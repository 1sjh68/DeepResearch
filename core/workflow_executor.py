"""用于在CLI和Web入口点运行Deep Research工作流的共享工具。"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime

from config import Config
from services.llm_interaction import preflight_llm_connectivity
from services.vector_db import VectorDBManager
from utils.text_processor import (
    consolidate_document_structure,
    final_post_processing,
    quality_check,
)
from workflows.graph_runner import run_graph_workflow


@dataclass
class WorkflowResult:
    """工作流执行结果的容器。"""

    raw_result: str
    final_answer: str | None
    quality_report: str | None
    saved_filepath: str | None
    success: bool
    error: str | None = None


def run_workflow_pipeline(
    config: Config,
    vector_db_manager: VectorDBManager | None,
    *,
    log_handler: logging.Handler | None = None,
    output_filename: str | None = None,
    save_result: bool = True,
) -> WorkflowResult:
    """执行主要的研究工作流，并根据请求持久化输出。"""
    # 预检 LLM 连通性（网络/代理/TLS），失败仅记录警告，不阻断流程
    try:
        if not preflight_llm_connectivity(config):
            logging.warning("LLM 连通性预检失败：后续步骤可能受到网络影响。建议检查直连/代理设置与超时重试配置。")
    except Exception as _exc:
        logging.debug("LLM 连通性预检异常: %s", _exc, exc_info=True)

    raw_result = run_graph_workflow(
        config,
        vector_db_manager,
        log_handler=log_handler,
    )

    if not raw_result or raw_result.startswith("错误："):
        error_text = raw_result or "工作流未返回任何结果。"
        logging.error("工作流执行失败: %s", error_text)
        return WorkflowResult(
            raw_result=raw_result,
            final_answer=None,
            quality_report=None,
            saved_filepath=None,
            success=False,
            error=error_text,
        )

    logging.info("\n--- 工作流完成，正在进行最终的后处理、评估与保存 ---")

    def _extract_heading_fingerprint(md_text: str) -> list[tuple[int, str, str | None]]:
        """提取有序的(level, title_text, section_id)列表，用于结构一致性校验。"""
        if not md_text:
            return []
        heading_re = re.compile(
            r"^(#{1,6})\s+(.*?)(?:\s*<!--\s*section_id:\s*([A-Za-z0-9_-]+)\s*-->)?\s*$",
            re.MULTILINE,
        )
        result: list[tuple[int, str, str | None]] = []
        for line in md_text.splitlines():
            m = heading_re.match(line)
            if m:
                level = len(m.group(1))
                title_text = (m.group(2) or "").strip()
                section_id = m.group(3)
                result.append((level, title_text, section_id))
        return result

    before_fp = _extract_heading_fingerprint(raw_result)
    structured_answer = consolidate_document_structure(raw_result)
    after_fp = _extract_heading_fingerprint(structured_answer)

    def _filter_fingerprint(fp: list[tuple[int, str, str | None]]):
        # 忽略无 section_id 的章节，这些章节在整合过程中可能被去重或重排
        return [item for item in fp if item[2]]

    # 实例级开关，若未配置则默认开启
    strict_enforce = getattr(config, "STRICT_STRUCTURE_ENFORCEMENT", True)
    fallback_on_mismatch = getattr(config, "FINAL_FALLBACK_ON_MISMATCH", True)

    use_fallback = False
    fallback_content: str | None = None
    if strict_enforce:
        before_filtered = _filter_fingerprint(before_fp)
        after_filtered = _filter_fingerprint(after_fp)
        if before_filtered != after_filtered:
            logging.warning("结构健康检查失败：合并后标题/ID 列表与合并前不一致。")
            logging.warning("  - 合并前: %s", before_filtered)
            logging.warning("  - 合并后: %s", after_filtered)
            if fallback_on_mismatch:
                # 优先回退到最近的 refine 快照
                latest_refine_path = None
                session_dir = config.session_dir
                try:
                    if session_dir and os.path.isdir(session_dir):
                        iter_dir = os.path.join(session_dir, "iterations")
                        if os.path.isdir(iter_dir):
                            candidates = [os.path.join(iter_dir, fn) for fn in os.listdir(iter_dir) if fn.startswith("iter_") and "_refine" in fn and fn.endswith(".md")]
                            if candidates:

                                def _candidate_key(path: str) -> tuple[int, float]:
                                    name = os.path.basename(path)
                                    match = re.search(r"iter_(\d+)", name)
                                    iter_index = int(match.group(1)) if match else -1
                                    try:
                                        mtime = os.path.getmtime(path)
                                    except OSError:
                                        mtime = 0.0
                                    return iter_index, mtime

                                latest_refine_path = max(candidates, key=_candidate_key)
                except Exception as _e:
                    logging.warning("扫描 refine 快照失败: %s", _e)

                if latest_refine_path and os.path.isfile(latest_refine_path):
                    try:
                        with open(latest_refine_path, encoding="utf-8") as rf:
                            fallback_content = rf.read()
                        logging.info("回退到最新 refine 快照: %s", latest_refine_path)
                    except Exception as _e:
                        logging.warning("读取 refine 快照失败，将回退到合并前的抛光文本: %s", _e)
                        fallback_content = raw_result
                else:
                    fallback_content = raw_result
                use_fallback = True

    final_answer = final_post_processing(fallback_content if use_fallback and fallback_content else structured_answer)

    quality_report = None
    if config.workflow.disable_final_quality_check:
        logging.info("\n--- 已禁用最终质量评估 (DISABLE_FINAL_QUALITY_CHECK=true) ---")
    else:
        logging.info("\n--- 最终产出质量评估报告 ---")
        quality_report = quality_check(config, final_answer)
        logging.info(quality_report)

    saved_filepath = None
    if save_result:
        filename = output_filename or f"final_solution_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        session_dir = config.session_dir
        if session_dir and os.path.isdir(session_dir):
            saved_filepath = os.path.join(session_dir, filename)
            try:
                with open(saved_filepath, "w", encoding="utf-8") as f:
                    f.write(final_answer)
                logging.info("🎉 最终报告已成功保存至: %s", saved_filepath)
            except Exception as exc:
                logging.error("保存最终报告时发生错误: %s", exc)
                saved_filepath = None
        else:
            logging.error("会话目录不存在，无法保存最终文件。")

    return WorkflowResult(
        raw_result=raw_result,
        final_answer=final_answer,
        quality_report=quality_report,
        saved_filepath=saved_filepath,
        success=True,
    )
