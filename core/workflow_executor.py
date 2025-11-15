"""用于在CLI和Web入口点运行Deep Research工作流的共享工具。"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass

from config import Config
from logic.file_saver import save_final_result
from logic.post_processing import (
    consolidate_document_structure,
    final_post_processing,
    quality_check,
)
from services.llm_interaction import preflight_llm_connectivity
from services.vector_db import VectorDBManager
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

    structured_answer = consolidate_document_structure(raw_result)

    # 实例级开关，若未配置则默认开启
    strict_enforce = getattr(config, "STRICT_STRUCTURE_ENFORCEMENT", True)
    fallback_on_mismatch = getattr(config, "FINAL_FALLBACK_ON_MISMATCH", True)

    use_fallback = False
    fallback_content: str | None = None
    if strict_enforce:
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
        saved_filepath = save_final_result(config, final_answer, output_filename)

    return WorkflowResult(
        raw_result=raw_result,
        final_answer=final_answer,
        quality_report=quality_report,
        saved_filepath=saved_filepath,
        success=True,
    )
