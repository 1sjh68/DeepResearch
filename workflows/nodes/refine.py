from __future__ import annotations

import json
import logging
import re
from collections.abc import Iterable
from difflib import SequenceMatcher
from typing import Any

from pydantic import ValidationError

from core.context_manager import ContextManager
from core.progress import StepOutput, step_result, workflow_step
from core.state_manager import WorkflowStateAdapter
from planning.tool_definitions import CritiqueModel, FineGrainedPatchList
from services.llm_interaction import call_ai_with_schema
from utils.progress_tracker import safe_pulse
from utils.text_processor import extract_json_from_ai_response
from workflows.graph_state import GraphState
from workflows.prompts import PATCH_SCHEMA_INSTRUCTIONS


@workflow_step("refine_node", "生成内容优化补丁")
def refine_node(state: GraphState) -> StepOutput:
    workflow_state = WorkflowStateAdapter.ensure(state)
    config = workflow_state.config
    current_iteration = workflow_state.refinement_count + 1
    max_rounds = config.max_refinement_rounds
    logging.info(
        "[RefineLoop] Iteration %s/%s -> refine_node",
        current_iteration,
        max_rounds,
    )
    safe_pulse(
        config.task_id,
        f"迭代 {current_iteration}/{max_rounds} · 生成内容优化补丁中...",
    )

    draft_content = workflow_state.draft_content or ""
    critique = workflow_state.critique or ""
    research_brief = workflow_state.research_brief or ""
    style_guide = workflow_state.style_guide or ""
    outline_data = workflow_state.outline

    # 检查是否有结构化研究数据
    structured_research_data = getattr(workflow_state, "structured_research_data", None)

    if not isinstance(outline_data, dict) or not outline_data.get("outline"):
        logging.warning("refine_node: Outline 数据缺失，跳过补丁生成。")
        return step_result({"patches": []}, "缺少大纲")

    external_data = workflow_state.external_data or ""
    raw_structured_critique = getattr(workflow_state, "structured_critique", None)
    structured_critique: CritiqueModel | None = None
    if raw_structured_critique:
        try:
            structured_critique = CritiqueModel.model_validate(raw_structured_critique)
        except (ValidationError, TypeError) as exc:
            logging.warning("refine_node: 无法解析结构化评审数据，改用文本回退。详情: %s", exc)
    knowledge_gaps: list[str] = []
    if structured_critique and structured_critique.knowledge_gaps:
        knowledge_gaps = structured_critique.knowledge_gaps
    else:
        knowledge_gaps = workflow_state.knowledge_gaps or []

    if not draft_content:
        logging.warning("refine_node: No draft_content found to refine. Skipping.")
        return step_result({"patches": []}, "缺少草稿内容")

    embedding_model = getattr(config, "embedding_model_instance", None)

    # 处理研究数据：优先使用结构化数据，否则使用传统字符串格式
    research_summary = ""
    research_context = ""
    if structured_research_data:
        research_summary, research_context = _process_structured_research_data(structured_research_data)
    else:
        research_summary = research_brief or ""
        research_context = research_brief or ""

    combined_context_segments = [segment.strip() for segment in (external_data, research_context) if segment and segment.strip()]
    combined_context_data = "\n\n".join(combined_context_segments)
    context_manager_for_patch = ContextManager(
        config,
        style_guide,
        outline_data,
        combined_context_data,
        embedding_model,
        repository=workflow_state.context_repository,
        rag_service=workflow_state.rag_service,
        assembler=workflow_state.context_assembler,
    )

    outline_chapters = outline_data.get("outline", [])

    # 🔧 修复：从 draft_content 中动态提取实际存在的 section_id（而非从 outline_data）
    # 这样保证映射表始终与当前文档内容同步
    actual_section_ids_in_draft = re.findall(r"<!--\s*section_id:\s*([A-Za-z0-9-]+)\s*-->", draft_content)

    if not actual_section_ids_in_draft:
        logging.warning("  ⚠️  draft_content 中未找到任何 section_id 注释，回退到 outline_data")
        # 回退方案：从 outline_data 生成映射表
        all_chapters_index = _build_chapter_index(outline_chapters)
        section_number_map = {idx + 1: item["id"] for idx, item in enumerate(all_chapters_index) if item.get("id")}
    else:
        # 正常流程：根据 draft 中实际出现的顺序生成映射表
        section_number_map = {idx + 1: section_id for idx, section_id in enumerate(actual_section_ids_in_draft)}
        logging.info("  ✅ 从 draft_content 中提取 %d 个章节ID，生成数字映射表", len(section_number_map))

    target_chapters = _select_target_chapters(outline_chapters, structured_critique, knowledge_gaps, critique)
    logging.info(
        "  - Patcher 目标章节: %s",
        ", ".join(f"{item['title']}" for item in target_chapters) if target_chapters else "无",
    )

    # 为 AI Prompt 生成 target_chapters 在全局编号中的位置
    target_chapter_ids = {item["id"] for item in target_chapters if item.get("id")}
    target_global_numbers = {num: uuid for num, uuid in section_number_map.items() if uuid in target_chapter_ids}

    chapter_contexts: list[str] = []
    # 🔧 使用全局编号而非局部 idx
    for chapter_info in target_chapters:
        chapter_title = chapter_info["title"]
        chapter_id = chapter_info.get("id") or ""
        chapter_path = chapter_info.get("path") or chapter_title
        existing_text = _chapter_body_snippet(draft_content, chapter_id)

        # 🔧 从内容中移除 UUID 注释，避免 AI 混淆
        existing_text_clean = re.sub(r"<!--\s*section_id:\s*[A-Za-z0-9-]+\s*-->", "", existing_text)

        # 查找该章节的全局编号
        global_num = next((num for num, uuid in section_number_map.items() if uuid == chapter_id), "?")

        context_packet = context_manager_for_patch.get_context_for_chapter_critique(chapter_title, draft_content, section_number_map)
        block_parts = [
            f"[Section #{global_num}] {chapter_path}",  # ✅ 使用全局稳定编号
            "[Existing Draft]\n" + (existing_text_clean or "<未找到该章节正文或章节为空>"),
        ]
        if context_packet:
            block_parts.append("[Supporting Context]\n" + context_packet)
        chapter_contexts.append("\n\n".join(block_parts))
    safe_pulse(
        config.task_id,
        f"迭代 {current_iteration}/{max_rounds} · Patcher上下文已准备，目标章节 {len(target_chapters)} 个；调用模型中...",
    )

    precise_context_for_patcher = "\n\n".join(chapter_contexts)
    issues_for_prompt: list[str] = []
    if structured_critique:
        for issue in structured_critique.priority_issues or []:
            _append_unique_text(issues_for_prompt, issue)
        for gap in structured_critique.knowledge_gaps or []:
            _append_unique_text(issues_for_prompt, gap)
    for gap in knowledge_gaps:
        _append_unique_text(issues_for_prompt, gap)
    knowledge_gap_block = "\n".join(f"- {item}" for item in issues_for_prompt) if issues_for_prompt else "None"

    # 为 AI 生成简单易用的章节列表（使用全局稳定编号）
    section_list_for_ai = []
    for num, uuid in sorted(target_global_numbers.items()):
        # 从 target_chapters 中找到对应章节
        chapter_info = next((ch for ch in target_chapters if ch.get("id") == uuid), None)
        if chapter_info:
            chapter_title = chapter_info.get("title", "未命名")
            section_list_for_ai.append(f"  [{num}] {chapter_title[:80]}")

    section_reference_block = "\n".join(section_list_for_ai) if section_list_for_ai else "  (无可用章节)"

    # 🔧 Phase 1 优化：计算问题数量并强调要求
    total_issues_count = len(issues_for_prompt)

    patch_user_prompt = (
        f"""[Original Problem]\n{config.user_problem}\n\n"""
        f"""[Latest Research Brief]\n{(research_summary or research_brief or "None")}\n\n"""
        f"""[Revision Feedback]\n---\n{critique}\n---\n\n"""
        f"""[Knowledge Gaps]\n{knowledge_gap_block}\n\n"""
        f"""[Available Sections for Editing]\n"""
        f"""Use the section number as target_id:\n"""
        f"""{section_reference_block}\n\n"""
        f"""Example: To edit section [1], use "target_id": 1\n"""
        f"""Example: To edit section [3], use "target_id": 3\n\n"""
        f"""[Target Chapter Context for Revision]\n---\n{precise_context_for_patcher}\n---\n\n"""
        f"""[TASK REQUIREMENTS]\n"""
        f"""Total issues identified: {total_issues_count}\n"""
        f"""You MUST generate AT LEAST {total_issues_count} patches (one per issue).\n"""
        f"""Do NOT skip any issues. Address each one with a specific sentence-level edit.\n\n"""
        """Now generate the patch list to resolve these issues. Use simple numeric target_id."""
    )

    # 🔍 调试：记录完整 prompt 到文件
    if getattr(config, "debug_prompts", False):
        import datetime

        debug_file = f"debug_patch_prompt_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(debug_file, "w", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write("SYSTEM PROMPT\n")
            f.write("=" * 80 + "\n")
            f.write("You are Patch-Bot. Generate sentence-level revision patches.\n")
            f.write(PATCH_SCHEMA_INSTRUCTIONS)
            f.write("\nEach patch must address issues from the critique and knowledge gaps only.\n\n")
            f.write("=" * 80 + "\n")
            f.write("USER PROMPT\n")
            f.write("=" * 80 + "\n")
            f.write(patch_user_prompt)
            f.write("\n" + "=" * 80 + "\n")
            f.write(f"Section Number Map: {section_number_map}\n")
        logging.info(f"  - Prompt 调试文件已保存: {debug_file}")

    model_name = config.patcher_model_name or getattr(config, "main_ai_model", None)
    if not model_name:
        raise ValueError("未配置 patcher 模型名称，无法生成补丁。")

    def _invoke_patch_request(prompt: str, temperature: float, attempt_label: str) -> tuple[list[dict[str, Any]], str | None]:
        try:
            # 🔧 使用结构化调用，AI 会看到 Pydantic Schema（target_id 必须是 int）
            # 🆕 优化：强调必须为每个问题生成补丁
            system_prompt = (
                "You are Patch-Bot. Generate sentence-level revision patches.\n\n"
                + PATCH_SCHEMA_INSTRUCTIONS
                + "\n\n**CRITICAL REQUIREMENTS:**\n"
                + "1. target_id MUST be an integer (1, 2, 3, 4, 5, ...). NEVER use UUID strings.\n"
                + "2. Generate AT LEAST ONE patch for EACH issue in the critique and knowledge gaps.\n"
                + "3. DO NOT skip issues or consolidate multiple issues into one patch.\n"
                + "4. If there are N issues, generate AT LEAST N patches.\n"
                + "5. Each patch should address ONE specific issue clearly.\n\n"
                + "If you cannot fix an issue with sentence edits, still create a patch that attempts improvement."
            )

            edit_obj, call_mode = call_ai_with_schema(
                config,
                model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                schema=FineGrainedPatchList,
                kwargs={
                    "temperature": temperature,
                    "max_tokens_output": getattr(config, "intermediate_edit_max_tokens", 2048),
                },
            )

            # 如果返回的是字符串（回退到普通调用），尝试手动解析
            if isinstance(edit_obj, str):
                json_response_text = extract_json_from_ai_response(
                    config,
                    edit_obj,
                    context_for_error_log=f"Patcher AI response ({attempt_label})",
                )
                if not json_response_text:
                    logging.info("  - [%s] 模型未返回补丁结果。", attempt_label)
                    return [], None

                parsed_payload = json.loads(json_response_text)
                if isinstance(parsed_payload, list):
                    logging.warning(
                        "  - [%s] 收到列表形式补丁，自动封装为 {'patches': ...}。",
                        attempt_label,
                    )
                    parsed_payload = {"patches": parsed_payload}
                    json_response_text = json.dumps(parsed_payload, ensure_ascii=False)

                edit_obj = FineGrainedPatchList.model_validate_json(json_response_text)

            patch_list = [patch.model_dump() for patch in edit_obj.patches]
            logging.info("  - [%s] 成功生成 %s 个补丁 (模式: %s)", attempt_label, len(patch_list), call_mode)
            # 🔍 调试：记录AI返回的 target_id
            if patch_list:
                target_ids = [p.get("target_id") for p in patch_list]
                logging.debug("  - [%s] AI返回的 target_id 列表: %s (类型: %s)", attempt_label, target_ids, [type(tid).__name__ for tid in target_ids])
            return patch_list, None
        except (ValidationError, json.JSONDecodeError) as exc:
            logging.error("  - [%s] 补丁生成失败: %s", attempt_label, exc)
            return [], str(exc)

    # 🆕 优化：确保最低温度，提高生成多样性
    # 🔧 Phase 1 优化：提高初始温度以增加补丁多样性
    base_temperature = max(config.temperature_factual, 0.4)  # 从 0.3 提升到 0.4
    patches, _ = _invoke_patch_request(patch_user_prompt, base_temperature, "primary")
    retry_attempted = False

    # 🆕 优化：计算期望的最小补丁数
    expected_min_patches = max(len(knowledge_gaps), len(issues_for_prompt), 1)

    # 🆕 优化：不仅检查是否为空，还检查数量是否充足
    patch_count_insufficient = len(patches) < expected_min_patches
    has_issues_to_fix = bool(issues_for_prompt or knowledge_gaps or critique.strip())
    should_retry = patch_count_insufficient and getattr(config, "enable_patch_retry", False) and has_issues_to_fix
    if should_retry:
        retry_attempted = True
        # 🔧 Phase 1 优化：进一步提高重试温度
        retry_temperature = min(base_temperature + 0.3, 0.95)  # 从 +0.2 提升到 +0.3，上限从 0.9 到 0.95

        # 🆕 优化：根据情况定制重试提示
        if not patches:
            retry_reason = "未生成任何补丁"
            retry_directive = (
                "\n\n[RETRY DIRECTIVE - CRITICAL]\n"
                f"The previous attempt returned NO patches, but there are {expected_min_patches} issues to address.\n\n"
                "**MANDATORY REQUIREMENTS:**\n"
                f"- You MUST generate at least {expected_min_patches} patches\n"
                "- Create ONE patch for EACH issue listed in critique and knowledge gaps\n"
                "- Do NOT consolidate multiple issues into one patch\n"
                "- If unsure how to fix, still propose an improvement attempt\n\n"
                "Re-evaluate ALL critique points and knowledge gaps one by one.\n"
                "Generate a separate patch for each item."
            )
        else:
            retry_reason = f"补丁数量不足（生成了 {len(patches)} 个，预期至少 {expected_min_patches} 个）"
            missing_count = expected_min_patches - len(patches)
            retry_directive = (
                "\n\n[RETRY DIRECTIVE - INSUFFICIENT PATCHES]\n"
                f"Previous attempt: {len(patches)} patches generated\n"
                f"Expected minimum: {expected_min_patches} patches\n"
                f"Missing: {missing_count} patches\n\n"
                "**ACTION REQUIRED:**\n"
                f"- Generate {missing_count} MORE patches to address remaining issues\n"
                "- Review the critique and knowledge gaps list\n"
                "- Identify which issues were NOT addressed in the first attempt\n"
                "- Create patches for ALL unaddressed issues\n\n"
                "Return ALL patches (previous + new ones) in your response."
            )

        retry_prompt = patch_user_prompt + retry_directive

        safe_pulse(
            config.task_id,
            f"迭代 {current_iteration}/{max_rounds} · {retry_reason}，重试生成中...",
        )
        logging.warning("=" * 60)
        logging.warning("⚠️  补丁生成不充分，尝试重试")
        logging.warning(f"  - 当前迭代: {current_iteration}/{max_rounds}")
        logging.warning(f"  - 生成的补丁数: {len(patches)}")
        logging.warning(f"  - 预期最小数量: {expected_min_patches}")
        logging.warning(f"  - Knowledge gaps: {len(knowledge_gaps)} 个")
        logging.warning(f"  - Critique issues: {len(issues_for_prompt)} 个")
        logging.warning(f"  - 重试温度: {retry_temperature}")
        logging.warning("=" * 60)

        retry_patches, _ = _invoke_patch_request(retry_prompt, retry_temperature, "retry")
        if retry_patches:
            logging.info(f"  ✓ 重试成功，新增 {len(retry_patches)} 个补丁")
            patches = retry_patches
        else:
            logging.warning("  ✗ 重试失败，仍未生成补丁")

    # 🔍 调试：记录最终的补丁和映射表
    if patches:
        logging.info("  - 补丁 target_id 列表: %s", [p.get("target_id") for p in patches])
    logging.info("  - section_number_map 映射表: %s", {k: v[:8] + "..." for k, v in section_number_map.items()} if section_number_map else "空")

    detail_msg = f"迭代 {current_iteration}/{max_rounds}，生成补丁 {len(patches)} 个"
    if retry_attempted:
        detail_msg += "（重试" + ("成功" if patches else "无补丁") + "）"

    repository, rag_service, assembler = context_manager_for_patch.export_components()
    return step_result(
        {
            "patches": patches,
            "section_number_map": section_number_map,  # 传递数字→UUID映射表
            "context_repository": repository,
            "rag_service": rag_service,
            "context_assembler": assembler,
            "knowledge_gaps": knowledge_gaps,
            "structured_critique": raw_structured_critique,
        },
        detail_msg,
    )


def _process_structured_research_data(structured_research_data: dict) -> tuple[str, str]:
    """
    处理结构化研究数据，生成用于补丁生成的上下文内容
    """
    if not structured_research_data:
        return "", ""

    briefs = structured_research_data.get("briefs", [])
    statistics = structured_research_data.get("statistics", {})

    if not briefs:
        return "", ""

    summary_lines: list[str] = []
    detail_lines: list[str] = []
    detail_lines.append("结构化研究数据（基于知识缺口搜索结果）:")
    detail_lines.append("")

    sorted_briefs = sorted(
        briefs,
        key=lambda x: (x.get("source_quality", "") == "high", x.get("confidence", 0)),
        reverse=True,
    )

    for i, brief in enumerate(sorted_briefs[:10]):
        url = brief.get("url", "")
        title = brief.get("title", "")
        summary = brief.get("summary", "")
        key_points = brief.get("key_points", [])
        confidence = brief.get("confidence", 0)
        source_quality = brief.get("source_quality", "unknown")
        relevance = brief.get("relevance_score", 0)

        research_part = f"[研究源 {i + 1}] {title or url or '未命名来源'}"
        if url:
            research_part += f" ({url})"
        research_part += f"\n  置信度: {confidence:.2f} | 质量: {source_quality} | 相关性: {relevance:.2f}"
        research_part += f"\n  总结: {summary}"

        if key_points:
            research_part += "\n  关键要点:"
            for point in key_points:
                research_part += f"\n    - {point}"

        detail_lines.append(research_part)
        detail_lines.append("")

        if title or summary:
            summary_excerpt = summary.strip() if summary else ""
            summary_lines.append(f"- {title or url}: {summary_excerpt[:160]}")

    if statistics:
        detail_lines.append("统计信息:")
        for key, value in statistics.items():
            detail_lines.append(f"- {key}: {value}")

    summary_text = "\n".join(summary_lines[:5])
    detail_text = "\n".join(detail_lines)
    return summary_text, detail_text


def _append_unique_text(collection: list[str], text: str | None) -> None:
    if not text:
        return
    candidate = text.strip()
    if not candidate:
        return
    if candidate not in collection:
        collection.append(candidate)


def _select_target_chapters(
    outline_chapters: Iterable[dict[str, Any]],
    structured_critique: CritiqueModel | None,
    knowledge_gaps: list[str],
    critique_text: str,
) -> list[dict[str, Any]]:
    chapters = _build_chapter_index(outline_chapters)
    if not chapters:
        return []

    score_map: dict[str, float] = {}

    def _bump_score(entry_id: str, weight: float) -> None:
        if not entry_id:
            return
        score_map[entry_id] = score_map.get(entry_id, 0.0) + weight

    def _match_text(text: str, base_weight: float = 1.0) -> None:
        if not text:
            return
        text_lower = text.lower()
        for entry in chapters:
            if entry["id_lower"] and entry["id_lower"] in text_lower:
                _bump_score(entry["id"], 3.0 * base_weight)
            if entry["title_lower"] and entry["title_lower"] in text_lower:
                _bump_score(entry["id"], 2.0 * base_weight)
            if entry["path_lower"] and entry["path_lower"] in text_lower:
                _bump_score(entry["id"], 1.2 * base_weight)
        best_entry = _best_fuzzy_match(text_lower, chapters)
        if best_entry:
            ratio = _similarity_ratio(text_lower, best_entry["title_lower"])
            _bump_score(best_entry["id"], max(1.0, ratio * 2.5) * base_weight)

    if structured_critique:
        for item in structured_critique.priority_issues or []:
            _match_text(item, base_weight=1.5)
        for item in structured_critique.knowledge_gaps or []:
            _match_text(item, base_weight=1.3)

    if not score_map:
        for gap in knowledge_gaps:
            _match_text(gap, base_weight=1.1)

    if not score_map and critique_text:
        _match_text(critique_text, base_weight=1.0)

    ranked = sorted(
        (entry for entry in chapters if score_map.get(entry["id"], 0) > 0),
        key=lambda item: (-score_map[item["id"]], item["order"]),
    )
    if ranked:
        max_targets = min(5, len(ranked))
        logging.debug(
            "refine_node: 章节评分分布: %s",
            ", ".join(f"{entry['id']}={score_map[entry['id']]:.2f}" for entry in ranked[:max_targets]),
        )
        logging.info(
            "refine_node: 识别到 %s 个候选章节用于修订：%s",
            max_targets,
            ", ".join(r["path"] for r in ranked[:max_targets]),
        )
        return ranked[:max_targets]

    if chapters:
        fallback_targets = chapters[: min(5, len(chapters))]
        logging.info(
            "refine_node: 未从评审中识别到特定章节，回退到前 %s 个章节上下文。",
            len(fallback_targets),
        )
        logging.debug(
            "refine_node: 回退章节列表: %s",
            ", ".join(item["path"] for item in fallback_targets),
        )
        return fallback_targets

    return []


def _build_chapter_index(outline_chapters: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    chapter_index: list[dict[str, Any]] = []
    order_counter = 0

    def _walk(chapters: Iterable[dict[str, Any]], parent_titles: list[str], parent_indices: tuple[int, ...]) -> None:
        nonlocal order_counter
        for local_idx, chapter in enumerate(chapters or []):
            if not isinstance(chapter, dict):
                continue
            chapter_id_raw = chapter.get("id") or chapter.get("title") or f"chapter_{order_counter + 1}"
            chapter_id = str(chapter_id_raw)
            title = str(chapter.get("title") or f"未命名章节-{chapter_id[:8]}")
            path_titles = parent_titles + [title]
            entry = {
                "id": chapter_id,
                "id_lower": chapter_id.lower(),
                "title": title,
                "title_lower": title.lower(),
                "path": " > ".join(path_titles),
                "path_lower": " > ".join(part.lower() for part in path_titles),
                "order": order_counter,
                "index": parent_indices + (local_idx,),
            }
            chapter_index.append(entry)
            order_counter += 1
            children = chapter.get("sections") or []
            if children:
                _walk(children, path_titles, entry["index"])

    _walk(outline_chapters or [], [], tuple())
    return chapter_index


def _best_fuzzy_match(text_lower: str, chapters: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not text_lower or not chapters:
        return None
    best_score = 0.0
    best_entry: dict[str, Any] | None = None
    for entry in chapters:
        title_lower = entry["title_lower"]
        if not title_lower:
            continue
        score = SequenceMatcher(None, text_lower, title_lower).ratio()
        if score > best_score:
            best_score = score
            best_entry = entry
    if best_entry and best_score >= 0.6:
        return best_entry
    return None


def _similarity_ratio(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    return SequenceMatcher(None, left, right).ratio()


def _chapter_body_snippet(draft_content: str, chapter_id: str, max_chars: int = 1600) -> str:
    if not draft_content or not chapter_id:
        return ""
    escaped_id = re.escape(chapter_id)
    pattern = re.compile(
        rf"(^#+.*?<!--\s*section_id:\s*{escaped_id}\s*-->.*?)(?=^#+ |\Z)",
        re.MULTILINE | re.DOTALL,
    )
    match = pattern.search(draft_content)
    if not match:
        return ""
    block = match.group(1).strip()
    lines = block.splitlines()
    if len(lines) > 1:
        body = "\n".join(lines[1:]).strip()
    else:
        body = block
    if not body:
        return ""
    if len(body) <= max_chars:
        return body
    trimmed = body[:max_chars].rstrip()
    return f"{trimmed}\n...[内容已截断]"


__all__ = ["refine_node"]
