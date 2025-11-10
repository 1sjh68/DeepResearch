from __future__ import annotations

import logging
import re
from typing import Any, cast

from core.context_manager import ContextManager
from core.progress import StepOutput, step_result, workflow_step
from core.state_manager import WorkflowStateAdapter
from planning.tool_definitions import DraftModel, SectionContent
from utils.id_manager import ensure_section_id
from utils.iteration_storage import archive_iteration_snapshot
from utils.progress_tracker import safe_pulse, safe_step_update
from workflows.graph_state import GraphState
from workflows.nodes.sub_workflows.drafting import (
    generate_section_content,
    generate_section_content_structured,
)
from workflows.prompts import DRAFT_SYSTEM_PROMPT

TOPOLOGY_STEP_NAME = "topology_writer_node"


@workflow_step(TOPOLOGY_STEP_NAME, "拓扑写作初稿")
def topology_writer_node(state: GraphState) -> StepOutput:
    workflow_state = WorkflowStateAdapter.ensure(state)

    if workflow_state.draft_content:
        logging.info("检测到预填充的草稿内容，跳过初稿生成。")
        return step_result({}, "草稿已存在")

    config = workflow_state.config
    safe_pulse(
        config.task_id,
        f"迭代 0/{config.max_refinement_rounds} · 生成初稿中...",
    )

    style_guide = workflow_state.style_guide or ""
    outline_data = workflow_state.outline
    if not outline_data:
        raise ValueError("topology_writer_node 无法从 state 中获取 'outline'。工作流可能已在 plan_node 中失败。")

    skeleton_outline = workflow_state.skeleton_outline or _synthesize_skeleton_from_outline(outline_data)
    skeleton_index = _flatten_skeleton_map(skeleton_outline)
    digest_index = _build_digest_index(workflow_state.section_digests)

    external_data = workflow_state.external_data or ""
    embedding_model = getattr(config, "embedding_model_instance", None)
    context_manager = ContextManager(
        config,
        style_guide,
        outline_data,
        external_data,
        embedding_model,
        repository=workflow_state.context_repository,
        rag_service=workflow_state.rag_service,
        assembler=workflow_state.context_assembler,
    )

    # 检查是否启用结构化输出
    use_structured_output = getattr(config, "use_structured_draft_output", False)

    if use_structured_output:
        result_data, detail = _generate_structured_draft(
            workflow_state,
            config,
            outline_data,
            context_manager,
            style_guide,
            external_data,
            skeleton_index,
            digest_index,
        )
    else:
        result_data, detail = _generate_traditional_draft(
            workflow_state,
            config,
            outline_data,
            context_manager,
            style_guide,
            external_data,
            skeleton_index,
            digest_index,
        )

    repository, rag_service, assembler = context_manager.export_components()
    mutable_result = cast(dict[str, Any], result_data)
    mutable_result.update(
        {
            "context_repository": repository,
            "rag_service": rag_service,
            "context_assembler": assembler,
        }
    )
    return step_result(mutable_result, detail)


def _generate_traditional_draft(
    workflow_state,
    config,
    outline_data,
    context_manager,
    style_guide,
    external_data,
    skeleton_index,
    digest_index,
) -> tuple[dict[str, Any], str]:
    """生成传统文本格式的草稿（向后兼容）"""
    assembled_parts = [f"# {outline_data.get('title', 'Untitled Document')}\n\n"]
    chapters_to_generate = outline_data.get("outline", [])
    total_chapters = len(chapters_to_generate) or 1

    for i, chapter in enumerate(chapters_to_generate):
        logging.info(
            "  -> 起草章节 %s/%s: %s",
            i + 1,
            len(chapters_to_generate),
            chapter.get("title"),
        )
        safe_step_update(
            config.task_id,
            TOPOLOGY_STEP_NAME,
            (i / total_chapters) * 100.0,
            f"起草章节 {i + 1}/{len(chapters_to_generate)}",
        )
        section_payload = dict(chapter)
        # 🔧 修复：使用 ensure_section_id 确保 ID 被保存
        section_id = ensure_section_id(section_payload)

        skeleton_meta = skeleton_index.get(section_id)
        if skeleton_meta:
            section_payload["must_include"] = skeleton_meta.get("must_include", [])
            section_payload["organization_hint"] = skeleton_meta.get("organization_hint", "")

        digest_points = digest_index.get(section_id, [])
        section_payload["digest_points"] = digest_points

        context_for_chapter = context_manager.get_context_for_standalone_chapter(section_payload.get("title"))
        structured_brief = _compose_section_brief(section_payload, digest_points)
        combined_context = f"{structured_brief}\n\n{context_for_chapter}".strip()

        chapter_content = generate_section_content(
            config,
            section_data=section_payload,
            system_prompt=DRAFT_SYSTEM_PROMPT,
            model_name=config.main_ai_model,
            overall_context=combined_context,
            is_subsection=False,
        )
        assembled_parts.append(chapter_content)
        context_manager.update_completed_chapter_content(section_payload.get("title"), chapter_content)
        safe_step_update(
            config.task_id,
            TOPOLOGY_STEP_NAME,
            ((i + 1) / total_chapters) * 100.0,
            f"已完成章节 {i + 1}/{len(chapters_to_generate)}",
        )

    draft_content = "".join(assembled_parts)
    archive_iteration_snapshot(config, 0, "initial_draft", draft_content)
    detail = f"生成草稿章节 {len(chapters_to_generate)} 个"
    return {"draft_content": draft_content}, detail


def _generate_structured_draft(
    workflow_state,
    config,
    outline_data,
    context_manager,
    style_guide,
    external_data,
    skeleton_index,
    digest_index,
) -> tuple[dict[str, Any], str]:
    """生成结构化格式的草稿"""
    logging.info("开始生成结构化草稿...")

    document_title = outline_data.get("title", "Untitled Document")
    sections = []

    chapters_to_generate = outline_data.get("outline", [])
    total_chapters = len(chapters_to_generate) or 1
    for i, chapter in enumerate(chapters_to_generate):
        logging.info(
            "  -> 生成结构化章节 %s/%s: %s",
            i + 1,
            len(chapters_to_generate),
            chapter.get("title"),
        )
        safe_step_update(
            config.task_id,
            TOPOLOGY_STEP_NAME,
            (i / total_chapters) * 100.0,
            f"生成结构化章节 {i + 1}/{len(chapters_to_generate)}",
        )

        section_payload = dict(chapter)
        # 🔧 修复：使用 ensure_section_id 确保 ID 被保存
        section_id = ensure_section_id(section_payload)

        skeleton_meta = skeleton_index.get(section_id)
        if skeleton_meta:
            section_payload["must_include"] = skeleton_meta.get("must_include", [])
            section_payload["organization_hint"] = skeleton_meta.get("organization_hint", "")

        digest_points = digest_index.get(section_id, [])
        section_payload["digest_points"] = digest_points

        context_for_chapter = context_manager.get_context_for_standalone_chapter(section_payload.get("title"))
        structured_brief = _compose_section_brief(section_payload, digest_points)
        combined_context = f"{structured_brief}\n\n{context_for_chapter}".strip()

        try:
            section_content = generate_section_content_structured(
                config,
                section_data=section_payload,
                system_prompt=DRAFT_SYSTEM_PROMPT,
                model_name=config.main_ai_model,
                overall_context=combined_context,
            )

            if section_content:
                sections.append(section_content)
                context_manager.update_completed_chapter_content(section_payload.get("title"), section_content.content)

        except Exception as e:
            logging.warning(f"章节 '{chapter.get('title')}' 结构化生成失败，使用回退机制: {e}")
            # 回退到传统生成方式
            fallback_content = generate_section_content(
                config,
                section_data=section_payload,
                system_prompt=DRAFT_SYSTEM_PROMPT,
                model_name=config.main_ai_model,
                overall_context=combined_context,
                is_subsection=False,
            )

            # 将传统内容转换为结构化格式
            key_claims = []
            todos = []

            section_obj = SectionContent(
                section_id=section_id,
                title=section_payload.get("title", f"章节 {i + 1}"),
                content=fallback_content,
                key_claims=key_claims,
                todos=todos,
                word_count=len(fallback_content) if fallback_content else 0,
            )
            sections.append(section_obj)
        safe_step_update(
            config.task_id,
            TOPOLOGY_STEP_NAME,
            ((i + 1) / total_chapters) * 100.0,
            f"已完成结构化章节 {i + 1}/{len(chapters_to_generate)}",
        )

    # 创建结构化草稿模型
    total_word_count = sum(section.word_count for section in sections if section.word_count)

    draft_model = DraftModel(
        sections=sections,
        document_title=document_title,
        summary=None,
        total_word_count=total_word_count,
        writing_style_notes=style_guide,
    )

    # 导出为传统文本格式以保持兼容性
    legacy_draft_content = _convert_structured_to_legacy_format(draft_model)
    archive_iteration_snapshot(config, 0, "initial_draft", legacy_draft_content)

    # 保存结构化数据
    structured_data = {
        "draft_model": draft_model.model_dump(),
        "legacy_content": legacy_draft_content,
    }

    detail = f"生成结构化草稿章节 {len(sections)} 个"
    return {
        "draft_content": legacy_draft_content,
        "draft_structure": structured_data,
    }, detail


def _convert_structured_to_legacy_format(draft_model: DraftModel) -> str:
    """将结构化草稿模型转换为传统文本格式（用于向后兼容）"""
    parts: list[str] = []

    if draft_model.document_title:
        parts.append(f"# {draft_model.document_title}\n\n")

    for index, section in enumerate(draft_model.sections):
        parts.append(f"\n## {section.title}\n\n")
        parts.append(section.content)

        # 添加关键主张到文档中
        if section.key_claims:
            parts.append("\n\n**关键主张：**\n")
            for claim in section.key_claims:
                parts.append(f"- {claim}\n")

        # 添加任务列表到文档中
        if section.todos:
            parts.append("\n\n**待办任务：**\n")
            for todo in section.todos:
                parts.append(f"- {todo}\n")

        parts.append("\n\n")

    return "".join(parts)


# 🔧 修复：删除本地的 _ensure_section_id 函数，使用 utils.id_manager.ensure_section_id


def _compose_section_brief(section_payload: dict[str, Any], digest_points: list[dict[str, str]]) -> str:
    title = section_payload.get("title", "未命名章节")
    must_include = section_payload.get("must_include") or []
    organization_hint = section_payload.get("organization_hint", "")
    lines: list[str] = [f"--- Skeleton Checklist · {title} ---"]
    if must_include:
        lines.extend([f"- {point}" for point in must_include[:6]])
    if organization_hint:
        lines.append(f"组织建议：{organization_hint}")

    if digest_points:
        lines.append("--- Indexed Facts (引用 ref:source#anchor) ---")
        for point in digest_points[:6]:
            fact_text = point.get("fact", "")
            citation = point.get("citation", "ref:pending")
            snippet = fact_text[:220].strip()
            lines.append(f"- {snippet} (ref: {citation})")
    return "\n".join(lines)


def _build_digest_index(section_digests: dict[str, Any] | None) -> dict[str, list[dict[str, str]]]:
    if not section_digests:
        return {}
    mapping: dict[str, list[dict[str, str]]] = {}
    for entry in section_digests.get("sections", []):
        section_id = entry.get("section_id")
        if not section_id:
            continue
        mapping[section_id] = entry.get("facts", [])
    return mapping


def _flatten_skeleton_map(skeleton_outline: dict[str, Any]) -> dict[str, dict[str, Any]]:
    mapping: dict[str, dict[str, Any]] = {}

    def _walk(sections: list[dict[str, Any]]):
        for section in sections:
            # 🔧 修复：使用 ensure_section_id 确保 ID 被保存
            section_id = ensure_section_id(section)
            mapping[section_id] = {
                "must_include": section.get("must_include", []),
                "organization_hint": section.get("organization_hint", ""),
            }
            children = section.get("children") or []
            if children:
                _walk(children)

    _walk(skeleton_outline.get("sections", []))
    return mapping


def _synthesize_skeleton_from_outline(outline_data: dict[str, Any]) -> dict[str, Any]:
    return {
        "title": outline_data.get("title", "Untitled Document"),
        "sections": [_outline_to_skeleton(chapter, depth=0) for chapter in outline_data.get("outline", [])],
    }


def _outline_to_skeleton(chapter: dict[str, Any], depth: int) -> dict[str, Any]:
    # 🔧 修复：使用 ensure_section_id 确保 ID 被保存
    section_id = ensure_section_id(chapter)
    title = chapter.get("title", f"未命名章节-{section_id[:8]}")
    return {
        "id": section_id,
        "title": title,
        "must_include": _derive_must_include_points(chapter),
        "organization_hint": _derive_org_hint(chapter, depth),
        "children": [_outline_to_skeleton(child, depth + 1) for child in chapter.get("sections", []) or []],
    }


def _derive_must_include_points(chapter: dict[str, Any]) -> list[str]:
    description = chapter.get("description", "") or ""
    sentences = [frag.strip() for frag in re.split(r"[。！!？?；;]", description) if frag and len(frag.strip()) >= 6]
    child_titles = [child.get("title", "").strip() for child in chapter.get("sections", []) or [] if child.get("title")]
    child_requirements = [f"覆盖子主题：{child_title}" for child_title in child_titles]
    merged = sentences + child_requirements
    if not merged:
        merged = ["说明本节的关键概念、数据与对比结论。"]
    return merged[:6]


def _derive_org_hint(chapter: dict[str, Any], depth: int) -> str:
    child_titles = [child.get("title", "").strip() for child in chapter.get("sections", []) or [] if child.get("title")]
    if not child_titles:
        return "围绕“背景→分析→结论”三段展开。"
    prefix = "叶子小节" if depth >= 1 else "父级章节"
    joined = " → ".join(child_titles[:4])
    return f"{prefix}建议依次阐述：{joined}，并在结尾总结对比。"


__all__ = ["topology_writer_node"]
