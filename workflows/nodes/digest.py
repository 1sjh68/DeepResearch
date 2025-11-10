from __future__ import annotations

import logging
import re
from typing import Any

from core.context_manager import ContextManager
from core.progress import StepOutput, step_result, workflow_step
from core.state_manager import WorkflowStateAdapter
from utils.iteration_storage import archive_iteration_snapshot
from utils.progress_tracker import safe_pulse
from workflows.graph_state import GraphState


@workflow_step("digest_node", "整理资料索引卡")
def digest_node(state: GraphState) -> StepOutput:
    """根据骨架为每个小节生成“索引卡片”（facts + anchors）。"""
    workflow_state = WorkflowStateAdapter.ensure(state)
    config = workflow_state.config

    skeleton_outline = workflow_state.skeleton_outline
    outline = workflow_state.outline
    if not skeleton_outline and outline:
        logging.warning("未检测到 skeleton_outline，回退至原始大纲构建骨架。")
        from workflows.nodes.skeleton import _build_skeleton_section  # type: ignore

        skeleton_outline = {
            "title": outline.get("title", "Untitled Document"),
            "sections": [_build_skeleton_section(chapter, depth=0) for chapter in outline.get("outline", [])],
        }

    if not skeleton_outline:
        raise ValueError("digest_node 需要 skeleton_outline 数据。")

    safe_pulse(config.task_id, "三段式 · 第2步：整理索引卡片")

    style_guide = workflow_state.style_guide or ""
    external_data = workflow_state.external_data or ""
    embedding_model = getattr(config, "embedding_model_instance", None)
    context_manager = ContextManager(
        config,
        style_guide,
        workflow_state.outline or {},
        external_data,
        embedding_model,
        repository=workflow_state.context_repository,
        rag_service=workflow_state.rag_service,
        assembler=workflow_state.context_assembler,
    )

    section_digests: list[dict[str, Any]] = []
    for node in _flatten_skeleton(skeleton_outline["sections"]):
        chapter_title = node["title"]
        context_bundle = context_manager.get_context_for_standalone_chapter(chapter_title)
        facts = _extract_facts_from_context(context_bundle, chapter_title, node["id"])
        section_digests.append(
            {
                "section_id": node["id"],
                "title": chapter_title,
                "facts": facts,
                "must_include": node.get("must_include", []),
                "organization_hint": node.get("organization_hint", ""),
            }
        )

    repository, rag_service, assembler = context_manager.export_components()
    digest_payload = {"sections": section_digests, "feedback": []}
    archive_iteration_snapshot(
        config,
        0,
        "skeleton_digest",
        f"索引卡片总数: {len(section_digests)}",
    )
    logging.info("索引卡片生成完成，共 %s 个章节。", len(section_digests))

    return step_result(
        {
            "section_digests": digest_payload,
            "context_repository": repository,
            "rag_service": rag_service,
            "context_assembler": assembler,
        },
        "索引卡片已构建",
    )


def _flatten_skeleton(sections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for section in sections:
        items.append(section)
        children = section.get("children") or []
        if children:
            items.extend(_flatten_skeleton(children))
    return items


def _extract_facts_from_context(context: str, title: str, section_id: str) -> list[dict[str, str]]:
    """将 ContextManager 返回的大段文本拆成 3-6 条事实要点，并生成引用锚点。"""
    cleaned = context.strip()
    if not cleaned:
        return [
            {
                "fact": "未找到可用资料，请在研究节点补充引用。",
                "citation": f"{section_id}#pending",
            }
        ]

    chunks = _split_context_chunks(cleaned)
    facts: list[dict[str, str]] = []
    for chunk_idx, chunk in enumerate(chunks, start=1):
        sentences = _split_into_sentences(chunk)
        if not sentences:
            continue
        citation = _derive_citation_anchor(chunk, section_id, chunk_idx)
        for sentence in sentences:
            facts.append({"fact": sentence, "citation": citation})
            if len(facts) >= 6:
                break
        if len(facts) >= 6:
            break

    if not facts:
        snippet = chunks[0][:220].strip() if chunks else cleaned[:220].strip()
        citation = _derive_citation_anchor(cleaned, section_id, 1)
        facts.append({"fact": snippet, "citation": citation})
    return facts


def _split_context_chunks(context: str) -> list[str]:
    """优先提取 RAG 片段，无法匹配时回退到整段文本。"""
    rag_start = context.find("--- 从参考PDF中检索到的高度相关原文片段 ---")
    if rag_start != -1:
        rag_section = context[rag_start:]
        rag_end = rag_section.find("--- 原文片段结束 ---")
        if rag_end != -1:
            rag_section = rag_section[:rag_end]
        context_segment = rag_section
    else:
        context_segment = context

    chunk_pattern = re.compile(
        r"\[相关原文片段\s*(\d+)\](.*?)(?=\[相关原文片段\s*\d+\]|---\s*原文片段结束\s*---|\Z)",
        re.IGNORECASE | re.DOTALL,
    )
    chunks = [match.group(2).strip() for match in chunk_pattern.finditer(context_segment) if match.group(2).strip()]
    if not chunks:
        # 按段落粗分，避免返回超长文本
        paragraphs = [block.strip() for block in re.split(r"\n{2,}", context_segment) if block.strip()]
        return paragraphs[:4] if paragraphs else [context]
    return chunks


def _split_into_sentences(text: str) -> list[str]:
    sentences = [sentence.strip() for sentence in re.split(r"(?<=[。.!?])\s+", text) if sentence.strip() and len(sentence.strip()) >= 20]
    return sentences[:3]


def _derive_citation_anchor(text: str, section_id: str, chunk_idx: int) -> str:
    """尽量提取真实引用锚点，若缺失则以 chunk 索引回退。"""
    ref_match = re.search(r"\[ref:\s*([^\]]+)\]", text)
    if ref_match:
        return ref_match.group(1).strip()

    url_match = re.search(r"(https?://[^\s\]\)]+)", text)
    if url_match:
        return url_match.group(1).rstrip(").,;")

    source_match = re.search(r"(?:来源|Source|URL)[:：]\s*([^\n，。]+)", text)
    if source_match:
        return source_match.group(1).strip()

    page_match = re.search(r"(第[0-9一二三四五六七八九十百千]+页)", text)
    if page_match:
        identifier = page_match.group(1).replace(" ", "")
        return f"{section_id}#{identifier}"

    return f"{section_id}#rag{chunk_idx}"


__all__ = ["digest_node"]
