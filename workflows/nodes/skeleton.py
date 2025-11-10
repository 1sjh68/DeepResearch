from __future__ import annotations

import logging
import re
from typing import Any

from core.progress import StepOutput, step_result, workflow_step
from core.state_manager import WorkflowStateAdapter
from utils.id_manager import ensure_section_id
from utils.progress_tracker import safe_pulse
from workflows.graph_state import GraphState


@workflow_step("skeleton_node", "构建骨架目录")
def skeleton_node(state: GraphState) -> StepOutput:
    """引导"三段式"骨架：为每一节定义必须覆盖的要点与组织方式。"""
    workflow_state = WorkflowStateAdapter.ensure(state)
    outline = workflow_state.outline
    if not outline or not outline.get("outline"):
        raise ValueError("skeleton_node 需要有效的 plan_node 输出（outline 数据缺失）。")

    config = workflow_state.config
    safe_pulse(config.task_id, "三段式 · 第1步：整理骨架")

    skeleton_outline = {
        "title": outline.get("title", "Untitled Document"),
        "sections": [_build_skeleton_section(chapter, depth=0) for chapter in outline.get("outline", [])],
    }

    logging.info("骨架构建完成，共生成 %s 个章节节点。", _count_nodes(skeleton_outline["sections"]))
    return step_result({"skeleton_outline": skeleton_outline}, "骨架+清单已就绪")


def _build_skeleton_section(chapter: dict[str, Any], depth: int) -> dict[str, Any]:
    """为单个章节生成骨架节点。"""
    # 🔧 修复：使用 ensure_section_id 确保 ID 被保存
    section_id = ensure_section_id(chapter)
    title = chapter.get("title", f"未命名章节-{section_id[:8]}")
    must_include = _derive_must_include(chapter)
    organization_hint = _derive_organization_hint(chapter, depth)
    children = [_build_skeleton_section(child, depth + 1) for child in chapter.get("sections", []) or []]

    return {
        "id": section_id,
        "title": title,
        "must_include": must_include,
        "organization_hint": organization_hint,
        "children": children,
    }


def _derive_must_include(chapter: dict[str, Any]) -> list[str]:
    description = chapter.get("description", "") or ""
    sentences = [frag.strip() for frag in re.split(r"[。！!？?；;]", description) if frag and len(frag.strip()) >= 6]
    child_titles = [child.get("title", "").strip() for child in chapter.get("sections", []) or [] if child.get("title")]
    child_requirements = [f"比较/覆盖子主题：{child_title}" for child_title in child_titles]

    merged = sentences + child_requirements
    if not merged:
        merged = ["梳理本节核心概念、关键数据与结论。"]
    return merged[:6]


def _derive_organization_hint(chapter: dict[str, Any], depth: int) -> str:
    child_titles = [child.get("title", "").strip() for child in chapter.get("sections", []) or [] if child.get("title")]
    if not child_titles:
        return "先定义概念，再列事实/数据，最后总结启示。"

    joined = " → ".join(child_titles[:4])
    prefix = "叶子小节" if depth >= 1 else "父级章节"
    return f"{prefix}建议按顺序串联：{joined}，并在段末进行横向比较。"


def _count_nodes(sections: list[dict[str, Any]]) -> int:
    total = 0
    for section in sections:
        total += 1
        children = section.get("children", [])
        if children:
            total += _count_nodes(children)
    return total


__all__ = ["skeleton_node"]
