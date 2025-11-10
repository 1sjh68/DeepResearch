"""
Polish模块的主入口

包含polish_node和perform_structured_polish等主要功能
"""

from __future__ import annotations

import logging
from typing import Any

from core.progress import StepOutput, step_result, workflow_step
from core.state_manager import WorkflowStateAdapter
from planning.tool_definitions import (
    PolishModel,
    PolishSection,
    extract_references_from_text,
    validate_references_quality,
)
from utils.iteration_storage import archive_iteration_snapshot
from utils.progress_tracker import safe_pulse, safe_step_update
from workflows.graph_state import GraphState
from workflows.nodes.sub_workflows.polishing import perform_final_polish

from .citation_handler import initialize_citation_manager, integrate_fact_checking
from .content_assembler import assemble_final_content, extract_document_title
from .content_processor import polish_section_structured
from .quality_checker import (
    _validate_final_solution,
    calculate_quality_score,
    generate_modification_summary,
)
from .utils import parse_document_structure

POLISH_STEP_NAME = "polish_node"


@workflow_step(POLISH_STEP_NAME, "润色文档")
def polish_node(state: GraphState) -> StepOutput:
    """
    润色文档节点 - 升级为结构化输出
    """
    workflow_state = WorkflowStateAdapter.ensure(state)
    config = workflow_state.config
    draft_content = workflow_state.draft_content or ""
    style_guide = workflow_state.style_guide or ""
    critique = workflow_state.critique or ""
    structured_research_data = workflow_state.structured_research_data or {}

    # 引用功能配置
    enable_citations = getattr(config, "enable_citation_management", True)
    citation_style = getattr(config, "citation_style", "numeric")

    if not draft_content:
        logging.warning("polish_node: No draft_content found to polish. Returning empty solution.")
        return step_result({"final_solution": ""}, "缺少草稿内容")

    safe_pulse(
        config.task_id,
        f"迭代 {workflow_state.refinement_count}/{config.max_refinement_rounds} · 结构化润色中...",
    )

    try:
        # 执行结构化润色
        iteration_info = f"迭代 {workflow_state.refinement_count}/{config.max_refinement_rounds}"
        polished_result = perform_structured_polish(
            config=config,
            original_content=draft_content,
            style_guide=style_guide,
            critique=critique,
            research_data=structured_research_data,
            iteration_info=iteration_info,
            enable_citations=enable_citations,
            citation_style=citation_style,
        )

        # 集成事实核查功能
        fact_check_results = integrate_fact_checking(
            polished_result=polished_result,
            structured_research_data=structured_research_data,
            config=config,
        )

        # 将事实核查结果添加到polish_result中
        polished_result.fact_check_results = fact_check_results

        # 保存迭代快照
        archive_iteration_snapshot(
            config,
            workflow_state.refinement_count,
            "structured_polish",
            polished_result.model_dump_json(indent=2, ensure_ascii=False),
        )

        # 修复5: 验证最终内容的有效性
        final_solution = polished_result.polished_content
        valid, reason = _validate_final_solution(final_solution, config)

        if not valid:
            logging.error("最终内容验证失败: %s", reason)
            logging.warning("使用回退方案: 原始草稿")
            final_solution = draft_content if draft_content else "# 错误\n\n内容生成失败"
        else:
            logging.info("最终内容验证通过: %s", reason)

        return step_result(
            {
                "final_solution": final_solution,
                "polish_model": polished_result,
                "fact_check_results": fact_check_results,
                "metadata": {
                    "sections_count": len(polished_result.sections),
                    "total_references": len(polished_result.all_references),
                    "quality_score": polished_result.overall_quality_score,
                    "validation_needed": polished_result.validation_needed,
                    "fact_check_performed": len(fact_check_results) > 0
                    if fact_check_results
                    else False,  # noqa: E501
                    "validation_status": reason,
                },
            },
            "结构化润色完成",
        )

    except Exception as e:
        logging.error(f"结构化润色失败，回退到传统模式: {e}", exc_info=True)
        return polish_node_fallback(state)


def polish_node_fallback(state: GraphState) -> StepOutput:
    """
    润色节点的回退机制 - 使用传统文本输出
    """
    workflow_state = WorkflowStateAdapter.ensure(state)
    config = workflow_state.config
    draft_content = workflow_state.draft_content or ""
    style_guide = workflow_state.style_guide or ""

    if not draft_content:
        logging.warning(
            "polish_node_fallback: No draft_content found to polish. Returning empty solution."
        )  # noqa: E501
        return step_result({"final_solution": ""}, "缺少草稿内容")

    safe_pulse(config.task_id, "回退模式 · 传统润色中...")

    iteration_info = "传统润色模式"
    polished_solution = perform_final_polish(
        config, draft_content, style_guide, iteration_info=iteration_info
    )  # noqa: E501

    return step_result({"final_solution": polished_solution}, "传统润色完成")


def perform_structured_polish(
    config,
    original_content: str,
    style_guide: str,
    critique: str = "",
    research_data: dict[str, Any] | None = None,
    iteration_info: str | None = None,
    enable_citations: bool = True,
    citation_style: str = "numeric",
) -> PolishModel:
    """
    执行结构化润色，返回PolishModel

    Args:
        config: 配置对象
        original_content: 原始内容
        style_guide: 风格指南
        critique: 评审意见
        research_data: 研究数据
        iteration_info: 迭代信息
        enable_citations: 是否启用引用
        citation_style: 引用风格

    Returns:
        PolishModel: 结构化的润色结果
    """
    logging.info("--- 开始结构化润色流程 ---")

    # 初始化引用管理器
    citation_manager = None
    if enable_citations:
        citation_manager = initialize_citation_manager(research_data)

    # 解析文档结构
    sections = parse_document_structure(original_content)
    total_sections = len(sections) or 1
    polished_sections = []
    all_references = []
    fact_check_points = []

    for i, section in enumerate(sections):
        logging.info(f"处理章节 {i + 1}/{len(sections)}: {section.get('title', '未命名')}")
        safe_step_update(
            config.task_id,
            POLISH_STEP_NAME,
            (i / total_sections) * 100.0,
            f"润色章节 {i + 1}/{len(sections)}",
        )

        try:
            # 提取章节引用
            section_references = extract_references_from_text(section["content"])

            # 执行章节润色
            polished_section = polish_section_structured(
                config=config,
                section=section,
                style_guide=style_guide,
                critique=critique,
                research_data=research_data,
                references=section_references,
                iteration_info=f"{iteration_info} · 章节 {i + 1}",
            )

            polished_sections.append(polished_section)
            all_references.extend(polished_section.references)

            safe_step_update(
                config.task_id,
                POLISH_STEP_NAME,
                ((i + 1) / total_sections) * 100.0,
                f"完成润色章节 {i + 1}/{len(sections)}",
            )

            # 如果启用引用标注，处理主张提取和引用匹配
            if citation_manager and enable_citations:
                try:
                    claims = citation_manager.extract_claims(section["content"])
                    matches = citation_manager.align_claims_to_sources(claims)
                    citation_manager.citations.extend(matches)
                except Exception as e:
                    logging.warning(f"章节 {i + 1} 引用处理失败: {e}")

            # 收集事实核查点（向后兼容无该字段的旧模型）
            _points = getattr(polished_section, "fact_check_points", None)
            if _points:
                fact_check_points.extend(_points)

        except Exception as e:
            logging.error(f"章节 {section.get('title', '未命名')} 润色失败: {e}")
            # 回退到原始章节
            section_content = section.get("content", "")
            polished_sections.append(
                PolishSection(
                    section_id=section.get("section_id", f"fallback_{i}"),
                    title=section.get("title", f"章节 {i + 1}"),
                    content=section_content,
                    original_content=section.get("original_content", section_content),
                    modifications=[],
                    references=[],
                    quality_metrics=None,
                    word_count=len(section_content),
                    revision_notes="润色失败，保持原样",
                )
            )

    # 组装最终内容
    document_title = extract_document_title(original_content)
    polished_content = assemble_final_content(
        sections=polished_sections,
        citation_manager=citation_manager,
        citation_style=citation_style,
        document_title=document_title,
    )

    # 计算整体质量评分
    overall_quality_score = calculate_quality_score(polished_sections)

    # 生成修改摘要
    modification_summary = generate_modification_summary(polished_sections)
    logging.info(f"润色完成 - {modification_summary}")

    # 验证引用质量
    validation_needed = False
    try:
        all_references_validated = validate_references_quality(all_references)
        if all_references_validated:
            validation_needed = not all(
                ref.confidence_level > 0.7 for ref in all_references_validated
            )  # noqa: E501
    except Exception:
        pass

    return PolishModel(
        document_title=document_title,  # ← 新增
        original_content=original_content,  # ← 新增
        polished_content=polished_content,
        sections=polished_sections,
        all_references=all_references,
        overall_quality_score=overall_quality_score,
        modification_summary=modification_summary,
        validation_needed=validation_needed,
        fact_check_points=fact_check_points,
        metadata={},
        fact_check_results=[],
    )
