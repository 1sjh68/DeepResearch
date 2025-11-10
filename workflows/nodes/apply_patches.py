from __future__ import annotations

import logging

from core.patch_manager import apply_fine_grained_edits
from core.progress import StepOutput, step_result, workflow_step
from core.state_manager import WorkflowStateAdapter
from utils.iteration_storage import archive_iteration_snapshot
from utils.progress_tracker import safe_pulse
from utils.text_processor import safe_get_dict  # 修复 8: 添加类型检查
from workflows.graph_state import GraphState


def _should_exit_refinement_loop(has_gaps: bool, open_issues: bool, had_effect: bool) -> bool:
    """
    修复 6: 清晰的退出条件逻辑

    规则:
    1. 如果有问题但补丁无效 → 退出（避免无限循环）
    2. 如果既无gaps也无问题 → 退出（完美了）
    3. 其他情况 → 继续迭代
    """
    has_problems = has_gaps or open_issues

    # 无问题 → 退出（成功完成）
    if not has_problems:
        logging.info("No issues found, ready for final polish")
        return True

    # 有问题但补丁无效 → 退出（避免无限循环）
    if has_problems and not had_effect:
        logging.info("Issues remain but patches ineffective, exiting to prevent loop")
        return True

    # 有问题但补丁有效 → 继续
    logging.info("Issues remain but patches effective, continuing refinement")
    return False


@workflow_step("apply_patches_node", "应用内容补丁")
def apply_patches_node(state: GraphState) -> StepOutput:
    workflow_state = WorkflowStateAdapter.ensure(state)
    config = workflow_state.config
    patches = workflow_state.patches or []
    draft_content = workflow_state.draft_content or ""
    structured_critique = getattr(workflow_state, "structured_critique", None) or {}
    section_number_map = getattr(workflow_state, "section_number_map", None) or {}  # 获取数字映射表
    current_iteration = workflow_state.refinement_count + 1
    max_rounds = config.max_refinement_rounds
    logging.info(
        "[RefineLoop] Iteration %s/%s -> apply_patches_node",
        current_iteration,
        max_rounds,
    )
    safe_pulse(config.task_id, f"迭代 {current_iteration}/{max_rounds} · 应用补丁中...")

    if not draft_content:
        logging.warning("apply_patches_node: No draft_content found to apply patches to. Skipping.")
        return step_result({}, f"迭代 {current_iteration}/{max_rounds}，缺少草稿内容")

    if not patches:
        logging.info("  - 无补丁需要应用，提前结束当前迭代。")
        # 修复 3: 统一递归计数逻辑 - 始终 +1
        refinement_count = workflow_state.refinement_count + 1
        archive_iteration_snapshot(
            config,
            refinement_count,
            "refine_no_changes",
            draft_content,
        )
        # 修复 8: 使用安全的类型检查函数
        structured_open_issues = (
            safe_get_dict(structured_critique, "priority_issues") or
            safe_get_dict(structured_critique, "knowledge_gaps")
        )
        has_gaps = bool(workflow_state.knowledge_gaps)
        open_issues = has_gaps or structured_open_issues
        previous_effect = workflow_state.last_refine_had_effect
        if open_issues:
            should_exit = previous_effect is False
            if should_exit:
                logging.warning("  - 连续两轮未生成补丁，终止修订循环并提示手动处理剩余问题。")
            else:
                logging.info("  - 本轮未生成补丁，将在下一轮再次尝试。")
        else:
            should_exit = True
        return step_result(
            {
                "refinement_count": refinement_count,
                "patches": [],
                "force_exit_refine": should_exit,
                "last_refine_had_effect": False,
                "structured_critique": structured_critique,
            },
            f"迭代 {refinement_count}/{max_rounds}，无补丁",
        )

    logging.info("  - 正在应用 %s 个补丁...", len(patches))
    if section_number_map:
        logging.info("  - 数字映射表可用：%d 个编号 → UUID", len(section_number_map))
    edit_result = apply_fine_grained_edits(draft_content, patches, section_number_map=section_number_map)
    updated_draft = edit_result.updated_text
    had_effect = edit_result.had_effect

    # 增强诊断：检测补丁完全失败的情况
    if len(patches) > 0 and edit_result.sections_modified == 0 and edit_result.successful_edits == 0:
        logging.error("=" * 60)
        logging.error("警告：所有补丁应用失败！")
        logging.error(f"  - 尝试应用: {len(patches)} 个补丁")
        logging.error(f"  - 成功修改: {edit_result.sections_modified} 个章节")
        logging.error(f"  - 成功编辑: {edit_result.successful_edits} 条")
        if edit_result.failed_edits:
            logging.error(f"  - 失败章节ID: {edit_result.failed_edits[:5]}")  # 显示前5个
        logging.error("  - 建议：章节ID映射可能存在问题，或需要采用全局修订模式")
        logging.error("=" * 60)

        # 标记建议使用全局修订（未来可在refine节点中检测此标记）
        workflow_state.suggest_global_refine = True

    # 修复 8: 使用安全的类型检查函数
    structured_open_issues = (
        safe_get_dict(structured_critique, "priority_issues") or
        safe_get_dict(structured_critique, "knowledge_gaps")
    )
    has_gaps = bool(workflow_state.knowledge_gaps)
    detail_stats = f"章节变更 {edit_result.sections_modified} 个，命中修订 {edit_result.successful_edits} 条"
    logging.info("  - 本轮补丁统计：%s", detail_stats)
    safe_pulse(
        config.task_id,
        f"迭代 {current_iteration}/{max_rounds} · 已应用补丁 {len(patches)} 个，准备归档快照...",
    )

    refinement_count = workflow_state.refinement_count + 1
    archive_iteration_snapshot(
        config,
        refinement_count,
        "refine",
        updated_draft,
    )
    detail_msg = f"迭代 {refinement_count}/{max_rounds}，应用补丁 {len(patches)} 个（{detail_stats}）"

    # 修复 6: 使用清晰的函数判断退出条件
    force_exit_refine = _should_exit_refinement_loop(
        has_gaps=has_gaps,
        open_issues=structured_open_issues,
        had_effect=had_effect
    )

    return step_result(
        {
            "draft_content": updated_draft,
            "refinement_count": refinement_count,
            "patches": [],
            "force_exit_refine": force_exit_refine,
            "last_refine_had_effect": had_effect,
            "structured_critique": structured_critique,
        },
        detail_msg,
    )


__all__ = ["apply_patches_node"]
