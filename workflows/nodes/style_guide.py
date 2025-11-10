from __future__ import annotations

from core.progress import StepOutput, step_result, workflow_step
from core.state_manager import WorkflowStateAdapter
from utils.progress_tracker import safe_pulse
from workflows.graph_state import GraphState
from workflows.nodes.sub_workflows.planning import generate_style_guide


@workflow_step("style_guide_node", "生成风格指南")
def style_guide_node(state: GraphState) -> StepOutput:
    workflow_state = WorkflowStateAdapter.ensure(state)
    config = workflow_state.config
    safe_pulse(config.task_id, "生成风格与声音指南中...")
    style_guide = generate_style_guide(config)
    return step_result({"style_guide": style_guide}, "生成风格指南完成")


__all__ = ["style_guide_node"]
