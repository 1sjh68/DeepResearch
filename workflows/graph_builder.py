import logging

from langgraph.graph import END, StateGraph

from utils.progress_tracker import safe_pulse
from workflows.graph_nodes import (
    apply_patches_node,
    critique_node,
    digest_node,
    memory_node,
    plan_node,
    polish_node,
    refine_node,
    research_node,
    skeleton_node,
    style_guide_node,
    topology_writer_node,
)
from workflows.graph_state import GraphState


def should_research(state: GraphState) -> str:
    """
    根据配置和识别出的知识空白，判断是否需要执行研究。
    """
    config = state.get("config")
    if config is None:
        raise KeyError("GraphState 缺少必需的 'config' 字段")
    knowledge_gaps = state.get("knowledge_gaps") or []
    if config.enable_web_research and knowledge_gaps:
        logging.info("---决策: 网络研究已启用且发现知识空白。进入研究阶段。---")
        safe_pulse(getattr(config, "task_id", None), "决策: 进入研究节点")
        return "research_node"
    else:
        if not config.enable_web_research:
            logging.info("---决策: 配置中已禁用网络研究。跳过研究。---")
        else:
            logging.info("---决策: 未发现知识空白。跳过研究。---")
        safe_pulse(getattr(config, "task_id", None), "决策: 跳过研究，进入生成补丁")
        return "refine_node"


def should_continue_refining(state: GraphState) -> str:
    """
    判断是应该继续优化循环，还是进入最终的润色阶段。
    """
    refinement_count = state.get("refinement_count", 0)
    config = state.get("config")
    if config is None:
        raise KeyError("GraphState 缺少必需的 'config' 字段")
    max_refinements = config.max_refinement_rounds

    logging.info(f"[should_continue_refining] Current state: refinement_count={refinement_count}, max={max_refinements}")

    # 若上游节点设置了提前退出标志（例如无新的知识空白与补丁），且未禁用提前退出，则直接进入润色
    exit_flag = bool(state.get("force_exit_refine"))
    disable_early_exit = bool(config.disable_early_exit)
    if exit_flag and not disable_early_exit:
        logging.info("[should_continue_refining] force_exit_refine=True -> Returning 'polish_node'")
        safe_pulse(state.get("task_id"), "决策: 无需继续迭代，进入润色")
        return "polish_node"

    if refinement_count < max_refinements:
        remaining = max_refinements - refinement_count
        # refinement_count 表示已完成的轮数，所以下一轮是 refinement_count + 1
        next_round = refinement_count + 1
        logging.info(
            "---决策: 优化迭代 %s/%s -> 下一步=critique_node (剩余 %s)---",
            next_round,
            max_refinements,
            remaining,
        )
        logging.info("[should_continue_refining] Returning 'critique_node' to continue loop (early-exit disabled)")
        safe_pulse(
            state.get("task_id"),
            f"决策: 进入第 {next_round}/{max_refinements} 轮评审（不提前收敛）",
        )
        return "critique_node"
    else:
        logging.info(
            "---决策: 已达到最大优化轮数(%s)。进入润色阶段。---",
            max_refinements,
        )
        logging.info("[should_continue_refining] Returning 'polish_node' to exit loop")
        safe_pulse(state.get("task_id"), "决策: 达到迭代上限，进入润色")
        return "polish_node"


def build_graph():
    """
    构建并编译用于“深度研究”工作流的LangGraph。
    """
    workflow = StateGraph(GraphState)

    # Add all the nodes to the graph
    workflow.add_node("style_guide_node", style_guide_node)
    workflow.add_node("plan_node", plan_node)
    workflow.add_node("skeleton_node", skeleton_node)
    workflow.add_node("digest_node", digest_node)
    workflow.add_node("topology_writer_node", topology_writer_node)
    workflow.add_node("critique_node", critique_node)
    workflow.add_node("research_node", research_node)
    workflow.add_node("refine_node", refine_node)
    workflow.add_node("apply_patches_node", apply_patches_node)
    workflow.add_node("polish_node", polish_node)
    workflow.add_node("memory_node", memory_node)

    # Set the entry point
    workflow.set_entry_point("style_guide_node")

    # Define the edges
    workflow.add_edge("style_guide_node", "plan_node")
    workflow.add_edge("plan_node", "skeleton_node")
    workflow.add_edge("skeleton_node", "digest_node")
    workflow.add_edge("digest_node", "topology_writer_node")
    workflow.add_edge("topology_writer_node", "critique_node")

    # Conditional edge for research
    workflow.add_conditional_edges(
        "critique_node",
        should_research,
        {
            "research_node": "research_node",
            "refine_node": "refine_node",
        },
    )
    workflow.add_edge("research_node", "refine_node")
    workflow.add_edge("refine_node", "apply_patches_node")

    # Conditional edge for the refinement loop
    workflow.add_conditional_edges(
        "apply_patches_node",
        should_continue_refining,
        {
            "critique_node": "critique_node",
            "polish_node": "polish_node",
        },
    )

    workflow.add_edge("polish_node", "memory_node")
    workflow.add_edge("memory_node", END)

    # Compile the graph
    app = workflow.compile()

    logging.info("LangGraph编译成功。")
    # Visualization is now saved by the runner into the session directory.

    return app


if __name__ == "__main__":
    # This allows for visualizing the graph structure if the file is run directly
    graph_app = build_graph()
    # The graph is built and a visualization is saved, nothing else to run here.
    logging.info("图构建完成。可视化已保存到graph.png（如果已安装依赖）。")
