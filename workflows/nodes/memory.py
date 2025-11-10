from __future__ import annotations

import logging

from core.progress import StepOutput, step_result, workflow_step
from core.state_manager import WorkflowStateAdapter
from utils.progress_tracker import safe_pulse
from workflows.graph_state import GraphState
from workflows.nodes.sub_workflows.memory import accumulate_experience


@workflow_step("memory_node", "保存经验")
def memory_node(state: GraphState) -> StepOutput:
    workflow_state = WorkflowStateAdapter.ensure(state)
    config = workflow_state.config
    user_problem = config.user_problem
    final_solution = workflow_state.final_solution
    vector_db_manager = workflow_state.vector_db_manager

    if not final_solution or not vector_db_manager:
        logging.warning("memory_node: No final_solution or vector_db_manager found. Skipping experience accumulation.")
        safe_pulse(config.task_id, "经验存储跳过：缺少结果或向量库实例")
        return step_result({}, "跳过经验写入")

    # 记录向量数据库已集成混合检索功能
    logging.info("内存节点：利用混合检索优化的向量数据库存储经验")

    # 显示混合检索配置状态
    hybrid_config = getattr(config, "enable_hybrid_search", True)
    bm25_config = getattr(config, "enable_bm25_search", True)
    rerank_config = getattr(config, "enable_rerank", True)
    bm25_weight = getattr(config, "bm25_weight", 0.3)

    logging.info(f"  - 混合检索启用状态: {hybrid_config}")
    logging.info(f"  - BM25检索启用状态: {bm25_config} (权重: {bm25_weight})")
    logging.info(f"  - 交叉编码器重排启用状态: {rerank_config}")

    if hasattr(vector_db_manager, "hybrid_retrieve_experience"):
        logging.info("  - 向量数据库支持混合检索（向量+BM25）")
        if hasattr(vector_db_manager, "cross_encoder") and vector_db_manager.cross_encoder:
            logging.info("  - 支持交叉编码器重排优化")
        if hasattr(vector_db_manager, "bm25_index") and vector_db_manager.bm25_index:
            logging.info("  - 支持BM25文本检索")

    accumulate_experience(
        config,
        vector_db_manager,
        user_problem,
        final_solution,
        [],
        [],
        [],
    )
    return step_result({}, "经验写入完成")


__all__ = ["memory_node"]
