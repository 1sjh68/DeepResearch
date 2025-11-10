# workflows/graph_runner.py

import logging
import os
from collections.abc import Mapping
from typing import Any, cast

from config import Config
from core.progress import StepOutput, StepPayload
from services.vector_db import VectorDBManager
from utils.file_handler import load_external_data
from utils.progress_tracker import (
    EnhancedProgressTracker,
    register_tracker,
    safe_pulse,
    unregister_tracker,
)
from workflows.graph_builder import build_graph
from workflows.graph_state import GraphState

PROGRESS_STEP_SEQUENCE = [
    ("style_guide_node", "生成风格指南"),
    ("plan_node", "生成文档大纲"),
    ("skeleton_node", "构建骨架目录"),
    ("digest_node", "整理资料索引卡"),
    ("topology_writer_node", "拓扑写作初稿"),
    ("critique_node", "评审草稿"),
    ("research_node", "执行外部研究"),
    ("refine_node", "生成内容优化补丁"),
    ("apply_patches_node", "应用内容补丁"),
    ("polish_node", "润色文档"),
    ("memory_node", "保存经验"),
]


def run_graph_workflow(
    config: Config,
    vector_db_manager: VectorDBManager | None,
    log_handler: logging.Handler | None = None,
) -> str:
    """
    (同步版本) 运行完整的内容创作工作流图。
    """
    tracker = EnhancedProgressTracker(config.task_id, total_steps=len(PROGRESS_STEP_SEQUENCE))
    for step_name, detail in PROGRESS_STEP_SEQUENCE:
        tracker.add_step(step_name, detail)
    register_tracker(tracker)

    # 如果提供了日志处理器，则将其添加到根记录器
    root_logger = logging.getLogger()
    if log_handler:
        root_logger.addHandler(log_handler)

    try:
        # 准备初始状态
        logging.info("--- 准备工作流初始状态 ---")

        # 修复 4: 向量数据库检索添加降级处理
        retrieved_exps = []
        if vector_db_manager:
            try:
                retrieved_exps = vector_db_manager.hybrid_retrieve_experience(config.user_problem)
                logging.info("检索到 %d 条经验记录", len(retrieved_exps))
            except TimeoutError as e:
                logging.warning("向量数据库检索超时 - 不使用缓存经验继续执行: %s", e)
            except Exception as e:
                logging.warning("向量数据库检索失败(%s) - 不使用缓存继续执行", type(e).__name__)

        exp_texts = [f"---历史经验 {i + 1} (混合相关度: {exp.get('hybrid_score', exp.get('distance', -1)):.4f})---\n{exp.get('document')}" for i, exp in enumerate(retrieved_exps)]
        retrieved_experience_text = "\n\n===== 检索到的相关历史经验 =====\n" + "\n\n".join(exp_texts) + "\n===== 历史经验结束 =====\n\n" if exp_texts else ""

        loaded_ext_data = load_external_data(config, config.external_data_files or [])
        initial_external_data = retrieved_experience_text + loaded_ext_data

        # 修复 2: GraphState 完整初始化保证
        initial_state = cast(
            GraphState,
            {
                # 必要字段 - 保证非空
                "config": config,
                "task_id": config.task_id,
                "refinement_count": 0,

                # 可选字段 - 初始化为默认值
                "vector_db_manager": vector_db_manager,
                "external_data": initial_external_data or "",
                "skeleton_outline": None,
                "section_digests": None,
                "style_guide": "",
                "outline": None,
                "draft_content": "",
                "context_repository": None,
                "context_assembler": None,
                "rag_service": None,

                # 评审相关
                "critique": "",
                "knowledge_gaps": [],
                "research_brief": "",
                "structured_research_data": None,
                "citation_data": None,
                "patches": [],

                # 控制字段
                "force_exit_refine": False,
                "last_refine_had_effect": None,
            },
        )

        # 验证必要字段
        required_fields = ["config", "task_id", "refinement_count"]
        for field in required_fields:
            if field not in initial_state or initial_state[field] is None:
                raise ValueError(f"Missing required field: {field}")
        logging.info("初始状态验证成功")
        # 构建并调用图
        logging.info("--- 正在构建并调用同步工作流图 ---")
        # LangGraph 默认递归限制较低，这里根据配置的迭代上限动态放宽。
        # 每次迭代包含: critique->research/refine->apply_patches + 条件边判断
        # 保守估计每轮迭代需要8-10个节点调用（包括条件边）
        estimated_steps_per_iteration = 10
        # 基础节点(style_guide+plan+draft+polish+memory) + 迭代循环 + 安全余量
        recursion_limit = max(50, 10 + (config.max_refinement_rounds * estimated_steps_per_iteration) + 20)
        logging.info(f"LangGraph recursion_limit set to: {recursion_limit} (max_refinement_rounds={config.max_refinement_rounds})")
        app = build_graph()
        use_simple = os.getenv("USE_SIMPLE_RUNNER", "true").lower() == "true"
        logging.info(
            "启动工作流。运行器模式: %s (USE_SIMPLE_RUNNER=%s)",
            "simple" if use_simple else "invoke",
            os.getenv("USE_SIMPLE_RUNNER", "true"),
        )
        if use_simple:
            # Default to SIMPLE runner to avoid environment-specific graph cycle stalls.
            logging.info("使用SIMPLE运行器（线性，无LangGraph循环）。")
            return _run_simple_runner(
                config,
                vector_db_manager,
            )
        logging.info("启动LangGraph invoke()（图模式）...")
        try:
            # Save graph visualization into the current session directory (best-effort)
            try:
                if config.session_dir and os.path.isdir(config.session_dir):
                    out_path = os.path.join(config.session_dir, "graph.png")
                    app.get_graph().draw_mermaid_png(output_file_path=out_path)
                    logging.info("图可视化已保存到 %s", out_path)
            except Exception as viz_err:
                logging.warning("无法保存图可视化: %s", viz_err)

            # 使用同步 invoke() 执行整个有向图，避免流式在循环边上偶发阻塞。
            # 节点内部仍会通过 pulse 输出细粒度心跳。
            final_state = app.invoke(initial_state, config={"recursion_limit": recursion_limit})
            if not isinstance(final_state, dict):
                logging.error("invoke()未返回字典状态，得到: %s", type(final_state))
                return "错误：工作流未返回有效最终状态。"
        except RecursionError as e:
            logging.error(f"RecursionError during graph execution: {e}")
            raise RuntimeError(f"工作流达到递归限制({recursion_limit}): {e}")
        except Exception as e:
            logging.error(f"图执行期间发生意外错误: {e}", exc_info=True)
            raise

        final_solution = final_state.get("final_solution", "")
        refinement_count = final_state.get("refinement_count", 0)
        logging.info(f"工作流完成，refinement_count={refinement_count}, max={config.max_refinement_rounds}")

        if not final_solution or final_solution.isspace():
            logging.error(f"工作流执行完毕，但未能生成最终解决方案。refinement_count={refinement_count}")
            return "错误：工作流未能生成最终解决方案。"

        logging.info(f"--- 同步工作流图执行完毕 (completed {refinement_count}/{config.max_refinement_rounds} iterations) ---")
        return final_solution

    except Exception as e:
        logging.critical(f"工作流执行期间发生严重错误: {e}", exc_info=True)
        return f"错误：工作流执行失败，原因: {e}"
    finally:
        # 修复 7: 安全清理，避免单个异常导致其他清理跳过
        cleanup_errors = []

        if log_handler:
            try:
                root_logger.removeHandler(log_handler)
                logging.debug("日志处理器移除成功")
            except Exception as e:
                cleanup_errors.append(f"removeHandler failed: {e}")
                logging.warning("移除日志处理器失败: %s", e)

        try:
            unregister_tracker(config.task_id)
            logging.debug("追踪器注销成功")
        except Exception as e:
            cleanup_errors.append(f"unregister_tracker failed: {e}")
            logging.warning("注销追踪器失败: %s", e)

        if cleanup_errors:
            logging.warning("清理错误: %s", "; ".join(cleanup_errors))


def _run_simple_runner(
    config: Config,
    vector_db_manager: VectorDBManager | None,
) -> str:
    """线性回退运行器，不使用LangGraph而是按顺序执行节点。

    在图循环/流式处理导致停滞的环境中很有用。
    """
    state = cast(
        GraphState,
        {
            "config": config,
            "vector_db_manager": vector_db_manager,
            "external_data": "",
            "task_id": config.task_id,
            "refinement_count": 0,
        },
    )

    # Load external data (same as graph runner)
    from utils.file_handler import load_external_data

    def _unwrap(result: StepOutput | StepPayload) -> tuple[dict[str, Any], str]:
        if isinstance(result, StepOutput):
            return dict(result.data), result.detail or ""
        if isinstance(result, Mapping):
            return dict(result), ""
        raise TypeError(f"不支持的节点输出类型: {type(result)}")

    def _update_state(payload: dict[str, Any]) -> None:
        """更新图状态同时满足TypedDict类型检查。"""
        cast(dict[str, Any], state).update(payload)

    retrieved_exps = []
    if vector_db_manager:
        try:
            retrieved_exps = vector_db_manager.hybrid_retrieve_experience(config.user_problem)
        except TimeoutError as e:
            logging.warning("向量数据库检索超时 - 不使用缓存经验继续执行: %s", e)
            retrieved_exps = []
        except Exception as e:
            logging.warning("向量数据库检索失败(%s) - 不使用缓存继续执行", type(e).__name__)
            retrieved_exps = []
    exp_texts = [f"---历史经验 {i + 1} (混合相关度: {exp.get('hybrid_score', exp.get('distance', -1)):.4f})---\n{exp.get('document')}" for i, exp in enumerate(retrieved_exps)]
    retrieved_experience_text = "\n\n===== 检索到的相关历史经验 =====\n" + "\n\n".join(exp_texts) + "\n===== 历史经验结束 =====\n\n" if exp_texts else ""
    loaded_ext = load_external_data(config, config.external_data_files or [])
    state["external_data"] = retrieved_experience_text + loaded_ext

    # Execute nodes sequentially
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

    def _pulse(message: str) -> None:
        """更新CLI脉冲而不过度充斥日志。"""
        safe_pulse(config.task_id, message)

    # style guide, plan, skeleton, digest, topology writer
    _pulse("加载风格指南节点…")
    style_payload, style_detail = _unwrap(style_guide_node(state))
    _update_state(style_payload)

    _pulse("生成文档大纲…")
    plan_payload, plan_detail = _unwrap(plan_node(state))
    _update_state(plan_payload)

    _pulse("构建骨架…")
    skeleton_payload, skeleton_detail = _unwrap(skeleton_node(state))
    _update_state(skeleton_payload)

    _pulse("生成索引卡片…")
    digest_payload, digest_detail = _unwrap(digest_node(state))
    _update_state(digest_payload)

    _pulse("拓扑写作初稿…")
    topo_payload, topo_detail = _unwrap(topology_writer_node(state))
    _update_state(topo_payload)

    # refinement loop
    max_rounds = config.max_refinement_rounds
    for iteration_index in range(max_rounds):
        # critique
        critique_payload, critique_detail = _unwrap(critique_node(state))
        _update_state(critique_payload)
        # research if any gaps and enabled
        gaps = state.get("knowledge_gaps") or []
        if config.enable_web_research and gaps:
            research_payload, research_detail = _unwrap(research_node(state))
            _update_state(research_payload)
        # refine
        refine_payload, refine_detail = _unwrap(refine_node(state))
        _update_state(refine_payload)
        # apply patches (this increments refinement_count)
        apply_payload, apply_detail = _unwrap(apply_patches_node(state))
        _update_state(apply_payload)

        if state.get("force_exit_refine") and not getattr(config, "disable_early_exit", False):
            break

        # 使用 refinement_count 而非 iteration_index 判断，因为 apply_patches_node 内部会更新它
        # refinement_count 表示实际完成的迭代轮数，更准确反映工作流状态
        current_refinement = state.get("refinement_count", 0)
        if current_refinement >= max_rounds:
            break
        # Ignore early-exit flags in simple runner to guarantee full iterations

    # polish and memory
    _pulse("润色文档…")
    polish_payload, polish_detail = _unwrap(polish_node(state))
    _update_state(polish_payload)

    _pulse("保存经验…")
    memory_payload, memory_detail = _unwrap(memory_node(state))
    _update_state(memory_payload)

    final = state.get("final_solution", "")
    if not final or final.isspace():
        return "错误：工作流未能生成最终解决方案。"
    return final
