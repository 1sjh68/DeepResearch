"""工作流图状态定义（LangGraph接口）。

本模块定义LangGraph使用的TypedDict状态结构。
状态字段应与core/state_manager.py中的WorkflowStateModel保持一致。
状态字段定义参考core/state_fields.py以保持同步。
"""

from typing import TypedDict

from config import Config
from core.context_components import ContextAssembler, ContextRepository, RAGService
from services.vector_db import VectorDBManager


class GraphState(TypedDict, total=False):
    """
    表示工作流图的状态。
    """

    # --- 核心上下文 ---
    task_id: str | None
    config: Config
    external_data: str | None
    vector_db_manager: VectorDBManager | None
    refinement_count: int

    # --- 规划/草稿 ---
    skeleton_outline: dict | None
    section_digests: dict | None
    style_guide: str | None
    outline: dict | None
    draft_content: str | None
    draft_structure: dict | None
    context_repository: ContextRepository | None
    context_assembler: ContextAssembler | None
    rag_service: RAGService | None

    # --- 评审与研究 ---
    critique: str | None
    knowledge_gaps: list[str] | None
    research_brief: str | None
    structured_research_data: dict | None  # 结构化研究数据
    citation_data: dict | None  # 引用管理数据
    patches: list[dict] | None
    section_number_map: dict[int, str] | None  # 数字编号→UUID映射表（用于补丁应用）

    # --- 最终输出 ---
    final_solution: str | None
    # --- 控制标志 ---
    force_exit_refine: bool | None
    last_refine_had_effect: bool | None
    suggest_global_refine: bool | None
