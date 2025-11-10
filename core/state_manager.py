"""工作流状态管理模块。

本模块提供工作流状态的Pydantic模型定义和适配器。
状态字段定义参考state_fields.py以保持一致性。

架构说明:
- GraphState (workflows/graph_state.py): LangGraph原生TypedDict接口
- WorkflowStateModel (本文件): Pydantic模型提供运行时验证
- STATE_FIELDS (state_fields.py): 单一数据源（未来计划整合）
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from config import Config
from services.vector_db import VectorDBManager

if TYPE_CHECKING:
    from core.context_components import ContextAssembler, ContextRepository, RAGService
else:
    # Import concrete types to satisfy Pydantic forward references when the model is built.
    from core.context_components import ContextAssembler, ContextRepository, RAGService


def _default_patch_list() -> list[dict[str, Any]]:
    return []


class WorkflowStateModel(BaseModel):
    """LangGraph工作流状态的类型化表示。"""

    task_id: str | None = None
    config: Config
    external_data: str | None = ""
    vector_db_manager: VectorDBManager | None = None
    refinement_count: int = 0

    # 规划/草稿
    skeleton_outline: dict[str, Any] | None = None
    section_digests: dict[str, Any] | None = None
    style_guide: str | None = None
    outline: dict[str, Any] | None = None
    draft_content: str | None = None
    draft_structure: dict[str, Any] | None = None
    context_repository: ContextRepository | None = None
    context_assembler: ContextAssembler | None = None
    rag_service: RAGService | None = None

    # 评审与研究
    critique: str | None = None
    knowledge_gaps: list[str] = Field(default_factory=list)
    research_brief: str | None = None
    patches: list[dict[str, Any]] = Field(default_factory=_default_patch_list)
    section_number_map: dict[int, str] | None = None  # 数字编号→UUID映射表（用于补丁应用）
    citation_data: dict[str, Any] | None = None
    structured_research_data: dict[str, Any] | None = Field(default_factory=dict)
    structured_critique: dict[str, Any] | None = None

    # 最终输出
    final_solution: str | None = None
    # 控制标志
    force_exit_refine: bool = False
    last_refine_had_effect: bool | None = None
    suggest_global_refine: bool = False

    model_config = ConfigDict(arbitrary_types_allowed=True)


class WorkflowStateAdapter:
    """用于安全操作LangGraph状态字典的工具类。"""

    @staticmethod
    def ensure(state: Mapping[str, Any]) -> WorkflowStateModel:
        """
        将原始LangGraph状态字典转换为类型化模型。
        为缺失的可选字段提供默认值。
        """
        try:
            return WorkflowStateModel(**dict(state))
        except ValidationError as exc:
            raise ValueError(f"工作流状态验证失败: {exc}") from exc


# 重建模型以在模块导入时解析前向引用
WorkflowStateModel.model_rebuild()
