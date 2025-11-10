"""工作流状态字段的统一定义。

本模块提供单一数据源，用于定义工作流状态的所有字段。
GraphState和WorkflowStateModel都从此处生成，确保一致性。
"""

from __future__ import annotations

from typing import Any, TypeAlias

from pydantic import Field

# 字段定义格式: (类型注解, 默认值, Field配置(可选))
# 默认值使用 ... 表示必需字段
# 默认值使用 None 或具体值表示可选字段

FieldDefinition: TypeAlias = tuple[Any, Any] | tuple[Any, Any, Any]

# 状态字段统一定义
# 注意：此定义是唯一数据源，GraphState和WorkflowStateModel都从此生成
STATE_FIELDS: dict[str, FieldDefinition] = {
    # --- 核心上下文 ---
    "task_id": (str | None, None),
    "config": ("Config", ...),  # 必需字段，使用字符串避免循环导入
    "external_data": (str | None, ""),
    "vector_db_manager": ("VectorDBManager | None", None),
    "refinement_count": (int, 0),

    # --- 规划/草稿 ---
    "skeleton_outline": (dict[str, Any] | None, None),
    "section_digests": (dict[str, Any] | None, None),
    "style_guide": (str | None, None),
    "outline": (dict[str, Any] | None, None),
    "draft_content": (str | None, None),
    "draft_structure": (dict[str, Any] | None, None),
    "context_repository": ("ContextRepository | None", None),
    "context_assembler": ("ContextAssembler | None", None),
    "rag_service": ("RAGService | None", None),

    # --- 评审与研究 ---
    "critique": (str | None, None),
    "knowledge_gaps": (list[str], Field(default_factory=list)),
    "research_brief": (str | None, None),
    "patches": (list[dict[str, Any]], Field(default_factory=lambda: [])),
    "citation_data": (dict[str, Any] | None, None),
    "structured_research_data": (dict[str, Any] | None, Field(default_factory=dict)),
    "structured_critique": (dict[str, Any] | None, None),

    # --- 最终输出 ---
    "final_solution": (str | None, None),

    # --- 控制标志 ---
    "force_exit_refine": (bool, False),
    "last_refine_had_effect": (bool | None, None),
    "suggest_global_refine": (bool, False),
}


def get_required_fields() -> list[str]:
    """返回所有必需字段的列表。"""
    return [
        field_name
        for field_name, field_def in STATE_FIELDS.items()
        if field_def[1] is ...
    ]


def get_field_type(field_name: str) -> Any:
    """获取字段的类型注解。"""
    if field_name not in STATE_FIELDS:
        raise ValueError(f"未知字段: {field_name}")
    return STATE_FIELDS[field_name][0]


def get_field_default(field_name: str) -> Any:
    """获取字段的默认值。"""
    if field_name not in STATE_FIELDS:
        raise ValueError(f"未知字段: {field_name}")
    return STATE_FIELDS[field_name][1]


def validate_state_consistency() -> tuple[bool, list[str]]:
    """验证STATE_FIELDS定义的一致性。

    返回:
        (是否一致, 错误列表)
    """
    errors = []

    # 检查必需字段至少包含config
    required = get_required_fields()
    if "config" not in required:
        errors.append("config必须是必需字段")

    # 检查所有字段都有类型注解
    for field_name, field_def in STATE_FIELDS.items():
        if len(field_def) < 2:
            errors.append(f"字段{field_name}缺少类型或默认值")

    return len(errors) == 0, errors


__all__ = [
    "STATE_FIELDS",
    "get_required_fields",
    "get_field_type",
    "get_field_default",
    "validate_state_consistency",
]

