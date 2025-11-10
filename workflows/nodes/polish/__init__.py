"""
Polish模块 - 文档润色功能

模块结构：
- polish_main.py - 主入口函数（polish_node, perform_structured_polish）
- content_processor.py - 核心润色处理逻辑
- quality_checker.py - 质量检查和评分
- content_assembler.py - 内容组装和去重
- citation_handler.py - 引用和脚注处理
- utils.py - 通用工具函数
"""

from __future__ import annotations

# 引用处理器
from .citation_handler import (
    initialize_citation_manager,
    integrate_fact_checking,
    render_citations_with_footnotes,
)

# 内容组装器
from .content_assembler import (
    assemble_final_content,
    extract_document_title,
)

# 内容处理器
from .content_processor import (
    build_section_polish_prompt,
    polish_section_structured,
    polish_section_text_fallback,
    process_structured_polish_response,
)

# 主入口函数
from .polish_main import (
    POLISH_STEP_NAME,
    perform_structured_polish,
    polish_node,
    polish_node_fallback,
)

# 质量检查器
from .quality_checker import (
    _validate_final_solution,
    calculate_quality_score,
    generate_modification_summary,
)

# 工具函数
from .utils import (
    _detect_unresolved_placeholders,
    _remove_unresolved_placeholders,
    parse_document_structure,
)

__all__ = [
    # 主入口
    "polish_node",
    "polish_node_fallback",
    "perform_structured_polish",
    "POLISH_STEP_NAME",
    # 内容处理
    "polish_section_structured",
    "build_section_polish_prompt",
    "process_structured_polish_response",
    "polish_section_text_fallback",
    # 质量检查
    "_validate_final_solution",
    "calculate_quality_score",
    "generate_modification_summary",
    # 内容组装
    "assemble_final_content",
    "extract_document_title",
    # 引用处理
    "render_citations_with_footnotes",
    "initialize_citation_manager",
    "integrate_fact_checking",
    # 工具函数
    "_detect_unresolved_placeholders",
    "_remove_unresolved_placeholders",
    "parse_document_structure",
]
