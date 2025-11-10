"""
Polish模块的工具函数

包含占位符检测和清理等通用工具函数
"""

from __future__ import annotations

import re

PLACEHOLDER_PATTERNS = [
    r"\[ref:[^\]]+\]",
    r"待补[充]?",
    r"参考文献\s*\[[^\]]*待补[^\]]*\]",
    r"references?\s*\[\d+\]\s*(todo|待补)",
    r"\btodo\b",
]

INTRO_TOKENS = ("引言", "绪论", "简介", "背景", "overview", "introduction")
CONCLUSION_TOKENS = ("结论", "总结", "展望", "结语", "conclusion", "outlook", "closing")


def _detect_unresolved_placeholders(content: str) -> set[str]:
    """
    检测未解决的占位符

    Args:
        content: 文档内容

    Returns:
        未解决的占位符集合
    """
    hits: set[str] = set()
    lowered = content.lower()
    for pattern in PLACEHOLDER_PATTERNS:
        for match in re.finditer(pattern, lowered, flags=re.IGNORECASE):
            hits.add(match.group())
    return hits


def _remove_unresolved_placeholders(content: str, placeholders: set[str]) -> str:
    """
    移除未解决的占位符（🔧 完整修复：强化清理）

    Args:
        content: 原始内容
        placeholders: 未解决的占位符集合

    Returns:
        清理后的内容
    """
    cleaned = content

    # 🔧 完整修复：添加更多占位符清理规则
    # 1. 清理检测到的特定占位符
    for placeholder in placeholders:
        pattern = re.escape(placeholder)
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)

    # 2. 🆕 强力清理：移除所有 [ref:...] 格式的占位符
    cleaned = re.sub(r"\[ref:\s*[^\]]+\]", "", cleaned)

    # 3. 🆕 清理常见的占位符模式
    cleaned = re.sub(r"\[citation needed\]", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\[待补充?\]", "", cleaned)
    cleaned = re.sub(r"\[TODO:?[^\]]*\]", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"TODO:?\s*[^\n]*", "", cleaned, flags=re.IGNORECASE)

    # 4. 🆕 清理孤立的 RAG 引用
    cleaned = re.sub(r"#rag\d+", "", cleaned)

    # 5. 清理多余的空格和换行
    cleaned = re.sub(r"\n\s*\n\s*\n+", "\n\n", cleaned)  # 多个空行合并为两个
    cleaned = re.sub(r" +", " ", cleaned)  # 多个空格合并为一个
    cleaned = re.sub(r"^\s+", "", cleaned, flags=re.MULTILINE)  # 清理行首空格

    return cleaned.strip()


def parse_document_structure(content: str) -> list[dict]:
    """
    解析文档的章节结构

    Args:
        content: 文档内容

    Returns:
        章节列表，每个章节包含title、content、section_id和original_content
    """
    import re

    from utils.id_manager import ensure_section_id

    sections = []
    lines = content.split("\n")
    current_section = None
    current_content = []

    for line in lines:
        # 检测markdown标题
        if line.strip().startswith("#"):
            # 保存前一个章节
            if current_section:
                section_content = "\n".join(current_content).strip()
                current_section["content"] = section_content
                current_section["original_content"] = section_content
                sections.append(current_section)

            # 开始新章节
            # 尝试从标题中提取 section_id（如果有注释）
            title_line = line.strip()
            match = re.search(r"<!--\s*section_id:\s*([a-zA-Z0-9\-_]+)\s*-->", title_line)

            # 🔧 修复：创建字典后使用 ensure_section_id 确保有 ID
            current_section = {
                "title": title_line,
                "content": "",
            }

            if match:
                # 如果标题中有 ID 注释，使用它
                current_section["section_id"] = match.group(1)
            else:
                # 否则生成新的 ID 并保存
                current_section["section_id"] = ensure_section_id(current_section, key="section_id")

            current_content = []
        elif current_section:
            current_content.append(line)

    # 保存最后一个章节
    if current_section:
        section_content = "\n".join(current_content).strip()
        current_section["content"] = section_content
        current_section["original_content"] = section_content
        sections.append(current_section)

    return sections
