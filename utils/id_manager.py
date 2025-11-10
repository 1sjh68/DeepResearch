"""
章节 ID 管理工具

提供统一的 UUID 生成和管理接口，确保 ID 的一致性。
"""

import uuid
from typing import Any


def ensure_section_id(section: dict[str, Any], key: str = "id") -> str:
    """
    确保章节有唯一的 ID，如果没有则生成并保存。

    Args:
        section: 章节字典对象
        key: ID 键名，默认为 "id"

    Returns:
        str: 章节的 UUID

    Example:
        >>> section = {"title": "Introduction"}
        >>> section_id = ensure_section_id(section)
        >>> assert "id" in section
        >>> assert section["id"] == section_id
    """
    existing = section.get(key)
    if existing:
        return existing

    # 生成新的 UUID 并保存回字典
    new_id = str(uuid.uuid4())
    section[key] = new_id
    return new_id


def ensure_ids_recursive(sections: list[dict[str, Any]], key: str = "id") -> None:
    """
    递归地为章节列表中的所有章节确保有 ID。

    Args:
        sections: 章节列表
        key: ID 键名，默认为 "id"

    Example:
        >>> sections = [
        ...     {"title": "Chapter 1", "sections": [{"title": "Section 1.1"}]},
        ...     {"title": "Chapter 2"}
        ... ]
        >>> ensure_ids_recursive(sections)
        >>> assert all("id" in s for s in sections)
    """
    for section in sections:
        if not isinstance(section, dict):
            continue

        # 确保当前章节有 ID
        ensure_section_id(section, key)

        # 递归处理子章节
        child_sections = section.get("sections") or section.get("children")
        if child_sections and isinstance(child_sections, list):
            ensure_ids_recursive(child_sections, key)
