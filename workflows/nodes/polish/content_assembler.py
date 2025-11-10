"""
Polish模块的内容组装功能

包含最终文档组装、章节去重、标题提取等功能
"""

from __future__ import annotations

import logging
import re

from planning.tool_definitions import PolishSection
from utils.citation import CitationManager

from .utils import (
    CONCLUSION_TOKENS,
    INTRO_TOKENS,
    _detect_unresolved_placeholders,
    _remove_unresolved_placeholders,
)


def extract_document_title(content: str) -> str:
    """
    提取文档标题
    """
    lines = content.split("\n")
    for line in lines:
        line = line.strip()
        if line.startswith("#"):
            cleaned = re.sub(r"^#+\s*", "", line)
            if cleaned:
                return cleaned.strip()
    return "未命名文档"


def _drop_duplicate_intro_and_conclusion(markdown: str) -> str:
    """
    移除重复的引言/结论章节，保留信息量更高的一份。
    """
    heading_re = re.compile(
        r"^(#+)\s*([^\n]+?)(?:\s*<!--.*?-->)?\s*$",
        re.MULTILINE,
    )

    headings: list[tuple[int, str, str]] = []  # (position, level, title_lower)
    for match in heading_re.finditer(markdown):
        level = len(match.group(1))
        title = match.group(2).strip()
        headings.append((match.start(), level, title.lower()))

    if not headings:
        return markdown

    intro_idxs = []
    conclusion_idxs = []

    for idx, (pos, level, title_lower) in enumerate(headings):
        if level == 2:  # Only check level-2 sections
            if any(token in title_lower for token in INTRO_TOKENS):
                intro_idxs.append(idx)
            elif any(token in title_lower for token in CONCLUSION_TOKENS):
                conclusion_idxs.append(idx)

    sections_to_remove: list[int] = []

    # 处理重复引言
    if len(intro_idxs) > 1:
        content_lengths = []
        for idx in intro_idxs:
            start = headings[idx][0]
            end = headings[idx + 1][0] if idx + 1 < len(headings) else len(markdown)
            section_text = markdown[start:end]
            content_lengths.append((idx, len(section_text)))
        content_lengths.sort(key=lambda x: x[1], reverse=True)
        # Keep the longest, remove others
        keep_idx = content_lengths[0][0]
        for idx in intro_idxs:
            if idx != keep_idx:
                sections_to_remove.append(idx)
                _, _, title_lower = headings[idx]
                logging.info("去除重复章节《%s》", title_lower)

    # 处理重复结论
    if len(conclusion_idxs) > 1:
        content_lengths = []
        for idx in conclusion_idxs:
            start = headings[idx][0]
            end = headings[idx + 1][0] if idx + 1 < len(headings) else len(markdown)
            section_text = markdown[start:end]
            content_lengths.append((idx, len(section_text)))
        content_lengths.sort(key=lambda x: x[1], reverse=True)
        keep_idx = content_lengths[0][0]
        for idx in conclusion_idxs:
            if idx != keep_idx:
                sections_to_remove.append(idx)
                _, _, title_lower = headings[idx]
                logging.info("去除重复章节《%s》", title_lower)

    if not sections_to_remove:
        return markdown

    # 从后往前删除章节
    sections_to_remove.sort(reverse=True)
    result = markdown
    for idx in sections_to_remove:
        start = headings[idx][0]
        end = headings[idx + 1][0] if idx + 1 < len(headings) else len(markdown)
        result = result[:start] + result[end:]

    return result.strip()


def assemble_final_content(
    sections: list[PolishSection],
    citation_manager: CitationManager | None = None,
    citation_style: str = "numeric",
    document_title: str | None = None,
) -> str:
    """
    重新组装最终文档内容，集成引用标注和脚注生成

    Args:
        sections: 章节列表
        citation_manager: 引用管理器实例
        citation_style: 引用格式 ("numeric", "symbol", "bracket", "parenthetical")
        document_title: 文档标题（可选）

    Returns:
        包含引用标注的完整文档
    """
    content_parts: list[str] = []

    # Optional document title (level-1 heading)
    if document_title:
        content_parts.append(f"# {document_title}\n\n")

    order: list[str] = []
    section_map: dict[str, tuple[PolishSection, str]] = {}

    for index, section in enumerate(sections):
        raw_title = section.title or ""
        # Remove existing HTML comments (e.g., legacy section_id markers)
        title_clean = re.sub(r"\s*<!--.*?-->\s*", "", raw_title).strip()
        title_clean = re.sub(r"^#+\s*", "", title_clean).strip()
        content_clean = (section.content or "").strip()

        if not title_clean or not content_clean:
            continue

        key = title_clean.lower()
        existing = section_map.get(key)
        if existing:
            existing_section, existing_title = existing
            if len(content_clean) > len(existing_section.content or ""):
                section_map[key] = (section, title_clean)
        else:
            section_map[key] = (section, title_clean)
            order.append(key)

    if not order:
        return ""

    def _priority(title: str, default_index: int) -> tuple[int, int]:
        lowered = title.lower()
        if any(token in lowered for token in ("引言", "背景", "简介")):
            return (0, default_index)
        if any(token in lowered for token in ("理论", "模型", "方法", "框架")):
            return (10, default_index)
        if any(token in lowered for token in ("实验", "实施", "方案", "方法学")):
            return (20, default_index)
        if any(token in lowered for token in ("结果", "分析", "讨论", "发现")):
            return (30, default_index)
        if any(token in lowered for token in ("结论", "展望", "总结", "未来")):
            return (40, default_index)
        if any(token in lowered for token in ("符号", "参数", "附录")):
            return (50, default_index)
        if any(token in lowered for token in ("参考", "文献", "致谢")):
            return (60, default_index)
        return (99, default_index)

    ordered_keys = sorted(
        order,
        key=lambda key: _priority(section_map[key][1], order.index(key)),
    )

    seen_paragraphs: set[str] = set()

    for key in ordered_keys:
        section, cleaned_title = section_map[key]
        # 添加标题（若存在 section_id，注入注释以便后续结构管理）
        heading = f"## {cleaned_title}"
        if getattr(section, "section_id", None):
            heading = f"{heading} <!-- section_id: {section.section_id} -->"
        content_parts.append(heading)

        # 添加内容
        if section.content:
            paragraphs = re.split(r"\n\s*\n", section.content.strip())
            filtered: list[str] = []
            for para in paragraphs:
                normalized = re.sub(r"\s+", " ", para.strip()).lower()
                if len(normalized) >= 60:
                    if normalized in seen_paragraphs:
                        logging.info(
                            "检测到重复段落，已跳过: %s...",
                            para.strip()[:40],
                        )
                        continue
                    seen_paragraphs.add(normalized)
                filtered.append(para.strip())

            if filtered:
                content_parts.append("\n" + "\n\n".join(filtered))

        # 添加分段符（除了最后一个章节）
        content_parts.append("\n")

    base_content = "".join(content_parts).strip()

    base_content = _drop_duplicate_intro_and_conclusion(base_content)

    placeholder_hits = _detect_unresolved_placeholders(base_content)
    if placeholder_hits:
        logging.warning(
            "最终文档存在未解决占位符，命中: %s。将尝试清理这些占位符。",
            ", ".join(sorted(placeholder_hits)),
        )
        # 修复：清理未解决的占位符，而不是崩溃
        base_content = _remove_unresolved_placeholders(base_content, placeholder_hits)
        removed_count = len(placeholder_hits)
        logging.info(f"已清理 {removed_count} 个未解决的占位符")

    # 如果没有引用管理器，返回基础内容
    if not citation_manager or not citation_manager.citations:
        from .citation_handler import _ensure_reference_section
        return _ensure_reference_section(base_content)

    # 生成引用标注和脚注
    try:
        from .citation_handler import render_citations_with_footnotes
        annotated_content, references_section = render_citations_with_footnotes(
            base_content, citation_manager.citations, citation_style
        )

        # 合并内容
        final_content = f"{annotated_content}\n\n{references_section}"
        return final_content

    except Exception as e:
        logging.warning(f"引用标注生成失败: {e}，返回基础内容")
        from .citation_handler import _ensure_reference_section
        return _ensure_reference_section(base_content)

