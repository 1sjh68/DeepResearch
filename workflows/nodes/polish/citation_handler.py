"""
Polish模块的引用处理功能

包含引用标注生成、脚注格式化、事实核查集成等功能
"""

from __future__ import annotations

import logging
import re
from typing import Any

from planning.tool_definitions import PolishModel, PolishSection
from utils.citation import CitationManager, CitationMatch, SourceInfo
from utils.factcheck import FactChecker, FactCheckResult, add_verification_markers


def _ensure_reference_section(content: str) -> str:
    """确保最终文档包含参考文献占位符，避免空白引用段。"""
    if re.search(r"^##\s*参考文献", content, flags=re.MULTILINE):
        return content
    placeholder = "\n\n## 参考文献\n\n（本次迭代未生成参考文献，请补充可信来源。）"
    return content + placeholder


def generate_citation_marker(number: int, style: str) -> str:
    """根据风格生成引用标记"""
    style_formatters = {
        "numeric": f"[{number}]",
        "symbol": f"[†{number}]",
        "bracket": f"[({number})]",
        "parenthetical": f"({number})",
    }
    return style_formatters.get(style, f"[{number}]")


def format_footnote_content(match: CitationMatch, number: int, style: str) -> str:
    """格式化脚注内容"""
    if not match.sources:
        return f"[{number}] 需要进一步验证的内容"

    source_texts = []
    for source in match.sources[:2]:  # 最多显示2个来源
        source_info = f"{source.title}"
        if source.url:
            source_info += f" - {source.url}"
        if source.date:
            source_info += f" ({source.date})"
        source_texts.append(source_info)

    # 添加验证状态
    verification_status = "✓ 已验证" if not match.requires_verification else "⚠ 需要验证"

    return f"[{number}] {'; '.join(source_texts)} - {verification_status}"


def find_sentence_end(text: str, start_pos: int) -> int:
    """查找句子的结束位置"""
    end_chars = [".", "。", "!", "！", "?", "？"]
    for i in range(start_pos, min(len(text), start_pos + 100)):
        if text[i] in end_chars:
            return i + 1
    return start_pos


def insert_citation_markers(content: str, citation_map: dict, style: str) -> str:
    """在文本中插入引用标记"""
    # 按匹配项的文本位置排序
    sorted_matches = sorted(
        citation_map.items(),
        key=lambda x: content.find(x[1].claim.text) if x[1].claim.text in content else -1,
    )

    # 从后往前插入，避免位置偏移
    for citation_marker, match in reversed(sorted_matches):
        if match.claim.text in content:
            # 在句末插入引用标记
            end_pos = content.find(match.claim.text) + len(match.claim.text)
            if end_pos <= len(content):
                # 查找句子结束位置
                sentence_end = find_sentence_end(content, end_pos)
                content = content[:sentence_end] + f" {citation_marker}" + content[sentence_end:]

    return content


def generate_footnotes_section(footnote_list: list[str], style: str) -> str:
    """生成脚注列表部分"""
    if not footnote_list:
        return ""

    section_title = {
        "numeric": "\n\n## 参考文献",
        "symbol": "\n\n## 符号脚注",
        "bracket": "\n\n## 参考资料",
        "parenthetical": "\n\n## 参考资料",
    }

    title = section_title.get(style, "\n\n## 参考资料")
    footnotes = "\n\n".join(f"{i + 1}. {footnote}" for i, footnote in enumerate(footnote_list))

    # 添加统计信息
    verified_count = sum(1 for footnote in footnote_list if "✓" in footnote)
    total_count = len(footnote_list)
    coverage = (verified_count / total_count * 100.0) if total_count > 0 else 0.0
    stats = f"\n\n*注：已验证来源 {verified_count}/{total_count}，引用覆盖率 {coverage:.1f}%*"

    return f"{title}\n\n{footnotes}{stats}"


def render_citations_with_footnotes(content: str, citations: list[CitationMatch], style: str = "numeric") -> tuple[str, str]:
    """
    根据引用风格生成带标注的文本和脚注列表

    Args:
        content: 原始内容
        citations: 引用匹配列表
        style: 引用风格

    Returns:
        带标注的文本和脚注列表
    """
    if not citations:
        return content, ""

    # 构建引用映射
    citation_map = {}
    footnote_list = []
    citation_counter = 1

    for match in citations:
        if match.sources:
            # 根据风格生成引用标记
            citation_marker = generate_citation_marker(citation_counter, style)
            citation_map[citation_marker] = match

            # 生成脚注内容
            footnote_content = format_footnote_content(match, citation_counter, style)
            if footnote_content not in footnote_list:
                footnote_list.append(footnote_content)
                citation_counter += 1

    # 在文本中插入引用标记
    annotated_text = insert_citation_markers(content, citation_map, style)

    # 生成脚注列表
    footnotes_section = generate_footnotes_section(footnote_list, style)

    return annotated_text, footnotes_section


def initialize_citation_manager(
    research_data: dict[str, Any] | None = None,
) -> CitationManager:
    """
    初始化引用管理器

    Args:
        research_data: 研究数据

    Returns:
        配置好的引用管理器
    """
    citation_manager = CitationManager()

    if research_data:
        # 从研究数据中提取源信息
        for item in research_data.get("sources", []):
            if isinstance(item, dict):
                source = SourceInfo(
                    id=item.get("id", ""),
                    url=item.get("url", ""),
                    title=item.get("title", ""),
                    date=item.get("date", ""),
                    content=item.get("content", ""),
                    summary=item.get("summary", ""),
                    confidence=item.get("confidence", 0.5),
                )
                citation_manager.add_source(source)

    return citation_manager


def integrate_fact_checking(
    polished_result: PolishModel,
    structured_research_data: dict[str, Any] | None = None,
    config: Any | None = None,
) -> list[FactCheckResult]:
    """
    集成事实核查功能到润色流程中

    Args:
        polished_result: 润色结果
        structured_research_data: 结构化研究数据
        config: 配置对象

    Returns:
        List[FactCheckResult]: 事实核查结果列表
    """
    logging.info("开始集成事实核查功能...")

    # 初始化事实核查器
    fact_checker = FactChecker()

    # 收集所有事实核查点
    all_fact_check_points = []
    for section in polished_result.sections:
        # 兼容旧的 PolishSection（无 fact_check_points 字段）
        _points = getattr(section, "fact_check_points", None)
        if _points:
            all_fact_check_points.extend(_points)

    if not all_fact_check_points:
        logging.info("未发现需要核查的事实点")
        return []

    logging.info(f"发现 {len(all_fact_check_points)} 个需要核查的事实点")

    # 准备数据源
    sources = prepare_fact_check_sources(structured_research_data)

    try:
        # 执行事实核查
        fact_check_results = fact_checker.check_minimal(all_fact_check_points, sources)

        # 生成核查报告
        if fact_check_results:
            verification_report = fact_checker.generate_unverifiable_report(fact_check_results)
            logging.info(f"事实核查完成 - 验证率: {verification_report['verification_rate']:.2%}")

            # 为需要修改的内容生成建议
            modification_suggestions = generate_fact_check_modification_suggestions(fact_check_results, polished_result.sections)

            # 如果有修改建议，添加到元数据中
            if modification_suggestions:
                polished_result.metadata["fact_check_modifications"] = modification_suggestions
                logging.info(f"生成了 {len(modification_suggestions)} 条事实核查修改建议")

        return fact_check_results

    except Exception as e:
        logging.error(f"事实核查过程发生错误: {e}", exc_info=True)
        return []


def prepare_fact_check_sources(
    structured_research_data: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """
    准备事实核查的数据源

    Args:
        structured_research_data: 结构化研究数据

    Returns:
        List[Dict]: 格式化后的数据源列表
    """
    sources: list[dict[str, Any]] = []

    if not structured_research_data:
        return sources

    # 提取研究数据中的各个部分作为数据源
    for key, value in structured_research_data.items():
        if isinstance(value, dict):
            # 学术论文格式
            if "title" in value and "abstract" in value:
                sources.append(
                    {
                        "title": value.get("title", ""),
                        "content": value.get("abstract", "") + " " + value.get("content", ""),
                        "url": value.get("url", ""),
                        "type": "academic_paper",
                    }
                )
            # 报告格式
            elif "summary" in value:
                sources.append(
                    {
                        "title": value.get("title", key),
                        "content": value.get("summary", ""),
                        "url": value.get("url", ""),
                        "type": "report",
                    }
                )
        elif isinstance(value, list):
            # 列表格式的数据
            for item in value:
                if isinstance(item, dict) and "content" in item:
                    sources.append(
                        {
                            "title": item.get("title", f"{key}_item"),
                            "content": item.get("content", ""),
                            "url": item.get("url", ""),
                            "type": "reference",
                        }
                    )
        elif isinstance(value, str):
            # 简单文本内容
            sources.append({"title": key, "content": value, "url": "", "type": "text"})

    logging.info(f"准备了 {len(sources)} 个数据源用于事实核查")
    return sources


def generate_fact_check_modification_suggestions(fact_check_results: list[FactCheckResult], sections: list[PolishSection]) -> list[dict]:
    """
    基于事实核查结果生成修改建议

    Args:
        fact_check_results: 事实核查结果
        sections: 润色章节列表

    Returns:
        List[Dict]: 修改建议列表
    """
    suggestions = []

    for result in fact_check_results:
        if result.confidence_level in ["low", "unverifiable"]:
            # 查找包含该主张的章节
            target_section = None
            for section in sections:
                if result.claim in section.content:
                    target_section = section
                    break

            if target_section:
                suggestion = {
                    "section_id": target_section.section_id,
                    "section_title": target_section.title,
                    "claim": result.claim,
                    "issue_type": "verification_concern",
                    "confidence_level": result.confidence_level,
                    "verification_score": result.verification_score,
                    "recommendation": generate_single_claim_suggestion(result),
                    "supporting_sources": result.supporting_sources,
                    "contradicting_sources": result.contradicting_sources,
                    "notes": result.notes,
                }
                suggestions.append(suggestion)

    return suggestions


def generate_single_claim_suggestion(result: FactCheckResult) -> str:
    """
    为单个主张生成修改建议

    Args:
        result: 事实核查结果

    Returns:
        str: 修改建议文本
    """
    if result.confidence_level == "unverifiable":
        return f"建议删除或修改该主张：'{result.claim[:50]}...'，因为无法验证其准确性。"
    elif result.confidence_level == "low":
        return f"建议为该主张补充权威证据：'{result.claim[:50]}...'，当前验证分数为 {result.verification_score:.2f}。"
    else:
        return f"该主张基本可信：'{result.claim[:50]}...'，验证分数为 {result.verification_score:.2f}。"


def enhance_content_with_verification_markers(content: str, fact_check_results: list[FactCheckResult]) -> str:
    """
    在内容中添加验证标记

    Args:
        content: 原始内容
        fact_check_results: 事实核查结果

    Returns:
        str: 带有验证标记的内容
    """
    if not fact_check_results:
        return content

    # 使用utils.factcheck模块中的函数添加标记
    return add_verification_markers(content, fact_check_results)

