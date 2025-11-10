"""
Polish模块的质量检查功能

包含质量评分、文本异常检测等质量控制功能
"""

from __future__ import annotations

import logging
import re

from planning.tool_definitions import PolishSection


def _validate_final_solution(content: str, config) -> tuple[bool, str]:
    """
    验证最终输出内容的质量。

    参数：
        content: 待验证的内容
        config: 配置对象

    返回：
        (是否有效, 错误消息或"OK")
    """
    min_length = getattr(config, "min_final_content_length", 100)
    max_length = getattr(config, "max_final_content_length", 1000000)

    # 检查1: 不为空
    if not content or not content.strip():
        return False, "内容为空"

    # 检查2: 长度范围
    if len(content) < min_length:
        return False, f"内容过短({len(content)} < {min_length})"

    if len(content) > max_length:
        logging.warning("内容超过推荐长度: %d > %d",
                       len(content), max_length)

    # 检查3: 有效的Markdown结构
    if not re.search(r'^#+\s+', content, re.MULTILINE):
        return False, "未找到Markdown标题"

    # 检查4: 足够的内容行
    non_empty_lines = len([line for line in content.split('\n') if line.strip()])
    if non_empty_lines < 3:
        return False, f"有效内容行数过少: {non_empty_lines} < 3"

    return True, "OK"


def _detect_text_anomalies(text: str) -> list[str]:
    """
    粗略检测文本是否存在截断、未闭合括号等明显完整性问题。
    """
    if not text:
        return []

    anomalies: list[str] = []
    normalized = text.replace("\r\n", "\n")

    delimiter_pairs = [
        ("(", ")"),
        ("（", "）"),
        ("[", "]"),
        ("【", "】"),
    ]
    for left, right in delimiter_pairs:
        if normalized.count(left) != normalized.count(right):
            anomalies.append(f"未闭合的括号 {left}{right}")

    truncated_number_pattern = re.compile(r"[（(][^）()\n]*\$?\d+(?:\.[^\d\)\]]*)?$")
    paragraphs = re.split(r"\n\s*\n", normalized)
    for para in paragraphs:
        stripped = para.strip()
        if not stripped:
            continue
        if len(stripped) > 80:
            tail = stripped[-1]
            end_punctuation = '。！？?!.;:）)】]"\''
            if tail not in end_punctuation:
                if not (stripped.endswith("$$") or stripped.endswith("$") or stripped.endswith("\\]")):
                    anomalies.append("长句缺乏结尾标点，疑似被截断")
                    break
        if truncated_number_pattern.search(stripped):
            anomalies.append("括号中的数字或公式疑似被截断")
            break

    # 去重
    seen = set()
    unique_anomalies = []
    for item in anomalies:
        if item not in seen:
            seen.add(item)
            unique_anomalies.append(item)
    return unique_anomalies


def _should_revert_due_to_anomalies(anomalies: list[str], original: str, candidate: str) -> bool:
    """
    决定是否因为检测到的异常而回退到原文。

    仅当异常明显且替换文本显著短于原始内容时才回退，避免轻微警告吞掉有效修改。
    """
    if not anomalies:
        return False

    original_clean = (original or "").strip()
    candidate_clean = (candidate or "").strip()
    if not candidate_clean:
        return True

    original_length = len(original_clean)
    candidate_length = len(candidate_clean)
    # original_length 已在前面确保非零，无需使用 max()
    if original_length and candidate_length / original_length < 0.45:
        return True

    # 多条异常基本可判断为不可恢复的输出
    return len(anomalies) >= 2


def calculate_quality_score(sections: list[PolishSection]) -> float:
    """
    计算整体质量评分
    """
    if not sections:
        return 0.5

    total_score = 0.0
    total_sections = len(sections)

    for section in sections:
        # 基于修改数量和字数计算质量评分
        word_count = max(section.word_count, 1)
        modification_ratio = len(section.modifications) / word_count

        # 合理的修改比例
        if modification_ratio < 0.1:  # 修改过少
            base_score = 0.7
        elif modification_ratio > 0.3:  # 修改过多
            base_score = 0.6
        else:  # 合理修改
            base_score = 0.85

        total_score += base_score

    return min(total_score / total_sections, 1.0)


def generate_modification_summary(sections: list[PolishSection]) -> str:
    """
    生成修改总结
    """
    total_modifications = sum(len(section.modifications) for section in sections)
    total_words = sum(section.word_count for section in sections)

    if total_modifications == 0:
        return "本次润色无需任何修改，文档质量良好。"

    avg_words_per_modification = total_words / total_modifications if total_modifications > 0 else 0

    summary = (
        f"本次润色共对 {len(sections)} 个章节进行处理，"
        f"总计 {total_modifications} 处修改，"
        f"平均每 {avg_words_per_modification:.1f} 字修改一处。"
    )
    return summary

