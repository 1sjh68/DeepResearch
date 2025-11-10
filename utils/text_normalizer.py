"""
文本规范化和清理模块

提供统一的文本清理、LaTeX修复和规范化功能。
"""

import logging
import re

logger = logging.getLogger(__name__)

# LaTeX修复相关常量
CONTROL_CHAR_PATTERN = re.compile(r"[\x00-\x08\x0b-\x0c\x0e-\x1f]")
LATEX_REPLACEMENTS: dict[str, str] = {
    "\x08oldsymbol": r"\\boldsymbol",
    "\x08oxed": r"\\boxed",
    "\x08eta": r"\\eta",
    "\x08theta": r"\\theta",
    "\x08tau": r"\\tau",
    "\x08odot": r"\\odot",
    "\x08big": r"\\big",
    "\x08left": r"\\left",
    "\x08right": r"\\right",
    "\x08circ": r"\\circ",
    "\x08propto": r"\\propto",
    "\x07lpha": r"\\alpha",
    "\x07pprox": r"\\approx",
    "\x07mega": r"\\Omega",
    "\x07theta": r"\\theta",
    "\x07star": r"\\star",
    "\x0crac": r"\\frac",
    "\tfrac": r"\\frac",
}


def clean_text_artifacts(text: str) -> str:
    """
    移除控制字符并修复常见的 LaTeX 破坏问题。

    Args:
        text: 原始文本

    Returns:
        清理后的文本
    """
    if not text:
        return text

    cleaned = text
    for src, dst in LATEX_REPLACEMENTS.items():
        if src in cleaned:
            cleaned = cleaned.replace(src, dst)

    cleaned = CONTROL_CHAR_PATTERN.sub("", cleaned)

    if "!\\left" in cleaned or "!\\right" in cleaned or "!\\big" in cleaned:
        cleaned = (
            cleaned.replace("!\\left", "\\left")
            .replace("!\\right", "\\right")
            .replace("!\\big", "\\big")
        )

    return cleaned


def normalize_whitespace(text: str) -> str:
    """
    规范化文本中的空白字符

    Args:
        text: 原始文本

    Returns:
        规范化后的文本
    """
    if not text:
        return text

    # 替换多个空格为单个空格
    text = re.sub(r' +', ' ', text)
    # 替换多个换行为最多两个换行
    text = re.sub(r'\n{3,}', '\n\n', text)
    # 移除行尾空格
    text = re.sub(r' +$', '', text, flags=re.MULTILINE)

    return text


def remove_control_characters(text: str, allowed: set[str] | None = None) -> str:
    """
    移除文本中不允许的控制字符

    Args:
        text: 原始文本
        allowed: 允许保留的控制字符集合，默认为 {'\t', '\n', '\r'}

    Returns:
        清理后的文本
    """
    if not text:
        return text

    if allowed is None:
        allowed = {'\t', '\n', '\r'}

    cleaned_chars = []
    removed_count = 0

    for ch in text:
        # 检查是否是不允许的控制字符
        if (ord(ch) < 0x20 or ord(ch) == 0x7f) and ch not in allowed:
            removed_count += 1
            continue
        cleaned_chars.append(ch)

    if removed_count > 0:
        logger.debug(
            "remove_control_characters: 移除了 %d 个不允许的控制字符",
            removed_count,
        )

    return ''.join(cleaned_chars)


def strip_markdown_fence(text: str) -> str:
    """
    移除Markdown代码块标记

    Args:
        text: 可能包含markdown代码块的文本

    Returns:
        移除代码块标记后的文本
    """
    if not text:
        return text

    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?", "", stripped, flags=re.IGNORECASE).strip()
        stripped = re.sub(r"```$", "", stripped).strip()
    return stripped


def normalize_unicode_quotes(text: str) -> str:
    """
    规范化Unicode引号为标准ASCII引号

    Args:
        text: 包含Unicode引号的文本

    Returns:
        规范化后的文本
    """
    if not text:
        return text

    replacements = {
        """: '"',  # U+201C LEFT DOUBLE QUOTATION MARK
        """: '"',  # U+201D RIGHT DOUBLE QUOTATION MARK
        "„": '"',  # U+201E DOUBLE LOW-9 QUOTATION MARK
        "‟": '"',  # U+201F DOUBLE HIGH-REVERSED-9 QUOTATION MARK
        "'": "'",  # U+2019 RIGHT SINGLE QUOTATION MARK
        "‚": "'",  # U+201A SINGLE LOW-9 QUOTATION MARK
    }

    for src, dst in replacements.items():
        if src in text:
            text = text.replace(src, dst)

    return text


class TextNormalizer:
    """文本规范化器的面向对象封装"""

    @staticmethod
    def clean(text: str) -> str:
        """清理文本中的artifacts"""
        return clean_text_artifacts(text)

    @staticmethod
    def normalize_whitespace(text: str) -> str:
        """规范化空白字符"""
        return normalize_whitespace(text)

    @staticmethod
    def remove_control_chars(text: str, allowed: set[str] | None = None) -> str:
        """移除控制字符"""
        return remove_control_characters(text, allowed)

    @staticmethod
    def strip_markdown(text: str) -> str:
        """移除markdown代码块标记"""
        return strip_markdown_fence(text)

    @staticmethod
    def normalize_quotes(text: str) -> str:
        """规范化Unicode引号"""
        return normalize_unicode_quotes(text)

    def normalize_all(self, text: str) -> str:
        """
        应用所有规范化步骤

        Args:
            text: 原始文本

        Returns:
            完全规范化后的文本
        """
        if not text:
            return text

        # 按顺序应用所有规范化
        text = self.clean(text)
        text = self.normalize_quotes(text)
        text = self.remove_control_chars(text)
        text = self.normalize_whitespace(text)

        return text

