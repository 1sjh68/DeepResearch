# ruff: noqa: E501
from __future__ import annotations

import codecs
import logging
import os
import re
import unicodedata
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass
from difflib import SequenceMatcher
from re import Match
from typing import Any, cast

SECTION_BLOCK_PATTERN = re.compile(r"(^#+.*?)(?=^#+ |\Z)", re.MULTILINE | re.DOTALL)


# 🔧 Phase 1修复：定义匹配常量，替换魔法数字
class MatchingThresholds:
    """句子匹配的阈值常量"""

    # 相似度阈值
    SIMILARITY_STRICT = 0.60  # 严格相似度匹配（默认）
    SIMILARITY_MODERATE = 0.50  # 中等相似度匹配
    SIMILARITY_RELAXED = 0.40  # 宽松相似度匹配
    SIMILARITY_FUZZY = 0.35  # 模糊匹配最低阈值

    # 覆盖率阈值
    COVERAGE_MINIMUM = 0.6  # 核心词汇最小覆盖率（60%）

    # 综合评分阈值
    COMBINED_SCORE_MIN = 0.45  # 综合分数最低要求

    # 模糊匹配参数
    MIN_FUZZY_LENGTH = 15  # 模糊匹配最小字符串长度


# 注意：补丁载荷来自外部JSON，因此值保持为`Any`类型
# 在此适配层中，在下游验证之前保持灵活性
JsonDict = dict[str, Any]
CorrectionResult = tuple[str, str, float, str, str, int]
Strategy = Callable[[str, str, int], CorrectionResult | None]


def _normalize_mapping(mapping: Any) -> JsonDict:
    if not isinstance(mapping, Mapping):
        return {}
    typed_mapping = cast(Mapping[Any, Any], mapping)
    normalized: JsonDict = {}
    for key, value in typed_mapping.items():
        normalized[str(key)] = value
    return normalized


def _find_section_by_id(document: str, section_id: str) -> Match[str] | None:
    """
    定位由注入的section_id注释标识的Markdown章节块。
    优先匹配包含<!-- section_id: ... -->标记的标题。
    """
    if not document or not section_id:
        return None

    escaped_id = re.escape(section_id)
    direct_pattern = re.compile(
        rf"(^#+.*?<!--\s*section_id:\s*{escaped_id}\s*-->.*?)(?=^#+ |\Z)",
        re.MULTILINE | re.DOTALL,
    )
    match = direct_pattern.search(document)
    if match:
        return match

    comment_pattern = re.compile(rf"<!--\s*section_id:\s*{escaped_id}\s*-->")
    comment_match = comment_pattern.search(document)
    if not comment_match:
        return None

    heading_pattern = re.compile(r"^#+.*", re.MULTILINE)
    heading_match: Match[str] | None = None
    for candidate in heading_pattern.finditer(document, 0, comment_match.start()):
        heading_match = candidate
    if not heading_match:
        return None

    return SECTION_BLOCK_PATTERN.search(document, heading_match.start())


def _find_section_by_title(document: str, section_title: str) -> Match[str] | None:
    """
    通过章节标题模糊匹配查找章节

    Args:
        document: 文档内容
        section_title: 章节标题

    Returns:
        匹配的章节，如果未找到则返回None
    """
    if not document or not section_title:
        return None

    # 移除特殊字符进行模糊匹配
    clean_title = re.sub(r"[^\w\s]", "", section_title)

    # 尝试匹配包含该标题的markdown标题行
    title_pattern = rf"(^##+\s+.*?{re.escape(clean_title)}.*?)(?=^#+ |\Z)"
    match = re.search(title_pattern, document, re.MULTILINE | re.DOTALL | re.IGNORECASE)

    if match:
        return match

    return None


def _list_available_section_ids(document: str) -> list[str]:
    """
    列出文档中所有可用的section_id

    Args:
        document: 文档内容

    Returns:
        section_id列表
    """
    pattern = r"<!--\s*section_id:\s*([^\s]+)\s*-->"
    matches = re.findall(pattern, document)
    return matches


def _find_section_by_multiple_methods(
    document: str,
    section_id: str,
    section_title: str | None = None,
) -> tuple[Match[str] | None, str | None]:
    """
    通过多种方式查找章节

    Args:
        document: 文档内容
        section_id: 目标章节ID
        section_title: 章节标题（可选，用于降级匹配）

    Returns:
        (匹配的章节, 匹配方式描述)
    """
    # 方法1: 通过section_id注释精确匹配
    match = _find_section_by_id(document, section_id)
    if match:
        return match, "section_id"

    # 方法2: 通过章节标题模糊匹配
    if section_title:
        match = _find_section_by_title(document, section_title)
        if match:
            logging.info(f"通过标题匹配找到章节: {section_title}")
            return match, "title"

    # 方法3: 列出可用的section_id帮助诊断
    available_ids = _list_available_section_ids(document)
    logging.error(f"章节ID映射失败。目标: {section_id}")
    if available_ids:
        logging.error(f"可用section_id (前10个): {available_ids[:10]}")
    else:
        logging.error("文档中未找到任何section_id注释")

    return None, None


def _normalize_sentence_tokens(sentence: str) -> list[str]:
    return [token for token in re.split(r"\s+", sentence.strip()) if token]


def _find_section_containing_offset(document: str, offset: int) -> Match[str] | None:
    if offset < 0:
        return None
    for match in SECTION_BLOCK_PATTERN.finditer(document):
        if match.start() <= offset < match.end():
            return match
    return None


def _find_section_by_sentences(document: str, sentences: Iterable[str]) -> Match[str] | None:
    """
    尝试通过扫描提供的句子片段来定位章节。
    当章节ID发生漂移但内容仍存在时很有用。
    """
    processed: list[tuple[list[str], str]] = []
    seen_keys: set[str] = set()
    for sentence in sentences:
        if not sentence:
            continue
        cleaned = sentence.strip()
        if not cleaned:
            continue
        tokens = _normalize_sentence_tokens(cleaned)
        key = " ".join(tokens)
        if not tokens or key in seen_keys:
            continue
        seen_keys.add(key)
        processed.append((tokens, cleaned))
    processed.sort(key=lambda item: len(item[1]), reverse=True)

    for tokens, original_sentence in processed:
        if not tokens:
            continue
        literal = original_sentence
        literal_index = document.find(literal)
        if literal_index != -1:
            section_match = _find_section_containing_offset(document, literal_index)
            if section_match:
                return section_match
        pattern = r"\s+".join(re.escape(token) for token in tokens)
        if not pattern:
            continue
        regex = re.compile(pattern, re.MULTILINE | re.IGNORECASE)
        regex_match = regex.search(document)
        if regex_match:
            section_match = _find_section_containing_offset(document, regex_match.start())
            if section_match:
                return section_match
    return None


@dataclass
class EditIntent:
    original_sentence: str
    revised_sentence: str
    expected_replacements: int = 1
    metadata: dict[str, Any] | None = None


@dataclass
class EditOutcome:
    applied: bool
    method: str
    similarity: float | None
    matched_fragment: str | None
    detail: str
    replacements: int = 0


@dataclass
class _TextSegment:
    raw_text: str
    clean_text: str
    start: int
    end: int
    label: str


@dataclass
class FineGrainedEditResult:
    """
    单次细粒度编辑应用的汇总统计。
    """

    updated_text: str
    sections_modified: int
    successful_edits: int
    failed_edits: list[str]
    total_replacements: int = 0

    @property
    def had_effect(self) -> bool:
        return self.sections_modified > 0 or self.successful_edits > 0


class EditCorrector:
    """
    句子级编辑的精确优先替换工具。
    """

    _SENTENCE_PATTERN = re.compile(r"[^。！？!?\n]*[。！？!?]+|[^\n]+", re.MULTILINE)

    def __init__(
        self,
        body_text: str,
        *,
        similarity_threshold: float = 0.65,  # 降低默认阈值以提高匹配成功率
        min_fuzzy_length: int = 15,  # 降低最小长度要求
    ):
        self._text = body_text
        self.similarity_threshold = max(0.4, min(similarity_threshold, 0.95))  # 更宽松的范围
        self.min_fuzzy_length = max(5, min_fuzzy_length)  # 更低的最小长度

    @property
    def text(self) -> str:
        return self._text

    def apply(self, intent: EditIntent) -> EditOutcome:
        original = (intent.original_sentence or "").strip()
        revised = intent.revised_sentence or ""
        expected = intent.expected_replacements or 1

        if not original or not revised.strip():
            return EditOutcome(
                applied=False,
                method="invalid",
                similarity=None,
                matched_fragment=None,
                detail="Empty original or revised sentence.",
            )

        if original == revised:
            return EditOutcome(
                applied=False,
                method="noop",
                similarity=1.0,
                matched_fragment=original,
                detail="Original and revised sentences are identical.",
            )

        candidate_pairs: list[tuple[str, str]] = [(original, revised)]

        display_original = self._convert_inline_math_to_display(original)
        display_revised = self._convert_inline_math_to_display(revised)
        if (display_original, display_revised) != (original, revised):
            candidate_pairs.append((display_original, display_revised))

        last_outcome: EditOutcome | None = None
        for candidate_original, candidate_revised in candidate_pairs:
            outcome = self._attempt_strategies(candidate_original, candidate_revised, expected)
            if outcome.applied:
                return outcome
            last_outcome = outcome

        return last_outcome or EditOutcome(
            applied=False,
            method="unmatched",
            similarity=None,
            matched_fragment=None,
            detail="No matching snippet satisfied similarity/occurrence guards.",
        )

    def _attempt_strategies(self, original: str, revised: str, expected: int) -> EditOutcome:
        strategies: tuple[Strategy, ...] = (
            self._try_literal,
            self._try_whitespace_normalised,
            self._try_unescaped_literal,
            self._try_casefold_literal,
            self._try_math_normalised_segment,
            self._try_segment_similarity,
            self._try_fuzzy_window,
            self._try_relaxed_segment_similarity,
            self._try_substring_fuzzy_match,  # 🆕 新增：最后的降级策略
        )

        for strategy in strategies:
            result = strategy(original, revised, expected)
            if result:
                new_text, method, score, fragment, detail, replacements = result
                if new_text != self._text:
                    self._text = new_text
                return EditOutcome(True, method, score, fragment, detail, replacements)

        return EditOutcome(
            applied=False,
            method="unmatched",
            similarity=None,
            matched_fragment=None,
            detail="No matching snippet satisfied similarity/occurrence guards.",
        )

    # --- Strategy implementations -------------------------------------------------

    def _try_literal(self, original: str, revised: str, expected: int) -> CorrectionResult | None:
        occurrences = self._text.count(original)
        if occurrences == 0:
            return None
        if occurrences > 1 and expected <= 1:
            logging.debug(
                "Literal replacement skipped because the target sentence appears %s times; deferring to contextual strategies.",
                occurrences,
            )
            return None

        max_replacements = expected if expected > 0 else occurrences
        new_text, replacements = self._safe_literal_replace(self._text, original, revised, max_replacements)
        if replacements == 0:
            return None
        if expected > 0 and replacements < expected:
            logging.debug(
                "Literal replacement applied %s/%s expected occurrences; proceeding with partial match.",
                replacements,
                expected,
            )

        return (
            new_text,
            "literal",
            1.0,
            original,
            f"Replaced literal match ({replacements}).",
            replacements,
        )

    def _try_whitespace_normalised(self, original: str, revised: str, expected: int) -> CorrectionResult | None:
        pattern = self._build_whitespace_flexible_pattern(original)
        matches = list(re.finditer(pattern, self._text, flags=re.MULTILINE))
        if len(matches) != 1:
            logging.debug(
                "Whitespace-normalised search matched %s fragments; expected singular.",
                len(matches),
            )
            return None

        match = matches[0]
        fragment = match.group(0)
        score = self._similarity(original, fragment)
        new_text = self._replace_span(match.start(), match.end(), revised)
        return (
            new_text,
            "whitespace_normalised",
            score,
            fragment,
            "Whitespace-normalised literal replacement.",
            1,
        )

    def _try_unescaped_literal(self, original: str, revised: str, expected: int) -> CorrectionResult | None:
        unescaped_original, changed = self._maybe_unescape_literal(original)
        if not changed:
            return None

        occurrences = self._text.count(unescaped_original)
        if occurrences == 0:
            return None

        unescaped_revised, revised_changed = self._maybe_unescape_literal(revised)
        replacement = unescaped_revised if revised_changed else revised
        max_replacements = expected if expected > 0 else occurrences
        new_text, replacements = self._safe_literal_replace(self._text, unescaped_original, replacement, max_replacements)
        if replacements == 0:
            return None
        if expected > 0 and replacements < expected:
            logging.debug(
                "Unescaped literal replacement applied %s/%s expected occurrences; proceeding with partial match.",
                replacements,
                expected,
            )

        score = self._similarity(original, unescaped_original)
        return (
            new_text,
            "unescaped_literal",
            score,
            unescaped_original,
            "Literal replacement after correcting escape sequences.",
            replacements,
        )

    def _try_casefold_literal(self, original: str, revised: str, expected: int) -> CorrectionResult | None:
        original_cf = original.casefold()
        text_cf = self._text.casefold()
        idx = text_cf.find(original_cf)
        if idx == -1:
            return None
        second_idx = text_cf.find(original_cf, idx + len(original_cf))
        if second_idx != -1:
            logging.debug("Casefold match not unique; skipping.")
            return None
        fragment = self._text[idx : idx + len(original)]
        new_text = self._replace_span(idx, idx + len(original), revised)
        return (
            new_text,
            "casefold_literal",
            1.0,
            fragment,
            "Case-insensitive literal replacement.",
            1,
        )

    def _try_math_normalised_segment(self, original: str, revised: str, expected: int) -> CorrectionResult | None:
        stripped_original = self._strip_math_expressions(original)
        if len(stripped_original.strip()) < 8:
            return None

        best_segment: _TextSegment | None = None
        best_score = 0.0

        for segment in self._iter_segments():
            if self._should_skip_segment(segment):
                continue
            stripped_segment = self._strip_math_expressions(segment.clean_text)
            if not stripped_segment.strip():
                continue
            score = self._similarity(stripped_original, stripped_segment)
            if score >= max(self.similarity_threshold - 0.2, 0.45) and score > best_score:  # 更宽松的匹配
                best_segment = segment
                best_score = score

        if not best_segment:
            return None

        replacement = self._preserve_whitespace(best_segment.raw_text, revised)
        new_text = self._replace_span(best_segment.start, best_segment.end, replacement)
        return (
            new_text,
            f"segment_math_{best_segment.label}",
            best_score,
            best_segment.raw_text,
            f"Math-normalised segment similarity {best_score:.2f}",
            1,
        )

    def _try_segment_similarity(self, original: str, revised: str, expected: int) -> CorrectionResult | None:
        best_segment: _TextSegment | None = None
        best_score = 0.0

        for segment in self._iter_segments():
            if self._should_skip_segment(segment):
                continue
            score = self._similarity(original, segment.clean_text)
            if score >= self.similarity_threshold and score > best_score:
                best_segment = segment
                best_score = score

        if not best_segment:
            return None

        replacement = self._preserve_whitespace(best_segment.raw_text, revised)
        new_text = self._replace_span(best_segment.start, best_segment.end, replacement)
        return (
            new_text,
            f"segment_{best_segment.label}",
            best_score,
            best_segment.raw_text,
            f"Segment similarity {best_score:.2f}",
            1,
        )

    def _try_fuzzy_window(self, original: str, revised: str, expected: int) -> CorrectionResult | None:
        norm_len = len(self._normalize_for_similarity(original))
        if norm_len < self.min_fuzzy_length:
            return None

        best_span: tuple[int, int, float] | None = None
        for start, end in self._candidate_paragraph_spans(original):
            refined_start, refined_end, score = self._refine_span(start, end, original)
            if score >= max(self.similarity_threshold - 0.15, 0.5):  # 更宽松的模糊窗口匹配
                if not best_span or score > best_span[2]:
                    best_span = (refined_start, refined_end, score)

        if not best_span:
            return None

        span_start, span_end, score = best_span
        fragment = self._text[span_start:span_end]
        replacement = self._preserve_whitespace(fragment, revised)
        new_text = self._replace_span(span_start, span_end, replacement)
        return (
            new_text,
            "fuzzy_window",
            score,
            fragment,
            f"Fuzzy window similarity {score:.2f}",
            1,
        )

    def _try_relaxed_segment_similarity(self, original: str, revised: str, expected: int) -> CorrectionResult | None:
        """
        Final safety net: choose the most similar segment even if it falls below the
        strict similarity threshold, provided it is still reasonably close.
        """
        best_segment: _TextSegment | None = None
        best_score = 0.0

        for segment in self._iter_segments():
            if self._should_skip_segment(segment):
                continue
            score = self._similarity(original, segment.clean_text)
            if score > best_score:
                best_score = score
                best_segment = segment

        # 要求宽松但非平凡的匹配，避免虚构的替换
        relaxed_threshold = max(self.similarity_threshold - 0.35, 0.4)  # 更宽松的最终回退策略
        if not best_segment or best_score < relaxed_threshold:
            return None

        replacement = self._preserve_whitespace(best_segment.raw_text, revised)
        new_text = self._replace_span(best_segment.start, best_segment.end, replacement)
        return (
            new_text,
            f"segment_relaxed_{best_segment.label}",
            best_score,
            best_segment.raw_text,
            f"Relaxed segment similarity {best_score:.2f}",
            1,
        )

    def _try_substring_fuzzy_match(self, original: str, revised: str, expected: int) -> CorrectionResult | None:
        """
        最后的降级策略：基于子串和关键词的模糊匹配

        当所有其他策略都失败时，尝试通过核心词汇覆盖率来匹配。
        这是为了处理 AI 生成的原句与文档有轻微差异的情况。
        """
        if len(original) < self.min_fuzzy_length:
            return None

        # 提取核心词汇（长度>=3的词，避免停用词）
        core_words = [word for word in re.findall(r"\w{3,}", original) if word]
        if not core_words or len(core_words) < 3:
            return None

        best_segment = None
        best_score = 0.0
        best_coverage = 0.0

        for segment in self._iter_segments():
            if self._should_skip_segment(segment):
                continue

            segment_text = segment.clean_text.lower()

            # 计算核心词汇覆盖率
            matches = sum(1 for word in core_words if word.lower() in segment_text)
            coverage = matches / len(core_words)

            # 至少需要60%的核心词匹配
            if coverage >= 0.6:
                similarity = self._similarity(original, segment.clean_text)

                # 综合考虑覆盖率和相似度
                combined_score = coverage * 0.4 + similarity * 0.6

                if combined_score > best_score:
                    best_segment = segment
                    best_score = combined_score
                    best_coverage = coverage

        # 更宽松的阈值：综合分数 >= 0.45 或相似度 >= 0.35
        if best_segment and (best_score >= 0.45 or self._similarity(original, best_segment.clean_text) >= 0.35):
            actual_similarity = self._similarity(original, best_segment.clean_text)
            logging.info("    · 使用子串模糊匹配 (coverage=%.0f%%, similarity=%.2f, combined=%.2f)", best_coverage * 100, actual_similarity, best_score)

            replacement = self._preserve_whitespace(best_segment.raw_text, revised)
            new_text = self._replace_span(best_segment.start, best_segment.end, replacement)

            if new_text == self._text:
                return None

            return (
                new_text,
                f"substring_fuzzy_{best_segment.label}",
                actual_similarity,
                best_segment.raw_text,
                f"Substring fuzzy match (coverage={best_coverage:.0%}, sim={actual_similarity:.2f})",
                1,
            )

        return None

    # --- Helper utilities --------------------------------------------------------

    @staticmethod
    def _safe_literal_replace(text: str, needle: str, replacement: str, max_replacements: int) -> tuple[str, int]:
        if not needle:
            return text, 0

        limit = max_replacements if max_replacements > 0 else 1
        pieces: list[str] = []
        start = 0
        replaced = 0

        while True:
            idx = text.find(needle, start)
            if idx == -1 or replaced >= limit:
                pieces.append(text[start:])
                break
            pieces.append(text[start:idx])
            pieces.append(replacement)
            replaced += 1
            start = idx + len(needle)

        return "".join(pieces), replaced

    def _replace_span(self, start: int, end: int, replacement: str) -> str:
        return self._text[:start] + replacement + self._text[end:]

    @staticmethod
    def _maybe_unescape_literal(text: str) -> tuple[str, bool]:
        if not text or "\\" not in text:
            return text, False
        try:
            candidate = codecs.decode(text, "unicode_escape")
        except Exception:
            return text, False
        if candidate == text:
            return text, False
        return candidate, True

    @staticmethod
    def _normalize_for_similarity(text: str) -> str:
        normalized = unicodedata.normalize("NFKC", text or "")
        normalized = normalized.casefold()
        normalized = re.sub(r"\s+", " ", normalized)
        return normalized.strip()

    def _similarity(self, left: str, right: str) -> float:
        left_norm = self._normalize_for_similarity(left)
        right_norm = self._normalize_for_similarity(right)
        if not left_norm or not right_norm:
            return 0.0
        return SequenceMatcher(None, left_norm, right_norm).ratio()

    @staticmethod
    def _build_whitespace_flexible_pattern(text: str) -> str:
        parts = [re.escape(part) for part in re.split(r"\s+", text.strip()) if part]
        return r"\s*".join(parts)

    @staticmethod
    def _strip_math_expressions(text: str) -> str:
        if not text:
            return ""
        # 先移除块级数学表达式
        cleaned = re.sub(r"\$\$.*?\$\$", " ", text, flags=re.DOTALL)
        # 再移除行内数学表达式
        cleaned = re.sub(r"\$(?:\\.|[^$])+\$", " ", cleaned)
        # 移除多余空白
        cleaned = re.sub(r"\s+", " ", cleaned)
        return cleaned.strip()

    @staticmethod
    def _convert_inline_math_to_display(text: str) -> str:
        if not text or "$" not in text or "$$" in text:
            return text

        pattern = re.compile(r"\$(.+?)\$", re.DOTALL)

        def _to_display(match: re.Match[str]) -> str:
            expr = match.group(1)
            if expr is None:
                return match.group(0)
            stripped = expr.strip()
            if not stripped:
                return match.group(0)
            if "=" not in stripped and len(stripped) <= 40:
                return match.group(0)
            return f"\n\n$${stripped}$$\n\n"

        converted = pattern.sub(_to_display, text)
        if converted != text:
            converted = re.sub(r"\n{3,}", "\n\n", converted)
        return converted

    def _iter_segments(self) -> Iterator[_TextSegment]:
        yield from self._line_window_segments()
        yield from self._sentence_segments()

    def _line_window_segments(self) -> Iterator[_TextSegment]:
        lines = self._text.splitlines(True)
        offsets = [0]
        for line in lines:
            offsets.append(offsets[-1] + len(line))

        max_window = 3
        seen: set[tuple[int, int]] = set()
        for i in range(len(lines)):
            for width in range(1, max_window + 1):
                if i + width > len(lines):
                    break
                start = offsets[i]
                end = offsets[i + width]
                span = (start, end)
                if span in seen:
                    continue
                seen.add(span)
                raw = self._text[start:end]
                clean = raw.strip()
                if not clean:
                    continue
                yield _TextSegment(raw, clean, start, end, f"line_window_{width}")

    def _sentence_segments(self) -> Iterator[_TextSegment]:
        seen: set[tuple[int, int]] = set()
        for match in self._SENTENCE_PATTERN.finditer(self._text):
            start, end = match.start(), match.end()
            span = (start, end)
            if span in seen:
                continue
            seen.add(span)
            raw = self._text[start:end]
            clean = raw.strip()
            if not clean:
                continue
            yield _TextSegment(raw, clean, start, end, "sentence")

    @staticmethod
    def _should_skip_segment(segment: _TextSegment) -> bool:
        clean = segment.clean_text.lstrip()
        if not clean:
            return True
        if clean.startswith("#"):
            return True
        if "section_id:" in segment.raw_text:
            return True
        return False

    @staticmethod
    def _preserve_whitespace(original_fragment: str, revised: str) -> str:
        leading_match = re.match(r"^\s*", original_fragment)
        trailing_match = re.search(r"\s*$", original_fragment)
        leading = leading_match.group(0) if leading_match else ""
        trailing = trailing_match.group(0) if trailing_match else ""
        stripped_revised = revised.strip("\n")
        replacement = stripped_revised
        if leading and not replacement.startswith(leading):
            replacement = leading + replacement
        if trailing and not replacement.endswith(trailing):
            replacement = replacement + trailing
        return replacement

    def _candidate_paragraph_spans(self, original: str) -> Iterator[tuple[int, int]]:
        tokens = [tok for tok in re.findall(r"\w+", original) if len(tok) > 3]
        token_pattern = re.compile("|".join(re.escape(tok) for tok in set(tokens)), re.IGNORECASE) if tokens else None
        pattern = re.compile(r".+?(?:\n\s*\n|\Z)", re.DOTALL)
        for match in pattern.finditer(self._text):
            raw = match.group(0)
            if token_pattern and not token_pattern.search(raw):
                continue
            start, end = match.start(), match.end()
            if raw.strip():
                yield start, end

    def _refine_span(self, start: int, end: int, original: str) -> tuple[int, int, float]:
        snippet = self._text[start:end]
        matcher = SequenceMatcher(None, original, snippet)
        blocks = [block for block in matcher.get_matching_blocks() if block.size]
        if not blocks:
            score = self._similarity(original, snippet)
            return start, end, score

        refined_start = start + blocks[0].b
        refined_end = start + blocks[-1].b + blocks[-1].size
        padding = max(2, int(len(original) * 0.1))
        refined_start = max(start, refined_start - padding)
        refined_end = min(end, refined_end + padding)

        candidate = self._text[refined_start:refined_end]
        score = self._similarity(original, candidate)
        return refined_start, refined_end, score


def apply_fine_grained_edits(
    current_solution: str,
    changes_list: Iterable[Any],
    section_number_map: dict[int, str] | None = None,
) -> FineGrainedEditResult:
    """
    在由section_id标识的章节内应用句子级细粒度补丁。

    Args:
        current_solution: 当前文档内容
        changes_list: 补丁列表
        section_number_map: 数字编号→UUID映射表（可选，用于支持简单数字引用）
    """
    # 检查文档大小，避免处理超大文档时性能问题
    if len(current_solution) > 1_000_000:  # 超过1MB
        logging.warning(f"⚠️  文档过大 ({len(current_solution):,} 字符 ≈ {len(current_solution) // 1024}KB)，可能处理较慢\n💡 建议: 考虑分段处理或优化文档结构")
    # noqa: W293
    modified_solution = current_solution
    changes = list(changes_list)
    logging.info("--- 开始应用 %s 个章节的细粒度修订 ---", len(changes))
    if section_number_map:
        logging.debug("  - 数字映射表：%s", {k: v[:8] + "..." for k, v in section_number_map.items()})
    total_successful_edits = 0
    total_failed_edits: list[str] = []
    sections_with_changes = 0
    total_replacements_applied = 0

    def _as_dict(change_item: Any) -> JsonDict:
        if isinstance(change_item, Mapping):
            return _normalize_mapping(change_item)
        try:
            dumped: Any = change_item.model_dump(mode="json")  # type: ignore[attr-defined]
        except AttributeError:
            try:
                dumped = dict(change_item)
            except (TypeError, ValueError):
                dumped = None
        if isinstance(dumped, Mapping):
            return _normalize_mapping(dumped)
        logging.warning("无法解析补丁变更项，已跳过：%s", change_item)
        return {}

    threshold_env = os.getenv("PATCH_SIMILARITY_THRESHOLD", "0.60")  # 降低默认环境变量阈值
    try:
        similarity_threshold = float(threshold_env)
        if similarity_threshold <= 0 or similarity_threshold > 1:
            raise ValueError
    except ValueError:
        similarity_threshold = 0.60  # 使用更低的回退阈值

    for change in changes:
        change_dict = _as_dict(change)
        raw_target_id = change_dict.get("target_id")

        # ═══════════════════════════════════════════════════════════════
        # 🥇 优先级 1：数字编号 → UUID 映射（最推荐）
        # ═══════════════════════════════════════════════════════════════
        if isinstance(raw_target_id, int):
            if section_number_map and raw_target_id in section_number_map:
                target_id = section_number_map[raw_target_id]
                logging.debug("  ✓ 数字编号 [%d] → UUID %s...", raw_target_id, target_id[:8] if len(target_id) >= 8 else target_id)
            else:
                valid_range = f"1-{len(section_number_map)}" if section_number_map else "无映射表"
                logging.warning("  ✗ 无效的数字编号 %d（有效范围:      %s），跳过此补丁", raw_target_id, valid_range)
                continue
        # ═══════════════════════════════════════════════════════════════
        # 🥈 优先级 2：字符串（UUID 或其他）
        # ═══════════════════════════════════════════════════════════════
        else:
            target_id = str(raw_target_id or "")

        edits_value = change_dict.get("edits", [])
        if not isinstance(edits_value, list):
            logging.debug("  - 章节 '%s' 的 edits 字段不是列表，跳过。", target_id or "<missing>")
            continue
        edits_raw: list[JsonDict] = [_normalize_mapping(edit_map) for edit_map in cast(list[Any], edits_value)]
        if not target_id:
            logging.warning("  - 遇到缺少 target_id 的补丁，已跳过。")
            continue
        if not edits_raw:
            logging.info("  - 章节 '%s' 无需修订，跳过。", target_id)
            continue

        original_sentences_for_fallback = [str(edit_dict.get("original_sentence", "")).strip() for edit_dict in edits_raw if str(edit_dict.get("original_sentence", "")).strip()]

        # 尝试从变更字典中提取章节标题（如果有）
        section_title = change_dict.get("section_title") or change_dict.get("title")

        # 使用增强的多方式匹配
        match, match_method = _find_section_by_multiple_methods(modified_solution, target_id, section_title)

        # 如果多方式匹配失败，尝试基于原句内容的回退匹配
        if not match and original_sentences_for_fallback:
            fallback_match = _find_section_by_sentences(modified_solution, original_sentences_for_fallback)
            if fallback_match:
                logging.warning("  - 未能通过ID '%s' 定位章节，已基于原句内容回退匹配。", target_id)
                match = fallback_match
                match_method = "sentences"

        if not match:
            logging.warning("  - 未能找到ID为 '%s' 的章节块，跳过此修订。", target_id)
            total_failed_edits.append(target_id)
            continue

        # 记录成功的匹配方式
        if match_method:
            logging.info(f"  - 章节 '{target_id}' 通过 {match_method} 方式匹配成功")

        original_section_content = match.group(1)
        section_lines = original_section_content.splitlines()
        heading_line = section_lines[0] if section_lines else ""
        body_text = "\n".join(section_lines[1:]) if len(section_lines) > 1 else ""

        corrector = EditCorrector(body_text, similarity_threshold=similarity_threshold)
        applied_count = 0
        replacement_total = 0
        section_failures: list[str] = []

        original_sentence = ""
        revised_sentence = ""
        for edit_dict in edits_raw:
            original_sentence = str(edit_dict.get("original_sentence", ""))
            revised_sentence = str(edit_dict.get("revised_sentence", ""))

            if not original_sentence.strip() or not revised_sentence.strip():
                logging.warning(
                    "  - 章节 '%s' 修订跳过：存在空的原句或修订句。",
                    target_id,
                )
                continue

            if original_sentence.lstrip().startswith("#") or "section_id:" in original_sentence:
                logging.debug("  - 跳过触及标题/ID 的修订。")
                continue

            metadata_obj = edit_dict.get("metadata")
            metadata: dict[str, Any] | None
            if isinstance(metadata_obj, Mapping):
                metadata = _normalize_mapping(metadata_obj)
            else:
                metadata = None
            expected_source = metadata.get("expected_replacements") if metadata else edit_dict.get("expected_replacements", 1)
            expected_raw: Any = expected_source
            try:
                expected_int = int(expected_raw)
            except (TypeError, ValueError):
                expected_int = 1
            if expected_int <= 0:
                expected_int = 1

            intent = EditIntent(
                original_sentence=original_sentence,
                revised_sentence=revised_sentence,
                expected_replacements=expected_int,
                metadata=metadata,
            )
            outcome = corrector.apply(intent)
            if outcome.applied:
                if expected_int > 0 and outcome.replacements != expected_int:
                    logging.warning(
                        "  - 章节 '%s' 修订替换次数不符 (expected=%s, actual=%s)。将继续应用此修订。",
                        target_id,
                        expected_int,
                        outcome.replacements,
                    )
                applied_count += 1
                replacement_total += max(1, outcome.replacements)
                if outcome.similarity is not None:
                    logging.info(
                        "    · 修订命中 (%s · replacements=%s · similarity=%.2f)",
                        outcome.method,
                        outcome.replacements or 1,
                        outcome.similarity,
                    )
                else:
                    logging.info(
                        "    · 修订命中 (%s · replacements=%s)",
                        outcome.method,
                        outcome.replacements or 1,
                    )
            else:
                # 简化日志：详细信息放到 DEBUG 级别
                logging.debug(
                    "  - 章节 '%s' 修订未命中: %s 原句片段: '%s...'",
                    target_id,
                    outcome.detail,
                    original_sentence[:50],
                )
                section_failures.append(f"{target_id}: {outcome.detail} ({original_sentence[:30]}...)")

        updated_body = corrector.text
        if applied_count > 0 and updated_body != body_text:
            new_section_content = heading_line
            if updated_body:
                new_section_content += "\n" + updated_body
            modified_solution = modified_solution.replace(original_section_content, new_section_content, 1)
            logging.info(
                "  - 成功向章节 '%s' 应用了 %s/%s 条句子级修订（实际替换 %s 次）。",
                target_id,
                applied_count,
                len(edits_raw),
                replacement_total,
            )
            sections_with_changes += 1
            total_replacements_applied += replacement_total
        else:
            logging.info(
                "  - 章节 '%s' 无实际内容变更（命中 %s 条修订）。",
                target_id,
                applied_count,
            )

        if section_failures:
            total_failed_edits.extend(section_failures)
        if applied_count:
            total_successful_edits += applied_count

    logging.info("--- 所有细粒度修订应用完毕 ---")
    if total_successful_edits:
        logging.info("  · 成功命中的修订条数：%s", total_successful_edits)
    if total_failed_edits:
        # 简化日志：只显示未命中数量，详细原因在 DEBUG
        logging.info("  · 未命中的修订条数：%s", len(total_failed_edits))
        logging.debug("  · 未命中详情：%s", "; ".join(total_failed_edits))
    return FineGrainedEditResult(
        updated_text=modified_solution,
        sections_modified=sections_with_changes,
        successful_edits=total_successful_edits,
        failed_edits=total_failed_edits,
        total_replacements=total_replacements_applied,
    )
