from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator, model_validator

# ------------------------------------------------------------------------------
# Shared sentence-level edit structures
# ------------------------------------------------------------------------------


class SentenceEdit(BaseModel):
    """用于润色和优化流程的单个句子编辑。"""

    original_sentence: str = Field(
        ...,
        description="需要修改的原句。**CRITICAL: MUST copy EXACTLY from the existing draft, character by character. DO NOT paraphrase, rewrite, or modify the original text in any way.**"
    )
    revised_sentence: str = Field(..., description="修改后的句子。")


class EditList(BaseModel):
    """润色编辑的顶层容器。"""

    edits: list[SentenceEdit] = Field(default_factory=list, description="句子级修改列表。")


# ------------------------------------------------------------------------------
# Iterative refinement patches
# ------------------------------------------------------------------------------


class FineGrainedPatch(BaseModel):
    """针对特定章节的句子级补丁。"""

    target_id: int = Field(
        ...,
        description="目标章节的简单数字编号，例如 1, 2, 3, 4, 5。使用 [Available Sections] 列表中对应的数字。",
        ge=1,  # 必须 >= 1
    )
    edits: list[SentenceEdit] = Field(default_factory=list, description="需要应用的句子修改。")

    @field_validator("target_id", mode="before")
    @classmethod
    def _coerce_target_id(cls, value: Any) -> int:
        """将各种输入格式强制转换为整数"""
        # 直接接受数字（最推荐）
        if isinstance(value, int):
            return value
        # 字符串转整数
        if isinstance(value, str):
            stripped = value.strip()
            # 尝试转换为整数
            try:
                return int(stripped)
            except (ValueError, TypeError):
                raise ValueError(f"target_id 必须是整数，收到: {repr(value)}")
        # 其他类型报错
        raise TypeError(f"target_id 必须是整数（1, 2, 3...），不支持类型: {type(value).__name__}")


class FineGrainedPatchList(BaseModel):
    """句子级补丁的集合。"""

    patches: list[FineGrainedPatch] = Field(default_factory=list, description="所有章节的补丁集合。每个补丁的 target_id 必须使用简单数字（1, 2, 3...）。")


# ------------------------------------------------------------------------------
# Outline planning structures
# ------------------------------------------------------------------------------


class PlanChapter(BaseModel):
    """层次化大纲节点。"""

    title: str = Field(..., description="章节标题。")
    description: str = Field(default="", description="章节描述（可选，AI 未提供时使用空字符串）。")
    target_chars_ratio: float | None = Field(default=None, description="章节占总字数比例；可留空以便后续自动分配。")
    sections: list[PlanChapter] | None = Field(default_factory=list, description="子章节，可选。")

    @field_validator("target_chars_ratio", mode="before")
    @classmethod
    def _convert_ratio_to_float(cls, value: Any) -> float | None:
        """将字符串转换为 float（处理 LLM 返回字符串数字的情况）"""
        if value is None or value == "":
            return None
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value.strip())
            except (ValueError, AttributeError):
                return None
        return None

    @field_validator("target_chars_ratio")
    @classmethod
    def _ensure_ratio_bounds(cls, value: float | None) -> float | None:
        if value is None:
            return None
        if 0 <= value <= 1:
            return value
        raise ValueError("target_chars_ratio must be between 0 and 1 when provided.")


class PlanModel(BaseModel):
    """大纲模型返回的结构化规划输出。"""

    title: str = Field(..., description="文档主标题。")
    outline: list[PlanChapter] = Field(..., description="章节列表。")
    total_estimated_chars: int | None = Field(None, description="预估总字数。")
    target_audience: str | None = Field(None, description="目标读者。")
    key_objectives: list[str] | None = Field(default_factory=list, description="核心目标。")


# ------------------------------------------------------------------------------
# Drafting structures
# ------------------------------------------------------------------------------


class SectionContent(BaseModel):
    """单个草稿章节。"""

    section_id: str = Field(..., description="章节 ID。")
    title: str = Field(..., description="章节标题。")
    content: str = Field(..., description="章节正文。")
    key_claims: list[str] = Field(default_factory=list, description="关键主张。")
    todos: list[str] = Field(default_factory=list, description="待办事项。")
    word_count: int | None = Field(None, description="字数统计。")


class DraftModel(BaseModel):
    """由章节组成的结构化草稿。"""

    sections: list[SectionContent] = Field(default_factory=list, description="所有章节。")
    document_title: str = Field(..., description="文档标题。")
    summary: str | None = Field(None, description="摘要。")
    total_word_count: int | None = Field(None, description="总字数。")
    writing_style_notes: str | None = Field(None, description="风格说明。")


# ------------------------------------------------------------------------------
# Critique structures
# ------------------------------------------------------------------------------


class RubricScores(BaseModel):
    """Numerical rubric used during critique."""

    coverage: int = Field(ge=1, le=10, description="覆盖度评分。")
    correctness: int = Field(ge=1, le=10, description="正确性评分。")
    verifiability: int = Field(ge=1, le=10, description="可验证性评分。")
    coherence: int = Field(ge=1, le=10, description="连贯性评分。")
    style_fit: int = Field(ge=1, le=10, description="风格契合度评分。")
    math_symbol_correctness: int = Field(ge=1, le=10, description="数学符号正确性评分。")
    chapter_balance: int = Field(ge=1, le=10, description="章节平衡评分。")


class ContradictionEntry(BaseModel):
    """Flexible contradiction entry supporting structured responses."""

    description: str = Field("", description="矛盾或冲突的描述。")
    location_type: str | None = Field(None, description="定位类型（如行、章节等）。")
    location_value: str | None = Field(None, description="定位值或标识。")
    context_snippet: str | None = Field(None, description="相关上下文片段。")
    metadata: dict[str, Any] = Field(default_factory=dict, description="额外原始字段。")

    @model_validator(mode="before")
    @classmethod
    def _coerce(cls, value: Any) -> Any:
        if isinstance(value, str):
            return {"description": value}
        if isinstance(value, Mapping):
            raw = dict(value)
            description = raw.get("description") or raw.get("summary") or raw.get("message") or raw.get("detail") or raw.get("context_snippet") or ""
            return {
                "description": description,
                "location_type": raw.get("location_type"),
                "location_value": raw.get("location_value") or raw.get("location"),
                "context_snippet": raw.get("context_snippet"),
                "metadata": {
                    key: val
                    for key, val in raw.items()
                    if key
                    not in {
                        "description",
                        "summary",
                        "message",
                        "detail",
                        "context_snippet",
                        "location_type",
                        "location_value",
                        "location",
                    }
                },
            }
        return value


class ImprovementEntry(BaseModel):
    """Structured improvement suggestion returned by the reviewer."""

    section_title: str = Field(..., description="针对的章节标题。")
    advice: str = Field(..., description="具体改进建议。")
    section_id: str | None = Field(None, description="可选的章节 ID。")

    @model_validator(mode="before")
    @classmethod
    def _coerce(cls, value: Any) -> Any:
        if isinstance(value, str):
            return {
                "section_title": "未命名章节",
                "advice": value,
                "section_id": None,
            }
        if isinstance(value, Mapping):
            raw = dict(value)
            title = raw.get("section_title") or raw.get("title") or raw.get("heading") or "未命名章节"
            advice = raw.get("advice") or raw.get("suggestion") or raw.get("recommendation") or ""
            return {
                "section_title": title,
                "advice": advice,
                "section_id": raw.get("section_id"),
            }
        return value


class CritiqueModel(BaseModel):
    """Structured critique payload."""

    critique: str = Field(..., description="详细评审意见。")
    knowledge_gaps: list[str] = Field(default_factory=list, description="知识空白列表。")
    rubric: RubricScores | None = Field(None, description="评分细节。")
    contradictions: list[ContradictionEntry] = Field(default_factory=list, description="发现的矛盾。")
    improvements: list[ImprovementEntry] = Field(default_factory=list, description="改进建议。")
    priority_issues: list[str] = Field(default_factory=list, description="优先处理的问题。")
    overall_quality_score: int | None = Field(None, ge=1, le=10, description="整体评分。")


# ------------------------------------------------------------------------------
# Polishing structures
# ------------------------------------------------------------------------------


class PolishReference(BaseModel):
    """Reference metadata captured during polishing."""

    reference_id: str = Field(..., description="引用 ID。")
    citation_text: str = Field(..., description="引用文本。")
    source_type: str = Field(..., description="来源类型。")
    confidence_level: float = Field(..., ge=0, le=1, description="可信度。")
    source_info: dict[str, Any] = Field(default_factory=dict, description="来源信息。")


class PolishSection(BaseModel):
    """Structured polished chapter."""

    section_id: str = Field(..., description="章节 ID。")
    title: str = Field(..., description="章节标题。")
    content: str = Field(..., description="润色后内容。")
    original_content: str | None = Field(None, description="原始内容。")
    modifications: list[SentenceEdit] = Field(default_factory=list, description="修改记录。")
    references: list[PolishReference] = Field(default_factory=list, description="引用列表。")
    quality_metrics: dict[str, float] | None = Field(None, description="质量指标。")
    word_count: int = Field(..., description="章节字数。")
    revision_notes: str | None = Field(None, description="修改说明。")
    fact_check_points: list[str] | None = Field(default=None, description="待事实核查要点。")


class PolishSectionResponse(BaseModel):
    """Minimal schema expected from the polishing LLM."""

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    revised_content: str = Field(..., description="润色后的章节内容。")
    modifications: list[SentenceEdit] = Field(default_factory=list, description="修改记录。")
    quality_metrics: dict[str, float] | None = Field(None, description="质量指标。")
    revision_notes: str | None = Field(None, description="修改说明。")
    fact_check_points: list[str] | None = Field(default=None, description="待事实核查要点。")
    references: list[Any] = Field(default_factory=list, description="引用列表，可为空或包含文本/对象。")
    word_count: int | None = Field(
        default=None,
        description="润色后的章节字数，可选。",
        validation_alias=AliasChoices("_word_count", "word_count"),
        serialization_alias="word_count",
    )

    @field_validator("quality_metrics", mode="before")
    @classmethod
    def _coerce_quality_metrics(cls, value: Any) -> dict[str, float] | None:
        """转换 quality_metrics 中的字符串值为浮点数"""
        if value is None:
            return None
        if isinstance(value, dict):
            result = {}
            for key, val in value.items():
                try:
                    # 尝试将值转换为浮点数
                    if isinstance(val, str):
                        result[key] = float(val)
                    elif isinstance(val, (int, float)):
                        result[key] = float(val)
                    else:
                        result[key] = val  # 保留原值
                except (ValueError, TypeError):
                    result[key] = val  # 转换失败时保留原值
            return result
        return value

    @field_validator("fact_check_points", mode="before")
    @classmethod
    def _coerce_fact_check_points(cls, value: Any) -> list[str] | None:
        """转换 fact_check_points，处理各种输入格式"""
        if value is None:
            return None
        if isinstance(value, list):
            result = []
            for item in value:
                # 如果是字典，提取 fact_check_point 字段
                if isinstance(item, dict):
                    if "fact_check_point" in item:
                        result.append(str(item["fact_check_point"]))
                    elif "point" in item:
                        result.append(str(item["point"]))
                    else:
                        # 如果没有这些字段，转换整个字典为字符串
                        result.append(str(item))
                elif isinstance(item, str):
                    result.append(item)
                else:
                    result.append(str(item))
            return result
        return value

    @field_validator("word_count", mode="before")
    @classmethod
    def _coerce_word_count(cls, value: Any) -> int | None:
        """强制转换 word_count 为非负整数，处理字符串、负数等各种输入"""
        if value in (None, "", "null"):
            return None
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, (int, float)):
            return max(0, int(value))
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped or stripped.lower() in ("null", "none", "n/a"):
                return None
            # 移除所有非数字字符（保留负号）
            digits_only = re.sub(r"[^0-9-]+", "", stripped)
            if not digits_only or digits_only == "-":
                return None
            try:
                parsed = int(digits_only)
                return max(0, parsed)  # 确保非负
            except ValueError:
                return None
        try:
            return max(0, int(value))
        except (TypeError, ValueError):
            return None


class PolishModel(BaseModel):
    """Structured output for the polishing workflow."""

    sections: list[PolishSection] = Field(default_factory=list, description="润色后的章节。")
    document_title: str = Field(..., description="文档标题。")
    polished_content: str = Field(..., description="完整润色内容。")
    original_content: str = Field(..., description="原始文档。")
    metadata: dict[str, Any] = Field(default_factory=dict, description="额外元数据。")
    overall_quality_score: float | None = Field(None, ge=0, le=1, description="整体质量评分。")
    modification_summary: str | None = Field(None, description="修改总结。")
    all_references: list[PolishReference] = Field(default_factory=list, description="全部引用。")
    reference_validation_status: dict[str, str] | None = Field(None, description="引用验证结果。")
    citations_enabled: bool | None = Field(False, description="是否启用引用。")
    citation_style: str | None = Field(None, description="引用风格。")
    fact_check_points: list[str] | None = Field(None, description="待事实核查要点。")
    validation_needed: bool | None = Field(False, description="是否需要事实核查。")
    fact_check_results: Any | None = Field(None, description="事实核查结果。")


# ------------------------------------------------------------------------------
# Helper utilities for polishing
# ------------------------------------------------------------------------------


def extract_references_from_text(text: str) -> list[PolishReference]:
    """Lightweight extraction of references using simple regex heuristics."""

    if not text:
        return []

    references: list[PolishReference] = []
    patterns: list[str] = [
        r"\[(\d+)\]",  # numeric references like [1]
        r"\((\d{4})\)",  # year references like (2023)
        r'"([^"]+)"',  # quoted titles
    ]

    for pattern_index, pattern in enumerate(patterns):
        for match_index, match in enumerate(re.findall(pattern, text)):
            references.append(
                PolishReference(
                    reference_id=f"ref_{pattern_index}_{match_index}",
                    citation_text=match,
                    source_type="extracted",
                    confidence_level=0.5,
                    source_info={"pattern_index": pattern_index, "method": "regex"},
                )
            )

    return references


def validate_references_quality(references: list[PolishReference]) -> dict[str, str]:
    """Classify references into quality buckets."""

    status: dict[str, str] = {}
    for ref in references:
        if ref.confidence_level < 0.3:
            status[ref.reference_id] = "需要验证"
        elif ref.confidence_level > 0.8:
            status[ref.reference_id] = "可信"
        else:
            status[ref.reference_id] = "部分可信"
    return status


__all__ = [
    "SentenceEdit",
    "EditList",
    "FineGrainedPatch",
    "FineGrainedPatchList",
    "PlanChapter",
    "PlanModel",
    "SectionContent",
    "DraftModel",
    "RubricScores",
    "ContradictionEntry",
    "ImprovementEntry",
    "CritiqueModel",
    "PolishReference",
    "PolishSection",
    "PolishSectionResponse",
    "PolishModel",
    "extract_references_from_text",
    "validate_references_quality",
]
