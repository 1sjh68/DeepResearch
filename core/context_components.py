# ruff: noqa: E501
from __future__ import annotations

import hashlib
import logging
import re
import uuid
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, cast

from config import Config
from services.vector_db import EmbeddingModel, VectorDBManager
from utils.text_processor import (
    chunk_document_for_rag,
    truncate_text_for_context_boundary_aware,
)

OutlineNode = dict[str, Any]
MetadataDict = dict[str, Any]
ChunkResult = tuple[list[str], list[MetadataDict]]


def _normalize_outline_node(mapping: Mapping[Any, Any]) -> OutlineNode:
    normalized: OutlineNode = {}
    for key, value in mapping.items():
        normalized[str(key)] = value
    return normalized


def _normalized_nodes(sequence: Sequence[Mapping[Any, Any]]) -> list[OutlineNode]:
    return [_normalize_outline_node(item) for item in sequence]


def _filter_mappings(sequence: Iterable[Any]) -> list[Mapping[Any, Any]]:
    return [item for item in sequence if isinstance(item, Mapping)]


def _normalize_metadata(mapping: Mapping[Any, Any]) -> MetadataDict:
    normalized: MetadataDict = {}
    for key, value in mapping.items():
        normalized[str(key)] = value
    return normalized


@dataclass
class RAGService:
    """轻量级协调器，管理外部数据的向量索引。"""

    config: Config
    embedding_model: EmbeddingModel | None
    _vector_manager: VectorDBManager | None = None
    _collection_name: str | None = None
    _initialized: bool = False
    _external_data_hash: str | None = None

    def _reset_index(self) -> None:
        manager = self._vector_manager
        if manager and manager.client and self._collection_name:
            try:
                manager.client.delete_collection(name=self._collection_name)
            except Exception as exc:
                logging.warning("RAGService: 删除旧向量集合 '%s' 失败: %s", self._collection_name, exc)
        self._vector_manager = None
        self._collection_name = None
        self._initialized = False
        self._external_data_hash = None

    def ensure_index(self, external_data: str) -> None:
        if not external_data or not self.embedding_model:
            return

        data_hash = hashlib.sha256(external_data.encode("utf-8")).hexdigest()

        if self._initialized:
            if data_hash == self._external_data_hash:
                logging.debug("RAGService: 外部数据未变化，跳过索引重建。")
                return
            logging.info("RAGService: 检测到外部数据已更新，正在重建 RAG 索引。")
            self._reset_index()

        try:
            manager = VectorDBManager(self.config, self.embedding_model)
            if not manager or not manager.client:
                logging.warning("RAGService: 无法初始化向量数据库管理器。")
                return

            session_id_part = ""
            if self.config.session_dir:
                session_id_part = self.config.session_dir.split("_")[-1]
            if not session_id_part or not session_id_part.isalnum():
                session_id_part = uuid.uuid4().hex[:8]

            doc_id: str = f"session_doc_{session_id_part}"
            chunks: list[str]
            metadatas: list[MetadataDict]
            chunks, metadatas = chunk_document_for_rag(self.config, external_data, doc_id)
            metadatas = [dict(meta) for meta in metadatas]
            if not chunks:
                logging.warning("RAGService: 外部数据未生成有效分块，跳过索引。")
                return

            collection_name: str = f"rag_{doc_id}"
            manager.collection = manager.client.get_or_create_collection(name=collection_name)
            if manager.add_experience(texts=chunks, metadatas=metadatas):
                logging.info("RAGService: RAG 分块已写入集合 '%s'。", collection_name)
                self._vector_manager = manager
                self._collection_name = collection_name
                self._initialized = True
                self._external_data_hash = data_hash
            else:
                logging.warning("RAGService: RAG 分块写入失败。")
        except Exception as exc:  # pragma: no cover - defensive logging
            logging.error("RAGService: 初始化索引时发生错误: %s", exc, exc_info=True)

    def retrieve(self, query_text: str, n_results: int = 3) -> list[MetadataDict]:
        if not self._initialized or not self._vector_manager:
            return []
        try:
            raw_results: list[MetadataDict] = self._vector_manager.hybrid_retrieve_experience(query_text, n_results=n_results)
            typed_results: list[MetadataDict] = [_normalize_metadata(item) for item in raw_results]
            return typed_results
        except Exception as exc:
            logging.error("RAGService: 检索失败: %s", exc, exc_info=True)
            return []


@dataclass
class ContextRepository:
    """存储生成的章节和小节内容以及摘要。"""

    chapter_summaries: dict[str, str] = field(default_factory=dict)
    chapter_content: dict[str, str] = field(default_factory=dict)
    subsection_content: dict[str, dict[str, str]] = field(default_factory=dict)

    def record_chapter(self, chapter_title: str, content: str) -> None:
        self.chapter_content[chapter_title] = content

    def get_previous_chapter(self, title: str) -> str | None:
        return self.chapter_content.get(title)

    def set_summary(self, chapter_title: str, summary: str) -> None:
        self.chapter_summaries[chapter_title] = summary

    def get_summary(self, chapter_title: str) -> str | None:
        return self.chapter_summaries.get(chapter_title)


@dataclass
class ContextAssembler:
    """为草稿、续写和评审工作流构建上下文包。"""

    config: Config
    outline: Mapping[str, Any]
    style_guide: str
    repository: ContextRepository
    rag_service: RAGService | None = None

    def _truncate(self, text: str, limit: int, mode: str = "tail") -> str:
        ratio = self.config.generation.prompt_budget_ratio
        return truncate_text_for_context_boundary_aware(self.config, text, int(limit * ratio), mode)

    def _rag_context(self, query_text: str, n_results: int = 3) -> str:
        if not self.rag_service:
            return ""
        results = self.rag_service.retrieve(query_text, n_results=n_results)
        if not results:
            return ""
        parts: list[str] = ["\n\n--- 从参考PDF中检索到的高度相关原文片段 ---\\n"]
        for idx, chunk in enumerate(results, start=1):
            document_text = chunk.get("document", "内容缺失")
            parts.append(f"\n[相关原文片段 {idx}]\\n")
            parts.append(f"{document_text}\\n")
        parts.append("--- 原文片段结束 ---\\n\n")
        return "".join(parts)

    def build_chapter_context(self, chapter_title: str) -> str:
        chapters: list[OutlineNode] = self._extract_chapters()
        chapter_obj: OutlineNode | None = None
        chapter_index = -1
        for idx, chapter in enumerate(chapters):
            if self._get_title(chapter) == chapter_title:
                chapter_obj = chapter
                chapter_index = idx
                break

        if chapter_obj is None or chapter_index == -1:
            return "[错误：无法定位当前独立章节信息]"

        chapter_description = self._get_description(chapter_obj) or ""
        rag_context = self._rag_context(f"{chapter_title}: {chapter_description}")
        style_guide = self.style_guide or "无特定风格指南。"

        other_titles: list[str] = []
        for chapter in chapters:
            title = self._get_title(chapter) or "未命名章节"
            if title == chapter_title:
                continue
            other_titles.append(title)
        other_titles_str = "\n - ".join(other_titles) if other_titles else "无其他章节。"

        prev_title = self._get_title(chapters[chapter_index - 1]) if chapter_index > 0 else None
        next_title = self._get_title(chapters[chapter_index + 1]) if chapter_index < len(chapters) - 1 else None

        prev_content = "这是报告的第一个主章节。"
        if prev_title:
            prev_content_raw = self.repository.get_previous_chapter(prev_title)
            if not prev_content_raw:
                prev_content_raw = f"前一主章节“{prev_title}”的内容尚未记录。"
            prev_content = self._truncate(prev_content_raw, 6000, "tail")

        next_desc = "这是报告的最后一个主章节。"
        if next_title:
            next_obj = chapters[chapter_index + 1]
            next_description = self._get_description(next_obj) or f"下一主章节“{next_title}”的描述未定义。"
            next_desc = f"下一主章节《{next_title}》计划阐述：{next_description}"

        return f"""
[报告的完整大纲]
{self._outline_to_json()}

[风格与声音指南]
{style_guide}
{rag_context}
[其他章节标题列表 (供结构参考)]
 - {other_titles_str}

[【章节 N-1】上一主章节《{prev_title if prev_title else "N/A"}》的完整内容回顾]
--- 前一章节内容开始 ---
{prev_content}
--- 前一章节内容结束 ---

[【章节 N】当前主章节《{chapter_title}》的核心目标与描述]
{chapter_description or "无详细描述。"}
重要提示: 你将一次性完成本章节的全部内容。

[【章节 N+1】下一主章节《{next_title if next_title else "N/A"}》的核心目标]
{next_desc}
"""

    def build_subsection_context(self, chapter_title: str, subsection_index: int) -> str:
        chapters: list[OutlineNode] = self._extract_chapters()
        chapter_obj: OutlineNode | None = None
        chapter_index = -1
        for idx, chapter in enumerate(chapters):
            if self._get_title(chapter) == chapter_title:
                chapter_obj = chapter
                chapter_index = idx
                break

        if chapter_obj is None or chapter_index == -1:
            return "[错误：无法定位当前主章节信息]"

        subsections: list[OutlineNode] = self._get_sections(chapter_obj)
        current_description = self._get_description(chapter_obj) or "无详细描述。"
        rag_context = ""
        if subsection_index < len(subsections):
            subsection = subsections[subsection_index]
            subsection_title = self._get_title(subsection) or ""
            subsection_desc = self._get_description(subsection) or ""
            query_text = f"{chapter_title}: {subsection_title} - {subsection_desc}"
            rag_context = self._rag_context(query_text)

        prev_chapter_title = self._get_title(chapters[chapter_index - 1]) if chapter_index > 0 else None
        prev_chapter_text = "这是报告的第一个主章节。"
        if prev_chapter_title:
            prev_chapter_raw = self.repository.get_previous_chapter(prev_chapter_title)
            if not prev_chapter_raw:
                prev_chapter_raw = f"前一主章节“{prev_chapter_title}”的内容尚未记录。"
            prev_chapter_text = self._truncate(prev_chapter_raw, 4000, "tail")

        accumulated_subsections: list[str] = []
        if subsection_index > 0:
            for sub_idx in range(subsection_index):
                if sub_idx < len(subsections):
                    sub_obj = subsections[sub_idx]
                    title = self._get_title(sub_obj)
                    if not title:
                        continue
                    subsection_map = self.repository.subsection_content.get(chapter_title)
                    if subsection_map:
                        content = subsection_map.get(title)
                        if content:
                            accumulated_subsections.append(f"--- 内容来自：{title} ---\\n{content}")
        chapter_progress_raw = "\n\n".join(accumulated_subsections) if accumulated_subsections else "这是本章的第一个子章节，之前没有内容。"
        chapter_progress = self._truncate(chapter_progress_raw, 4000, "tail")

        next_context = "这是报告的最后一个部分。"
        if subsection_index < len(subsections) - 1:
            next_sub = subsections[subsection_index + 1]
            next_sub_title = self._get_title(next_sub) or "未命名子章节"
            next_sub_desc = self._get_description(next_sub) or "无详细描述。"
            next_context = f"下一个子章节《{next_sub_title}》计划阐述：{next_sub_desc}"
        elif chapter_index < len(chapters) - 1:
            next_main = chapters[chapter_index + 1]
            next_main_title = self._get_title(next_main) or "未命名章节"
            next_main_desc = self._get_description(next_main) or "无描述"
            next_context = f"完成本章节后，下一个主章节是《{next_main_title}》，其核心目标是：{next_main_desc}"

        return f"""
[报告的完整大纲]
{self._outline_to_json()}

[风格与声音指南]
{self.style_guide or "无特定风格指南。"}
{rag_context}
[上一主章节《{prev_chapter_title if prev_chapter_title else "N/A"}》的核心内容回顾]
{prev_chapter_text}

[当前主章节《{chapter_title}》的核心目标]
{current_description}

[当前主章节《{chapter_title}》已生成的小节内容（你正在续写）]
{chapter_progress}

[为后续内容的铺垫信息]
{next_context}
"""

    def build_critique_context(self, chapter_title: str, full_document_text: str, section_number_map: dict[int, str] | None = None) -> str:
        chapters: list[OutlineNode] = self._extract_chapters()
        try:
            chapter_index = next(i for i, ch in enumerate(chapters) if self._get_title(ch) == chapter_title)
        except StopIteration:
            return "[错误：无法定位被评审章节信息]"

        prev_obj: OutlineNode | None = chapters[chapter_index - 1] if chapter_index > 0 else None
        next_obj: OutlineNode | None = chapters[chapter_index + 1] if chapter_index < len(chapters) - 1 else None

        rag_context = self._rag_context(f"Reviewing section: {chapter_title}", n_results=5)

        prev_title = self._get_title(prev_obj)
        prev_title_display = prev_title or "N/A"
        prev_content_raw: str | None
        if prev_obj and prev_title:
            prev_content_raw = self.repository.get_previous_chapter(prev_title)
            if not prev_content_raw:
                prev_content_raw = f"章节《{prev_title}》的内容尚未记录。"
        else:
            prev_content_raw = "这是报告的第一个主章节。"
        prev_content_raw_cleaned = re.sub(r"<!--\s*section_id:\s*[A-Za-z0-9-]+\s*-->", "", prev_content_raw or "")
        prev_content = self._truncate(prev_content_raw_cleaned, 6000, "middle")

        chapter_text = self._extract_chapter_text(chapter_title, full_document_text)
        # 🔧 移除 section_id 注释，避免 AI 看到 UUID
        chapter_text = re.sub(r"<!--\s*section_id:\s*[A-Za-z0-9-]+\s*-->", "", chapter_text)

        next_title = self._get_title(next_obj)
        if next_obj and next_title:
            next_summary = self.repository.get_summary(next_title)
            if not next_summary:
                next_summary = f"章节《{next_title}》的摘要尚未生成。"
        else:
            next_summary = "这是报告的最后一个主章节。"

        raw_packet = f"""
[报告的完整大纲]
{self._outline_to_json(section_number_map)}

[风格与声音指南]
{self.style_guide or "无特定风格指南。"}
{rag_context}
[【章节 N-1】《{prev_title_display}》的全文回顾]
--- 内容开始 ---
{prev_content}
--- 内容结束 ---

[【章节 N】《{chapter_title}》的当前全文 (此为重点评审/修改对象)]
--- 内容开始 ---
{chapter_text}
--- 内容结束 ---

[【章节 N+1】《{next_title or "N/A"}》的核心摘要]
--- 内容开始 ---
{next_summary}
--- 内容结束 ---
"""
        limit = self.config.generation.max_context_for_long_text_review_tokens
        return truncate_text_for_context_boundary_aware(
            self.config,
            raw_packet,
            int(limit * self.config.generation.prompt_budget_ratio),
            "middle",
        )

    def _extract_chapters(self) -> list[OutlineNode]:
        raw_outline = self.outline.get("outline")
        if not isinstance(raw_outline, list):
            return []
        mapping_items: list[Mapping[Any, Any]] = _filter_mappings(cast(list[Any], raw_outline))
        return _normalized_nodes(mapping_items)

    @staticmethod
    def _get_sections(chapter: OutlineNode) -> list[OutlineNode]:
        sections = chapter.get("sections", [])
        if not isinstance(sections, list):
            return []
        mapping_items: list[Mapping[Any, Any]] = _filter_mappings(cast(list[Any], sections))
        return _normalized_nodes(mapping_items)

    @staticmethod
    def _get_title(node: OutlineNode | None) -> str | None:
        if not node:
            return None
        title = node.get("title")
        return title if isinstance(title, str) else None

    @staticmethod
    def _get_description(node: OutlineNode | None) -> str | None:
        if not node:
            return None
        description = node.get("description")
        return description if isinstance(description, str) else None

    def _outline_to_json(self, section_number_map: dict[int, str] | None = None) -> str:
        import json  # local import to avoid top-level dependency if unused

        def _replace_uuids_with_numbers(obj, uuid_to_number: dict[str, int]):
            """递归将所有UUID替换为对应的数字编号"""
            if isinstance(obj, dict):
                result = {}
                for k, v in obj.items():
                    if k == "id" and isinstance(v, str) and v in uuid_to_number:
                        # 将 UUID 替换为数字编号
                        result[k] = uuid_to_number[v]
                    else:
                        result[k] = _replace_uuids_with_numbers(v, uuid_to_number)
                return result
            elif isinstance(obj, list):
                return [_replace_uuids_with_numbers(item, uuid_to_number) for item in obj]
            return obj

        try:
            # 如果提供了数字映射表，创建反向映射（UUID → 数字）
            if section_number_map:
                uuid_to_number = {uuid_val: num for num, uuid_val in section_number_map.items()}
                clean_outline = _replace_uuids_with_numbers(self.outline, uuid_to_number)
            else:
                clean_outline = self.outline

            return json.dumps(clean_outline, ensure_ascii=False, indent=2)
        except TypeError:
            return str(self.outline)

    @staticmethod
    def _extract_chapter_text(chapter_title: str, full_document_text: str) -> str:
        import re

        escaped_title = re.escape(chapter_title)
        pattern = re.compile(rf"^(##\s*{escaped_title}.*?)(?=^## |\Z)", re.MULTILINE | re.DOTALL)
        match = pattern.search(full_document_text or "")
        if match:
            return match.group(1).strip()
        # 简化日志：降级到 DEBUG
        logging.debug("ContextAssembler: 无法从文档中提取章节 '%s' 的内容。", chapter_title)
        return f"未能从文档中提取章节《{chapter_title}》的完整内容。"
