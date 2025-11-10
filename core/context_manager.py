# ruff: noqa: E501
from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any, cast

from config import Config
from services.llm_interaction import call_ai
from services.vector_db import EmbeddingModel
from utils.text_processor import truncate_text_for_context_boundary_aware

from .context_components import ContextAssembler, ContextRepository, RAGService
from .message_types import ChatMessage


class ContextManager:
    """轻量级门面类，连接上下文仓库、组装器和RAG服务。"""

    def __init__(
        self,
        config: Config,
        style_guide: str,
        outline: Mapping[str, Any],
        external_data: str = "",
        embedding_model_instance: EmbeddingModel | None = None,
        repository: ContextRepository | None = None,
        rag_service: RAGService | None = None,
        assembler: ContextAssembler | None = None,
    ):
        self.config = config
        self.style_guide = style_guide
        self.outline: dict[str, Any] = dict(outline)

        self.repository: ContextRepository = repository or ContextRepository()
        self.rag_service: RAGService | None = None
        if self.config.workflow.disable_rag_for_patch:
            logging.info("ContextManager: 已禁用补丁阶段 RAG（DISABLE_RAG_FOR_PATCH=true）。")
        else:
            self.rag_service = rag_service or RAGService(
                config=config,
                embedding_model=embedding_model_instance,
            )
            if self.rag_service:
                self.rag_service.ensure_index(external_data)

        self.assembler: ContextAssembler = assembler or ContextAssembler(
            config=config,
            outline=self.outline,
            style_guide=style_guide,
            repository=self.repository,
            rag_service=self.rag_service,
        )

    # --- 持久化辅助方法 -------------------------------------------------
    def export_components(self) -> tuple[ContextRepository, RAGService | None, ContextAssembler]:
        """导出底层组件以便在工作流状态中持久化。"""
        return self.repository, self.rag_service, self.assembler

    def update_completed_chapter_content(self, chapter_title: str, full_chapter_content: str) -> None:
        logging.info('ContextManager: 存储章节 "%s" 的完整内容并生成摘要。', chapter_title)
        self.repository.record_chapter(chapter_title, full_chapter_content)

        ratio = self.config.generation.prompt_budget_ratio
        trimmed = truncate_text_for_context_boundary_aware(
            self.config,
            full_chapter_content,
            int(6000 * ratio),
        )

        summary_prompt = f'请为以下标题为 "{chapter_title}" 的章节提供一个简洁的摘要（约200-300字）。重点关注本章内的关键论点、发现和结论。此摘要将用作撰写后续章节的上下文，因此需要信息丰富且简短。\n\n--- 章节内容开始 ---\n{trimmed}\n--- 章节内容结束 ---'

        messages: list[ChatMessage] = [{"role": "user", "content": summary_prompt}]
        try:
            summary = call_ai(
                self.config,
                self.config.models.summary_model_name,
                cast(list[dict[str, str]], messages),
                max_tokens_output=512,
            )
        except Exception as exc:  # pragma: no cover - defensive logging
            logging.error('ContextManager: 生成章节 "%s" 摘要时发生异常: %s', chapter_title, exc, exc_info=True)
            summary = "AI模型调用失败"

        if "AI模型调用失败" in summary or not summary.strip():
            logging.warning('ContextManager: 章节 "%s" 摘要生成失败。AI 响应: %s', chapter_title, summary)
            self.repository.set_summary(chapter_title, "本章节摘要生成失败。")
        else:
            self.repository.set_summary(chapter_title, summary.strip())
            logging.info('ContextManager: 章节 "%s" 的摘要已创建并存储。', chapter_title)

    # --- 上下文组装API ----------------------------------------------
    def get_context_for_standalone_chapter(self, chapter_title: str) -> str:
        logging.info('ContextManager: 生成独立章节 "%s" 的上下文包。', chapter_title)
        return self.assembler.build_chapter_context(chapter_title)

    def get_context_for_chapter_critique(self, chapter_title: str, full_document_text: str, section_number_map: dict[int, str] | None = None) -> str:
        logging.info('ContextManager: 生成章节 "%s" 的精确评审上下文。', chapter_title)
        return self.assembler.build_critique_context(chapter_title, full_document_text, section_number_map)
