from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from collections.abc import Sequence
from typing import Any

from config import Config
from services.llm_interaction import call_ai
from services.web_research.cache import ResponseCache
from services.web_research.citation import build_anchors
from services.web_research.dedupe import dedupe_passages
from services.web_research.fetch_strategy import FETCHER_MANAGER
from services.web_research.instrumentation import log_event, track_stage
from services.web_research.models import ResearchResult, ResearchSource
from services.web_research.parser import parse_html

logger = logging.getLogger(__name__)

try:  # pragma: no cover - optional dependency
    import nest_asyncio
except ImportError:
    logger.info("nest_asyncio not installed; continuing without event-loop patch.")
else:
    try:
        nest_asyncio.apply()
    except Exception as exc:
        if isinstance(exc, (SystemExit, KeyboardInterrupt)):
            raise
        logger.warning("nest_asyncio.apply() failed; continuing without patch: %s", exc, exc_info=True)

_LLM_FAILURE_SENTINEL = "AI模型调用失败"


class ResearchPipeline:
    def __init__(self, config: Config):
        self.config = config
        self.cache = ResponseCache(
            ttl_seconds=getattr(config, "research_cache_ttl", 900),
            redis_url=getattr(config, "research_cache_redis_url", None),
        )
        proxies = {}
        http_proxy = os.environ.get("HTTP_PROXY")
        https_proxy = os.environ.get("HTTPS_PROXY")
        if http_proxy:
            proxies["http"] = http_proxy
        if https_proxy:
            proxies["https"] = https_proxy
        self.fetcher = FETCHER_MANAGER.ensure(
            user_agent=config.api.user_agent if getattr(config, "api", None) else None,
            proxies=proxies or None,
        )

    async def run_async(
        self,
        knowledge_gaps: Sequence[str],
        full_document_context: str,
        *,
        topic: str | None,
    ) -> ResearchResult:
        summaries: list[str] = []
        sources: list[ResearchSource] = []

        for gap in knowledge_gaps:
            result = await self._process_gap_async(gap, full_document_context, topic=topic)
            if result:
                sources.extend(result.sources)
                summaries.append(result.body)

        if not sources:
            return ResearchResult(body="No additional research available.")

        anchors = build_anchors(sources)
        combined_body = "\n\n".join(summaries)
        return ResearchResult(body=combined_body, sources=sources, anchors=anchors)

    async def _call_ai_text(
        self,
        *,
        model_name: str,
        messages: list[dict[str, str]],
        context: str | None = None,
        **kwargs: Any,
    ) -> str | None:
        """Run call_ai in a background thread and normalize failure responses."""

        label = context or model_name

        def _invoke() -> str | None:
            try:
                result = call_ai(self.config, model_name, messages, **kwargs)
            except Exception as exc:  # pragma: no cover - defensive
                if isinstance(exc, (SystemExit, KeyboardInterrupt)):
                    raise
                logger.warning("LLM call failed for %s: %s", label, exc, exc_info=True)
                return None
            if result is None:
                return None
            if not isinstance(result, str):
                try:
                    result = str(result)
                except Exception as stringify_error:
                    if isinstance(stringify_error, (SystemExit, KeyboardInterrupt)):
                        raise
                    logger.debug("Unable to stringify LLM result for %s", label, exc_info=True)
                    return None
            if _LLM_FAILURE_SENTINEL in result:
                logger.warning("LLM call returned failure sentinel for %s", label)
                return None
            return result

        return await asyncio.to_thread(_invoke)

    async def _process_gap_async(
        self,
        knowledge_gap: str,
        full_document_context: str,
        *,
        topic: str | None,
    ) -> ResearchResult | None:
        if not knowledge_gap.strip():
            return None

        queries = await self._build_queries_async(knowledge_gap, full_document_context, topic=topic)
        serp_items = []
        for query in queries:
            serp_items.extend(self._perform_search(query))

        if not serp_items:
            return None

        passages: list[tuple[str, str]] = []
        sources: list[ResearchSource] = []

        for item in serp_items:
            url = item.get("link")
            title = item.get("title")
            if not url:
                continue
            fetch_cache_hit = self.cache.get("fetch", url)
            if fetch_cache_hit:
                status = fetch_cache_hit.get("status", 200)
                if status < 200 or status >= 400:
                    log_event(
                        task_id=self.config.task_id,
                        topic=topic,
                        node="web_research",
                        stage_key="fetch",
                        message=f"Cached response not usable (status {status}) for {url}",
                        level=logging.WARNING,
                    )
                    continue
                html_text = fetch_cache_hit["content"]
                latency = fetch_cache_hit.get("latency_ms", 0.0)
                retries = fetch_cache_hit.get("retries", 0)
            else:
                try:
                    with track_stage(
                        task_id=self.config.task_id,
                        topic=topic,
                        node="web_research",
                        stage_key="fetch",
                        message=f"Fetching {url}",
                    ):
                        response = await self.fetcher.fetch(url)
                except Exception as exc:
                    if isinstance(exc, (SystemExit, KeyboardInterrupt)):
                        raise
                    log_event(
                        task_id=self.config.task_id,
                        topic=topic,
                        node="web_research",
                        stage_key="fetch",
                        message=f"Fetch failed for {url}: {exc}",
                        level=logging.DEBUG,  # 降低日志级别，避免污染控制台
                    )
                    continue
                if response.status < 200 or response.status >= 400:
                    log_event(
                        task_id=self.config.task_id,
                        topic=topic,
                        node="web_research",
                        stage_key="fetch",
                        message=f"Ignoring non-success status {response.status} for {url}",
                        level=logging.WARNING,
                    )
                    continue
                html_text = response.content
                latency = response.elapsed_ms
                retries = response.retries
                self.cache.set(
                    {"content": html_text, "latency_ms": latency, "retries": retries, "status": response.status},
                    "fetch",
                    url,
                )

            parsed = parse_html(html_text)
            text = parsed.get("text") or ""
            if not text.strip():
                continue
            passages.append((url, text))

            summary = await self._summarize_async(
                knowledge_gap=knowledge_gap,
                url=url,
                text=text,
                topic=topic,
                cached_title=title or parsed.get("title"),
            )
            if not summary:
                continue

            source = ResearchSource(
                url=url,
                title=title or parsed.get("title"),
                summary=summary,
                raw_content=text[:4000],
                score=item.get("score", 0.5),
                latency_ms=latency,
                retries=retries,
            )
            sources.append(source)

        if not sources:
            return None

        deduped_passages = dedupe_passages(passages)
        deduped_urls = {url for url, _ in deduped_passages}
        sources = [src for src in sources if src.url in deduped_urls][: get_top_sources(self.config)]
        anchors = build_anchors(sources)
        body = "\n\n".join(f"[{idx + 1}] {src.summary}" for idx, src in enumerate(sources))
        return ResearchResult(body=body, sources=sources, anchors=anchors)

    async def _build_queries_async(
        self,
        knowledge_gap: str,
        full_document_context: str,
        *,
        topic: str | None,
    ) -> list[str]:
        cached = self.cache.get("queries", knowledge_gap)
        if cached:
            return cached
        max_queries = max(1, getattr(self.config, "max_queries_per_gap", 3))
        multilingual_enabled = bool(getattr(self.config, "enable_multilingual_search", False))
        context_snippet = full_document_context[:2000]
        if multilingual_enabled:
            prompt = (
                f"知识缺口: {knowledge_gap}\n"
                f"现有文稿片段: {context_snippet}\n"
                f"请生成不超过 {max_queries} 条高质量的搜索引擎查询，覆盖专业术语与最新进展。"
                "必须同时提供中文和英文（若适用可包含其他语言）检索词，保留关键术语原文。"
                "每行使用格式如 '[zh] 自由落体 稳定性' 或 'en: rigid body stability'，无需额外解释。"
            )
        else:
            prompt = (
                f"知识缺口: {knowledge_gap}\n"
                f"现有文稿片段: {context_snippet}\n"
                f"请生成最多 {max_queries} 条适用于搜索引擎的关键词查询，覆盖专业术语与最新进展。"
            )
        message = [{"role": "user", "content": prompt}]
        response = await self._call_ai_text(
            model_name=self.config.models.researcher_model_name,
            messages=message,
            temperature=0.2,
            max_tokens_output=200,
            context=f"query generation for gap '{knowledge_gap[:60]}'",
        )
        queries: list[str] = []
        if response:
            raw_lines = [line.strip() for line in response.splitlines() if line.strip()]
            seen: set[str] = set()
            for line in raw_lines:
                normalized = line.lstrip("-• ").strip()
                normalized = re.sub(r"^\d+[\)\.、]\s*", "", normalized)
                normalized = re.sub(r"^\s*(\[[^\]]+\]|[A-Za-z]{2,8}:)\s*", "", normalized)
                if not normalized:
                    continue
                lowered = normalized.lower()
                if lowered in seen:
                    continue
                seen.add(lowered)
                queries.append(normalized)
        if not queries:
            fallback = knowledge_gap.strip() or knowledge_gap
            queries = [fallback]
        queries = queries[:max_queries]
        self.cache.set(queries, "queries", knowledge_gap)
        log_event(
            task_id=self.config.task_id,
            topic=topic,
            node="web_research",
            stage_key="query",
            message=f"Generated {len(queries)} queries for gap (multilingual={multilingual_enabled}).",
        )
        return queries

    def _perform_search(self, query: str) -> list[dict[str, Any]]:
        from services.web_research.pipeline.search import perform_search

        items = perform_search(self.config, query)
        log_event(
            task_id=self.config.task_id,
            topic=None,
            node="web_research",
            stage_key="search",
            message=f"Search '{query}' yielded {len(items)} results.",
        )
        return items

    async def _summarize_async(
        self,
        *,
        knowledge_gap: str,
        url: str,
        text: str,
        topic: str | None,
        cached_title: str | None,
    ) -> str | None:
        cache_key = f"{url}:{knowledge_gap}"
        cached = self.cache.get("summary", cache_key)
        if cached:
            return cached

        prompt = f"背景缺口：{knowledge_gap}\n来源：{url}\n标题：{cached_title or '未知'}\n请提取与缺口直接相关的要点，回答 2-3 句话，并包含可引用的数据。"
        truncated = text[: min(6000, len(text))]
        messages = [
            {"role": "system", "content": "你是专业研究助理，擅长抽取网页要点。"},
            {"role": "user", "content": prompt + "\n\n正文:\n" + truncated},
        ]
        summary = await self._call_ai_text(
            model_name=self.config.models.summary_model_name,
            messages=messages,
            max_tokens_output=512,
            temperature=self.config.temperature_factual,
            context=f"summarizing {url}",
        )
        if summary:
            summary = summary.strip()
        if summary:
            self.cache.set(summary, "summary", cache_key)
            log_event(
                task_id=self.config.task_id,
                topic=topic,
                node="web_research",
                stage_key="summarize",
                message=f"Summarized {url}",
            )
        return summary or None


def get_top_sources(config: Config) -> int:
    return getattr(config, "structured_research_max_briefs", 6)


def _to_result_dict(result: ResearchResult) -> dict[str, Any]:
    return {
        "briefs": [
            {
                "url": src.url,
                "title": src.title,
                "summary": src.summary,
                "score": src.score,
            }
            for src in result.sources
        ],
        "body": result.body,
        "anchors": [anchor.__dict__ for anchor in result.anchors],
        "statistics": {
            "total_sources": len(result.sources),
            "cached_hits": 0,
        },
    }


def run_research_cycle(config: Config, knowledge_gaps: list[str], full_document_context: str) -> ResearchResult:
    pipeline = ResearchPipeline(config)
    return _run_sync(pipeline.run_async(knowledge_gaps, full_document_context, topic=config.user_problem))


async def run_research_cycle_async(
    config: Config,
    knowledge_gaps: list[str],
    full_document_context: str,
) -> ResearchResult:
    pipeline = ResearchPipeline(config)
    return await pipeline.run_async(knowledge_gaps, full_document_context, topic=config.user_problem)


def run_structured_research_cycle(config: Config, knowledge_gaps: list[str], full_document_context: str) -> dict[str, Any]:
    result = run_research_cycle(config, knowledge_gaps, full_document_context)
    return _to_result_dict(result)


async def run_structured_research_cycle_async(config: Config, knowledge_gaps: list[str], full_document_context: str) -> dict[str, Any]:
    result = await run_research_cycle_async(config, knowledge_gaps, full_document_context)
    return _to_result_dict(result)


def enhanced_research_cycle_with_hyde(config: Config, knowledge_gaps: list[str], full_document_context: str) -> ResearchResult:
    return run_research_cycle(config, knowledge_gaps, full_document_context)


async def enhanced_research_cycle_with_hyde_async(config: Config, knowledge_gaps: list[str], full_document_context: str) -> ResearchResult:
    return await run_research_cycle_async(config, knowledge_gaps, full_document_context)


def create_hyde_search_queries(config: Config, gap: str, context: str) -> list[str]:
    pipeline = ResearchPipeline(config)
    return _run_sync(pipeline._build_queries_async(gap, context, topic=config.user_problem))


def _run_sync(coro):
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)


def enable_diagnostic_logging(**_: Any) -> None:
    logger.info("Diagnostic logging enabled for web research pipeline.")


def get_diagnostic_statistics() -> dict[str, Any]:
    return FETCHER_MANAGER.stats()


def get_diagnostic_summary() -> str:
    stats = FETCHER_MANAGER.stats()
    return json.dumps(stats)


def get_fetcher_statistics(config: Config) -> dict[str, Any]:
    return FETCHER_MANAGER.stats()


def ensure_fetcher_initialized(config: Config) -> None:
    FETCHER_MANAGER.ensure(
        user_agent=config.api.user_agent if getattr(config, "api", None) else None,
        proxies=None,
    )


async def cleanup_fetcher() -> None:
    """异步清理 Fetcher 资源，确保 httpx 客户端正确关闭"""
    await FETCHER_MANAGER.close()
