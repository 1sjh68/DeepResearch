from __future__ import annotations

import asyncio
import json
import logging
import re
from collections.abc import Awaitable, Callable
from datetime import datetime
from typing import TYPE_CHECKING, Any, cast

import services.web_research as web_research
from config import Config
from core.progress import StepOutput, step_result, workflow_step
from core.state_manager import WorkflowStateAdapter
from services.vector_db import EmbeddingModel, VectorDBManager
from utils.citation import CitationManager, CitationMatch, ClaimInfo, SourceInfo
from utils.progress_tracker import safe_pulse
from workflows.graph_state import GraphState

if TYPE_CHECKING:
    from services.web_research import ResearchResult

# Required web research functions
run_research_cycle = web_research.run_research_cycle
run_research_cycle_async = web_research.run_research_cycle_async
enhanced_research_cycle_with_hyde = getattr(web_research, "enhanced_research_cycle_with_hyde", run_research_cycle)
create_hyde_search_queries = getattr(web_research, "create_hyde_search_queries", lambda _config, _gap, _context: [])
ensure_fetcher_initialized = getattr(web_research, "ensure_fetcher_initialized")
cleanup_fetcher = getattr(web_research, "cleanup_fetcher")

# Optional advanced integrations
_STRUCTURED_SYNC_FN: Callable[[Config, list[str], str], dict[str, Any]] | None = getattr(web_research, "run_structured_research_cycle", None)
_STRUCTURED_ASYNC_FN: Callable[[Config, list[str], str], Awaitable[dict[str, Any]]] | None = getattr(web_research, "run_structured_research_cycle_async", None)
_ENHANCED_ASYNC_FN: Callable[[Config, list[str], str], Awaitable[ResearchResult | str | None]] | None = getattr(web_research, "enhanced_research_cycle_with_hyde_async", None)
_FETCHER_STATS_FN: Callable[[Config], dict[str, Any]] | None = getattr(web_research, "get_fetcher_statistics", None)
_ENABLE_DIAGNOSTIC_FN: Callable[..., None] | None = getattr(web_research, "enable_diagnostic_logging", None)
_DIAGNOSTIC_STATS_FN: Callable[[], dict[str, Any]] | None = getattr(web_research, "get_diagnostic_statistics", None)
_DIAGNOSTIC_SUMMARY_FN: Callable[[], str] | None = getattr(web_research, "get_diagnostic_summary", None)

# 使用统一的缓存管理器
from utils.cache_manager import get_cache  # noqa: E402

# 获取研究缓存实例
_research_cache = get_cache(
    namespace="research",
    ttl=3600,  # 默认1小时，会被config覆盖
    max_size=32,
    backend="memory",
)


def _normalized_gaps(knowledge_gaps: list[str] | None) -> tuple[str, ...]:
    if not knowledge_gaps:
        return tuple()
    return tuple(sorted({gap.strip(): None for gap in knowledge_gaps if isinstance(gap, str) and gap.strip()}.keys()))


def _research_cache_key(config: Config, knowledge_gaps: list[str], research_mode: str) -> str:
    """生成研究缓存的键"""
    task_identifier = getattr(config, "task_id", None) or getattr(config, "user_problem", "")
    researcher_model = getattr(config, "researcher_model_name", "")
    normalized_gaps = _normalized_gaps(knowledge_gaps)
    # 将元组转换为字符串键
    return json.dumps({"task": str(task_identifier), "gaps": list(normalized_gaps), "mode": research_mode, "model": str(researcher_model)}, sort_keys=True)


def _read_research_cache(config: Config, knowledge_gaps: list[str], research_mode: str) -> tuple[dict[str, Any], str | None] | None:
    """从缓存读取研究结果"""
    if not getattr(config, "enable_research_cache", False):
        return None

    # 更新缓存TTL（如果配置了）
    ttl = getattr(config, "research_cache_ttl_seconds", 0) or 0
    if ttl and _research_cache.ttl != ttl:
        _research_cache.ttl = ttl

    key = _research_cache_key(config, knowledge_gaps, research_mode)
    cached = _research_cache.get(key)

    if cached:
        data = cached.get("data")
        detail = cached.get("detail")
        return data, detail

    return None


def _write_research_cache(config: Config, knowledge_gaps: list[str], research_mode: str, data: dict[str, Any], detail: str | None) -> None:
    """将研究结果写入缓存"""
    if not getattr(config, "enable_research_cache", False):
        return
    if not knowledge_gaps:
        return

    key = _research_cache_key(config, knowledge_gaps, research_mode)
    payload = {"data": data, "detail": detail}
    _research_cache.set(key, payload)


def _apply_cached_research(workflow_state: Any, cached_data: dict[str, Any]) -> None:
    workflow_state.research_brief = cached_data.get("research_brief") or ""
    workflow_state.structured_research_data = cached_data.get("structured_research_data") or None
    workflow_state.citation_data = cached_data.get("citation_data") or None


def _maybe_cache_research_result(config: Config, knowledge_gaps: list[str], research_mode: str, data: dict[str, Any], detail: str | None) -> None:
    try:
        _write_research_cache(config, knowledge_gaps, research_mode, data, detail)
    except Exception as exc:  # pragma: no cover - defensive logging
        logging.debug("research_node: 缓存研究结果失败，跳过。详情: %s", exc)


def _default_statistics(total_queries: int, successes: int) -> dict[str, Any]:
    success_rate = successes / total_queries if total_queries else 0.0
    return {
        "total_queries": total_queries,
        "successful_extractions": successes,
        "success_rate": success_rate,
        "average_confidence": 0.0,
        "high_quality_rate": 0.0,
    }


def _build_structured_results(brief_text: str | None, knowledge_gaps: list[str], result: Any | None = None) -> dict[str, Any]:
    cleaned = (brief_text or "").strip()
    successes = 1 if cleaned else 0
    statistics = _default_statistics(len(knowledge_gaps), successes)
    if result and hasattr(result, "sources"):
        try:
            from services.web_research.models import ResearchResult as _RR  # type: ignore

            if isinstance(result, _RR):
                return {
                    "briefs": [
                        {
                            "url": src.url,
                            "specific_query": "",
                            "summary": src.summary,
                            "key_points": [],
                            "confidence": src.score,
                            "source_quality": "unknown",
                        }
                        for src in result.sources
                    ],
                    "statistics": {
                        "total_queries": len(knowledge_gaps),
                        "successful_extractions": len(result.sources),
                        "success_rate": len(result.sources) / max(1, len(knowledge_gaps)),
                    },
                }
        except (AttributeError, TypeError, ValueError) as exc:
            logging.warning(
                "Failed to adapt structured research result; falling back to summary-only payload: %s",
                exc,
                exc_info=True,
            )
        except Exception as exc:  # pragma: no cover - defensive
            if isinstance(exc, (SystemExit, KeyboardInterrupt)):
                raise
            logging.warning(
                "Unexpected error building structured research results; continuing with fallback: %s",
                exc,
                exc_info=True,
            )
    if not cleaned:
        return {"briefs": [], "statistics": statistics}
    return {
        "briefs": [
            {
                "url": "",
                "specific_query": "",
                "summary": cleaned,
                "key_points": [],
                "confidence": 0.0,
                "source_quality": "unknown",
            }
        ],
        "statistics": statistics,
    }


def _invoke_structured_sync(config: Config, knowledge_gaps: list[str], draft_content: str) -> dict[str, Any]:
    if _STRUCTURED_SYNC_FN:
        try:
            return _STRUCTURED_SYNC_FN(config, knowledge_gaps, draft_content)
        except Exception as exc:
            if isinstance(exc, (SystemExit, KeyboardInterrupt)):
                raise
            logging.debug("结构化研究失败，回退到传统模式: %s", exc)  # 降低日志级别
    result = run_research_cycle(config, knowledge_gaps, draft_content)
    brief = result.to_brief()
    return _build_structured_results(brief, knowledge_gaps, result)


async def _invoke_structured_async(config: Config, knowledge_gaps: list[str], draft_content: str) -> dict[str, Any]:
    if _STRUCTURED_ASYNC_FN:
        try:
            return await _STRUCTURED_ASYNC_FN(config, knowledge_gaps, draft_content)
        except Exception as exc:
            if isinstance(exc, (SystemExit, KeyboardInterrupt)):
                raise
            logging.debug("异步结构化研究失败，回退到传统模式: %s", exc)  # 降低日志级别
    result = await run_research_cycle_async(config, knowledge_gaps, draft_content)
    brief = result.to_brief()
    return _build_structured_results(brief, knowledge_gaps, result)


async def _invoke_enhanced_async(config: Config, knowledge_gaps: list[str], draft_content: str) -> str:
    if _ENHANCED_ASYNC_FN:
        try:
            result = await _ENHANCED_ASYNC_FN(config, knowledge_gaps, draft_content)
            if result is None:
                return ""
            if isinstance(result, str):
                return result
            research_result = cast("ResearchResult", result)
            return research_result.to_brief()
        except Exception as exc:
            if isinstance(exc, (SystemExit, KeyboardInterrupt)):
                raise
            logging.warning("增强异步研究失败，回退到传统模式: %s", exc)
    fallback = await run_research_cycle_async(config, knowledge_gaps, draft_content)
    return fallback.to_brief()


def _get_fetcher_stats(config: Config) -> dict[str, Any]:
    if _FETCHER_STATS_FN:
        try:
            return _FETCHER_STATS_FN(config)
        except Exception as exc:
            logging.debug("获取 SmartFetcher 统计信息失败: %s", exc)
    return {
        "smart_fetcher_enabled": False,
        "supports_async": False,
        "batch_processing": False,
    }


def _get_diagnostic_stats() -> dict[str, Any]:
    if _DIAGNOSTIC_STATS_FN:
        try:
            return _DIAGNOSTIC_STATS_FN()
        except Exception as exc:
            logging.debug("获取诊断统计失败: %s", exc)
    return {
        "diagnostic_system_enabled": False,
        "supported_metrics": [],
        "blocking_patterns_detected": [],
    }


def _diagnostic_summary() -> str:
    if _DIAGNOSTIC_SUMMARY_FN:
        try:
            return _DIAGNOSTIC_SUMMARY_FN()
        except Exception as exc:
            logging.debug("获取诊断摘要失败: %s", exc)
    return "Diagnostic logging not available."


def _maybe_enable_diagnostics(level: str, json_output: bool) -> None:
    if _ENABLE_DIAGNOSTIC_FN:
        try:
            _ENABLE_DIAGNOSTIC_FN(level=level, json_output=json_output)
        except Exception as exc:
            logging.debug("启用诊断日志失败: %s", exc)


@workflow_step("research_node", "执行外部研究")
def research_node(state: GraphState) -> StepOutput:
    """执行研究节点，支持同步和异步模式"""
    workflow_state = WorkflowStateAdapter.ensure(state)
    config = workflow_state.config
    current_iteration = workflow_state.refinement_count + 1
    max_rounds = config.max_refinement_rounds

    safe_pulse(
        config.task_id,
        f"迭代 {current_iteration}/{max_rounds} · 执行外部研究中...",
    )

    if not config.enable_web_research:
        return step_result(
            {
                "research_brief": "",
                "structured_research_data": None,
                "citation_data": None,
            },
            "功能已禁用",
        )

    knowledge_gaps = workflow_state.knowledge_gaps or []

    if not knowledge_gaps:
        return step_result(
            {
                "research_brief": "",
                "structured_research_data": None,
                "citation_data": None,
            },
            "无研究需求",
        )

    research_mode = getattr(config, "research_mode", "intelligent")
    cached_payload = _read_research_cache(config, knowledge_gaps, research_mode)
    if cached_payload:
        cached_data, cached_detail = cached_payload
        _apply_cached_research(workflow_state, cached_data)
        detail_text = cached_detail or f"缓存命中：知识缺口 {len(knowledge_gaps)} 项"
        logging.info("research_node: 使用缓存的研究结果（%s 个知识缺口）。", len(knowledge_gaps))
        safe_pulse(config.task_id, f"迭代 {current_iteration}/{max_rounds} · 研究结果命中缓存")
        return step_result(cached_data, detail_text)

    # 启用诊断日志系统
    if getattr(config, "enable_diagnostic_logging", True):
        _maybe_enable_diagnostics(
            level=getattr(config, "diagnostic_log_level", "INFO"),
            json_output=getattr(config, "enable_json_logs", True),
        )

    # 选择研究模式：智能模式（默认），结构化模式，增强模式或传统模式
    use_async = getattr(config, "use_async_research", True)
    citation_manager = _init_citation_manager(config)

    # 智能模式使用异步优先，但可以回退到同步
    if research_mode == "intelligent" and use_async:
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # 事件循环已在运行（如Jupyter环境），无法使用asyncio.run()
                logging.info("检测到运行中的事件循环，回退到同步研究模式")
            else:
                return asyncio.run(
                    _research_node_async_impl(
                        workflow_state,
                        config,
                        citation_manager,
                        current_iteration,
                        max_rounds,
                    )
                )
        except Exception as exc:
            logging.warning("异步研究模式失败，回退到同步模式: %s", exc)

    # 使用同步模式
    return _research_node_sync_impl(workflow_state, config, citation_manager, current_iteration, max_rounds)


async def _research_node_async_impl(
    workflow_state,
    config,
    citation_manager,
    current_iteration,
    max_rounds,
) -> StepOutput:
    """异步研究节点实现"""
    # 初始化SmartFetcher
    ensure_fetcher_initialized(config)

    knowledge_gaps = workflow_state.knowledge_gaps or []
    draft_content = workflow_state.draft_content or ""
    research_mode = getattr(config, "research_mode", "intelligent")

    structured_results: dict[str, Any] = {"briefs": [], "statistics": {}}
    research_brief: str = ""
    try:
        if research_mode == "intelligent":
            structured_results = await _invoke_structured_async(config, knowledge_gaps, draft_content)

        elif research_mode == "enhanced":
            research_brief = await _invoke_enhanced_async(config, knowledge_gaps, draft_content)
            structured_results = _convert_enhanced_to_structured(research_brief)

        elif research_mode == "structured":
            structured_results = await _invoke_structured_async(config, knowledge_gaps, draft_content)

        else:
            result = await run_research_cycle_async(config, knowledge_gaps, draft_content)
            research_brief = result.to_brief()
            structured_results = _build_structured_results(research_brief, knowledge_gaps, result)

        # 处理引用和主张
        if research_mode not in ["enhanced", "traditional"]:
            # 添加信息源到引用管理器
            _add_structured_sources_to_citation_manager(citation_manager, structured_results, knowledge_gaps)

            # 转换为传统格式以保持向后兼容
            research_brief = _convert_structured_to_brief(structured_results)

            # 提取主张并绑定到数据源
            claims, citation_matches = _process_citations_from_research(
                citation_manager,
                research_brief,
                structured_results,
                knowledge_gaps,
            )

            # 保存结构化数据和引用信息到工作流状态
            workflow_state.structured_research_data = structured_results
            workflow_state.citation_data = {
                "citation_matches": citation_matches,
                "total_claims": len(claims),
                "total_sources": len(citation_manager.sources),
                "citation_statistics": citation_manager.get_citation_statistics(),
                "smart_fetcher_stats": _get_fetcher_stats(config),
                "diagnostic_stats": _get_diagnostic_stats(),
                "enhanced_stats": _get_enhanced_statistics(config, structured_results),
            }

            detail_msg = f"迭代 {current_iteration}/{max_rounds}，完成智能研究：{len(structured_results.get('briefs', []))} 个源，提取 {len(claims)} 个主张"
            result_payload = {
                "research_brief": research_brief,
                "structured_research_data": structured_results or None,
                "citation_data": workflow_state.citation_data,
            }
            _maybe_cache_research_result(config, knowledge_gaps, research_mode, result_payload, detail_msg)
            return step_result(result_payload, detail_msg)
        else:
            # 增强模式或传统模式处理
            if research_mode == "enhanced":
                research_brief = research_brief or ""
            # 提取主张并绑定到数据源
            claims, citation_matches = _process_citations_from_research(
                citation_manager,
                research_brief,
                None,
                knowledge_gaps,
            )

            workflow_state.citation_data = {
                "citation_matches": citation_matches,
                "total_claims": len(claims),
                "total_sources": len(citation_manager.sources),
                "citation_statistics": citation_manager.get_citation_statistics(),
                "smart_fetcher_stats": _get_fetcher_stats(config),
                "diagnostic_stats": _get_diagnostic_stats(),
            }

            detail_msg = f"迭代 {current_iteration}/{max_rounds}，完成传统异步研究并提取 {len(claims)} 个主张"
            result_payload = {
                "research_brief": research_brief,
                "structured_research_data": structured_results or None,
                "citation_data": workflow_state.citation_data,
            }
            _maybe_cache_research_result(config, knowledge_gaps, research_mode, result_payload, detail_msg)
            return step_result(result_payload, detail_msg)

    except Exception as e:
        logging.error(f"异步研究节点执行失败: {e}", exc_info=True)
        return step_result(
            {
                "research_brief": "",
                "structured_research_data": None,
                "citation_data": None,
            },
            f"研究失败: {str(e)}",
        )
    finally:
        # 清理资源
        try:
            await cleanup_fetcher()
        except Exception as e:
            logging.warning(f"清理SmartFetcher资源失败: {e}")


def _research_node_sync_impl(
    workflow_state,
    config,
    citation_manager,
    current_iteration,
    max_rounds,
) -> StepOutput:
    """同步研究节点实现"""
    knowledge_gaps = workflow_state.knowledge_gaps or []
    draft_content = workflow_state.draft_content or ""
    research_mode = getattr(config, "research_mode", "intelligent")

    structured_results: dict[str, Any] = {"briefs": [], "statistics": {}}
    research_brief: str = ""
    try:
        if research_mode == "intelligent":
            structured_results = _invoke_structured_sync(config, knowledge_gaps, draft_content)

        elif research_mode == "enhanced":
            # 增强模式：使用 HyDE + 混合检索
            if hasattr(config, "enable_hybrid_search") and config.enable_hybrid_search:
                research_brief = _enhanced_hybrid_research(config, knowledge_gaps, draft_content)
                structured_results = {"briefs": [], "statistics": {}}
            else:
                enhanced_result = enhanced_research_cycle_with_hyde(config, knowledge_gaps, draft_content)
                research_brief = enhanced_result.to_brief()
                structured_results = _build_structured_results(research_brief, knowledge_gaps, enhanced_result)

        elif research_mode == "structured":
            structured_results = _invoke_structured_sync(config, knowledge_gaps, draft_content)

        else:
            # 传统模式（向后兼容）
            result = run_research_cycle(config, knowledge_gaps, draft_content)
            research_brief = result.to_brief()
            structured_results = _build_structured_results(research_brief, knowledge_gaps, result)

        # 处理引用和主张
        if research_mode not in ["enhanced", "traditional"]:
            # 添加信息源到引用管理器
            _add_structured_sources_to_citation_manager(citation_manager, structured_results, knowledge_gaps)

            # 转换为传统格式以保持向后兼容
            research_brief = _convert_structured_to_brief(structured_results)

            # 提取主张并绑定到数据源
            claims, citation_matches = _process_citations_from_research(
                citation_manager,
                research_brief,
                structured_results,
                knowledge_gaps,
            )

            # 保存结构化数据和引用信息到工作流状态
            workflow_state.structured_research_data = structured_results
            workflow_state.citation_data = {
                "citation_matches": citation_matches,
                "total_claims": len(claims),
                "total_sources": len(citation_manager.sources),
                "citation_statistics": citation_manager.get_citation_statistics(),
                "smart_fetcher_stats": _get_fetcher_stats(config),
                "diagnostic_stats": _get_diagnostic_stats(),
                "enhanced_stats": _get_enhanced_statistics(config, structured_results),
            }

            detail_msg = f"迭代 {current_iteration}/{max_rounds}，完成同步研究：{len(structured_results.get('briefs', []))} 个源，提取 {len(claims)} 个主张"
            result_payload = {
                "research_brief": research_brief,
                "structured_research_data": structured_results,
                "citation_data": workflow_state.citation_data,
            }
            _maybe_cache_research_result(config, knowledge_gaps, research_mode, result_payload, detail_msg)
            return step_result(result_payload, detail_msg)
        else:
            # 增强模式或传统模式处理
            # 提取主张并绑定到数据源
            claims, citation_matches = _process_citations_from_research(
                citation_manager,
                research_brief,
                None,
                knowledge_gaps,
            )

            # 保存引用数据到工作流状态
            workflow_state.citation_data = {
                "citation_matches": citation_matches,
                "total_claims": len(claims),
                "total_sources": len(citation_manager.sources),
                "citation_statistics": citation_manager.get_citation_statistics(),
                "smart_fetcher_stats": _get_fetcher_stats(config),
                "diagnostic_stats": _get_diagnostic_stats(),
            }

            detail_msg = f"迭代 {current_iteration}/{max_rounds}，完成同步研究并提取 {len(claims)} 个主张"
            result_payload = {
                "research_brief": research_brief,
                "structured_research_data": structured_results or None,
                "citation_data": workflow_state.citation_data,
            }
            _maybe_cache_research_result(config, knowledge_gaps, research_mode, result_payload, detail_msg)
            return step_result(result_payload, detail_msg)

    except Exception as e:
        logging.debug(f"研究节点执行失败: {e}")  # 降低日志级别，避免污染控制台
        return step_result(
            {
                "research_brief": "",
                "structured_research_data": None,
                "citation_data": None,
            },
            f"研究失败: {str(e)}",
        )


def _enhanced_hybrid_research(config: Config, knowledge_gaps: list[str], draft_content: str) -> str:
    """
    增强的混合检索研究：结合向量数据库和网络检索
    """
    try:
        # 初始化向量数据库管理器
        embedding_model = EmbeddingModel(config)
        vector_db = VectorDBManager(config, embedding_model)

        all_research_results = []

        for gap_text in knowledge_gaps:
            # 1. 使用 HyDE 生成增强查询
            enhanced_queries = create_hyde_search_queries(config, gap_text, draft_content)
            if enhanced_queries:
                formatted_queries = "\n".join(f"- {query}" for query in enhanced_queries)
                all_research_results.append(f"[HyDE 查询] {gap_text}:\n{formatted_queries}")

            # 2. 向量数据库检索
            vector_results = vector_db.hybrid_retrieve_experience(gap_text, n_results=5)
            if vector_results:
                vector_summary = _summarize_vector_results(vector_results, gap_text)
                all_research_results.append(f"[向量库检索] {gap_text}:\n{vector_summary}")

            # 3. 网络检索（使用增强查询）
            # 注意：这里使用传统的网络检索，因为 enhanced_research_cycle_with_hyde 内部会使用增强查询
            web_results = enhanced_research_cycle_with_hyde(config, [gap_text], draft_content)
            if web_results:
                all_research_results.append(f"[网络检索] {gap_text}:\n{web_results.to_brief()}")

        if all_research_results:
            return "\n\n===== 混合检索研究简报开始 =====\n\n" + "\n".join(all_research_results) + "\n===== 混合检索研究简报结束 =====\n\n"
        else:
            return ""

    except Exception as e:
        # 如果混合检索失败，回退到基础网络检索
        logging.warning(f"混合检索失败，回退到基础检索: {e}")
        fallback_result = enhanced_research_cycle_with_hyde(config, knowledge_gaps, draft_content)
        return fallback_result.to_brief()


def _summarize_vector_results(vector_results: list[dict], knowledge_gap: str) -> str:
    """
    总结向量检索结果
    """
    if not vector_results:
        return "无相关经验记录。"

    summary_parts = []
    for i, result in enumerate(vector_results[:3]):  # 只显示前3个最相关的结果
        doc = result.get("document", "")[:500]  # 限制长度
        score = result.get("hybrid_score", result.get("distance", 0))
        source = result.get("source", "unknown")

        summary_parts.append(f"结果 {i + 1} (相似度: {score:.3f}, 来源: {source}):\n{doc}...")

    return "\n\n".join(summary_parts)


def _convert_structured_to_brief(structured_results: dict[str, Any]) -> str:
    """
    将结构化研究结果转换为传统格式，保持向后兼容
    """
    briefs = structured_results.get("briefs", [])
    statistics = structured_results.get("statistics", {})

    if not briefs:
        return ""

    formatted_briefs = []
    for brief in briefs:
        formatted_brief = f"URL: {brief.get('url', '')}\n"
        formatted_brief += f"查询: {brief.get('specific_query', '')}\n"
        formatted_brief += f"总结: {brief.get('summary', '')}\n"
        formatted_brief += f"要点: {', '.join(brief.get('key_points', []))}\n"
        formatted_brief += f"置信度: {brief.get('confidence', 0):.2f}\n"
        formatted_brief += f"来源质量: {brief.get('source_quality', 'unknown')}\n"

        formatted_briefs.append(formatted_brief)

    # 添加统计信息
    stats_info = "\n研究统计:\n"
    stats_info += f"- 总查询数: {statistics.get('total_queries', 0)}\n"
    stats_info += f"- 成功提取: {statistics.get('successful_extractions', 0)}\n"
    stats_info += f"- 成功率: {statistics.get('success_rate', 0):.2%}\n"
    stats_info += f"- 平均置信度: {statistics.get('average_confidence', 0):.2f}\n"

    return "\n\n===== 结构化研究简报开始 =====\n\n" + "\n".join(formatted_briefs) + stats_info + "\n===== 结构化研究简报结束 =====\n\n"


def _init_citation_manager(config: Config) -> CitationManager:
    """初始化引用管理器"""
    # 优先复用工作流已经初始化的嵌入模型，避免重复建连
    embedding_model = getattr(config, "embedding_model_instance", None)
    if embedding_model is not None:
        if not getattr(embedding_model, "client", None):
            logging.info("嵌入客户端不可用，将使用关键词匹配进行引用对齐")
    else:
        try:
            embedding_model = EmbeddingModel(config)
            if not embedding_model.client:
                logging.info("嵌入客户端不可用，将使用关键词匹配进行引用对齐")
        except Exception as e:
            logging.warning(f"初始化嵌入模型失败: {e}，将使用关键词匹配")
            embedding_model = None

    return CitationManager(embedding_model=embedding_model)


def _extract_gap_keywords(knowledge_gaps: list[str]) -> tuple[set[str], set[str]]:
    ascii_keywords: set[str] = set()
    cjk_keywords: set[str] = set()
    for gap in knowledge_gaps:
        if not isinstance(gap, str):
            continue
        text = gap.strip()
        if not text:
            continue
        for token in re.findall(r"[A-Za-z0-9]{3,}", text):
            ascii_keywords.add(token.lower())
        for token in re.findall(r"[\u4e00-\u9fff]{2,}", text):
            cjk_keywords.add(token)
    return ascii_keywords, cjk_keywords


def _add_structured_sources_to_citation_manager(
    citation_manager: CitationManager,
    structured_results: dict[str, Any],
    knowledge_gaps: list[str],
) -> None:
    """从结构化研究结果中添加信息源到引用管理器"""
    briefs = structured_results.get("briefs", [])
    ascii_keywords, cjk_keywords = _extract_gap_keywords(knowledge_gaps)
    keyword_filter_active = bool(ascii_keywords or cjk_keywords)
    accepted_sources = 0

    for brief in briefs:
        try:
            url = (brief.get("url") or "").strip()
            summary = (brief.get("summary") or "").strip()
            title = (brief.get("title") or "").strip()
            if not summary:
                continue
            if not url.startswith(("http://", "https://")):
                continue

            combined_text = f"{title}\n{summary}"
            combined_lower = combined_text.lower()

            if keyword_filter_active:
                ascii_match = any(keyword in combined_lower for keyword in ascii_keywords)
                cjk_match = any(keyword in combined_text for keyword in cjk_keywords)
                if not ascii_match and not cjk_match:
                    logging.debug("跳过信息源 '%s'：与知识缺口不匹配。", title or url)
                    continue

            confidence_raw = brief.get("confidence", 0.8)
            try:
                confidence = float(confidence_raw)
            except (TypeError, ValueError):
                confidence = 0.8

            source = SourceInfo(
                id=brief.get("id", ""),
                url=url,
                title=title,
                date=brief.get("date", ""),
                content="",
                summary=summary,
                confidence=confidence,
            )
            citation_manager.add_source(source)
            accepted_sources += 1
        except Exception as e:
            logging.warning(f"添加信息源失败: {e}")

    if keyword_filter_active and accepted_sources == 0 and briefs:
        logging.info("结构化研究结果与知识缺口不匹配，已跳过 %s 个候选信息源。", len(briefs))


def _convert_enhanced_to_structured(research_brief: str) -> dict[str, Any]:
    """将增强研究简报转换为结构化格式"""
    if not research_brief:
        return {"briefs": [], "statistics": {}}

    # 解析简报内容，提取结构化信息
    # 这里简化处理，实际应用中可能需要更复杂的解析

    briefs = []
    statistics = {
        "total_queries": 0,
        "successful_extractions": 0,
        "failed_extractions": 0,
        "total_urls_processed": 0,
        "high_quality_sources": 0,
        "success_rate": 0.0,
        "average_confidence": 0.8,
    }

    # 简单的URL提取和解析
    url_pattern = r"URL:\s*(https?://[^\s\n]+)"
    url_matches = re.findall(url_pattern, research_brief)

    for i, url in enumerate(url_matches):
        brief = {
            "id": f"enhanced_{i}",
            "url": url.strip(),
            "title": url.strip(),
            "knowledge_gap": "enhanced_research",
            "specific_query": "enhanced_query",
            "summary": "Enhanced research result",
            "key_points": [],
            "confidence": 0.8,
            "source_quality": "high",
            "relevance_score": 0.8,
            "extraction_date": datetime.now().strftime("%Y-%m-%d"),
            "text_length": len(research_brief),
            "raw_content": research_brief[:1000] if len(research_brief) > 1000 else research_brief,
            "created_at": datetime.now().isoformat(),
            "source_type": "enhanced_web",
        }
        briefs.append(brief)
        statistics["successful_extractions"] += 1
        statistics["total_urls_processed"] += 1

    # 计算成功率
    if statistics["total_urls_processed"] > 0:
        statistics["success_rate"] = statistics["successful_extractions"] / statistics["total_urls_processed"]

    return {"briefs": briefs, "statistics": statistics, "timestamp": datetime.now().isoformat()}


def _process_citations_from_research(
    citation_manager: CitationManager,
    research_brief: str,
    structured_results: dict[str, Any] | None,
    knowledge_gaps: list[str],
) -> tuple[list[ClaimInfo], list[CitationMatch]]:
    """从研究结果中提取主张并绑定到数据源"""
    claims = []
    citation_matches = []

    if not research_brief:
        return claims, citation_matches

    try:
        # 从研究简报中提取主张
        claims = citation_manager.extract_claims(research_brief)

        if claims:
            # 将主张与数据源对齐
            citation_matches = citation_manager.align_claims_to_sources(claims)
            citation_manager.citations = citation_matches

            # 记录处理结果
            logging.info(f"完成引用处理：提取 {len(claims)} 个主张，生成 {len(citation_matches)} 个匹配")

            # 如果有结构化数据，确保所有源都被添加到引用管理器中
            if structured_results:
                _add_structured_sources_to_citation_manager(citation_manager, structured_results, knowledge_gaps)
        else:
            logging.info("未提取到任何主张")

    except Exception as e:
        logging.error(f"处理引用时出错: {e}")

    return claims, citation_matches


def _get_enhanced_statistics(config: Config, structured_results: dict[str, Any]) -> dict[str, Any]:
    """获取增强的研究统计信息"""
    statistics = structured_results.get("statistics", {})

    # 添加SmartFetcher统计
    fetcher_stats = _get_fetcher_stats(config)
    diagnostic_stats = _get_diagnostic_stats()

    # 组合所有统计信息
    enhanced_stats = {
        **statistics,
        "smart_fetcher_integration": {
            "enabled": fetcher_stats.get("smart_fetcher_enabled", False),
            "supports_async": fetcher_stats.get("supports_async", False),
            "batch_processing": fetcher_stats.get("batch_processing", False),
        },
        "diagnostic_system": {
            "enabled": diagnostic_stats.get("diagnostic_system_enabled", False),
            "supported_metrics": diagnostic_stats.get("supported_metrics", []),
            "blocking_patterns_detected": diagnostic_stats.get("blocking_patterns_detected", []),
        },
        "performance_metrics": {
            "average_confidence": statistics.get("average_confidence", 0.0),
            "success_rate": statistics.get("success_rate", 0.0),
            "high_quality_rate": statistics.get("high_quality_rate", 0.0),
            "total_extraction_time": getattr(config, "total_extraction_time", None),
        },
    }

    return enhanced_stats


__all__ = ["research_node"]
