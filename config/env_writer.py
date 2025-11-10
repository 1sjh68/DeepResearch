from __future__ import annotations

import logging
import os
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from .config import Config

logger = logging.getLogger(__name__)


def _quote_if_needed(value: str) -> str:
    if any(c in value for c in [",", " ", "(", ")"]):
        return f'"{value}"'
    return value


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
        iterable = cast(Iterable[Any], value)
        parts: list[str] = []
        for item in iterable:
            parts.append(str(item))
        return ",".join(parts)
    return str(value)


def write_env_file(
    config: Config,
    env_file_path: str = ".env",
    *,
    confirm: bool = False,
    example_only: bool = False,
) -> bool:
    """
    Persist managed configuration values to an env file.
    Guards against accidentally writing secrets unless confirm=True.
    """
    if example_only:
        return write_env_example(config, env_file_path)

    if not confirm:
        logger.warning(
            "保护机制：未开启 confirm=True，已阻止将敏感配置写入 %s。",
            env_file_path,
        )
        logger.warning(
            "如需安全共享配置，请调用 Config.save_env_example(env_example_path='.env.example') "
            "生成示例文件。"
        )
        return False

    logger.info("正在尝试将配置安全地保存到 %s...", env_file_path)
    managed_keys: set[str] = {
        "DEEPSEEK_API_KEY",
        "DEEPSEEK_BASE_URL",
        "EMBEDDING_API_BASE_URL",
        "EMBEDDING_API_KEY",
        "GOOGLE_API_KEYS",
        "GOOGLE_CSE_IDS",
        "MAX_CHUNKS_PER_SECTION",
        "MAX_ITERATIONS",
        "INITIAL_SOLUTION_TARGET_CHARS",
        "NUM_RETRIEVED_EXPERIENCES",
        "MAIN_AI_MODEL",
        "MAIN_AI_MODEL_HEAVY",
        "SECONDARY_AI_MODEL",
        "SUMMARY_MODEL_NAME",
        "RESEARCHER_MODEL_NAME",
        "OUTLINE_MODEL_NAME",
        "PLANNING_REVIEW_MODEL_NAME",
        "EDITORIAL_MODEL_NAME",
        "JSON_FIXER_MODEL_NAME",
        "PATCHER_MODEL_NAME",
        "EMBEDDING_MODEL_NAME",
        "ENABLE_WEB_RESEARCH",
        "MAX_CONCURRENT_TASKS",
        "STRUCTURED_RESEARCH_MAX_BRIEFS",
        # 混合检索配置
        "ENABLE_HYBRID_SEARCH",
        "ENABLE_BM25_SEARCH",
        "ENABLE_RERANK",
        "BM25_WEIGHT",
        "RERANK_MODEL_NAME",
        # 结构化输出配置
        "USE_STRUCTURED_PLAN_OUTPUT",
        "STRUCTURED_OUTPUT_MAX_TOKENS",
        "ENABLE_OUTPUT_VALIDATION",
        # 引用管理配置
        "ENABLE_CITATION_MANAGEMENT",
        "CITATION_STYLE",
        "CITATION_MAX_SOURCES",
        "CITATION_MIN_RELEVANCE_SCORE",
        # 事实核查配置
        "ENABLE_FACT_CHECKING",
        "FACT_CHECK_THRESHOLD",
        "FACT_CHECK_MAX_CLAIMS",
        "FACT_CHECK_MODEL_NAME",
        "ENABLE_CROSS_VALIDATION",
        # 高级检索配置
        "ENABLE_SEMANTIC_SEARCH",
        "SEARCH_RESULT_DIVERSITY",
        "MAX_SEARCH_ITERATIONS",
        "SEARCH_CONFIDENCE_THRESHOLD",
        "USE_HYDE_FOR_STRUCTURED",
        # 抓取相关配置
        "PROXY_URL",
        "FETCH_TIMEOUT",
        "PER_HOST_RPS",
        "MAX_CONCURRENT",
        "PLAYWRIGHT_ENABLED",
        "EXTRACTOR_ORDER",
        "RESPECT_ROBOTS",
        "RETRY_MAX",
        "RETRY_BACKOFF_BASE",
        "BLOCK_PATTERNS",
        "ENABLE_DIAGNOSTICS",
        "FETCH_USER_AGENT",
        "FETCH_HEADERS",
        "DOMAIN_CONFIGS",
        "ENABLE_CACHE",
        "CACHE_TTL",
        "ENABLE_CONTENT_FILTER",
        "MIN_CONTENT_LENGTH",
        "MAX_CONTENT_LENGTH",
        # 提示词预算比例
        "PROMPT_BUDGET_RATIO",
    }

    new_values: dict[str, str] = {
        "DEEPSEEK_API_KEY": _stringify(config.api.deepseek_api_key),
        "DEEPSEEK_BASE_URL": _stringify(config.api.deepseek_base_url),
        "EMBEDDING_API_BASE_URL": _stringify(config.api.embedding_api_base_url),
        "EMBEDDING_API_KEY": _stringify(config.api.embedding_api_key),
        "GOOGLE_API_KEYS": ",".join(config.api.google_api_keys),
        "GOOGLE_CSE_IDS": ",".join(config.api.google_cse_ids),
        "MAX_CHUNKS_PER_SECTION": _stringify(config.generation.max_chunks_per_section),
        "POLISH_SECTION_MAX_TOKENS": _stringify(config.generation.polish_section_max_tokens),
        "MAX_ITERATIONS": _stringify(config.runtime.max_refinement_rounds),
        "INITIAL_SOLUTION_TARGET_CHARS": _stringify(config.runtime.initial_solution_target_chars),
        "NUM_RETRIEVED_EXPERIENCES": _stringify(config.vector.num_retrieved_experiences),
        "MAIN_AI_MODEL": _stringify(config.models.main_ai_model),
        "MAIN_AI_MODEL_HEAVY": _stringify(config.models.main_ai_model_heavy),
        "SECONDARY_AI_MODEL": _stringify(config.models.secondary_ai_model),
        "SUMMARY_MODEL_NAME": _stringify(config.models.summary_model_name),
        "RESEARCHER_MODEL_NAME": _stringify(config.models.researcher_model_name),
        "OUTLINE_MODEL_NAME": _stringify(config.models.outline_model_name),
        "PLANNING_REVIEW_MODEL_NAME": _stringify(config.models.planning_review_model_name),
        "EDITORIAL_MODEL_NAME": _stringify(config.models.editorial_model_name),
        "JSON_FIXER_MODEL_NAME": _stringify(config.models.json_fixer_model_name),
        "PATCHER_MODEL_NAME": _stringify(config.models.patcher_model_name),
        "EMBEDDING_MODEL_NAME": _stringify(config.api.embedding_model_name),
        "ENABLE_WEB_RESEARCH": _stringify(config.workflow.enable_web_research),
        "MAX_CONCURRENT_TASKS": _stringify(config.runtime.max_concurrent_tasks),
        "STRUCTURED_RESEARCH_MAX_BRIEFS": _stringify(config.output.structured_research_max_briefs),
        # 混合检索配置
        "ENABLE_HYBRID_SEARCH": _stringify(config.vector.enable_hybrid_search),
        "ENABLE_BM25_SEARCH": _stringify(config.vector.enable_bm25_search),
        "ENABLE_RERANK": _stringify(config.vector.enable_rerank),
        "BM25_WEIGHT": _stringify(config.vector.bm25_weight),
        "RERANK_MODEL_NAME": _stringify(config.vector.rerank_model_name),
        # 结构化输出配置
        "USE_STRUCTURED_PLAN_OUTPUT": _stringify(config.output.use_structured_plan_output),
        "STRUCTURED_OUTPUT_MAX_TOKENS": _stringify(config.output.structured_output_max_tokens),
        "ENABLE_OUTPUT_VALIDATION": _stringify(config.output.enable_output_validation),
        # 引用管理配置
        "ENABLE_CITATION_MANAGEMENT": _stringify(config.output.enable_citation_management),
        "CITATION_STYLE": _stringify(config.output.citation_style),
        "CITATION_MAX_SOURCES": _stringify(config.output.citation_max_sources),
        "CITATION_MIN_RELEVANCE_SCORE": _stringify(config.output.citation_min_relevance_score),
        # 事实核查配置
        "ENABLE_FACT_CHECKING": _stringify(config.output.enable_fact_checking),
        "FACT_CHECK_THRESHOLD": _stringify(config.output.fact_check_threshold),
        "FACT_CHECK_MAX_CLAIMS": _stringify(config.output.fact_check_max_claims),
        "FACT_CHECK_MODEL_NAME": _stringify(config.output.fact_check_model_name),
        "ENABLE_CROSS_VALIDATION": _stringify(config.output.enable_cross_validation),
        # 高级检索配置
        "ENABLE_SEMANTIC_SEARCH": _stringify(config.vector.enable_semantic_search),
        "SEARCH_RESULT_DIVERSITY": _stringify(config.vector.search_result_diversity),
        "MAX_SEARCH_ITERATIONS": _stringify(config.vector.max_search_iterations),
        "SEARCH_CONFIDENCE_THRESHOLD": _stringify(config.vector.search_confidence_threshold),
        "USE_HYDE_FOR_STRUCTURED": _stringify(config.workflow.use_hyde_for_structured),
        # 抓取相关配置
        "PROXY_URL": _stringify(config.fetch.proxy_url),
        "FETCH_TIMEOUT": _stringify(config.fetch.fetch_timeout),
        "PER_HOST_RPS": _stringify(config.fetch.per_host_rps),
        "MAX_CONCURRENT": _stringify(config.fetch.max_concurrent),
        "PLAYWRIGHT_ENABLED": _stringify(config.fetch.playwright_enabled),
        "EXTRACTOR_ORDER": _stringify(config.fetch.extractor_order),
        "RESPECT_ROBOTS": _stringify(config.fetch.respect_robots),
        "RETRY_MAX": _stringify(config.fetch.retry_max),
        "RETRY_BACKOFF_BASE": _stringify(config.fetch.retry_backoff_base),
        "BLOCK_PATTERNS": _stringify(config.fetch.block_patterns),
        "ENABLE_DIAGNOSTICS": _stringify(config.fetch.enable_diagnostics),
        "FETCH_USER_AGENT": _stringify(config.fetch.fetch_user_agent),
        "FETCH_HEADERS": _stringify(config.fetch.fetch_headers),
        "DOMAIN_CONFIGS": _stringify(config.fetch.domain_configs),
        "ENABLE_CACHE": _stringify(config.fetch.enable_cache),
        "CACHE_TTL": _stringify(config.fetch.cache_ttl),
        "ENABLE_CONTENT_FILTER": _stringify(config.fetch.enable_content_filter),
        "MIN_CONTENT_LENGTH": _stringify(config.fetch.min_content_length),
        "MAX_CONTENT_LENGTH": _stringify(config.fetch.max_content_length),
        "PROMPT_BUDGET_RATIO": _stringify(config.generation.prompt_budget_ratio),
    }

    updated_lines: list[str] = []
    keys_to_update: set[str] = set(managed_keys)

    try:
        if os.path.exists(env_file_path):
            with open(env_file_path, encoding="utf-8") as file:
                for line in file:
                    stripped = line.strip()
                    if not stripped or stripped.startswith("#") or "=" not in stripped:
                        updated_lines.append(line)
                        continue
                    key, _ = stripped.split("=", 1)
                    key = key.strip()
                    if key in managed_keys:
                        value: str = new_values.get(key, "")
                        updated_lines.append(f"{key}={_quote_if_needed(value)}\n")
                        keys_to_update.discard(key)
                    else:
                        updated_lines.append(line)

        for key in keys_to_update:
            value: str = new_values.get(key, "")
            updated_lines.append(f"{key}={_quote_if_needed(value)}\n")
            logger.info("  - 在 .env 文件中新增了配置项: %s", key)

        with open(env_file_path, "w", encoding="utf-8") as file:
            file.writelines(updated_lines)

        logger.info("配置已成功保存到 %s", env_file_path)
        return True
    except Exception as exc:  # pragma: no cover - IO safeguard
        logger.error("保存配置到 .env 文件时发生严重错误: %s", exc, exc_info=True)
        return False


def write_env_example(_config: Config, env_example_path: str = ".env.example") -> bool:
    """生成清理过的环境变量示例文件，所有管理的键留空。"""
    logger.info("正在生成示例环境文件: %s", env_example_path)
    try:
        managed_keys: Iterable[str] = [
            "DEEPSEEK_API_KEY",
            "DEEPSEEK_BASE_URL",
            "EMBEDDING_API_BASE_URL",
            "EMBEDDING_API_KEY",
            "EMBEDDING_MODEL_NAME",
            "GOOGLE_API_KEYS",
            "GOOGLE_CSE_IDS",
            "GOOGLE_APPLICATION_CREDENTIALS",
            "VECTOR_DB_PATH",
            "VECTOR_DB_COLLECTION_NAME",
            "EMBEDDING_BATCH_SIZE",
            "NUM_RETRIEVED_EXPERIENCES",
            "MAIN_AI_MODEL",
            "MAIN_AI_MODEL_HEAVY",
            "SECONDARY_AI_MODEL",
            "SUMMARY_MODEL_NAME",
            "RESEARCHER_MODEL_NAME",
            "OUTLINE_MODEL_NAME",
            "PLANNING_REVIEW_MODEL_NAME",
            "EDITORIAL_MODEL_NAME",
            "JSON_FIXER_MODEL_NAME",
            "PATCHER_MODEL_NAME",
            "LLM_TEMPERATURE_FACTUAL",
            "LLM_TOP_P_FACTUAL",
            "LLM_TEMPERATURE_CREATIVE",
            "LLM_TOP_P_CREATIVE",
            "LLM_FREQUENCY_PENALTY",
            "LLM_PRESENCE_PENALTY",
            "API_TIMEOUT_SECONDS",
            "MAX_ITERATIONS",
            "MAX_CONCURRENT_TASKS",
            "INITIAL_SOLUTION_TARGET_CHARS",
            "EMBEDDING_MODEL_MAX_TOKENS",
            "MAX_CONTEXT_TOKENS_REVIEW",
            "INTERMEDIATE_EDIT_MAX_TOKENS",
            "POLISH_SECTION_MAX_TOKENS",
            "MAX_CHUNK_TOKENS",
            "OVERLAP_CHARS",
            "MAX_CHUNKS_PER_SECTION",
            "API_RETRY_MAX_ATTEMPTS",
            "API_RETRY_WAIT_MULTIPLIER",
            "API_RETRY_MAX_WAIT",
            "NUM_SEARCH_RESULTS",
            "MAX_QUERIES_PER_GAP",
            "INTERACTIVE_MODE",
            "USE_ASYNC_RESEARCH",
            "ENABLE_DYNAMIC_OUTLINE_CORRECTION",
            "ENABLE_WEB_RESEARCH",
            "USER_AGENT",
            "SESSION_BASE_DIR",
            "MIN_ALLOCATED_CHARS_SECTION",
            "USER_PROBLEM",
            "EXTERNAL_FILES",
            # 混合检索配置
            "ENABLE_HYBRID_SEARCH",
            "ENABLE_BM25_SEARCH",
            "ENABLE_RERANK",
            "BM25_WEIGHT",
            "RERANK_MODEL_NAME",
            # 结构化输出配置
            "USE_STRUCTURED_PLAN_OUTPUT",
            "STRUCTURED_OUTPUT_MAX_TOKENS",
            "ENABLE_OUTPUT_VALIDATION",
            # 引用管理配置
            "ENABLE_CITATION_MANAGEMENT",
            "CITATION_STYLE",
            "CITATION_MAX_SOURCES",
            "CITATION_MIN_RELEVANCE_SCORE",
            # 事实核查配置
            "ENABLE_FACT_CHECKING",
            "FACT_CHECK_THRESHOLD",
            "FACT_CHECK_MAX_CLAIMS",
            "FACT_CHECK_MODEL_NAME",
            "ENABLE_CROSS_VALIDATION",
            # 高级检索配置
            "ENABLE_SEMANTIC_SEARCH",
            "SEARCH_RESULT_DIVERSITY",
            "MAX_SEARCH_ITERATIONS",
            "SEARCH_CONFIDENCE_THRESHOLD",
            # 抓取相关配置
            "PROXY_URL",
            "FETCH_TIMEOUT",
            "PER_HOST_RPS",
            "MAX_CONCURRENT",
            "PLAYWRIGHT_ENABLED",
            "EXTRACTOR_ORDER",
            "RESPECT_ROBOTS",
            "RETRY_MAX",
            "RETRY_BACKOFF_BASE",
            "BLOCK_PATTERNS",
            "ENABLE_DIAGNOSTICS",
            "FETCH_USER_AGENT",
            "FETCH_HEADERS",
            "DOMAIN_CONFIGS",
            "ENABLE_CACHE",
            "CACHE_TTL",
            "ENABLE_CONTENT_FILTER",
            "MIN_CONTENT_LENGTH",
            "MAX_CONTENT_LENGTH",
        ]

        lines: list[str] = [
            "# Example environment file. All values intentionally left blank.\n",
            "# Do NOT commit real secrets.\n",
        ]
        for key in managed_keys:
            lines.append(f"{key}=\n")

        with open(env_example_path, "w", encoding="utf-8") as file:
            file.writelines(lines)
        logger.info("示例环境文件已生成: %s", env_example_path)
        return True
    except Exception as exc:  # pragma: no cover - IO safeguard
        logger.error("生成示例环境文件失败: %s", exc, exc_info=True)
        return False


__all__ = ["write_env_file", "write_env_example"]
