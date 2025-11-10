from __future__ import annotations

import hashlib
import logging
import os
from collections.abc import Mapping
from collections.abc import Mapping as MappingABC
from dataclasses import asdict, dataclass, is_dataclass
from functools import lru_cache
from typing import TYPE_CHECKING, Any, cast

try:
    import tiktoken
except ImportError:  # pragma: no cover - optional dependency
    tiktoken = None

from pydantic import ValidationError

from .client_factory import create_deepseek_client
from .env_loader import EnvironmentSettings, load_environment_settings
from .logging_setup import setup_session_logging

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from services.vector_db import EmbeddingModel  # pragma: no cover


@dataclass(frozen=True)
class APISettings:
    """顶层API凭证和相关元数据。"""

    deepseek_api_key: str | None
    deepseek_base_url: str | None
    api_request_timeout_seconds: int
    embedding_api_base_url: str | None
    embedding_api_key: str | None
    embedding_model_name: str | None
    google_api_keys: list[str]
    google_cse_ids: list[str]
    google_service_account_path: str | None
    user_agent: str
    current_google_api_key_index: int = 0

    @classmethod
    def from_env(cls, env: EnvironmentSettings) -> APISettings:
        return cls(
            deepseek_api_key=env.deepseek_api_key,
            deepseek_base_url=env.deepseek_base_url,
            api_request_timeout_seconds=env.api_request_timeout_seconds,
            embedding_api_base_url=env.embedding_api_base_url,
            embedding_api_key=env.embedding_api_key,
            embedding_model_name=env.embedding_model_name,
            google_api_keys=env.google_api_keys,
            google_cse_ids=env.google_cse_ids,
            google_service_account_path=env.google_service_account_path,
            user_agent=env.user_agent,
        )


@dataclass(frozen=True)
class VectorRetrievalSettings:
    """向量存储、混合检索和搜索预算的配置。"""

    vector_db_path: str
    vector_db_collection_name: str
    embedding_batch_size: int
    num_retrieved_experiences: int
    enable_hybrid_search: bool
    enable_bm25_search: bool
    enable_rerank: bool
    bm25_weight: float
    rerank_model_name: str | None
    enable_semantic_search: bool
    search_result_diversity: float
    max_search_iterations: int
    search_confidence_threshold: float
    num_search_results: int
    max_queries_per_gap: int
    enable_multilingual_search: bool

    @classmethod
    def from_env(cls, env: EnvironmentSettings) -> VectorRetrievalSettings:
        return cls(
            vector_db_path=env.vector_db_path,
            vector_db_collection_name=env.vector_db_collection_name,
            embedding_batch_size=env.embedding_batch_size,
            num_retrieved_experiences=env.num_retrieved_experiences,
            enable_hybrid_search=env.enable_hybrid_search,
            enable_bm25_search=env.enable_bm25_search,
            enable_rerank=env.enable_rerank,
            bm25_weight=env.bm25_weight,
            rerank_model_name=env.rerank_model_name,
            enable_semantic_search=env.enable_semantic_search,
            search_result_diversity=env.search_result_diversity,
            max_search_iterations=env.max_search_iterations,
            search_confidence_threshold=env.search_confidence_threshold,
            num_search_results=env.num_search_results,
            max_queries_per_gap=env.max_queries_per_gap,
            enable_multilingual_search=env.enable_multilingual_search,
        )


@dataclass(frozen=True)
class OutputQualitySettings:
    """结构化输出、引用和事实检查配置。"""

    use_structured_plan_output: bool
    structured_output_max_tokens: int
    structured_research_max_briefs: int
    enable_output_validation: bool
    enable_citation_management: bool
    citation_style: str
    citation_max_sources: int
    citation_min_relevance_score: float
    enable_fact_checking: bool
    fact_check_threshold: float
    fact_check_max_claims: int
    fact_check_model_name: str | None
    enable_cross_validation: bool

    @classmethod
    def from_env(cls, env: EnvironmentSettings) -> OutputQualitySettings:
        return cls(
            use_structured_plan_output=env.use_structured_plan_output,
            structured_output_max_tokens=env.structured_output_max_tokens,
            structured_research_max_briefs=env.structured_research_max_briefs,
            enable_output_validation=env.enable_output_validation,
            enable_citation_management=env.enable_citation_management,
            citation_style=env.citation_style,
            citation_max_sources=env.citation_max_sources,
            citation_min_relevance_score=env.citation_min_relevance_score,
            enable_fact_checking=env.enable_fact_checking,
            fact_check_threshold=env.fact_check_threshold,
            fact_check_max_claims=env.fact_check_max_claims,
            fact_check_model_name=env.fact_check_model_name,
            enable_cross_validation=env.enable_cross_validation,
        )


@dataclass(frozen=True)
class ModelSettings:
    """整个工作流中使用的模型系列。"""

    main_ai_model: str
    main_ai_model_heavy: str | None
    secondary_ai_model: str
    summary_model_name: str
    researcher_model_name: str
    outline_model_name: str
    planning_review_model_name: str | None
    editorial_model_name: str
    json_fixer_model_name: str | None
    patcher_model_name: str | None

    @classmethod
    def from_env(cls, env: EnvironmentSettings) -> ModelSettings:
        editorial_model = env.editorial_model_name or env.main_ai_model
        json_fixer = env.json_fixer_model_name or "deepseek-coder"
        if "reasoner" in json_fixer.lower():
            logger.info(
                "JSON fixer 模型设置为 reasoner，已自动切换到 deepseek-coder 以提高 "
                "JSON 修复稳定性。"
            )
            json_fixer = "deepseek-coder"
        patcher_model = env.patcher_model_name or json_fixer
        return cls(
            main_ai_model=env.main_ai_model,
            main_ai_model_heavy=env.main_ai_model_heavy,
            secondary_ai_model=env.secondary_ai_model,
            summary_model_name=env.summary_model_name,
            researcher_model_name=env.researcher_model_name,
            outline_model_name=env.outline_model_name,
            planning_review_model_name=env.planning_review_model_name,
            editorial_model_name=editorial_model,
            json_fixer_model_name=json_fixer,
            patcher_model_name=patcher_model,
        )


@dataclass(frozen=True)
class GenerationSettings:
    """Token预算和采样超参数。"""

    temperature_factual: float
    top_p_factual: float
    temperature_creative: float
    top_p_creative: float
    frequency_penalty: float
    presence_penalty: float
    embedding_model_max_tokens: int
    max_context_for_long_text_review_tokens: int
    intermediate_edit_max_tokens: int
    polish_section_max_tokens: int
    max_chunk_tokens: int
    overlap_chars: int
    max_chunks_per_section: int
    min_allocated_chars_for_section: int
    # 为输出留足空间的提示词预算比例（0-1）
    prompt_budget_ratio: float

    @classmethod
    def from_env(cls, env: EnvironmentSettings) -> GenerationSettings:
        return cls(
            temperature_factual=env.temperature_factual,
            top_p_factual=env.top_p_factual,
            temperature_creative=env.temperature_creative,
            top_p_creative=env.top_p_creative,
            frequency_penalty=env.frequency_penalty,
            presence_penalty=env.presence_penalty,
            embedding_model_max_tokens=env.embedding_model_max_tokens,
            max_context_for_long_text_review_tokens=env.max_context_for_long_text_review_tokens,
            intermediate_edit_max_tokens=env.intermediate_edit_max_tokens,
            polish_section_max_tokens=env.polish_section_max_tokens,
            max_chunk_tokens=env.max_chunk_tokens,
            overlap_chars=env.overlap_chars,
            max_chunks_per_section=env.max_chunks_per_section,
            min_allocated_chars_for_section=env.min_allocated_chars_for_section,
            prompt_budget_ratio=env.prompt_budget_ratio,
        )


@dataclass(frozen=True)
class WorkflowFlags:
    """控制工作流行为的布尔开关。"""

    interactive_mode: bool
    use_async_research: bool
    enable_dynamic_outline_correction: bool
    enable_web_research: bool
    use_structured_draft_output: bool
    enable_multilingual_search: bool
    use_hyde_for_structured: bool
    enable_research_cache: bool
    enable_patch_retry: bool
    disable_early_exit: bool
    disable_final_quality_check: bool
    disable_memory_node: bool
    disable_rag_for_patch: bool
    debug_json_repair: bool

    @classmethod
    def from_env(cls, env: EnvironmentSettings) -> WorkflowFlags:
        return cls(
            interactive_mode=env.interactive_mode,
            use_async_research=env.use_async_research,
            enable_dynamic_outline_correction=env.enable_dynamic_outline_correction,
            enable_web_research=env.enable_web_research,
            use_structured_draft_output=env.use_structured_draft_output,
            enable_multilingual_search=env.enable_multilingual_search,
            use_hyde_for_structured=env.use_hyde_for_structured,
            enable_research_cache=env.enable_research_cache,
            enable_patch_retry=env.enable_patch_retry,
            disable_early_exit=env.disable_early_exit,
            disable_final_quality_check=env.disable_final_quality_check,
            disable_memory_node=env.disable_memory_node,
            disable_rag_for_patch=env.disable_rag_for_patch,
            debug_json_repair=env.debug_json_repair,
        )


@dataclass(frozen=True)
class RuntimeSettings:
    """时间、并发和重试控制。"""

    api_request_timeout_seconds: int
    max_refinement_rounds: int
    max_concurrent_tasks: int
    initial_solution_target_chars: int
    api_retry_max_attempts: int
    api_retry_wait_multiplier: float
    api_retry_max_wait: float
    research_cache_ttl_seconds: int

    @classmethod
    def from_env(cls, env: EnvironmentSettings) -> RuntimeSettings:
        return cls(
            api_request_timeout_seconds=env.api_request_timeout_seconds,
            max_refinement_rounds=env.max_iterations,
            max_concurrent_tasks=env.max_concurrent_tasks,
            initial_solution_target_chars=env.initial_solution_target_chars,
            api_retry_max_attempts=env.api_retry_max_attempts,
            api_retry_wait_multiplier=env.api_retry_wait_multiplier,
            api_retry_max_wait=env.api_retry_max_wait,
            research_cache_ttl_seconds=env.research_cache_ttl_seconds,
        )


@dataclass(frozen=True)
class ProxySettings:
    """LLM代理控制。"""

    llm_disable_proxy: bool
    llm_http_proxy: str | None
    llm_https_proxy: str | None
    llm_proxy_trust_env: bool

    @classmethod
    def from_env(cls, env: EnvironmentSettings) -> ProxySettings:
        return cls(
            llm_disable_proxy=env.llm_disable_proxy,
            llm_http_proxy=env.llm_http_proxy,
            llm_https_proxy=env.llm_https_proxy,
            llm_proxy_trust_env=env.llm_proxy_trust_env,
        )


@dataclass(frozen=True)
class FetcherSettings:
    """智能抓取器和爬虫配置。"""

    proxy_url: str | None
    fetch_timeout: int
    per_host_rps: float
    max_concurrent: int
    playwright_enabled: bool
    extractor_order: str
    respect_robots: bool
    retry_max: int
    retry_backoff_base: float
    block_patterns: str
    enable_diagnostics: bool
    fetch_user_agent: str | None
    fetch_headers: str
    domain_configs: str
    enable_cache: bool
    cache_ttl: int
    enable_content_filter: bool
    min_content_length: int
    max_content_length: int

    @classmethod
    def from_env(cls, env: EnvironmentSettings) -> FetcherSettings:
        return cls(
            proxy_url=env.proxy_url,
            fetch_timeout=env.fetch_timeout,
            per_host_rps=env.per_host_rps,
            max_concurrent=env.max_concurrent,
            playwright_enabled=env.playwright_enabled,
            extractor_order=env.extractor_order,
            respect_robots=env.respect_robots,
            retry_max=env.retry_max,
            retry_backoff_base=env.retry_backoff_base,
            block_patterns=env.block_patterns,
            enable_diagnostics=env.enable_diagnostics,
            fetch_user_agent=env.fetch_user_agent,
            fetch_headers=env.fetch_headers,
            domain_configs=env.domain_configs,
            enable_cache=env.enable_cache,
            cache_ttl=env.cache_ttl,
            enable_content_filter=env.enable_content_filter,
            min_content_length=env.min_content_length,
            max_content_length=env.max_content_length,
        )


class Config:
    """聚合运行时设置和辅助工具的中央配置对象。"""

    api: APISettings
    vector: VectorRetrievalSettings
    output: OutputQualitySettings
    models: ModelSettings
    generation: GenerationSettings
    workflow: WorkflowFlags
    runtime: RuntimeSettings
    proxies: ProxySettings
    fetch: FetcherSettings
    fetch_settings: FetcherSettings

    # Individual attributes copied from dataclass sections (kept for backwards compatibility)
    deepseek_api_key: str | None
    deepseek_base_url: str | None
    embedding_api_base_url: str | None
    embedding_api_key: str | None
    embedding_model_name: str | None
    google_api_keys: list[str]
    google_cse_ids: list[str]
    google_service_account_path: str | None
    user_agent: str
    current_google_api_key_index: int

    vector_db_path: str
    vector_db_collection_name: str
    embedding_batch_size: int
    num_retrieved_experiences: int
    enable_hybrid_search: bool
    enable_bm25_search: bool
    enable_rerank: bool
    bm25_weight: float
    rerank_model_name: str | None
    enable_semantic_search: bool
    search_result_diversity: float
    max_search_iterations: int
    search_confidence_threshold: float
    num_search_results: int
    max_queries_per_gap: int

    use_structured_plan_output: bool
    structured_output_max_tokens: int
    structured_research_max_briefs: int
    enable_output_validation: bool
    enable_citation_management: bool
    citation_style: str
    citation_max_sources: int
    citation_min_relevance_score: float
    enable_fact_checking: bool
    fact_check_threshold: float
    fact_check_max_claims: int
    fact_check_model_name: str | None
    enable_cross_validation: bool

    main_ai_model: str
    main_ai_model_heavy: str | None
    secondary_ai_model: str
    summary_model_name: str
    researcher_model_name: str
    outline_model_name: str
    planning_review_model_name: str | None
    editorial_model_name: str
    json_fixer_model_name: str | None
    patcher_model_name: str | None

    temperature_factual: float
    top_p_factual: float
    temperature_creative: float
    top_p_creative: float
    frequency_penalty: float
    presence_penalty: float
    embedding_model_max_tokens: int
    max_context_for_long_text_review_tokens: int
    intermediate_edit_max_tokens: int
    polish_section_max_tokens: int
    max_chunk_tokens: int
    overlap_chars: int
    max_chunks_per_section: int
    min_allocated_chars_for_section: int
    prompt_budget_ratio: float

    interactive_mode: bool
    use_async_research: bool
    enable_dynamic_outline_correction: bool
    enable_web_research: bool
    use_structured_draft_output: bool
    enable_multilingual_search: bool
    use_hyde_for_structured: bool
    enable_research_cache: bool
    enable_patch_retry: bool
    disable_early_exit: bool
    disable_final_quality_check: bool
    disable_memory_node: bool
    disable_rag_for_patch: bool
    debug_json_repair: bool

    api_request_timeout_seconds: int
    max_refinement_rounds: int
    max_concurrent_tasks: int
    initial_solution_target_chars: int
    api_retry_max_attempts: int
    api_retry_wait_multiplier: float
    api_retry_max_wait: float
    research_cache_ttl_seconds: int

    llm_disable_proxy: bool
    llm_http_proxy: str | None
    llm_https_proxy: str | None
    llm_proxy_trust_env: bool

    proxy_url: str | None
    fetch_timeout: int
    per_host_rps: float
    max_concurrent: int
    playwright_enabled: bool
    extractor_order: str
    respect_robots: bool
    retry_max: int
    retry_backoff_base: float
    block_patterns: str
    enable_diagnostics: bool
    fetch_user_agent: str | None
    fetch_headers: str
    domain_configs: str
    enable_cache: bool
    cache_ttl: int
    enable_content_filter: bool
    min_content_length: int
    max_content_length: int

    # Runtime-populated attributes
    session_base_dir: str
    session_dir: str
    log_file_path: str
    user_problem: str
    external_data_files: list[str]
    embedding_model_instance: EmbeddingModel | None
    client: Any
    async_client: Any
    encoder: Any
    task_id: str

    def __init__(self):
        try:
            env_settings = load_environment_settings()
        except ValidationError:
            raise

        self._env_settings: EnvironmentSettings = env_settings

        # Populate grouped settings while preserving attribute-level backwards compatibility.
        self.api = self._assign_settings(APISettings.from_env(env_settings), alias="api")
        self.vector = self._assign_settings(
            VectorRetrievalSettings.from_env(env_settings),
            alias="vector",
        )
        self.output = self._assign_settings(
            OutputQualitySettings.from_env(env_settings),
            alias="output",
        )
        self.models = self._assign_settings(
            ModelSettings.from_env(env_settings),
            alias="models",
        )
        self.generation = self._assign_settings(
            GenerationSettings.from_env(env_settings),
            alias="generation",
        )
        self.workflow = self._assign_settings(
            WorkflowFlags.from_env(env_settings),
            alias="workflow",
        )
        self.runtime = self._assign_settings(
            RuntimeSettings.from_env(env_settings),
            alias="runtime",
        )
        self.proxies = self._assign_settings(
            ProxySettings.from_env(env_settings),
            alias="proxies",
        )
        self.fetch = self._assign_settings(
            FetcherSettings.from_env(env_settings),
            alias="fetch_settings",
        )

        self._initialize_paths(env_settings)
        self._initialize_defaults()
        self._initialize_encoder()
        self._configure_user_agent_env()

    @staticmethod
    def _dataclass_as_dict(section: Any) -> dict[str, Any]:
        if is_dataclass(section) and not isinstance(section, type):
            return asdict(section)
        if isinstance(section, MappingABC):
            mapping_section = cast(Mapping[str, Any], section)
            normalized: dict[str, Any] = {}
            for key, value in mapping_section.items():
                normalized[str(key)] = value
            return normalized
        raise TypeError(
            f"Unsupported settings section type: {type(section)!r}. "
            "Expected dataclass or mapping."
        )

    def _assign_settings(self, section: Any, *, alias: str | None = None) -> Any:
        """
        Copy fields from a dataclass section onto the config instance.
        """
        for key, value in self._dataclass_as_dict(section).items():
            setattr(self, key, value)
        if alias:
            setattr(self, alias, section)
        return section

    def _initialize_paths(self, env_settings: EnvironmentSettings) -> None:
        """
        Resolve session paths and ensure the session directory exists.
        """
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        default_session_base_dir = os.path.join(project_root, "outputs")
        session_base = env_settings.session_base_dir or default_session_base_dir
        if not os.path.isabs(session_base):
            session_base = os.path.join(project_root, session_base)
        self.session_base_dir = os.path.abspath(os.path.normpath(session_base))
        os.makedirs(self.session_base_dir, exist_ok=True)
        self.session_dir = ""
        self.log_file_path = ""

    def _initialize_defaults(self) -> None:
        """
        Set mutable runtime defaults that are populated later in the workflow.
        """
        # 从环境变量加载用户任务和外部文件
        self.user_problem = self._env_settings.user_problem
        external_files_str = self._env_settings.external_files
        if external_files_str:
            # 分割逗号分隔的路径，并清理空格
            self.external_data_files = [
                f.strip()
                for f in external_files_str.split(",")
                if f.strip()
            ]
        else:
            self.external_data_files = []

        self.embedding_model_instance = None
        self.client = None
        self.async_client = None
        self.task_id = ""
        self.encoder = None

    def _initialize_encoder(self) -> None:
        """
        Attempt to load the tiktoken encoder, falling back gracefully when unavailable.
        """
        if not tiktoken:
            logger.warning("tiktoken 库未安装，将使用近似 token 计数。")
            self.encoder = None
            return
        try:
            self.encoder = tiktoken.get_encoding("cl100k_base")
        except Exception as exc:
            logger.error("严重错误：初始化 tiktoken 编码器失败: %s", exc)
            self.encoder = None

    def _configure_user_agent_env(self) -> None:
        """
        Ensure the configured User-Agent is propagated to the environment variables.
        """
        if not os.getenv("USER_AGENT"):
            os.environ["USER_AGENT"] = self.user_agent
            logger.info("USER_AGENT 环境变量未设置，已自动应用默认值: %s", self.user_agent)

    @property
    def max_iterations(self) -> int:
        """向后兼容：提供原有命名的访问入口。"""
        return self.max_refinement_rounds

    @max_iterations.setter
    def max_iterations(self, value: int) -> None:
        self.max_refinement_rounds = value

    def setup_logging(self, logging_level: int = logging.INFO) -> None:
        """为当前运行会话配置日志记录器。"""
        setup_session_logging(self, logging_level)

    def initialize_deepseek_client(self) -> None:
        """初始化与 DeepSeek API 通信的客户端。"""
        self.client = create_deepseek_client(self)

    def count_tokens(self, text: str) -> int:
        """使用 tiktoken 计算文本的 token 数量（带LRU缓存），如果失败则回退到近似计算。"""
        if not text:
            return 0

        # 使用缓存的辅助函数（通过文本哈希避免存储大文本）
        text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
        return self._count_tokens_cached(text_hash, text)

    @lru_cache(maxsize=1024)
    def _count_tokens_cached(self, text_hash: str, text: str) -> int:
        """
        带缓存的token计数实现

        Args:
            text_hash: 文本的MD5哈希（用于缓存键）
            text: 实际文本内容

        Returns:
            token数量
        """
        if self.encoder:
            try:
                return len(self.encoder.encode(text))
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("Tiktoken 编码失败: %s。回退到近似计算。", exc)
                return len(text) // 3
        logger.warning("Tiktoken 编码器不可用，Token 计数使用近似值。")
        return len(text) // 3


__all__ = ["Config", "EnvironmentSettings"]
