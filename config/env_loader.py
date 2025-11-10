from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from pydantic import Field, ValidationError, field_validator

try:
    from pydantic_settings import BaseSettings, SettingsConfigDict

    PYDANTIC_SETTINGS_AVAILABLE = True
except ImportError:
    # 回退到 pydantic v2 的 BaseSettings（如果可用）
    try:
        from pydantic import BaseSettings as BaseSettingsV2  # type: ignore[attr-defined]

        BaseSettings = BaseSettingsV2  # type: ignore[misc,assignment]
        PYDANTIC_SETTINGS_AVAILABLE = False
        SettingsConfigDict = None  # type: ignore[assignment,misc]
    except ImportError:
        from pydantic import BaseModel as BaseSettings  # type: ignore[assignment,misc,name-defined]

        PYDANTIC_SETTINGS_AVAILABLE = False
        SettingsConfigDict = None  # type: ignore[assignment,misc]

logger = logging.getLogger(__name__)

# 获取env文件的绝对路径（相对于此模块所在的config目录的父目录）
_CONFIG_DIR = Path(__file__).parent
_PROJECT_ROOT = _CONFIG_DIR.parent
_ENV_FILES = [
    str(_PROJECT_ROOT / "env"),
    str(_PROJECT_ROOT / ".env"),
    str(_PROJECT_ROOT / ".env.local"),
]

# 所有旧的辅助函数已删除，pydantic-settings 会自动处理类型转换


class EnvironmentSettings(BaseSettings):  # type: ignore[misc]
    """环境配置（使用 pydantic-settings 自动解析和验证）"""

    # Pydantic Settings 配置
    if PYDANTIC_SETTINGS_AVAILABLE and SettingsConfigDict:
        model_config = SettingsConfigDict(
            env_file=tuple(_ENV_FILES),  # 使用绝对路径，支持从任何目录运行
            env_file_encoding="utf-8",
            case_sensitive=False,
            extra="ignore",  # 忽略未定义的环境变量
            env_parse_enums=False,  # 避免过度解析
            populate_by_name=True,  # 支持使用字段别名
        )

    # API 配置
    deepseek_api_key: str
    deepseek_base_url: str = "https://api.deepseek.com"
    embedding_api_base_url: str | None = None
    embedding_api_key: str | None = None
    embedding_model_name: str = "bge-m3"

    # Google API 配置
    google_api_keys: list[str] = Field(default_factory=list)
    google_cse_ids: list[str] = Field(default_factory=list)
    google_service_account_path: str | None = None

    vector_db_path: str = "./chroma_db"
    vector_db_collection_name: str = "experience_store"
    embedding_batch_size: int = 25
    num_retrieved_experiences: int = 3

    main_ai_model: str = "deepseek-chat"
    main_ai_model_heavy: str = "deepseek-reasoner"
    secondary_ai_model: str = "deepseek-reasoner"
    summary_model_name: str = "deepseek-coder"
    researcher_model_name: str = "deepseek-reasoner"
    outline_model_name: str = "deepseek-coder"
    planning_review_model_name: str = "deepseek-coder"
    editorial_model_name: str | None = None
    json_fixer_model_name: str = "deepseek-coder"
    patcher_model_name: str | None = None

    temperature_factual: float = Field(default=0.1, alias="llm_temperature_factual")
    top_p_factual: float = 0.95
    temperature_creative: float = Field(default=0.3, alias="llm_temperature_creative")
    top_p_creative: float = 0.95
    frequency_penalty: float = Field(default=0.2, alias="llm_frequency_penalty")
    presence_penalty: float = Field(default=0.0, alias="llm_presence_penalty")

    api_request_timeout_seconds: int = 900
    max_iterations: int = 4
    max_concurrent_tasks: int = 2
    initial_solution_target_chars: int = 15000

    embedding_model_max_tokens: int = 1024
    max_context_for_long_text_review_tokens: int = Field(
        default=30000,
        alias="max_context_tokens_review",  # 支持 env 文件中的简短名称
    )
    intermediate_edit_max_tokens: int = 8192
    polish_section_max_tokens: int = 8192
    max_chunk_tokens: int = 4096
    overlap_chars: int = 800
    max_chunks_per_section: int = 20
    # 提示词预算比例（为输出留足空间）
    prompt_budget_ratio: float = 0.9

    api_retry_max_attempts: int = 3
    api_retry_wait_multiplier: int = 1
    api_retry_max_wait: int = 60

    num_search_results: int = 3
    max_queries_per_gap: int = 5
    structured_research_max_briefs: int = 6

    interactive_mode: bool = False
    use_async_research: bool = True
    enable_dynamic_outline_correction: bool = True
    enable_web_research: bool = True
    use_structured_draft_output: bool = False
    enable_multilingual_search: bool = True

    user_agent: str = "DeepResearch/1.0"

    session_base_dir: str | None = None
    min_allocated_chars_for_section: int = 100
    disable_early_exit: bool = True
    disable_final_quality_check: bool = True
    disable_memory_node: bool = True
    disable_rag_for_patch: bool = True

    llm_disable_proxy: bool = False
    llm_http_proxy: str | None = None
    llm_https_proxy: str | None = None
    llm_proxy_trust_env: bool = True

    # 混合检索配置
    enable_hybrid_search: bool = True
    enable_bm25_search: bool = True
    enable_rerank: bool = True
    bm25_weight: float = 0.3
    rerank_model_name: str | None = None

    # 结构化输出配置
    use_structured_plan_output: bool = False
    structured_output_max_tokens: int = 4096
    enable_output_validation: bool = True

    # 引用管理配置
    enable_citation_management: bool = True
    citation_style: str = "apa"
    citation_max_sources: int = 50
    citation_min_relevance_score: float = 0.7

    # 事实核查配置
    enable_fact_checking: bool = True
    fact_check_threshold: float = 0.8
    fact_check_max_claims: int = 10
    fact_check_model_name: str = "deepseek-reasoner"
    enable_cross_validation: bool = True

    # 高级检索配置
    enable_semantic_search: bool = True
    search_result_diversity: float = 0.1
    max_search_iterations: int = 3
    search_confidence_threshold: float = 0.6
    use_hyde_for_structured: bool = True

    # 用户任务和外部文件配置
    user_problem: str = ""  # 核心写作任务/研究问题
    external_files: str = ""  # 本地参考文件或目录路径（逗号分隔）

    # 抓取相关配置
    proxy_url: str | None = None
    fetch_timeout: int = 30
    per_host_rps: int = 5
    max_concurrent: int = 10
    playwright_enabled: bool = False
    extractor_order: str = "readability,transformers,regex"
    respect_robots: bool = True
    retry_max: int = 3
    retry_backoff_base: float = 1.0
    block_patterns: str = "google-analytics,googletagmanager,facebook,ads"
    enable_diagnostics: bool = True
    fetch_user_agent: str = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    )
    fetch_headers: str = (
        '{"Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"}'
    )
    domain_configs: str = "{}"
    enable_cache: bool = True
    cache_ttl: int = 3600
    enable_content_filter: bool = True
    min_content_length: int = 100
    max_content_length: int = 1048576
    enable_research_cache: bool = True
    research_cache_ttl_seconds: int = 14400
    enable_patch_retry: bool = True

    # 调试配置
    debug_json_repair: bool = False

    # Google API 配对验证
    @field_validator("google_api_keys", "google_cse_ids", mode="after")
    @classmethod
    def ensure_list(cls, v: Any) -> list[str]:
        """确保字段是列表"""
        if isinstance(v, list):
            return v
        if isinstance(v, str) and v:
            # 回退处理：如果仍然是字符串，尝试分割
            return [item.strip() for item in v.split(",") if item.strip()]
        return []

    @field_validator("fetch_headers", "domain_configs", mode="before")
    @classmethod
    def parse_json_string(cls, v: Any) -> str:
        """保持 JSON 字符串格式（后续会被解析）"""
        if isinstance(v, str):
            return v
        if isinstance(v, dict):
            import json

            return json.dumps(v)
        return "{}"

    @field_validator("*", mode="before")
    @classmethod
    def strip_quotes(cls, v: Any) -> Any:
        """移除环境变量中的引号"""
        if isinstance(v, str):
            v = v.strip()
            if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
                return v[1:-1]
        return v

    @field_validator("*", mode="after")
    @classmethod
    def _align_google_pairs(cls, info) -> Any:
        """验证 Google API 配对"""
        if hasattr(info, "google_api_keys") and hasattr(info, "google_cse_ids"):
            if info.google_api_keys and info.google_cse_ids:
                if len(info.google_api_keys) != len(info.google_cse_ids):
                    min_len = min(len(info.google_api_keys), len(info.google_cse_ids))
                    logger.warning(
                        "GOOGLE_API_KEYS 和 GOOGLE_CSE_IDS 数量不匹配，将仅使用前 %s 对。",
                        min_len,
                    )
                    info.google_api_keys = info.google_api_keys[:min_len]
                    info.google_cse_ids = info.google_cse_ids[:min_len]
        return info


def load_environment_settings() -> EnvironmentSettings:
    """
    加载环境变量，验证它们，并返回 EnvironmentSettings 实例。
    使用 pydantic-settings 自动加载 .env 文件和系统环境变量。
    """
    try:
        # pydantic-settings 会自动从 .env 文件和环境变量加载
        settings = EnvironmentSettings()  # type: ignore
        logger.info(".env 文件和环境变量加载成功。")
        return settings
    except ValidationError as exc:
        logger.critical("配置初始化失败，请检查环境变量设置: %s", exc)
        raise
    except Exception as exc:
        logger.critical("环境设置加载异常: %s", exc)
        raise


__all__ = ["EnvironmentSettings", "load_environment_settings"]
