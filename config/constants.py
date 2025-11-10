"""
配置常量模块

集中管理所有硬编码的常量、模型限制和默认值。
"""


# ==================== 模型限制 ====================

class ModelLimits:
    """各种模型的上下文和输出token限制"""

    # 上下文窗口大小（输入tokens）
    CONTEXT_LIMITS: dict[str, int] = {
        "deepseek-reasoner": 64000,
        "deepseek-chat": 64000,
        "deepseek-coder": 128000,
    }

    # 最大输出tokens
    MAX_OUTPUT_TOKENS: dict[str, int] = {
        "deepseek-reasoner": 4096,
        "deepseek-chat": 8192,
        "deepseek-coder": 8192,
    }

    # Reasoner模型的最小输出tokens（为思维链预留空间）
    REASONER_MIN_TOKENS: int = 2048

    @classmethod
    def get_context_limit(cls, model_name: str, default: int = 64000) -> int:
        """获取模型的上下文限制"""
        return cls.CONTEXT_LIMITS.get(model_name, default)

    @classmethod
    def get_max_output(cls, model_name: str, default: int = 4096) -> int:
        """获取模型的最大输出tokens"""
        return cls.MAX_OUTPUT_TOKENS.get(model_name, default)


# ==================== 重试配置 ====================

class RetryConfig:
    """API重试相关配置"""

    # 默认重试次数
    MAX_ATTEMPTS: int = 3

    # 指数退避乘数
    BACKOFF_MULTIPLIER: float = 2.0

    # 最大等待时间（秒）
    MAX_WAIT: float = 60.0

    # 初始等待时间（秒）
    INITIAL_WAIT: float = 1.0


# ==================== Token预算 ====================

class TokenBudget:
    """Token分配和预算相关常量"""

    # 提示词占上下文的比例（剩余留给输出）
    PROMPT_RATIO: float = 0.7

    # 保留的安全边距tokens
    SAFETY_MARGIN: int = 100

    # 最小可用输出tokens
    MIN_OUTPUT_TOKENS: int = 100


# ==================== 文本处理 ====================

class TextProcessing:
    """文本处理相关常量"""

    # Token估算比例（字符数/3作为近似）
    CHARS_PER_TOKEN: int = 3

    # 最大chunk tokens
    MAX_CHUNK_TOKENS: int = 4000

    # Chunk重叠字符数
    OVERLAP_CHARS: int = 200

    # 每个section最多chunks数
    MAX_CHUNKS_PER_SECTION: int = 10

    # Section最小分配字符数
    MIN_ALLOCATED_CHARS_FOR_SECTION: int = 500


# ==================== 缓存配置 ====================

class CacheConfig:
    """缓存相关配置"""

    # 研究缓存TTL（秒）
    RESEARCH_CACHE_TTL: int = 3600  # 1小时

    # 研究缓存最大条目数
    RESEARCH_CACHE_MAX_SIZE: int = 32

    # 网络缓存TTL（秒）
    NETWORK_CACHE_TTL: int = 7200  # 2小时

    # 网络缓存最大条目数
    NETWORK_CACHE_MAX_SIZE: int = 128


# ==================== JSON修复 ====================

class JSONRepairConfig:
    """JSON修复相关常量"""

    # 修复策略的最大迭代次数
    MAX_REPAIR_ITERATIONS: int = 5

    # 截断噪声的阈值（字符数）
    TRUNCATE_NOISE_THRESHOLD: int = 100

    # 最大括号不平衡差异
    MAX_BRACKET_IMBALANCE: int = 10


# ==================== 向量检索 ====================

class VectorRetrievalConfig:
    """向量检索相关配置"""

    # 默认检索数量
    DEFAULT_NUM_RESULTS: int = 5

    # 嵌入批次大小
    EMBEDDING_BATCH_SIZE: int = 32

    # BM25权重
    BM25_WEIGHT: float = 0.3

    # 搜索结果多样性阈值
    SEARCH_DIVERSITY_THRESHOLD: float = 0.7


# ==================== 工作流配置 ====================

class WorkflowConfig:
    """工作流相关常量"""

    # 默认最大迭代轮数
    DEFAULT_MAX_ITERATIONS: int = 3

    # 并发任务数
    MAX_CONCURRENT_TASKS: int = 5

    # 初始方案目标字符数
    INITIAL_SOLUTION_TARGET_CHARS: int = 5000


# ==================== HTTP/网络 ====================

class NetworkConfig:
    """网络请求相关配置"""

    # 默认超时（秒）
    DEFAULT_TIMEOUT: int = 30

    # 最大重试次数
    MAX_RETRIES: int = 3

    # 每主机RPS限制
    PER_HOST_RPS: float = 1.0

    # User-Agent
    DEFAULT_USER_AGENT: str = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    )


# ==================== 导出便捷访问 ====================

# 为了向后兼容，导出常用常量
MODEL_CONTEXT_LIMITS = ModelLimits.CONTEXT_LIMITS
MODEL_MAX_OUTPUT_TOKENS = ModelLimits.MAX_OUTPUT_TOKENS
REASONER_MIN_TOKENS = ModelLimits.REASONER_MIN_TOKENS

API_RETRY_MAX_ATTEMPTS = RetryConfig.MAX_ATTEMPTS
API_RETRY_BACKOFF_MULTIPLIER = RetryConfig.BACKOFF_MULTIPLIER
API_RETRY_MAX_WAIT = RetryConfig.MAX_WAIT

PROMPT_BUDGET_RATIO = TokenBudget.PROMPT_RATIO


__all__ = [
    "ModelLimits",
    "RetryConfig",
    "TokenBudget",
    "TextProcessing",
    "CacheConfig",
    "JSONRepairConfig",
    "VectorRetrievalConfig",
    "WorkflowConfig",
    "NetworkConfig",
    # 向后兼容导出
    "MODEL_CONTEXT_LIMITS",
    "MODEL_MAX_OUTPUT_TOKENS",
    "REASONER_MIN_TOKENS",
    "API_RETRY_MAX_ATTEMPTS",
    "API_RETRY_BACKOFF_MULTIPLIER",
    "API_RETRY_MAX_WAIT",
    "PROMPT_BUDGET_RATIO",
]

