"""
LLM API 重试策略模块

提供统一的重试逻辑和异常类型管理。
"""

import logging
from typing import TYPE_CHECKING

import openai
import tenacity

if TYPE_CHECKING:
    from config import Config

logger = logging.getLogger(__name__)

# 尝试导入可选依赖
try:
    import httpx
except ImportError:
    httpx = None  # type: ignore

try:
    import httpcore
except ImportError:
    httpcore = None  # type: ignore

try:
    from instructor.exceptions import InstructorRetryException
    instructor_available = True
except Exception:
    InstructorRetryException = None  # type: ignore
    instructor_available = False


class EmptyResponseFromReasonerError(Exception):
    """当 Reasoner 模型在剥离 <RichMediaReference> 标签后返回空内容时抛出的自定义异常。"""
    pass


def build_retry_exception_types(include_reasoner_empty: bool = True) -> tuple[type[BaseException], ...]:
    """
    组合应该触发tenacity重试的异常类型元组。
    允许为推理器模型可选地包含EmptyResponseFromReasonerError。

    Args:
        include_reasoner_empty: 是否包含EmptyResponseFromReasonerError

    Returns:
        异常类型元组
    """
    exceptions: tuple[type[BaseException], ...] = (
        openai.APITimeoutError,
        openai.APIConnectionError,
        openai.InternalServerError,
        openai.RateLimitError,
        openai.NotFoundError,  # 添加404错误支持,处理模型端点暂时不可用的情况
    )
    if include_reasoner_empty:
        exceptions = exceptions + (EmptyResponseFromReasonerError,)
    if instructor_available and InstructorRetryException is not None:
        exceptions = exceptions + (InstructorRetryException,)  # type: ignore
    if httpx is not None:
        exceptions = exceptions + (
            httpx.RemoteProtocolError,
            httpx.ReadTimeout,
            httpx.ConnectError,
        )
    if httpcore is not None:
        exceptions = exceptions + (
            httpcore.RemoteProtocolError,
            httpcore.ConnectError,
        )
    return exceptions


def build_retryer(config: "Config", exception_types: tuple[type[BaseException], ...]) -> tenacity.Retrying:
    """
    创建具有一致策略的Tenacity重试器，用于同步调用。

    Args:
        config: 配置对象
        exception_types: 需要重试的异常类型元组

    Returns:
        Tenacity重试器实例
    """
    return tenacity.Retrying(
        wait=tenacity.wait_random_exponential(
            multiplier=config.runtime.api_retry_wait_multiplier,
            max=config.runtime.api_retry_max_wait,
        ),
        stop=tenacity.stop_after_attempt(config.runtime.api_retry_max_attempts),
        retry=tenacity.retry_if_exception_type(exception_types),
        before_sleep=tenacity.before_sleep_log(logging.getLogger(__name__), logging.WARNING),
        reraise=True,
    )


class RetryStrategy:
    """重试策略的面向对象封装"""

    def __init__(self, config: "Config"):
        """
        初始化重试策略

        Args:
            config: 配置对象
        """
        self.config = config

    def get_exception_types(self, include_reasoner_empty: bool = True) -> tuple[type[BaseException], ...]:
        """
        获取需要重试的异常类型

        Args:
            include_reasoner_empty: 是否包含EmptyResponseFromReasonerError

        Returns:
            异常类型元组
        """
        return build_retry_exception_types(include_reasoner_empty)

    def create_retryer(self, exception_types: tuple[type[BaseException], ...] | None = None) -> tenacity.Retrying:
        """
        创建重试器

        Args:
            exception_types: 异常类型元组，如果为None则使用默认类型

        Returns:
            Tenacity重试器实例
        """
        if exception_types is None:
            exception_types = self.get_exception_types()
        return build_retryer(self.config, exception_types)

