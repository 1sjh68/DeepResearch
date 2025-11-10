# utils/error_handler.py

import json
import logging
from collections.abc import Callable
from contextlib import contextmanager
from enum import Enum
from typing import Any


class ErrorSeverity(Enum):
    """错误严重程度"""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ErrorType(Enum):
    """错误类型"""

    API_TIMEOUT = "api_timeout"
    API_ERROR = "api_error"
    EMPTY_RESPONSE = "empty_response"
    TOKEN_LIMIT = "token_limit"
    NETWORK_ERROR = "network_error"
    VALIDATION_ERROR = "validation_error"
    CONTENT_FILTER = "content_filter"
    RATE_LIMIT = "rate_limit"
    UNKNOWN = "unknown"


class FriendlyError:
    """友好的错误信息"""

    def __init__(
        self,
        error_type: ErrorType,
        title: str,
        message: str,
        suggestions: list[str],
        severity: ErrorSeverity = ErrorSeverity.ERROR,
        recoverable: bool = True,
        technical_details: str | None = None,
    ):
        self.error_type = error_type
        self.title = title
        self.message = message
        self.suggestions = suggestions
        self.severity = severity
        self.recoverable = recoverable
        self.technical_details = technical_details

    def to_dict(self) -> dict:
        return {
            "type": "error",
            "error_type": self.error_type.value,
            "title": self.title,
            "message": self.message,
            "suggestions": self.suggestions,
            "severity": self.severity.value,
            "recoverable": self.recoverable,
            "technical_details": self.technical_details,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)


class ErrorHandler:
    """统一错误处理和友好提示"""

    # 错误模板定义
    ERROR_TEMPLATES = {
        ErrorType.API_TIMEOUT: {
            "title": "API 响应超时",
            "message": "模型响应时间过长，可能是网络问题或服务繁忙",
            "suggestions": [
                "系统已自动重试",
                "如持续失败，请稍后再试",
                "检查网络连接",
                "尝试减少目标长度或简化提示词",
            ],
            "severity": ErrorSeverity.WARNING,
            "recoverable": True,
        },
        ErrorType.EMPTY_RESPONSE: {
            "title": "模型返回空内容",
            "message": "AI 模型未能生成有效内容，这可能是由于提示词或输入内容的问题",
            "suggestions": [
                "系统已自动切换到备用模型",
                "如果持续失败，请尝试简化您的提示词",
                "检查是否包含敏感内容触发了过滤",
            ],
            "severity": ErrorSeverity.WARNING,
            "recoverable": True,
        },
        ErrorType.TOKEN_LIMIT: {
            "title": "内容长度超限",
            "message": "输入或输出内容超过了模型的最大处理限制",
            "suggestions": [
                "减少外部资料的长度（当前建议：<50,000字）",
                "降低目标字数设置",
                "使用更简洁的提示词",
                "移除不必要的上下文信息",
            ],
            "severity": ErrorSeverity.ERROR,
            "recoverable": True,
        },
        ErrorType.NETWORK_ERROR: {
            "title": "网络连接错误",
            "message": "无法连接到 API 服务器",
            "suggestions": ["检查网络连接", "确认 API 服务是否正常", "检查防火墙设置", "稍后重试"],
            "severity": ErrorSeverity.ERROR,
            "recoverable": True,
        },
        ErrorType.VALIDATION_ERROR: {
            "title": "数据验证失败",
            "message": "AI 返回的数据格式不符合预期",
            "suggestions": [
                "系统正在自动修复",
                "如持续失败，可能需要调整提示词",
                "这通常是暂时性问题，请重试",
            ],
            "severity": ErrorSeverity.WARNING,
            "recoverable": True,
        },
        ErrorType.CONTENT_FILTER: {
            "title": "内容被过滤",
            "message": "输入或输出内容触发了安全过滤规则",
            "suggestions": [
                "检查输入内容是否包含敏感信息",
                "尝试重新表述您的问题",
                "避免使用可能被误判的词汇",
            ],
            "severity": ErrorSeverity.WARNING,
            "recoverable": True,
        },
        ErrorType.RATE_LIMIT: {
            "title": "API 调用频率限制",
            "message": "请求过于频繁，已达到 API 速率限制",
            "suggestions": ["系统将自动等待后重试", "如果是多个任务并发，建议分批执行"],
            "severity": ErrorSeverity.WARNING,
            "recoverable": True,
        },
        ErrorType.API_ERROR: {
            "title": "API 服务错误",
            "message": "API 服务返回错误",
            "suggestions": ["系统正在重试", "如持续失败，可能是 API 服务问题", "请稍后再试"],
            "severity": ErrorSeverity.ERROR,
            "recoverable": True,
        },
        ErrorType.UNKNOWN: {
            "title": "未知错误",
            "message": "发生了意外错误",
            "suggestions": ["请查看详细日志", "如问题持续，请联系技术支持", "尝试重新启动任务"],
            "severity": ErrorSeverity.ERROR,
            "recoverable": True,
        },
    }

    @classmethod
    def handle_error(
        cls,
        error_type: ErrorType,
        context: dict | None = None,
        technical_details: str | None = None,
    ) -> FriendlyError:
        """处理错误并返回友好的错误信息"""
        template = cls.ERROR_TEMPLATES.get(error_type, cls.ERROR_TEMPLATES[ErrorType.UNKNOWN])

        # 可以根据 context 动态调整消息
        message = template["message"]
        suggestions = template["suggestions"].copy()

        if context:
            # 根据上下文添加额外建议
            if context.get("retry_count", 0) > 2:
                suggestions.append("已重试多次，建议检查配置或稍后再试")

            if context.get("input_tokens", 0) > 50000:
                suggestions.append("输入内容过长，建议减少外部资料")

        error = FriendlyError(
            error_type=error_type,
            title=template["title"],
            message=message,
            suggestions=suggestions,
            severity=template["severity"],
            recoverable=template["recoverable"],
            technical_details=technical_details,
        )

        # 记录到日志
        cls._log_error(error)

        return error

    @classmethod
    def _log_error(cls, error: FriendlyError):
        """记录错误到日志"""
        log_data = error.to_dict()
        log_level = {
            ErrorSeverity.INFO: logging.INFO,
            ErrorSeverity.WARNING: logging.WARNING,
            ErrorSeverity.ERROR: logging.ERROR,
            ErrorSeverity.CRITICAL: logging.CRITICAL,
        }.get(error.severity, logging.ERROR)

        logging.log(log_level, json.dumps(log_data, ensure_ascii=False))

    @classmethod
    def detect_error_type(cls, exception: Exception) -> ErrorType:
        """从异常对象检测错误类型"""
        error_str = str(exception).lower()
        exception_type = type(exception).__name__

        if "timeout" in error_str or "timeout" in exception_type.lower():
            return ErrorType.API_TIMEOUT
        elif "empty" in error_str or "no content" in error_str:
            return ErrorType.EMPTY_RESPONSE
        elif "token" in error_str or "length" in error_str or "limit" in error_str:
            return ErrorType.TOKEN_LIMIT
        elif "network" in error_str or "connection" in error_str:
            return ErrorType.NETWORK_ERROR
        elif "validation" in error_str or "invalid" in error_str:
            return ErrorType.VALIDATION_ERROR
        elif "filter" in error_str or "safety" in error_str:
            return ErrorType.CONTENT_FILTER
        elif "rate" in error_str or "429" in error_str:
            return ErrorType.RATE_LIMIT
        elif "api" in error_str and ("error" in error_str or "fail" in error_str):
            return ErrorType.API_ERROR
        else:
            return ErrorType.UNKNOWN

    @classmethod
    def from_exception(cls, exception: Exception, context: dict | None = None) -> FriendlyError:
        """从异常创建友好错误"""
        error_type = cls.detect_error_type(exception)
        technical_details = f"{type(exception).__name__}: {str(exception)}"

        return cls.handle_error(error_type, context, technical_details)

    @classmethod
    def repair_json_once(cls, text: str, schema=None) -> tuple[str, bool]:
        """
        兼容旧调用入口：委托核心 JSON 修复实现（services.llm_interaction.repair_json_once）。
        这样可以避免两套实现产生分歧，统一修复策略与日志。
        """
        try:
            from utils.json_repair import repair_json_once as _core_repair  # 延迟导入以避免循环依赖
            if schema is None:
                # 无 schema 时退化为字典结构尝试
                schema = object  # 类型占位，不影响核心实现流程
            return _core_repair(text, schema)
        except Exception as exc:
            logging.warning("ErrorHandler.repair_json_once 委托失败: %s", exc, exc_info=True)
            # 最后兜底：原样返回，标记失败
            return text, False


# ==================== 上下文管理器 ====================

@contextmanager
def handle_api_errors(
    context: str,
    *,
    fallback: Any = None,
    log_errors: bool = True,
    reraise: bool = False
):
    """
    统一的API错误处理上下文管理器

    使用示例:
        with handle_api_errors("调用模型生成大纲", fallback=""):
            result = call_ai(...)

    Args:
        context: 操作上下文描述
        fallback: 发生错误时返回的备用值
        log_errors: 是否记录错误日志
        reraise: 是否重新抛出异常

    Yields:
        None
    """
    try:
        yield
    except Exception as exc:
        if log_errors:
            error = ErrorHandler.from_exception(exc)
            logging.error(f"[{context}] {error.title}: {error.message}")
            for suggestion in error.suggestions:
                logging.info(f"  建议: {suggestion}")

        if reraise:
            raise

        return fallback


@contextmanager
def suppress_errors(
    *error_types,
    log_warning: bool = True,
    context: str | None = None
):
    """
    抑制特定类型的异常

    使用示例:
        with suppress_errors(FileNotFoundError, KeyError, context="加载配置"):
            config = load_optional_config()

    Args:
        *error_types: 要抑制的异常类型
        log_warning: 是否记录警告日志
        context: 操作上下文描述

    Yields:
        None
    """
    try:
        yield
    except error_types as exc:
        if log_warning:
            ctx_msg = f"[{context}] " if context else ""
            logging.warning(f"{ctx_msg}抑制异常: {type(exc).__name__}: {exc}")


def log_and_return(
    error: Exception,
    message: str,
    default: Any = None,
    *,
    log_level: int = logging.ERROR,
    include_traceback: bool = False
) -> Any:
    """
    标准化的日志+返回模式

    使用示例:
        try:
            return process_data()
        except Exception as e:
            return log_and_return(e, "处理数据失败", default=[])

    Args:
        error: 异常对象
        message: 日志消息
        default: 返回的默认值
        log_level: 日志级别
        include_traceback: 是否包含完整堆栈跟踪

    Returns:
        default值
    """
    if include_traceback:
        logging.log(log_level, f"{message}: {error}", exc_info=True)
    else:
        logging.log(log_level, f"{message}: {type(error).__name__}: {error}")

    return default


def safe_execute(
    func: Callable[[], Any],
    *,
    fallback: Any = None,
    context: str | None = None,
    log_errors: bool = True
) -> Any:
    """
    安全执行函数，捕获所有异常

    使用示例:
        result = safe_execute(
            lambda: expensive_operation(),
            fallback={},
            context="执行耗时操作"
        )

    Args:
        func: 要执行的函数（无参数）
        fallback: 失败时的备用值
        context: 操作上下文描述
        log_errors: 是否记录错误

    Returns:
        函数执行结果或fallback
    """
    try:
        return func()
    except Exception as exc:
        if log_errors:
            ctx_msg = f"[{context}] " if context else ""
            logging.error(f"{ctx_msg}执行失败: {type(exc).__name__}: {exc}")
        return fallback
