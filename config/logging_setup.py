from __future__ import annotations

import logging
import os
import sys
from datetime import datetime
from typing import TYPE_CHECKING

# 使用 loguru 替代标准 logging
try:
    from loguru import logger  # type: ignore[import-untyped]
    LOGURU_AVAILABLE = True
except ImportError:
    # 回退到标准 logging
    logger = logging.getLogger(__name__)  # type: ignore[assignment]
    LOGURU_AVAILABLE = False

if TYPE_CHECKING:
    from .config import Config  # pragma: no cover


def _resolve_console_level(default: str = "WARNING") -> str:
    """确定控制台日志级别"""
    explicit_level = os.getenv("CONSOLE_LOG_LEVEL")
    if explicit_level:
        return explicit_level.strip().upper()

    compact = os.getenv("COMPACT_CONSOLE_PROGRESS", "true").strip().lower()
    if compact in {"1", "true", "yes", "on"}:
        return default
    else:
        return "INFO"


def setup_session_logging(config: Config, logging_level: str | int = "INFO") -> None:
    """为当前会话配置日志输出（使用 loguru，自动拦截标准 logging）"""
    # 将 int 级别转换为 str（兼容标准 logging.INFO 等常量）
    if isinstance(logging_level, int):
        import logging as std_logging
        level_name = std_logging.getLevelName(logging_level)
        if isinstance(level_name, str) and not level_name.startswith("Level"):
            logging_level = level_name
        else:
            logging_level = "INFO"
    now = datetime.now()
    session_timestamp = now.strftime("%Y%m%d_%H%M%S")
    config.session_dir = os.path.join(config.session_base_dir, f"session_{session_timestamp}")
    os.makedirs(config.session_dir, exist_ok=True)
    config.log_file_path = os.path.join(config.session_dir, "session.log")
    json_log_path = os.path.join(config.session_dir, "diagnostics.jsonl")

    if LOGURU_AVAILABLE:
        # 使用 loguru 配置日志
        logger.remove()  # type: ignore[attr-defined]

        # 配置 loguru 拦截标准 logging 调用
        # 这样所有现有代码无需修改就能使用 loguru
        import logging as std_logging

        class InterceptHandler(std_logging.Handler):
            def emit(self, record: std_logging.LogRecord) -> None:
                # 获取对应的 loguru 级别
                try:
                    level = logger.level(record.levelname).name  # type: ignore[attr-defined]
                except ValueError:
                    level = record.levelno

                # 找到调用者
                frame, depth = sys._getframe(6), 6
                while frame and frame.f_code.co_filename == std_logging.__file__:
                    frame = frame.f_back
                    depth += 1

                logger.opt(depth=depth, exception=record.exc_info).log(  # type: ignore[attr-defined]
                    level, record.getMessage()
                )

        # 配置标准 logging 使用拦截器
        std_logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)

        # 控制台日志（非 JSON，带颜色）
        console_level = _resolve_console_level()
        logger.add(  # type: ignore[attr-defined]
            sys.stdout,
            format="<level>{message}</level>",
            level=console_level,
            filter=lambda record: not (
                str(record["message"]).startswith("{") and str(record["message"]).endswith("}")
            ),
            colorize=True,
        )

        # 文件日志（所有日志，传统格式）
        # 改进的日志轮转：10MB大小限制，每天轮转，保留7个备份，自动压缩
        logger.add(  # type: ignore[attr-defined]
            config.log_file_path,
            format="{time:YYYY-MM-DD HH:mm:ss} - {level} - [{file}:{line}] - {message}",
            level=logging_level,
            encoding="utf-8",
            rotation="10 MB",  # 每10MB轮转一次
            retention="7 days",  # 保留7天的日志
            compression="zip",  # 压缩旧日志节省空间
            enqueue=True,  # 异步写入，提高性能
        )

        # JSON 诊断日志
        logger.add(  # type: ignore[attr-defined]
            json_log_path,
            format="{message}",
            level=logging_level,
            filter=lambda record: (
                str(record["message"]).startswith("{") and str(record["message"]).endswith("}")
            ),
            encoding="utf-8",
            serialize=False,
            rotation="5 MB",  # JSON日志也添加轮转
            retention="3 days",
            compression="zip",
            enqueue=True,
        )

        # 静默第三方库日志
        for noisy_logger in ("httpx", "httpcore", "openai", "urllib3"):
            std_logging.getLogger(noisy_logger).setLevel(std_logging.WARNING)

        logger.info(
            "日志记录已初始化（loguru + 标准 logging 拦截）。会话目录: %s",
            config.session_dir,
        )
        logger.info(f"📝 日志文件: {config.log_file_path}")
        logger.info(f"📊 JSON 日志: {json_log_path}")
        logger.info("🔄 日志轮转: 10MB/文件, 保留7天, 自动压缩")

    else:
        # 回退到标准 logging（保持向后兼容）
        import logging

        root_logger = logging.getLogger()
        if root_logger.hasHandlers():
            for handler in root_logger.handlers[:]:
                root_logger.removeHandler(handler)
                handler.close()

        level_int = getattr(logging, logging_level, logging.INFO)
        root_logger.setLevel(level_int)

        # 控制台处理器
        console_handler = logging.StreamHandler(sys.stdout)
        console_level_str = _resolve_console_level()
        console_handler.setLevel(getattr(logging, console_level_str, logging.WARNING))
        console_handler.setFormatter(logging.Formatter("%(message)s"))
        root_logger.addHandler(console_handler)

        # 文件处理器（使用 RotatingFileHandler 实现日志轮转）
        from logging.handlers import RotatingFileHandler
        file_handler = RotatingFileHandler(
            config.log_file_path,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=7,  # 保留7个备份
            encoding="utf-8"
        )
        file_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s"
            )
        )
        root_logger.addHandler(file_handler)

        # JSON 日志处理器（也使用轮转）
        from logging.handlers import RotatingFileHandler
        json_handler = RotatingFileHandler(
            json_log_path,
            maxBytes=5*1024*1024,  # 5MB
            backupCount=3,
            encoding="utf-8"
        )
        json_handler.setFormatter(logging.Formatter("%(message)s"))
        root_logger.addHandler(json_handler)

        logging.info(f"📝 日志记录已初始化（标准 logging）。会话目录: {config.session_dir}")
        logging.info("🔄 日志轮转: 10MB/文件, 保留7个备份")


__all__ = ["setup_session_logging", "logger"]
