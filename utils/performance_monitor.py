"""
性能监控和基准测试模块

提供装饰器和工具来监控函数执行时间、内存使用等。
"""

import functools
import logging
import time
from collections.abc import Callable
from contextlib import contextmanager
from typing import Any

logger = logging.getLogger(__name__)

# 性能统计存储
_performance_stats: dict[str, dict[str, Any]] = {}


def track_time(func: Callable | None = None, *, name: str | None = None):
    """
    函数执行时间追踪装饰器

    使用示例:
        @track_time
        def expensive_function():
            ...

        @track_time(name="自定义名称")
        def another_function():
            ...

    Args:
        func: 被装饰的函数
        name: 自定义追踪名称

    Returns:
        装饰后的函数
    """
    def decorator(f: Callable) -> Callable:
        func_name = name or f.__name__

        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            try:
                result = f(*args, **kwargs)
                return result
            finally:
                duration = time.perf_counter() - start_time

                # 更新统计
                if func_name not in _performance_stats:
                    _performance_stats[func_name] = {
                        "total_calls": 0,
                        "total_time": 0.0,
                        "min_time": float('inf'),
                        "max_time": 0.0,
                    }

                stats = _performance_stats[func_name]
                stats["total_calls"] += 1
                stats["total_time"] += duration
                stats["min_time"] = min(stats["min_time"], duration)
                stats["max_time"] = max(stats["max_time"], duration)
                stats["avg_time"] = stats["total_time"] / stats["total_calls"]

                logger.debug(
                    f"[Performance] {func_name} 执行时间: {duration:.3f}s "
                    f"(avg: {stats['avg_time']:.3f}s)"
                )

        return wrapper

    # 支持两种调用方式：@track_time 和 @track_time(name="...")
    if func is None:
        return decorator
    else:
        return decorator(func)


@contextmanager
def measure_time(operation: str):
    """
    代码块执行时间测量上下文管理器

    使用示例:
        with measure_time("数据处理"):
            process_data()

    Args:
        operation: 操作描述

    Yields:
        None
    """
    start_time = time.perf_counter()
    try:
        yield
    finally:
        duration = time.perf_counter() - start_time
        logger.info(f"[Performance] {operation} 耗时: {duration:.3f}s")


def get_performance_stats() -> dict[str, dict[str, Any]]:
    """
    获取所有性能统计数据

    Returns:
        性能统计字典
    """
    return dict(_performance_stats)


def reset_performance_stats() -> None:
    """重置所有性能统计"""
    _performance_stats.clear()
    logger.debug("Performance stats reset")


def print_performance_report() -> None:
    """打印性能报告"""
    if not _performance_stats:
        logger.info("No performance data collected")
        return

    logger.info("=" * 60)
    logger.info("性能监控报告")
    logger.info("=" * 60)

    # 按总时间排序
    sorted_stats = sorted(
        _performance_stats.items(),
        key=lambda x: x[1]["total_time"],
        reverse=True
    )

    for func_name, stats in sorted_stats:
        logger.info(f"\n函数: {func_name}")
        logger.info(f"  调用次数: {stats['total_calls']}")
        logger.info(f"  总耗时: {stats['total_time']:.3f}s")
        logger.info(f"  平均耗时: {stats['avg_time']:.3f}s")
        logger.info(f"  最小耗时: {stats['min_time']:.3f}s")
        logger.info(f"  最大耗时: {stats['max_time']:.3f}s")

    logger.info("=" * 60)


class PerformanceMonitor:
    """性能监控器类"""

    def __init__(self, enabled: bool = True):
        """
        初始化性能监控器

        Args:
            enabled: 是否启用监控
        """
        self.enabled = enabled
        self._checkpoints: dict[str, float] = {}

    def checkpoint(self, name: str) -> None:
        """
        设置性能检查点

        Args:
            name: 检查点名称
        """
        if not self.enabled:
            return
        self._checkpoints[name] = time.perf_counter()

    def measure_since(self, checkpoint: str, operation: str | None = None) -> float:
        """
        测量自检查点以来的时间

        Args:
            checkpoint: 检查点名称
            operation: 操作描述（用于日志）

        Returns:
            经过的时间（秒）
        """
        if not self.enabled or checkpoint not in self._checkpoints:
            return 0.0

        start_time = self._checkpoints[checkpoint]
        duration = time.perf_counter() - start_time

        if operation:
            logger.debug(f"[Performance] {operation} 耗时: {duration:.3f}s")

        return duration

    def clear(self) -> None:
        """清空所有检查点"""
        self._checkpoints.clear()


# 创建全局监控器实例
_global_monitor = PerformanceMonitor()


def get_monitor() -> PerformanceMonitor:
    """获取全局监控器实例"""
    return _global_monitor


def benchmark(iterations: int = 10):
    """
    基准测试装饰器

    使用示例:
        @benchmark(iterations=100)
        def test_function():
            ...

    Args:
        iterations: 运行次数

    Returns:
        装饰后的函数
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            times = []
            result = None

            for i in range(iterations):
                start = time.perf_counter()
                result = func(*args, **kwargs)
                duration = time.perf_counter() - start
                times.append(duration)

            avg_time = sum(times) / len(times)
            min_time = min(times)
            max_time = max(times)

            logger.info(
                f"[Benchmark] {func.__name__} "
                f"({iterations} 次运行): "
                f"平均={avg_time:.6f}s, "
                f"最小={min_time:.6f}s, "
                f"最大={max_time:.6f}s"
            )

            return result

        return wrapper

    return decorator


__all__ = [
    "track_time",
    "measure_time",
    "get_performance_stats",
    "reset_performance_stats",
    "print_performance_report",
    "PerformanceMonitor",
    "get_monitor",
    "benchmark",
]

