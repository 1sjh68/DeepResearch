"""
进度条工具模块
提供统一的进度显示功能，支持文件处理、API调用等场景

注意：本模块已弃用，请使用 utils.progress_tracker 中的 rich 进度条
为了向后兼容，保留简化版本，仅使用日志输出
"""

import logging
from collections.abc import Iterable
from typing import Any

logger = logging.getLogger(__name__)

# 不再使用 tqdm，避免与 rich 进度条冲突
TQDM_AVAILABLE = False


class ProgressBar:
    """简化的进度条包装器，仅使用日志输出（避免与 rich 进度条冲突）"""

    def __init__(
        self,
        iterable: Iterable[Any] | None = None,
        total: int | None = None,
        desc: str = "Processing",
        unit: str = "item",
        disable: bool = False,
        leave: bool = True,
    ):
        self.iterable = iterable
        self.total = total
        self.desc = desc
        self.unit = unit
        self.disable = disable
        self.leave = leave
        self._count = 0
        self._last_logged_percent = -1

        # 仅使用日志输出
        if not disable and total:
            logger.debug(f"开始 {desc}: 共 {total} {unit}")
    
    def __iter__(self):
        """迭代器支持 - 静默处理，仅在开始和结束时记录"""
        if self.iterable:
            for i, item in enumerate(self.iterable, 1):
                self._count = i
                # 仅在特定百分比时记录（减少日志噪音）
                if not self.disable and self.total and self.total > 0:
                    percent = int((i / self.total) * 100)
                    # 每25%记录一次
                    if percent >= self._last_logged_percent + 25:
                        logger.debug(f"{self.desc}: {i}/{self.total} ({percent}%)")
                        self._last_logged_percent = percent
                yield item
        return iter([])
    
    def update(self, n: int = 1):
        """手动更新进度 - 静默处理"""
        self._count += n
        # 不输出日志，避免干扰 rich 进度条
    
    def set_description(self, desc: str):
        """更新描述"""
        self.desc = desc

    def set_postfix(self, **kwargs):
        """设置后缀信息 - 无操作"""
        pass

    def close(self):
        """关闭进度条"""
        if not self.disable and self.total:
            logger.debug(f"{self.desc} 完成: {self._count}/{self.total} {self.unit}")
    
    def __enter__(self):
        """上下文管理器支持"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """退出时自动关闭"""
        self.close()


def create_progress_bar(
    iterable: Iterable[Any] | None = None,
    total: int | None = None,
    desc: str = "Processing",
    unit: str = "item",
    disable: bool = False,
) -> ProgressBar:
    """
    创建进度条的便捷函数
    
    Args:
        iterable: 可迭代对象
        total: 总数
        desc: 描述文本
        unit: 单位名称
        disable: 是否禁用进度条
    
    Returns:
        ProgressBar 实例
    
    Example:
        >>> for item in create_progress_bar(items, desc="处理文件"):
        >>>     process(item)
    """
    return ProgressBar(
        iterable=iterable,
        total=total,
        desc=desc,
        unit=unit,
        disable=disable,
    )


__all__ = ["ProgressBar", "create_progress_bar", "TQDM_AVAILABLE"]
