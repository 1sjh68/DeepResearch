"""
统一缓存管理模块

提供通用的缓存接口，支持内存、磁盘和Redis后端。
"""

import copy
import logging
import time
from collections.abc import Callable
from hashlib import sha256
from threading import Lock
from typing import Any

logger = logging.getLogger(__name__)

# 尝试导入可选依赖
try:
    from diskcache import Cache as DiskCache
    DISKCACHE_AVAILABLE = True
except ImportError:
    DISKCACHE_AVAILABLE = False
    DiskCache = None  # type: ignore

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    redis = None  # type: ignore
    REDIS_AVAILABLE = False


class CacheManager:
    """
    通用缓存管理器

    支持：
    - 内存缓存（默认，线程安全）
    - 磁盘缓存（可选，使用diskcache）
    - Redis缓存（可选）
    - TTL过期
    - LRU淘汰策略
    """

    def __init__(
        self,
        *,
        ttl: int = 3600,
        max_size: int = 1000,
        namespace: str = "cache",
        backend: str = "memory",
        disk_cache_dir: str | None = None,
        redis_url: str | None = None,
    ):
        """
        初始化缓存管理器

        Args:
            ttl: 缓存过期时间（秒），0表示永不过期
            max_size: 最大缓存条目数（仅内存缓存有效）
            namespace: 缓存命名空间
            backend: 后端类型 ("memory", "disk", "redis")
            disk_cache_dir: 磁盘缓存目录
            redis_url: Redis连接URL
        """
        self.ttl = ttl
        self.max_size = max_size
        self.namespace = namespace
        self.backend = backend

        # 内存存储和锁
        self._memory_store: dict[str, tuple[float, Any]] = {}
        self._access_order: list[str] = []  # 用于LRU
        self._lock = Lock()

        # 磁盘缓存
        self._disk_cache = None
        if backend == "disk" and DISKCACHE_AVAILABLE and disk_cache_dir:
            try:
                self._disk_cache = DiskCache(
                    disk_cache_dir,
                    eviction_policy='least-recently-used',
                    size_limit=500 * 1024 * 1024,  # 500MB
                )
                logger.info(f"Cache '{namespace}' initialized with disk backend at {disk_cache_dir}")
            except Exception as exc:
                logger.warning(f"Failed to initialize disk cache: {exc}, falling back to memory")
                self.backend = "memory"

        # Redis缓存
        self._redis = None
        if backend == "redis" and REDIS_AVAILABLE and redis_url:
            try:
                self._redis = redis.from_url(redis_url)
                self._redis.ping()  # 测试连接
                logger.info(f"Cache '{namespace}' initialized with Redis backend")
            except Exception as exc:
                logger.warning(f"Failed to initialize Redis cache: {exc}, falling back to memory")
                self.backend = "memory"

    def _make_key(self, key: str) -> str:
        """生成带命名空间的缓存键"""
        if self.backend in ("disk", "redis"):
            # 对于磁盘和Redis，使用哈希避免键名过长
            hashed = sha256(key.encode('utf-8')).hexdigest()
            return f"{self.namespace}:{hashed}"
        return f"{self.namespace}:{key}"

    def get(self, key: str, default: Any = None) -> Any:
        """
        获取缓存值

        Args:
            key: 缓存键
            default: 默认值（未命中时返回）

        Returns:
            缓存的值或默认值
        """
        cache_key = self._make_key(key)

        # Redis后端
        if self.backend == "redis" and self._redis:
            try:
                import json
                payload = self._redis.get(cache_key)
                if payload:
                    return json.loads(payload)
            except Exception as exc:
                logger.debug(f"Redis get failed for key={key}: {exc}")
            return default

        # 磁盘后端
        if self.backend == "disk" and self._disk_cache:
            try:
                value = self._disk_cache.get(cache_key, default=None, retry=True)
                if value is not None:
                    return value
            except Exception as exc:
                logger.debug(f"Disk cache get failed for key={key}: {exc}")
            return default

        # 内存后端
        with self._lock:
            record = self._memory_store.get(cache_key)
            if not record:
                return default

            expires_at, value = record

            # 检查是否过期
            if self.ttl > 0 and expires_at < time.time():
                self._memory_store.pop(cache_key, None)
                if cache_key in self._access_order:
                    self._access_order.remove(cache_key)
                return default

            # 更新LRU顺序
            if cache_key in self._access_order:
                self._access_order.remove(cache_key)
            self._access_order.append(cache_key)

            # 深拷贝以避免外部修改
            return copy.deepcopy(value)

    def set(self, key: str, value: Any) -> None:
        """
        设置缓存值

        Args:
            key: 缓存键
            value: 要缓存的值
        """
        cache_key = self._make_key(key)

        # Redis后端
        if self.backend == "redis" and self._redis:
            try:
                import json
                if self.ttl > 0:
                    self._redis.setex(cache_key, self.ttl, json.dumps(value))
                else:
                    self._redis.set(cache_key, json.dumps(value))
                return
            except Exception as exc:
                logger.warning(f"Redis set failed for key={key}: {exc}")
                return

        # 磁盘后端
        if self.backend == "disk" and self._disk_cache:
            try:
                if self.ttl > 0:
                    self._disk_cache.set(cache_key, value, expire=self.ttl, retry=True)
                else:
                    self._disk_cache.set(cache_key, value, retry=True)
                return
            except Exception as exc:
                logger.warning(f"Disk cache set failed for key={key}: {exc}")
                return

        # 内存后端
        with self._lock:
            # 计算过期时间
            expires_at = time.time() + self.ttl if self.ttl > 0 else float('inf')

            # 深拷贝以避免外部修改
            self._memory_store[cache_key] = (expires_at, copy.deepcopy(value))

            # 更新LRU顺序
            if cache_key in self._access_order:
                self._access_order.remove(cache_key)
            self._access_order.append(cache_key)

            # LRU淘汰
            while len(self._memory_store) > self.max_size:
                if not self._access_order:
                    break
                oldest_key = self._access_order.pop(0)
                self._memory_store.pop(oldest_key, None)

    def delete(self, key: str) -> bool:
        """
        删除缓存值

        Args:
            key: 缓存键

        Returns:
            是否成功删除
        """
        cache_key = self._make_key(key)

        # Redis后端
        if self.backend == "redis" and self._redis:
            try:
                return bool(self._redis.delete(cache_key))
            except Exception as exc:
                logger.debug(f"Redis delete failed for key={key}: {exc}")
                return False

        # 磁盘后端
        if self.backend == "disk" and self._disk_cache:
            try:
                return self._disk_cache.delete(cache_key, retry=True)
            except Exception as exc:
                logger.debug(f"Disk cache delete failed for key={key}: {exc}")
                return False

        # 内存后端
        with self._lock:
            if cache_key in self._memory_store:
                del self._memory_store[cache_key]
                if cache_key in self._access_order:
                    self._access_order.remove(cache_key)
                return True
            return False

    def clear(self) -> None:
        """清空所有缓存"""
        # Redis后端
        if self.backend == "redis" and self._redis:
            try:
                # 删除所有带命名空间前缀的键
                pattern = f"{self.namespace}:*"
                for key in self._redis.scan_iter(match=pattern):
                    self._redis.delete(key)
                logger.info(f"Cleared Redis cache for namespace '{self.namespace}'")
                return
            except Exception as exc:
                logger.warning(f"Redis clear failed: {exc}")
                return

        # 磁盘后端
        if self.backend == "disk" and self._disk_cache:
            try:
                self._disk_cache.clear(retry=True)
                logger.info(f"Cleared disk cache for namespace '{self.namespace}'")
                return
            except Exception as exc:
                logger.warning(f"Disk cache clear failed: {exc}")
                return

        # 内存后端
        with self._lock:
            self._memory_store.clear()
            self._access_order.clear()
            logger.debug(f"Cleared memory cache for namespace '{self.namespace}'")

    def stats(self) -> dict[str, Any]:
        """
        获取缓存统计信息

        Returns:
            统计信息字典
        """
        stats = {
            "namespace": self.namespace,
            "backend": self.backend,
            "ttl": self.ttl,
        }

        # 内存后端统计
        if self.backend == "memory":
            with self._lock:
                stats.update({
                    "size": len(self._memory_store),
                    "max_size": self.max_size,
                    "utilization": len(self._memory_store) / self.max_size if self.max_size > 0 else 0,
                })

        # 磁盘后端统计
        elif self.backend == "disk" and self._disk_cache:
            try:
                stats.update({
                    "size": len(self._disk_cache),
                    "volume": self._disk_cache.volume(),
                })
            except Exception:
                pass

        # Redis后端统计
        elif self.backend == "redis" and self._redis:
            try:
                pattern = f"{self.namespace}:*"
                keys = list(self._redis.scan_iter(match=pattern, count=100))
                stats.update({
                    "size": len(keys),
                })
            except Exception:
                pass

        return stats

    def get_or_compute(self, key: str, compute_fn: Callable[[], Any]) -> Any:
        """
        获取缓存值，如果不存在则计算并缓存

        Args:
            key: 缓存键
            compute_fn: 计算函数（无参数）

        Returns:
            缓存的或新计算的值
        """
        # 先尝试获取
        value = self.get(key)
        if value is not None:
            return value

        # 计算新值
        value = compute_fn()

        # 缓存结果
        if value is not None:
            self.set(key, value)

        return value


# 创建全局缓存实例（单例模式）
_cache_instances: dict[str, CacheManager] = {}
_cache_instances_lock = Lock()


def get_cache(
    namespace: str,
    ttl: int = 3600,
    max_size: int = 1000,
    backend: str = "memory",
    **kwargs
) -> CacheManager:
    """
    获取或创建缓存实例（单例）

    Args:
        namespace: 缓存命名空间
        ttl: 过期时间
        max_size: 最大条目数
        backend: 后端类型
        **kwargs: 其他参数传递给CacheManager

    Returns:
        CacheManager实例
    """
    with _cache_instances_lock:
        if namespace not in _cache_instances:
            _cache_instances[namespace] = CacheManager(
                ttl=ttl,
                max_size=max_size,
                namespace=namespace,
                backend=backend,
                **kwargs
            )
        return _cache_instances[namespace]


__all__ = [
    "CacheManager",
    "get_cache",
]

