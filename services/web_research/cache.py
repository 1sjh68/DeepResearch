from __future__ import annotations

import json
import logging
import time
from hashlib import sha256
from typing import Any

# 持久化缓存库
try:
    from diskcache import Cache as DiskCache
    DISKCACHE_AVAILABLE = True
except ImportError:
    DISKCACHE_AVAILABLE = False
    logging.debug("diskcache not installed. Persistent cache will not be available.")

try:  # Optional dependency
    import redis  # type: ignore
except ImportError:  # pragma: no cover - optional
    redis = None

logger = logging.getLogger(__name__)


class ResponseCache:
    """TTL cache with persistent disk storage (diskcache) or optional Redis backing."""

    def __init__(
        self,
        *,
        ttl_seconds: int = 900,
        namespace: str = "web_research",
        redis_url: str | None = None,
        cache_dir: str = "./cache",
    ):
        self.ttl = ttl_seconds
        self.namespace = namespace
        self._memory_store: dict[str, tuple[float, Any]] = {}
        self._disk_cache = None
        self._redis = None

        # 优先使用diskcache（持久化）
        if DISKCACHE_AVAILABLE and not redis_url:
            try:
                self._disk_cache = DiskCache(
                    cache_dir,
                    eviction_policy='least-recently-used',
                    size_limit=500 * 1024 * 1024,  # 500MB
                )
                logger.info(f"Response cache configured with diskcache backend at {cache_dir}")
            except Exception as exc:
                logger.warning("Failed to initialize diskcache: %s", exc)

        # 如果指定了redis_url，使用Redis
        if redis_url and redis:
            try:
                self._redis = redis.from_url(redis_url)
                logger.info("Response cache configured with Redis backend.")
            except Exception as exc:  # pragma: no cover - redis optional
                logger.warning("Failed to initialize Redis cache: %s", exc)

    def _key(self, *parts: str) -> str:
        hashed = sha256("::".join(parts).encode("utf-8")).hexdigest()
        return f"{self.namespace}:{hashed}"

    def get(self, *parts: str) -> Any | None:
        key = self._key(*parts)

        # Redis优先（如果配置）
        if self._redis:
            payload = self._redis.get(key)
            if payload:
                try:
                    return json.loads(payload)
                except json.JSONDecodeError:
                    logger.debug("Redis cache payload decode failed for key=%s", key)
                    return None
            return None

        # diskcache次之（持久化）
        if self._disk_cache:
            try:
                value = self._disk_cache.get(key, default=None, retry=True)
                return value
            except Exception as exc:
                logger.debug("diskcache get failed for key=%s: %s", key, exc)
                return None

        # 内存缓存兜底
        record = self._memory_store.get(key)
        if not record:
            return None
        expires_at, value = record
        if expires_at < time.time():
            self._memory_store.pop(key, None)
            return None
        return value

    def set(self, value: Any, *parts: str) -> None:
        key = self._key(*parts)
        expiry = time.time() + self.ttl

        # Redis优先
        if self._redis:
            try:
                self._redis.setex(key, self.ttl, json.dumps(value))
                return
            except Exception as exc:  # pragma: no cover
                logger.warning("Redis cache set failed for key=%s: %s", key, exc)

        # diskcache持久化
        if self._disk_cache:
            try:
                self._disk_cache.set(key, value, expire=self.ttl, retry=True)
                return
            except Exception as exc:
                logger.warning("diskcache set failed for key=%s: %s", key, exc)

        # 内存缓存兜底
        self._memory_store[key] = (expiry, value)
