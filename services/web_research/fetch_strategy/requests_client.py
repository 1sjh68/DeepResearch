from __future__ import annotations

import importlib
import logging
import time
from collections import defaultdict
from urllib.parse import urlparse

import httpx

try:
    aiolimiter_mod = importlib.import_module("aiolimiter")
    AsyncLimiter = aiolimiter_mod.AsyncLimiter  # type: ignore[attr-defined]
except Exception:  # pragma: no cover - optional dependency missing

    class AsyncLimiter:  # type: ignore
        def __init__(self, *_args, **_kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False


from tenacity import AsyncRetrying, retry_if_exception_type, stop_after_attempt, wait_exponential

from services.web_research.models import FetchResponse

logger = logging.getLogger(__name__)


class RequestsFetchError(RuntimeError):
    """Raised when the httpx fetcher exhausts retries."""


class RequestsFetchStrategy:
    """Native async HTTP client using httpx with rate limits and retries."""

    def __init__(
        self,
        *,
        connect_timeout: float = 10.0,
        read_timeout: float = 20.0,
        max_retries: int = 3,
        global_rps: float = 8.0,
        per_domain_rps: float = 2.0,
        user_agent: str | None = None,
        proxies: dict[str, str] | None = None,
    ):
        self._connect_timeout = connect_timeout
        self._read_timeout = read_timeout
        self._max_retries = max(1, max_retries)

        # 使用 httpx.AsyncClient（原生异步，性能提升20-50%）
        headers = {"User-Agent": user_agent} if user_agent else {}

        # 构建客户端配置（兼容 httpx >= 0.24）
        client_kwargs = {
            "timeout": httpx.Timeout(
                connect=connect_timeout,
                read=read_timeout,
                write=read_timeout,  # 写入超时通常与读取相同
                pool=5.0,  # 连接池获取超时
            ),
            "limits": httpx.Limits(max_connections=100, max_keepalive_connections=20),
            "follow_redirects": True,
            "http2": True,  # 启用HTTP/2支持
            "headers": headers,
        }

        # 处理代理配置（适配 httpx >= 0.24 的新 API）
        if proxies:
            # 优先使用 https 代理，如果没有则使用 http 代理
            proxy_url = proxies.get("https") or proxies.get("http")
            if proxy_url:
                client_kwargs["proxy"] = proxy_url  # 新版 API 使用 proxy（单数）

        self._client = httpx.AsyncClient(**client_kwargs)

        self._global_limiter = AsyncLimiter(max(1, int(global_rps)), time_period=1)
        self._domain_limiters: dict[str, AsyncLimiter] = defaultdict(lambda: AsyncLimiter(max(1, int(per_domain_rps)), time_period=1))
        self._stats = {
            "total_requests": 0,
            "successful": 0,
            "failed": 0,
            "last_error": None,
        }

    async def fetch(self, url: str) -> FetchResponse:
        """原生异步fetch，无需asyncio.to_thread"""
        domain = urlparse(url).netloc.lower()
        limiter = self._domain_limiters[domain]
        async with self._global_limiter:
            async with limiter:
                return await self._fetch_async(url)

    async def close(self) -> None:
        """关闭httpx客户端"""
        await self._client.aclose()

    def stats(self) -> dict[str, str | None]:
        return {
            "total_requests": str(self._stats["total_requests"]),
            "successful": str(self._stats["successful"]),
            "failed": str(self._stats["failed"]),
            "last_error": self._stats["last_error"],
        }

    async def _fetch_async(self, url: str) -> FetchResponse:
        """原生异步fetch实现"""
        async for attempt in AsyncRetrying(
            reraise=True,
            stop=stop_after_attempt(self._max_retries),
            wait=wait_exponential(multiplier=1, min=1, max=8),
            retry=retry_if_exception_type((httpx.HTTPError,)),
        ):
            with attempt:
                return await self._attempt_fetch_async(url)

    async def _attempt_fetch_async(self, url: str) -> FetchResponse:
        """单次异步fetch尝试"""
        self._stats["total_requests"] += 1
        start = time.perf_counter()
        try:
            response = await self._client.get(url)
            elapsed = (time.perf_counter() - start) * 1000
            content = response.text if response.text else ""
            headers = {k: v for k, v in response.headers.items()}

            if 200 <= response.status_code < 400:
                self._stats["successful"] += 1
            else:
                self._stats["failed"] += 1

            return FetchResponse(
                url=str(response.url),  # httpx返回的是URL对象
                status=response.status_code,
                content=content,
                headers=headers,
                elapsed_ms=elapsed,
                retries=0,
            )
        except httpx.HTTPError as exc:
            elapsed = (time.perf_counter() - start) * 1000
            self._stats["failed"] += 1
            self._stats["last_error"] = str(exc)
            logger.debug("Request error for %s (%.1f ms): %s", url, elapsed, exc)
            raise


class FetcherManager:
    """Lifecycle manager shared by pipeline entry points."""

    def __init__(self):
        self._fetcher: RequestsFetchStrategy | None = None

    def ensure(self, *, user_agent: str | None, proxies: dict[str, str] | None) -> RequestsFetchStrategy:
        if self._fetcher is None:
            self._fetcher = RequestsFetchStrategy(user_agent=user_agent, proxies=proxies)
        return self._fetcher

    async def close(self) -> None:
        """异步关闭httpx客户端"""
        if self._fetcher:
            await self._fetcher.close()
            self._fetcher = None

    def stats(self) -> dict[str, str | None]:
        if not self._fetcher:
            return {
                "total_requests": "0",
                "successful": "0",
                "failed": "0",
                "last_error": "",
            }
        return self._fetcher.stats()


FETCHER_MANAGER = FetcherManager()
