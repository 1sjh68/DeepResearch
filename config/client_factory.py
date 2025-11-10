from __future__ import annotations

import logging
import sys
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse, urlunparse

import openai

if TYPE_CHECKING:
    from .config import Config  # pragma: no cover

try:
    import httpx
except ImportError:  # pragma: no cover - optional dependency
    httpx = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


def create_deepseek_client(config: Config) -> openai.OpenAI:
    """创建并配置同步的DeepSeek/OpenAI兼容客户端。"""
    if not config.deepseek_api_key:
        logger.critical("DEEPSEEK_API_KEY 环境变量未设置。")
        raise ValueError("DEEPSEEK_API_KEY 环境变量未设置。")

    http_client: Any = None
    try:
        raw_base_url = (config.deepseek_base_url or "").strip()
        if not raw_base_url:
            logger.critical("未提供有效的 DeepSeek 基础地址。")
            raise ValueError("DEEPSEEK_BASE_URL 环境变量未设置或为空。")

        parsed_url = urlparse(raw_base_url.rstrip("/"))
        path = parsed_url.path or ""
        if path in {"", "/"}:
            parsed_url = parsed_url._replace(path="/v1")
        base_url = urlunparse(parsed_url)

        if httpx is not None:
            timeout_seconds = float(config.api.api_request_timeout_seconds)
            limits = httpx.Limits(max_connections=20, max_keepalive_connections=10)
            base_kwargs: dict[str, Any] = {
                "timeout": timeout_seconds,
                "trust_env": config.proxies.llm_proxy_trust_env,
                "http2": False,
                "limits": limits,
            }
            if config.proxies.llm_disable_proxy:
                base_kwargs["trust_env"] = False
                http_client = httpx.Client(**base_kwargs)
                logger.info("LLM 客户端将绕过系统代理 (LLM_DISABLE_PROXY=true)")
            else:
                proxies: dict[str, str] = {}
                if config.proxies.llm_http_proxy:
                    proxies["http://"] = config.proxies.llm_http_proxy
                if config.proxies.llm_https_proxy:
                    proxies["https://"] = config.proxies.llm_https_proxy

                if proxies or not config.proxies.llm_proxy_trust_env:
                    client_kwargs = dict(base_kwargs)
                    client_kwargs["trust_env"] = config.proxies.llm_proxy_trust_env
                    client_kwargs["proxies"] = proxies or None
                    http_client = httpx.Client(**client_kwargs)
                    logger.info(
                        "LLM 客户端代理配置: proxies=%s trust_env=%s",
                        bool(proxies),
                        config.proxies.llm_proxy_trust_env,
                    )

        try:
            import httpcore  # type: ignore
        except Exception:  # pragma: no cover - optional dependency
            httpcore = None  # type: ignore

        logger.info(
            "Runtime info: sys.executable=%s | openai=%s | httpx=%s | httpcore=%s",
            sys.executable,
            getattr(openai, "__version__", "unknown"),
            getattr(httpx, "__version__", "unknown") if httpx else "n/a",
            getattr(httpcore, "__version__", "unknown") if httpcore else "n/a",
        )

        client = openai.OpenAI(
            api_key=config.deepseek_api_key,
            base_url=base_url,
            timeout=float(config.api.api_request_timeout_seconds),
            http_client=http_client,
        )
        logger.info("DeepSeek 客户端初始化成功，连接至 %s", base_url)
        return client
    except Exception as exc:
        logger.critical("DeepSeek 客户端初始化期间出错: %s。请检查 API 密钥/URL 和网络连接。", exc)
        raise RuntimeError(f"DeepSeek 客户端初始化失败: {exc}") from exc


__all__ = ["create_deepseek_client"]
