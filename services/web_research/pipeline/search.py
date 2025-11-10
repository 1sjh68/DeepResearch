from __future__ import annotations

import json
import logging
import os

from config import Config

logger = logging.getLogger(__name__)

build = None
HttpError = None
httplib2 = None
socks = None
AuthorizedHttp = None
Credentials = None


def _lazy_imports() -> None:
    global build, HttpError, httplib2, socks, AuthorizedHttp, Credentials
    if build is None:
        try:
            from googleapiclient.discovery import build as _build
            from googleapiclient.errors import HttpError as _HttpError

            build = _build
            HttpError = _HttpError
        except Exception as exc:  # pragma: no cover - optional
            logger.warning("Google API client libraries unavailable: %s", exc)
    if httplib2 is None:
        try:
            import httplib2 as _httplib2

            httplib2 = _httplib2
        except Exception as exc:
            logger.warning("httplib2 unavailable: %s", exc)
    if socks is None:
        try:
            import socks as _socks

            socks = _socks
        except Exception:
            socks = None
    if AuthorizedHttp is None or Credentials is None:
        try:
            from google.oauth2.service_account import Credentials as _Credentials
            from google_auth_httplib2 import AuthorizedHttp as _AuthorizedHttp

            AuthorizedHttp = _AuthorizedHttp
            Credentials = _Credentials
        except Exception:
            AuthorizedHttp = None
            Credentials = None


def _build_proxy_info(config: Config):
    if not httplib2:
        return None
    proxy_url = os.environ.get("HTTP_PROXY") or os.environ.get("HTTPS_PROXY")
    if not proxy_url:
        return None
    if not socks:
        logger.warning("PySocks is required for proxy support; continuing without proxy")
        return None
    from urllib.parse import urlparse

    parsed = urlparse(proxy_url)
    return httplib2.ProxyInfo(
        socks.PROXY_TYPE_HTTP,
        parsed.hostname,
        parsed.port or 80,
        proxy_user=parsed.username,
        proxy_pass=parsed.password,
    )


def get_google_auth_http(config: Config):
    _lazy_imports()
    if not httplib2:
        raise RuntimeError("httplib2 is required for Google CSE requests")
    proxy_info = _build_proxy_info(config)
    base_http = httplib2.Http(
        timeout=getattr(config, "api_request_timeout_seconds", 60),
        proxy_info=proxy_info,
    )
    credentials_path = getattr(config.api, "google_service_account_path", None)
    if credentials_path and AuthorizedHttp and Credentials and os.path.exists(credentials_path):
        try:
            creds = Credentials.from_service_account_file(credentials_path, scopes=["https://www.googleapis.com/auth/cse"])
            return AuthorizedHttp(creds, http=base_http)
        except Exception as exc:
            logger.warning("Failed to initialize Google service account: %s", exc)
    return base_http


def perform_search(config: Config, query: str) -> list[dict]:
    _lazy_imports()
    if not build or not HttpError:
        logger.error("Google API client is not available; search disabled.")
        return []

    api_keys = getattr(config.api, "google_api_keys", [])
    cse_ids = getattr(config.api, "google_cse_ids", [])
    if not api_keys or not cse_ids:
        logger.error("Google API keys or CSE IDs missing; set GOOGLE_API_KEYS/GOOGLE_CSE_IDS.")
        return []

    max_items = getattr(
        getattr(config, "vector", None),
        "num_search_results",
        getattr(config, "num_search_results", 3),
    )
    pairs = min(len(api_keys), len(cse_ids))
    for _ in range(pairs):
        idx = getattr(config, "current_google_api_key_index", 0) % pairs
        api_key = api_keys[idx]
        cse_id = cse_ids[idx]
        try:
            http_client = get_google_auth_http(config)
            service = build("customsearch", "v1", developerKey=api_key, http=http_client)
            resp = service.cse().list(q=query, cx=cse_id, num=max_items).execute()
            return resp.get("items", [])
        except HttpError as err:
            reason = None
            try:
                payload = json.loads(err.content.decode("utf-8", "ignore"))
                reason = payload.get("error", {}).get("errors", [{}])[0].get("reason")
            except Exception:
                reason = None
            if err.resp.status == 429 and reason == "rateLimitExceeded":
                setattr(config, "current_google_api_key_index", idx + 1)
                logger.warning("Google CSE quota exceeded for key #%s, rotating...", idx + 1)
                continue
            logger.error("Google search failed (%s): %s", err.resp.status, err.content)
            return []
        except Exception as exc:
            logger.debug("Google search error: %s", exc)  # 降低日志级别，避免污染控制台
            return []
    logger.error("All Google API keys exhausted.")
    return []
