"""
统一抓取器服务层
提供智能网页抓取功能，支持多种抓取策略和内容抽取引擎
"""

import asyncio
import importlib
import logging
import random
import re
import time
import urllib.robotparser
from collections import defaultdict
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any, cast
from urllib.parse import urlparse

import httpx
from playwright.async_api import Browser, Page, async_playwright

trafilatura = None
Document = None
DefaultExtractor = None

try:
    trafilatura = importlib.import_module("trafilatura")
except ImportError:  # pragma: no cover - optional dependency
    trafilatura = None

try:
    readability_module = importlib.import_module("readability")
    Document = getattr(readability_module, "Document", None)
except ImportError:  # pragma: no cover - optional dependency
    Document = None

try:
    boilerpy3_module = importlib.import_module("boilerpy3.heuristics")
    DefaultExtractor = getattr(boilerpy3_module, "DefaultExtractor", None)
except ImportError:  # pragma: no cover - optional dependency
    DefaultExtractor = None

# 配置日志
logger = logging.getLogger(__name__)


@dataclass
class FetchConfig:
    """抓取配置类"""

    timeout: int = 30
    max_retries: int = 5  # 增加重试次数
    retry_delay: float = 1.0
    retry_backoff_factor: float = 1.5  # 更温和的退避因子
    rate_limit: float = 1.0  # 每秒请求数
    max_concurrent: int = 10
    user_agent_rotation: bool = True
    proxy_enabled: bool = False
    browser_fallback: bool = True
    content_extraction: bool = True
    robots_check: bool = True  # 是否检查robots.txt
    blocked_fallback: bool = True  # 是否启用拦截回退


@dataclass
class FetchMetrics:
    """抓取性能指标"""

    dns_ms: float | None = None
    tcp_ms: float | None = None
    tls_ms: float | None = None
    http_proto: str | None = None
    ip_country: str | None = None
    status: int | None = None
    content_length: int | None = None
    mime_type: str | None = None
    title: str | None = None
    server: str | None = None
    cf_ray: str | None = None
    total_time_ms: float | None = None
    extraction_quality: float | None = None
    soft_block_detected: bool = False
    hard_block_detected: bool = False
    retry_count: int = 0
    error_message: str | None = None


@dataclass
class FetchResult:
    """抓取结果"""

    url: str
    content: str | None = None
    html: str | None = None
    metadata: dict[str, Any] | None = None
    metrics: FetchMetrics | None = None
    strategy_used: str | None = None
    success: bool = False
    error: str | None = None


class RateLimiter:
    """令牌桶限流器"""

    def __init__(self, rate_per_second: float = 1.0):
        self.rate = rate_per_second
        self.tokens = rate_per_second
        self.last_update = time.time()
        self.lock = asyncio.Lock()

    async def acquire(self) -> bool:
        """获取令牌"""
        async with self.lock:
            now = time.time()
            elapsed = now - self.last_update

            # 添加令牌
            self.tokens = min(self.rate, self.tokens + elapsed * self.rate)
            self.last_update = now

            if self.tokens >= 1:
                self.tokens -= 1
                return True
            else:
                # 等待下一个令牌
                wait_time = (1 - self.tokens) / self.rate
                await asyncio.sleep(wait_time)
                self.tokens = 0
                return True


class SoftBlockDetector:
    """软拦截检测器"""

    def __init__(self):
        self.block_patterns = {
            "cloudflare": [
                r"DDos-GUARD",
                r"Checking your browser",
                r"Please enable cookies",
                r"cf-ray:",
                r"Cloudflare Ray ID",
            ],
            "aws": [
                r"This page is being protected by AWS WAF",
                r"challenge",
                r"Access Denied",
                r"Amazon CloudFront",
            ],
            "general": [
                r"access to this page has been denied",
                r"this site is temporarily unavailable",
                r"you have been blocked",
                r"rate limit exceeded",
                r"too many requests",
                r"captcha",
                r"javascript is required",
            ],
        }

    def detect_block(self, content: str, headers: dict[str, str]) -> tuple[bool, str]:
        """检测是否存在软拦截"""
        content_lower = content.lower()

        # 检查内容模式
        for category, patterns in self.block_patterns.items():
            for pattern in patterns:
                if re.search(pattern, content_lower, re.IGNORECASE):
                    return True, f"soft_block_detected_{category}"

        # 检查响应头
        if "cf-ray" in headers:
            return True, "soft_block_detected_cloudflare"

        # 检查状态码模式
        # 这里可以根据具体需求添加更多检测逻辑

        return False, ""


class UserAgentRotator:
    """User-Agent轮换器"""

    def __init__(self):
        # 更多样化的User-Agent列表，包含最新版本
        self.user_agents = [
            # Chrome最新版本
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
            # Firefox最新版本  
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:133.0) Gecko/20100101 Firefox/133.0",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 14.0) Gecko/20100101 Firefox/133.0",
            # Safari
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.0 Safari/605.1.15",
            # Edge
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36 Edg/131.0.0.0",
            # 移动端User-Agent
            "Mozilla/5.0 (iPhone; CPU iPhone OS 18_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.0 Mobile/15E148 Safari/604.1",
            "Mozilla/5.0 (Linux; Android 14; SM-G991B) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Mobile Safari/537.36",
        ]
        self.last_used_index = -1

    def get_random_ua(self) -> str:
        """获取随机User-Agent，避免连续重复"""
        # 避免连续使用相同的UA
        available_indices = list(range(len(self.user_agents)))
        if self.last_used_index >= 0:
            available_indices.remove(self.last_used_index)
        
        index = random.choice(available_indices)
        self.last_used_index = index
        return self.user_agents[index]
    
    def get_mobile_ua(self) -> str:
        """获取移动端User-Agent"""
        mobile_uas = [ua for ua in self.user_agents if "Mobile" in ua or "Android" in ua or "iPhone" in ua]
        return random.choice(mobile_uas) if mobile_uas else self.get_random_ua()


class ContentExtractor:
    """多引擎内容抽取器"""

    def __init__(self):
        self.extractors: dict[str, Callable[[str], dict[str, Any] | None]] = {}
        if trafilatura is not None:
            self.extractors["trafilatura"] = self._extract_with_trafilatura
        if Document is not None:
            self.extractors["readability"] = self._extract_with_readability
        if DefaultExtractor is not None:
            self.extractors["boilerpy3"] = self._extract_with_boilerpy3
        self.preference_order = list(self.extractors.keys())

    def _extract_with_trafilatura(self, html: str) -> dict[str, Any] | None:
        """使用trafilatura抽取内容"""
        if trafilatura is None:
            return None
        try:
            # 抽取主要文本
            text = trafilatura.extract(html, include_comments=False, include_tables=True)

            # 抽取元数据
            metadata = trafilatura.extract_metadata(html)

            if text:
                return {
                    "text": text.strip(),
                    "title": metadata.title if metadata else None,
                    "description": metadata.description if metadata else None,
                    "author": metadata.author if metadata else None,
                    "date": metadata.date if metadata else None,
                }
        except Exception as e:
            logger.warning(f"Trafilatura extraction failed: {e}")
        return None

    def _extract_with_readability(self, html: str) -> dict[str, Any] | None:
        """使用readability抽取内容"""
        if Document is None:
            return None
        try:
            doc = Document(html)

            # 获取标题
            title = doc.short_title() or doc.title()

            # 获取内容摘要
            summary = doc.summary(html_partial=True)

            # 移除HTML标签获取纯文本
            text = re.sub(r"<[^>]+>", "", summary).strip()

            return {
                "text": text,
                "title": title,
                "byline": doc.byline() if hasattr(doc, "byline") else None,
                "excerpt": doc.summary(html_partial=True)[:200] + "..." if summary else None,
            }
        except Exception as e:
            logger.warning(f"Readability extraction failed: {e}")
        return None

    def _extract_with_boilerpy3(self, html: str) -> dict[str, Any] | None:
        """使用boilerpy3抽取内容"""
        if DefaultExtractor is None:
            return None
        try:
            extractor = DefaultExtractor()
            content = extractor.get_doc(html)

            if content:
                return {
                    "text": content.content.strip() if content.content else "",
                    "title": getattr(content, "title", None),
                    "publish_date": getattr(content, "publish_date", None),
                }
        except Exception as e:
            logger.warning(f"Boilerpy3 extraction failed: {e}")
        return None

    def extract_best(self, html: str) -> tuple[dict[str, Any] | None, float]:
        """使用首选引擎抽取内容，失败时回退到备选引擎"""
        extracted = None
        quality_score = 0.0

        for extractor_name in self.preference_order:
            result = self.extractors[extractor_name](html)
            if result and result.get("text"):
                # 计算质量分数
                quality_score = self._calculate_quality_score(result)
                extracted = result
                extracted["extraction_engine"] = extractor_name
                break

        return extracted, quality_score

    def _calculate_quality_score(self, result: dict[str, Any]) -> float:
        """计算抽取质量分数"""
        score = 0.0

        # 文本长度分数
        text_len = len(result.get("text", ""))
        if text_len > 1000:
            score += 0.4
        elif text_len > 500:
            score += 0.3
        elif text_len > 100:
            score += 0.2

        # 标题分数
        if result.get("title"):
            score += 0.2

        # 元数据完整性分数
        metadata_count = sum(1 for key in ["author", "date", "description", "byline", "publish_date"] if result.get(key))
        score += metadata_count * 0.1

        return min(score, 1.0)


class RobotsCache:
    """robots.txt缓存管理器"""

    def __init__(self):
        self.cache: dict[str, urllib.robotparser.RobotFileParser] = {}  # host -> robotparser
        self.cache_time: dict[str, float] = {}  # host -> timestamp
        self.cache_ttl = 3600  # 1小时缓存

    async def can_fetch(self, url: str, user_agent: str = "*") -> bool:
        """检查是否允许抓取URL"""
        try:
            parsed = urlparse(url)
            host = parsed.netloc

            # 检查缓存
            if host in self.cache:
                cache_time = self.cache_time.get(host, 0)
                if time.time() - cache_time < self.cache_ttl:
                    return cast(urllib.robotparser.RobotFileParser, self.cache[host]).can_fetch(user_agent, url)

            # 获取robots.txt
            robots_url = f"{parsed.scheme}://{host}/robots.txt"
            rp = urllib.robotparser.RobotFileParser()
            rp.set_url(robots_url)

            try:
                rp.read()
                # 缓存结果
                self.cache[host] = rp
                self.cache_time[host] = time.time()

                return rp.can_fetch(user_agent, url)
            except Exception as e:
                # 如果无法获取robots.txt，默认允许抓取
                logger.debug(f"获取robots.txt失败: {str(e)}")
                return True

        except Exception as e:
            logger.debug(f"robots.txt检查失败: {str(e)}")
            return True

class ProxyManager:
    """代理管理器"""

    def __init__(self):
        self.proxies: list[str] = []
        self.current_index = 0
        self.failed_proxies: set[str] = set()

    def add_proxy(self, proxy: str):
        """添加代理"""
        if proxy not in self.proxies:
            self.proxies.append(proxy)

    def add_proxies(self, proxies: list[str]):
        """批量添加代理"""
        for proxy in proxies:
            self.add_proxy(proxy)

    def get_next_proxy(self) -> str | None:
        """获取下一个可用代理"""
        if not self.proxies:
            return None

        # 尝试找到未失败的代理
        for _ in range(len(self.proxies)):
            proxy = self.proxies[self.current_index]
            self.current_index = (self.current_index + 1) % len(self.proxies)

            if proxy not in self.failed_proxies:
                return proxy

        # 如果所有代理都失败了，重置失败列表
        self.failed_proxies.clear()
        return self.proxies[0] if self.proxies else None

    def mark_proxy_failed(self, proxy: str):
        """标记代理失败"""
        self.failed_proxies.add(proxy)

        # 如果所有代理都失败了，重置失败列表
        if len(self.failed_proxies) >= len(self.proxies):
            self.failed_proxies.clear()


class BrowserManager:
    """浏览器管理器"""

    def __init__(self):
        self.browser: Browser | None = None
        self.playwright = None
        self.pages: list[Page] = []
        self.max_pages = 5

    async def start(self):
        """启动浏览器"""
        if not self.browser:
            self.playwright = await async_playwright().start()
            self.browser = await self.playwright.chromium.launch(
                headless=True,
                args=[
                    "--no-sandbox",
                    "--disable-dev-shm-usage",
                    "--disable-blink-features=AutomationControlled",
                    "--disable-extensions",
                    "--disable-plugins-discovery",
                ],
            )

    async def get_page(self) -> Page:
        """获取页面"""
        if not self.browser:
            await self.start()

        browser = self.browser
        if browser is None:
            raise RuntimeError("BrowserManager: 浏览器未能初始化。")

        if len(self.pages) < self.max_pages:
            page = await browser.new_page()
            self.pages.append(page)
            return page

        # 重用现有页面
        return random.choice(self.pages)

    async def close(self):
        """关闭浏览器"""
        for page in self.pages:
            await page.close()
        if self.browser:
            await self.browser.close()
        if self.playwright:
            await self.playwright.stop()


class SmartFetcher:
    """智能抓取器主类"""

    def __init__(self, config: FetchConfig | None = None):
        self.config = config or FetchConfig()

        # 初始化组件
        self.rate_limiter = RateLimiter(self.config.rate_limit)
        self.block_detector = SoftBlockDetector()
        self.ua_rotator = UserAgentRotator()
        self.content_extractor = ContentExtractor()
        self.proxy_manager = ProxyManager()
        self.browser_manager = BrowserManager()
        self.robots_cache = RobotsCache()

        # HTTP客户端池
        self.http_client: httpx.AsyncClient | None = None

        # 会话缓存
        self.session_cache: dict[str, Any] = {}

        # 性能统计
        self.stats: defaultdict[str, int] = defaultdict(int)

        # 指数退避状态
        self.retry_delays: dict[str, dict[str, Any]] = {}  # url -> last_retry_time

        logger.info("SmartFetcher initialized with config: %s", asdict(self.config))

    async def __aenter__(self):
        """异步上下文管理器入口"""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口"""
        await self.close()

    async def start(self):
        """启动抓取器"""
        # 初始化HTTP客户端（增强的请求头）
        headers = {
            "User-Agent": self.ua_rotator.get_random_ua(),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9,zh-CN;q=0.8,zh;q=0.7",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "none",
            "Sec-Fetch-User": "?1",
            "Cache-Control": "max-age=0",
            "DNT": "1",
        }

        self.http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(self.config.timeout),
            headers=headers,
            limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
        )

        # 启动浏览器
        await self.browser_manager.start()

        logger.info("SmartFetcher started successfully")

    async def close(self):
        """关闭抓取器"""
        if self.http_client:
            await self.http_client.aclose()

        await self.browser_manager.close()

        logger.info("SmartFetcher closed. Statistics: %s", dict(self.stats))

    async def fetch(self, url: str, strategy: str = "auto") -> FetchResult:
        """抓取网页内容

        Args:
            url: 要抓取的URL
            strategy: 抓取策略 ("auto", "direct", "browser", "proxy")

        Returns:
            FetchResult: 抓取结果
        """
        start_time = time.time()
        metrics = FetchMetrics()

        try:
            # 限流
            await self.rate_limiter.acquire()

            # 根据策略选择抓取方法
            if strategy == "auto":
                result = await self._auto_fetch(url, metrics)
            elif strategy == "direct":
                result = await self._direct_fetch(url, metrics)
            elif strategy == "browser":
                result = await self._browser_fetch(url, metrics)
            elif strategy == "proxy":
                result = await self._proxy_fetch(url, metrics)
            else:
                raise ValueError(f"Unknown strategy: {strategy}")

            # 记录性能统计
            total_time = (time.time() - start_time) * 1000
            metrics.total_time_ms = total_time

            result.metrics = metrics
            result.url = url

            # 更新统计
            self.stats["total_requests"] += 1
            if result.success:
                self.stats["successful_requests"] += 1
                self.stats[f"strategy_{result.strategy_used}"] += 1
                # 清除重试记录
                if url in self.retry_delays:
                    del self.retry_delays[url]
            else:
                self.stats["failed_requests"] += 1

                # 更新拦截统计
                if metrics.hard_block_detected:
                    self.stats["blocked_hard"] += 1
                elif metrics.soft_block_detected:
                    self.stats["blocked_soft"] += 1

            logger.info(
                "Fetch completed for %s: %s (%.2fms)",
                url,
                "success" if result.success else "failed",
                total_time,
            )

            return result

        except Exception as e:
            total_time = (time.time() - start_time) * 1000
            metrics.total_time_ms = total_time
            metrics.error_message = str(e)

            self.stats["failed_requests"] += 1
            self.stats["total_requests"] += 1

            logger.debug("Fetch failed for %s: %s", url, str(e))  # 降低日志级别，避免污染控制台

            return FetchResult(url=url, success=False, error=str(e), metrics=metrics)

    async def _auto_fetch(self, url: str, metrics: FetchMetrics) -> FetchResult:
        """自动策略抓取：直连→浏览器→代理"""

        # 检查robots.txt
        if self.config.robots_check:
            can_fetch = await self.robots_cache.can_fetch(url)
            if not can_fetch:
                logger.info("Robots.txt disallows fetching: %s", url)
                return FetchResult(
                    url=url,
                    success=False,
                    error="Robots.txt disallows this URL",
                    strategy_used="robots_blocked",
                )

        # 策略1：直接HTTP请求
        result = await self._direct_fetch(url, metrics, raise_on_error=False)
        if result.success and not metrics.soft_block_detected and not metrics.hard_block_detected:
            return result

        # 策略2：浏览器回退（处理403/429）
        if self.config.browser_fallback:
            result = await self._browser_fetch(url, metrics, raise_on_error=False)
            if result.success:
                return result

        # 策略3：代理回退（处理403）
        if self.config.proxy_enabled and self.proxy_manager.proxies:
            result = await self._proxy_fetch(url, metrics, raise_on_error=False)
            if result.success:
                return result

        # 返回最后的错误结果
        return result

    async def _direct_fetch(self, url: str, metrics: FetchMetrics, raise_on_error: bool = True) -> FetchResult:
        """直接HTTP请求抓取"""

        try:
            client = self.http_client
            if client is None:
                raise RuntimeError("SmartFetcher: HTTP 客户端未初始化。")

            # 检查是否为重试请求
            retry_count = self.retry_delays.get(url, {}).get("count", 0)
            metrics.retry_count = retry_count

            # 随机User-Agent
            if self.config.user_agent_rotation:
                client.headers.update({"User-Agent": self.ua_rotator.get_random_ua()})

            # 发送请求
            response = await client.get(url)

            # 记录基础信息
            headers_dict = dict(response.headers.items())
            metrics.status = response.status_code
            metrics.content_length = len(response.content)
            metrics.mime_type = headers_dict.get("content-type")
            metrics.server = headers_dict.get("server")
            metrics.cf_ray = headers_dict.get("cf-ray")

            # 检测软拦截
            content = response.text
            is_blocked, _ = self.block_detector.detect_block(content, headers_dict)
            metrics.soft_block_detected = is_blocked

            # 显式处理403/429状态码
            if response.status_code in {403, 429}:
                metrics.hard_block_detected = True

                if response.status_code == 429:
                    # 429: 限流错误，实现指数退避
                    await self._handle_rate_limit(url, retry_count)

                    if retry_count < self.config.max_retries:
                        # 记录重试信息
                        self.retry_delays[url] = {
                            "count": retry_count + 1,
                            "last_attempt": time.time(),
                        }

                        logger.info(
                            "Rate limited (429) for %s, retry %d/%d",
                            url,
                            retry_count + 1,
                            self.config.max_retries,
                        )

                        # 递归重试
                        return await self._direct_fetch(url, metrics, raise_on_error)

                elif response.status_code == 403:
                    # 403: 禁止访问，先尝试更换User-Agent重试
                    if retry_count < 2:  # 尝试两次不同的UA
                        self.retry_delays[url] = {
                            "count": retry_count + 1,
                            "last_attempt": time.time(),
                        }
                        
                        # 尝试使用移动端UA（有时更容易通过）
                        if retry_count == 1:
                            client.headers.update({"User-Agent": self.ua_rotator.get_mobile_ua()})
                            logger.info("403 on %s, retrying with mobile UA", url)
                        else:
                            client.headers.update({"User-Agent": self.ua_rotator.get_random_ua()})  
                            logger.info("403 on %s, retrying with different UA", url)
                        
                        await asyncio.sleep(random.uniform(0.5, 2.0))  # 短暂随机延迟
                        return await self._direct_fetch(url, metrics, raise_on_error)
                    
                    # 多次UA重试失败后，尝试其他策略
                    if self.config.browser_fallback:
                        logger.info("Access forbidden (403) for %s after UA retries, trying browser", url)
                        return await self._browser_fetch(url, metrics, raise_on_error)
                        
                    if self.config.proxy_enabled and self.proxy_manager.proxies:
                        logger.info("Access forbidden (403) for %s after UA retries, trying proxy", url)
                        return await self._proxy_fetch(url, metrics, raise_on_error)

            # 检查响应状态
            if response.status_code >= 400:
                raise httpx.HTTPStatusError(f"HTTP {response.status_code}", request=response.request, response=response)

            # 内容抽取
            if self.config.content_extraction:
                extracted, quality = self.content_extractor.extract_best(content)
                metrics.extraction_quality = quality

                if extracted:
                    return FetchResult(
                        url=url,
                        content=extracted["text"],
                        html=content,
                        metadata={
                            "title": extracted.get("title"),
                            "extraction_engine": extracted.get("extraction_engine"),
                            "raw_metadata": extracted,
                        },
                        strategy_used="direct",
                        success=True,
                    )

            # 返回原始内容
            return FetchResult(url=url, content=content, html=content, strategy_used="direct", success=True)

        except Exception as e:
            logger.warning("Direct fetch failed for %s: %s", url, str(e))

            if raise_on_error:
                raise

            return FetchResult(url=url, success=False, error=str(e), strategy_used="direct")

    async def _browser_fetch(self, url: str, metrics: FetchMetrics, raise_on_error: bool = True) -> FetchResult:
        """浏览器抓取"""

        try:
            # 检查是否为重试请求
            retry_count = self.retry_delays.get(url, {}).get("count", 0)
            metrics.retry_count = retry_count

            page = await self.browser_manager.get_page()

            # 设置浏览器指纹
            await self._setup_browser_fingerprint(page)

            # 访问页面
            response = await page.goto(url, timeout=self.config.timeout * 1000)
            if response is None:
                raise RuntimeError("Playwright 未返回有效的响应。")

            # 记录状态码
            metrics.status = response.status

            # 显式处理403/429状态码
            if response.status in {403, 429}:
                metrics.hard_block_detected = True

                if response.status == 429:
                    # 429: 限流错误，实现指数退避
                    await self._handle_rate_limit(url, retry_count)

                    if retry_count < self.config.max_retries:
                        # 记录重试信息
                        self.retry_delays[url] = {
                            "count": retry_count + 1,
                            "last_attempt": time.time(),
                        }

                        logger.info(
                            "Browser rate limited (429) for %s, retry %d/%d",
                            url,
                            retry_count + 1,
                            self.config.max_retries,
                        )

                        # 递归重试
                        return await self._browser_fetch(url, metrics, raise_on_error)

                elif response.status == 403:
                    # 403: 禁止访问，尝试代理回退
                    if self.config.proxy_enabled and self.proxy_manager.proxies:
                        logger.info("Browser access forbidden (403) for %s, trying proxy", url)
                        return await self._proxy_fetch(url, metrics, raise_on_error)

            # 等待页面加载
            await page.wait_for_load_state("networkidle", timeout=10000)

            # 获取页面内容
            content = await page.content()
            title = await page.title()

            # 记录信息
            metrics.content_length = len(content.encode())
            metrics.title = title
            metrics.mime_type = "text/html"

            # 检测软拦截
            is_blocked, _ = self.block_detector.detect_block(content, {})
            metrics.soft_block_detected = is_blocked

            # 内容抽取
            if self.config.content_extraction:
                extracted, quality = self.content_extractor.extract_best(content)
                metrics.extraction_quality = quality

                if extracted:
                    return FetchResult(
                        url=url,
                        content=extracted["text"],
                        html=content,
                        metadata={
                            "title": extracted.get("title"),
                            "extraction_engine": extracted.get("extraction_engine"),
                            "raw_metadata": extracted,
                        },
                        strategy_used="browser",
                        success=True,
                    )

            return FetchResult(url=url, content=content, html=content, strategy_used="browser", success=True)

        except Exception as e:
            logger.warning("Browser fetch failed for %s: %s", url, str(e))

            if raise_on_error:
                raise

            return FetchResult(url=url, success=False, error=str(e), strategy_used="browser")

    async def _proxy_fetch(self, url: str, metrics: FetchMetrics, raise_on_error: bool = True) -> FetchResult:
        """代理抓取"""

        proxy: str | None = None
        try:
            proxy = self.proxy_manager.get_next_proxy()
            if proxy is None:
                raise ValueError("No proxy available")

            headers: dict[str, str] = {}
            if self.config.user_agent_rotation:
                headers["User-Agent"] = self.ua_rotator.get_random_ua()

            client_kwargs: dict[str, Any] = {
                "timeout": httpx.Timeout(self.config.timeout),
                "proxies": {"http://": proxy, "https://": proxy},
            }
            if headers:
                client_kwargs["headers"] = headers

            async_client_cls = cast(Any, httpx.AsyncClient)
            async with async_client_cls(**client_kwargs) as client:
                response = await client.get(url)

            # 记录信息
            headers_dict = dict(response.headers.items())
            metrics.status = response.status_code
            metrics.content_length = len(response.content)
            metrics.mime_type = headers_dict.get("content-type")

            # 检测软拦截
            content = response.text
            is_blocked, block_type = self.block_detector.detect_block(content, headers_dict)
            metrics.soft_block_detected = is_blocked

            # 内容抽取
            if self.config.content_extraction:
                extracted, quality = self.content_extractor.extract_best(content)
                metrics.extraction_quality = quality

                if extracted:
                    return FetchResult(
                        url=url,
                        content=extracted["text"],
                        html=content,
                        metadata={
                            "title": extracted.get("title"),
                            "extraction_engine": extracted.get("extraction_engine"),
                            "raw_metadata": extracted,
                            "proxy_used": proxy,
                        },
                        strategy_used="proxy",
                        success=True,
                    )

            return FetchResult(
                url=url,
                content=content,
                html=content,
                metadata={"proxy_used": proxy},
                strategy_used="proxy",
                success=True,
            )

        except Exception as e:
            logger.warning("Proxy fetch failed for %s: %s", url, str(e))

            # 标记代理失败
            if proxy:
                self.proxy_manager.mark_proxy_failed(proxy)

            if raise_on_error:
                raise

            return FetchResult(url=url, success=False, error=str(e), strategy_used="proxy")

    async def _handle_rate_limit(self, url: str, retry_count: int):
        """处理429限流错误，实现智能退避"""
        # 计算退避延迟，加入随机因子避免批量请求同时重试
        base_delay = self.config.retry_delay
        exponential_delay = base_delay * (self.config.retry_backoff_factor**retry_count)
        max_delay = 30  # 降低最大延迟到30秒
        
        # 添加随机抖动（jitter）
        jitter = random.uniform(0.5, 1.5)
        delay = min(exponential_delay * jitter, max_delay)

        logger.info("Applying exponential backoff for %s: %.2f seconds delay", url, delay)

        # 等待指定时间
        await asyncio.sleep(delay)

        # 更新统计
        self.stats["rate_limited"] += 1

    async def _setup_browser_fingerprint(self, page: Page):
        """设置浏览器指纹"""

        # 移除webdriver属性
        await page.add_init_script(
            "\n".join(
                [
                    "Object.defineProperty(navigator, 'webdriver', {",
                    "    get: () => undefined,",
                    "});",
                    "",
                    "// 模拟真实浏览器特征",
                    "Object.defineProperty(navigator, 'plugins', {",
                    "    get: () => [1, 2, 3, 4, 5],",
                    "});",
                    "",
                    "Object.defineProperty(navigator, 'languages', {",
                    "    get: () => ['en-US', 'en'],",
                    "});",
                ]
            )
        )

    async def batch_fetch(self, urls: list[str], strategy: str = "auto", max_concurrent: int | None = None) -> list[FetchResult]:
        """批量抓取URLs

        Args:
            urls: URL列表
            strategy: 抓取策略
            max_concurrent: 最大并发数

        Returns:
            List[FetchResult]: 抓取结果列表
        """
        max_concurrent = max_concurrent or self.config.max_concurrent
        semaphore = asyncio.Semaphore(max_concurrent)

        async def fetch_with_semaphore(url: str) -> FetchResult:
            async with semaphore:
                return await self.fetch(url, strategy)

        # 执行批量抓取
        tasks = [fetch_with_semaphore(url) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 处理异常结果
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append(FetchResult(url=urls[i], success=False, error=str(result)))
            else:
                processed_results.append(result)

        return processed_results

    def add_proxies(self, proxies: list[str]):
        """添加代理列表"""
        self.proxy_manager.add_proxies(proxies)
        self.config.proxy_enabled = True
        logger.info("Added %d proxies", len(proxies))

# 便捷函数
async def fetch_url(url: str, config: FetchConfig | None = None) -> FetchResult:
    """便捷函数：抓取单个URL"""
    async with SmartFetcher(config) as fetcher:
        return await fetcher.fetch(url)


async def fetch_urls(
    urls: list[str],
    config: FetchConfig | None = None,
    strategy: str = "auto",
    max_concurrent: int = 10,
) -> list[FetchResult]:
    """便捷函数：批量抓取URLs"""
    async with SmartFetcher(config) as fetcher:
        return await fetcher.batch_fetch(urls, strategy, max_concurrent)


# 配置示例
DEFAULT_CONFIG = FetchConfig(
    timeout=30,
    max_retries=3,
    retry_delay=1.0,
    rate_limit=1.0,  # 每秒1个请求
    max_concurrent=10,
    user_agent_rotation=True,
    proxy_enabled=False,
    browser_fallback=True,
    content_extraction=True,
)

FAST_CONFIG = FetchConfig(
    timeout=10,
    max_retries=1,
    retry_delay=0.5,
    rate_limit=5.0,  # 每秒5个请求
    max_concurrent=20,
    user_agent_rotation=True,
    proxy_enabled=False,
    browser_fallback=False,  # 禁用浏览器回退以提高速度
    content_extraction=False,  # 禁用内容抽取以提高速度
)

STEALTH_CONFIG = FetchConfig(
    timeout=60,
    max_retries=5,
    retry_delay=2.0,
    rate_limit=0.5,  # 每秒0.5个请求
    max_concurrent=5,
    user_agent_rotation=True,
    proxy_enabled=True,  # 启用代理
    browser_fallback=True,
    content_extraction=True,
)


if __name__ == "__main__":
    # 示例用法
    async def main():
        # 配置
        config = DEFAULT_CONFIG

        # 抓取单个URL
        result = await fetch_url("https://example.com", config)
        logging.info(
            f"Single fetch → success={result.success} · content_len={len(result.content) if result.content else 0}",
        )
        logging.info(
            f"Metrics snapshot: {asdict(result.metrics) if result.metrics else {}}",
        )

        # 批量抓取
        urls = ["https://example.com", "https://httpbin.org/html", "https://httpbin.org/json"]

        results = await fetch_urls(urls, config, max_concurrent=3)
        for idx, r in enumerate(results, start=1):
            logging.info(f"[{idx}/{len(results)}] {r.url} · success={r.success}")

        logging.info("批量抓取完成。")

    # 运行示例
    # asyncio.run(main())
