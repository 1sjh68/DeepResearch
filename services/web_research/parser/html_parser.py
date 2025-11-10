from __future__ import annotations

import importlib
import logging

from bs4 import BeautifulSoup

try:
    trafilatura = importlib.import_module("trafilatura")
    from trafilatura.settings import use_config
    TRAFILATURA_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    trafilatura = None
    TRAFILATURA_AVAILABLE = False

try:
    readability_module = importlib.import_module("readability")
    Document = getattr(readability_module, "Document", None)
except Exception:  # pragma: no cover - optional dependency
    Document = None

# jusText作为备份提取器（特别适合新闻网站）
try:
    import justext
    JUSTEXT_AVAILABLE = True
except ImportError:
    JUSTEXT_AVAILABLE = False
    logging.debug("justext not installed. Advanced news extraction not available.")

logger = logging.getLogger(__name__)


def parse_html(content: str) -> dict[str, str | None]:
    """Parse HTML content with advanced trafilatura configuration and jusText fallback."""
    if not content:
        return {"text": None, "title": None, "description": None}

    # 优先使用trafilatura（高级配置）
    if TRAFILATURA_AVAILABLE and trafilatura:
        try:
            # 配置trafilatura以获得更好的提取效果
            config = use_config()
            config.set("DEFAULT", "EXTRACTION_TIMEOUT", "0")
            config.set("DEFAULT", "MIN_EXTRACTED_SIZE", "100")
            config.set("DEFAULT", "MIN_EXTRACTED_COMM_SIZE", "50")

            extracted = trafilatura.extract(
                content,
                include_comments=False,
                include_tables=True,       # 保留表格（之前禁用了）
                include_formatting=True,   # 保留格式
                include_links=True,        # 保留链接
                output_format='txt',       # 纯文本输出
                config=config,
            )

            metadata = trafilatura.extract_metadata(content)

            if extracted and len(extracted.strip()) > 100:
                return {
                    "text": extracted.strip(),
                    "title": metadata.title if metadata else None,
                    "description": metadata.description if metadata else None,
                }
        except Exception as exc:  # pragma: no cover - best effort
            logger.debug("Trafilatura extraction failed: %s", exc)

    # jusText作为备份（特别适合新闻网站）
    if JUSTEXT_AVAILABLE and (not content or len(content.strip()) < 100):
        try:
            paragraphs = justext.justext(
                content,
                justext.get_stoplist("English"),  # 可以根据需要切换语言
                length_low=70,
                length_high=200,
                stopwords_low=0.30,
                stopwords_high=0.32,
                max_link_density=0.2,
            )
            extracted_text = "\n\n".join(
                [p.text for p in paragraphs if not p.is_boilerplate and len(p.text) > 50]
            )
            if extracted_text:
                # 从BeautifulSoup提取标题
                soup = BeautifulSoup(content, "lxml")
                title = soup.title.string.strip() if soup.title and soup.title.string else None
                meta_desc = soup.find("meta", attrs={"name": "description"})
                description = meta_desc["content"].strip() if meta_desc and meta_desc.get("content") else None
                return {"text": extracted_text, "title": title, "description": description}
        except Exception as exc:
            logger.debug("jusText extraction failed: %s", exc)

    # Readability fallback
    if Document:
        try:
            doc = Document(content)
            summary_html = doc.summary()
            text = BeautifulSoup(summary_html, "lxml").get_text(separator=" ", strip=True)
            return {
                "text": text,
                "title": doc.short_title(),
                "description": None,
            }
        except Exception as exc:
            logger.debug("Readability extraction failed: %s", exc)

    # BeautifulSoup兜底
    soup = BeautifulSoup(content, "lxml")
    text = soup.get_text(separator=" ", strip=True)
    title = soup.title.string.strip() if soup.title and soup.title.string else None
    meta_desc = soup.find("meta", attrs={"name": "description"})
    description = meta_desc["content"].strip() if meta_desc and meta_desc.get("content") else None
    return {"text": text, "title": title, "description": description}
