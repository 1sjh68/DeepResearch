from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ResearchAnchor:
    """Anchor linking a summarized fragment back to its source."""

    text: str
    url: str
    offset: int
    confidence: float = 0.0


@dataclass
class ResearchSource:
    """Metadata describing a single researched source."""

    url: str
    title: str | None
    summary: str
    raw_content: str
    score: float
    tokens_used: int = 0
    latency_ms: float = 0.0
    retries: int = 0
    anchors: list[ResearchAnchor] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ResearchResult:
    """
    Canonical research response surfaced to downstream nodes.

    body: Aggregated summary paragraphs suitable for injection into drafting nodes.
    sources: Ordered list of ResearchSource entries.
    anchors: Flattened anchors for quick citation lookup.
    """

    body: str
    sources: list[ResearchSource] = field(default_factory=list)
    anchors: list[ResearchAnchor] = field(default_factory=list)

    def to_brief(self) -> str:
        """Backward-compatible string brief for legacy consumers."""
        if not self.sources:
            return self.body
        parts: list[str] = [self.body.strip(), "\n\n--- Sources ---"]
        for idx, src in enumerate(self.sources, start=1):
            snippet = src.summary.strip() or src.raw_content[:280]
            parts.append(f"[{idx}] {src.title or src.url}\nURL: {src.url}\n{snippet}\n")
        return "\n".join(parts).strip()


@dataclass
class FetchResponse:
    """Normalized HTTP fetch result."""

    url: str
    status: int
    content: str
    headers: dict[str, str]
    elapsed_ms: float
    retries: int
