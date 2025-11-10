from __future__ import annotations

from services.web_research.models import ResearchAnchor, ResearchSource


def build_anchors(sources: list[ResearchSource]) -> list[ResearchAnchor]:
    anchors: list[ResearchAnchor] = []
    for idx, src in enumerate(sources):
        snippet = src.summary.splitlines()[0] if src.summary else src.raw_content[:120]
        anchors.append(
            ResearchAnchor(
                text=snippet.strip(),
                url=src.url,
                offset=idx,
                confidence=min(1.0, max(0.1, src.score)),
            )
        )
    return anchors
