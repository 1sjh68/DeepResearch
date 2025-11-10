from __future__ import annotations

import hashlib
from collections.abc import Iterable


def dedupe_passages(passages: Iterable[tuple[str, str]]) -> list[tuple[str, str]]:
    """
    Remove duplicate passages based on a hash of normalized text.
    passages: iterable of (url, text)
    """
    seen = set()
    output: list[tuple[str, str]] = []
    for url, text in passages:
        norm = " ".join(text.strip().split()).lower()
        digest = hashlib.sha1(norm.encode("utf-8")).hexdigest()
        if digest in seen:
            continue
        seen.add(digest)
        output.append((url, text))
    return output
