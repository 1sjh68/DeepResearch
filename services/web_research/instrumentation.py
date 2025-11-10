from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from typing import Any

logger = logging.getLogger(__name__)


def log_event(
    *,
    task_id: str | None,
    topic: str | None,
    node: str,
    stage_key: str,
    message: str,
    tokens: int | None = None,
    latency_ms: float | None = None,
    retries: int | None = None,
    extra: dict[str, Any] | None = None,
    level: int = logging.INFO,
) -> None:
    payload = {
        "task_id": task_id,
        "topic": topic,
        "node": node,
        "stage": stage_key,
        "tokens": tokens,
        "latency_ms": latency_ms,
        "retries": retries,
    }
    if extra:
        payload.update(extra)
    logger.log(level, message, extra=payload)


@contextmanager
def track_stage(
    *,
    task_id: str | None,
    topic: str | None,
    node: str,
    stage_key: str,
    message: str,
):
    start = time.perf_counter()
    log_event(
        task_id=task_id,
        topic=topic,
        node=node,
        stage_key=stage_key,
        message=f"START · {message}",
    )
    try:
        yield
    finally:
        elapsed = (time.perf_counter() - start) * 1000
        log_event(
            task_id=task_id,
            topic=topic,
            node=node,
            stage_key=stage_key,
            message=f"END · {message}",
            latency_ms=elapsed,
        )
