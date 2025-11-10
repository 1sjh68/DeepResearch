from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import wraps
from typing import Any

from utils.progress_tracker import EnhancedProgressTracker, get_tracker

StateMapping = Mapping[str, Any]
StepPayload = Mapping[str, Any]


@dataclass
class StepOutput:
    data: dict[str, Any]
    detail: str | None = None


NodeResult = StepPayload | StepOutput


def _fetch_tracker(state: StateMapping) -> EnhancedProgressTracker | None:
    task_id = state.get("task_id")
    return get_tracker(task_id) if task_id else None


def step_result(data: dict[str, Any], detail: str | None = None) -> StepOutput:
    """辅助函数，同时返回节点输出数据和进度详情。"""
    return StepOutput(data=data, detail=detail)


def workflow_step(
    step_name: str,
    message: str,
) -> Callable[[Callable[..., NodeResult]], Callable[..., StepPayload]]:
    """
    装饰器，为LangGraph节点函数添加进度追踪功能。

    节点函数可以返回dict（标准LangGraph更新）
    或`StepOutput(data=..., detail="...")`以在完成时记录额外详情。
    """

    def decorator(func: Callable[..., NodeResult]) -> Callable[..., StepPayload]:
        @wraps(func)
        def wrapper(state: StateMapping, *args: Any, **kwargs: Any) -> StepPayload:
            tracker = _fetch_tracker(state)
            if tracker:
                tracker.start_step(step_name, message)
            # 已移除冗余的工作流步骤DEBUG日志
            try:
                result: NodeResult = func(state, *args, **kwargs)
            except Exception as exc:
                if tracker:
                    tracker.fail_step(step_name, str(exc))
                logging.error("工作流步骤 %s 中发生异常: %s", step_name, exc, exc_info=True)
                raise

            detail = ""
            payload: StepPayload
            if isinstance(result, StepOutput):
                payload = result.data
                detail = result.detail or ""
            else:
                payload = result

            if tracker:
                tracker.complete_step(step_name, detail)
            # 已移除冗余的工作流步骤DEBUG日志

            return payload

        return wrapper

    return decorator
