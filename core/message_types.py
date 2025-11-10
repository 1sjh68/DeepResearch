from __future__ import annotations

from typing import TypedDict


class ChatMessage(TypedDict):
    """用于LLM交互的标准聊天消息载荷。"""

    role: str
    content: str
