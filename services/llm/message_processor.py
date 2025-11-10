"""
LLM 消息处理和规范化模块

提供消息内容规范化、JSON指令注入等功能。
"""

import json
import logging
from typing import Any

# 导入文本规范化功能
from utils.text_normalizer import clean_text_artifacts

logger = logging.getLogger(__name__)


def ensure_json_instruction(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    确保在使用json_object时提示词明确指示返回JSON。

    某些提供商要求在使用response_format={"type": "json_object"}时
    提示词中必须出现'JSON'一词。如果没有消息提及JSON，
    此辅助函数会追加一个简洁的系统提示。
    """
    try:
        text = "\n".join(str(m.get("content", "")) for m in messages)
    except Exception:
        text = ""
    if "json" in text.lower():
        return messages
    # Append a minimal system message to satisfy the requirement and guide output
    hint = {
        "role": "system",
        "content": ("When responding, return ONLY valid JSON as a single object. No prose, no markdown fences, just raw JSON."),
    }
    return [*messages, hint]


def coerce_message_content(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    规范化消息内容，使其符合OpenAI API的期望。

    主要处理：
    - 将非字符串的content转换为字符串
    - 移除tool_calls和function_call字段并序列化到content中
    - 清理文本artifacts
    """
    coerced: list[dict[str, Any]] = []
    for msg in messages:
        new_msg = dict(msg)
        content = new_msg.get("content")

        if isinstance(content, str):
            pass
        elif content is None:
            new_msg["content"] = ""
        elif isinstance(content, list):
            pieces: list[str] = []
            for item in content:
                if isinstance(item, str):
                    pieces.append(item)
                elif isinstance(item, dict):
                    text_val = item.get("text")
                    if isinstance(text_val, str):
                        pieces.append(text_val)
                    else:
                        pieces.append(json.dumps(item, ensure_ascii=False))
                else:
                    pieces.append(json.dumps(item, ensure_ascii=False))
            new_msg["content"] = "\n".join(pieces)
        else:
            # Fallback: serialize unknown content types
            new_msg["content"] = json.dumps(content, ensure_ascii=False)

        tool_calls = new_msg.pop("tool_calls", None)
        if tool_calls:
            try:
                serialized_calls = json.dumps(tool_calls, ensure_ascii=False)
            except (TypeError, ValueError):
                serialized_calls = str(tool_calls)
            existing_content = new_msg.get("content") or ""
            new_msg["content"] = (
                f"{existing_content}\n{serialized_calls}" if existing_content else serialized_calls
            )

        function_call = new_msg.pop("function_call", None)
        if function_call:
            try:
                serialized_call = json.dumps(function_call, ensure_ascii=False)
            except (TypeError, ValueError):
                serialized_call = str(function_call)
            existing_content = new_msg.get("content") or ""
            new_msg["content"] = (
                f"{existing_content}\n{serialized_call}" if existing_content else serialized_call
            )

        coerced.append(new_msg)

    # 验证：确保所有消息的content都是字符串（防御性编程）
    for idx, msg in enumerate(coerced):
        content = msg.get("content")
        if not isinstance(content, str):
            logging.warning(
                "Message %d content is not str after coercion: %s, converting to string",
                idx,
                type(content).__name__,
            )
            msg["content"] = str(content) if content is not None else ""
        elif content:
            msg["content"] = clean_text_artifacts(content)

    return coerced


class MessageProcessor:
    """消息处理器的面向对象封装"""

    @staticmethod
    def process_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        处理消息列表，规范化内容

        Args:
            messages: 原始消息列表

        Returns:
            规范化后的消息列表
        """
        return coerce_message_content(messages)

    @staticmethod
    def add_json_instruction(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        确保消息中包含JSON指令

        Args:
            messages: 消息列表

        Returns:
            添加JSON指令后的消息列表
        """
        return ensure_json_instruction(messages)

    @staticmethod
    def clean_text(text: str) -> str:
        """
        清理文本中的artifacts

        Args:
            text: 原始文本

        Returns:
            清理后的文本
        """
        return clean_text_artifacts(text)

