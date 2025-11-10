"""核心接口定义，用于打破循环依赖。

本模块定义协议(Protocol)接口，允许模块之间通过接口而非具体实现进行交互，
从而解耦utils和services之间的循环依赖关系。

使用Protocol而非ABC是为了：
1. 支持结构化子类型（structural subtyping）
2. 避免需要显式继承
3. 更灵活的鸭子类型检查
"""

from __future__ import annotations

from typing import Any, Protocol


class JSONRepairProtocol(Protocol):
    """JSON修复接口协议。

    定义JSON修复功能的标准接口，允许不同的实现策略。
    """

    def __call__(self, text: str, schema: type[Any]) -> tuple[str, bool]:
        """修复JSON文本。

        参数：
            text: 待修复的JSON文本
            schema: 目标Pydantic模式类

        返回：
            (修复后的文本, 是否成功修复)
        """
        ...


class LLMCallProtocol(Protocol):
    """LLM调用接口协议。

    定义LLM调用功能的标准接口，支持依赖注入。
    """

    def __call__(
        self,
        config: Any,  # Config
        model_name: str,
        messages: list[dict[str, str]],
        **kwargs: Any
    ) -> Any | str:
        """调用LLM模型。

        参数：
            config: 配置对象
            model_name: 模型名称
            messages: 消息列表
            **kwargs: 其他参数（temperature、max_tokens等）

        返回：
            模型响应（字符串或结构化对象）
        """
        ...


class TextProcessorProtocol(Protocol):
    """文本处理接口协议。

    定义文本处理功能的标准接口。
    """

    def preprocess_json(self, json_string: str) -> str:
        """预处理JSON字符串。

        参数：
            json_string: 原始JSON字符串

        返回：
            预处理后的JSON字符串
        """
        ...

    def extract_json(
        self,
        config: Any,  # Config
        response_text: str,
        llm_caller: LLMCallProtocol,
        context: str = ""
    ) -> str | None:
        """从响应中提取JSON。

        参数：
            config: 配置对象
            response_text: AI响应文本
            llm_caller: LLM调用函数（依赖注入）
            context: 上下文信息用于错误日志

        返回：
            提取的JSON字符串，失败返回None
        """
        ...


__all__ = [
    "JSONRepairProtocol",
    "LLMCallProtocol",
    "TextProcessorProtocol",
]

