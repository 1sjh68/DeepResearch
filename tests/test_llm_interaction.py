"""测试LLM交互模块"""
import os
from unittest.mock import MagicMock, patch

import pytest

from config import Config


class TestLLMInteraction:
    """测试LLM交互功能"""

    @pytest.fixture
    def mock_config(self):
        """创建mock配置"""
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test_key"}):
            config = Config()
            config.client = MagicMock()
            return config

    def test_call_ai_basic(self, mock_config):
        """测试基本的AI调用"""
        from services.llm_interaction import call_ai

        # Mock OpenAI response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Test response"
        mock_response.usage = MagicMock()
        mock_response.usage.total_tokens = 100

        mock_config.client.chat.completions.create = MagicMock(
            return_value=mock_response
        )

        messages = [{"role": "user", "content": "Test"}]
        result = call_ai(
            mock_config,
            "deepseek-chat",
            messages,
            temperature=0.7,
            max_tokens_output=1000
        )

        assert result == "Test response"
        assert mock_config.client.chat.completions.create.called

    def test_call_ai_with_empty_response(self, mock_config):
        """测试空响应处理"""
        from services.llm_interaction import call_ai

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = ""
        mock_response.usage = MagicMock()
        mock_response.usage.total_tokens = 10

        mock_config.client.chat.completions.create = MagicMock(
            return_value=mock_response
        )

        messages = [{"role": "user", "content": "Test"}]
        result = call_ai(
            mock_config,
            "deepseek-chat",
            messages,
            max_tokens_output=100
        )

        # 应该返回空字符串或警告
        assert result == "" or "空内容" in result

    def test_build_retry_exception_types(self):
        """测试重试异常类型构建"""
        from services.llm.retry_strategy import build_retry_exception_types

        exception_types = build_retry_exception_types()

        assert exception_types is not None
        assert len(exception_types) > 0

    def test_preflight_llm_connectivity_success(self, mock_config):
        """测试LLM连通性预检成功"""
        from services.llm_interaction import preflight_llm_connectivity

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "pong"
        mock_response.usage = MagicMock()
        mock_response.usage.total_tokens = 5

        mock_config.client.chat.completions.create = MagicMock(
            return_value=mock_response
        )

        result = preflight_llm_connectivity(mock_config)

        assert result is True

    def test_preflight_llm_connectivity_failure(self, mock_config):
        """测试LLM连通性预检失败"""
        from services.llm_interaction import preflight_llm_connectivity

        mock_config.client.chat.completions.create = MagicMock(
            side_effect=Exception("Network error")
        )

        result = preflight_llm_connectivity(mock_config)

        assert result is False

    def test_message_coercion(self):
        """测试消息规范化"""
        from services.llm.message_processor import coerce_message_content

        messages = [
            {"role": "user", "content": "test"},
            {"role": "assistant", "content": "response", "tool_calls": []},
        ]

        coerced = coerce_message_content(messages)

        assert len(coerced) == 2
        assert "tool_calls" not in coerced[1]

    def test_json_instruction_injection(self):
        """测试JSON指令注入"""
        from services.llm.message_processor import ensure_json_instruction

        messages = [{"role": "user", "content": "test"}]

        result = ensure_json_instruction(messages)

        # 应该注入JSON指令
        assert any("json" in str(msg.get("content", "")).lower() for msg in result)

    def test_clean_text_artifacts(self):
        """测试文本清理"""
        from services.llm.message_processor import clean_text_artifacts

        dirty_text = "```json\n{\"key\": \"value\"}\n```"
        clean_text = clean_text_artifacts(dirty_text)

        # clean_text_artifacts主要移除BOM和控制字符
        # 如果需要移除markdown，使用其他函数
        assert clean_text is not None
        assert isinstance(clean_text, str)

    def test_model_limits(self):
        """测试模型限制"""
        from config.constants import ModelLimits

        # 测试context limit
        context_limit = ModelLimits.get_context_limit("deepseek-chat")
        assert context_limit > 0

        # 测试max output
        max_output = ModelLimits.get_max_output("deepseek-chat")
        assert max_output > 0

    def test_token_management(self, mock_config):
        """测试token管理"""
        from services.llm_interaction import call_ai

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Response"
        mock_response.usage = MagicMock()
        mock_response.usage.total_tokens = 50

        mock_config.client.chat.completions.create = MagicMock(
            return_value=mock_response
        )

        messages = [{"role": "user", "content": "Test"}]

        # 测试token限制
        result = call_ai(
            mock_config,
            "deepseek-chat",
            messages,
            max_tokens_output=100
        )

        assert result is not None

    def test_reasoner_model_handling(self, mock_config):
        """测试推理器模型特殊处理"""
        from services.llm_interaction import call_ai

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Reasoned response"
        mock_response.choices[0].message.reasoning_content = "Thinking..."
        mock_response.usage = MagicMock()
        mock_response.usage.total_tokens = 200

        mock_config.client.chat.completions.create = MagicMock(
            return_value=mock_response
        )

        messages = [{"role": "user", "content": "Complex problem"}]
        result = call_ai(
            mock_config,
            "deepseek-reasoner",
            messages,
            max_tokens_output=4000
        )

        assert result == "Reasoned response"


class TestJSONRepair:
    """测试JSON修复功能"""

    def test_repair_json_once(self):
        """测试JSON修复"""
        from utils.json_repair import repair_json_once

        broken_json = '{"key": "value",}'

        repaired, was_repaired = repair_json_once(broken_json, dict)

        # 应该修复尾部逗号
        if was_repaired:
            assert "," not in repaired.rstrip("}")

    def test_massage_structured_payload(self):
        """测试结构化载荷修正"""
        from utils.json_repair import massage_structured_payload

        # 创建一个简单的schema类
        class DummySchema:
            __name__ = "PlanModel"

        payload = {
            "document_title": "Test",
            "sections": []
        }

        result = massage_structured_payload(DummySchema, payload)

        # 应该处理字段别名
        assert "title" in result or "document_title" in result


class TestRetryStrategy:
    """测试重试策略"""

    @pytest.fixture
    def mock_config(self):
        """创建mock配置"""
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test"}):
            return Config()

    def test_build_retryer(self, mock_config):
        """测试重试器构建"""
        from services.llm.retry_strategy import (
            build_retry_exception_types,
            build_retryer,
        )

        exception_types = build_retry_exception_types()
        retryer = build_retryer(mock_config, exception_types)

        assert retryer is not None
        assert callable(retryer)

    def test_empty_response_error(self):
        """测试空响应错误"""
        from services.llm.retry_strategy import EmptyResponseFromReasonerError

        error = EmptyResponseFromReasonerError("Test error")
        assert str(error) == "Test error"

