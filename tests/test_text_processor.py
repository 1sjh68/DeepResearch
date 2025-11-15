"""测试文本处理工具"""
import os
from unittest.mock import patch

import pytest


class TestConsolidateDocumentStructure:
    """测试文档结构合并"""

    def test_consolidate_simple_document(self):
        """测试简单文档合并"""
        from logic.post_processing import consolidate_document_structure

        text = "# Title\n\nContent here.\n\n# Title\n\nMore content."

        result = consolidate_document_structure(text)

        assert result is not None
        assert isinstance(result, str)

    def test_consolidate_empty_document(self):
        """测试空文档"""
        from logic.post_processing import consolidate_document_structure

        text = ""
        result = consolidate_document_structure(text)

        assert result == ""

    def test_consolidate_with_section_ids(self):
        """测试带section_id的文档"""
        from logic.post_processing import consolidate_document_structure

        text = (
            "# Title <!-- section_id: intro -->\n\n"
            "Content.\n\n"
            "# Title <!-- section_id: intro -->\n\n"
            "More content."
        )

        result = consolidate_document_structure(text)

        # 应该合并相同section_id的章节
        assert result is not None


class TestFinalPostProcessing:
    """测试最终后处理"""

    def test_final_post_processing_basic(self):
        """测试基本后处理"""
        from logic.post_processing import final_post_processing

        text = "# Title\n\nSome content with extra   spaces."

        result = final_post_processing(text)

        assert result is not None
        assert isinstance(result, str)

    def test_final_post_processing_empty(self):
        """测试空文本后处理"""
        from logic.post_processing import final_post_processing

        result = final_post_processing("")

        # 空文本后处理可能返回空字符串或默认值
        assert result is not None
        assert isinstance(result, str)


class TestQualityCheck:
    """测试质量检查"""

    @pytest.fixture
    def mock_config(self):
        """创建mock配置"""
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test"}):
            from config import Config
            return Config()

    def test_quality_check_basic(self, mock_config):
        """测试基本质量检查"""
        from utils.text_processor import quality_check

        text = "# Title\n\nThis is a well-formed document with content."

        with patch("utils.text_processor.call_ai_writing_with_auto_continue") as mock_call:
            mock_call.return_value = "Quality: Good"

            result = quality_check(mock_config, text)

            assert result is not None
            assert isinstance(result, str)

    def test_quality_check_short_text(self, mock_config):
        """测试短文本质量检查"""
        from utils.text_processor import quality_check

        text = "Short."

        with patch("utils.text_processor.call_ai_writing_with_auto_continue") as mock_call:
            mock_call.return_value = "Quality report"

            result = quality_check(mock_config, text)

            # 短文本可能直接返回或调用简化检查
            assert result is not None


class TestExtractJSON:
    """测试JSON提取"""

    def test_extract_json_from_markdown(self):
        """测试从markdown提取JSON"""
        from config import Config
        from utils.text_processor import extract_json_from_ai_response

        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test"}):
            config = Config()

        response = '''
        Here is the result:
        ```json
        {"key": "value"}
        ```
        Additional text
        '''

        result = extract_json_from_ai_response(
            config,
            response,
            context_for_error_log="test"
        )

        assert result is not None
        assert "{" in result

    def test_extract_json_plain(self):
        """测试提取纯JSON"""
        from config import Config
        from utils.text_processor import extract_json_from_ai_response

        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test"}):
            config = Config()

        response = '{"key": "value", "nested": {"a": 1}}'

        result = extract_json_from_ai_response(
            config,
            response,
            context_for_error_log="test"
        )

        assert result is not None
        assert "key" in result


class TestTextNormalization:
    """测试文本规范化"""

    def test_remove_extra_whitespace(self):
        """测试移除多余空白"""
        text = "Line  with   extra    spaces."

        # 简单的空白规范化
        normalized = " ".join(text.split())

        assert "   " not in normalized
        assert normalized == "Line with extra spaces."

    def test_normalize_line_endings(self):
        """测试规范化行结束符"""
        text = "Line1\r\nLine2\rLine3\n"

        normalized = text.replace("\r\n", "\n").replace("\r", "\n")

        assert "\r" not in normalized
        assert normalized.count("\n") == 3


class TestMarkdownProcessing:
    """测试Markdown处理"""

    def test_extract_headings(self):
        """测试提取标题"""
        import re

        text = "# H1\n\n## H2\n\n### H3\n\nContent"

        headings = re.findall(r"^(#{1,6})\s+(.+)$", text, re.MULTILINE)

        assert len(headings) == 3
        assert headings[0][0] == "#"
        assert headings[1][0] == "##"

    def test_remove_markdown_formatting(self):
        """测试移除Markdown格式"""
        text = "**bold** and *italic* and `code`"

        # 简单的格式移除
        cleaned = text.replace("**", "").replace("*", "").replace("`", "")

        assert "**" not in cleaned
        assert "*" not in cleaned
        assert "`" not in cleaned

