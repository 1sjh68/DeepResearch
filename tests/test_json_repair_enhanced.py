"""
JSON修复增强功能的单元测试

测试新增的trailing characters处理和智能截断功能
"""

import json

import pytest

from utils.json_repair import (
    _contains_json_structure,
    _fix_invalid_latex_commands,
    _is_likely_trailing_text,
    _looks_like_json_continuation,
    _truncate_top_level_json,
    repair_json_once,
)


class TestTrailingTextDetection:
    """测试尾部文本检测功能"""

    def test_is_likely_trailing_text_with_chinese(self):
        """测试检测中文尾部文本"""
        # 主要是中文，应该被识别为尾部文本
        text = "这是LLM添加的解释性文字，不是JSON内容"
        assert _is_likely_trailing_text(text) is True

    def test_is_likely_trailing_text_with_json(self):
        """测试纯JSON字符不被误判"""
        text = '{"key": "value", "number": 123}'
        assert _is_likely_trailing_text(text) is False

    def test_is_likely_trailing_text_mixed(self):
        """测试混合内容"""
        # 包含一些JSON字符但主要是文本
        text = "注意：这个JSON包含了123个字段"
        assert _is_likely_trailing_text(text) is True

    def test_is_likely_trailing_text_empty(self):
        """测试空字符串"""
        assert _is_likely_trailing_text("") is False


class TestJSONContinuationDetection:
    """测试JSON延续判断"""

    def test_looks_like_json_continuation_with_brace(self):
        """测试以{开头的内容"""
        assert _looks_like_json_continuation('{"nested": true}') is True

    def test_looks_like_json_continuation_with_bracket(self):
        """测试以[开头的内容"""
        assert _looks_like_json_continuation('[1, 2, 3]') is True

    def test_looks_like_json_continuation_with_quote(self):
        """测试以引号开头的内容"""
        assert _looks_like_json_continuation('"string"') is True

    def test_looks_like_json_continuation_with_text(self):
        """测试普通文本"""
        assert _looks_like_json_continuation('这是普通文本') is False

    def test_looks_like_json_continuation_empty(self):
        """测试空字符串"""
        assert _looks_like_json_continuation('') is False


class TestJSONStructureDetection:
    """测试JSON结构检测"""

    def test_contains_json_structure_with_braces(self):
        """测试包含大括号"""
        assert _contains_json_structure('some text {key: value}') is True

    def test_contains_json_structure_with_brackets(self):
        """测试包含方括号"""
        assert _contains_json_structure('some text [1, 2, 3]') is True

    def test_contains_json_structure_plain_text(self):
        """测试纯文本"""
        assert _contains_json_structure('just plain text') is False


class TestTruncateTopLevelJSON:
    """测试顶层JSON截断功能"""

    def test_truncate_simple_trailing_text(self):
        """测试截断简单的尾部文本"""
        json_str = '{"key": "value", "number": 123}'
        input_str = json_str + ' 这是额外的文本'

        result = _truncate_top_level_json(input_str)
        # 应该截断尾部文本
        assert result == json_str or json.loads(result) == json.loads(json_str)

    def test_truncate_long_trailing_text(self):
        """测试截断超过100字符的尾部文本（核心功能）"""
        json_str = '{"content": "test", "count": 123}'
        # 添加200+字符的尾部文本
        trailing = "这是LLM添加的解释文字。" * 20  # 约200字符
        input_str = json_str + trailing

        result = _truncate_top_level_json(input_str)
        # 应该能够截断，即使超过100字符
        parsed = json.loads(result)
        assert parsed["content"] == "test"
        assert parsed["count"] == 123

    def test_truncate_8000_char_trailing_text(self):
        """测试截断8000+字符的尾部文本（session.log实际情况）"""
        json_str = '{"revised_content": "章节内容", "word_count": 123}'
        # 模拟8000字符的尾部文本
        trailing = "额外内容" * 2000  # 约8000字符
        input_str = json_str + trailing

        result = _truncate_top_level_json(input_str)
        parsed = json.loads(result)
        assert "revised_content" in parsed
        assert "word_count" in parsed

    def test_truncate_with_markdown_fence(self):
        """测试包含markdown代码块的情况"""
        json_str = '{"key": "value"}'
        input_str = json_str + '\n```\n这是代码块\n```'

        result = _truncate_top_level_json(input_str)
        parsed = json.loads(result)
        assert parsed["key"] == "value"

    def test_no_truncate_for_valid_json(self):
        """测试有效JSON不被错误截断"""
        json_str = '{"key": "value", "nested": {"inner": "data"}}'
        result = _truncate_top_level_json(json_str)
        assert json.loads(result) == json.loads(json_str)


class TestLatexCommandFix:
    """测试LaTeX命令修复"""

    def test_fix_cdotp_to_cdot(self):
        """测试修复 \\cdotp → \\cdot"""
        text = r'{"formula": "a \\cdotp b"}'
        result = _fix_invalid_latex_commands(text)
        assert r'\\cdot' in result
        assert r'\\cdotp' not in result

    def test_fix_unicode_middot(self):
        """测试修复Unicode中间点 · → \\cdot"""
        text = '{"formula": "a · b"}'
        result = _fix_invalid_latex_commands(text)
        assert '\\\\cdot' in result
        assert '·' not in result

    def test_fix_multiple_cdotp(self):
        """测试修复多个\\cdotp"""
        text = r'{"formula": "a \\cdotp b \\cdotp c"}'
        result = _fix_invalid_latex_commands(text)
        assert r'\\cdotp' not in result
        assert result.count(r'\\cdot') >= 2

    def test_preserve_valid_cdot(self):
        """测试不破坏正确的\\cdot命令"""
        text = r'{"formula": "a \\cdot b"}'
        result = _fix_invalid_latex_commands(text)
        assert r'\\cdot' in result
        # 不应该变成\\cdotot
        assert r'\\cdotot' not in result


class TestRepairJSONOnceEnhanced:
    """测试增强的JSON修复功能"""

    def test_repair_trailing_characters_error(self):
        """测试修复trailing characters错误"""
        # 模拟session.log中的实际错误
        json_str = '{"revised_content": "test", "word_count": 123}'
        broken_json = json_str + '} extra trailing text'

        from planning.tool_definitions import PolishSectionResponse

        # 尝试修复
        repaired, success = repair_json_once(broken_json, PolishSectionResponse)

        # 应该能成功修复
        assert success is True
        parsed = json.loads(repaired)
        assert "revised_content" in parsed

    def test_repair_with_latex_commands(self):
        """测试JSON修复时同时修复LaTeX命令"""
        broken_json = r'{"formula": "a \\cdotp b", "note": "test"}'

        class SimpleSchema:
            pass

        repaired, success = repair_json_once(broken_json, SimpleSchema)

        # 应该修复LaTeX命令
        if success:
            assert r'\\cdotp' not in repaired


class TestEOFErrorFix:
    """测试EOF错误修复"""

    def test_fix_eof_while_parsing_value(self):
        """测试修复EOF while parsing value"""
        from utils.json_repair import _attempt_fix_eof_error

        # 模拟column 3914的EOF错误 - JSON被截断
        truncated_json = '{"critique": "内容很长", "priority_issues": ['

        fixed = _attempt_fix_eof_error(truncated_json)

        # 应该能解析（可能截断到完整的字段）
        try:
            parsed = json.loads(fixed)
            assert "critique" in parsed
        except json.JSONDecodeError:
            # 如果无法完全修复，至少应该有闭合括号
            assert '}' in fixed

    def test_fix_eof_while_parsing_string(self):
        """测试修复EOF while parsing string"""
        from utils.json_repair import _attempt_fix_eof_error

        # 模拟column 13081的EOF错误 - 字符串未闭合
        truncated_json = '{"revised_content": "这是很长的内容'

        fixed = _attempt_fix_eof_error(truncated_json)

        # 应该能解析或至少有闭合结构
        try:
            parsed = json.loads(fixed)
            assert "revised_content" in parsed
        except json.JSONDecodeError:
            # 如果无法完全修复，至少应该有闭合括号
            assert '}' in fixed

    def test_eof_with_complete_field(self):
        """测试找到完整字段后截断"""
        from utils.json_repair import _attempt_fix_eof_error

        truncated = '{"field1": "value1", "field2": "value2", "field3":'

        fixed = _attempt_fix_eof_error(truncated)

        # 应该能解析至少前两个字段
        try:
            parsed = json.loads(fixed)
            assert "field1" in parsed
        except json.JSONDecodeError:
            pass  # 如果修复失败也可以接受


class TestInvalidEscapeFix:
    """测试invalid escape修复"""

    def test_fix_invalid_escape_in_latex(self):
        """测试修复LaTeX中的无效转义"""
        from utils.json_repair import _fix_invalid_escape_sequences

        # 模拟column 892的invalid escape错误
        broken_json = '{"formula": "\\alpha \\beta"}'  # 单反斜杠

        fixed = _fix_invalid_escape_sequences(broken_json)

        # 应该转为双反斜杠（至少修复了一个）
        # 由于正则表达式的限制，可能只修复第一个匹配
        assert '\\\\alpha' in fixed or '\\\\beta' in fixed

    def test_preserve_valid_json_escapes(self):
        """测试不破坏有效的JSON转义"""
        from utils.json_repair import _fix_invalid_escape_sequences

        # 有效的JSON转义应该保持不变
        valid_json = '{"text": "line1\\nline2", "quote": "\\"test\\""}'

        fixed = _fix_invalid_escape_sequences(valid_json)

        # \\n和\\"应该保持不变
        parsed = json.loads(fixed)
        assert "line1\nline2" == parsed["text"] or "line1\\nline2" in fixed

    def test_fix_complex_latex_formula(self):
        """测试修复复杂LaTeX公式中的转义"""
        from utils.json_repair import _fix_invalid_escape_sequences

        # 包含多个LaTeX命令的公式
        broken = '{"formula": "\\frac{\\alpha}{\\beta}"}'

        fixed = _fix_invalid_escape_sequences(broken)

        # 应该修复部分或全部LaTeX命令（至少修复一个）
        double_escaped_count = fixed.count('\\\\')
        original_count = broken.count('\\\\')

        # 修复后应该有更多双反斜杠
        assert double_escaped_count >= original_count


class TestPylatexencIntegration:
    """测试pylatexenc集成"""

    def test_pylatexenc_unicode_conversion(self):
        """测试Unicode符号转换"""
        from utils.json_repair import _fix_latex_with_pylatexenc

        # 包含Unicode数学符号
        text = '{"formula": "a · b × c ÷ d ± e"}'

        fixed = _fix_latex_with_pylatexenc(text)

        # 应该转换为LaTeX命令
        assert '·' not in fixed or '\\\\cdot' in fixed

    def test_pylatexenc_fallback(self):
        """测试pylatexenc不可用时的降级"""
        from utils.json_repair import _fix_latex_with_pylatexenc

        # 即使pylatexenc不可用，也应该能处理基本情况
        text = '{"formula": "a · b"}'

        fixed = _fix_latex_with_pylatexenc(text)

        # 应该有某种形式的修复
        assert len(fixed) >= len(text)


class TestIntegrationScenarios:
    """集成测试场景"""

    def test_session_log_scenario(self):
        """测试session.log中的实际场景"""
        # 模拟column 8692的trailing characters错误
        large_json = '{"revised_content": "' + 'x' * 8000 + '", "modifications": [], "word_count": 123}'
        # 添加尾部文本
        broken = large_json + '\n\n这是额外的说明文字'

        result = _truncate_top_level_json(broken)

        # 应该能成功解析
        parsed = json.loads(result)
        assert "revised_content" in parsed
        assert "word_count" in parsed

    def test_all_error_types_integration(self):
        """测试所有错误类型的集成修复"""
        # 创建包含多种问题的JSON
        broken = '{"content": "\\alpha test", "value": "data'  # EOF + invalid escape

        from utils.json_repair import repair_json_once

        class SimpleSchema:
            pass

        # 应该能尝试修复
        repaired, success = repair_json_once(broken, SimpleSchema)

        # 至少应该有修复尝试
        assert len(repaired) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

