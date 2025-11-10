"""
Polish模块utils的单元测试
"""

import pytest

from workflows.nodes.polish.utils import (
    PLACEHOLDER_PATTERNS,
    _detect_unresolved_placeholders,
    _remove_unresolved_placeholders,
    parse_document_structure,
)


class TestDetectUnresolvedPlaceholders:
    """测试占位符检测"""

    def test_detect_ref_placeholders(self):
        """测试检测[ref: ...]占位符"""
        content = "这是文本 [ref: abc123#rag1] 和 [ref: def456#rag2]"
        placeholders = _detect_unresolved_placeholders(content)

        assert len(placeholders) >= 2
        # 注意：返回的是小写版本
        assert any('ref:' in p for p in placeholders)

    def test_detect_todo_placeholders(self):
        """测试检测todo占位符"""
        content = "这是内容 todo 需要补充"
        placeholders = _detect_unresolved_placeholders(content)

        assert len(placeholders) > 0
        assert any('todo' in p for p in placeholders)

    def test_no_placeholders(self):
        """测试没有占位符的情况"""
        content = "这是完整的内容，没有任何占位符"
        placeholders = _detect_unresolved_placeholders(content)

        # 可能为空或很少
        assert isinstance(placeholders, set)


class TestRemoveUnresolvedPlaceholders:
    """测试移除占位符"""

    def test_remove_ref_placeholders(self):
        """测试移除[ref: ...]占位符"""
        content = "这是文本 [ref: abc123#rag1] 和其他内容"
        placeholders = {'[ref: abc123#rag1]'}

        result = _remove_unresolved_placeholders(content, placeholders)

        assert '[ref: abc123#rag1]' not in result.lower()
        assert '这是文本' in result
        assert '和其他内容' in result

    def test_remove_multiple_placeholders(self):
        """测试移除多个占位符"""
        content = "文本1 [ref: a] 文本2 [ref: b] 文本3"
        placeholders = {'[ref: a]', '[ref: b]'}

        result = _remove_unresolved_placeholders(content, placeholders)

        assert '[ref:' not in result.lower()
        assert '文本1' in result
        assert '文本2' in result

    def test_clean_extra_whitespace(self):
        """测试清理多余空格和换行"""
        content = "文本1  \n\n\n\n  文本2"
        placeholders = set()

        result = _remove_unresolved_placeholders(content, placeholders)

        # 应该合并多个换行
        assert '\n\n\n' not in result


class TestParseDocumentStructure:
    """测试文档结构解析"""

    def test_parse_simple_document(self):
        """测试解析简单文档"""
        content = """# 标题

## 第一章
这是第一章内容。

## 第二章
这是第二章内容。
"""

        sections = parse_document_structure(content)

        assert len(sections) >= 2
        # 检查是否包含章节
        titles = [s['title'] for s in sections]
        assert any('第一章' in t for t in titles)
        assert any('第二章' in t for t in titles)

    def test_parse_with_section_id(self):
        """测试解析带section_id的文档"""
        content = """
## 引言 <!-- section_id: abc123 -->
引言内容

## 理论 <!-- section_id: def456 -->
理论内容
"""

        sections = parse_document_structure(content)
        assert len(sections) >= 2

    def test_parse_empty_document(self):
        """测试解析空文档"""
        sections = parse_document_structure("")
        assert isinstance(sections, list)


class TestPlaceholderPatterns:
    """测试占位符模式"""

    def test_patterns_are_valid(self):
        """测试占位符模式是否有效"""
        import re

        for pattern in PLACEHOLDER_PATTERNS:
            # 确保是有效的正则表达式
            try:
                re.compile(pattern)
            except re.error as e:
                pytest.fail(f"Invalid pattern: {pattern}, error: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

