"""
Polish模块content_assembler的单元测试
"""

import pytest

from planning.tool_definitions import PolishSection
from workflows.nodes.polish.content_assembler import (
    assemble_final_content,
    extract_document_title,
)


class TestExtractDocumentTitle:
    """测试文档标题提取"""

    def test_extract_from_h1(self):
        """测试从一级标题提取"""
        content = """# 主标题

## 第一章
内容
"""

        title = extract_document_title(content)
        assert title == "主标题"

    def test_extract_from_h2(self):
        """测试从二级标题提取"""
        content = """
## 第一章
内容
"""

        title = extract_document_title(content)
        assert "第一章" in title

    def test_no_title(self):
        """测试没有标题"""
        content = "只是普通文本"
        title = extract_document_title(content)
        assert title == "未命名文档"


class TestAssembleFinalContent:
    """测试最终内容组装"""

    def test_assemble_simple_sections(self):
        """测试组装简单章节"""
        sections = [
            PolishSection(
                section_id="sec1",
                title="引言",
                content="这是引言内容。",
                original_content="原引言",
                modifications=[],
                references=[],
                quality_metrics=None,
                word_count=10,
            ),
            PolishSection(
                section_id="sec2",
                title="理论基础",
                content="这是理论内容。",
                original_content="原理论",
                modifications=[],
                references=[],
                quality_metrics=None,
                word_count=10,
            ),
        ]

        result = assemble_final_content(sections, citation_manager=None)

        assert "引言" in result
        assert "理论基础" in result
        assert "这是引言内容" in result
        assert "这是理论内容" in result

    def test_assemble_with_document_title(self):
        """测试带文档标题的组装"""
        sections = [
            PolishSection(
                section_id="sec1",
                title="章节",
                content="内容",
                original_content="原内容",
                modifications=[],
                references=[],
                quality_metrics=None,
                word_count=10,
            ),
        ]

        result = assemble_final_content(sections, citation_manager=None, document_title="我的文档")

        assert "# 我的文档" in result

    def test_assemble_removes_duplicates(self):
        """测试去除重复章节"""
        sections = [
            PolishSection(
                section_id="sec1",
                title="引言",
                content="短内容",
                original_content="原内容",
                modifications=[],
                references=[],
                quality_metrics=None,
                word_count=3,
            ),
            PolishSection(
                section_id="sec2",
                title="引言",  # 同名章节
                content="这是更长的引言内容，包含更多信息。",
                original_content="原内容",
                modifications=[],
                references=[],
                quality_metrics=None,
                word_count=20,
            ),
        ]

        result = assemble_final_content(sections, citation_manager=None)

        # 应该只保留更长的那个
        assert "更长的引言内容" in result
        # 短的应该被过滤
        count = result.count("## 引言")
        assert count == 1  # 只有一个引言章节

    def test_empty_sections(self):
        """测试空章节列表"""
        result = assemble_final_content([], citation_manager=None)
        assert result == "" or "参考文献" in result  # 可能只有参考文献占位符


class TestSectionOrdering:
    """测试章节排序"""

    def test_intro_comes_first(self):
        """测试引言排在前面"""
        sections = [
            PolishSection(
                section_id="sec1",
                title="理论基础",
                content="理论内容",
                original_content="原内容",
                modifications=[],
                references=[],
                quality_metrics=None,
                word_count=10,
            ),
            PolishSection(
                section_id="sec2",
                title="引言",
                content="引言内容",
                original_content="原内容",
                modifications=[],
                references=[],
                quality_metrics=None,
                word_count=10,
            ),
        ]

        result = assemble_final_content(sections, citation_manager=None)

        # 检查结果包含两个章节
        assert "引言" in result
        assert "理论基础" in result

        # 引言应该排在理论之前
        intro_pos = result.find("引言")
        theory_pos = result.find("理论基础")

        if intro_pos > 0 and theory_pos > 0:
            assert intro_pos < theory_pos  # 引言在前


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
