"""
章节匹配功能的单元测试

测试增强的多方式章节匹配功能
"""

import pytest

from core.patch_manager import (
    _find_section_by_id,
    _find_section_by_multiple_methods,
    _find_section_by_title,
    _list_available_section_ids,
)


class TestFindSectionById:
    """测试通过section_id查找章节"""

    def test_find_section_with_valid_id(self):
        """测试有效的section_id"""
        document = """
## 引言与研究背景 <!-- section_id: f5e2b926-a31f-482c-bc32-ff3822a81e37 -->
这是引言内容。

## 理论基础 <!-- section_id: fc617217-e988-4c15-a3ee-6756f96c5522 -->
这是理论内容。
"""

        match = _find_section_by_id(document, "f5e2b926-a31f-482c-bc32-ff3822a81e37")
        assert match is not None
        content = match.group(1)
        assert "引言内容" in content

    def test_find_section_with_invalid_id(self):
        """测试无效的section_id"""
        document = """
## 章节 <!-- section_id: valid-id -->
内容
"""

        match = _find_section_by_id(document, "invalid-id")
        assert match is None


class TestFindSectionByTitle:
    """测试通过标题查找章节"""

    def test_find_section_by_exact_title(self):
        """测试精确标题匹配"""
        document = """
## 引言与研究背景
这是引言内容。

## 理论基础
这是理论内容。
"""

        match = _find_section_by_title(document, "引言与研究背景")
        assert match is not None
        content = match.group(1)
        assert "引言内容" in content

    def test_find_section_by_partial_title(self):
        """测试部分标题匹配"""
        document = """
## 引言与研究背景 <!-- section_id: abc123 -->
这是引言内容。

## 理论基础
这是理论内容。
"""

        match = _find_section_by_title(document, "研究背景")
        assert match is not None
        content = match.group(1)
        assert "引言内容" in content

    def test_find_section_case_insensitive(self):
        """测试大小写不敏感匹配"""
        document = """
## Introduction and Background
Content here.
"""

        match = _find_section_by_title(document, "introduction")
        assert match is not None


class TestListAvailableSectionIds:
    """测试列出可用的section_id"""

    def test_list_multiple_section_ids(self):
        """测试列出多个section_id"""
        document = """
## Section 1 <!-- section_id: id-001 -->
Content 1

## Section 2 <!-- section_id: id-002 -->
Content 2

## Section 3 <!-- section_id: id-003 -->
Content 3
"""

        ids = _list_available_section_ids(document)
        assert len(ids) == 3
        assert "id-001" in ids
        assert "id-002" in ids
        assert "id-003" in ids

    def test_list_no_section_ids(self):
        """测试没有section_id的情况"""
        document = """
## Section 1
Content 1
"""

        ids = _list_available_section_ids(document)
        assert len(ids) == 0


class TestFindSectionByMultipleMethods:
    """测试多方式章节查找"""

    def test_find_by_section_id_first(self):
        """测试优先使用section_id匹配"""
        document = """
## 引言 <!-- section_id: target-id -->
这是引言。

## 引言 <!-- section_id: other-id -->
这是另一个引言。
"""

        match, method = _find_section_by_multiple_methods(
            document,
            "target-id",
            "引言"
        )

        assert match is not None
        assert method == "section_id"
        assert "这是引言" in match.group(1)

    def test_fallback_to_title_match(self):
        """测试section_id失败时降级到标题匹配"""
        document = """
## 理论基础 <!-- section_id: other-id -->
这是理论内容。
"""

        match, method = _find_section_by_multiple_methods(
            document,
            "nonexistent-id",
            "理论基础"
        )

        assert match is not None
        assert method == "title"
        assert "理论内容" in match.group(1)

    def test_no_match_returns_none(self):
        """测试所有方法都失败时返回None"""
        document = """
## 章节 <!-- section_id: some-id -->
内容
"""

        match, method = _find_section_by_multiple_methods(
            document,
            "nonexistent-id",
            "不存在的标题"
        )

        assert match is None
        assert method is None


class TestSessionLogScenario:
    """测试session.log中的实际场景"""

    def test_session_log_id_not_found(self):
        """测试session.log中的章节ID未找到场景"""
        # 模拟session.log中的实际情况
        document = """
## 引言与研究背景 <!-- section_id: f5e2b926-a31f-482c-bc32-ff3822a81e37 -->
自由落体旋转稳定性研究关注物体在重力作用下下落时...

## 刚体旋转动力学基础 <!-- section_id: a1b2c3d4-e5f6-7890-abcd-ef1234567890 -->
根据欧拉动力学方程...
"""

        # 尝试查找session.log中失败的ID
        target_id = "ffeb8203-92c7-4a96-82aa-95730b8411f0"
        section_title = "刚体旋转动力学基础"

        match, method = _find_section_by_multiple_methods(
            document,
            target_id,
            section_title
        )

        # 应该通过标题匹配找到
        assert match is not None
        assert method == "title"
        assert "欧拉动力学方程" in match.group(1)


class TestEdgeCases:
    """测试边界情况"""

    def test_empty_document(self):
        """测试空文档"""
        match, method = _find_section_by_multiple_methods("", "any-id", "任何标题")
        assert match is None

    def test_empty_section_id(self):
        """测试空section_id"""
        document = "## Section\nContent"
        match, method = _find_section_by_multiple_methods(document, "", "Section")
        # 应该降级到标题匹配
        assert match is not None
        assert method == "title"

    def test_special_characters_in_title(self):
        """测试标题中包含特殊字符"""
        document = """
## 第1章：引言 (Introduction)
内容
"""

        match = _find_section_by_title(document, "第1章")
        assert match is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

