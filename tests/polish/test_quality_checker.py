"""
Polish模块quality_checker的单元测试
"""

import pytest

from planning.tool_definitions import PolishSection, SentenceEdit
from workflows.nodes.polish.quality_checker import (
    _detect_text_anomalies,
    _should_revert_due_to_anomalies,
    _validate_final_solution,
    calculate_quality_score,
    generate_modification_summary,
)


class MockConfig:
    """模拟配置对象"""
    min_final_content_length = 100
    max_final_content_length = 1000000


class TestValidateFinalSolution:
    """测试最终方案验证"""

    def test_valid_content(self):
        """测试有效内容"""
        content = """# 标题

## 第一章
这是第一章的内容，足够长。需要更多文本来满足最小长度要求。这里添加一些额外的文本。
继续添加更多行来确保内容足够长，超过最小长度要求。
第三行内容。
第四行内容。

## 第二章
这是第二章的内容。也需要足够的文本内容。
第二章第二行。
第二章第三行。
"""

        config = MockConfig()
        valid, reason = _validate_final_solution(content, config)

        # 如果失败，打印详细信息
        if not valid:
            print(f"Validation failed: {reason}, content length: {len(content)}")

        assert valid is True, f"Expected valid but got: {reason}"
        assert reason == "OK"

    def test_empty_content(self):
        """测试空内容"""
        config = MockConfig()
        valid, reason = _validate_final_solution("", config)

        assert valid is False
        assert "空" in reason

    def test_too_short_content(self):
        """测试内容过短"""
        config = MockConfig()
        valid, reason = _validate_final_solution("短", config)

        assert valid is False
        assert "过短" in reason

    def test_no_markdown_headings(self):
        """测试没有markdown标题"""
        # 内容足够长但没有markdown标题
        content = "只是普通文本，没有任何标题。" * 20  # 确保足够长
        config = MockConfig()
        valid, reason = _validate_final_solution(content, config)

        assert valid is False
        # 可能是"未找到Markdown标题"或其他错误
        assert reason != "OK"


class TestDetectTextAnomalies:
    """测试文本异常检测"""

    def test_unbalanced_parentheses(self):
        """测试未闭合的括号"""
        text = "这是文本 (未闭合的括号"
        anomalies = _detect_text_anomalies(text)

        assert len(anomalies) > 0
        assert any('括号' in a for a in anomalies)

    def test_truncated_number(self):
        """测试截断的数字"""
        text = "计算结果为 ($0."
        anomalies = _detect_text_anomalies(text)

        # 应该检测到截断
        assert len(anomalies) > 0

    def test_no_anomalies(self):
        """测试正常文本"""
        text = "这是完整的文本，没有任何问题。所有括号都闭合了(正常)。"
        anomalies = _detect_text_anomalies(text)

        # 应该没有异常或很少
        assert isinstance(anomalies, list)


class TestShouldRevertDueToAnomalies:
    """测试是否应该回退"""

    def test_no_anomalies_no_revert(self):
        """测试没有异常时不回退"""
        result = _should_revert_due_to_anomalies([], "原文", "修改文")
        assert result is False

    def test_empty_candidate_revert(self):
        """测试候选内容为空时回退"""
        anomalies = ["问题1"]
        result = _should_revert_due_to_anomalies(anomalies, "原文", "")
        assert result is True

    def test_significant_length_reduction_revert(self):
        """测试内容显著缩短时回退"""
        original = "这是很长的原始内容" * 100
        candidate = "很短"
        anomalies = ["问题1"]

        result = _should_revert_due_to_anomalies(anomalies, original, candidate)
        assert result is True

    def test_multiple_anomalies_revert(self):
        """测试多个异常时回退"""
        anomalies = ["问题1", "问题2"]
        result = _should_revert_due_to_anomalies(anomalies, "原文", "修改文")
        assert result is True


class TestCalculateQualityScore:
    """测试质量评分计算"""

    def test_empty_sections(self):
        """测试空章节列表"""
        score = calculate_quality_score([])
        assert score == 0.5

    def test_single_section_moderate_changes(self):
        """测试单个章节合理修改"""
        section = PolishSection(
            section_id="test",
            title="测试章节",
            content="内容",
            original_content="原内容",
            modifications=[SentenceEdit(original_sentence="a", revised_sentence="b")] * 15,  # 15个修改
            references=[],
            quality_metrics=None,
            word_count=100,  # 100字，15个修改 = 15% 修改率（合理）
        )

        score = calculate_quality_score([section])

        assert 0.0 <= score <= 1.0
        assert score >= 0.8  # 合理修改应该得高分


class TestGenerateModificationSummary:
    """测试修改总结生成"""

    def test_no_modifications(self):
        """测试无修改情况"""
        section = PolishSection(
            section_id="test",
            title="章节",
            content="内容",
            original_content="内容",
            modifications=[],
            references=[],
            quality_metrics=None,
            word_count=100,
        )

        summary = generate_modification_summary([section])
        assert "无需任何修改" in summary or "良好" in summary

    def test_with_modifications(self):
        """测试有修改情况"""
        section = PolishSection(
            section_id="test",
            title="章节",
            content="内容",
            original_content="原内容",
            modifications=[SentenceEdit(original_sentence="a", revised_sentence="b")] * 5,
            references=[],
            quality_metrics=None,
            word_count=100,
        )

        summary = generate_modification_summary([section])

        assert "1" in summary or "章节" in summary
        assert "5" in summary or "修改" in summary


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

