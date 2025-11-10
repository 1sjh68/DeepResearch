"""
文本规范化模块的单元测试
"""

import unittest

from utils.text_normalizer import (
    TextNormalizer,
    clean_text_artifacts,
    normalize_unicode_quotes,
    normalize_whitespace,
    remove_control_characters,
    strip_markdown_fence,
)


class TestCleanTextArtifacts(unittest.TestCase):
    """测试文本artifacts清理"""

    def test_remove_control_chars(self):
        """测试移除控制字符"""
        text = "hello\x08world\x07test"
        cleaned = clean_text_artifacts(text)
        self.assertNotIn("\x08", cleaned)
        self.assertNotIn("\x07", cleaned)

    def test_fix_latex_commands(self):
        """测试修复LaTeX命令"""
        text = "\x08oldsymbol{x} + \x0crac{1}{2}"
        cleaned = clean_text_artifacts(text)
        self.assertIn(r"\\boldsymbol", cleaned)
        self.assertIn(r"\\frac", cleaned)


class TestNormalizeWhitespace(unittest.TestCase):
    """测试空白字符规范化"""

    def test_multiple_spaces(self):
        """测试多个空格压缩"""
        text = "hello    world"
        normalized = normalize_whitespace(text)
        self.assertEqual(normalized, "hello world")

    def test_multiple_newlines(self):
        """测试多个换行压缩"""
        text = "line1\n\n\n\nline2"
        normalized = normalize_whitespace(text)
        self.assertEqual(normalized, "line1\n\nline2")

    def test_trailing_spaces(self):
        """测试移除行尾空格"""
        text = "hello world   \nfoo bar  "
        normalized = normalize_whitespace(text)
        self.assertNotIn("   \n", normalized)


class TestRemoveControlCharacters(unittest.TestCase):
    """测试控制字符移除"""

    def test_remove_disallowed_chars(self):
        """测试移除不允许的控制字符"""
        text = "hello\x00\x01world"
        cleaned = remove_control_characters(text)
        self.assertNotIn("\x00", cleaned)
        self.assertNotIn("\x01", cleaned)
        self.assertEqual(cleaned, "helloworld")

    def test_keep_allowed_chars(self):
        """测试保留允许的字符"""
        text = "hello\tworld\nfoo\rbar"
        cleaned = remove_control_characters(text)
        self.assertIn("\t", cleaned)
        self.assertIn("\n", cleaned)
        self.assertIn("\r", cleaned)

    def test_custom_allowed(self):
        """测试自定义允许字符"""
        text = "hello\tworld\nfoo"
        cleaned = remove_control_characters(text, allowed={'\n'})
        self.assertNotIn("\t", cleaned)
        self.assertIn("\n", cleaned)


class TestStripMarkdownFence(unittest.TestCase):
    """测试Markdown代码块移除"""

    def test_remove_json_fence(self):
        """测试移除JSON代码块"""
        text = "```json\n{\"key\": \"value\"}\n```"
        stripped = strip_markdown_fence(text)
        self.assertNotIn("```", stripped)
        self.assertIn("{\"key\": \"value\"}", stripped)

    def test_remove_generic_fence(self):
        """测试移除通用代码块"""
        text = "```\nsome code\n```"
        stripped = strip_markdown_fence(text)
        self.assertNotIn("```", stripped)

    def test_no_fence_unchanged(self):
        """测试没有代码块时不变"""
        text = "just plain text"
        stripped = strip_markdown_fence(text)
        self.assertEqual(stripped, text)


class TestNormalizeUnicodeQuotes(unittest.TestCase):
    """测试Unicode引号规范化"""

    def test_function_exists(self):
        """测试函数存在且可调用"""
        text = "hello world"
        result = normalize_unicode_quotes(text)
        self.assertIsInstance(result, str)
        self.assertEqual(result, "hello world")

    def test_empty_input(self):
        """测试空输入"""
        result = normalize_unicode_quotes("")
        self.assertEqual(result, "")

    def test_normal_quotes_unchanged(self):
        """测试普通引号不变"""
        text = '"hello" and \'world\''
        result = normalize_unicode_quotes(text)
        self.assertEqual(result, text)


class TestTextNormalizerClass(unittest.TestCase):
    """测试TextNormalizer类"""

    def setUp(self):
        self.normalizer = TextNormalizer()

    def test_clean_method(self):
        """测试clean方法"""
        text = "hello\x08world"
        cleaned = self.normalizer.clean(text)
        self.assertNotIn("\x08", cleaned)

    def test_normalize_all(self):
        """测试normalize_all方法"""
        # 测试多余空格和换行的规范化
        text = "  hello    world  \n\n\n\ntest"
        normalized = self.normalizer.normalize_all(text)

        # 应该应用所有规范化
        self.assertNotIn("    ", normalized)  # 多余空格
        self.assertNotIn("\n\n\n", normalized)  # 多余换行
        # 验证基本内容保留
        self.assertIn("hello", normalized)
        self.assertIn("world", normalized)


if __name__ == "__main__":
    unittest.main()

