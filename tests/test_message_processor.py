"""
消息处理器的单元测试
"""

import unittest

from services.llm.message_processor import (
    MessageProcessor,
    clean_text_artifacts,
    coerce_message_content,
    ensure_json_instruction,
)


class TestCleanTextArtifacts(unittest.TestCase):
    """测试文本清理功能"""

    def test_clean_control_chars(self):
        """测试控制字符清理"""
        text = "hello\x08world"
        cleaned = clean_text_artifacts(text)
        self.assertNotIn("\x08", cleaned)

    def test_fix_latex_replacements(self):
        """测试LaTeX修复"""
        text = "\x08oldsymbol{x}"
        cleaned = clean_text_artifacts(text)
        self.assertIn(r"\\boldsymbol", cleaned)

    def test_fix_latex_exclamation(self):
        """测试LaTeX感叹号修复"""
        text = "!\\left( x \\right)"
        cleaned = clean_text_artifacts(text)
        # 检查是否修复了感叹号
        self.assertIn("\\left", cleaned)
        self.assertNotIn("!\\left", cleaned)


class TestEnsureJSONInstruction(unittest.TestCase):
    """测试JSON指令注入"""

    def test_add_json_instruction_when_missing(self):
        """测试缺少JSON指令时添加"""
        messages = [
            {"role": "user", "content": "Generate data"}
        ]
        result = ensure_json_instruction(messages)

        # 应该添加一个系统消息
        self.assertEqual(len(result), 2)
        self.assertEqual(result[1]["role"], "system")
        self.assertIn("JSON", result[1]["content"])

    def test_no_add_when_json_present(self):
        """测试已有JSON指令时不添加"""
        messages = [
            {"role": "user", "content": "Return JSON format"}
        ]
        result = ensure_json_instruction(messages)

        # 不应该添加新消息
        self.assertEqual(len(result), 1)


class TestCoerceMessageContent(unittest.TestCase):
    """测试消息内容规范化"""

    def test_string_content_preserved(self):
        """测试字符串内容保持不变"""
        messages = [
            {"role": "user", "content": "Hello"}
        ]
        result = coerce_message_content(messages)
        self.assertEqual(result[0]["content"], "Hello")

    def test_none_content_converted(self):
        """测试None内容转换为空字符串"""
        messages = [
            {"role": "user", "content": None}
        ]
        result = coerce_message_content(messages)
        self.assertEqual(result[0]["content"], "")

    def test_list_content_joined(self):
        """测试列表内容合并"""
        messages = [
            {"role": "user", "content": ["Hello", "World"]}
        ]
        result = coerce_message_content(messages)
        self.assertIn("Hello", result[0]["content"])
        self.assertIn("World", result[0]["content"])

    def test_tool_calls_serialized(self):
        """测试tool_calls序列化"""
        messages = [
            {
                "role": "assistant",
                "content": "Result",
                "tool_calls": [{"name": "tool1", "args": {}}]
            }
        ]
        result = coerce_message_content(messages)

        # tool_calls应该被移除并序列化到content中
        self.assertNotIn("tool_calls", result[0])
        self.assertIn("tool1", result[0]["content"])

    def test_function_call_serialized(self):
        """测试function_call序列化"""
        messages = [
            {
                "role": "assistant",
                "content": "Result",
                "function_call": {"name": "func1", "arguments": "{}"}
            }
        ]
        result = coerce_message_content(messages)

        # function_call应该被移除并序列化到content中
        self.assertNotIn("function_call", result[0])
        self.assertIn("func1", result[0]["content"])


class TestMessageProcessor(unittest.TestCase):
    """测试MessageProcessor类接口"""

    def test_process_messages(self):
        """测试处理消息"""
        messages = [
            {"role": "user", "content": "Hello"}
        ]
        result = MessageProcessor.process_messages(messages)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["content"], "Hello")

    def test_add_json_instruction(self):
        """测试添加JSON指令"""
        messages = [
            {"role": "user", "content": "Generate data"}
        ]
        result = MessageProcessor.add_json_instruction(messages)
        self.assertGreaterEqual(len(result), 1)

    def test_clean_text(self):
        """测试文本清理"""
        text = "hello\x08world"
        cleaned = MessageProcessor.clean_text(text)
        self.assertNotIn("\x08", cleaned)


if __name__ == "__main__":
    unittest.main()

