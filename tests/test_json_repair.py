"""
JSON修复模块的单元测试
"""

import json
import unittest
from typing import Any

from pydantic import BaseModel

from utils.json_repair import (
    JSONRepairEngine,
    massage_structured_payload,
    repair_json_once,
)


class SimpleModel(BaseModel):
    """测试用的简单模型"""
    name: str
    value: int


class PlanModel(BaseModel):
    """测试用的Plan模型"""
    title: str
    outline: list[dict[str, Any]]


class TestJSONRepair(unittest.TestCase):
    """测试JSON修复功能"""

    def test_repair_simple_json(self):
        """测试简单JSON修复"""
        # 缺少引号
        text = '{"name": test, "value": 123}'
        repaired, success = repair_json_once(text, SimpleModel)
        self.assertTrue(success or repaired != text)

    def test_repair_with_markdown_fence(self):
        """测试去除markdown代码块"""
        text = '```json\n{"name": "test", "value": 123}\n```'
        repaired, success = repair_json_once(text, SimpleModel)
        self.assertTrue(success)
        data = json.loads(repaired)
        self.assertEqual(data["name"], "test")

    def test_massage_structured_payload_plan_model(self):
        """测试PlanModel的载荷按摩"""
        payload = {
            "document_title": "测试文档",  # 别名
            "chapters": [  # 别名
                {"section_title": "章节1", "description": "描述"}
            ]
        }

        result = massage_structured_payload(PlanModel, payload)

        # 应该转换别名
        self.assertIn("title", result)
        self.assertIn("outline", result)
        self.assertEqual(result["title"], "测试文档")

    def test_json_repair_engine_interface(self):
        """测试JSONRepairEngine类接口"""
        engine = JSONRepairEngine(debug=False)

        text = '{"name": "test", "value": 123}'
        repaired, success = engine.repair(text, SimpleModel)
        self.assertTrue(success)

    def test_normalize_mapping(self):
        """测试映射规范化"""
        engine = JSONRepairEngine()

        data = {"key1": "value1", "key2": 123}
        result = engine.normalize_mapping(data)

        self.assertIsInstance(result, dict)
        self.assertEqual(result["key1"], "value1")


class TestJSONRepairEdgeCases(unittest.TestCase):
    """测试JSON修复的边缘情况"""

    def test_empty_input(self):
        """测试空输入"""
        repaired, success = repair_json_once("", SimpleModel)
        self.assertFalse(success)

    def test_unbalanced_brackets(self):
        """测试不平衡的括号"""
        text = '{"name": "test", "value": 123'
        repaired, success = repair_json_once(text, SimpleModel)
        # 应该尝试修复
        self.assertIn("}", repaired)

    def test_unicode_quotes(self):
        """测试Unicode引号"""
        text = '{"name": "test", "value": 123}'
        repaired, success = repair_json_once(text, SimpleModel)
        self.assertNotIn(""", repaired)
        self.assertNotIn(""", repaired)

    def test_trailing_text_cleanup(self):
        """测试尾部非JSON文本截断"""
        text = '{"name": "test", "value": 123}} extra-notes'
        repaired, success = repair_json_once(text, SimpleModel)
        self.assertTrue(success)
        self.assertTrue(repaired.strip().endswith("}"), msg=repaired)
        data = json.loads(repaired)
        self.assertEqual(data["name"], "test")

    def test_invalid_escape_sequences(self):
        """测试修复未转义的数学命令"""
        text = '{"name": "\\omega 项", "value": 1}'
        repaired, success = repair_json_once(text, SimpleModel)
        self.assertTrue(success)
        # 修复后的JSON应该能被解析并保留双反斜杠
        data = json.loads(repaired)
        self.assertIn("\\\\omega", repaired)
        self.assertEqual(data["name"], "\\omega 项")

    def test_negative_word_count_sanitized(self):
        """测试负 word_count 自动归零"""

        class SectionListModel(BaseModel):
            sections: list[dict[str, Any]]

        text = '{"sections": [{"title": "demo", "_word_count": -5}]}'
        repaired, success = repair_json_once(text, SectionListModel)
        self.assertTrue(success)
        payload = json.loads(repaired)
        self.assertGreaterEqual(payload["sections"][0]["_word_count"], 0)


if __name__ == "__main__":
    unittest.main()

