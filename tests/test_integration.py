"""
集成测试：验证模块间协作和完整工作流
"""

import sys
import unittest
from pathlib import Path

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class TestModuleIntegration(unittest.TestCase):
    """测试模块集成"""

    def test_import_all_modules(self):
        """测试所有新模块可以正常导入"""
        try:
            # 导入新创建的模块
            from config import constants
            from services.llm import message_processor, retry_strategy
            from utils import cache_manager, json_repair, performance_monitor, text_normalizer

            # 验证关键类和函数存在
            self.assertTrue(hasattr(json_repair, 'repair_json_once'))
            self.assertTrue(hasattr(json_repair, 'JSONRepairEngine'))
            self.assertTrue(hasattr(text_normalizer, 'TextNormalizer'))
            self.assertTrue(hasattr(cache_manager, 'CacheManager'))
            self.assertTrue(hasattr(performance_monitor, 'PerformanceMonitor'))
            self.assertTrue(hasattr(message_processor, 'MessageProcessor'))
            self.assertTrue(hasattr(retry_strategy, 'RetryStrategy'))
            self.assertTrue(hasattr(constants, 'ModelLimits'))

        except ImportError as e:
            self.fail(f"模块导入失败: {e}")

    def test_llm_interaction_imports(self):
        """测试llm_interaction模块的导入依赖"""
        try:
            from services import llm_interaction

            # 验证关键函数存在
            self.assertTrue(hasattr(llm_interaction, 'call_ai'))
            self.assertTrue(hasattr(llm_interaction, 'call_ai_with_schema'))
            self.assertTrue(hasattr(llm_interaction, 'preflight_llm_connectivity'))

        except ImportError as e:
            self.fail(f"llm_interaction导入失败: {e}")

    def test_workflow_nodes_imports(self):
        """测试工作流节点导入"""
        try:
            from workflows.nodes import critique, polish, research

            # 验证节点函数存在
            self.assertTrue(hasattr(polish, 'polish_node'))
            self.assertTrue(hasattr(critique, 'critique_node'))
            self.assertTrue(hasattr(research, 'research_node'))

        except ImportError as e:
            self.fail(f"工作流节点导入失败: {e}")

    def test_config_integration(self):
        """测试配置模块集成"""
        try:
            from config import Config
            from config.constants import ModelLimits

            # 验证可以创建Config实例
            # 注意：不实际初始化，因为需要API密钥
            self.assertTrue(callable(Config))

            # 验证常量可访问
            self.assertIsInstance(ModelLimits.CONTEXT_LIMITS, dict)

        except Exception as e:
            self.fail(f"配置模块集成失败: {e}")


class TestCrossModuleFunctionality(unittest.TestCase):
    """测试跨模块功能"""

    def test_json_repair_with_text_normalizer(self):
        """测试JSON修复和文本规范化协作"""
        from utils.json_repair import repair_json_once
        from utils.text_normalizer import clean_text_artifacts

        # 先清理文本
        dirty_text = '{"name": "test\x08value", "count": 123}'
        cleaned = clean_text_artifacts(dirty_text)

        # 然后修复JSON
        class DummySchema:
            __name__ = "DummySchema"

        repaired, success = repair_json_once(cleaned, DummySchema)
        # 验证至少运行了
        self.assertIsInstance(repaired, str)

    def test_cache_with_performance_monitor(self):
        """测试缓存和性能监控协作"""
        from utils.cache_manager import CacheManager
        from utils.performance_monitor import track_time

        cache = CacheManager(ttl=60, max_size=10, namespace="test")

        @track_time
        def cached_operation(key: str):
            result = cache.get(key)
            if result is None:
                result = f"computed_{key}"
                cache.set(key, result)
            return result

        # 第一次调用应该计算
        result1 = cached_operation("key1")
        self.assertEqual(result1, "computed_key1")

        # 第二次应该从缓存获取
        result2 = cached_operation("key1")
        self.assertEqual(result2, "computed_key1")


class TestWorkflowComponentsReady(unittest.TestCase):
    """测试工作流组件准备就绪"""

    def test_graph_builder_imports(self):
        """测试图构建器导入"""
        try:
            from workflows.graph_builder import build_graph

            # 验证函数可调用
            self.assertTrue(callable(build_graph))

        except ImportError as e:
            self.fail(f"图构建器导入失败: {e}")

    def test_core_modules_imports(self):
        """测试核心模块导入"""
        try:
            from core import context_components
            from core.patch_manager import apply_fine_grained_edits
            from core.state_manager import WorkflowStateAdapter

            # 验证类和函数存在
            self.assertTrue(callable(WorkflowStateAdapter))
            self.assertTrue(callable(apply_fine_grained_edits))
            # 验证模块存在
            self.assertIsNotNone(context_components)

        except ImportError as e:
            self.fail(f"核心模块导入失败: {e}")


if __name__ == "__main__":
    unittest.main()

