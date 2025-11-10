"""
配置常量模块的单元测试
"""

import unittest

from config.constants import (
    CacheConfig,
    JSONRepairConfig,
    ModelLimits,
    RetryConfig,
    TokenBudget,
    WorkflowConfig,
)


class TestModelLimits(unittest.TestCase):
    """测试模型限制配置"""

    def test_context_limits(self):
        """测试上下文限制"""
        self.assertEqual(ModelLimits.CONTEXT_LIMITS["deepseek-chat"], 64000)
        self.assertEqual(ModelLimits.CONTEXT_LIMITS["deepseek-coder"], 128000)

    def test_max_output_tokens(self):
        """测试最大输出tokens"""
        self.assertEqual(ModelLimits.MAX_OUTPUT_TOKENS["deepseek-chat"], 8192)
        self.assertEqual(ModelLimits.MAX_OUTPUT_TOKENS["deepseek-reasoner"], 4096)

    def test_get_context_limit(self):
        """测试获取上下文限制方法"""
        limit = ModelLimits.get_context_limit("deepseek-chat")
        self.assertEqual(limit, 64000)

        # 测试默认值
        default_limit = ModelLimits.get_context_limit("unknown-model")
        self.assertEqual(default_limit, 64000)

    def test_get_max_output(self):
        """测试获取最大输出方法"""
        max_out = ModelLimits.get_max_output("deepseek-chat")
        self.assertEqual(max_out, 8192)

        # 测试默认值
        default_max = ModelLimits.get_max_output("unknown-model")
        self.assertEqual(default_max, 4096)

    def test_reasoner_min_tokens(self):
        """测试Reasoner最小tokens"""
        self.assertGreaterEqual(ModelLimits.REASONER_MIN_TOKENS, 2000)


class TestRetryConfig(unittest.TestCase):
    """测试重试配置"""

    def test_max_attempts(self):
        """测试最大重试次数"""
        self.assertGreaterEqual(RetryConfig.MAX_ATTEMPTS, 1)

    def test_backoff_multiplier(self):
        """测试退避乘数"""
        self.assertGreater(RetryConfig.BACKOFF_MULTIPLIER, 1.0)

    def test_max_wait(self):
        """测试最大等待时间"""
        self.assertGreater(RetryConfig.MAX_WAIT, 0)


class TestTokenBudget(unittest.TestCase):
    """测试Token预算配置"""

    def test_prompt_ratio(self):
        """测试提示词比例"""
        self.assertGreater(TokenBudget.PROMPT_RATIO, 0)
        self.assertLess(TokenBudget.PROMPT_RATIO, 1)

    def test_safety_margin(self):
        """测试安全边距"""
        self.assertGreaterEqual(TokenBudget.SAFETY_MARGIN, 0)

    def test_min_output_tokens(self):
        """测试最小输出tokens"""
        self.assertGreater(TokenBudget.MIN_OUTPUT_TOKENS, 0)


class TestCacheConfig(unittest.TestCase):
    """测试缓存配置"""

    def test_research_cache_ttl(self):
        """测试研究缓存TTL"""
        self.assertGreater(CacheConfig.RESEARCH_CACHE_TTL, 0)

    def test_cache_max_size(self):
        """测试缓存最大大小"""
        self.assertGreater(CacheConfig.RESEARCH_CACHE_MAX_SIZE, 0)
        self.assertGreater(CacheConfig.NETWORK_CACHE_MAX_SIZE, 0)


class TestJSONRepairConfig(unittest.TestCase):
    """测试JSON修复配置"""

    def test_max_iterations(self):
        """测试最大迭代次数"""
        self.assertGreater(JSONRepairConfig.MAX_REPAIR_ITERATIONS, 0)
        self.assertLess(JSONRepairConfig.MAX_REPAIR_ITERATIONS, 20)

    def test_thresholds(self):
        """测试阈值"""
        self.assertGreater(JSONRepairConfig.TRUNCATE_NOISE_THRESHOLD, 0)
        self.assertGreater(JSONRepairConfig.MAX_BRACKET_IMBALANCE, 0)


class TestWorkflowConfig(unittest.TestCase):
    """测试工作流配置"""

    def test_default_max_iterations(self):
        """测试默认最大迭代数"""
        self.assertGreater(WorkflowConfig.DEFAULT_MAX_ITERATIONS, 0)

    def test_concurrency_limits(self):
        """测试并发限制"""
        self.assertGreater(WorkflowConfig.MAX_CONCURRENT_TASKS, 0)


class TestBackwardCompatibility(unittest.TestCase):
    """测试向后兼容性"""

    def test_exported_constants(self):
        """测试导出的常量"""
        from config.constants import (
            MODEL_CONTEXT_LIMITS,
            MODEL_MAX_OUTPUT_TOKENS,
            REASONER_MIN_TOKENS,
        )

        self.assertIsInstance(MODEL_CONTEXT_LIMITS, dict)
        self.assertIsInstance(MODEL_MAX_OUTPUT_TOKENS, dict)
        self.assertIsInstance(REASONER_MIN_TOKENS, int)


if __name__ == "__main__":
    unittest.main()

