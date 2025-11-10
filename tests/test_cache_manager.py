"""
缓存管理器的单元测试
"""

import time
import unittest

from utils.cache_manager import CacheManager, get_cache


class TestCacheManager(unittest.TestCase):
    """测试CacheManager基本功能"""

    def setUp(self):
        """测试前准备"""
        self.cache = CacheManager(
            ttl=2,  # 2秒TTL用于快速测试
            max_size=3,
            namespace="test",
            backend="memory"
        )

    def test_set_and_get(self):
        """测试设置和获取"""
        self.cache.set("key1", "value1")
        result = self.cache.get("key1")
        self.assertEqual(result, "value1")

    def test_get_nonexistent(self):
        """测试获取不存在的键"""
        result = self.cache.get("nonexistent", default="default_value")
        self.assertEqual(result, "default_value")

    def test_ttl_expiration(self):
        """测试TTL过期"""
        self.cache.set("key1", "value1")
        self.assertEqual(self.cache.get("key1"), "value1")

        # 等待过期
        time.sleep(2.5)
        result = self.cache.get("key1", default="expired")
        self.assertEqual(result, "expired")

    def test_lru_eviction(self):
        """测试LRU淘汰策略"""
        # 设置3个值（max_size=3）
        self.cache.set("key1", "value1")
        self.cache.set("key2", "value2")
        self.cache.set("key3", "value3")

        # 再设置一个，应该淘汰最旧的
        self.cache.set("key4", "value4")

        # key1应该被淘汰
        result = self.cache.get("key1", default="evicted")
        self.assertEqual(result, "evicted")

        # 其他键应该存在
        self.assertEqual(self.cache.get("key4"), "value4")

    def test_delete(self):
        """测试删除"""
        self.cache.set("key1", "value1")
        self.assertTrue(self.cache.delete("key1"))
        result = self.cache.get("key1", default="deleted")
        self.assertEqual(result, "deleted")

    def test_clear(self):
        """测试清空"""
        self.cache.set("key1", "value1")
        self.cache.set("key2", "value2")
        self.cache.clear()

        self.assertEqual(self.cache.get("key1", default="cleared"), "cleared")
        self.assertEqual(self.cache.get("key2", default="cleared"), "cleared")

    def test_stats(self):
        """测试统计信息"""
        self.cache.set("key1", "value1")
        self.cache.set("key2", "value2")

        stats = self.cache.stats()
        self.assertEqual(stats["namespace"], "test")
        self.assertEqual(stats["backend"], "memory")
        self.assertGreaterEqual(stats["size"], 2)

    def test_get_or_compute(self):
        """测试get_or_compute"""
        call_count = [0]

        def compute():
            call_count[0] += 1
            return f"computed_{call_count[0]}"

        # 第一次应该计算
        result1 = self.cache.get_or_compute("key1", compute)
        self.assertEqual(result1, "computed_1")
        self.assertEqual(call_count[0], 1)

        # 第二次应该使用缓存
        result2 = self.cache.get_or_compute("key1", compute)
        self.assertEqual(result2, "computed_1")
        self.assertEqual(call_count[0], 1)  # 没有再次调用


class TestGetCache(unittest.TestCase):
    """测试get_cache单例功能"""

    def test_singleton_behavior(self):
        """测试单例行为"""
        cache1 = get_cache("singleton_test", ttl=60)
        cache2 = get_cache("singleton_test", ttl=60)

        # 应该返回同一个实例
        self.assertIs(cache1, cache2)

    def test_different_namespaces(self):
        """测试不同命名空间"""
        cache1 = get_cache("namespace1", ttl=60)
        cache2 = get_cache("namespace2", ttl=60)

        # 应该返回不同的实例
        self.assertIsNot(cache1, cache2)


class TestCacheManagerDeepCopy(unittest.TestCase):
    """测试缓存的深拷贝行为"""

    def setUp(self):
        self.cache = CacheManager(
            ttl=60,
            max_size=10,
            namespace="deepcopy_test",
            backend="memory"
        )

    def test_deep_copy_on_set(self):
        """测试设置时的深拷贝"""
        original = {"data": [1, 2, 3]}
        self.cache.set("key1", original)

        # 修改原始数据
        original["data"].append(4)

        # 缓存中的数据不应该被修改
        cached = self.cache.get("key1")
        self.assertEqual(cached["data"], [1, 2, 3])

    def test_deep_copy_on_get(self):
        """测试获取时的深拷贝"""
        original = {"data": [1, 2, 3]}
        self.cache.set("key1", original)

        # 获取并修改
        cached1 = self.cache.get("key1")
        cached1["data"].append(4)

        # 再次获取应该是原始数据
        cached2 = self.cache.get("key1")
        self.assertEqual(cached2["data"], [1, 2, 3])


if __name__ == "__main__":
    unittest.main()

