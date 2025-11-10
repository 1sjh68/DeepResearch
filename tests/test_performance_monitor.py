"""
性能监控模块的单元测试
"""

import time
import unittest

from utils.performance_monitor import (
    PerformanceMonitor,
    benchmark,
    get_monitor,
    get_performance_stats,
    measure_time,
    reset_performance_stats,
    track_time,
)


class TestTrackTime(unittest.TestCase):
    """测试track_time装饰器"""

    def setUp(self):
        """清理统计"""
        reset_performance_stats()

    def test_basic_tracking(self):
        """测试基本追踪"""
        @track_time
        def test_func():
            time.sleep(0.01)
            return "result"

        result = test_func()
        self.assertEqual(result, "result")

        stats = get_performance_stats()
        self.assertIn("test_func", stats)
        self.assertEqual(stats["test_func"]["total_calls"], 1)

    def test_custom_name(self):
        """测试自定义名称"""
        @track_time(name="custom_name")
        def test_func():
            return "result"

        test_func()

        stats = get_performance_stats()
        self.assertIn("custom_name", stats)

    def test_multiple_calls(self):
        """测试多次调用"""
        @track_time
        def test_func():
            time.sleep(0.01)

        test_func()
        test_func()
        test_func()

        stats = get_performance_stats()
        self.assertEqual(stats["test_func"]["total_calls"], 3)
        self.assertGreater(stats["test_func"]["avg_time"], 0)


class TestMeasureTime(unittest.TestCase):
    """测试measure_time上下文管理器"""

    def test_measure_context(self):
        """测试时间测量上下文"""
        with measure_time("test operation"):
            time.sleep(0.01)
        # 如果没有异常，测试通过


class TestPerformanceMonitor(unittest.TestCase):
    """测试PerformanceMonitor类"""

    def test_checkpoint_and_measure(self):
        """测试检查点和测量"""
        monitor = PerformanceMonitor(enabled=True)

        monitor.checkpoint("start")
        time.sleep(0.01)
        duration = monitor.measure_since("start", "test operation")

        self.assertGreater(duration, 0.01)

    def test_disabled_monitor(self):
        """测试禁用的监控器"""
        monitor = PerformanceMonitor(enabled=False)

        monitor.checkpoint("start")
        duration = monitor.measure_since("start")

        self.assertEqual(duration, 0.0)

    def test_clear_checkpoints(self):
        """测试清空检查点"""
        monitor = PerformanceMonitor()

        monitor.checkpoint("cp1")
        monitor.checkpoint("cp2")
        monitor.clear()

        duration = monitor.measure_since("cp1")
        self.assertEqual(duration, 0.0)

    def test_global_monitor(self):
        """测试全局监控器"""
        monitor = get_monitor()
        self.assertIsInstance(monitor, PerformanceMonitor)


class TestBenchmark(unittest.TestCase):
    """测试benchmark装饰器"""

    def test_benchmark_decorator(self):
        """测试基准测试装饰器"""
        @benchmark(iterations=5)
        def test_func():
            time.sleep(0.001)
            return "result"

        result = test_func()
        self.assertEqual(result, "result")


class TestResetStats(unittest.TestCase):
    """测试重置统计"""

    def test_reset(self):
        """测试重置功能"""
        @track_time
        def test_func():
            pass

        test_func()

        stats_before = get_performance_stats()
        self.assertGreater(len(stats_before), 0)

        reset_performance_stats()

        stats_after = get_performance_stats()
        self.assertEqual(len(stats_after), 0)


if __name__ == "__main__":
    unittest.main()

