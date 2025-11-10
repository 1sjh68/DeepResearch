"""
错误处理模块的单元测试
"""

import unittest

from utils.error_handler import (
    ErrorHandler,
    ErrorSeverity,
    ErrorType,
    FriendlyError,
    handle_api_errors,
    log_and_return,
    safe_execute,
    suppress_errors,
)


class TestFriendlyError(unittest.TestCase):
    """测试FriendlyError类"""

    def test_create_friendly_error(self):
        """测试创建友好错误"""
        error = FriendlyError(
            error_type=ErrorType.API_TIMEOUT,
            title="测试错误",
            message="这是一个测试",
            suggestions=["建议1", "建议2"],
            severity=ErrorSeverity.WARNING,
        )

        self.assertEqual(error.title, "测试错误")
        self.assertEqual(len(error.suggestions), 2)

    def test_to_dict(self):
        """测试转换为字典"""
        error = FriendlyError(
            error_type=ErrorType.API_ERROR,
            title="API错误",
            message="测试消息",
            suggestions=["建议"],
        )

        error_dict = error.to_dict()
        self.assertEqual(error_dict["type"], "error")
        self.assertEqual(error_dict["title"], "API错误")

    def test_to_json(self):
        """测试转换为JSON"""
        error = FriendlyError(
            error_type=ErrorType.NETWORK_ERROR,
            title="网络错误",
            message="测试",
            suggestions=[],
        )

        json_str = error.to_json()
        self.assertIsInstance(json_str, str)
        self.assertIn("network_error", json_str)


class TestErrorHandler(unittest.TestCase):
    """测试ErrorHandler类"""

    def test_handle_error(self):
        """测试错误处理"""
        error = ErrorHandler.handle_error(ErrorType.API_TIMEOUT)

        self.assertIsInstance(error, FriendlyError)
        self.assertEqual(error.error_type, ErrorType.API_TIMEOUT)
        self.assertGreater(len(error.suggestions), 0)

    def test_detect_error_type(self):
        """测试错误类型检测"""
        timeout_exc = Exception("Request timeout occurred")
        error_type = ErrorHandler.detect_error_type(timeout_exc)
        self.assertEqual(error_type, ErrorType.API_TIMEOUT)

        network_exc = Exception("Network connection failed")
        error_type = ErrorHandler.detect_error_type(network_exc)
        self.assertEqual(error_type, ErrorType.NETWORK_ERROR)

    def test_from_exception(self):
        """测试从异常创建错误"""
        exc = Exception("Network connection failed")
        error = ErrorHandler.from_exception(exc)

        self.assertIsInstance(error, FriendlyError)
        # 验证可以从异常创建错误（具体类型取决于错误消息）
        self.assertIn(error.error_type, [ErrorType.NETWORK_ERROR, ErrorType.UNKNOWN])


class TestHandleAPIErrors(unittest.TestCase):
    """测试handle_api_errors上下文管理器"""

    def test_no_error(self):
        """测试没有错误时"""
        with handle_api_errors("test operation"):
            result = "success"

        # 应该正常执行
        self.assertEqual(result, "success")

    def test_with_fallback(self):
        """测试带fallback"""
        with handle_api_errors("test operation", fallback="default"):
            raise Exception("Test error")

        # 不应该抛出异常


class TestSuppressErrors(unittest.TestCase):
    """测试suppress_errors上下文管理器"""

    def test_suppress_specific_error(self):
        """测试抑制特定错误"""
        with suppress_errors(ValueError, KeyError):
            raise ValueError("This should be suppressed")

        # 不应该抛出异常

    def test_dont_suppress_other_errors(self):
        """测试不抑制其他错误"""
        with self.assertRaises(TypeError):
            with suppress_errors(ValueError):
                raise TypeError("This should not be suppressed")


class TestLogAndReturn(unittest.TestCase):
    """测试log_and_return函数"""

    def test_returns_default(self):
        """测试返回默认值"""
        exc = Exception("Test error")
        result = log_and_return(exc, "Operation failed", default="default")

        self.assertEqual(result, "default")

    def test_returns_none(self):
        """测试返回None"""
        exc = Exception("Test error")
        result = log_and_return(exc, "Operation failed")

        self.assertIsNone(result)


class TestSafeExecute(unittest.TestCase):
    """测试safe_execute函数"""

    def test_successful_execution(self):
        """测试成功执行"""
        result = safe_execute(lambda: "success", fallback="failed")
        self.assertEqual(result, "success")

    def test_failed_execution(self):
        """测试失败执行"""
        def failing_func():
            raise Exception("Test error")

        result = safe_execute(failing_func, fallback="fallback_value")
        self.assertEqual(result, "fallback_value")

    def test_with_context(self):
        """测试带上下文"""
        def failing_func():
            raise ValueError("Test")

        result = safe_execute(
            failing_func,
            fallback=None,
            context="测试操作"
        )
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()

