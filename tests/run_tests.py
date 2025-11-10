"""
测试运行器脚本

运行所有单元测试并生成报告。
"""

import sys
import unittest
from pathlib import Path

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def run_all_tests():
    """运行所有测试"""
    # 发现所有测试
    loader = unittest.TestLoader()
    start_dir = Path(__file__).parent
    suite = loader.discover(str(start_dir), pattern='test_*.py')

    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # 返回结果
    return result.wasSuccessful()


def run_specific_test(test_name: str):
    """
    运行特定测试模块

    Args:
        test_name: 测试模块名（不含.py后缀）
    """
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromName(test_name)

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="运行单元测试")
    parser.add_argument(
        "--test",
        type=str,
        help="运行特定测试模块（例如：test_json_repair）"
    )

    args = parser.parse_args()

    if args.test:
        success = run_specific_test(args.test)
    else:
        success = run_all_tests()

    sys.exit(0 if success else 1)

