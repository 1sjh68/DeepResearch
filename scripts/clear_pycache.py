#!/usr/bin/env python3
"""
一键清理Python缓存文件脚本

清理以下缓存：
- __pycache__/ 目录
- *.pyc 文件
- *.pyo 文件
- .pytest_cache 目录
- .mypy_cache 目录
"""

import shutil
import sys
from pathlib import Path


def get_project_root():
    """获取项目根目录"""
    return Path(__file__).resolve().parent.parent


def clear_pycache(directory=None):
    """清理Python缓存文件"""
    if directory is None:
        directory = get_project_root()

    directory = Path(directory)

    cache_patterns = [
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
    ]

    file_patterns = [
        "*.pyc",
        "*.pyo",
        "*.pyd",
        "*.so",
        ".DS_Store",
    ]

    removed_count = 0
    removed_size = 0

    print("=" * 60)
    print("🧹 清理Python缓存文件")
    print("=" * 60)

    # 删除缓存目录
    print("\n📁 清理缓存目录...")
    for pattern in cache_patterns:
        for cache_dir in directory.rglob(pattern):
            if cache_dir.is_dir():
                try:
                    size = sum(f.stat().st_size for f in cache_dir.rglob("*") if f.is_file())
                    shutil.rmtree(cache_dir)
                    removed_count += 1
                    removed_size += size
                    print(f"  ✓ 删除: {cache_dir.relative_to(directory)}")
                except Exception as e:
                    print(f"  ✗ 失败: {cache_dir} - {e}")

    # 删除缓存文件
    print("\n📄 清理缓存文件...")
    for pattern in file_patterns:
        for cache_file in directory.rglob(pattern):
            if cache_file.is_file():
                try:
                    size = cache_file.stat().st_size
                    cache_file.unlink()
                    removed_count += 1
                    removed_size += size
                    print(f"  ✓ 删除: {cache_file.relative_to(directory)}")
                except Exception as e:
                    print(f"  ✗ 失败: {cache_file} - {e}")

    # 统计信息
    print("\n" + "=" * 60)
    size_mb = removed_size / (1024 * 1024)
    print("✅ 清理完成！")
    print(f"   删除项目数: {removed_count}")
    print(f"   释放空间: {size_mb:.2f} MB")
    print("=" * 60 + "\n")

    return removed_count, removed_size


def main():
    """主函数"""
    try:
        # 清理项目根目录
        count, size = clear_pycache()

        if count > 0:
            print("✨ 缓存已清理！\n")
            sys.exit(0)
        else:
            print("⚠️  没有找到需要清理的缓存文件\n")
            sys.exit(0)

    except KeyboardInterrupt:
        print("\n⚠️  清理已取消")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
