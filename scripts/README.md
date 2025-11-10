# Scripts目录文档

生产工具和辅助脚本集合。

## ℹ️ 说明

该目录以前包含多个调试脚本。在代码质量改进和测试覆盖率提升后（35-40%覆盖率，286个通过测试），这些调试脚本已不再需要。相关调试功能已整合到：

- **单元测试**：`tests/` 目录（全部通过）
- **集成测试**：`tests/test_integration.py`
- **核心测试**：各模块对应的测试文件

## 📚 可用工具脚本

### clear_pycache.py ⭐ **保留**
**用途**：一键清理Python缓存文件

**功能**：
- 删除所有 `__pycache__/` 目录
- 清理 `*.pyc`, `*.pyo` 文件
- 清除 `.pytest_cache`, `.mypy_cache` 等编译缓存
- 统计清理的文件数和释放的空间

**使用方法**：
```bash
# 一键清理
python scripts/clear_pycache.py

# 输出示例
# 🧹 清理Python缓存文件
# ✓ 删除: __pycache__
# ✓ 删除: .pytest_cache
# ✅ 清理完成！
#    删除项目数: 42
#    释放空间: 12.34 MB
```

**建议频率**：
- 定期运行（每天/每周）
- 切换分支后运行
- 安装/更新依赖后运行

---

## 目前状态

✅ **已清理的调试脚本**：
- ✓ `debug_polish_toolcall.py` - 功能已整合到单元测试
- ✓ `replay_failing_toolcall.py` - 已被增强的测试套件替代
- ✓ `extract_failed_jsons.py` - 测试覆盖率已充分
- ✓ `test_json_repair.py` - 单元测试已覆盖
- ✓ `benchmark_optimizations.py` - 性能基准已达标
- ✓ `install_optimizations.py` - 依赖已稳定
- ✓ 临时数据文件（`.jsonl`, `.log`, `.txt`）- 已清理

## 推荐的调试方式

### 运行单元测试
```bash
# 运行所有测试
pytest tests/ -v

# 运行特定模块测试
pytest tests/test_json_repair.py -v

# 查看覆盖率
pytest --cov=deepresearch tests/
```

### 查看日志
```bash
# 启用调试日志
export DEBUG_JSON_REPAIR=true
export LOG_LEVEL=DEBUG

python main.py
```

### 性能检查
```bash
# 使用内置性能监控
from utils.performance_monitor import PerformanceMonitor

monitor = PerformanceMonitor()
monitor.checkpoint("milestone_1")
monitor.report()
```

---

## 添加新脚本

创建新的调试脚本：

```python
# scripts/my_debug_script.py

import logging
from config import Config

def main():
    """脚本主函数。"""
    config = Config()
    config.setup_logging(logging.DEBUG)
    
    # 调试逻辑
    ...

if __name__ == "__main__":
    main()
```

---

**维护者**：开发团队

