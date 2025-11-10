# 💻 开发指南

**版本**: v1.0  
**最后更新**: 2025-11-07

---

## 目录

1. [开发环境设置](#1-开发环境设置)
2. [开发工作流](#2-开发工作流)
3. [调试指南](#3-调试指南)
4. [测试指南](#4-测试指南)
5. [常见开发任务](#5-常见开发任务)

---

## 1. 开发环境设置

### 1.1 系统要求

- Python 3.11+
- Git
- 4GB+ RAM
- Windows/Linux/macOS

### 1.2 首次设置

```bash
# 1. 克隆项目
git clone <repository-url>
cd deepresearch

# 2. 创建虚拟环境
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/macOS
source venv/bin/activate

# 3. 安装依赖
pip install -r requirements.txt

# 4. 安装开发依赖
pip install black isort pylint mypy pytest pytest-cov

# 5. 配置环境变量
cp .env.example .env
# 编辑.env，设置必需的API密钥
```

### 1.3 IDE配置

#### VS Code

创建`.vscode/settings.json`:
```json
{
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": true,
    "python.formatting.provider": "black",
    "python.formatting.blackArgs": ["--line-length=120"],
    "editor.formatOnSave": true,
    "python.linting.mypyEnabled": true,
    "[python]": {
        "editor.codeActionsOnSave": {
            "source.organizeImports": true
        }
    }
}
```

#### PyCharm

1. Settings → Tools → Black → 启用
2. Settings → Tools → External Tools → 添加isort
3. Settings → Editor → Inspections → Python → 启用类型检查

---

## 2. 开发工作流

### 2.1 日常开发流程

```bash
# 1. 创建功能分支
git checkout -b feature/my-feature

# 2. 编写代码
# 遵循代码规范（见README.md）

# 3. 运行测试
pytest tests/ -v

# 4. 代码检查
black deepresearch/ --check
isort deepresearch/ --check-only
pylint deepresearch/
mypy deepresearch/

# 5. 自动格式化
black deepresearch/
isort deepresearch/

# 6. 提交
git add .
git commit -m "feat(module): 添加新功能"

# 7. 推送并创建PR
git push origin feature/my-feature
```

### 2.2 提交前检查清单

- [ ] 所有测试通过
- [ ] 代码已格式化（black）
- [ ] Imports已排序（isort）
- [ ] 无Pylint警告
- [ ] 类型检查通过（mypy）
- [ ] 添加必要的注释（中文）
- [ ] 更新相关文档

---

## 3. 调试指南

### 3.1 启用详细日志

```bash
# 在main.py中设置
config.setup_logging(logging.DEBUG)

# 或环境变量
export LOG_LEVEL=DEBUG
```

### 3.2 调试JSON修复

```bash
# 启用JSON修复调试
export DEBUG_JSON_REPAIR=true

# 运行并查看详细日志
python main.py 2>&1 | tee debug.log
```

#### JSON修复机制

系统使用**多层JSON修复策略**，自动处理AI生成JSON中的各种格式错误：

**修复层次**：
1. **控制字符清理**: LaTeX智能还原（`utils/latex_handler.py`）
   - 自动识别`\x07lpha` → `\\alpha`等损坏的LaTeX命令
   - 支持143个常用LaTeX命令
   - 详细统计和日志

2. **专业库修复**: json-repair处理常见格式错误
   - 缺失引号（开头/结尾）
   - Trailing comma
   - 缺失字段值
   - 多余闭合括号
   - Python字面量（True/False/None）

3. **正则预处理**: `preprocess_json_string`
   - 移除Markdown代码块
   - 修复常见格式问题
   - LaTeX反斜杠修复

4. **自定义修复**: 针对特定问题
   - `_fix_missing_field_values`: "field": , → "field": null,
   - `_fix_unbalanced_quotes_in_arrays`: , 文本" → , "文本"
   - `_remove_duplicate_closers`: }} → }

**依赖说明**：
- `json-repair>=0.7.0`: **推荐安装**，显著提升修复成功率（50% → 85%+）
- 如果未安装，系统会自动降级到内置修复逻辑
- 安装命令：`pip install json-repair`

**调试工具**：
```bash
# 提取失败的JSON案例
python scripts/extract_failed_jsons.py

# 测试修复能力
python scripts/test_json_repair.py

# 查看修复统计
grep "JSON 修复成功" outputs/session_*/session.log
```

### 3.3 调试工具脚本

项目提供了几个调试脚本：

```bash
# 调试润色工具调用
python scripts/debug_polish_toolcall.py --section-text "测试内容"

# 重放失败的工具调用
python scripts/replay_failing_toolcall.py

# 清理Python缓存
python scripts/clear_pycache.py
```

### 3.4 断点调试

```python
# 在代码中添加断点
import pdb; pdb.set_trace()

# VS Code调试配置 (.vscode/launch.json)
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python: Main",
            "type": "python",
            "request": "launch",
            "program": "${workspaceFolder}/deepresearch/main.py",
            "console": "integratedTerminal",
            "env": {
                "USER_PROBLEM": "测试问题",
                "DEEPSEEK_API_KEY": "your-key"
            }
        }
    ]
}
```

---

## 4. 测试指南

### 4.1 测试结构

```
tests/
├── test_core/
│   ├── test_context_manager.py
│   ├── test_state_manager.py
│   └── test_workflow_executor.py
├── test_workflows/
│   ├── test_graph_builder.py
│   └── test_nodes/
├── test_services/
│   └── test_llm_interaction.py
├── test_utils/
│   └── test_text_processor.py
└── conftest.py  # 共享fixtures
```

### 4.2 编写测试

```python
# tests/test_core/test_state_manager.py

import pytest
from core.state_manager import WorkflowStateModel, WorkflowStateAdapter
from config import Config

class TestWorkflowStateModel:
    """测试工作流状态模型。"""
    
    def test_create_minimal_state(self):
        """测试：创建最小状态"""
        config = Config()
        state = WorkflowStateModel(config=config)
        
        assert state.config == config
        assert state.refinement_count == 0
        assert state.final_solution is None
    
    def test_validation_fails_without_config(self):
        """测试：缺少config时验证失败"""
        with pytest.raises(ValueError):
            WorkflowStateModel()
```

### 4.3 运行测试

```bash
# 运行所有测试
pytest

# 详细输出
pytest -v

# 运行特定测试
pytest tests/test_core/test_state_manager.py::TestWorkflowStateModel::test_create_minimal_state

# 查看覆盖率
pytest --cov=deepresearch --cov-report=html

# 生成覆盖率报告
open htmlcov/index.html
```

---

## 5. 常见开发任务

### 5.1 添加新的工作流节点

参见 [ARCHITECTURE.md#7.1](ARCHITECTURE.md#71-添加新节点)

### 5.2 添加新的配置选项

```python
# 1. 在config/env_loader.py添加环境变量解析
"new_option": _parse_bool_env("NEW_OPTION", False),

# 2. 在EnvironmentSettings添加字段
new_option: bool = False

# 3. 在对应的Settings类添加
class WorkflowFlags:
    new_option: bool
    
    @classmethod
    def from_env(cls, env):
        return cls(..., new_option=env.new_option)

# 4. 在Config类添加扁平化访问
new_option: bool
```

### 5.3 修改LLM提示词

```python
# workflows/prompts.py

# 修改现有提示词常量
DRAFT_SYSTEM_PROMPT = """..."""

# 在节点中使用
prompt = DRAFT_SYSTEM_PROMPT + user_context
```

### 5.4 添加新的数据处理函数

```python
# utils/text_processor.py

def my_text_processor(config: Config, text: str) -> str:
    """我的文本处理函数。
    
    参数：
        config: 配置对象
        text: 待处理文本
        
    返回：
        处理后的文本
    """
    # 处理逻辑
    return processed_text

# 在其他模块中导入使用
from utils.text_processor import my_text_processor
```

---

## 6. 性能分析

### 6.1 性能分析工具

```python
# 使用cProfile
python -m cProfile -o profile.stats main.py

# 分析结果
python -m pstats profile.stats
> sort cumulative
> stats 20
```

### 6.2 内存分析

```python
# 使用memory_profiler
pip install memory_profiler

# 装饰要分析的函数
from memory_profiler import profile

@profile
def my_function():
    # ...
    
# 运行
python -m memory_profiler main.py
```

---

## 7. 故障排除

### 7.1 依赖问题

```bash
# 重新安装所有依赖
pip install -r requirements.txt --force-reinstall

# 检查依赖冲突
pip check
```

### 7.2 导入错误

```bash
# 确保在项目根目录
cd deepresearch

# 检查PYTHONPATH
echo $PYTHONPATH

# 设置PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### 7.3 测试失败

```bash
# 详细输出
pytest -vv

# 显示打印语句
pytest -s

# 在失败时进入调试器
pytest --pdb
```

---

## 8. 最佳实践

### 8.1 代码组织

- 单个文件不超过1000行
- 函数不超过50行
- 类不超过300行
- 使用有意义的命名

### 8.2 错误处理

- 使用特定异常类型
- 保留异常链（`from e`）
- 记录完整堆栈（`exc_info=True`）
- 详细信息参见utils/error_handler.py

### 8.3 性能考虑

- 避免重复计算（使用缓存）
- 批量处理API调用
- 使用异步I/O（适当时）
- 延迟加载大对象

---

**维护者**：开发团队  
**问题反馈**：提交issue或联系负责人

