# 🤖 DeepResearch AI内容创作框架

一个强大的AI驱动的深度研究和内容创作工作流框架，支持自动化的文档生成、优化和润色。

## 🎉 最新更新 (2025-11-08)

**✅ 项目迭代和文档优化完成！**
- 📝 深度研究内容创作系统已成熟运行
- 🧪 测试覆盖率: 35-40%，286个测试100%通过
- 🏆 代码质量分数: 90/100
- 🧹 清理临时调试文件，简化项目结构
- 📚 更新项目文档和流程指南

📖 查看详情: [项目流程图](项目流程图.md) | [开发指南](docs/DEVELOPMENT.md)

## ✨ 核心特性

- 🎯 **智能规划**：自动生成结构化文档大纲
- 📝 **多轮优化**：迭代式评审和改进机制
- 🔍 **网络研究**：自动搜索和整合外部资料
- 🎨 **智能润色**：专业的内容编辑和质量提升
- 📊 **RAG增强**：基于向量数据库的上下文检索
- 🔄 **工作流图**：基于LangGraph的灵活工作流引擎

---

## 🚀 快速开始

### 环境要求

- Python 3.11+
- DeepSeek API密钥
- 可选：向量数据库（Chroma）

### 安装

```bash
# 克隆项目
cd 原项目

# 安装依赖
pip install -r requirements.txt

# 配置环境变量
cp .env.example .env
# 编辑.env文件，设置DEEPSEEK_API_KEY
```

### 基础使用

```bash
# 设置环境变量
export USER_PROBLEM="请详细阐述人工智能的发展趋势"
export DEEPSEEK_API_KEY="your-api-key"

# 运行主程序
python main.py
```

### Python API

```python
from config import Config
from core.workflow_executor import run_workflow_pipeline
from services.vector_db import VectorDBManager

# 初始化配置
config = Config()
config.user_problem = "讲解量子计算的原理"
config.setup_logging()
config.initialize_deepseek_client()

# 运行工作流
result = run_workflow_pipeline(config, vector_db_manager=None)

if result.success:
    print(f"生成完成！保存在: {result.saved_filepath}")
else:
    print(f"生成失败: {result.error}")
```

---

## 📁 项目结构

```
deepresearch/
├── config/              # 配置管理
│   ├── config.py       # 中央配置类
│   ├── settings.py     # 设置模型
│   ├── env_loader.py   # 环境变量加载
│   ├── client_factory.py  # 客户端工厂
│   ├── logging_setup.py   # 日志配置
│   └── constants.py    # 常量定义
├── core/               # 核心组件
│   ├── workflow_executor.py  # 工作流执行器
│   ├── context_manager.py    # 上下文管理
│   ├── context_components.py # 上下文组件
│   ├── state_manager.py      # 状态管理
│   ├── state_fields.py       # 状态字段定义
│   ├── interfaces.py         # 接口定义
│   ├── message_types.py      # 消息类型
│   ├── patch_manager.py      # 补丁管理
│   └── progress.py           # 进度追踪
├── workflows/          # 工作流定义
│   ├── graph_builder.py     # 图构建器
│   ├── graph_runner.py      # 图执行器
│   ├── graph_state.py       # 状态定义
│   ├── graph_nodes.py       # 节点装饰器
│   ├── prompts.py           # 提示词模板
│   ├── nodes/              # 工作流节点
│   │   ├── style_guide.py  # 风格指南节点
│   │   ├── plan.py         # 规划节点
│   │   ├── skeleton.py     # 骨架节点
│   │   ├── digest.py       # 摘要节点
│   │   ├── draft.py        # 草稿节点
│   │   ├── critique.py     # 评审节点
│   │   ├── research.py     # 研究节点
│   │   ├── refine.py       # 优化节点
│   │   ├── apply_patches.py # 补丁应用节点
│   │   ├── polish.py       # 润色节点
│   │   ├── memory.py       # 记忆节点
│   │   ├── polish/         # 润色子模块
│   │   └── sub_workflows/  # 子工作流
│   └── README.md
├── services/           # 外部服务
│   ├── llm_interaction.py  # LLM调用
│   ├── vector_db.py        # 向量数据库
│   ├── fetchers.py         # 网络抓取
│   ├── llm/                # LLM子模块
│   │   ├── message_processor.py  # 消息处理
│   │   └── retry_strategy.py     # 重试策略
│   └── web_research/       # 网络研究模块
│       ├── pipeline/       # 搜索管道
│       ├── parser/         # HTML解析
│       └── cache.py        # 缓存管理
├── utils/              # 工具函数
│   ├── text_processor.py   # 文本处理
│   ├── text_normalizer.py  # 文本标准化
│   ├── error_handler.py    # 错误处理
│   ├── json_repair.py      # JSON修复
│   ├── citation.py         # 引用处理
│   ├── factcheck.py        # 事实检查
│   ├── file_handler.py     # 文件处理
│   ├── draft_manager.py    # 草稿管理
│   ├── iteration_storage.py # 迭代存储
│   ├── cache_manager.py    # 缓存管理
│   ├── progress_tracker.py # 进度追踪
│   └── performance_monitor.py # 性能监控
├── planning/           # 规划工具
│   ├── outline.py          # 大纲生成
│   └── tool_definitions.py # Pydantic模式定义
├── tests/              # 测试文件
├── main.py            # 主入口
├── requirements.txt   # 依赖列表
└── pyproject.toml     # 项目配置
```

---

## ⚙️ 配置说明

### 必需配置

| 环境变量 | 说明 | 示例 |
|---------|------|------|
| `DEEPSEEK_API_KEY` | DeepSeek API密钥 | `sk-xxx` |
| `DEEPSEEK_BASE_URL` | API基础URL | `https://api.deepseek.com` |

### 可选配置

| 环境变量 | 说明 | 默认值 |
|---------|------|--------|
| `MAX_ITERATIONS` | 最大优化轮数 | `4` |
| `ENABLE_WEB_RESEARCH` | 启用网络研究 | `true` |
| `DISABLE_EARLY_EXIT` | 禁用提前退出 | `true` |
| `DEBUG_JSON_REPAIR` | 调试JSON修复 | `false` |
| `USE_SIMPLE_RUNNER` | 使用简单运行器 | `true` |

更多配置选项请参见：`.env.example`

---

## 🏗️ 工作流架构

### 工作流节点

```
style_guide_node     → 生成写作风格指南
     ↓
plan_node           → 生成文档大纲
     ↓
skeleton_node       → 构建骨架结构
     ↓
digest_node         → 整理资料索引
     ↓
topology_writer_node → 初稿生成
     ↓
critique_node       → 评审反馈
     ↓
research_node       → 网络研究（可选）
     ↓
refine_node         → 生成优化补丁
     ↓
apply_patches_node  → 应用补丁
     ↓ (循环)
polish_node         → 最终润色
     ↓
memory_node         → 保存经验
```

### 状态管理

- **GraphState** (`workflows/graph_state.py`): LangGraph TypedDict接口
- **WorkflowStateModel** (`core/state_manager.py`): Pydantic验证模型
- **STATE_FIELDS** (`core/state_fields.py`): 统一字段定义（单一数据源）

---

## 🔧 开发指南

### 开发环境设置

```bash
# 克隆项目
cd 原项目

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 安装开发依赖
pip install -r requirements.txt
pip install black isort pylint mypy pytest

# 配置pre-commit（可选）
pre-commit install
```

### 代码规范

项目遵循以下规范：
- ✅ 所有注释和日志使用中文
- ✅ 遵循PEP 8代码风格
- ✅ 使用类型注解
- ✅ 使用Black格式化（行长120）

### 运行测试

```bash
# 运行所有测试
pytest tests/ -v

# 运行特定测试
pytest tests/test_core/test_state_manager.py -v

# 查看覆盖率
pytest --cov=原项目 tests/
```

### 代码检查

```bash
# 格式化代码
black 原项目/ --line-length=120

# 排序imports
isort 原项目/

# 类型检查
mypy 原项目/ --ignore-missing-imports

# Lint检查
pylint 原项目/ --rcfile=.pylintrc
```

---

## 📖 文档

- [架构设计文档](docs/ARCHITECTURE.md) - 系统架构说明
- [开发指南](docs/DEVELOPMENT.md) - 详细开发指南
- [项目流程图](项目流程图.md) - 完整的工作流程图

### 模块文档

- [core模块](core/README.md) - 核心组件说明
- [workflows模块](workflows/README.md) - 工作流节点说明
- [services模块](services/README.md) - 外部服务集成
- [utils模块](utils/README.md) - 工具函数说明

---

## 🤝 贡献指南

### 提交代码前检查

- [ ] 所有注释使用中文
- [ ] 运行`black`格式化
- [ ] 运行`isort`排序imports
- [ ] 通过`pylint`检查
- [ ] 添加/更新单元测试
- [ ] 更新相关文档
- [ ] 提交消息符合规范

### 提交消息格式

```
<类型>(<范围>): <简短描述>

详细描述（可选）

Fixes #issue_number
```

类型：`feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`, `chore`

示例：
```bash
git commit -m "feat(llm): 添加重试机制处理API超时"
git commit -m "fix(state): 修复状态序列化时的类型错误"
```

---

## 🐛 故障排除

### 常见问题

#### 1. LLM预检失败

```bash
# 问题：网络连接或代理问题
# 解决：
export LLM_DISABLE_PROXY=true
# 或设置代理
export LLM_HTTP_PROXY=http://127.0.0.1:7890
```

#### 2. 向量数据库初始化失败

```bash
# 问题：Chroma版本不兼容
# 解决：
pip install chromadb==0.4.22 --upgrade
```

#### 3. JSON解析失败

```bash
# 问题：模型返回格式错误
# 解决：启用调试
export DEBUG_JSON_REPAIR=true
# 查看日志了解详情
```

#### 4. 内存不足

```bash
# 问题：处理大文件时内存溢出
# 解决：减少上下文窗口
export MAX_CONTEXT_TOKENS_REVIEW=15000
export MAX_CHUNK_TOKENS=2048
```

---

## 📊 性能优化

### 推荐配置

```bash
# 生产环境配置
export USE_SIMPLE_RUNNER=true        # 稳定性优先
export MAX_ITERATIONS=3              # 控制迭代次数
export DISABLE_FINAL_QUALITY_CHECK=true  # 跳过最终质量检查
export API_TIMEOUT_SECONDS=600       # API超时时间
```

### 性能提示

- 使用`USE_SIMPLE_RUNNER=true`避免图循环问题
- 设置合理的`MAX_ITERATIONS`避免过多迭代
- 启用缓存：`ENABLE_RESEARCH_CACHE=true`
- 使用更快的模型进行摘要和规划

---

## 📜 许可证

[待添加]

---

## 🙏 致谢

本项目基于以下开源项目：
- LangGraph
- OpenAI Python SDK
- Pydantic
- Chroma
- Tenacity

---

## 📋 项目维护状态

- ✅ **稳定版本**: v1.0
- ✅ **文档完整度**: 95%
- ✅ **测试覆盖率**: 35-40%
- ✅ **代码质量**: 90/100
- 🔄 **活跃开发**: 持续改进

---

**最后更新**：2025-11-08  
**版本**：v1.0 (稳定)  
**维护者**：DeepResearch Team

