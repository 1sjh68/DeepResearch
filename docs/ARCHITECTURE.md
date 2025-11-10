# 🏛️ 系统架构设计文档

**版本**: v1.0  
**最后更新**: 2025-11-07

---

## 目录

1. [系统概述](#1-系统概述)
2. [分层架构](#2-分层架构)
3. [核心组件](#3-核心组件)
4. [工作流引擎](#4-工作流引擎)
5. [状态管理](#5-状态管理)
6. [数据流](#6-数据流)
7. [扩展点](#7-扩展点)

---

## 1. 系统概述

DeepResearch是一个基于AI的自动化内容创作框架，采用工作流图架构，
支持从规划、草稿、评审到润色的完整写作流程。

### 1.1 核心价值

- **自动化**：端到端的自动化文档生成
- **可靠性**：多轮迭代和质量检查机制
- **可扩展**：插件式节点架构
- **智能化**：RAG增强和网络研究集成

### 1.2 技术栈

- **工作流引擎**：LangGraph
- **AI模型**：DeepSeek (Chat/Reasoner/Coder)
- **向量数据库**：Chroma
- **数据验证**：Pydantic
- **重试机制**：Tenacity

---

## 2. 分层架构

```
┌─────────────────────────────────────────────────────────┐
│                     入口层 (main.py)                     │
│                  命令行接口 / API接口                      │
└───────────────────────┬─────────────────────────────────┘
                        │
┌───────────────────────▼─────────────────────────────────┐
│               工作流执行层 (core/)                         │
│   - workflow_executor: 管道编排                           │
│   - context_manager: 上下文管理                           │
│   - state_manager: 状态验证                              │
└───────────────────────┬─────────────────────────────────┘
                        │
┌───────────────────────▼─────────────────────────────────┐
│               工作流图层 (workflows/)                      │
│   - graph_builder: 构建工作流DAG                          │
│   - graph_runner: 执行工作流                              │
│   - nodes/*: 各个工作流节点实现                           │
└───────────────────────┬─────────────────────────────────┘
                        │
        ┌───────────────┼───────────────┐
        │               │               │
┌───────▼──────┐ ┌─────▼──────┐ ┌─────▼──────┐
│ services/    │ │ planning/  │ │ utils/     │
│ 外部服务集成  │ │ 规划工具   │ │ 工具函数   │
│              │ │            │ │            │
│ - LLM调用    │ │ - Schema   │ │ - 文本处理 │
│ - 向量DB     │ │ - 大纲     │ │ - 错误处理 │
│ - 网络抓取   │ │            │ │ - 进度追踪 │
└──────────────┘ └────────────┘ └────────────┘
```

---

## 3. 核心组件

### 3.1 配置管理 (config/)

**职责**：统一的配置加载、验证和管理

**关键类**：
- `Config`: 中央配置对象
- `EnvironmentSettings`: 环境变量Pydantic模型
- `APISettings`, `ModelSettings`, `WorkflowFlags`: 分组配置

**设计模式**：
- 分层配置（API、模型、工作流等）
- 向后兼容（扁平化属性访问）
- Pydantic验证确保类型安全

### 3.2 工作流执行器 (core/workflow_executor.py)

**职责**：工作流执行的顶层协调

```python
def run_workflow_pipeline(
    config: Config,
    vector_db_manager: VectorDBManager | None,
    ...
) -> WorkflowResult:
    # 1. 预检LLM连通性
    # 2. 运行工作流图
    # 3. 后处理（结构整合）
    # 4. 质量评估
    # 5. 保存结果
```

### 3.3 上下文管理器 (core/context_manager.py)

**职责**：为各个节点组装上下文

**核心功能**：
- 章节内容存储
- 摘要生成
- RAG检索集成
- 上下文包构建

### 3.4 状态管理 (core/state_manager.py)

**职责**：工作流状态的类型安全管理

**双模型架构**：
```
GraphState (TypedDict)
    ↓ (运行时使用)
LangGraph 工作流
    ↓ (验证需要时)
WorkflowStateAdapter.ensure()
    ↓
WorkflowStateModel (Pydantic)
```

**设计考虑**：
- GraphState：LangGraph原生接口要求
- WorkflowStateModel：提供运行时验证
- STATE_FIELDS：单一数据源（未来整合点）

---

## 4. 工作流引擎

### 4.1 工作流图 (workflows/graph_builder.py)

使用LangGraph构建有向无环图（DAG）：

```python
workflow = StateGraph(GraphState)

# 添加节点
workflow.add_node("plan_node", plan_node)
workflow.add_node("draft_node", draft_node)
...

# 添加边
workflow.add_edge("plan_node", "draft_node")

# 条件边
workflow.add_conditional_edges(
    "critique_node",
    should_research,  # 决策函数
    {"research_node": "research_node", "refine_node": "refine_node"}
)

# 编译
app = workflow.compile()
```

### 4.2 执行模式

#### 模式1：LangGraph模式（默认禁用）
- 使用`app.invoke()`执行
- 支持复杂的条件边和循环
- 可能在某些环境出现停滞

#### 模式2：Simple Runner（推荐）
- 线性执行所有节点
- 手动控制循环逻辑
- 更稳定可靠

配置：`USE_SIMPLE_RUNNER=true`（默认）

### 4.3 节点装饰器

```python
@workflow_step("plan_node", "生成文档大纲")
def plan_node(state: GraphState) -> StepPayload:
    # 1. 自动进度追踪
    # 2. 异常处理
    # 3. 日志记录
    return {"outline": generated_outline}
```

---

## 5. 状态管理

### 5.1 状态字段分组

```python
GraphState:
    # 核心上下文
    task_id: str | None
    config: Config
    refinement_count: int
    
    # 规划/草稿
    outline: dict | None
    draft_content: str | None
    context_repository: ContextRepository | None
    
    # 评审与研究
    critique: str | None
    knowledge_gaps: list[str]
    patches: list[dict]
    
    # 最终输出
    final_solution: str | None
    
    # 控制标志
    force_exit_refine: bool
```

### 5.2 状态转换

```
初始状态
    ↓ plan_node
task_id, config → outline, style_guide
    ↓ draft_node
outline → draft_content
    ↓ critique_node
draft_content → critique, knowledge_gaps
    ↓ research_node (条件)
knowledge_gaps → research_brief
    ↓ refine_node
critique, research_brief → patches
    ↓ apply_patches_node
patches → draft_content (更新), refinement_count++
    ↓ (循环或进入polish)
polish_node
draft_content → final_solution
```

---

## 6. 数据流

### 6.1 用户输入流

```
用户问题 (config.user_problem)
    ↓
style_guide_node → 风格指南
    ↓
plan_node → 结构化大纲
    ↓
skeleton_node → 骨架目录
    ↓
digest_node → 资料索引
    ↓
topology_writer_node → 初稿
```

### 6.2 优化循环流

```
初稿
    ↓
critique_node → 发现问题和知识空白
    ↓
[条件] 有知识空白且启用网络研究？
    ├─ 是 → research_node → 补充资料
    └─ 否 → 直接进入refine
    ↓
refine_node → 生成句子级补丁
    ↓
apply_patches_node → 应用补丁，更新草稿
    ↓
[条件] 达到最大轮数？
    ├─ 否 → 回到critique_node
    └─ 是 → 进入polish_node
```

### 6.3 上下文流

```
外部数据 (PDF/DOCX等)
    ↓
RAGService.ensure_index() → 向量索引
    ↓
ContextAssembler.build_context()
    ↓ (for each node)
组装上下文包：
- 大纲
- 风格指南
- RAG检索结果
- 前序章节内容
- 后续章节描述
    ↓
发送给LLM
```

---

## 7. 扩展点

### 7.1 添加新节点

```python
# 1. 在workflows/nodes/创建新文件
# workflows/nodes/my_node.py

from core.progress import workflow_step
from workflows.graph_state import GraphState

@workflow_step("my_node", "执行我的任务")
def my_node(state: GraphState) -> dict:
    """我的自定义节点。"""
    # 处理逻辑
    return {"my_result": result}

# 2. 在workflows/nodes/__init__.py导出
from .my_node import my_node

# 3. 在graph_builder.py中注册
workflow.add_node("my_node", my_node)
workflow.add_edge("previous_node", "my_node")
```

### 7.2 添加新LLM提供商

```python
# config/client_factory.py

def create_custom_client(config: Config) -> CustomClient:
    """创建自定义LLM客户端。"""
    return CustomClient(api_key=config.custom_api_key)

# config/config.py
# 添加配置字段

# services/llm_interaction.py
# 适配调用接口
```

### 7.3 自定义文本处理

```python
# utils/text_processor.py

def custom_post_processing(text: str) -> str:
    """自定义后处理逻辑。"""
    # 添加处理步骤
    return processed_text

# core/workflow_executor.py
# 在final_post_processing中集成
```

---

## 附录：设计决策

### A.1 为什么使用双状态模型？

**问题**：GraphState和WorkflowStateModel看似重复

**原因**：
1. LangGraph要求TypedDict作为状态类型
2. Pydantic提供运行时验证
3. 两者服务不同目的，互补

**未来**：考虑统一到STATE_FIELDS生成

### A.2 为什么默认使用Simple Runner？

**问题**：LangGraph提供完整的图能力，为什么不用？

**原因**：
1. 某些环境下图循环可能停滞
2. Simple Runner更稳定可预测
3. 对本项目的线性流程足够

**选择**：保留两种模式，让用户选择

### A.3 为什么使用多个AI模型？

**配置**：
- `main_ai_model`: deepseek-chat（主写作）
- `main_ai_model_heavy`: deepseek-reasoner（深度推理）
- `secondary_ai_model`: deepseek-reasoner（评审）
- `summary_model_name`: deepseek-coder（摘要）

**原因**：
1. 不同任务适合不同模型
2. Reasoner模型适合推理和评审
3. Coder模型适合结构化输出
4. 分离可控制成本和质量

---

**维护者**：开发团队  
**反馈**：请提交issue

