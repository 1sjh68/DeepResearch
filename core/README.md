# Core模块文档

核心组件模块，提供工作流执行、上下文管理、状态管理等基础功能。

## 模块组成

### workflow_executor.py
**职责**：工作流执行的顶层协调器

**核心函数**：
- `run_workflow_pipeline()`: 执行完整工作流并返回结果

**使用示例**：
```python
from core.workflow_executor import run_workflow_pipeline

result = run_workflow_pipeline(config, vector_db_manager)
if result.success:
    print(f"成功！结果：{result.final_answer}")
```

### context_manager.py
**职责**：为各节点组装上下文包

**核心类**：
- `ContextManager`: 上下文管理门面

**功能**：
- 章节内容存储
- 摘要自动生成
- 上下文包组装

### state_manager.py
**职责**：工作流状态的类型安全管理

**核心类**：
- `WorkflowStateModel`: Pydantic状态模型
- `WorkflowStateAdapter`: 状态适配器

### state_fields.py
**职责**：状态字段的统一定义

**核心变量**：
- `STATE_FIELDS`: 所有状态字段的单一数据源

### context_components.py
**职责**：上下文相关的底层组件

**核心类**：
- `RAGService`: RAG检索服务
- `ContextRepository`: 内容仓库
- `ContextAssembler`: 上下文组装器

### patch_manager.py
**职责**：细粒度文本编辑和补丁应用

**核心类**：
- `EditCorrector`: 句子级编辑器
- `FineGrainedEditResult`: 编辑结果

**核心函数**：
- `apply_fine_grained_edits()`: 应用句子级补丁

### progress.py
**职责**：进度追踪装饰器

**核心装饰器**：
- `@workflow_step`: 为节点添加进度追踪

### message_types.py
**职责**：消息类型定义

**核心类型**：
- `ChatMessage`: 聊天消息TypedDict

### interfaces.py
**职责**：接口协议定义，打破循环依赖

**核心协议**：
- `LLMCallProtocol`: LLM调用接口
- `JSONRepairProtocol`: JSON修复接口
- `TextProcessorProtocol`: 文本处理接口

---

## 依赖关系

```
context_manager
    ↓
context_components (RAGService, ContextRepository)
    ↓
patch_manager (EditCorrector)
    ↓
state_manager (WorkflowStateModel)
    ↓
state_fields (STATE_FIELDS)
```

---

**维护者**：核心团队

