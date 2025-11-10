# Workflows模块文档

工作流定义和节点实现模块，包含LangGraph图构建和所有工作流节点。

## 模块组成

### 图相关文件

#### graph_builder.py
**职责**：构建LangGraph工作流图

**核心函数**：
- `build_graph()`: 构建并编译工作流图
- `should_research()`: 决策函数：是否执行研究
- `should_continue_refining()`: 决策函数：继续优化或进入润色

#### graph_runner.py  
**职责**：执行工作流图

**核心函数**：
- `run_graph_workflow()`: 同步执行工作流
- `_run_simple_runner()`: 简单线性执行器（推荐）

#### graph_state.py
**职责**：定义工作流状态（TypedDict）

**核心类**：
- `GraphState`: LangGraph使用的状态类型

#### graph_nodes.py
**职责**：节点聚合导出

---

## 工作流节点

所有节点位于`workflows/nodes/`目录。

### 规划阶段

#### style_guide_node.py
**职责**：生成写作风格指南

**输出**：
- `style_guide`: 风格指南文本

#### plan_node.py
**职责**：生成文档大纲

**输出**：
- `outline`: 结构化大纲（JSON）

#### skeleton_node.py
**职责**：构建骨架目录

**输出**：
- `skeleton_outline`: 骨架结构

#### digest_node.py
**职责**：整理资料索引卡

**输出**：
- `section_digests`: 资料摘要

### 写作阶段

#### draft.py (topology_writer_node)
**职责**：拓扑写作初稿

**输出**：
- `draft_content`: 初稿内容
- `context_manager`: 上下文管理器

### 优化循环

#### critique_node.py
**职责**：评审草稿，识别问题

**输出**：
- `critique`: 评审反馈
- `knowledge_gaps`: 知识空白列表
- `refinement_count`: 递增

#### research_node.py
**职责**：执行网络研究填补知识空白

**输出**：
- `research_brief`: 研究摘要
- `structured_research_data`: 结构化研究数据

#### refine_node.py
**职责**：生成优化补丁

**输出**：
- `patches`: 句子级补丁列表

#### apply_patches.py (apply_patches_node)
**职责**：应用补丁到草稿

**输出**：
- `draft_content`: 更新后的草稿
- `last_refine_had_effect`: 是否有效果

### 最终阶段

#### polish_node.py
**职责**：最终润色和质量提升

**输出**：
- `final_solution`: 最终文档

#### memory_node.py
**职责**：保存经验到向量数据库

---

## 子工作流

位于`workflows/nodes/sub_workflows/`：

- `drafting.py`: 多块草稿子流程
- `planning.py`: 规划子流程
- `polishing.py`: 润色子流程
- `memory.py`: 记忆子流程

---

## 提示词模板

### prompts.py
包含所有LLM提示词模板：

- `DRAFT_SYSTEM_PROMPT`: 草稿系统提示
- `CRITIQUE_SYSTEM_PROMPT`: 评审系统提示
- `POLISH_SYSTEM_PROMPT`: 润色系统提示
- `RESEARCH_QUERY_PROMPT`: 研究查询提示
- 等等...

---

## 节点开发规范

### 标准节点结构

```python
from core.progress import workflow_step, step_result
from workflows.graph_state import GraphState

@workflow_step("my_node", "执行我的任务")
def my_node(state: GraphState) -> dict:
    """节点说明。
    
    输入（从state读取）：
        - input_field: 描述
    
    输出（返回到state）：
        - output_field: 描述
    """
    # 1. 从state提取输入
    input_data = state.get("input_field")
    
    # 2. 执行处理逻辑
    result = process(input_data)
    
    # 3. 返回更新
    return {"output_field": result}
```

---

**维护者**：工作流团队

