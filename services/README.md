# Services模块文档

外部服务集成模块，提供LLM调用、向量数据库、网络抓取等功能。

## 模块组成

### llm_interaction.py
**职责**：与LLM API交互的核心模块

**核心函数**：
- `call_ai()`: 标准AI调用（带重试）
- `call_ai_with_schema()`: 结构化输出调用
- `call_ai_writing_with_auto_continue()`: 写作型调用（自动续写）
- `repair_json_once()`: JSON修复
- `preflight_llm_connectivity()`: 连通性预检

**特性**：
- 自动重试（Tenacity）
- Token管理
- Reasoner模型适配
- JSON强制格式支持
- 消息规范化

**使用示例**：
```python
from services.llm_interaction import call_ai

response = call_ai(
    config,
    "deepseek-chat",
    [{"role": "user", "content": "你好"}],
    temperature=0.7,
    max_tokens_output=2000
)
```

### vector_db.py
**职责**：向量数据库管理

**核心类**：
- `EmbeddingModel`: 嵌入模型封装
- `VectorDBManager`: 向量数据库管理器

**功能**：
- 文本嵌入
- 向量索引
- 混合检索（向量+BM25）
- 重排序（可选）

**使用示例**：
```python
from services.vector_db import VectorDBManager, EmbeddingModel

embedding = EmbeddingModel(config)
db_manager = VectorDBManager(config, embedding)

# 添加经验
db_manager.add_experience(
    texts=["内容1", "内容2"],
    metadatas=[{"source": "doc1"}, {"source": "doc2"}]
)

# 检索
results = db_manager.hybrid_retrieve_experience("查询文本")
```

### fetchers.py
**职责**：智能网络内容抓取

**核心类**：
- `SmartFetcher`: 智能抓取器
- `RateLimiter`: 速率限制器
- `UserAgentRotator`: User-Agent轮换

**功能**：
- 智能重试
- 速率限制
- robots.txt遵守
- 内容解析

### web_research/
**职责**：网络研究子模块

#### pipeline/
- `executor.py`: 研究执行器
- `search.py`: 搜索策略

#### parser/
- `html_parser.py`: HTML解析器

#### fetch_strategy/
- `requests_client.py`: Requests客户端

#### 其他
- `cache.py`: 研究缓存
- `models.py`: 数据模型
- `instrumentation.py`: 诊断工具

---

## 配置

### LLM相关

```bash
DEEPSEEK_API_KEY=sk-xxx
DEEPSEEK_BASE_URL=https://api.deepseek.com
API_TIMEOUT_SECONDS=900

# 代理设置
LLM_DISABLE_PROXY=false
LLM_HTTP_PROXY=http://127.0.0.1:7890
```

### 向量数据库

```bash
VECTOR_DB_PATH=./chroma_db
EMBEDDING_BATCH_SIZE=25
ENABLE_HYBRID_SEARCH=true
ENABLE_BM25_SEARCH=true
ENABLE_RERANK=true
```

### 网络抓取

```bash
FETCH_TIMEOUT=30
PER_HOST_RPS=5
MAX_CONCURRENT=10
RESPECT_ROBOTS=true
```

---

## 最佳实践

### LLM调用

```python
# ✅ 好的做法：指定合理的token限制
response = call_ai(
    config, 
    model_name,
    messages,
    max_tokens_output=2000  # 明确限制
)

# ❌ 避免：不设限制导致超时
response = call_ai(config, model_name, messages)
```

### 错误处理

```python
# ✅ 捕获特定异常
try:
    response = call_ai(...)
except LLMTimeoutError:
    # 处理超时
except LLMEmptyResponseError:
    # 处理空响应
```

---

**维护者**：服务团队

