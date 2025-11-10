# Utils模块文档

工具函数模块，提供文本处理、错误处理、文件处理等辅助功能。

## 模块组成

### text_processor.py
**职责**：文本处理工具集

**核心函数**：
- `consolidate_document_structure()`: 文档结构整合
- `truncate_text_for_context_boundary_aware()`: 边界感知截断
- `chunk_document_for_rag()`: RAG分块
- `extract_json_from_ai_response()`: JSON提取
- `preprocess_json_string()`: JSON预处理
- `final_post_processing()`: 最终后处理
- `quality_check()`: 质量检查

### error_handler.py
**职责**：统一错误处理

**核心类**：
- `ErrorHandler`: 错误处理器
- `FriendlyError`: 友好错误信息
- `ErrorType`: 错误类型枚举

### file_handler.py
**职责**：文件读取和路径处理

**支持格式**：
- .txt
- .pdf
- .docx
- .pptx

### progress_tracker.py
**职责**：进度追踪和显示

**核心类**：
- `EnhancedProgressTracker`: 增强进度追踪器

**功能**：
- 步骤追踪
- 进度条显示
- 脉冲更新

### iteration_storage.py
**职责**：迭代内容存储

### draft_manager.py
**职责**：草稿管理

### citation.py
**职责**：引用管理

### factcheck.py
**职责**：事实检查

### log_streamer.py
**职责**：日志流式输出

---

**维护者**：工具团队

