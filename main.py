# main.py

import logging
import os
import sys
import uuid

# --- 核心模块导入 ---
from config import Config
from core.workflow_executor import run_workflow_pipeline
from services.llm_interaction import preflight_llm_connectivity
from services.vector_db import EmbeddingModel, VectorDBManager

# --- 路径修正代码 ---
# 优先使用 __file__ 来确定项目根目录，如果失败（例如在交互式环境中），则回退到 getcwd()
project_root = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
if project_root not in sys.path:
    sys.path.insert(0, project_root)


def main():
    # Force unbuffered output to prevent log interleaving
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    # Force simple runner and stable end-of-run behavior in this environment
    # Ensure full iterations (no early exit)
    os.environ.setdefault("DISABLE_EARLY_EXIT", "true")
    # Avoid long post-run LLM calls by default
    os.environ.setdefault("DISABLE_FINAL_QUALITY_CHECK", "true")
    os.environ.setdefault("DISABLE_MEMORY_NODE", "true")
    os.environ.setdefault("DISABLE_RAG_FOR_PATCH", "true")

    config = Config()
    config.task_id = str(uuid.uuid4())
    config.setup_logging(logging.DEBUG)

    # Config已经在初始化时从env文件加载了user_problem和external_data_files
    # 不需要再次从环境变量读取，避免覆盖env文件中的配置
    logging.info(f"任务问题 (前100字符): {config.user_problem[:100] if config.user_problem else '未设置'}...")  # noqa: E501
    logging.info(f"外部文件数量: {len(config.external_data_files)}")

    try:
        config.initialize_deepseek_client()
    except Exception as e:
        logging.critical(f"致命错误：无法初始化 DeepSeek 客户端: {e}. 程序即将退出。")
        sys.exit(1)

    # 运行一次轻量连通性预检，尽早暴露代理/TLS问题
    if not preflight_llm_connectivity(config):
        logging.critical(
            "LLM 预检失败：请检查网络/代理设置。建议：设置 LLM_DISABLE_PROXY=true 或正确配置 LLM_HTTP_PROXY/LLM_HTTPS_PROXY；若环境不稳定，可设置 USE_SIMPLE_RUNNER=true，降低 API 超时时间 API_TIMEOUT_SECONDS=60 并减少 API_RETRY_MAX_ATTEMPTS=1。"  # noqa: E501
        )  # noqa: E501
        # 将最后100行日志留给用户定位
        try:
            if config.log_file_path and os.path.isfile(config.log_file_path):
                with open(config.log_file_path, encoding="utf-8", errors="ignore") as f:
                    tail = f.readlines()[-100:]
                logging.error(
                    "\n==== session.log (last 100 lines) ====%s\n=====================",
                    "".join(tail),
                )
        except Exception as e:
            logging.warning(f"读取日志文件失败: {str(e)}")
        sys.exit(2)

    vector_db_manager_instance = None
    try:
        embedding_model_instance = EmbeddingModel(config)
        config.embedding_model_instance = embedding_model_instance
        if embedding_model_instance and embedding_model_instance.client:
            vector_db_manager_instance = VectorDBManager(config, embedding_model_instance)
    except Exception as e:
        logging.error(f"初始化嵌入或向量数据库管理器时出错: {e}。功能将受限。", exc_info=True)

    logging.info("--- [标准模式] 启动完整AI内容创作框架 ---")

    workflow_result = run_workflow_pipeline(
        config,
        vector_db_manager_instance,
    )

    if not workflow_result.success:
        logging.error("脚本执行结束，但最终结果是错误或为空: %s", workflow_result.error)
        return

    if not workflow_result.final_answer:
        logging.error("脚本执行结束，但未获得最终输出。")
        return

    # 最终结果处理
    # logging.info(f"最终答案: {workflow_result.final_answer}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logging.warning("程序被用户手动中断。")
    except Exception as e:
        logging.critical(f"在启动或运行主异步任务时发生致命错误: {e}", exc_info=True)
