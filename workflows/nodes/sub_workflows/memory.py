# workflows/sub_workflows/memory.py

import hashlib
import json
import logging
import time
import uuid
from datetime import datetime

from config import Config
from services.vector_db import VectorDBManager
from utils.text_processor import chunk_document_for_rag


def accumulate_experience(
    config: Config,
    db_manager: VectorDBManager,
    problem: str,
    final_solution: str | None,
    feedback_history: list[str],
    successful_patches: list[dict],
    research_briefs: list[str],
):
    """
    (V2 - 精细化分块版) 将本次运行的经验积累到向量数据库。
    此版本会对最终的长篇解决方案进行分块，确保其能被成功向量化。
    """
    # 可配置跳过经验写入，避免不稳定环境下的 I/O 延迟
    if getattr(config, "disable_memory_node", False):
        logging.info("\n--- 已禁用经验写入 (DISABLE_MEMORY_NODE=true) ---")
        return

    logging.info("\n--- 正在将经验积累到向量数据库 ---")
    if not db_manager or not db_manager.collection:
        logging.info("数据库未就绪，跳过经验积累（预期可关闭）。")
        return

    experience_items, metadatas, ids = [], [], []
    current_time_iso = datetime.now().isoformat()
    problem_hash = hashlib.md5(problem.encode()).hexdigest()

    if final_solution:
        logging.info("  - 正在为最终解决方案分块以便存入长期记忆...")
        solution_chunks, solution_metadatas = chunk_document_for_rag(config, final_solution, f"solution_{problem_hash}")

        for i, chunk in enumerate(solution_chunks):
            experience_items.append(chunk)
            meta = solution_metadatas[i]
            meta.update({"type": "final_solution_chunk", "problem": problem[:200], "date": current_time_iso})
            metadatas.append(meta)
            ids.append(f"solution_{problem_hash}_{int(time.time())}_{i}")

    for i, feedback in enumerate(feedback_history):
        if not feedback or not str(feedback).strip():
            continue
        feedback_text = str(feedback).strip()
        experience_items.append(f"Feedback Note #{i + 1}:\n{feedback_text}")
        metadatas.append(
            {
                "type": "feedback_note",
                "problem": problem[:200],
                "date": current_time_iso,
                "origin": "review_feedback",
            }
        )
        ids.append(f"feedback_{problem_hash}_{int(time.time())}_{i}")

    for i, patch_info in enumerate(successful_patches):
        patch_dict = patch_info if isinstance(patch_info, dict) else patch_info.model_dump(mode="json")
        if "target_id" in patch_dict and isinstance(patch_dict["target_id"], uuid.UUID):
            patch_dict["target_id"] = str(patch_dict["target_id"])
        patch_content = f"Feedback:\n{patch_dict.get('feedback_applied', 'N/A')}\n\nApplied Patch:\n{json.dumps(patch_dict, ensure_ascii=False, indent=2)}"
        experience_items.append(patch_content)
        metadatas.append(
            {
                "type": "feedback_patch_pair",
                "problem": problem[:200],
                "iteration": patch_dict.get("iteration", "N/A"),
                "date": current_time_iso,
            }
        )
        ids.append(f"patch_{problem_hash}_{i}")

    for i, brief in enumerate(research_briefs):
        experience_items.append(brief)
        metadatas.append({"type": "research_brief", "problem": problem[:200], "date": current_time_iso})
        ids.append(f"research_{problem_hash}_{int(time.time())}_{i}")

    if experience_items:
        db_manager.add_experience(texts=experience_items, metadatas=metadatas, ids=ids)
