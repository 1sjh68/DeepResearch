# logic/file_saver.py

"""
This module handles the saving of the final output to a file.
"""

import logging
import os
from datetime import datetime


def save_final_result(config, final_answer: str, output_filename: str | None = None) -> str | None:
    """
    Saves the final result to a file.
    """
    if not final_answer:
        return None

    filename = output_filename or f"final_solution_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    session_dir = config.session_dir
    if session_dir and os.path.isdir(session_dir):
        saved_filepath = os.path.join(session_dir, filename)
        try:
            with open(saved_filepath, "w", encoding="utf-8") as f:
                f.write(final_answer)
            logging.info("🎉 最终报告已成功保存至: %s", saved_filepath)
            return saved_filepath
        except Exception as exc:
            logging.error("保存最终报告时发生错误: %s", exc)
            return None
    else:
        logging.error("会话目录不存在，无法保存最终文件。")
        return None
