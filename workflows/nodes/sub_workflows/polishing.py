# workflows/sub_workflows/polishing.py

import logging
import re
import time

from pydantic import ValidationError

from config import Config
from planning.tool_definitions import EditList
from services.llm_interaction import call_ai  # Use sync call_ai
from utils.progress_tracker import safe_pulse


def polish_chunk_with_diffs(config: Config, chunk_text: str, style_guide: str) -> str:
    """
    (同步版本) 对单个文本块进行差异化润色。
    """
    if not chunk_text or not chunk_text.strip():
        return ""
    logging.info(f"  - 正在对一个 {len(chunk_text)} 字符的文本块进行差异化润色 (JSON模式)...")

    polish_prompt = f"""
你是一位顶级的润色编辑，任务是根据提供的《风格指南》，对下面的【待润色文本】进行精细的语言优化。

**至关重要的规则：**
1.  **JSON输出**: 你的输出必须是一个严格符合`EditList` Pydantic模型的、单一的、合法的JSON对象。不要添加任何解释、道歉或Markdown标记。
2.  **句子级修改**: 你的所有修改都应以句子为单位。
3.  **忠于原意**: 润色是为了提升表达效果，绝不能改变原文的核心意思和事实信息。
4.  **无需修改则返回空**: 如果一段文本无需修改，请必须返回一个空的 "edits" 列表 `[]`。
5.  **禁止修改结构**: 绝对不要修改任何以 `#` 开头的标题行，也不要以任何方式添加、删除或修改 HTML 注释标记。这些结构化标记必须保持不变。

**JSON格式示例:**
```json
{{
  "edits": [
    {{
      "original_sentence": "This is a sentence that could be improved.",
      "revised_sentence": "This is an improved sentence."
    }}
  ]
}}
```

**《风格指南》**
---
{style_guide}
---

**【待润色文本】**
---
{chunk_text}
---

现在，请只生成修正后的JSON对象。
"""
    try:
        # This is a sync call now
        raw_response_text = call_ai(
            config,
            config.editorial_model_name,
            messages=[{"role": "user", "content": polish_prompt}],
            response_format={"type": "json_object"},
            temperature=config.temperature_creative,
            max_tokens_output=4096,
        )
        if raw_response_text and not raw_response_text.isspace():
            edit_list_obj = EditList.model_validate_json(raw_response_text)

            def _apply_edits_safely(text: str) -> str:
                # 仅在非标题、且不含 section_id 注释的行上进行句子级替换
                lines = text.split("\n")
                if not edit_list_obj.edits:
                    return text
                for edit in edit_list_obj.edits:
                    original = (edit.original_sentence or "").strip()
                    revised = (edit.revised_sentence or "").strip()
                    if not original or not revised:
                        continue
                    # 保护规则：跳过任何可能触及标题/ID的编辑
                    if original.startswith("#") or "section_id:" in original:
                        continue
                    # 在首个匹配的、非标题/非ID行中替换一次
                    replaced = False
                    for idx, line in enumerate(lines):
                        if line.lstrip().startswith("#") or "section_id:" in line:
                            continue
                        if original in line:
                            lines[idx] = line.replace(original, revised, 1)
                            replaced = True
                            break
                    if not replaced:
                        # 若整段无匹配，则保留原文，不做全局替换以避免误伤
                        pass
                return "\n".join(lines)

            polished_text = _apply_edits_safely(chunk_text)
            if edit_list_obj.edits:
                logging.info(f"    - AI 提出了 {len(edit_list_obj.edits)} 处修改建议。已进行安全替换。")
            else:
                logging.info("    - AI 评估后认为此文本块无需修改。")
            return polished_text
        else:
            logging.warning("  - Polishing AI 返回了空响应或无法解析的JSON，将使用原始文本块。")
            return chunk_text

    except (ValidationError, ValueError) as e:
        logging.error(f"  - JSON验证失败，将使用原始文本块。错误: {e}")
        return chunk_text
    except Exception as e:
        logging.error(f"  - 润色过程中发生未知错误，将使用原始文本块。错误: {e}", exc_info=True)
        return chunk_text


def perform_final_polish(
    config: Config,
    full_document_text: str,
    style_guide: str,
    iteration_info: str | None = None,
) -> str:
    """
    (同步版本) 对全文进行最终润色。
    """
    logging.info("\n--- 正在对全文进行最终润色 ---")
    chapters = [ch.strip() for ch in re.split(r"(?=^##\s)", full_document_text, flags=re.MULTILINE) if ch.strip()]
    if not chapters:
        logging.warning("  - 未在文档中找到任何'##'开头的章节，润色流程终止。")
        return full_document_text

    polished_chapters = []
    for i, chapter_text in enumerate(chapters):
        chapter_title_match = re.search(r"^(##\s*.*)", chapter_text)
        chapter_title = chapter_title_match.group(1).strip().split("\n")[0] if chapter_title_match else f"章节 {i + 1}"
        logging.info(f"--- 正在处理章节 {i + 1}/{len(chapters)}: '{chapter_title}' ---")
        prefix = f"{iteration_info} · " if iteration_info else ""
        safe_pulse(
            config.task_id,
            f"{prefix}润色 · 章节 {i + 1}/{len(chapters)}: {chapter_title}",
        )
        # 保护章节首行标题（含 section_id 注释）不参与润色
        lines = chapter_text.splitlines()
        if not lines:
            polished_chapters.append(chapter_text)
            continue
        first_line = lines[0]
        body_lines = lines[1:] if len(lines) > 1 else []

        # 将正文按空段落分块润色，标题和任何以 # 开头的行都不修改
        body_text = "\n".join(body_lines)
        chunk_texts = body_text.split("\n\n") if body_text else []
        processed_chunks: list[str] = []
        for chunk in chunk_texts:
            processed_chunks.append(polish_chunk_with_diffs(config, chunk, style_guide))
            time.sleep(0.2)  # Prevent overwhelming API

        # 重新组装：保留原始标题行 + 润色后的正文
        if processed_chunks:
            polished_chapter = first_line + "\n\n" + "\n\n".join(processed_chunks)
        else:
            polished_chapter = first_line
        polished_chapters.append(polished_chapter)

    final_polished_text = "\n\n".join(polished_chapters)
    logging.info("--- 全文润色完成 ---")

    length_ratio = len(final_polished_text) / len(full_document_text) if len(full_document_text) > 0 else 1.0
    if length_ratio < 0.75:
        logging.warning(f"  - 润色后的文档长度 ({len(final_polished_text)}) 过短，仅为原文 ({len(full_document_text)}) 的 {length_ratio:.1%}。为安全起见，将返回原始文档。")
        return full_document_text

    return final_polished_text
