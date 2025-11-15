# logic/post_processing.py

"""
This module handles the post-processing of the raw output from the AI workflow.
"""

import logging
import re


def consolidate_document_structure(final_markdown_content: str) -> str:
    """
    对最终生成的Markdown文档进行结构性整合和清理（按 section_id 主键、保序不覆盖）。
    - 若行中包含 `<!-- section_id: ... -->`，则以该 ID 作为唯一键；
    - 若无 ID，则按出现顺序分配顺序键；
    - 不进行基于标题文本的去重覆盖，避免丢失内容。
    """
    logging.info("--- 开始执行最终文档结构整合 (V4 - 按 section_id 保序) ---")

    if not final_markdown_content or not final_markdown_content.strip():
        return ""

    heading_re = re.compile(
        r"^(#{1,6})\s+(.*?)(\s*<!--\s*section_id:\s*([A-Za-z0-9_-]+)\s*-->)?\s*$",
        re.MULTILINE,
    )

    lines = final_markdown_content.splitlines()
    intro_lines: list[str] = []
    sections: list[dict] = []

    idx = 0
    current = None
    while idx < len(lines):
        line = lines[idx]
        m = heading_re.match(line)
        if m:
            # 完成上一节
            if current is not None:
                sections.append(current)
            level = len(m.group(1))
            title_text = m.group(2).strip()
            # 规范化标题，剥离可能嵌入的 # 前缀与脚手架标记（如“标题:”）
            title_text = re.sub(r"^#+\s*", "", title_text)
            title_text = re.sub(r"^标题\s*:\s*", "", title_text).strip()
            section_id = m.group(4)
            current = {
                "level": level,
                "heading_line": line.strip(),
                "title": title_text,
                "section_id": section_id,
                "content_lines": [],
            }
        else:
            if current is None:
                intro_lines.append(line)
            else:
                current["content_lines"].append(line)
        idx += 1
    # flush last
    if current is not None:
        sections.append(current)

    # 以 section_id 优先作为主键；无 ID 用顺序键，保持顺序，不覆盖。
    seen_keys: set[str] = set()
    seen_section_ids: set[str] = set()
    consolidated: list[dict] = []
    seq_counter = 1

    # 预计算：按规范化标题记录“最后一个带ID的章节”的索引，用于移除先出现的无ID重复
    def _norm_title(t: str) -> str:
        t = (t or "").strip()
        t = re.sub(r"^#+\s*", "", t)
        t = re.sub(r"^标题\s*:\s*", "", t)
        t = re.sub(r"\s+", " ", t)
        return t

    last_labeled_index: dict[str, int] = {}
    for idx, sec in enumerate(sections):
        if sec.get("section_id"):
            last_labeled_index[_norm_title(sec.get("title", ""))] = idx

    for idx, sec in enumerate(sections):
        section_id = sec.get("section_id")

        # 检测并跳过重复的 section_id
        if section_id and section_id in seen_section_ids:
            logging.warning("检测到重复章节 ID '%s'（标题: '%s'），跳过重复项（保留首次出现）", section_id, sec.get("title", "")[:50])
            continue  # 跳过重复的章节

        # 记录已见过的 section_id
        if section_id:
            seen_section_ids.add(section_id)

        key = None
        if section_id:
            key = f"id::{section_id}"
        else:
            key = f"seq::{seq_counter:04d}"
            seq_counter += 1

        # 注意：此时 key 应该不会重复了，因为 section_id 已去重
        if key in seen_keys:
            logging.warning("检测到重复主键 %s（这不应该发生，请检查逻辑）", key)

        # 若该章节无ID，且其规范化标题在后续存在“带ID”的重复，则丢弃该无ID重复，避免“引言之前的重复章节”问题
        if not section_id:
            ntitle = _norm_title(sec.get("title", ""))
            last_idx = last_labeled_index.get(ntitle)
            if last_idx is not None and last_idx > idx:
                logging.info(
                    "移除无ID重复章节 '%s'（后续存在带ID版本，索引 %s）",
                    ntitle,
                    last_idx,
                )
                continue
        seen_keys.add(key)
        consolidated.append(sec)

    final_parts: list[str] = []
    intro = "\n".join(intro_lines).strip()
    if intro:
        final_parts.append(intro)

    for sec in consolidated:
        final_parts.append(sec["heading_line"])
        body = "\n".join(sec["content_lines"]).strip()
        if body:
            final_parts.append(body)

    final_document = "\n\n".join(part for part in final_parts if part)
    logging.info("--- 文档结构整合完成 ---")
    return final_document


def final_post_processing(text: str) -> str:
    """
    对最终文档进行后处理修复。
    """
    logging.info("\n--- 正在对最终文档进行后处理修复 ---")

    processed_text = text or ""
    # 移除迭代元数据注释（保留 section_id 注释）
    processed_text = re.sub(r"(?m)^<!--\s*iteration:[^>]*-->\s*\n?", "", processed_text)
    # 标题规范化：修正 "## # 标题" -> "# 标题"
    processed_text = re.sub(r"(?m)^##\s*#\s+", "# ", processed_text)
    processed_text = re.sub(r"(?m)^#\s*#\s+", "# ", processed_text)
    # 移除可能的脚手架标记行："标题:" / "内容:" / 关键主张/待办任务 标签
    processed_text = re.sub(r"(?m)^\s*标题\s*:\s*.*$", "", processed_text)
    processed_text = re.sub(r"(?m)^\s*内容\s*:\s*.*$", "", processed_text)
    processed_text = re.sub(r"(?m)^\s*[*\-]?\s*关键主张：\s*$", "", processed_text)
    processed_text = re.sub(r"(?m)^\s*[*\-]?\s*待办任务：\s*$", "", processed_text)
    processed_text = re.sub(r"(?m)^\s*分析报告\s*$", "", processed_text)

    # 只保留首个 H1 标题，其余 H1 删除（避免二重标题）
    lines = processed_text.splitlines()
    cleaned_lines: list[str] = []
    seen_h1 = False
    for ln in lines:
        if re.match(r"^#\s+", ln) and not re.match(r"^##\s+", ln):
            if seen_h1:
                # 跳过后续 H1
                continue
            seen_h1 = True
        cleaned_lines.append(ln)
    processed_text = "\n".join(cleaned_lines)

    # 合并多余空行
    rules = [
        ("合并3个及以上的换行符", r"\n{3,}", "\n\n"),
    ]

    for description, pattern, replacement in rules:
        original_text = processed_text
        processed_text = re.sub(pattern, replacement, processed_text)
        if original_text != processed_text:
            logging.info(f"  - (规则) {description}")

    lines = processed_text.splitlines()
    cleaned_lines = [line.rstrip() for line in lines]
    processed_text = "\n".join(cleaned_lines)
    logging.info("  - (规则) 已清理所有行首/行尾的空白字符。")

    # 数学公式完整性检查
    double_dollar_count = processed_text.count("$$")
    if double_dollar_count % 2 != 0:
        processed_text += "\n$$"
        logging.warning("  - 检测到未闭合的块级公式分隔符，已自动补全结尾的 $$ 。")

    inline_dollar_count = processed_text.count("$") - double_dollar_count * 2
    if inline_dollar_count % 2 != 0:
        processed_text += "$"
        logging.warning("  - 检测到未闭合的行内公式分隔符，已自动补全结尾的 $ 。")

    # 记号歧义修正（温和替换）：将文本中的“惯性矩比γ/衰减率γ”区分为 k 与 ζ
    # 仅在相关词附近替换，避免过度影响公式
    processed_text = re.sub(r"惯性矩比\s*γ", "惯性矩比 k", processed_text)
    processed_text = re.sub(r"特征?衰减率\s*γ", "特征衰减率 ζ", processed_text)
    processed_text = re.sub(r"衰减(?:率|参数)\s*γ", "衰减率 ζ", processed_text)

    # 已禁用自动生成占位符：不再添加"符号与参数表"和空的"参考文献"章节

    # 轻度消重：压缩“结论与展望”中与“参数分析与结果讨论”完全重复的句/行
    try:
        concl_m = re.search(r"(?ms)(^##\s*结论与展望.*?$)(.*?)(?=^##\s+|\Z)", processed_text)
        param_m = re.search(r"(?ms)(^##\s*参数分析与结果讨论.*?$)(.*?)(?=^##\s+|\Z)", processed_text)
        if concl_m and param_m:
            concl_head, concl_body = concl_m.group(1), concl_m.group(2)
            param_body = param_m.group(2)
            concl_lines = [ln for ln in concl_body.splitlines()]
            param_lines_set = set(ln.strip() for ln in param_body.splitlines() if ln.strip())
            dedup_lines = []
            for ln in concl_lines:
                if ln.strip() and ln.strip() in param_lines_set:
                    continue
                dedup_lines.append(ln)
            new_concl_block = concl_head + "\n" + "\n".join(dedup_lines)
            processed_text = processed_text[: concl_m.start()] + new_concl_block + processed_text[concl_m.end() :]
    except Exception as exc:
        if isinstance(exc, (SystemExit, KeyboardInterrupt)):
            raise
        logging.warning(
            "Post-processing deduplication skipped due to error: %s",
            exc,
            exc_info=True,
        )

    logging.info("--- 后处理完成 ---")
    return processed_text


def quality_check(config, content: str) -> str:
    """
    (同步版本) 对最终内容进行质量评估。
    """
    from utils.text_processor import truncate_text_for_context_boundary_aware
    from services.llm_interaction import call_ai_writing_with_auto_continue

    content_for_review = truncate_text_for_context_boundary_aware(config, content, int(10000 * getattr(config, "prompt_budget_ratio", 0.9)))
    prompt = f"请深入评估以下内容的质量。为以下方面提供评分(0-10分): 深度、细节、结构、连贯性、问题契合度。并列出主要优缺点。\n\n内容:\n{content_for_review}"
    return call_ai_writing_with_auto_continue(
        config,
        config.secondary_ai_model,
        [{"role": "user", "content": prompt}],
        temperature=config.temperature_factual,
        max_continues=1,
    )
