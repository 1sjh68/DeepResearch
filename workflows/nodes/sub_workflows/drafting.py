import json
import logging
import re
import time

from config import Config
from planning.tool_definitions import SectionContent
from services.llm_interaction import call_ai, call_ai_writing_with_auto_continue  # Use sync call_ai
from utils.progress_tracker import safe_pulse
from utils.text_processor import truncate_text_for_context_boundary_aware


class SectionDraftPipeline:
    """封装单个章节的多块草稿工作流。"""

    def __init__(
        self,
        config: Config,
        section_data: dict,
        system_prompt: str,
        model_name: str,
        overall_context: str = "",
        is_subsection: bool = False,
    ):
        self.config = config
        self.section_data = section_data
        self.system_prompt = system_prompt
        self.model_name = model_name
        self.overall_context = overall_context
        self.is_subsection = is_subsection

        self.section_title = section_data.get("title", "无标题章节")
        self.section_id = section_data.get("id")
        self.section_goal = section_data.get("description", "无特定描述")
        self.must_include_points = section_data.get("must_include") or []
        self.organization_hint = section_data.get("organization_hint", "")
        self.digest_points = section_data.get("digest_points") or []
        self.target_length = section_data.get("allocated_chars", 1000)

        self.header_prefix = "###" if is_subsection else "##"
        self.generated_chunks: list[str] = []
        self.current_total_chars = 0
        self.force_continue = False

        ratio = getattr(config, "prompt_budget_ratio", 0.9)
        self.trimmed_overall_context = truncate_text_for_context_boundary_aware(
            config,
            overall_context,
            max(1024, int((config.max_chunk_tokens // 2) * ratio)),
        )
        self.rag_instructions = "重要提示：为了帮助你写作，系统已从核心参考PDF中检索到以下与本节主题高度相关的原文片段。请在撰写时优先参考、整合和阐述这些片段中的信息，以确保内容的准确性和深度。\n" if "--- 从参考PDF中检索到的高度相关原文片段 ---" in self.trimmed_overall_context else ""

    def run(self) -> str:
        logging.info(
            "\n--- 正在为部分生成内容 (裁剪模式): '%s' (目标: %s 字符) ---",
            self.section_title,
            self.target_length,
        )
        for index in range(self.config.max_chunks_per_section):
            if not self._step(index):
                break

        if self.force_continue:
            logging.warning("  - '%s' 在达到最大迭代次数后仍存在可能未完成的公式或段落。", self.section_title)
            self.generated_chunks.append("\n\n[提示：本节存在可能未完成的公式或段落，请复核补写。]\n")

        return self._compose_final_content()

    # --- Step orchestration -------------------------------------------------
    def _step(self, index: int) -> bool:
        safe_pulse(
            self.config.task_id,
            f"起草 · {self.section_title} · 生成块 {index + 1}/{self.config.max_chunks_per_section}",
        )
        remaining = self.target_length - self.current_total_chars
        if self._should_stop_iteration(remaining):
            logging.info("  - 部分 '%s' 已接近目标长度，停止生成。", self.section_title)
            return False

        if self.force_continue:
            remaining = max(remaining, self.config.min_allocated_chars_for_section)

        messages, max_tokens = self._build_prompt(index, remaining)
        chunk = self._invoke_model(messages, max_tokens)
        if not chunk:
            return False

        clean_chunk = self._post_process_chunk(chunk)
        self.generated_chunks.append(clean_chunk)
        self.current_total_chars += len(clean_chunk)
        safe_pulse(
            self.config.task_id,
            f"起草 · {self.section_title} · 已生成 {self.current_total_chars}/{self.target_length} 字符",
        )
        self.force_continue = self._needs_additional_generation(clean_chunk)
        if self.force_continue:
            logging.info("  - 检测到 '%s' 的文本可能未完整，将尝试继续生成。", self.section_title)
        time.sleep(0.2)  # Prevent overwhelming the API
        return True

    def _should_stop_iteration(self, remaining_chars: int) -> bool:
        return remaining_chars < self.config.min_allocated_chars_for_section and not self.force_continue

    # --- Prompt & model interaction ----------------------------------------
    def _build_prompt(self, index: int, remaining_chars: int) -> tuple[list[dict], int]:
        chars_to_generate = min(remaining_chars, int(self.config.max_chunk_tokens * 2.5))
        max_tokens_for_chunk = max(200, int((chars_to_generate / 2.0) * 1.3))

        if index == 0:
            user_prompt = (
                f"你正在撰写报告中的一个新部分，标题是：'{self.section_title}'。\n"
                f"本部分的写作目标是：{self.section_goal}\n\n"
                f"{self._format_must_include()}"
                f"**格式化规则：**\n"
                f"- **数学公式**: 所有独立成行的数学公式必须使用 `$$ ... $$` 包裹。"
                f"所有嵌入在文本行内的公式必须使用 `$ ... $` 包裹。不要使用 `\\\\[ ... \\\\]` 或 `[ ... ]`。\n\n"
                f"{self._format_digest_block()}"
                f"{self.rag_instructions}"
                f"--- 报告的整体上下文与参考资料 ---\n{self.trimmed_overall_context}\n--- 上下文与参考资料结束 ---\n\n"
                f"请严格依据以上所有信息，开始撰写 '{self.section_title}' 部分的正文。"
                f"每个关键主张都要在句末附上对应的引用标记（例如 [ref: source_id#anchor]）。不要重复标题。"
            )
        else:
            last_chunk_context = truncate_text_for_context_boundary_aware(self.config, self.generated_chunks[-1], self.config.overlap_chars, "tail")
            user_prompt = (
                f"你正在续写关于 '{self.section_title}' 的部分。以下是本部分已生成内容的结尾，供你衔接：\n"
                f"--- 已有内容结尾 ---\n...{last_chunk_context}\n--- 已有内容结尾 ---\n\n"
                f"**格式化规则提醒：**\n"
                f"- **数学公式**: 所有独立成行的数学公式必须使用 `$$ ... $$` 包裹。"
                f"所有嵌入在文本行内的公式必须使用 `$ ... $` 包裹。不要使用 `\\\\[ ... \\\\]` 或 `[ ... ]`。\n\n"
                f"{self._format_must_include(reminder=True)}"
                f"{self._format_digest_block(reminder=True)}"
                f"{self.rag_instructions}"
                f"--- 报告的整体上下文与参考资料（供回顾） ---\n{self.trimmed_overall_context}\n--- 上下文与参考资料结束 ---\n\n"
                f"请基于所有信息，流畅地续写后续内容，确保与前文衔接自然，并继续在主张后附上 [ref: ...] 引用。不要重复标题或已有内容。"
            )

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        return messages, max_tokens_for_chunk

    def _format_must_include(self, reminder: bool = False) -> str:
        if not self.must_include_points and not self.organization_hint:
            return ""
        suffix = "（提醒）" if reminder else ""
        lines: list[str] = []
        if self.must_include_points:
            lines.append(f"必须覆盖的要点{suffix}：")
            for point in self.must_include_points[:6]:
                lines.append(f"- {point}")
        if self.organization_hint:
            lines.append(f"组织建议{suffix}：{self.organization_hint}")
        return "\n".join(lines) + "\n\n"

    def _format_digest_block(self, reminder: bool = False) -> str:
        if not self.digest_points:
            return ""
        suffix = "（提醒：引用格式 [ref: source#anchor]）" if reminder else "（引用格式 [ref: source#anchor]）"
        lines = [f"索引卡片{suffix}:"]
        for point in self.digest_points[:6]:
            fact = point.get("fact", "")
            citation = point.get("citation", "ref:pending")
            snippet = fact[:220].strip()
            lines.append(f"- {snippet} (ref: {citation})")
        return "\n".join(lines) + "\n\n"

    def _invoke_model(self, messages: list[dict], max_tokens: int) -> str | None:
        logging.info("  -> 开始API调用，模型: %s, max_tokens: %s", self.model_name, max_tokens)
        call_start = time.time()
        chunk = call_ai_writing_with_auto_continue(
            self.config,
            self.model_name,
            messages,
            max_tokens_output=max_tokens,
            temperature=self.config.temperature_creative,
            max_continues=1,
        )
        call_duration = time.time() - call_start
        logging.info(
            "  <- API调用完成，返回长度: %s, 耗时: %.2f秒",
            len(chunk) if chunk else 0,
            call_duration,
        )

        if not chunk or "AI模型调用失败" in chunk or not chunk.strip():
            logging.error("  - '%s' 的文本块生成失败，将停止本章节的生成。", self.section_title)
            return None
        return chunk.strip()

    # --- Post processing ----------------------------------------------------
    def _post_process_chunk(self, chunk: str) -> str:
        clean_chunk = chunk.strip()
        last_sentence_end = -1
        for terminator in ["。", "！", "？", ".", "!", "?", "\n\n"]:
            pos = clean_chunk.rfind(terminator)
            if pos != -1 and pos < len(clean_chunk) - len(terminator):
                potential_end = pos + len(terminator)
                if potential_end > last_sentence_end:
                    last_sentence_end = potential_end

        if last_sentence_end != -1 and len(clean_chunk) > last_sentence_end:
            original_len = len(clean_chunk)
            clean_chunk = clean_chunk[:last_sentence_end]
            logging.info(
                "  - 智能裁剪已执行：将文本块从 %s 字符裁剪至 %s 字符，以确保句子完整。",
                original_len,
                len(clean_chunk),
            )
        return clean_chunk

    def _needs_additional_generation(self, text: str) -> bool:
        if not text.strip():
            return False
        double_dollar_count = text.count("$$")
        if double_dollar_count % 2 != 0:
            return True
        inline_dollar_count = text.count("$") - double_dollar_count * 2
        if inline_dollar_count % 2 != 0:
            return True
        if text.count("\\begin{") > text.count("\\end{"):
            return True
        if re.search(r"(\\begin\{[^\}]+)$", text.strip()):
            return True
        return False

    def _compose_final_content(self) -> str:
        if not self.generated_chunks:
            logging.warning("  - 未能为 '%s' 生成任何内容。", self.section_title)
            id_comment = f" <!-- section_id: {self.section_id} -->" if self.section_id else ""
            return f"\n\n{self.header_prefix} {self.section_title}{id_comment} \n\n[本部分内容生成失败或为空]\n\n"

        if len(self.generated_chunks) == 1:
            final_body = self.generated_chunks[0]
        else:
            final_body = self._smooth_transitions()

        logging.info(
            "--- 部分 '%s' 内容生成完毕 (最终长度: %s 字符) ---",
            self.section_title,
            len(final_body),
        )
        id_comment = f" <!-- section_id: {self.section_id} -->" if self.section_id else ""
        return f"\n\n{self.header_prefix} {self.section_title}{id_comment} \n\n{final_body.strip()}\n\n"

    def _smooth_transitions(self) -> str:
        logging.info(
            "  - 开始为 '%s' 优化 %s 个衔接点...",
            self.section_title,
            len(self.generated_chunks) - 1,
        )
        system_prompt = "你是一位精通文本衔接的编辑。你的任务是重写或添加一小段文字，使得两段独立的文本能够流畅、自然地连接在一起。只输出用于衔接的文本，不要有任何额外的解释。"
        final_parts = [self.generated_chunks[0]]

        for idx in range(len(self.generated_chunks) - 1):
            chunk_a = self.generated_chunks[idx]
            chunk_b = self.generated_chunks[idx + 1]
            context_a_end = truncate_text_for_context_boundary_aware(self.config, chunk_a, 500, "tail")
            context_b_start = truncate_text_for_context_boundary_aware(self.config, chunk_b, 500, "head")
            prompt = f"下面是两段连续文本的结尾和开头。请生成一小段（一到三句话）流畅的过渡文字，将它们无缝连接起来。\n[前一段的结尾]\n...{context_a_end}\n\n[后一段的开头]\n{context_b_start}...\n\n现在，请只输出你创作的、用于替换和衔接的过渡段落："
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]
            logging.info("  -> 开始衔接点优化，第 %s 个衔接点", idx + 1)
            transition = call_ai_writing_with_auto_continue(
                self.config,
                self.config.editorial_model_name,
                messages,
                max_tokens_output=256,
                temperature=self.config.temperature_creative,
                max_continues=1,
            )
            logging.info(
                "  <- 衔接点优化完成，返回长度: %s",
                len(transition) if transition else 0,
            )
            if transition and "AI模型调用失败" not in transition and transition.strip():
                logging.info("  - 衔接点 %s 优化成功。", idx + 1)
                final_parts.append("\n\n" + transition.strip())
                final_parts.append("\n\n" + chunk_b)
            else:
                logging.warning("  - 衔接点 %s 优化失败，将使用硬连接。", idx + 1)
                final_parts.append("\n\n" + chunk_b)
        return "".join(final_parts)


def generate_section_content(
    config: Config,
    section_data: dict,
    system_prompt: str,
    model_name: str,
    overall_context: str = "",
    is_subsection: bool = False,
) -> str:
    """
    (同步版本) 为指定的章节生成内容。
    """
    pipeline = SectionDraftPipeline(
        config=config,
        section_data=section_data,
        system_prompt=system_prompt,
        model_name=model_name,
        overall_context=overall_context,
        is_subsection=is_subsection,
    )
    return pipeline.run()


def generate_section_content_structured(
    config: Config,
    section_data: dict,
    system_prompt: str,
    model_name: str,
    overall_context: str = "",
) -> SectionContent | None:
    """
    生成结构化章节内容的函数。
    """
    section_title = section_data.get("title", "无标题章节")
    section_id = section_data.get("id", f"section_{section_title.replace(' ', '_')}")
    section_specific_user_prompt = section_data.get("description", "无特定描述")
    must_include = section_data.get("must_include") or []
    organization_hint = section_data.get("organization_hint", "")
    digest_points = section_data.get("digest_points") or []

    logging.info(f"\n--- 正在生成结构化内容: '{section_title}' ---")

    ratio = getattr(config, "prompt_budget_ratio", 0.9)
    trimmed_overall_context = truncate_text_for_context_boundary_aware(
        config,
        overall_context,
        max(1024, int((config.max_chunk_tokens // 2) * ratio)),
    )

    structured_brief = _compose_structured_brief_for_section(section_title, must_include, organization_hint, digest_points)
    rag_instruction = _rag_instruction_for_context(trimmed_overall_context)
    guidance_note = rag_instruction or "重要提示：严格按照骨架清单逐条覆盖，缺失证据时在 TODO 中标注并在正文引用 [ref: pending]。"

    # 构建结构化提示
    user_prompt = f"""
你正在撰写报告中的一个新部分，标题是：'{section_title}'。
本部分的写作目标是：{section_specific_user_prompt}

{structured_brief}

{guidance_note}

请严格按照以下JSON格式返回你的回答，不要包含任何其他文本：

{{
    "section_id": "{section_id}",
    "title": "{section_title}",
    "content": "这里是章节的正文内容...",
    "key_claims": ["关键主张1", "关键主张2"],
    "todos": ["需要后续处理的任务1", "需要后续处理的任务2"],
    "word_count": 1000
}}

写作与格式化要求：
- 严格使用骨架清单控制段落顺序，逐条核对 must_include 列表。
- 每个关键主张紧跟引用标记，例如 [ref: source_id#anchor]。
- 数学公式：独立成行的使用 $$...$$，嵌入文本的使用 $...$，不要使用 \\\\([ ... \\\\]) 或 [ ... ]。
- key_claims：列出3-5个本章节的核心观点或结论。
- todos：列出需要后续进一步研究或补写的任务点（若无可留空数组）。
- word_count：正文内容的实际字数。

--- 报告的整体上下文与参考资料 ---
{trimmed_overall_context}
--- 上下文与参考资料结束 ---

请直接返回JSON格式的结构化内容，不要包含任何解释或前缀。
"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    try:
        # 使用结构化调用
        response = call_ai(
            config=config,
            model_name=model_name,
            messages=messages,
            max_tokens_output=4000,
            temperature=config.temperature_factual,
            schema=SectionContent,
        )

        if isinstance(response, SectionContent):
            logging.info(f"  <- 结构化生成成功，章节: {response.title}, 字数: {response.word_count}")
            return response
        else:
            logging.warning("结构化调用返回非预期格式，使用文本解析")
            return _parse_structured_response(response, section_id, section_title)

    except Exception as e:
        logging.warning(f"结构化生成失败: {e}，使用回退解析方法")
        return _parse_structured_response("", section_id, section_title)


def _compose_structured_brief_for_section(
    title: str,
    must_include: list[str],
    organization_hint: str,
    digest_points: list[dict[str, str]],
) -> str:
    """镜像非结构化草稿中的骨架/索引卡片提示，保证结构一致。"""
    lines: list[str] = [f"--- Skeleton Checklist · {title} ---"]
    if must_include:
        lines.extend(f"- {point}" for point in must_include[:6])
    else:
        lines.append("- 梳理核心概念、关键数据和比较结论。")

    if organization_hint:
        lines.append(f"- 组织建议：{organization_hint}")

    if digest_points:
        lines.append("--- Indexed Facts (引用 ref: source#anchor) ---")
        for fact in digest_points[:6]:
            snippet = (fact.get("fact") or "").strip().replace("\n", " ")
            citation = fact.get("citation") or "ref:pending"
            if snippet:
                lines.append(f"- {snippet[:220]} (ref: {citation})")
    else:
        lines.append("--- Indexed Facts (引用 ref: source#anchor) ---")
        lines.append("- （当前章节缺少索引卡片，写作时如引用不足请添加 TODO 并标记 [ref: pending]。）")
    return "\n".join(lines)


def _rag_instruction_for_context(context_bundle: str) -> str:
    marker = "--- 从参考PDF中检索到的高度相关原文片段 ---"
    if marker in context_bundle:
        return "重要提示：上方索引卡片已绑定至检索到的原文片段，请优先合并这些事实并逐条附上 [ref: ...] 引用，必要时可在 TODO 中记录缺失证据。"
    return ""


def _parse_structured_response(response_text: str, section_id: str, section_title: str) -> SectionContent | None:
    """尝试从文本响应中解析结构化数据"""
    try:
        # 尝试直接解析JSON
        if response_text.strip().startswith("{"):
            data = json.loads(response_text)
            return SectionContent(
                section_id=data.get("section_id", section_id),
                title=data.get("title", section_title),
                content=data.get("content", ""),
                key_claims=data.get("key_claims", []),
                todos=data.get("todos", []),
                word_count=data.get("word_count", len(data.get("content", ""))),
            )
    except json.JSONDecodeError as e:
        logging.warning(f"JSON parsing failed during structured response parsing: {e}. Content was: '{response_text[:200]}...'")
        pass

    # 如果JSON解析失败，尝试从文本中提取信息
    lines = response_text.split("\n")
    content_lines = []
    key_claims = []
    todos = []

    in_content = True
    in_claims = False
    in_todos = False

    for line in lines:
        line = line.strip()
        if not line:
            continue

        if "关键主张" in line or "key claims" in line.lower():
            in_content = False
            in_claims = True
            continue
        elif "待办" in line or "todos" in line.lower():
            in_claims = False
            in_todos = True
            continue
        elif line.startswith("```"):
            continue

        if in_content:
            content_lines.append(line)
        elif in_claims and line.startswith("-"):
            key_claims.append(line[1:].strip())
        elif in_todos and line.startswith("-"):
            todos.append(line[1:].strip())

    content = "\n".join(content_lines)
    if not content:
        content = f"章节 '{section_title}' 内容生成失败，请手动补充。"

    return SectionContent(
        section_id=section_id,
        title=section_title,
        content=content,
        key_claims=key_claims,
        todos=todos,
        word_count=len(content),
    )


__all__ = ["generate_section_content", "generate_section_content_structured"]
