# utils/text_processor.py

import hashlib
import json
import logging
import re
from collections import Counter
from collections.abc import Mapping
from typing import Any

# 从重构后的模块中导入依赖
from config import Config
from services.llm_interaction import call_ai, call_ai_writing_with_auto_continue  # Use sync call_ai


# 修复 8: 通用类型检查和安全取值函数
def safe_get_dict(obj: Any, key: str, default: bool = False) -> bool:
    """
    修复 8: 安全地从字典中获取布尔值

    处理类型转换和异常情况
    """
    if not isinstance(obj, Mapping):
        return default

    value = obj.get(key)
    if value is None:
        return default

    if isinstance(value, bool):
        return value

    if isinstance(value, (list, str)):
        return bool(len(value))

    return bool(value)


def safe_get_dict_value(obj: Any, key: str, default: Any = None) -> Any:
    """安全地从字典中获取任意类型值"""
    if not isinstance(obj, Mapping):
        return default

    value = obj.get(key)
    return value if value is not None else default


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


def truncate_text_for_context(config: Config, text: str, max_tokens: int, truncation_style: str = "middle") -> str:
    """
    根据 token 数量安全地截断文本，以适应模型的上下文窗口。
    """
    if not text:
        return ""

    if not config.encoder:
        logging.warning("Tiktoken 编码器不可用，将使用基于字符的近似截断。")
        char_limit = max_tokens * 3
        if len(text) <= char_limit:
            return text
        logging.info(f"    - 正在截断文本: {len(text)} chars -> {char_limit} chars (方式: {truncation_style})")
        if truncation_style == "head":
            return text[:char_limit] + "\n... [内容已截断] ..."
        if truncation_style == "tail":
            return "... [内容已截断] ...\n" + text[-char_limit:]
        half = char_limit // 2
        return text[:half] + "\n... [中间内容已截断] ...\n" + text[-half:]

    tokens = config.encoder.encode(text)
    if len(tokens) <= max_tokens:
        return text

    logging.info(f"    - 正在截断文本: {len(tokens)} tokens -> {max_tokens} tokens (方式: {truncation_style})")

    decode_fn = config.encoder.decode
    if truncation_style == "head":
        truncated_tokens = tokens[:max_tokens]
        return decode_fn(truncated_tokens) + "\n... [内容已截断，只显示开头部分] ..."
    elif truncation_style == "tail":
        truncated_tokens = tokens[-max_tokens:]
        return "... [内容已截断，只显示结尾部分] ...\n" + decode_fn(truncated_tokens)
    else:  # middle
        h_len = max_tokens // 2
        t_len = max_tokens - h_len
        head_part = decode_fn(tokens[:h_len])
        tail_part = decode_fn(tokens[-t_len:])
        return head_part + "\n... [中间内容已截断] ...\n" + tail_part


def truncate_text_for_context_boundary_aware(config: Config, text: str, max_tokens: int, truncation_style: str = "middle") -> str:
    """
    边界感知的安全截断：在不改变既有策略的基础上，尽量在句/段/公式边界处截断，减少半句/半公式现象。
    - 不负责应用提示词预算比例（由调用方控制）。
    - 在 tiktoken 缺失时回退到字符近似，并做轻量边界调整。
    """
    if not text:
        return ""

    def _balance_inline_math(s: str) -> str:
        # 尝试平衡行内 $...$（粗略处理，忽略 $$ 块）
        doubles = s.count("$$")
        singles = s.count("$") - doubles * 2
        if singles % 2 != 0:
            s += "$"
        return s

    def _trim_unmatched_block_math(s: str) -> str:
        # 若 $$ 计数为奇数，说明处于块公式内部，向前回退到最近的 $$ 之前
        if s.count("$$") % 2 == 1:
            last = s.rfind("$$")
            if last != -1:
                s = s[:last]
        return s

    def _trim_unbalanced_brackets(s: str) -> str:
        # 轻量配对检查：若存在未闭合的 ( [ {，回退到最后一个未闭合处之前
        pairs = [("(", ")"), ("[", "]"), ("{", "}")]
        for open_ch, close_ch in pairs:
            if s.count(open_ch) > s.count(close_ch):
                last_open = s.rfind(open_ch)
                if last_open != -1 and last_open > len(s) - 300:
                    s = s[:last_open]
        return s

    def _find_last_sentence_boundary(s: str) -> int:
        # 优先选择句末边界；对小数点进行保护（避免 2.5 这种小数被当作句点）
        candidates = []
        boundary_chars = ["。", "!", "?", "！", "？", "；", ";", "\n"]
        for ch in boundary_chars:
            pos = s.rfind(ch)
            if pos != -1:
                # 小数点保护：若是 '.'，确保不是数字相邻
                if ch == ".":
                    prev_c = s[pos - 1] if pos - 1 >= 0 else ""
                    next_c = s[pos + 1] if pos + 1 < len(s) else ""
                    if prev_c.isdigit() and next_c.isdigit():
                        continue
                candidates.append(pos)
        return max(candidates) if candidates else -1

    def _align_head_boundary(s: str) -> str:
        # 从末尾向前寻找句子边界或换行
        idx = _find_last_sentence_boundary(s)
        if idx != -1:
            s = s[: idx + 1]
        # 尝试修复块公式与括号
        s = _trim_unmatched_block_math(s)
        s = _trim_unbalanced_brackets(s)
        return _balance_inline_math(s)

    def _align_tail_boundary(s: str) -> str:
        # 从开头向后跳到最近的边界之后，避免在半句中开始
        if not s:
            return s
        boundary_chars = "\n。.!?！？；;"
        first_positions = [pos for pos in (s.find(ch) for ch in boundary_chars) if pos != -1]
        if first_positions:
            cut = min(first_positions)
            s = s[cut + 1 :].lstrip()
        # 若以块公式中途开始，跳至下一个 $$ 之后
        prefix = s[:300]
        if prefix.count("$$") == 1:
            nxt = s.find("$$", len(prefix))
            if nxt != -1:
                s = s[nxt + 2 :].lstrip()
        # 若以 LaTeX 环境中途开始，跳至第一个 \end{...} 之后
        begin_pos = s.find("\\begin{")
        end_pos = s.find("\\end{")
        if 0 <= end_pos < begin_pos or (end_pos != -1 and begin_pos == -1):
            # 认为开头在环境内部
            end_match = re.search(r"\\end\{[^\}]+\}", s)
            if end_match:
                s = s[end_match.end() :].lstrip()
        # 避免以关闭括号开头
        while s and s[0] in ")]}】》〉」』﴾］］｣」】〉》":
            s = s[1:].lstrip()
        return s

    # 无编码器：字符近似
    if not config.encoder:
        char_limit = max_tokens * 3
        if len(text) <= char_limit:
            return text
        if truncation_style == "head":
            head = text[:char_limit]
            return _align_head_boundary(head) + "\n... [内容已截断] ..."
        if truncation_style == "tail":
            tail = text[-char_limit:]
            return "... [内容已截断] ...\n" + _align_tail_boundary(tail)
        # middle
        half = char_limit // 2
        head = _align_head_boundary(text[:half])
        tail = _align_tail_boundary(text[-(char_limit - half) :])
        return head + "\n... [中间内容已截断] ...\n" + tail

    # token 截断
    tokens = config.encoder.encode(text)
    if len(tokens) <= max_tokens:
        return text

    decode_fn = config.encoder.decode
    if truncation_style == "head":
        truncated_tokens = tokens[:max_tokens]
        head = decode_fn(truncated_tokens)
        return _align_head_boundary(head) + "\n... [内容已截断，只显示开头部分] ..."
    if truncation_style == "tail":
        truncated_tokens = tokens[-max_tokens:]
        tail = decode_fn(truncated_tokens)
        return "... [内容已截断，只显示结尾部分] ...\n" + _align_tail_boundary(tail)

    # middle
    h_len = max_tokens // 2
    t_len = max_tokens - h_len
    head_part = decode_fn(tokens[:h_len])
    tail_part = decode_fn(tokens[-t_len:])
    head_part = _align_head_boundary(head_part)
    tail_part = _align_tail_boundary(tail_part)
    return head_part + "\n... [中间内容已截断] ...\n" + tail_part


def calculate_checksum(data: str) -> str:
    """计算字符串的 SHA256 校验和，用于比较内容是否有变化。"""
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


# _strip_json_comments函数已被json-repair库替代，已删除


def preprocess_json_string(json_string: str) -> str:
    """
    使用 json-repair 库修复常见的 LLM 生成的 JSON 错误。
    这个函数已被简化，使用专业库替代了 200+ 行自定义逻辑。
    """
    if not json_string or json_string.isspace():
        return ""

    try:
        # 使用 json-repair 库进行修复
        from json_repair import repair_json

        # 先做一些基本的预处理（保留原有的关键修复）
        processed_string = json_string.strip()

        # 移除markdown代码块标记
        match = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", processed_string, re.DOTALL)
        if match:
            processed_string = match.group(1).strip()

        # 使用 json-repair 进行智能修复
        repaired = repair_json(processed_string, return_objects=False)

        if isinstance(repaired, str):
            return repaired
        else:
            # 如果返回的是对象，转换回字符串
            import json

            return json.dumps(repaired, ensure_ascii=False)

    except ImportError:
        # 如果 json-repair 未安装，回退到基本修复
        logging.warning("json-repair library not installed. Using basic fallback.")
        processed_string = json_string.strip()

        # 基本修复（保留最关键的几个）
        processed_string = re.sub(r",\s*([}\]])", r"\1", processed_string)  # 移除尾逗号
        processed_string = re.sub(r"\bTrue\b", "true", processed_string)
        processed_string = re.sub(r"\bFalse\b", "false", processed_string)
        processed_string = re.sub(r"\bNone\b", "null", processed_string)

        return processed_string
    except Exception as e:
        logging.warning(f"JSON修复失败: {e}")
        return json_string.strip()


def extract_json_from_ai_response(config: Config, response_text: str, context_for_error_log: str = "AI response") -> str | None:
    """
    (同步版本) 使用“三振出局”策略从 AI 的文本响应中稳健地提取 JSON 字符串。
    """
    logging.debug(f"尝试从以下内容提取JSON: {response_text[:300]}... 上下文: {context_for_error_log}")

    def _try_parse(s_to_parse, stage_msg):
        if not s_to_parse or s_to_parse.isspace():
            return None
        try:
            json.loads(s_to_parse)
            logging.info(f"  JSON 在 {stage_msg} 阶段解析成功。")
            return s_to_parse
        except json.JSONDecodeError:
            logging.debug(f"  JSON 在 {stage_msg} 阶段解析失败。")
            return None

    if (parsed_str := _try_parse(response_text, "直接解析")) is not None:
        return parsed_str

    pre_repaired_str = preprocess_json_string(response_text)
    if (parsed_str := _try_parse(pre_repaired_str, "正则预处理")) is not None:
        return parsed_str

    logging.info("  JSON 解析在预处理后仍然失败，尝试调用 AI 修复...")
    fixer_prompt = (
        "The following text is supposed to be a valid JSON string, but it's malformed. "
        "Please fix it and return ONLY the corrected, valid JSON string. Do not add any "
        "explanations, apologies, or markdown formatting like ```json ... ```.\n\n"
        "Malformed JSON attempt:\n"
        "```\n"
        f"{pre_repaired_str}\n"
        "```\n\n"
        "Corrected JSON string:"
    )

    fixer_model = getattr(config, "json_fixer_model_name", None)
    if not fixer_model or "reasoner" in str(fixer_model).lower():
        fixer_model = "deepseek-coder"
    ai_fixed_str = call_ai(
        config,
        fixer_model,
        [{"role": "user", "content": fixer_prompt}],
        max_tokens_output=max(2048, int(len(pre_repaired_str) * 1.5)),
        temperature=0.0,
    )

    if "AI模型调用失败" in ai_fixed_str or not ai_fixed_str.strip():
        logging.error("  AI JSON 修复调用失败或返回空。")
        return None

    final_attempt_str = preprocess_json_string(ai_fixed_str)
    if (parsed_str := _try_parse(final_attempt_str, "AI 修复后")) is not None:
        return parsed_str

    logging.error("在所有三个阶段（直接、预处理、AI修复）后，都无法从响应中解析出有效的 JSON。")
    return None


def extract_knowledge_gaps(feedback: str) -> list[str]:
    """从审稿人的反馈中提取知识空白列表。"""
    if not feedback:
        return []

    # 尝试直接解析 JSON 结构
    try:
        parsed = json.loads(feedback)
        if isinstance(parsed, dict):
            gaps = parsed.get("knowledge_gaps")
            if isinstance(gaps, list):
                return [str(gap).strip() for gap in gaps if str(gap).strip()]
    except json.JSONDecodeError as exc:
        logging.debug("反馈 JSON 解析失败，尝试正则提取知识空白: %s", exc)

    match = re.search(
        r"###?\s*(KNOWLEDGE GAPS|知识鸿沟)\s*###?\s*\n(.*?)(?=\n###?|\Z)",
        feedback,
        re.DOTALL | re.IGNORECASE,
    )
    if not match:
        logging.info("反馈中未找到 'KNOWLEDGE GAPS' 或 '知识鸿沟' 部分。")
        return []

    content = match.group(2).strip()
    gaps = [g.strip() for g in re.split(r"\n\s*(?:\d+\.|–|•)\s*", content) if g.strip()]

    logging.info(f"从反馈中提取了 {len(gaps)} 个知识空白。")
    return gaps


MetadataDict = dict[str, Any]
ChunkResult = tuple[list[str], list[MetadataDict]]


def chunk_document_for_rag(config: Config, document_text: str, doc_id: str) -> ChunkResult:
    """
    为 RAG 对原始文本文档进行分块（使用 LangChain 智能分块）。
    """
    logging.info(f"  正在为 RAG 对文档 (doc_id: {doc_id}) 进行分块...")

    if not document_text or not document_text.strip():
        logging.warning("  chunk_document_for_rag: 文档文本为空/无效。返回空块。")
        return [], []

    try:
        # 使用 LangChain 的智能分块器
        # 兼容新旧版本的langchain
        try:
            from langchain_text_splitters import RecursiveCharacterTextSplitter
        except ImportError:
            from langchain.text_splitter import RecursiveCharacterTextSplitter  # pyright: ignore[reportMissingImports]

        # 配置分块器
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.embedding_model_max_tokens * 3,  # 近似字符数
            chunk_overlap=config.overlap_chars,
            length_function=config.count_tokens if config.encoder else len,  # 使用 token 计数或字符数
            separators=["\n\n", "\n", "。", ".", " ", ""],  # 智能分隔符（中英文支持）
        )

        # 执行分块
        chunks = splitter.split_text(document_text)

        # 生成元数据
        metadatas = [{"doc_id": doc_id, "chunk_index": i} for i in range(len(chunks))]

        logging.info(f"  文档 RAG 分块完成（LangChain 智能分块）。共生成 {len(chunks)} 个块。")
        return chunks, metadatas

    except ImportError:
        # 如果 LangChain 未安装，回退到基本分块
        logging.warning("  LangChain not installed. Using basic fallback chunking.")
        chunks: list[str] = []
        metadatas: list[MetadataDict] = []

        max_chars = max(1, config.embedding_model_max_tokens * 3)
        overlap_chars = max(0, min(config.overlap_chars, max_chars - 1))
        step = max(1, max_chars - overlap_chars)

        for start in range(0, len(document_text), step):
            end = min(start + max_chars, len(document_text))
            chunk = document_text[start:end]
            if chunk and not chunk.isspace():
                chunks.append(chunk)
                metadatas.append({"doc_id": doc_id, "chunk_index": len(chunks) - 1})
            if end == len(document_text):
                break

        logging.info(f"  文档 RAG 分块完成（字符回退模式）。共生成 {len(chunks)} 个块。")
        return chunks, metadatas


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

    begin_envs = Counter(re.findall(r"\\begin\{([^\}]+)\}", processed_text))
    end_envs = Counter(re.findall(r"\\end\{([^\}]+)\}", processed_text))
    for env, count in begin_envs.items():
        diff = count - end_envs.get(env, 0)
        if diff > 0:
            processed_text += "".join(f"\n\\end{{{env}}}" for _ in range(diff))
            logging.warning(f"  - 检测到 {env} 环境未闭合，已自动追加 {diff} 个 \\end{{{env}}} 。")

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


def quality_check(config: Config, content: str) -> str:
    """
    (同步版本) 对最终内容进行质量评估。
    """
    content_for_review = truncate_text_for_context_boundary_aware(config, content, int(10000 * getattr(config, "prompt_budget_ratio", 0.9)))
    prompt = f"请深入评估以下内容的质量。为以下方面提供评分(0-10分): 深度、细节、结构、连贯性、问题契合度。并列出主要优缺点。\n\n内容:\n{content_for_review}"
    return call_ai_writing_with_auto_continue(
        config,
        config.secondary_ai_model,
        [{"role": "user", "content": prompt}],
        temperature=config.temperature_factual,
        max_continues=1,
    )
