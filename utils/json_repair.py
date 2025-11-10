"""
JSON 修复和规范化引擎

提供统一的JSON修复、规范化和验证功能，从 llm_interaction.py 中提取。
"""

import ast
import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

# 导入文本处理依赖（避免循环导入）
try:
    from utils.latex_handler import smart_remove_control_chars_with_latex_recovery
except ImportError:
    smart_remove_control_chars_with_latex_recovery = None  # type: ignore

try:
    from utils.text_processor import preprocess_json_string
except ImportError:
    preprocess_json_string = None  # type: ignore

# 尝试导入专业JSON修复库
try:
    from json_repair import repair_json as repair_json_lib

    JSON_REPAIR_AVAILABLE = True
    logger.info("json-repair库已加载，启用增强JSON修复功能")
except ImportError:
    JSON_REPAIR_AVAILABLE = False
    repair_json_lib = None  # type: ignore
    logger.info("json-repair库未安装，使用内置JSON修复逻辑。建议: pip install json-repair")

# 尝试导入 Pydantic
try:
    from pydantic import BaseModel as PydanticBaseModel
except ImportError:
    PydanticBaseModel = None  # type: ignore


def _safe_model_validate(schema: type[Any], data: Any) -> Any:
    """兼容不同 Pydantic 版本的模型校验。"""
    if hasattr(schema, "model_validate"):
        return schema.model_validate(data)  # type: ignore[attr-defined]
    return schema(**data)


def _normalize_plan_chapter(node: Any) -> None:
    """递归修补 PlanChapter 中缺失或错误的字段，并过滤无效章节。处理常见字段别名。"""
    if not isinstance(node, dict):
        return

    # === 处理章节内的字段别名 ===
    # section_title → title
    if "section_title" in node and not node.get("title"):
        node["title"] = node.pop("section_title")

    # estimated_characters → target_chars_ratio（绝对值字段，移除让后续自动分配）
    if "estimated_characters" in node and not node.get("target_chars_ratio"):
        node.pop("estimated_characters", None)

    # === 修复标题 ===
    title = str(node.get("title") or "").strip()
    if not title:
        title = "未命名章节"
    node["title"] = title

    # === 修复描述 ===
    description = str(node.get("description") or "").strip()
    if not description:
        node["description"] = title
    else:
        node["description"] = description

    # === 修复比例 ===
    ratio = node.get("target_chars_ratio")
    if isinstance(ratio, str):
        try:
            node["target_chars_ratio"] = float(ratio)
        except ValueError:
            node["target_chars_ratio"] = None

    # === 递归处理子章节（同时过滤无效章节）===
    sections = node.get("sections")
    if not isinstance(sections, list):
        node["sections"] = []
    else:
        # 递归规范化并过滤无效子章节
        valid_sections = []
        for child in sections:
            if isinstance(child, dict):
                # 先递归规范化
                _normalize_plan_chapter(child)

                # 过滤完全空的章节（标题为空且无实质内容）
                child_title = str(child.get("title") or "").strip()
                child_desc = str(child.get("description") or "").strip()
                child_sections = child.get("sections") or []

                # 保留有效章节：有非空标题，或有描述，或有子章节
                if (child_title and child_title != "未命名章节") or child_desc or child_sections:
                    valid_sections.append(child)
                else:
                    logging.debug(
                        "_normalize_plan_chapter: 过滤无效章节 (title='%s', desc='%s', subsections=%d)",
                        child_title,
                        child_desc,
                        len(child_sections),
                    )

        node["sections"] = valid_sections


def massage_structured_payload(schema: type[Any], payload: Any) -> Any:
    """
    根据 schema 特性修补模型响应，主要针对 PlanModel。
    处理常见的字段名变体和别名，提升解析鲁棒性。
    """
    schema_name = getattr(schema, "__name__", "")
    if schema_name != "PlanModel":
        return payload

    if not isinstance(payload, dict):
        return payload

    # === 第1步：处理顶层字段别名 ===
    # 别名映射：document_title → title
    if "document_title" in payload and not payload.get("title"):
        payload["title"] = payload.pop("document_title")

    # 别名映射：sections → outline
    if "sections" in payload and not payload.get("outline"):
        payload["outline"] = payload.pop("sections")

    # 别名映射：chapters → outline（保留原有逻辑）
    if "chapters" in payload and not payload.get("outline"):
        payload["outline"] = payload.pop("chapters")

    # 清理其他可能的无用字段
    for unwanted_key in ["core_objective", "document_plan", "total_estimated_characters"]:
        payload.pop(unwanted_key, None)

    # === 第2步：确保 title 字段存在且非空 ===
    title = str(payload.get("title") or "").strip()
    if not title:
        payload["title"] = "未命名文档"

    # === 第3步：确保 outline 字段存在且是列表 ===
    outline = payload.get("outline")
    if not isinstance(outline, list):
        if outline is None:
            payload["outline"] = []
        else:
            payload["outline"] = []

    # === 第4步：递归规范化每个章节 ===
    outline_list = payload.get("outline")
    if isinstance(outline_list, list):
        for chapter in outline_list:
            _normalize_plan_chapter(chapter)
    else:
        payload["outline"] = []

    return payload


def _fix_invalid_escape_sequences(json_str: str) -> str:
    """
    修复无效的转义序列

    Args:
        json_str: JSON字符串

    Returns:
        修复后的JSON字符串
    """
    if not json_str:
        return json_str

    fixed = json_str

    # 策略：将无效的单反斜杠转为双反斜杠（除了有效的JSON转义）
    # 有效的JSON转义：\" \\ \/ \b \f \n \r \t \uXXXX
    # 无效的LaTeX转义：\alpha \beta \gamma等应该是\\alpha

    # 核心修复：匹配单个反斜杠后跟字母（LaTeX命令）
    # 需要使用负向后顾确保前面不是反斜杠
    try:
        fixed = re.sub(
            r"(?<!\\)\\(?![\"\\/ bfnrtu])([a-zA-Z])",  # 单反斜杠+字母
            r"\\\\\1",  # 替换为双反斜杠+字母
            fixed,
        )

    except Exception as e:
        logging.debug(f"修复invalid escape时出错: {e}")
    return fixed


def _strip_trailing_non_json(content: str) -> str:
    """裁剪掉JSON体之后的非JSON尾部内容（例如多余的注释或Markdown）。"""
    if not content:
        return content

    idx = len(content) - 1
    while idx >= 0 and content[idx] not in "}]":
        idx -= 1

    if idx <= 0:
        return content

    trimmed = content[: idx + 1]
    if len(trimmed) != len(content):
        logging.debug(
            "_strip_trailing_non_json: 截断尾部非JSON片段 (%s -> %s)",
            len(content),
            len(trimmed),
        )
    return trimmed


def _aggressive_truncate_json(content: str) -> str:
    """更激进地截断尾部内容，确保以'}'或']'结束。"""
    if not content:
        return content

    truncated = _truncate_top_level_json(content)
    if truncated != content:
        return truncated

    trimmed = _strip_trailing_non_json(content)
    return trimmed


def _attempt_fix_eof_error(json_str: str) -> str:
    """
    尝试修复EOF错误（JSON被截断）

    Args:
        json_str: 可能被截断的JSON字符串

    Returns:
        尝试补全后的JSON字符串
    """
    if not json_str:
        return json_str

    fixed = json_str.strip()

    # 策略1: 检查是否在字符串内被截断
    # 查找最后一个完整的字段（以 ", 或 ": 结尾）
    last_quote_comma = fixed.rfind('",')
    last_quote_colon = fixed.rfind('":')
    last_complete = max(last_quote_comma, last_quote_colon)

    if last_complete > 0:
        # 截断到最后一个完整位置
        if last_quote_comma > last_quote_colon:
            # 最后是 ", 说明字段值完整
            truncated = fixed[: last_quote_comma + 2]  # 包含 ",
        else:
            # 最后是 ": 说明字段名完整但值缺失，截断到前一个字段
            prev_comma = fixed.rfind(",", 0, last_complete)
            if prev_comma > 0:
                truncated = fixed[: prev_comma + 1]
            else:
                truncated = fixed[:last_complete]

        # 补全未闭合的括号和引号
        open_braces = truncated.count("{")
        close_braces = truncated.count("}")
        open_brackets = truncated.count("[")
        close_brackets = truncated.count("]")

        # 检查字符串引号是否未闭合
        quotes = truncated.count('"')
        # 如果引号数是奇数，添加一个引号
        if quotes % 2 == 1:
            truncated += '"'

        # 添加缺失的闭合符号
        if close_brackets < open_brackets:
            truncated += "]" * (open_brackets - close_brackets)
        if close_braces < open_braces:
            truncated += "}" * (open_braces - close_braces)

        return truncated

    # 策略2: 如果找不到完整字段，尝试简单的括号补全
    return _balance_brackets(fixed)


def _fix_missing_field_values(json_str: str) -> str:
    """
    修复缺失的字段值。

    处理模式:
    - "field": , → "field": null,
    - "field": } → "field": null}
    - "field": ] → "field": null]
    """
    # Pattern 1: 冒号后直接是逗号
    json_str = re.sub(r":\s*,", r": null,", json_str)

    # Pattern 2: 冒号后直接是闭合括号
    json_str = re.sub(r":\s*}", r": null}", json_str)
    json_str = re.sub(r":\s*]", r": null]", json_str)

    return json_str


def _fix_unbalanced_quotes_in_arrays(json_str: str) -> str:
    """
    修复数组中缺失引号的字符串。

    处理模式:
    - , 文本" → , "文本"
    - [ 文本" → [ "文本"
    - \\" → "
    """
    # Pattern 1: 数组中缺少开头引号（逗号后）
    # 匹配: , 后跟非引号/括号的字符，再跟引号
    json_str = re.sub(r',\s*([^",\[\]{}:]+)"', r', "\1"', json_str)

    # Pattern 2: 数组开始处缺少开头引号
    json_str = re.sub(r'\[\s*([^",\[\]{}:]+)"', r'[ "\1"', json_str)

    # Pattern 3: 修复错误的双反斜杠引号（\\")
    json_str = re.sub(r'\\\\"', r'"', json_str)

    return json_str


def _remove_duplicate_closers(json_str: str) -> str:
    """
    移除重复的闭合括号。

    处理模式:
    - }} → }
    - ]] → ]
    """
    # 迭代移除重复的括号
    changed = True
    iterations = 0
    max_iterations = 5  # 防止无限循环

    while changed and iterations < max_iterations:
        original = json_str
        json_str = json_str.replace("}}", "}")
        json_str = json_str.replace("]]", "]")
        changed = json_str != original
        iterations += 1

    return json_str


def _strip_markdown_fence(value: str) -> str:
    """移除Markdown代码块标记"""
    stripped = value.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?", "", stripped, flags=re.IGNORECASE).strip()
        stripped = re.sub(r"```$", "", stripped).strip()
    return stripped


def _normalize_unicode_quotes(value: str) -> str:
    """规范化Unicode引号"""
    replacements = {
        """: '"',  # U+201C LEFT DOUBLE QUOTATION MARK
        """: '"',  # U+201D RIGHT DOUBLE QUOTATION MARK
        "„": '"',  # U+201E DOUBLE LOW-9 QUOTATION MARK
        "‟": '"',  # U+201F DOUBLE HIGH-REVERSED-9 QUOTATION MARK
        "'": "'",  # U+2018 LEFT SINGLE QUOTATION MARK
        "'": "'",  # U+2019 RIGHT SINGLE QUOTATION MARK  # noqa: F601
        "‚": "'",  # U+201A SINGLE LOW-9 QUOTATION MARK
    }
    normalized = value
    for src, dst in replacements.items():
        if src in normalized:
            normalized = normalized.replace(src, dst)
    return normalized


def _fix_latex_with_pylatexenc(text: str) -> str:
    """
    使用pylatexenc专业处理LaTeX（如果可用）

    Args:
        text: 输入文本（可能包含JSON转义的LaTeX）

    Returns:
        修复后的文本
    """
    if not text:
        return text

    try:
        # 尝试导入pylatexenc（用于检测可用性）
        import pylatexenc  # noqa: F401

        fixed = text

        # 1. 修复已知的无效命令
        invalid_to_valid = {
            "cdotp": "cdot",  # KaTeX不支持
            # 可以添加更多映射
        }

        for invalid_cmd, valid_cmd in invalid_to_valid.items():
            pattern = rf"(\\+){invalid_cmd}\b"
            fixed = re.sub(pattern, rf"\1{valid_cmd}", fixed)

        # 2. 转换Unicode数学符号为LaTeX（在JSON中需要双反斜杠）
        unicode_to_latex = {
            "·": "\\\\cdot",
            "×": "\\\\times",
            "÷": "\\\\div",
            "±": "\\\\pm",
            "∈": "\\\\in",
            "∞": "\\\\infty",
            "≈": "\\\\approx",
            "≠": "\\\\neq",
            "≤": "\\\\leq",
            "≥": "\\\\geq",
            "∑": "\\\\sum",
            "∏": "\\\\prod",
            "∫": "\\\\int",
            "√": "\\\\sqrt",
            "α": "\\\\alpha",
            "β": "\\\\beta",
            "γ": "\\\\gamma",
            "δ": "\\\\delta",
            "ε": "\\\\epsilon",
            "θ": "\\\\theta",
            "λ": "\\\\lambda",
            "μ": "\\\\mu",
            "π": "\\\\pi",
            "σ": "\\\\sigma",
            "τ": "\\\\tau",
            "φ": "\\\\phi",
            "ω": "\\\\omega",
        }

        for unicode_char, latex_cmd in unicode_to_latex.items():
            if unicode_char in fixed:
                fixed = fixed.replace(unicode_char, latex_cmd)

        return fixed

    except ImportError:
        # pylatexenc未安装，降级到基础修复
        return _fix_invalid_latex_commands(text)


def _fix_invalid_latex_commands(text: str) -> str:
    """
    修复无效的LaTeX命令（基础版本）

    Args:
        text: 输入文本（可能包含JSON转义的LaTeX）

    Returns:
        修复后的文本
    """
    if not text:
        return text

    fixed = text

    # 修复1: \cdotp → \cdot（KaTeX不支持 \cdotp）
    # 需要处理JSON中的各种转义情况：\cdotp, \\cdotp, \\\\cdotp 等
    # 使用正则表达式匹配任意数量的反斜杠后跟cdotp
    fixed = re.sub(r"(\\+)cdotp\b", r"\1cdot", fixed)

    # 修复2: Unicode中间点符号 · → LaTeX \cdot
    # 在JSON字符串中应该是 \\cdot（双反斜杠）
    if "·" in fixed:
        # 检查是否在JSON字符串上下文中（简单判断：如果有引号包围）
        # 如果是JSON，替换为 \\cdot；否则替换为 \cdot
        # 为了安全起见，统一替换为双反斜杠版本
        fixed = fixed.replace("·", "\\\\cdot")

    return fixed


def _is_likely_trailing_text(text: str) -> bool:
    """
    检测文本是否可能是尾部多余文本（如LLM添加的解释）

    Args:
        text: 待检测的文本

    Returns:
        如果文本主要由非JSON字符组成，返回True
    """
    if not text:
        return False

    # JSON合法字符集（包括数字、关键字、标点、空白）
    json_chars = set('{}[]",:0123456789truefalsenull \\n\\t-._eE+')

    # 计算非JSON字符的比例
    non_json_count = sum(1 for c in text if c not in json_chars)
    non_json_ratio = non_json_count / len(text) if len(text) > 0 else 0

    # 如果超过30%是非JSON字符（如中文、markdown），认为是多余文本
    return non_json_ratio > 0.3


def _looks_like_json_continuation(text: str) -> bool:
    """
    判断文本是否看起来像JSON的延续

    Args:
        text: 待检测的文本

    Returns:
        如果文本以JSON结构字符开头，返回True
    """
    stripped = text.strip()
    if not stripped:
        return False

    # 以JSON结构字符开头可能是延续
    return stripped[0] in '{[,"'


def _contains_json_structure(text: str) -> bool:
    """
    检查文本是否包含JSON结构符号

    Args:
        text: 待检测的文本

    Returns:
        如果文本包含{}或[]，返回True
    """
    return "{" in text or "[" in text


def _truncate_top_level_json(value: str) -> str:
    """
    Trim any trailing fragments that appear after the first well-formed JSON
    object/array (e.g., spurious `, "" }` or duplicated closers injected by
    the provider). This is a defensive heuristic and does not attempt to fix
    unmatched braces; that is handled separately.

    增强版：支持智能检测尾部多余文本，即使超过100字符也能正确截断。
    """
    if not value:
        return value

    working = value.lstrip()
    if working != value:
        logging.debug(
            "_truncate_top_level_json: leading whitespace trimmed (%s -> %s)",
            len(value),
            len(working),
        )
    if not working:
        return working

    first_obj = working.find("{")
    first_arr = working.find("[")

    start = -1
    open_ch = ""
    close_ch = ""

    if first_obj != -1 and (first_arr == -1 or first_obj < first_arr):
        start = first_obj
        open_ch = "{"
        close_ch = "}"
    elif first_arr != -1:
        start = first_arr
        open_ch = "["
        close_ch = "]"
    else:
        return working

    depth = 0
    in_string = False
    escape = False

    for idx in range(start, len(working)):
        ch = working[idx]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
            continue

        if ch == open_ch:
            depth += 1
        elif ch == close_ch:
            if depth == 0:
                # Unexpected extra closer before matching opener; abandon.
                return working
            depth -= 1
            if depth == 0:
                # 找到顶层 JSON 闭合点
                truncated = working[start : idx + 1]
                remaining = working[idx + 1 :].strip()

                # 如果没有剩余内容，直接返回截断结果
                if not remaining:
                    if len(truncated) != len(working):
                        logging.debug(
                            "_truncate_top_level_json: 截断尾部空白 (%s -> %s)",
                            len(working),
                            len(truncated),
                        )
                    return truncated

                # 增强版智能截断逻辑（更aggressive版本）
                # 先尝试解析截断后的JSON，验证其完整性
                try:
                    json.loads(truncated)
                    # JSON解析成功，现在检查剩余内容的性质

                    # 策略0: 如果没有剩余内容，直接返回
                    # （已在前面处理）

                    # 策略1: 剩余内容看起来不像JSON延续 - 优先截断
                    if not _looks_like_json_continuation(remaining):
                        logging.debug(
                            "_truncate_top_level_json: 剩余内容不像JSON延续，截断 (%s chars剩余)",
                            len(remaining),
                        )
                        return truncated

                    # 策略2: 剩余内容主要是非JSON字符（如中文解释、markdown）
                    if _is_likely_trailing_text(remaining):
                        logging.debug(
                            "_truncate_top_level_json: 检测到尾部非JSON文本（%s chars），截断",
                            len(remaining),
                        )
                        return truncated

                    # 策略3: 剩余内容不包含JSON结构符号 - 更aggressive
                    # 降低阈值：只要超过50字符且无JSON结构就截断
                    if len(remaining) > 50 and not _contains_json_structure(remaining):
                        logging.debug(
                            "_truncate_top_level_json: 尾部文本无JSON结构（%s chars），截断",
                            len(remaining),
                        )
                        return truncated

                    # 策略4: 剩余内容较短（<=200字符），更宽松的截断
                    if len(remaining) <= 200:
                        logging.debug(
                            "_truncate_top_level_json: 截断尾部内容 (%s chars)",
                            len(remaining),
                        )
                        return truncated

                    # 策略5: 即使剩余内容较长，如果主要不是JSON也截断
                    # 这是最后的aggressive策略
                    if len(remaining) > 200:
                        # 计算剩余内容中JSON关键字符的比例
                        json_key_chars = sum(1 for c in remaining if c in '{}[]":,')
                        json_ratio = json_key_chars / len(remaining) if remaining else 0

                        # 如果JSON关键字符<20%，认为不是JSON
                        if json_ratio < 0.2:
                            logging.debug(
                                "_truncate_top_level_json: 长尾部内容JSON特征弱（%s chars, ratio=%.2f），截断",
                                len(remaining),
                                json_ratio,
                            )
                            return truncated

                    # 只有在明确是JSON延续时才保留
                    logging.debug(
                        "_truncate_top_level_json: 剩余内容可能是JSON延续 (%s chars)，保留原始内容",
                        len(remaining),
                    )
                    return working

                except json.JSONDecodeError:
                    # 截断后的JSON解析失败，可能需要更多内容，保留原样
                    logging.debug(
                        "_truncate_top_level_json: 截断后JSON解析失败，保留原始内容",
                    )
                    return working

    # 未能找到匹配的闭合符号，返回原始内容以便由其它修复策略处理
    logging.debug("_truncate_top_level_json: 未找到完整 JSON 结构，保留原始内容")
    return working


def _balance_brackets(value: str) -> str:
    """平衡括号"""
    balanced = value
    for open_ch, close_ch in (("{", "}"), ("[", "]")):
        diff = balanced.count(open_ch) - balanced.count(close_ch)
        if diff > 0 and diff < 10:
            balanced = f"{balanced}{close_ch * diff}"
        elif diff < 0 and diff > -10:
            over = -diff
            trimmed = balanced.rstrip()
            # DeepSeek tool-call occasionally emits an extra closing brace at the end.
            # Strip only terminal closers to preserve internal structure.
            while over > 0 and trimmed.endswith(close_ch):
                trimmed = trimmed[:-1]
                over -= 1
            if over == 0:
                balanced = trimmed
    return balanced


def _strip_overclosed_anywhere(value: str) -> str:
    """剔除任意位置的多余关闭符号"""
    buf: list[str] = []
    depth_curly = 0
    depth_square = 0
    in_string = False
    escape = False
    dropped_curly = 0
    dropped_square = 0
    for ch in value:
        if in_string:
            buf.append(ch)
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
            buf.append(ch)
        elif ch == "{":
            depth_curly += 1
            buf.append(ch)
        elif ch == "}":
            if depth_curly > 0:
                depth_curly -= 1
                buf.append(ch)
            else:
                dropped_curly += 1
        elif ch == "[":
            depth_square += 1
            buf.append(ch)
        elif ch == "]":
            if depth_square > 0:
                depth_square -= 1
                buf.append(ch)
            else:
                dropped_square += 1
        else:
            buf.append(ch)

    if dropped_curly or dropped_square:
        logging.debug(
            "_strip_overclosed_anywhere: 删除多余关闭符号 }=%s ]=%s",
            dropped_curly,
            dropped_square,
        )
    return "".join(buf)


def _dump_with_schema(parsed: Any, schema: type[Any]) -> str | None:
    """尝试使用schema规范化已解析的数据"""
    if PydanticBaseModel is None:
        return None
    try:
        if isinstance(parsed, PydanticBaseModel):
            return parsed.model_dump_json(ensure_ascii=False)
        if isinstance(schema, type) and issubclass(schema, PydanticBaseModel):
            instance = schema.model_validate(parsed)  # type: ignore[attr-defined]
            return instance.model_dump_json(ensure_ascii=False)
    except Exception as schema_exc:
        logging.debug("Pydantic schema 归一化失败: %s", schema_exc)
    return None


def _try_normalize(candidate: str, label: str, schema: type[Any]) -> str | None:
    """尝试规范化JSON字符串"""
    candidate = candidate.strip()
    if not candidate:
        return None

    parsed: Any | None = None
    try:
        parsed = json.loads(candidate)
        logging.info("JSON 解析成功（策略：%s）", label)
    except json.JSONDecodeError as json_err:
        logging.debug("JSON 解析失败（策略：%s）: %s", label, json_err)
        needs_literal = any(token in candidate for token in ("'", "True", "False", "None"))
        if needs_literal:
            try:
                parsed = ast.literal_eval(candidate)
                logging.info("通过 ast.literal_eval 修复 JSON（策略：%s）", label)
            except (SyntaxError, ValueError) as literal_err:
                logging.debug("ast.literal_eval 解析失败（策略：%s）: %s", label, literal_err)
                parsed = None
        if parsed is None:
            return None

    if parsed is not None:
        _sanitize_parsed_json(parsed)

    normalized = _dump_with_schema(parsed, schema)
    if normalized:
        return normalized

    try:
        return json.dumps(parsed, ensure_ascii=False)
    except (TypeError, ValueError) as dump_err:
        logging.debug("json.dumps 归一化失败（策略：%s）: %s", label, dump_err)
        return None


def _sanitize_parsed_json(obj: Any) -> None:
    """对成功解析的JSON进行递归清理，例如修正负数的word_count。"""

    if isinstance(obj, dict):
        for key, value in list(obj.items()):
            if isinstance(value, (dict, list)):
                _sanitize_parsed_json(value)
            elif isinstance(value, (int, float)):
                lowered = str(key).lower()
                if "word_count" in lowered and value < 0:
                    logging.debug("_sanitize_parsed_json: 修正负word_count (%s -> 0)", value)
                    obj[key] = 0
            elif isinstance(value, str):
                obj[key] = value.strip()
    elif isinstance(obj, list):
        for item in obj:
            if isinstance(item, (dict, list)):
                _sanitize_parsed_json(item)


def repair_json_once(text: str, schema: type[Any], *, debug: bool = False) -> tuple[str, bool]:
    """
    尝试修复无效 JSON（仅单次尝试），返回修复后的文本及是否获得有效 JSON。

    Args:
        text: 待修复的JSON文本
        schema: Pydantic schema类型
        debug: 是否启用调试模式

    Returns:
        (修复后的文本, 是否成功)
    """
    schema_name = getattr(schema, "__name__", str(schema))
    logging.debug("JSON修复尝试启动，目标schema: %s", schema_name)

    if not text or not text.strip():
        logging.debug("待修复文本为空，跳过JSON修复。")
        return text, False

    attempts: list[tuple[str, str]] = []
    seen: set[str] = set()

    def _register(label: str, candidate: str) -> None:
        candidate = candidate.strip()
        if not candidate:
            return
        if candidate in seen:
            return
        attempts.append((label, candidate))
        logging.debug("repair_json_once - 通过%s注册候选方案(长度=%s)", label, len(candidate))
        seen.add(candidate)

    stripped = _strip_markdown_fence(text)
    _register("原始文本", text)
    _register("去除 Markdown 代码块", stripped)

    # 关键修复：使用LaTeX智能处理模块，先移除控制字符并尝试还原LaTeX命令
    if smart_remove_control_chars_with_latex_recovery:
        control_chars_removed, latex_recovered = smart_remove_control_chars_with_latex_recovery(stripped, debug=debug)
        if control_chars_removed != stripped:
            if latex_recovered > 0:
                _register("LaTeX智能还原", control_chars_removed)
                logging.debug("repair_json_once: LaTeX智能修复还原了 %d 个命令", latex_recovered)
            else:
                _register("移除控制字符", control_chars_removed)
    else:
        control_chars_removed = stripped

    if preprocess_json_string:
        preprocessed = preprocess_json_string(control_chars_removed)
        if preprocessed != control_chars_removed:
            logging.debug(
                "repair_json_once: preprocess_json_string modified content (len=%s -> %s)",
                len(control_chars_removed),
                len(preprocessed),
            )
        _register("正则预处理+反斜杠修复", preprocessed)
    else:
        preprocessed = control_chars_removed

    truncated = _truncate_top_level_json(preprocessed)
    if truncated != preprocessed:
        logging.debug(
            "repair_json_once: _truncate_top_level_json applied (len=%s -> %s)",
            len(preprocessed),
            len(truncated),
        )
        _register("截断尾部噪声", truncated)
    else:
        truncated = preprocessed

    normalized_quotes = _normalize_unicode_quotes(truncated)
    _register("替换特殊引号", normalized_quotes)

    # 修复无效的LaTeX命令（优先使用pylatexenc）
    latex_fixed = _fix_latex_with_pylatexenc(normalized_quotes)
    if latex_fixed != normalized_quotes:
        _register("修复LaTeX命令(pylatexenc)", latex_fixed)
        normalized_quotes = latex_fixed

    escape_prefixed = _fix_invalid_escape_sequences(normalized_quotes)
    if escape_prefixed != normalized_quotes:
        _register("预修复转义", escape_prefixed)
        normalized_quotes = escape_prefixed

    balanced = _balance_brackets(normalized_quotes)
    _register("补全括号", balanced)

    stripped_overclosers = _strip_overclosed_anywhere(balanced)
    _register("剔除多余关闭符号", stripped_overclosers)

    # === 新增：增强修复流程 ===

    # 步骤1: 使用json-repair库（如果可用）
    if JSON_REPAIR_AVAILABLE and repair_json_lib:
        try:
            library_repaired = repair_json_lib(stripped_overclosers, return_objects=False)
            _register("json-repair库修复", library_repaired)

            # 验证修复结果
            normalized = _try_normalize(library_repaired, "json-repair库", schema)
            if normalized is not None:
                logging.debug("JSON 修复成功（策略：json-repair库，原始长度=%s，修复后长度=%s）", len(stripped_overclosers), len(normalized))
                return normalized, True
        except Exception as e:
            logging.debug("json-repair库处理失败，继续使用备用修复: %s", e)

    # 步骤2: 自定义针对性修复（备用）
    value_fixed = _fix_missing_field_values(stripped_overclosers)
    if value_fixed != stripped_overclosers:
        _register("修复缺失字段值", value_fixed)
        normalized = _try_normalize(value_fixed, "修复缺失字段值", schema)
        if normalized is not None:
            logging.debug("JSON 修复成功（策略：修复缺失字段值，原始长度=%s，修复后长度=%s）", len(stripped_overclosers), len(normalized))
            return normalized, True

    quote_fixed = _fix_unbalanced_quotes_in_arrays(value_fixed)
    if quote_fixed != value_fixed:
        _register("修复引号配对", quote_fixed)
        normalized = _try_normalize(quote_fixed, "修复引号配对", schema)
        if normalized is not None:
            logging.debug("JSON 修复成功（策略：修复引号配对，原始长度=%s，修复后长度=%s）", len(value_fixed), len(normalized))
            return normalized, True

    final_clean = _remove_duplicate_closers(quote_fixed)
    if final_clean != quote_fixed:
        _register("最终清理重复括号", final_clean)
        normalized = _try_normalize(final_clean, "最终清理", schema)
        if normalized is not None:
            logging.debug("JSON 修复成功（策略：最终清理，原始长度=%s，修复后长度=%s）", len(quote_fixed), len(normalized))
            return normalized, True

    aggressive = _aggressive_truncate_json(final_clean)
    if aggressive != final_clean:
        _register("强截断尾部噪声", aggressive)
        normalized = _try_normalize(aggressive, "强截断", schema)
        if normalized is not None:
            logging.debug("JSON 修复成功（策略：强截断，原始长度=%s，修复后长度=%s）", len(final_clean), len(normalized))
            return normalized, True

    # === 步骤3: 专门处理各种JSON错误类型 ===
    # 如果到这里还没有成功，尝试专门检测并修复各种JSON错误
    error_specific_candidates = []

    for label, candidate in attempts:
        try:
            json.loads(candidate)
            # 如果能解析成功，直接使用
            normalized = _try_normalize(candidate, label, schema)
            if normalized is not None:
                logging.debug("JSON 修复成功（策略：%s，原始长度=%s，修复后长度=%s）", label, len(candidate), len(normalized))
                return normalized, True
        except json.JSONDecodeError as e:
            error_msg = str(e).lower()

            # 错误类型1: EOF错误（JSON被截断）
            if "eof while parsing" in error_msg:
                logging.debug("检测到EOF错误: %s", e)
                eof_fixed = _attempt_fix_eof_error(candidate)
                if eof_fixed != candidate:
                    error_specific_candidates.append(("eof_修复", eof_fixed))

            # 错误类型2: invalid escape（无效转义）
            elif "invalid escape" in error_msg or "invalid \\escape" in error_msg:
                logging.debug("检测到invalid escape错误: %s", e)
                escape_fixed = _fix_invalid_escape_sequences(candidate)
                if escape_fixed != candidate:
                    error_specific_candidates.append(("escape_修复", escape_fixed))

            # 错误类型3: trailing characters（尾部多余字符）
            elif "trailing characters" in error_msg:
                # 检测到trailing characters错误，尝试多种截断策略
                logging.debug("检测到trailing characters错误: %s", e)

                # 策略1: 查找第一个完整的JSON对象/数组
                first_brace = candidate.find("{")
                first_bracket = candidate.find("[")

                if first_brace != -1 or first_bracket != -1:
                    # 使用增强的_truncate_top_level_json
                    truncated = _truncate_top_level_json(candidate)
                    if truncated != candidate:
                        error_specific_candidates.append(("trailing_chars_截断", truncated))

                    # 策略2: 手动查找最后一个}或]
                    last_brace = candidate.rfind("}")
                    last_bracket = candidate.rfind("]")
                    last_close = max(last_brace, last_bracket)

                    if last_close > 0:
                        manual_truncate = candidate[: last_close + 1]
                        error_specific_candidates.append(("trailing_chars_手动截断", manual_truncate))

                    # 策略3: 移除markdown代码块标记后截断
                    no_fence = _strip_markdown_fence(candidate)
                    if no_fence != candidate:
                        truncated_no_fence = _truncate_top_level_json(no_fence)
                        error_specific_candidates.append(("trailing_chars_无fence", truncated_no_fence))

                    # 策略4: 更激进的截断
                    aggressive_tail = _aggressive_truncate_json(candidate)
                    if aggressive_tail != candidate:
                        error_specific_candidates.append(("trailing_chars_强截断", aggressive_tail))

    # 尝试所有错误特定的修复候选方案
    for label, candidate in error_specific_candidates:
        _register(label, candidate)
        normalized = _try_normalize(candidate, label, schema)
        if normalized is not None:
            logging.debug("JSON 修复成功（策略：%s，原始长度=%s，修复后长度=%s）", label, len(candidate), len(normalized))
            return normalized, True

    # === 原有逻辑继续 ===

    for label, candidate in attempts:
        normalized = _try_normalize(candidate, label, schema)
        if normalized is not None:
            logging.debug("JSON 修复成功（策略：%s，原始长度=%s，修复后长度=%s）", label, len(candidate), len(normalized))
            return normalized, True

    logging.error(
        "JSON 修复失败：schema=%s，尝试策略=%s，原始长度=%s，原始片段=%s",
        schema_name,
        ",".join(label for label, _ in attempts),
        len(text) if isinstance(text, str) else "n/a",
        text[:200].replace("\n", " "),
    )
    return text, False


class JSONRepairEngine:
    """JSON修复引擎的面向对象封装"""

    def __init__(self, *, debug: bool = False):
        """
        初始化JSON修复引擎

        Args:
            debug: 是否启用调试模式
        """
        self.debug = debug

    def repair(self, text: str, schema: type[Any]) -> tuple[str, bool]:
        """
        修复JSON文本

        Args:
            text: 待修复的JSON文本
            schema: Pydantic schema类型

        Returns:
            (修复后的文本, 是否成功)
        """
        return repair_json_once(text, schema, debug=self.debug)

    @staticmethod
    def normalize_mapping(data: Any) -> dict:
        """
        规范化映射数据为字典

        Args:
            data: 任意数据类型

        Returns:
            规范化后的字典
        """
        from collections.abc import Mapping

        if not isinstance(data, Mapping):
            return {}

        normalized: dict[str, Any] = {}
        for key, value in data.items():
            normalized[str(key)] = value
        return normalized

    @staticmethod
    def validate_schema(data: dict, schema: type[Any]) -> Any:
        """
        使用schema验证数据

        Args:
            data: 待验证的字典
            schema: Pydantic schema类型

        Returns:
            验证后的schema实例
        """
        # 先应用结构化载荷按摩（针对PlanModel）
        massaged_data = massage_structured_payload(schema, data)
        # 然后验证
        return _safe_model_validate(schema, massaged_data)
