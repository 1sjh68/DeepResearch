"""
Polish模块的内容处理器

包含结构化润色、提示词构建、响应处理等核心逻辑
"""

from __future__ import annotations

import logging
import re
from typing import Any

from pydantic import ValidationError

from planning.tool_definitions import (
    PolishReference,
    PolishSection,
    PolishSectionResponse,
    SentenceEdit,
)
from services.llm_interaction import call_ai
from utils.json_repair import repair_json_once
from utils.text_processor import extract_json_from_ai_response


def _sanitize_section_content(raw: str) -> str:
    """
    清理润色结果中常见的脚手架痕迹（如"标题:""内容:" 标记、代码围栏等）。
    """
    if not raw:
        return ""

    content = raw.replace("\r\n", "\n").strip()
    if not content:
        return ""

    # 去除包裹的代码块围栏
    content = re.sub(r"^```[a-zA-Z0-9_-]*\s*\n", "", content)
    content = re.sub(r"\n```$", "", content)

    # 剔除"标题:"/"内容:"行（无论是否带补充文字或 section_id 注释）
    content = re.sub(r"^\s*标题\s*:\s*.*(?:<!--.*?-->)?\s*$", "", content, flags=re.MULTILINE)
    content = re.sub(r"^\s*内容\s*:\s*.*$", "", content, flags=re.MULTILINE)
    content = re.sub(r"^\s*分析报告\s*$", "", content, flags=re.MULTILINE)

    # 合并多余空行并裁剪
    content = re.sub(r"\n{3,}", "\n\n", content)
    return content.strip()


def build_section_polish_prompt(
    section: dict[str, Any],
    style_guide: str,
    critique: str = "",
    research_data: dict[str, Any] | None = None,
    iteration_info: str | None = None,
) -> str:
    """
    构建章节润色提示
    """
    prompt = rf"""
你是一位专业的学术润色编辑，任务是对以下【待润色章节】进行精细的结构化润色。

**输出格式要求（严格遵守）：**
你的响应必须是且仅是一个有效的JSON对象。不要在JSON前后添加任何文本、解释或标记。
请返回严格的 JSON 对象，字段必须完全匹配以下结构：
{{
  "revised_content": "润色后的章节内容",
  "modifications": [
    {{
      "original_sentence": "原句",
      "revised_sentence": "修改后句子"
    }}
  ],
  "references": [
    {{
      "reference_id": "ref_0",
      "citation_text": "引用内容或来源提示",
      "source_type": "extracted",
      "confidence_level": 0.6,
      "source_info": {{"note": "可为空的补充信息"}}
    }}
  ],
  "quality_metrics": {{
    "clarity": 0.8,
    "coherence": 0.9,
    "style_alignment": 0.85
  }},
  "fact_check_points": ["需要事实核查的要点1", "需要事实核查的要点2"],
  "revision_notes": "修改说明，可用于提示残留问题或保留原文的原因",
  "word_count": 123
}}

**重要规则：**
1. 保持原意不变，仅改善表达。
2. 维持学术写作的专业性。
3. `revised_content` 只能包含润色后的正文 Markdown/LaTeX，不得添加"标题:""内容:"等脚手架，也不得嵌入 JSON 或额外元数据；确保所有 LaTeX 环境完整闭合。
4. 输出必须是单一 JSON 对象，严禁在前后添加解释、道歉或 ```json 代码块。
5. 如无修改，`modifications` 返回空数组 `[]`，不要伪造占位编辑。
6. 所有数值字段（如 `confidence_level`、`clarity` 等）必须为 0~1 范围内的数字；若未知请使用 `0.5`，禁止留空字符串。
7. 不要修改标题行和 `section_id` 注释。
8. 若因信息缺失无法生成合规 JSON，请返回 `{{"error":"unrecoverable_polish_failure","reason":"<简要原因>"}}`，不要输出自由文本。
9. **JSON转义规则（极其重要）**：在JSON字符串中，所有LaTeX/数学公式中的反斜杠必须转义为双反斜杠。例如：`\\alpha` 而不是 `\alpha`，`\\mathbf{{x}}` 而不是 `\mathbf{{x}}`，`\\frac{{1}}{{2}}` 而不是 `\frac{{1}}{{2}}`。这是JSON标准要求，否则会导致解析失败。
10. **数学符号使用规则（重要）**：
   - **禁止直接使用Unicode中间点符号 `·`**。必须使用LaTeX命令 `\\cdot` 表示数学乘法（注意JSON中需要双反斜杠）。
   - **禁止使用 `\\cdotp`**（KaTeX不支持此命令）。正确的命令是 `\\cdot`。
   - 数学符号示例：使用 `a \\cdot b` 而不是 `a · b` 或 `a \\cdotp b`。
   - 其他常用数学符号：`\\times`（乘号×）、`\\div`（除号÷）、`\\pm`（正负号±）。
   - 所有数学公式必须使用标准LaTeX命令，确保KaTeX能正确渲染。

**内容完整性要求（强化版）：**
- **CRITICAL**: 所有句子必须以正确的标点符号结束（句号、感叹号、问号），不得在句子中间截断。
- **CRITICAL**: 所有括号必须成对出现：圆括号、方括号、花括号、书名号。检查每个左括号都有对应的右括号。
- **CRITICAL**: 数学公式必须完整，不得出现截断的公式（如"L^2 = ..."、"($0."等）。
- **CRITICAL**: 如果内容接近 token 限制，优先保证最后一句话的完整性，必要时删减前面的次要内容。
- 若段落与原文或其他章节重复，请改写或合并，避免冗余。
- 若本章节与上一章节类型不同（例如从理论转到实验），请添加过渡句，说明实验如何验证模型假设或预测。
- 涉及实验或数据时说明数据处理流程、噪声抑制与误差分析方法。
- 使用引用占位符（如 [1]）或提供来源提示，并确保所有符号在正文和符号表中保持一致；若存在歧义，请在 revision_notes 中标注。
- 无引用时 `references` 返回空数组；无法提供事实核查要点时 `fact_check_points` 返回空数组。禁止使用 null 或空字符串。

**输出前自检**:
1. 检查最后一句是否完整（有结尾标点）
2. 数一数左括号和右括号的数量是否相等
3. 检查是否有未完成的公式或数字

**JSON格式检查清单（输出前必须检查）：**
✓ 确保所有字段名用双引号包裹，如 "revised_content"（不是 'revised_content' 或 revised_content）
✓ 确保所有字符串值用双引号包裹
✓ 确保所有键值对之间有冒号 :（不是 = 或其他符号）
✓ 确保数组元素之间、对象字段之间用逗号分隔（最后一个元素/字段后不要有逗号）
✓ 确保JSON对象以 {{ 开始，以 }} 结束，中间没有遗漏的括号
✓ 确保JSON输出完整，没有被截断
✓ 确保JSON之后没有任何额外字符或说明

**风格指南：**
{style_guide}

**评审意见（供参考）：**
{critique if critique else "无特定评审意见"}

**研究数据（供参考）：**
{str(research_data) if research_data else "无额外研究数据"}

**待润色章节：**
标题: {section["title"]}
内容:
{section["content"]}

现在请返回结构化的JSON结果。
"""
    return prompt


def process_structured_polish_response(
    section: dict[str, Any],
    response_data: dict[str, Any],
    original_references: list[PolishReference] | None = None,
) -> PolishSection:
    """
    处理结构化润色响应
    """
    from .quality_checker import _detect_text_anomalies, _should_revert_due_to_anomalies

    if original_references is None:
        original_references = []

    # 若模型提供引用信息，尝试解析并合并
    response_references = response_data.get("references")
    if response_references:
        parsed_references: list[PolishReference] = []
        for ref in response_references:
            try:
                parsed_references.append(PolishReference.model_validate(ref))
            except ValidationError as exc:
                logging.debug("引用解析失败，忽略该条目: %s (%s)", ref, exc, exc_info=True)
        if parsed_references:
            original_references = parsed_references

    # 提取修改记录
    modifications = []
    if "modifications" in response_data and response_data["modifications"]:
        for mod in response_data["modifications"]:
            if "original_sentence" in mod and "revised_sentence" in mod:
                modifications.append(
                    SentenceEdit(
                        original_sentence=mod["original_sentence"],
                        revised_sentence=mod["revised_sentence"],
                    )
                )

    # 计算字数
    revised_content = response_data.get("revised_content", section["content"])
    revised_content = _sanitize_section_content(revised_content)

    anomalies = _detect_text_anomalies(revised_content)
    revision_notes = response_data.get("revision_notes")
    revert_due_to_anomalies = _should_revert_due_to_anomalies(anomalies, section.get("content", ""), revised_content)
    if anomalies:
        anomaly_text = "; ".join(anomalies)
        if revert_due_to_anomalies:
            logging.warning(
                "章节 '%s' 润色输出存在严重完整性问题：%s。已回退为原始内容。",
                section.get("title", "未命名"),
                anomaly_text,
            )
            revised_content = section["content"]
            revision_notes = (revision_notes + " | " if revision_notes else "") + f"AI 输出存在完整性问题（{anomaly_text}），已保留原始内容。"
        else:
            # 简化日志：只显示章节标题，详细信息放到 DEBUG
            logging.info("章节 '%s' 润色完成（有轻微告警）", section.get("title", "未命名")[:30])
            logging.debug("  完整性告警详情: %s", anomaly_text)
            revision_notes = (revision_notes + " | " if revision_notes else "") + f"检测到潜在完整性告警（{anomaly_text}），请人工复核。"

    word_count = len(revised_content)

    return PolishSection(
        section_id=section.get("section_id", "unknown"),
        title=section.get("title", ""),
        content=revised_content,
        original_content=section.get("original_content", section.get("content", "")),
        modifications=modifications,
        references=original_references,
        quality_metrics=response_data.get("quality_metrics"),
        word_count=word_count,
        revision_notes=revision_notes,
        fact_check_points=response_data.get("fact_check_points", []),
    )


def polish_section_text_fallback(
    section: dict[str, Any],
    response_text: str,
    original_references: list[PolishReference] | None = None,
    config: Any = None,
) -> PolishSection:
    """
    章节润色的文本模式回退
    """
    from .quality_checker import _detect_text_anomalies, _should_revert_due_to_anomalies

    if original_references is None:
        original_references = []

    # 简单的文本处理与清理：移除可能的"标题:"/"内容:"脚手架残留
    original_clean = _sanitize_section_content(section["content"])
    revision_notes = "文本模式润色"
    raw_response = response_text or ""
    stripped_raw = raw_response.strip()
    force_revert_to_original = False

    if stripped_raw.startswith("{") or stripped_raw.startswith("["):
        logging.info("检测到文本模式响应疑似 JSON，尝试恢复结构化内容。")
        json_text: str | None = None
        try:
            debug_mode = getattr(getattr(config, "workflow", None), "debug_json_repair", False) if config else False
            repaired_text, repaired = repair_json_once(stripped_raw, PolishSectionResponse, debug=debug_mode)
            logging.debug(
                "polish_section_text_fallback: repair_json_once repaired=%s, length=%s",
                repaired,
                len(repaired_text) if isinstance(repaired_text, str) else "n/a",
            )
            json_text = repaired_text if repaired else stripped_raw
        except Exception as exc:  # pragma: no cover - 防御性日志
            logging.debug("repair_json_once 在文本回退阶段失败: %s", exc, exc_info=True)

        if json_text:
            try:
                parsed_obj = PolishSectionResponse.model_validate_json(json_text)
                logging.info("文本模式回退成功恢复结构化响应。")
                return process_structured_polish_response(
                    section=section,
                    response_data=parsed_obj.model_dump(),
                    original_references=original_references,
                )
            except ValidationError as exc:
                logging.debug(
                    "回退 JSON 解析失败: %s | 片段=%s",
                    exc,
                    json_text[:200].replace("\n", " ") if isinstance(json_text, str) else "<non-str>",
                    exc_info=True,
                )

        logging.warning(
            "章节 '%s' 文本模式响应疑似 JSON 但无法解析，保留原始内容。片段=%s",
            section.get("title", "未命名"),
            json_text[:200].replace("\n", " ") if isinstance(json_text, str) else "<non-str>",
        )
        force_revert_to_original = True

    candidate_clean = _sanitize_section_content(response_text) if response_text else original_clean

    if force_revert_to_original:
        candidate_clean = original_clean
        revision_notes += "（AI 输出疑似 JSON 但无法解析，已保留原文）"

    # 生成简单的修改记录
    modifications = []
    if candidate_clean != original_clean:
        # 基本的修改检测（可扩展为更智能的句子比较）
        if len(section["content"]) > 0:
            modifications.append(
                SentenceEdit(
                    original_sentence=section["content"][:100] + "...",
                    revised_sentence=candidate_clean[:100] + "...",
                )
            )

    anomalies = _detect_text_anomalies(candidate_clean)
    revert_due_to_anomalies = _should_revert_due_to_anomalies(anomalies, section.get("content", ""), candidate_clean)
    if anomalies:
        anomaly_text = "; ".join(anomalies)
        if revert_due_to_anomalies:
            logging.warning(
                "章节 '%s' 文本模式输出存在严重完整性问题：%s。保留原始内容。",
                section.get("title", "未命名"),
                anomaly_text,
            )
            candidate_clean = original_clean
            revision_notes += "（AI 输出存在完整性问题，已保留原文）"
            modifications = []
        else:
            logging.warning(
                "章节 '%s' 文本模式输出存在轻微完整性告警：%s。保留润色文本，标记复核。",
                section.get("title", "未命名"),
                anomaly_text,
            )
            revision_notes += f"（检测到潜在完整性告警：{anomaly_text}，请复核）"

    return PolishSection(
        section_id=section.get("section_id", "unknown"),
        title=section.get("title", ""),
        content=candidate_clean,
        original_content=section.get("original_content", section.get("content", "")),
        modifications=modifications,
        references=original_references,
        quality_metrics=None,
        word_count=len(candidate_clean),
        revision_notes=revision_notes,
        fact_check_points=None,
    )


def polish_section_structured(
    config,
    section: dict[str, Any],
    style_guide: str,
    critique: str = "",
    research_data: dict[str, Any] | None = None,
    references: list[PolishReference] | None = None,
    iteration_info: str | None = None,
) -> PolishSection:
    """
    对单个章节进行结构化润色
    """
    if references is None:
        references = []

    # 构建章节级润色提示
    polish_prompt = build_section_polish_prompt(
        section=section,
        style_guide=style_guide,
        critique=critique,
        research_data=research_data,
        iteration_info=iteration_info,
    )

    try:
        polished_section: PolishSection
        # 调用AI进行润色
        # 统一上限为 8192，避免超过底层模型的安全上限
        section_token_budget = getattr(config, "polish_section_max_tokens", 8192)
        structured_response = call_ai(
            config,
            config.editorial_model_name,
            messages=[{"role": "user", "content": polish_prompt}],
            temperature=config.temperature_creative,
            max_tokens_output=section_token_budget,
            schema=PolishSectionResponse,
        )

        if isinstance(structured_response, PolishSectionResponse):
            response_obj: PolishSectionResponse = structured_response
            response_data = response_obj.model_dump()
            return process_structured_polish_response(
                section=section,
                response_data=response_data,
                original_references=references,
            )

        if isinstance(structured_response, str) and structured_response.strip():
            logging.warning("章节润色返回原始字符串，尝试解析 JSON。")
            json_response_text = extract_json_from_ai_response(
                config,
                structured_response,
                context_for_error_log="Polish section response",
            )
            if json_response_text:
                try:
                    response_obj: PolishSectionResponse = PolishSectionResponse.model_validate_json(json_response_text)
                    polished_section = process_structured_polish_response(
                        section=section,
                        response_data=response_obj.model_dump(),
                        original_references=references,
                    )
                    return polished_section
                except ValidationError as exc:
                    logging.warning("章节润色 JSON 解析失败: %s", exc)
            polished_section = polish_section_text_fallback(section, structured_response, references, config)
            return polished_section

        raise ValueError("章节润色调用返回空响应或未知格式。")

    except Exception as e:
        logging.error(f"章节润色AI调用失败: {e}")
        return polish_section_text_fallback(section, "", references, config)
