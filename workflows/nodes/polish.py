from __future__ import annotations

import logging
import re
import time
from typing import Any

from pydantic import ValidationError

from core.progress import StepOutput, step_result, workflow_step
from core.state_manager import WorkflowStateAdapter
from planning.tool_definitions import (
    PolishModel,
    PolishReference,
    PolishSection,
    PolishSectionResponse,
    SentenceEdit,
    extract_references_from_text,
    validate_references_quality,
)
from services.llm_interaction import call_ai
from utils.citation import CitationManager, CitationMatch, SourceInfo
from utils.factcheck import FactChecker, FactCheckResult, add_verification_markers
from utils.iteration_storage import archive_iteration_snapshot
from utils.json_repair import repair_json_once
from utils.progress_tracker import safe_pulse, safe_step_update
from utils.text_processor import extract_json_from_ai_response
from workflows.graph_state import GraphState
from workflows.nodes.sub_workflows.polishing import perform_final_polish

POLISH_STEP_NAME = "polish_node"


# 修复5: 最终内容验证函数
def _validate_final_solution(content: str, config) -> tuple[bool, str]:
    """
    验证最终输出内容的质量。

    参数：
        content: 待验证的内容
        config: 配置对象

    返回：
        (是否有效, 错误消息或"OK")
    """
    min_length = getattr(config, "min_final_content_length", 100)
    max_length = getattr(config, "max_final_content_length", 1000000)

    # 检查1: 不为空
    if not content or not content.strip():
        return False, "内容为空"

    # 检查2: 长度范围
    if len(content) < min_length:
        return False, f"内容过短({len(content)} < {min_length})"

    if len(content) > max_length:
        logging.warning("内容超过推荐长度: %d > %d", len(content), max_length)

    # 检查3: 有效的Markdown结构
    if not re.search(r"^#+\s+", content, re.MULTILINE):
        return False, "未找到Markdown标题"

    # 检查4: 足够的内容行
    non_empty_lines = len([line for line in content.split("\n") if line.strip()])
    if non_empty_lines < 3:
        return False, f"有效内容行数过少: {non_empty_lines} < 3"

    return True, "OK"


@workflow_step(POLISH_STEP_NAME, "润色文档")
def polish_node(state: GraphState) -> StepOutput:
    """
    润色文档节点 - 升级为结构化输出
    """
    workflow_state = WorkflowStateAdapter.ensure(state)
    config = workflow_state.config
    draft_content = workflow_state.draft_content or ""
    style_guide = workflow_state.style_guide or ""
    critique = workflow_state.critique or ""
    structured_research_data = workflow_state.structured_research_data or {}

    # 引用功能配置
    enable_citations = getattr(config, "enable_citation_management", True)
    citation_style = getattr(config, "citation_style", "numeric")

    if not draft_content:
        logging.warning("polish_node: No draft_content found to polish. Returning empty solution.")
        return step_result({"final_solution": ""}, "缺少草稿内容")

    safe_pulse(
        config.task_id,
        f"迭代 {workflow_state.refinement_count}/{config.max_refinement_rounds} · 结构化润色中...",
    )

    try:
        # 执行结构化润色
        iteration_info = f"迭代 {workflow_state.refinement_count}/{config.max_refinement_rounds}"
        polished_result = perform_structured_polish(
            config=config,
            original_content=draft_content,
            style_guide=style_guide,
            critique=critique,
            research_data=structured_research_data,
            iteration_info=iteration_info,
            enable_citations=enable_citations,
            citation_style=citation_style,
        )

        # 集成事实核查功能
        fact_check_results = integrate_fact_checking(
            polished_result=polished_result,
            structured_research_data=structured_research_data,
            config=config,
        )

        # 将事实核查结果添加到polish_result中
        polished_result.fact_check_results = fact_check_results

        # 保存迭代快照
        archive_iteration_snapshot(
            config,
            workflow_state.refinement_count,
            "structured_polish",
            polished_result.model_dump_json(indent=2, ensure_ascii=False),
        )

        # 修复5: 验证最终内容的有效性
        final_solution = polished_result.polished_content
        valid, reason = _validate_final_solution(final_solution, config)

        if not valid:
            logging.error("最终内容验证失败: %s", reason)
            logging.warning("使用回退方案: 原始草稿")
            final_solution = draft_content if draft_content else "# 错误\n\n内容生成失败"
        else:
            logging.info("最终内容验证通过: %s", reason)

        return step_result(
            {
                "final_solution": final_solution,
                "polish_model": polished_result,
                "fact_check_results": fact_check_results,
                "metadata": {
                    "sections_count": len(polished_result.sections),
                    "total_references": len(polished_result.all_references),
                    "quality_score": polished_result.overall_quality_score,
                    "validation_needed": polished_result.validation_needed,
                    "fact_check_performed": len(fact_check_results) > 0 if fact_check_results else False,
                    "validation_status": reason,
                },
            },
            "结构化润色完成",
        )

    except Exception as e:
        logging.error(f"结构化润色失败，回退到传统模式: {e}", exc_info=True)
        return polish_node_fallback(state)


def polish_node_fallback(state: GraphState) -> StepOutput:
    """
    润色节点的回退机制 - 使用传统文本输出
    """
    workflow_state = WorkflowStateAdapter.ensure(state)
    config = workflow_state.config
    draft_content = workflow_state.draft_content or ""
    style_guide = workflow_state.style_guide or ""

    if not draft_content:
        logging.warning("polish_node_fallback: No draft_content found to polish. Returning empty solution.")
        return step_result({"final_solution": ""}, "缺少草稿内容")

    safe_pulse(config.task_id, "回退模式 · 传统润色中...")

    iteration_info = "传统润色模式"
    polished_solution = perform_final_polish(config, draft_content, style_guide, iteration_info=iteration_info)

    return step_result({"final_solution": polished_solution}, "传统润色完成")


def perform_structured_polish(
    config,
    original_content: str,
    style_guide: str,
    critique: str = "",
    research_data: dict[str, Any] | None = None,
    iteration_info: str | None = None,
    enable_citations: bool = True,
    citation_style: str = "numeric",
) -> PolishModel:
    """
    执行结构化润色，返回PolishModel

    Args:
        config: 配置对象
        original_content: 原始内容
        style_guide: 风格指南
        critique: 评审意见
        research_data: 研究数据
        iteration_info: 迭代信息

    Returns:
        PolishModel: 结构化的润色结果
    """
    logging.info("--- 开始结构化润色流程 ---")

    # 初始化引用管理器
    citation_manager = None
    if enable_citations:
        citation_manager = initialize_citation_manager(research_data)

    # 解析文档结构
    sections = parse_document_structure(original_content)
    total_sections = len(sections) or 1
    polished_sections = []
    all_references = []
    fact_check_points = []

    for i, section in enumerate(sections):
        logging.info(f"处理章节 {i + 1}/{len(sections)}: {section.get('title', '未命名')}")
        safe_step_update(
            config.task_id,
            POLISH_STEP_NAME,
            (i / total_sections) * 100.0,
            f"润色章节 {i + 1}/{len(sections)}",
        )

        try:
            # 提取章节引用
            section_references = extract_references_from_text(section["content"])

            # 执行章节润色
            polished_section = polish_section_structured(
                config=config,
                section=section,
                style_guide=style_guide,
                critique=critique,
                research_data=research_data,
                references=section_references,
                iteration_info=f"{iteration_info} · 章节 {i + 1}",
            )

            polished_sections.append(polished_section)
            all_references.extend(polished_section.references)

            safe_step_update(
                config.task_id,
                POLISH_STEP_NAME,
                ((i + 1) / total_sections) * 100.0,
                f"完成润色章节 {i + 1}/{len(sections)}",
            )

            # 如果启用引用标注，处理主张提取和引用匹配
            if citation_manager and enable_citations:
                try:
                    claims = citation_manager.extract_claims(section["content"])
                    matches = citation_manager.align_claims_to_sources(claims)
                    citation_manager.citations.extend(matches)
                except Exception as e:
                    logging.warning(f"章节 {i + 1} 引用处理失败: {e}")

            # 收集事实核查点（向后兼容无该字段的旧模型）
            _points = getattr(polished_section, "fact_check_points", None)
            if _points:
                fact_check_points.extend(_points)

        except Exception as e:
            logging.error(f"章节 {section.get('title', '未命名')} 润色失败: {e}")
            # 回退到原始章节
            polished_sections.append(
                PolishSection(
                    section_id=section.get("section_id", f"fallback_{i}"),
                    title=section.get("title", f"章节 {i + 1}"),
                    content=section["content"],
                    original_content=section["content"],
                    modifications=[],
                    references=[],
                    quality_metrics=None,
                    word_count=len(section["content"]),
                    revision_notes="润色失败，保持原样",
                    fact_check_points=None,
                )
            )
            safe_step_update(
                config.task_id,
                POLISH_STEP_NAME,
                ((i + 1) / total_sections) * 100.0,
                f"完成润色章节 {i + 1}/{len(sections)}（回退原文）",
            )

    # 组装最终结果
    document_title = extract_document_title(original_content)

    polished_content = assemble_final_content(
        polished_sections,
        citation_manager,
        citation_style,
        document_title=document_title,
    )

    # 验证引用质量
    reference_validation = validate_references_quality(all_references)

    # 计算整体质量评分
    quality_score = calculate_quality_score(polished_sections)

    # 生成修改总结
    modification_summary = generate_modification_summary(polished_sections)

    # 创建结构化结果
    result = PolishModel(
        sections=polished_sections,
        document_title=document_title,
        polished_content=polished_content,
        original_content=original_content,
        metadata={
            "polish_timestamp": time.time(),
            "style_guide_applied": bool(style_guide),
            "critique_considered": bool(critique),
            "research_data_incorporated": bool(research_data),
            "total_modifications": sum(len(s.modifications) for s in polished_sections),
        },
        overall_quality_score=quality_score,
        modification_summary=modification_summary,
        all_references=all_references,
        reference_validation_status=reference_validation,
        fact_check_points=fact_check_points if fact_check_points else None,
        validation_needed=len(fact_check_points) > 0,
        citations_enabled=enable_citations,
        citation_style=citation_style,
        fact_check_results=None,
    )

    logging.info(f"结构化润色完成 - 章节数: {len(polished_sections)}, 引用数: {len(all_references)}, 质量评分: {quality_score:.2f}")
    if not sections:
        safe_step_update(
            config.task_id,
            POLISH_STEP_NAME,
            100.0,
            "润色完成",
        )

    return result


def parse_document_structure(content: str) -> list[dict]:
    """
    解析文档结构，提取章节信息
    """
    sections = []

    # 按##标题分割章节
    chapter_pattern = r"(?=^##\s+.+$)"
    chapters = re.split(chapter_pattern, content, flags=re.MULTILINE)

    for i, chapter in enumerate(chapters):
        if not chapter.strip():
            continue

        lines = chapter.split("\n")
        title_line = lines[0] if lines else f"章节 {i + 1}"
        body = "\n".join(lines[1:]) if len(lines) > 1 else ""

        # 提取标题，优先复用既有 section_id 注释
        section_id = None
        section_id_match = re.search(
            r"<!--\s*section_id:\s*([A-Za-z0-9_-]+)\s*-->",
            title_line,
        )
        if section_id_match:
            section_id = section_id_match.group(1).strip() or None
            title_line = (title_line[: section_id_match.start()] + title_line[section_id_match.end() :]).strip()

        title_match = re.match(r"^##\s*(.+)$", title_line)
        title = title_match.group(1).strip() if title_match else title_line.strip()
        # 规范化标题，避免出现诸如 "## # 标题" 的嵌套级别
        title = re.sub(r"^#+\s*", "", title).strip()
        section_id = section_id or f"section_{i + 1}"

        sections.append({
            "section_id": section_id,
            "title": title,
            "content": body.strip(),
            "original_content": body.strip(),
        })

    return sections


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

**内容完整性要求：**
- 输出完整句子，严禁出现被截断的括号、公式或数字（如"($0."、"L^2 = ..."等）。
- 若段落与原文或其他章节重复，请改写或合并，避免冗余。
- 若本章节与上一章节类型不同（例如从理论转到实验），请添加过渡句，说明实验如何验证模型假设或预测。
- 涉及实验或数据时说明数据处理流程、噪声抑制与误差分析方法。
- 使用引用占位符（如 [1]）或提供来源提示，并确保所有符号在正文和符号表中保持一致；若存在歧义，请在 revision_notes 中标注。
- 无引用时 `references` 返回空数组；无法提供事实核查要点时 `fact_check_points` 返回空数组。禁止使用 null 或空字符串。

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


def _sanitize_section_content(raw: str) -> str:
    """
    清理润色结果中常见的脚手架痕迹（如“标题:”“内容:” 标记、代码围栏等）。
    """
    if not raw:
        return ""

    content = raw.replace("\r\n", "\n").strip()
    if not content:
        return ""

    # 去除包裹的代码块围栏
    content = re.sub(r"^```[a-zA-Z0-9_-]*\s*\n", "", content)
    content = re.sub(r"\n```$", "", content)

    # 剔除“标题:”/“内容:”行（无论是否带补充文字或 section_id 注释）
    content = re.sub(r"^\s*标题\s*:\s*.*(?:<!--.*?-->)?\s*$", "", content, flags=re.MULTILINE)
    content = re.sub(r"^\s*内容\s*:\s*.*$", "", content, flags=re.MULTILINE)
    content = re.sub(r"^\s*分析报告\s*$", "", content, flags=re.MULTILINE)

    # 合并多余空行并裁剪
    content = re.sub(r"\n{3,}", "\n\n", content)
    return content.strip()


def _detect_text_anomalies(text: str) -> list[str]:
    """
    粗略检测文本是否存在截断、未闭合括号等明显完整性问题。
    """
    if not text:
        return []

    anomalies: list[str] = []
    normalized = text.replace("\r\n", "\n")

    delimiter_pairs = [
        ("(", ")"),
        ("（", "）"),
        ("[", "]"),
        ("【", "】"),
    ]
    for left, right in delimiter_pairs:
        if normalized.count(left) != normalized.count(right):
            anomalies.append(f"未闭合的括号 {left}{right}")

    truncated_number_pattern = re.compile(r"[（(][^）()\n]*\$?\d+(?:\.[^\d\)\]]*)?$")
    paragraphs = re.split(r"\n\s*\n", normalized)
    for para in paragraphs:
        stripped = para.strip()
        if not stripped:
            continue
        if len(stripped) > 80:
            tail = stripped[-1]
            if tail not in "。！？?!.;:）)】]”\"'":
                if not (stripped.endswith("$$") or stripped.endswith("$") or stripped.endswith("\\]")):
                    anomalies.append("长句缺乏结尾标点，疑似被截断")
                    break
        if truncated_number_pattern.search(stripped):
            anomalies.append("括号中的数字或公式疑似被截断")
            break

    # 去重
    seen = set()
    unique_anomalies = []
    for item in anomalies:
        if item not in seen:
            seen.add(item)
            unique_anomalies.append(item)
    return unique_anomalies


def _should_revert_due_to_anomalies(anomalies: list[str], original: str, candidate: str) -> bool:
    """
    决定是否因为检测到的异常而回退到原文。

    仅当异常明显且替换文本显著短于原始内容时才回退，避免轻微警告吞掉有效修改。
    """
    if not anomalies:
        return False

    original_clean = (original or "").strip()
    candidate_clean = (candidate or "").strip()
    if not candidate_clean:
        return True

    original_length = len(original_clean)
    candidate_length = len(candidate_clean)
    # original_length 已在前面确保非零，无需使用 max()
    if original_length and candidate_length / original_length < 0.45:
        return True

    # 多条异常基本可判断为不可恢复的输出
    return len(anomalies) >= 2


def process_structured_polish_response(
    section: dict[str, Any],
    response_data: dict[str, Any],
    original_references: list[PolishReference] | None = None,
) -> PolishSection:
    """
    处理结构化润色响应
    """
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
            logging.warning(
                "章节 '%s' 润色输出存在轻微完整性告警：%s。保留润色结果，标记复核。",
                section.get("title", "未命名"),
                anomaly_text,
            )
            revision_notes = (revision_notes + " | " if revision_notes else "") + f"检测到潜在完整性告警（{anomaly_text}），请人工复核。"

    word_count = len(revised_content)

    return PolishSection(
        section_id=section["section_id"],
        title=section["title"],
        content=revised_content,
        original_content=section["original_content"],
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
    if original_references is None:
        original_references = []

    # 简单的文本处理与清理：移除可能的“标题:”/“内容:”脚手架残留
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
        section_id=section["section_id"],
        title=section["title"],
        content=candidate_clean,
        original_content=section["original_content"],
        modifications=modifications,
        references=original_references,
        quality_metrics=None,
        word_count=len(candidate_clean),
        revision_notes=revision_notes,
        fact_check_points=None,
    )


def extract_document_title(content: str) -> str:
    """
    提取文档标题
    """
    lines = content.split("\n")
    for line in lines:
        line = line.strip()
        if line.startswith("#"):
            cleaned = re.sub(r"^#+\s*", "", line)
            if cleaned:
                return cleaned.strip()
    return "未命名文档"


def calculate_quality_score(sections: list[PolishSection]) -> float:
    """
    计算整体质量评分
    """
    if not sections:
        return 0.5

    total_score = 0.0
    total_sections = len(sections)

    for section in sections:
        # 基于修改数量和字数计算质量评分
        word_count = max(section.word_count, 1)
        modification_ratio = len(section.modifications) / word_count

        # 合理的修改比例
        if modification_ratio < 0.1:  # 修改过少
            base_score = 0.7
        elif modification_ratio > 0.3:  # 修改过多
            base_score = 0.6
        else:  # 合理修改
            base_score = 0.85

        total_score += base_score

    return min(total_score / total_sections, 1.0)


def generate_modification_summary(sections: list[PolishSection]) -> str:
    """
    生成修改总结
    """
    total_modifications = sum(len(section.modifications) for section in sections)
    total_words = sum(section.word_count for section in sections)

    if total_modifications == 0:
        return "本次润色无需任何修改，文档质量良好。"

    avg_words_per_modification = total_words / total_modifications if total_modifications > 0 else 0

    summary_parts = [
        f"共对 {len(sections)} 个章节进行了润色",
        f"提出了 {total_modifications} 处修改建议",
        f"平均每 {avg_words_per_modification:.1f} 字提出 1 处修改",
    ]

    # 按章节汇总修改情况
    chapter_summaries = []
    for section in sections:
        if section.modifications:
            chapter_summaries.append(f"{section.title}: {len(section.modifications)}处修改")

    if chapter_summaries:
        summary_parts.append("详细修改情况: " + "; ".join(chapter_summaries))

    return " | ".join(summary_parts)


def assemble_final_content(
    sections: list[PolishSection],
    citation_manager: CitationManager | None = None,
    citation_style: str = "numeric",
    document_title: str | None = None,
) -> str:
    """
    重新组装最终文档内容，集成引用标注和脚注生成

    Args:
        sections: 章节列表
        citation_manager: 引用管理器实例
        citation_style: 引用格式 ("numeric", "symbol", "bracket", "parenthetical")

    Returns:
        包含引用标注的完整文档
    """
    content_parts: list[str] = []

    # Optional document title (level-1 heading)
    if document_title:
        content_parts.append(f"# {document_title}\n\n")

    order: list[str] = []
    section_map: dict[str, tuple[PolishSection, str]] = {}

    for index, section in enumerate(sections):
        raw_title = section.title or ""
        # Remove existing HTML comments (e.g., legacy section_id markers)
        title_clean = re.sub(r"\s*<!--.*?-->\s*", "", raw_title).strip()
        title_clean = re.sub(r"^#+\s*", "", title_clean).strip()
        content_clean = (section.content or "").strip()

        if not title_clean or not content_clean:
            continue

        key = title_clean.lower()
        existing = section_map.get(key)
        if existing:
            existing_section, existing_title = existing
            if len(content_clean) > len(existing_section.content or ""):
                section_map[key] = (section, title_clean)
        else:
            section_map[key] = (section, title_clean)
            order.append(key)

    if not order:
        return ""

    def _priority(title: str, default_index: int) -> tuple[int, int]:
        lowered = title.lower()
        if any(token in lowered for token in ("引言", "背景", "简介")):
            return (0, default_index)
        if any(token in lowered for token in ("理论", "模型", "方法", "框架")):
            return (10, default_index)
        if any(token in lowered for token in ("实验", "实施", "方案", "方法学")):
            return (20, default_index)
        if any(token in lowered for token in ("结果", "分析", "讨论", "发现")):
            return (30, default_index)
        if any(token in lowered for token in ("结论", "展望", "总结", "未来")):
            return (40, default_index)
        if any(token in lowered for token in ("符号", "参数", "附录")):
            return (50, default_index)
        if any(token in lowered for token in ("参考", "文献", "致谢")):
            return (60, default_index)
        return (99, default_index)

    ordered_keys = sorted(
        order,
        key=lambda key: _priority(section_map[key][1], order.index(key)),
    )

    seen_paragraphs: set[str] = set()

    for key in ordered_keys:
        section, cleaned_title = section_map[key]
        # 添加标题（若存在 section_id，注入注释以便后续结构管理）
        heading = f"## {cleaned_title}"
        if getattr(section, "section_id", None):
            heading = f"{heading} <!-- section_id: {section.section_id} -->"
        content_parts.append(heading)

        # 添加内容
        if section.content:
            paragraphs = re.split(r"\n\s*\n", section.content.strip())
            filtered: list[str] = []
            for para in paragraphs:
                normalized = re.sub(r"\s+", " ", para.strip()).lower()
                if len(normalized) >= 60:
                    if normalized in seen_paragraphs:
                        logging.info(
                            "检测到重复段落，已跳过: %s...",
                            para.strip()[:40],
                        )
                        continue
                    seen_paragraphs.add(normalized)
                filtered.append(para.strip())

            if filtered:
                content_parts.append("\n" + "\n\n".join(filtered))

        # 添加分段符（除了最后一个章节）
        content_parts.append("\n")

    base_content = "".join(content_parts).strip()

    base_content = _drop_duplicate_intro_and_conclusion(base_content)

    placeholder_hits = _detect_unresolved_placeholders(base_content)
    if placeholder_hits:
        logging.warning(
            "最终文档存在未解决占位符，命中: %s。将尝试清理这些占位符。",
            ", ".join(sorted(placeholder_hits)),
        )
        # 修复：清理未解决的占位符，而不是崩溃
        base_content = _remove_unresolved_placeholders(base_content, placeholder_hits)
        removed_count = len(placeholder_hits)
        logging.info(f"已清理 {removed_count} 个未解决的占位符")
        # 在文档末尾添加说明（可选，避免过度干扰）
        # 如果需要更明显的提示，可以取消下面的注释
        # if removed_count > 0:
        #     cleanup_note = f"\n\n> **注意**：已移除 {removed_count} 个未解决的引用占位符。"
        #     base_content = base_content + cleanup_note

    # 如果没有引用管理器，返回基础内容
    if not citation_manager or not citation_manager.citations:
        return _ensure_reference_section(base_content)

    # 生成引用标注和脚注
    try:
        annotated_content, references_section = render_citations_with_footnotes(base_content, citation_manager.citations, citation_style)

        # 合并内容
        final_content = f"{annotated_content}\n\n{references_section}"
        return final_content

    except Exception as e:
        logging.warning(f"引用标注生成失败: {e}，返回基础内容")
        return _ensure_reference_section(base_content)


PLACEHOLDER_PATTERNS = [
    r"\[ref:[^\]]+\]",
    r"待补[充]?",
    r"参考文献\s*\[[^\]]*待补[^\]]*\]",
    r"references?\s*\[\d+\]\s*(todo|待补)",
    r"\btodo\b",
]

INTRO_TOKENS = ("引言", "绪论", "简介", "背景", "overview", "introduction")
CONCLUSION_TOKENS = ("结论", "总结", "展望", "结语", "conclusion", "outlook", "closing")


def _detect_unresolved_placeholders(content: str) -> set[str]:
    hits: set[str] = set()
    lowered = content.lower()
    for pattern in PLACEHOLDER_PATTERNS:
        for match in re.finditer(pattern, lowered, flags=re.IGNORECASE):
            snippet = match.group(0)
            hits.add(snippet)
    return hits


def _remove_unresolved_placeholders(content: str, placeholders: set[str]) -> str:
    """
    移除未解决的占位符

    Args:
        content: 原始内容
        placeholders: 未解决的占位符集合（小写）

    Returns:
        清理后的内容
    """
    cleaned = content
    original_lower = content.lower()

    # 找到所有占位符在原文中的实际位置（大小写可能不同）
    placeholder_positions = []
    for placeholder_lower in placeholders:
        # 在原文中查找（不区分大小写）
        pattern = re.escape(placeholder_lower)
        for match in re.finditer(pattern, original_lower, flags=re.IGNORECASE):
            # 获取原文中的实际匹配
            start, end = match.span()
            actual_placeholder = content[start:end]
            placeholder_positions.append((start, end, actual_placeholder))

    # 按位置从后往前删除，避免位置偏移
    placeholder_positions.sort(key=lambda x: x[0], reverse=True)
    for start, end, actual_placeholder in placeholder_positions:
        cleaned = cleaned[:start] + cleaned[end:]

    # 清理多余的空格和换行
    cleaned = re.sub(r"\n\s*\n\s*\n+", "\n\n", cleaned)  # 多个空行合并为两个
    cleaned = re.sub(r" +", " ", cleaned)  # 多个空格合并为一个

    return cleaned.strip()


def _drop_duplicate_intro_and_conclusion(markdown: str) -> str:
    """
    移除重复的引言/结论章节，保留信息量更高的一份。
    """
    heading_re = re.compile(
        r"^##\s+(.*?)(?:\s<!--\s*section_id:\s*([A-Za-z0-9_-]+)\s*-->)?\s*$",
        re.MULTILINE,
    )

    matches = list(heading_re.finditer(markdown))
    if not matches:
        return markdown

    sections: list[dict[str, Any]] = []
    for idx, match in enumerate(matches):
        start = match.start()
        header_end = match.end()
        heading = match.group(1).strip()
        next_start = matches[idx + 1].start() if idx + 1 < len(matches) else len(markdown)
        body = markdown[header_end:next_start]
        sections.append({
            "index": idx,
            "start": start,
            "end": next_start,
            "heading": heading,
            "body": body,
        })

    def _classify(title: str) -> str | None:
        lowered = title.lower()
        if any(token in lowered for token in INTRO_TOKENS):
            return "intro"
        if any(token in lowered for token in CONCLUSION_TOKENS):
            return "conclusion"
        return None

    best_by_category: dict[str, tuple[int, int]] = {}
    to_drop: set[int] = set()

    for section in sections:
        category = _classify(section["heading"])
        if not category:
            continue
        body_length = len(section["body"].strip())
        if category not in best_by_category:
            best_by_category[category] = (section["index"], body_length)
            continue
        best_idx, best_len = best_by_category[category]
        if body_length > best_len:
            to_drop.add(best_idx)
            best_by_category[category] = (section["index"], body_length)
        else:
            to_drop.add(section["index"])

    if not to_drop:
        return markdown

    result_parts: list[str] = []
    last_pos = 0
    for section in sections:
        if section["index"] in to_drop:
            logging.info("去除重复章节《%s》", section["heading"])
            result_parts.append(markdown[last_pos : section["start"]])
            last_pos = section["end"]
        else:
            result_parts.append(markdown[last_pos : section["end"]])
            last_pos = section["end"]

    result_parts.append(markdown[last_pos:])
    return "".join(result_parts).strip()


def render_citations_with_footnotes(content: str, citations: list[CitationMatch], style: str = "numeric") -> tuple[str, str]:
    """
    根据引用风格生成带标注的文本和脚注列表

    Args:
        content: 原始内容
        citations: 引用匹配列表
        style: 引用风格

    Returns:
        带标注的文本和脚注列表
    """
    if not citations:
        return content, ""

    # 构建引用映射
    citation_map = {}
    footnote_list = []
    citation_counter = 1

    for match in citations:
        if match.sources:
            # 根据风格生成引用标记
            citation_marker = generate_citation_marker(citation_counter, style)
            citation_map[citation_marker] = match

            # 生成脚注内容
            footnote_content = format_footnote_content(match, citation_counter, style)
            if footnote_content not in footnote_list:
                footnote_list.append(footnote_content)
                citation_counter += 1

    # 在文本中插入引用标记
    annotated_text = insert_citation_markers(content, citation_map, style)

    # 生成脚注列表
    footnotes_section = generate_footnotes_section(footnote_list, style)

    return annotated_text, footnotes_section


def _ensure_reference_section(content: str) -> str:
    """确保最终文档包含参考文献占位符，避免空白引用段。"""
    if re.search(r"^##\s*参考文献", content, flags=re.MULTILINE):
        return content
    placeholder = "\n\n## 参考文献\n\n（本次迭代未生成参考文献，请补充可信来源。）"
    return content + placeholder


def generate_citation_marker(number: int, style: str) -> str:
    """根据风格生成引用标记"""
    style_formatters = {
        "numeric": f"[{number}]",
        "symbol": f"[†{number}]",
        "bracket": f"[({number})]",
        "parenthetical": f"({number})",
    }
    return style_formatters.get(style, f"[{number}]")


def format_footnote_content(match: CitationMatch, number: int, style: str) -> str:
    """格式化脚注内容"""
    if not match.sources:
        return f"[{number}] 需要进一步验证的内容"

    source_texts = []
    for source in match.sources[:2]:  # 最多显示2个来源
        source_info = f"{source.title}"
        if source.url:
            source_info += f" - {source.url}"
        if source.date:
            source_info += f" ({source.date})"
        source_texts.append(source_info)

    # 添加验证状态
    verification_status = "✓ 已验证" if not match.requires_verification else "⚠ 需要验证"

    return f"[{number}] {'; '.join(source_texts)} - {verification_status}"


def insert_citation_markers(content: str, citation_map: dict, style: str) -> str:
    """在文本中插入引用标记"""
    # 按匹配项的文本位置排序
    sorted_matches = sorted(
        citation_map.items(),
        key=lambda x: content.find(x[1].claim.text) if x[1].claim.text in content else -1,
    )

    # 从后往前插入，避免位置偏移
    for citation_marker, match in reversed(sorted_matches):
        if match.claim.text in content:
            # 在句末插入引用标记
            end_pos = content.find(match.claim.text) + len(match.claim.text)
            if end_pos <= len(content):
                # 查找句子结束位置
                sentence_end = find_sentence_end(content, end_pos)
                content = content[:sentence_end] + f" {citation_marker}" + content[sentence_end:]

    return content


def find_sentence_end(text: str, start_pos: int) -> int:
    """查找句子的结束位置"""
    end_chars = [".", "。", "!", "！", "?", "？"]
    for i in range(start_pos, min(len(text), start_pos + 100)):
        if text[i] in end_chars:
            return i + 1
    return start_pos


def generate_footnotes_section(footnote_list: list[str], style: str) -> str:
    """生成脚注列表部分"""
    if not footnote_list:
        return ""

    section_title = {
        "numeric": "\n\n## 参考文献",
        "symbol": "\n\n## 符号脚注",
        "bracket": "\n\n## 参考资料",
        "parenthetical": "\n\n## 参考资料",
    }

    title = section_title.get(style, "\n\n## 参考资料")
    footnotes = "\n\n".join(f"{i + 1}. {footnote}" for i, footnote in enumerate(footnote_list))

    # 添加统计信息
    verified_count = sum(1 for footnote in footnote_list if "✓" in footnote)
    total_count = len(footnote_list)
    coverage = (verified_count / total_count * 100.0) if total_count > 0 else 0.0
    stats = f"\n\n*注：已验证来源 {verified_count}/{total_count}，引用覆盖率 {coverage:.1f}%*"

    return f"{title}\n\n{footnotes}{stats}"


def initialize_citation_manager(
    research_data: dict[str, Any] | None = None,
) -> CitationManager:
    """
    初始化引用管理器

    Args:
        research_data: 研究数据

    Returns:
        配置好的引用管理器
    """
    citation_manager = CitationManager()

    if research_data:
        # 从研究数据中提取源信息
        for item in research_data.get("sources", []):
            if isinstance(item, dict):
                source = SourceInfo(
                    id=item.get("id", ""),
                    url=item.get("url", ""),
                    title=item.get("title", ""),
                    date=item.get("date", ""),
                    content=item.get("content", ""),
                    summary=item.get("summary", ""),
                    confidence=item.get("confidence", 0.5),
                )
                citation_manager.add_source(source)

    return citation_manager


def integrate_fact_checking(
    polished_result: PolishModel,
    structured_research_data: dict[str, Any] | None = None,
    config: Any | None = None,
) -> list[FactCheckResult]:
    """
    集成事实核查功能到润色流程中

    Args:
        polished_result: 润色结果
        structured_research_data: 结构化研究数据
        config: 配置对象

    Returns:
        List[FactCheckResult]: 事实核查结果列表
    """
    logging.info("开始集成事实核查功能...")

    # 初始化事实核查器
    fact_checker = FactChecker()

    # 收集所有事实核查点
    all_fact_check_points = []
    for section in polished_result.sections:
        # 兼容旧的 PolishSection（无 fact_check_points 字段）
        _points = getattr(section, "fact_check_points", None)
        if _points:
            all_fact_check_points.extend(_points)

    if not all_fact_check_points:
        logging.info("未发现需要核查的事实点")
        return []

    logging.info(f"发现 {len(all_fact_check_points)} 个需要核查的事实点")

    # 准备数据源
    sources = prepare_fact_check_sources(structured_research_data)

    try:
        # 执行事实核查
        fact_check_results = fact_checker.check_minimal(all_fact_check_points, sources)

        # 生成核查报告
        if fact_check_results:
            verification_report = fact_checker.generate_unverifiable_report(fact_check_results)
            logging.info(f"事实核查完成 - 验证率: {verification_report['verification_rate']:.2%}")

            # 为需要修改的内容生成建议
            modification_suggestions = generate_fact_check_modification_suggestions(fact_check_results, polished_result.sections)

            # 如果有修改建议，添加到元数据中
            if modification_suggestions:
                polished_result.metadata["fact_check_modifications"] = modification_suggestions
                logging.info(f"生成了 {len(modification_suggestions)} 条事实核查修改建议")

        return fact_check_results

    except Exception as e:
        logging.error(f"事实核查过程发生错误: {e}", exc_info=True)
        return []


def prepare_fact_check_sources(
    structured_research_data: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """
    准备事实核查的数据源

    Args:
        structured_research_data: 结构化研究数据

    Returns:
        List[Dict]: 格式化后的数据源列表
    """
    sources: list[dict[str, Any]] = []

    if not structured_research_data:
        return sources

    # 提取研究数据中的各个部分作为数据源
    for key, value in structured_research_data.items():
        if isinstance(value, dict):
            # 学术论文格式
            if "title" in value and "abstract" in value:
                sources.append({
                    "title": value.get("title", ""),
                    "content": value.get("abstract", "") + " " + value.get("content", ""),
                    "url": value.get("url", ""),
                    "type": "academic_paper",
                })
            # 报告格式
            elif "summary" in value:
                sources.append({
                    "title": value.get("title", key),
                    "content": value.get("summary", ""),
                    "url": value.get("url", ""),
                    "type": "report",
                })
        elif isinstance(value, list):
            # 列表格式的数据
            for item in value:
                if isinstance(item, dict) and "content" in item:
                    sources.append({
                        "title": item.get("title", f"{key}_item"),
                        "content": item.get("content", ""),
                        "url": item.get("url", ""),
                        "type": "reference",
                    })
        elif isinstance(value, str):
            # 简单文本内容
            sources.append({"title": key, "content": value, "url": "", "type": "text"})

    logging.info(f"准备了 {len(sources)} 个数据源用于事实核查")
    return sources


def generate_fact_check_modification_suggestions(fact_check_results: list[FactCheckResult], sections: list[PolishSection]) -> list[dict]:
    """
    基于事实核查结果生成修改建议

    Args:
        fact_check_results: 事实核查结果
        sections: 润色章节列表

    Returns:
        List[Dict]: 修改建议列表
    """
    suggestions = []

    for result in fact_check_results:
        if result.confidence_level in ["low", "unverifiable"]:
            # 查找包含该主张的章节
            target_section = None
            for section in sections:
                if result.claim in section.content:
                    target_section = section
                    break

            if target_section:
                suggestion = {
                    "section_id": target_section.section_id,
                    "section_title": target_section.title,
                    "claim": result.claim,
                    "issue_type": "verification_concern",
                    "confidence_level": result.confidence_level,
                    "verification_score": result.verification_score,
                    "recommendation": generate_single_claim_suggestion(result),
                    "supporting_sources": result.supporting_sources,
                    "contradicting_sources": result.contradicting_sources,
                    "notes": result.notes,
                }
                suggestions.append(suggestion)

    return suggestions


def generate_single_claim_suggestion(result: FactCheckResult) -> str:
    """
    为单个主张生成修改建议

    Args:
        result: 事实核查结果

    Returns:
        str: 修改建议文本
    """
    if result.confidence_level == "unverifiable":
        return f"建议删除或修改该主张：'{result.claim[:50]}...'，因为无法验证其准确性。"
    elif result.confidence_level == "low":
        return f"建议为该主张补充权威证据：'{result.claim[:50]}...'，当前验证分数为 {result.verification_score:.2f}。"
    else:
        return f"该主张基本可信：'{result.claim[:50]}...'，验证分数为 {result.verification_score:.2f}。"


def enhance_content_with_verification_markers(content: str, fact_check_results: list[FactCheckResult]) -> str:
    """
    在内容中添加验证标记

    Args:
        content: 原始内容
        fact_check_results: 事实核查结果

    Returns:
        str: 带有验证标记的内容
    """
    if not fact_check_results:
        return content

    # 使用utils.factcheck模块中的函数添加标记
    return add_verification_markers(content, fact_check_results)


__all__ = ["polish_node"]
