from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

from pydantic import ValidationError

from core.progress import StepOutput, step_result, workflow_step
from core.state_manager import WorkflowStateAdapter
from planning.tool_definitions import CritiqueModel
from services.llm_interaction import call_ai
from utils.iteration_storage import archive_iteration_snapshot
from utils.json_repair import repair_json_once
from utils.progress_tracker import safe_pulse
from utils.text_processor import (
    extract_json_from_ai_response,
    extract_knowledge_gaps,
    truncate_text_for_context_boundary_aware,
)
from workflows.graph_state import GraphState
from workflows.prompts import CRITIQUE_SYSTEM_PROMPT


# 修复 1: 添加异常捕获的安全 AI 调用包装
def _safe_call_ai_with_fallback(config, model, messages, schema, max_tokens):
    """安全调用 AI，自动降级处理异常"""
    try:
        return call_ai(
            config, model, messages,
            temperature=getattr(config, "temperature_factual", 0.3),
            max_tokens_output=max_tokens,
            schema=schema,
        )
    except TimeoutError as e:
        logging.warning("AI call timeout, using fallback response: %s", e)
        # 返回默认的空响应对象
        return schema(critique="(AI 超时，已使用默认回复)", knowledge_gaps=[])
    except ValidationError as e:
        logging.warning("Schema validation failed, retrying without schema: %s", e)
        try:
            raw_response = call_ai(
                config, model, messages,
                temperature=getattr(config, "temperature_factual", 0.3),
                max_tokens_output=max_tokens,
            )
            return schema(critique=raw_response[:1000], knowledge_gaps=[])
        except Exception as e2:
            logging.error("Fallback AI call also failed: %s", e2)
            return schema(critique="", knowledge_gaps=[])
    except Exception as e:
        logging.error("Unexpected error in AI call (%s): %s", type(e).__name__, e, exc_info=True)
        # 最后的保险
        return schema(critique="", knowledge_gaps=[])


def _format_knowledge_gaps(knowledge_gaps: Sequence[str]) -> str:
    if not knowledge_gaps:
        return ""
    lines = "\n".join(f"- {gap}" for gap in knowledge_gaps)
    return f"## 知识空白 ({len(knowledge_gaps)} 条)\n\n{lines}\n\n"


def _build_snapshot_body(
    structured_feedback: object,
    critique_text: str | None,
    knowledge_gaps: Sequence[str] | None = None,
    *,
    title: str = "Reviewer Critique",
) -> str:
    critic_section = (critique_text or "").strip()
    knowledge_section = _format_knowledge_gaps(knowledge_gaps or [])
    if isinstance(structured_feedback, CritiqueModel):
        return (
            f"# {title}\n\n"
            "## 评审意见\n\n"
            f"{critic_section}\n\n"
            f"{knowledge_section}"
            "## 结构化数据\n\n"
            "````json\n"
            f"{structured_feedback.model_dump_json(indent=2)}\n"
            "````\n"
        )
    raw_response = str(structured_feedback) if structured_feedback else "None"
    return (
        f"# {title}\n\n"
        "## 评审意见\n\n"
        f"{critic_section}\n\n"
        f"{knowledge_section}"
        "## 原始响应\n\n"
        f"{raw_response}"
    )


@workflow_step("critique_node", "评审草稿")
def critique_node(state: GraphState) -> StepOutput:
    workflow_state = WorkflowStateAdapter.ensure(state)
    config = workflow_state.config

    current_iteration = workflow_state.refinement_count + 1
    max_rounds = config.max_refinement_rounds
    logging.info(
        "[RefineLoop] Iteration %s/%s -> critique_node [NODE ENTRY - refinement_count=%s]",
        current_iteration,
        max_rounds,
        workflow_state.refinement_count,
    )
    logging.info(
        "[CRITIQUE_DEBUG] Iteration %s/%s - preparing critique payload",
        current_iteration,
        max_rounds,
    )
    safe_pulse(config.task_id, f"迭代 {current_iteration}/{max_rounds} · 评审草稿中...")

    draft_content = workflow_state.draft_content
    if not draft_content:
        logging.warning("critique_node: No draft_content found to critique. Skipping.")
        return step_result(
            {"critique": "", "knowledge_gaps": []},
            "无草稿可评审",
        )

    solution_for_critic = truncate_text_for_context_boundary_aware(
        config,
        draft_content,
        int(config.max_context_for_long_text_review_tokens * getattr(config, "prompt_budget_ratio", 0.9)),
    )

    critic_prompt = f"Original problem:\n---\n{config.user_problem}\n---\nSolution to be reviewed:\n---\n{solution_for_critic}\n---\nPlease provide your review:"

    logging.info(
        "[CRITIQUE_DEBUG] Iteration %s/%s - calling critique model %s",
        current_iteration,
        max_rounds,
        config.secondary_ai_model,
    )

    # 使用结构化输出调用 - 使用安全包装处理异常
    critique_token_budget = getattr(config, "critique_max_tokens", 4096)
    structured_feedback = _safe_call_ai_with_fallback(
        config,
        config.secondary_ai_model,
        [
            {"role": "system", "content": CRITIQUE_SYSTEM_PROMPT},
            {"role": "user", "content": critic_prompt},
        ],
        CritiqueModel,
        critique_token_budget,
    )
    structured_payload: dict[str, Any] | None = None
    knowledge_gaps: list[str] = []
    critique_text = ""
    critique_model: CritiqueModel | None = None

    # 处理结构化输出
    if isinstance(structured_feedback, CritiqueModel):
        critique_model = structured_feedback
    elif isinstance(structured_feedback, str) and structured_feedback.strip():
        logging.warning(
            "critique_node: 结构化解析回退为原始字符串，尝试手动解析 JSON。"
        )
        repaired_json: str | None = None
        try:
            debug_mode = getattr(getattr(config, "workflow", None), "debug_json_repair", False)
            repaired_text, repaired = repair_json_once(structured_feedback.strip(), CritiqueModel, debug=debug_mode)
            logging.debug(
                "critique_node: repair_json_once result repaired=%s (len=%s)",
                repaired,
                len(repaired_text or ""),
            )
            if repaired:
                repaired_json = repaired_text
        except Exception as exc:  # pragma: no cover - 防御性日志
            logging.debug("critique_node: repair_json_once 处理失败: %s", exc, exc_info=True)

        if not repaired_json:
            logging.debug("critique_node: repair_json_once 未奏效，调用 extract_json_from_ai_response。")
            repaired_json = extract_json_from_ai_response(
                config,
                structured_feedback,
                context_for_error_log="Critique fallback response",
            )

        if repaired_json:
            try:
                structured_feedback_obj = CritiqueModel.model_validate_json(repaired_json)
                critique_model = structured_feedback_obj
                logging.info("critique_node: 手动解析结构化评审成功。")
            except ValidationError as exc:
                logging.warning("critique_node: 手动解析结构化评审失败: %s", exc)

    if critique_model:
        structured_feedback = critique_model
        critique_text = critique_model.critique or ""
        knowledge_gaps = list(critique_model.knowledge_gaps or [])
        structured_payload = critique_model.model_dump()
        try:
            if critique_model.rubric:
                rubric_data = critique_model.rubric.dict()
                logging.info(
                    "Critique rubric: %s | contradictions: %s",
                    {
                        k: rubric_data.get(k)
                        for k in [
                            "coverage",
                            "correctness",
                            "verifiability",
                            "coherence",
                            "style_fit",
                            "math_symbol_correctness",
                            "chapter_balance",
                        ]
                    },
                    len(critique_model.contradictions or []),
                )
        except Exception as exc:
            if isinstance(exc, (SystemExit, KeyboardInterrupt)):
                raise
            logging.warning("记录评审打分时出错: %s", exc, exc_info=True)
    else:
        # 回退到原始逻辑（字符串输出）
        critique_text = structured_feedback if isinstance(structured_feedback, str) else str(structured_feedback or "")
        logging.warning("结构化调用失败，回退到字符串解析")
        try:
            # 尝试从字符串中提取知识空白
            knowledge_gaps = extract_knowledge_gaps(critique_text)
        except Exception as exc:
            if isinstance(exc, (SystemExit, KeyboardInterrupt)):
                raise
            logging.warning("提取知识空白时出错: %s", exc, exc_info=True)
            knowledge_gaps = []
        structured_payload = None

    if not knowledge_gaps:
        fallback_source = critique_text or (structured_feedback if isinstance(structured_feedback, str) else str(structured_feedback or ""))
        if fallback_source:
            try:
                extracted_gaps = extract_knowledge_gaps(fallback_source)
            except Exception as exc:
                if isinstance(exc, (SystemExit, KeyboardInterrupt)):
                    raise
                logging.debug("Fallback knowledge gap extraction failed: %s", exc, exc_info=True)
            else:
                if extracted_gaps:
                    knowledge_gaps = extracted_gaps

    if not critique_text:
        critique_text = structured_feedback if isinstance(structured_feedback, str) else str(structured_feedback or "")

    logging.info(
        "[CRITIQUE_DEBUG] Iteration %s/%s - processed critique feedback (%s chars, %s gaps)",
        current_iteration,
        max_rounds,
        len(critique_text) if critique_text else 0,
        len(knowledge_gaps),
    )

    if isinstance(structured_feedback, CritiqueModel):
        try:
            structured_snapshot = _build_snapshot_body(
                structured_feedback,
                critique_text,
                knowledge_gaps,
                title="结构化评审结果",
            )
            archive_iteration_snapshot(
                config,
                current_iteration,
                "critique_structured",
                structured_snapshot,
            )
        except Exception as exc:
            if isinstance(exc, (SystemExit, KeyboardInterrupt)):
                raise
            logging.warning("归档结构化评审结果时出错: %s", exc, exc_info=True)

    detail_msg = f"迭代 {current_iteration}/{max_rounds}，知识空白 {len(knowledge_gaps)} 条"
    try:
        snapshot_body = _build_snapshot_body(structured_feedback, critique_text, knowledge_gaps)
        archive_iteration_snapshot(
            config,
            current_iteration,
            "critique",
            snapshot_body,
        )
        preview = (critique_text or "").strip()
        if preview:
            logging.info(
                "[Reviewer] 迭代 %s/%s 评审摘要: %s...",
                current_iteration,
                max_rounds,
                preview[:300],
            )
    except Exception as exc:
        if isinstance(exc, (SystemExit, KeyboardInterrupt)):
            raise
        logging.debug(
            "无法归档评审快照（iteration=%s）: %s",
            current_iteration,
            exc,
            exc_info=True,
        )
    return step_result(
        {
            "critique": critique_text,
            "knowledge_gaps": knowledge_gaps,
            "structured_critique": structured_payload,
        },
        detail_msg,
    )


__all__ = ["critique_node"]
