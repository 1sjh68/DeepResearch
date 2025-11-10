from __future__ import annotations

import json
import logging
import uuid
from typing import Any

from core.progress import StepOutput, step_result, workflow_step
from core.state_manager import WorkflowStateAdapter
from planning.outline import allocate_content_lengths
from planning.tool_definitions import PlanModel
from services.llm_interaction import call_ai
from utils.progress_tracker import safe_pulse
from workflows.graph_state import GraphState


@workflow_step("plan_node", "生成文档大纲")
def plan_node(state: GraphState) -> StepOutput:
    workflow_state = WorkflowStateAdapter.ensure(state)
    config = workflow_state.config
    safe_pulse(config.task_id, "生成文档大纲中...")

    # 使用结构化输出生成大纲
    raw_outline = generate_structured_plan_outline(config, config.user_problem)
    if not raw_outline:
        raise RuntimeError("未能从AI生成有效的结构化文档大纲。")

    outline_data = allocate_content_lengths(config, raw_outline, config.initial_solution_target_chars)

    if not outline_data or not outline_data.get("outline"):
        raise RuntimeError("分配内容长度后，未能生成有效的文档大纲。")

    return step_result({"outline": outline_data}, "生成大纲完成")


def generate_structured_plan_outline(config, problem_statement: str) -> dict[str, Any] | None:
    """
    使用结构化输出生成文档大纲，支持回退机制。
    优先使用 PlanModel 进行结构化输出，失败时回退到字符串解析。
    """
    logging.info(f"\n--- 使用结构化输出生成文档大纲: {problem_statement[:100]}... ---")

    # 构建提示词
    system_prompt = """你是一位专业的结构分析师和文档规划专家。你的任务是根据用户的需求创建一个全面、结构化的文档大纲。
请确保输出包含：
- 清晰的主标题
- 逻辑合理的章节结构
- 每个章节都有详细的描述
- 合适的字数分配比例
- 文档的核心目标

请以结构化的方式组织你的响应。"""

    user_prompt = f"""请为以下用户需求生成一个详细的文档大纲：

用户需求："{problem_statement}"

目标文档长度：约 {config.initial_solution_target_chars} 字符

请生成一个结构化的文档规划，包含主标题、章节列表和核心目标。"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    try:
        # 首先尝试结构化输出
        logging.info("尝试使用结构化输出生成大纲...")
        plan_result = call_ai(
            config=config,
            model_name=config.outline_model_name,
            messages=messages,
            schema=PlanModel,
            temperature=0.1,
            max_tokens_output=4096,
        )

        # 检查是否成功返回 PlanModel 实例
        if isinstance(plan_result, PlanModel):
            logging.info("结构化输出成功，转换为标准格式")

            # 提取关键信息
            outline_dict = plan_result.model_dump()

            # 验证必需字段
            if not outline_dict.get("outline") or not outline_dict.get("title"):
                logging.warning("结构化输出缺少必需字段，尝试回退到字符串解析")
                raise ValueError("缺少必需字段")

            # 为大纲章节添加唯一ID（与现有系统兼容）
            _add_ids_to_outline(outline_dict.get("outline", []))

            # 记录成功生成的结构化信息
            if config.session_dir:
                path = f"{config.session_dir}/structured_plan_outline.json"
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(outline_dict, f, ensure_ascii=False, indent=4)

            logging.info(f"结构化大纲生成成功：主标题='{outline_dict['title']}', {len(outline_dict['outline'])} 个主要章节")
            return outline_dict

        else:
            # 回退到字符串解析模式
            logging.info("结构化输出失败或返回非 PlanModel 类型，使用字符串解析")
            return parse_outline_from_string(plan_result, problem_statement, config)

    except Exception as e:
        logging.warning(f"结构化输出异常: {e}，尝试回退到字符串解析")
        # 回退到字符串解析模式
        try:
            # 重新调用但不指定 schema
            plain_result = call_ai(
                config=config,
                model_name=config.outline_model_name,
                messages=messages,
                temperature=0.1,
                max_tokens_output=4096,
                response_format={"type": "json_object"},
            )
            return parse_outline_from_string(plain_result, problem_statement, config)
        except Exception as fallback_error:
            logging.error(f"回退机制也失败: {fallback_error}")
            return None


def parse_outline_from_string(text_content: str, problem_statement: str, config) -> dict[str, Any] | None:
    """
    从字符串内容中解析出标准格式的大纲数据。
    这是结构化输出失败时的回退机制。
    """
    logging.info("使用字符串解析模式生成大纲...")

    if not text_content or "AI模型调用失败" in text_content:
        logging.error("回退调用也失败了")
        return None

    try:
        # 尝试解析JSON格式
        if text_content.strip().startswith("{") and text_content.strip().endswith("}"):
            try:
                data = json.loads(text_content)
                if isinstance(data, dict) and "outline" in data and "title" in data:
                    logging.info("从JSON字符串成功解析大纲")
                    return data
            except json.JSONDecodeError as e:
                logging.warning(f"JSON parsing failed during fallback: {e}. Content was: '{text_content[:200]}...'")
                pass

        # 如果不是有效JSON，创建基础结构
        logging.info("生成基础大纲结构")

        # 提取章节标题（简单的文本解析）
        lines = text_content.split("\n")
        sections = []
        current_section = None

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # 检测标题行（简单的启发式方法）
            if (
                line.startswith("#")
                or any(
                    keyword in line.lower()
                    for keyword in [
                        "章节",
                        "section",
                        "部分",
                        "段落",
                        "一、",
                        "二、",
                        "三、",
                    ]
                )
                or (len(line) < 100 and not line.endswith("。") and not line.endswith("，"))
            ):
                if current_section:
                    sections.append(current_section)

                current_section = {
                    "title": line.replace("#", "").strip(),
                    "description": "",
                    "target_chars_ratio": 1.0 / 6,  # 默认平均分配
                    "sections": [],
                }

        if current_section:
            sections.append(current_section)

        # 如果没有检测到明确的章节，创建默认结构
        if not sections:
            sections = [
                {
                    "title": "引言",
                    "description": f"介绍{problem_statement[:50]}...的背景和重要性",
                    "target_chars_ratio": 0.15,
                    "sections": [],
                },
                {
                    "title": "主要内容",
                    "description": f"深入分析{problem_statement[:50]}...的核心问题",
                    "target_chars_ratio": 0.60,
                    "sections": [],
                },
                {
                    "title": "结论与建议",
                    "description": f"总结{problem_statement[:50]}...的关键发现和未来方向",
                    "target_chars_ratio": 0.25,
                    "sections": [],
                },
            ]

        # 归一化比例
        total_ratio = sum(section["target_chars_ratio"] for section in sections)
        if total_ratio > 0:
            for section in sections:
                section["target_chars_ratio"] /= total_ratio

        # 创建标准格式的大纲数据
        outline_data = {
            "title": f"{problem_statement}分析报告",
            "outline": sections,
            "total_estimated_chars": config.initial_solution_target_chars,
            "target_audience": "专业读者",
            "key_objectives": [
                f"分析{problem_statement[:30]}...",
                "提供深入见解",
                "给出实用建议",
            ],
        }

        # 为大纲章节添加唯一ID（确保兼容性）
        _add_ids_to_outline(outline_data.get("outline", []))

        logging.info(f"字符串解析成功：生成 {len(sections)} 个章节的基础大纲")
        return outline_data

    except Exception as e:
        logging.error(f"字符串解析失败: {e}")
        return None


def _add_ids_to_outline(chapters: list):
    """
    递归地为每个章节和子章节添加一个唯一的UUID。
    确保与现有系统兼容。
    """
    for chapter in chapters:
        if isinstance(chapter, dict) and "id" not in chapter:
            chapter["id"] = str(uuid.uuid4())
        if isinstance(chapter, dict) and "sections" in chapter and chapter["sections"]:
            _add_ids_to_outline(chapter["sections"])


__all__ = ["plan_node"]
