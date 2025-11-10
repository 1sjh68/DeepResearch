# workflows/sub_workflows/planning.py

import logging

from config import Config
from services.llm_interaction import call_ai_writing_with_auto_continue  # Use sync call_ai


def generate_style_guide(config: Config) -> str:
    """
    根据用户的核心问题生成风格与声音指南 (同步版本)。
    """
    logging.info("\n--- 正在生成风格与声音指南 ---")
    prompt = f"""
    你是一位经验丰富的总编辑。请根据用户的核心问题，为即将撰写的深度报告制定一份简明扼要的《风格与声音指南》。
    # 核心问题:
    "{config.user_problem}"
    # 你的任务是定义以下几点:
    1.  **核心论点 (Core Thesis)**: 用一句话总结本文最关键、最想证明的中心思想。
    2.  **目标读者 (Audience)**: 这篇文章是写给谁看的？
    3.  **写作语气 (Tone)**: 文章应该是什么感觉？（例如：学术严谨、科普风趣、客观中立）
    4.  **叙事节奏 (Narrative Pace)**: 内容应该如何展开？
    5.  **关键术语 (Key Terminology)**: 列出本文必须统一使用的3-5个核心术语及其简要定义。
    请直接输出这份指南，不要添加任何额外的解释。
    """
    messages = [
        {"role": "system", "content": "你是一位创作风格指南的大师级编辑。"},
        {"role": "user", "content": prompt},
    ]
    style_guide = call_ai_writing_with_auto_continue(
        config,
        config.editorial_model_name,
        messages,
        max_tokens_output=1024,
        temperature=config.temperature_creative,
        max_continues=1,
    )
    if "AI模型调用失败" in style_guide:
        logging.error(f"生成风格指南失败: {style_guide}")
        return ""
    logging.info("--- 风格与声音指南已生成 ---")
    logging.info(style_guide)
    return style_guide
