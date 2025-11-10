# planning/outline.py
from __future__ import annotations

import logging
import os
from typing import Any

from config import Config


def allocate_content_lengths(config: Config, outline_data: dict[str, Any], total_target_chars: int) -> dict[str, Any]:
    """
    将总目标字数按比例分配给大纲中的每个章节和子章节。
    """
    logging.info(f"\n--- 正在为大纲分配内容长度 (总目标: {total_target_chars} 字符) ---")
    if not outline_data or "outline" not in outline_data or not outline_data["outline"]:
        logging.error("  用于长度分配的大纲数据无效或为空。")
        return outline_data

    outline_items: list[dict[str, Any]] = [item for item in outline_data["outline"] if isinstance(item, dict)]
    if not outline_items:
        logging.error("  大纲中未找到有效的章节项用于长度分配。")
        return outline_data

    # 递归地为所有层级分配字数
    _allocate_recursive(config, outline_items, total_target_chars)

    # 保存分配后的大纲以供调试
    if config.session_dir:
        import json

        path = os.path.join(config.session_dir, "allocated_document_outline.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(outline_data, f, ensure_ascii=False, indent=4)

    logging.info("--- 内容长度分配完成 ---")
    return outline_data


def _allocate_recursive(config: Config, sections_list: list[dict[str, Any]], parent_allocated_chars: int) -> None:
    """
    一个递归的辅助函数，用于将父章节的字数分配给其子章节。
    """
    if not sections_list:
        return

    # 1. 标准化比例：处理缺失或无效的比例
    items_with_ratio: list[dict[str, Any]] = []
    items_without_ratio: list[dict[str, Any]] = []
    current_total_ratio = 0.0

    for item in sections_list:
        ratio: float | None = item.get("target_chars_ratio")
        if ratio is None:
            items_without_ratio.append(item)
            continue
        try:
            ratio_float = float(ratio)
            if ratio_float > 0:
                items_with_ratio.append(item)
                item["_ratio_numeric"] = ratio_float
                current_total_ratio += ratio_float
            else:
                items_without_ratio.append(item)
        except (ValueError, TypeError):
            items_without_ratio.append(item)

    # 为没有比例的项分配剩余比例
    if items_without_ratio:
        remaining_ratio = max(0, 1.0 - current_total_ratio)
        ratio_per_item = remaining_ratio / len(items_without_ratio) if items_without_ratio else 0
        for item in items_without_ratio:
            item["_ratio_numeric"] = ratio_per_item

    # 2. 归一化：确保当前层级所有比例总和为 1.0
    final_total_ratio = sum(item.get("_ratio_numeric", 0.0) for item in sections_list)
    if final_total_ratio > 0:
        for item in sections_list:
            item["_ratio_numeric"] = item.get("_ratio_numeric", 0.0) / final_total_ratio
    else:  # 如果所有比例都为0，则平分
        equal_share = 1.0 / len(sections_list) if sections_list else 0
        for item in sections_list:
            item["_ratio_numeric"] = equal_share

    # 3. 分配字数并处理余数
    total_allocated = 0
    for item in sections_list:
        item["allocated_chars"] = int(round(item.get("_ratio_numeric", 0.0) * parent_allocated_chars))
        total_allocated += item["allocated_chars"]

    remainder = parent_allocated_chars - total_allocated
    # 按比例大小，将余数逐一分配给最大的项，以减少相对误差
    sorted_items: list[dict[str, Any]] = sorted(sections_list, key=lambda x: x.get("_ratio_numeric", 0.0), reverse=True)
    for i in range(abs(remainder)):
        item_to_adjust = sorted_items[i % len(sorted_items)]
        item_to_adjust["allocated_chars"] += 1 if remainder > 0 else -1

    # 4. 递归调用子章节并清理临时键
    for item in sections_list:
        logging.info(f"    - {'  ' * item.get('title', '').count('#')}章节 '{item.get('title', 'N/A')}': 分配 {item['allocated_chars']} 字符")
        if "sections" in item and item["sections"]:
            _allocate_recursive(config, item["sections"], item["allocated_chars"])
        del item["_ratio_numeric"]  # 清理临时键
