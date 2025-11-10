# utils/draft_manager.py

import logging
import threading
from collections.abc import Callable
from datetime import datetime
from typing import Any


class DraftSection:
    """草稿章节"""

    def __init__(self, section_id: str, title: str, content: str = "", status: str = "pending", order_key: str | None = None):
        self.section_id = section_id
        self.title = title
        self.content = content
        self.status = status  # 'pending', 'generating', 'completed'
        self.updated_at = datetime.now()
        self.order_key = order_key

    def to_dict(self):
        return {
            "section_id": self.section_id,
            "title": self.title,
            "content": self.content,
            "status": self.status,
            "updated_at": self.updated_at.isoformat(),
            "word_count": len(self.content),
            "order_key": self.order_key,
        }


class DraftManager:
    """管理草稿的增量更新和实时预览"""

    def __init__(self, task_id: str):
        self.task_id = task_id
        self.sections: dict[str, DraftSection] = {}
        self.outline: dict = {}
        self.style_guide: str = ""
        self.metadata: dict = {
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            "total_words": 0,
        }
        self._lock = threading.RLock()
        self._update_callbacks: list[Callable[[dict[str, Any]], None]] = []
        self._section_order: dict[str, tuple[int, ...]] = {}
        self._order_counter = 0

    def set_outline(self, outline: dict):
        """设置大纲"""
        with self._lock:
            self.outline = outline or {}
            self.metadata["last_updated"] = datetime.now().isoformat()
            existing_sections = self.sections
            self.sections = {}
            self._section_order = {}
            self._order_counter = 0
            sections = self.outline.get("outline") if isinstance(self.outline, dict) else None
            self._create_sections_from_outline(sections, (), existing_sections)

    def _create_sections_from_outline(
        self,
        sections: Any,
        parent_path: tuple[int, ...],
        existing_sections: dict[str, DraftSection],
    ) -> None:
        """根据大纲递归创建章节"""
        if not isinstance(sections, list):
            return

        for idx, section in enumerate(sections, start=1):
            if not isinstance(section, dict):
                continue

            order_path = parent_path + (idx,)
            order_key = ".".join(str(part) for part in order_path)
            section_id = str(section.get("id") or section.get("section_id") or order_key)
            title = section.get("title", f"章节 {order_key}")

            draft_section = existing_sections.get(section_id)
            if draft_section:
                draft_section.title = title
                draft_section.order_key = order_key
            else:
                draft_section = DraftSection(
                    section_id=section_id,
                    title=title,
                    content="",
                    status="pending",
                    order_key=order_key,
                )

            self.sections[section_id] = draft_section
            self._section_order[section_id] = order_path

            child_sections = section.get("sections")
            if isinstance(child_sections, list):
                self._create_sections_from_outline(child_sections, order_path, existing_sections)

    def set_style_guide(self, style_guide: str):
        """设置风格指南"""
        with self._lock:
            self.style_guide = style_guide
            self.metadata["last_updated"] = datetime.now().isoformat()

    def update_section(self, section_id: str, content: str, status: str = "generating"):
        """更新章节内容"""
        with self._lock:
            if section_id in self.sections:
                self.sections[section_id].content = content
                self.sections[section_id].status = status
                self.sections[section_id].updated_at = datetime.now()
            else:
                # 如果章节不存在，创建它
                self.sections[section_id] = DraftSection(
                    section_id=section_id,
                    title=f"章节 {section_id}",
                    content=content,
                    status=status,
                )
            self._ensure_fallback_order(section_id)

            self.metadata["last_updated"] = datetime.now().isoformat()
            self._update_total_words()

            # 触发更新回调
            self._notify_updates()

    def complete_section(self, section_id: str):
        """标记章节完成"""
        with self._lock:
            if section_id in self.sections:
                self.sections[section_id].status = "completed"
                self.sections[section_id].updated_at = datetime.now()
                self.metadata["last_updated"] = datetime.now().isoformat()
                self._notify_updates()

    def get_section(self, section_id: str) -> DraftSection | None:
        """获取单个章节"""
        return self.sections.get(section_id)

    def get_current_draft(self) -> str:
        """获取当前完整草稿（Markdown 格式）"""
        with self._lock:
            draft_lines = []

            # 添加标题
            if self.outline.get("title"):
                draft_lines.append(f"# {self.outline['title']}\n")

            # 按章节 ID 排序
            sorted_sections = sorted(self.sections.items(), key=lambda x: self._section_sort_key(x[0]))

            # 组装内容
            for section_id, section in sorted_sections:
                depth = 0
                if section.order_key:
                    depth = section.order_key.count(".")
                else:
                    depth = section_id.count(".")
                level = depth + 2  # 从 ## 开始
                header = "#" * level

                if section.status == "pending":
                    draft_lines.append(f"{header} {section.title}\n")
                    draft_lines.append("*[待生成...]*\n")
                elif section.status == "generating":
                    draft_lines.append(f"{header} {section.title}\n")
                    if section.content:
                        draft_lines.append(section.content)
                    draft_lines.append("\n*[生成中...]*\n")
                else:  # completed
                    draft_lines.append(f"{header} {section.title}\n")
                    if section.content:
                        draft_lines.append(section.content)
                    draft_lines.append("\n")

            return "\n".join(draft_lines)

    def get_preview_data(self) -> dict:
        """获取预览数据（JSON 格式）"""
        with self._lock:
            return {
                "task_id": self.task_id,
                "draft_content": self.get_current_draft(),
                "metadata": self.metadata,
                "sections": [s.to_dict() for s in self.sections.values()],
                "outline": self.outline,
                "style_guide": self.style_guide,
                "timestamp": datetime.now().isoformat(),
            }

    def _update_total_words(self):
        """更新总字数"""
        total = sum(len(s.content) for s in self.sections.values())
        self.metadata["total_words"] = total

    def _section_sort_key(self, section_id: str) -> tuple:
        """生成章节排序键"""
        order = self._section_order.get(section_id)
        if order is None:
            order = self._ensure_fallback_order(section_id)
        return order

    def _ensure_fallback_order(self, section_id: str) -> tuple[int, ...]:
        """为缺少顺序信息的章节生成稳定的排序键。"""
        if section_id not in self._section_order:
            self._order_counter += 1
            self._section_order[section_id] = (999999, self._order_counter)
        return self._section_order[section_id]

    def register_update_callback(self, callback: Callable[[dict[str, Any]], None]):
        """注册更新回调（用于实时推送）"""
        self._update_callbacks.append(callback)

    def _notify_updates(self):
        """通知所有更新监听器"""
        for callback in self._update_callbacks:
            try:
                callback(self.get_preview_data())
            except Exception as e:
                logging.error(f"Draft update callback failed: {e}")


# 全局草稿管理器注册表
_active_drafts: dict[str, DraftManager] = {}


def get_draft_manager(task_id: str) -> DraftManager | None:
    """获取指定任务的草稿管理器"""
    return _active_drafts.get(task_id)


def register_draft_manager(manager: DraftManager):
    """注册草稿管理器"""
    _active_drafts[manager.task_id] = manager


def unregister_draft_manager(task_id: str):
    """注销草稿管理器"""
    _active_drafts.pop(task_id, None)
