# utils/progress_tracker.py

import json
import logging
import time
from dataclasses import asdict, dataclass

# 使用 rich 进度条库
try:
    from rich.console import Console
    from rich.progress import (
        BarColumn,
        Progress,
        SpinnerColumn,
        TaskProgressColumn,
        TextColumn,
        TimeElapsedColumn,
        TimeRemainingColumn,
    )
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    logging.warning("rich library not installed. Progress bar will be disabled.")


@dataclass
class ProgressStep:
    """单个进度步骤的详细信息"""

    name: str
    status: str  # 'pending', 'running', 'completed', 'failed'
    start_time: float | None = None
    end_time: float | None = None
    progress: float = 0.0  # 0-100
    detail: str = ""
    error: str | None = None

    def to_dict(self):
        return asdict(self)

    @property
    def duration(self) -> float | None:
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None


class EnhancedProgressTracker:
    """使用 rich 库的增强进度追踪器"""

    def __init__(self, task_id: str, total_steps: int = 0, start_time: float | None = None):
        self.task_id = task_id
        self.total_steps = total_steps
        self.current_step_index = 0
        self.steps: list[ProgressStep] = []
        self.step_times: list[float] = []
        self.start_time = start_time if start_time is not None else time.time()
        self.paused_time = 0.0
        self.pause_start = None

        # 步骤权重（某些步骤比其他步骤更耗时）
        self.step_weights = {
            "style_guide_node": 1,
            "plan_node": 2,
            "topology_writer_node": 10,
            "draft_node": 10,
            "critique_node": 3,
            "research_node": 5,
            "refine_node": 4,
            "apply_patches_node": 1,
            "polish_node": 3,
            "memory_node": 1,
        }

        # Rich 进度条（如果可用）
        if RICH_AVAILABLE:
            self.console = Console()  # type: ignore[possibly-unbound]
            self.progress = Progress(  # type: ignore[possibly-unbound]
                SpinnerColumn(),  # type: ignore[possibly-unbound]
                TextColumn("[progress.description]{task.description}"),  # type: ignore[possibly-unbound]
                BarColumn(),  # type: ignore[possibly-unbound]
                TaskProgressColumn(),  # type: ignore[possibly-unbound]
                TimeElapsedColumn(),  # type: ignore[possibly-unbound]
                TimeRemainingColumn(),  # type: ignore[possibly-unbound]
                console=self.console,
                transient=False,  # 完成后不消失
            )
            self.progress.start()
            self._rich_task = self.progress.add_task(
                f"[cyan]{task_id}",
                total=100.0,  # 使用百分比
            )
        else:
            self.console = None
            self.progress = None
            self._rich_task = None

    def add_step(self, name: str, detail: str = ""):
        """添加新步骤"""
        step = ProgressStep(name=name, status="pending", detail=detail)
        self.steps.append(step)
        if not self.total_steps:
            self.total_steps = len(self.steps)

    def start_step(self, name: str, detail: str = ""):
        """开始一个步骤"""
        step = next((s for s in self.steps if s.name == name and s.status == "pending"), None)
        if not step:
            # 尝试复用或创建新步骤
            step = next((s for s in self.steps if s.name == name), None)
            if step:
                step.status = "pending"
                step.start_time = None
                step.end_time = None
                step.progress = 0.0
                step.error = None
                step.detail = detail
            else:
                step = ProgressStep(name=name, status="pending", detail=detail)
                self.steps.append(step)
                if not self.total_steps:
                    self.total_steps = len(self.steps)
                else:
                    self.total_steps = max(self.total_steps, len(self.steps))

        step.status = "running"
        step.start_time = time.time()
        step.detail = detail
        self.current_step_index = self.steps.index(step)

        self._update_progress()

    def update_step(self, name: str, progress: float, detail: str = ""):
        """更新步骤进度"""
        step = next((s for s in self.steps if s.name == name and s.status == "running"), None)
        if step:
            step.progress = min(100.0, max(0.0, progress))
            if detail:
                step.detail = detail
        self._update_progress()

    def complete_step(self, name: str, detail: str = ""):
        """完成一个步骤"""
        step = next((s for s in self.steps if s.name == name and s.status == "running"), None)
        if step:
            step.status = "completed"
            step.end_time = time.time()
            step.progress = 100.0
            if detail:
                step.detail = detail
            if step.duration:
                self.step_times.append(step.duration)
        self._update_progress()

    def fail_step(self, name: str, error: str):
        """标记步骤失败"""
        step = next((s for s in self.steps if s.name == name), None)
        if step:
            step.status = "failed"
            step.end_time = time.time()
            step.error = error
        self._update_progress()

    def pause(self):
        """暂停追踪"""
        if not self.pause_start:
            self.pause_start = time.time()

    def resume(self):
        """恢复追踪"""
        if self.pause_start:
            self.paused_time += time.time() - self.pause_start
            self.pause_start = None

    def calculate_eta(self) -> float | None:
        """计算预估剩余时间（秒）"""
        import statistics

        completed_steps = [s for s in self.steps if s.status == "completed"]
        remaining_steps = [s for s in self.steps if s.status in ["pending", "running"]]
        current_running = next((s for s in self.steps if s.status == "running"), None)

        if not self.step_times:
            overall_progress = self.get_overall_progress()
            if overall_progress > 0:
                elapsed = self.get_elapsed_time()
                total_estimate = elapsed / (overall_progress / 100.0)
                return max(0.0, total_estimate - elapsed)
            return None

        if not remaining_steps:
            return 0.0

        total_weight = sum(self.step_weights.get(s.name, 1) for s in completed_steps)
        remaining_weight = sum(self.step_weights.get(s.name, 1) for s in remaining_steps)

        if total_weight == 0:
            avg_time = statistics.mean(self.step_times)
            return avg_time * len(remaining_steps)

        avg_time_per_weight = sum(self.step_times) / total_weight
        eta = avg_time_per_weight * remaining_weight

        if current_running and current_running.progress > 0:
            current_weight = self.step_weights.get(current_running.name, 1)
            current_remaining = current_weight * (1 - current_running.progress / 100.0)
            eta += (current_remaining - current_weight) * avg_time_per_weight

        return max(0.0, eta)

    def get_overall_progress(self) -> float:
        """获取总体进度（0-100）"""
        if not self.steps:
            return 0.0
        total_steps = max(self.total_steps, len(self.steps))
        completed_steps = len([s for s in self.steps if s.status == "completed"])
        running_step = next((s for s in self.steps if s.status == "running"), None)
        running_fraction = (running_step.progress / 100.0) if running_step else 0.0

        overall = (completed_steps + running_fraction) / total_steps
        return max(0.0, min(100.0, overall * 100.0))

    def get_elapsed_time(self) -> float:
        """获取已用时间（不包括暂停时间）"""
        elapsed = time.time() - self.start_time - self.paused_time
        if self.pause_start:
            elapsed -= time.time() - self.pause_start
        return elapsed

    def format_duration(self, seconds: float) -> str:
        """格式化时间显示"""
        if seconds < 60:
            return f"{int(seconds)}秒"
        elif seconds < 3600:
            minutes = int(seconds / 60)
            secs = int(seconds % 60)
            return f"{minutes}分{secs}秒"
        else:
            hours = int(seconds / 3600)
            minutes = int((seconds % 3600) / 60)
            return f"{hours}小时{minutes}分"

    def get_status_summary(self) -> dict:
        """获取完整的状态摘要"""
        eta = self.calculate_eta()
        overall_progress = self.get_overall_progress()
        elapsed = self.get_elapsed_time()

        completed_count = len([s for s in self.steps if s.status == "completed"])
        failed_count = len([s for s in self.steps if s.status == "failed"])
        current_step = next((s for s in self.steps if s.status == "running"), None)

        if current_step:
            current_step_name = current_step.detail
        else:
            last_completed = None
            for s in reversed(self.steps):
                if s.status == "completed":
                    last_completed = s
                    break
            if last_completed and last_completed.detail:
                current_step_name = f"{last_completed.detail} → 正在切换下一节点..."
            else:
                current_step_name = "准备中..."

        current_step_progress = current_step.progress if current_step else 0.0
        current_step_elapsed = 0.0
        if current_step and current_step.start_time:
            current_step_elapsed = time.time() - current_step.start_time

        return {
            "type": "enhanced_progress",
            "task_id": self.task_id,
            "overall_progress": round(overall_progress, 1),
            "current_step_index": self.current_step_index,
            "current_step_name": current_step_name,
            "current_step_progress": round(current_step_progress, 1),
            "current_step_elapsed": round(current_step_elapsed, 1),
            "current_step_elapsed_readable": self.format_duration(current_step_elapsed),
            "total_steps": self.total_steps,
            "completed_steps": completed_count,
            "failed_steps": failed_count,
            "eta_seconds": round(eta, 1) if eta is not None else None,
            "eta_readable": self.format_duration(eta) if eta is not None else "计算中...",
            "elapsed_time": round(elapsed, 1),
            "elapsed_readable": self.format_duration(elapsed),
            "steps": [s.to_dict() for s in self.steps],
            "timestamp": time.time(),
        }

    def _update_progress(self):
        """更新 rich 进度条"""
        if not RICH_AVAILABLE or not self.progress or self._rich_task is None:
            return

        try:
            overall_progress = self.get_overall_progress()
            current_step = next((s for s in self.steps if s.status == "running"), None)

            description = f"[cyan]{self.task_id}"
            if current_step:
                description += f" | {current_step.detail[:50]}"

            self.progress.update(
                self._rich_task,
                completed=overall_progress,
                description=description,
            )
        except Exception as e:
            logging.debug(f"Progress update failed: {e}")

    def clear_console_for_external_log(self) -> None:
        """清空进度条（rich 自动处理，无需手动清空）"""
        pass

    def redraw_console_line(self) -> None:
        """重绘进度条（rich 自动处理）"""
        pass

    def to_json(self) -> str:
        """导出为 JSON 字符串"""
        return json.dumps(self.get_status_summary(), ensure_ascii=False, indent=2)

    @classmethod
    def from_json(cls, data: str):
        """从 JSON 恢复进度追踪器"""
        state = json.loads(data)
        tracker = cls(state["task_id"], state["total_steps"])
        tracker.current_step_index = state["current_step_index"]
        tracker.start_time = time.time() - state["elapsed_time"]
        tracker.steps = [ProgressStep(**step_data) for step_data in state["steps"]]
        return tracker

    def close(self):
        """停止进度条并清理资源"""
        if RICH_AVAILABLE and self.progress:
            try:
                self.progress.stop()
            except Exception:
                pass

    def pulse(self, message: str | None = None):
        """发送心跳，更新当前步骤描述"""
        running = next((s for s in self.steps if s.status == "running"), None)
        if running and message:
            running.detail = message
        self._update_progress()


# 全局追踪器注册表
_active_trackers: dict[str, EnhancedProgressTracker] = {}
_recent_tracker: EnhancedProgressTracker | None = None


def _refresh_recent_tracker() -> EnhancedProgressTracker | None:
    global _recent_tracker
    for tracker in _active_trackers.values():
        _recent_tracker = tracker
        break
    else:
        _recent_tracker = None
    return _recent_tracker


def _get_recent_tracker() -> EnhancedProgressTracker | None:
    global _recent_tracker
    if _recent_tracker and _recent_tracker.task_id in _active_trackers:
        return _recent_tracker
    return _refresh_recent_tracker()


def get_tracker(task_id: str) -> EnhancedProgressTracker | None:
    """获取指定任务的追踪器"""
    return _active_trackers.get(task_id)


def safe_pulse(task_id: str | None, message: str) -> None:
    """安全发送心跳提示"""
    if not task_id:
        return
    tracker = get_tracker(task_id)
    if not tracker:
        return
    try:
        tracker.pulse(message)
    except Exception as exc:
        logging.debug(f"Failed to pulse tracker {task_id}: {exc}")


def safe_step_update(task_id: str | None, step_name: str, progress: float, detail: str = "") -> None:
    """安全更新步骤进度"""
    if not task_id or not step_name:
        return
    tracker = get_tracker(task_id)
    if not tracker:
        return
    try:
        tracker.update_step(step_name, progress, detail)
    except Exception as exc:
        logging.debug(f"Failed to update step '{step_name}' for task {task_id}: {exc}")


def register_tracker(tracker: EnhancedProgressTracker):
    """注册追踪器"""
    global _recent_tracker
    _active_trackers[tracker.task_id] = tracker
    _recent_tracker = tracker


def unregister_tracker(task_id: str):
    """注销追踪器"""
    global _recent_tracker
    tracker = _active_trackers.pop(task_id, None)
    if tracker:
        tracker.close()
    if tracker is _recent_tracker:
        _refresh_recent_tracker()


def before_console_log() -> None:
    """日志输出前的钩子（rich 自动处理，保留接口兼容）"""
    pass


def after_console_log() -> None:
    """日志输出后的钩子（rich 自动处理，保留接口兼容）"""
    pass
