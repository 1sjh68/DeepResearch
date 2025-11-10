# utils/log_streamer.py

import asyncio
import logging

logger = logging.getLogger(__name__)


class LogStreamHandler(logging.Handler):
    """
    一个自定义的日志处理器，用于将日志通过SSE发送给前端。
    (V3 - 任务特定队列版)
    """

    def __init__(self):
        super().__init__()
        self.queues: dict[str, asyncio.Queue] = {}
        self.active_task_id: str | None = None

    def set_active_task(self, task_id: str):
        """设置当前活动的任务ID，以便日志能被正确路由。"""
        if task_id not in self.queues:
            self.queues[task_id] = asyncio.Queue()
        self.active_task_id = task_id

    def clear_active_task(self):
        """清除活动的任务ID。"""
        self.active_task_id = None

    def emit(self, record: logging.LogRecord):
        """将日志记录放入当前活动任务的队列中。"""
        if self.active_task_id:
            try:
                msg = self.format(record)
                self.queues[self.active_task_id].put_nowait(msg)
            except asyncio.QueueFull:
                # 在队列满时可以决定是丢弃还是如何处理
                logger.warning(
                    "Log queue for task %s is full; dropping newest log entry.",
                    self.active_task_id,
                )
            except Exception:
                self.handleError(record)

    async def log_generator(self, task_id: str):
        """
        (V2 - 健壮版) 为特定任务ID创建异步日志生成器。
        能处理客户端断开连接等异常。
        """
        if task_id not in self.queues:
            self.queues[task_id] = asyncio.Queue()

        q = self.queues[task_id]
        logging.info(f"日志流生成器已为任务 {task_id} 启动。")
        while True:
            try:
                log_entry = await q.get()
                yield f"data: {log_entry}\n\n"
                q.task_done()
            except asyncio.CancelledError:
                # 客户端断开连接时的正常、预期行为
                logging.info(f"任务 {task_id} 的日志流被客户端主动取消。")
                break
            except Exception as e:
                # 捕捉其他因连接断开导致的发送异常 (如 socket.send() 错误)
                logging.warning(f"任务 {task_id} 的日志流因异常而中断: {e}")
                break  # 退出循环，防止无限刷警告
        logging.info(f"日志流生成器已为任务 {task_id} 正常关闭。")


# 创建一个全局的日志流处理器实例
log_stream_handler = LogStreamHandler()
