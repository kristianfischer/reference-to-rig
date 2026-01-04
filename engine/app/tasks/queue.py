"""In-memory task queue for background processing.

This is a simple implementation for MVP. The interface is designed
to be swappable with Celery/Redis for production use.
"""

import threading
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Optional
from uuid import uuid4

import structlog

from app.models import TaskStatus, TaskStatusResponse

logger = structlog.get_logger()


class Task(ABC):
    """Base class for background tasks."""

    def __init__(self):
        self.task_id: str = str(uuid4())
        self.status: TaskStatus = TaskStatus.PENDING
        self.progress: float = 0.0
        self.message: Optional[str] = None
        self.result: Optional[Any] = None
        self.error: Optional[str] = None
        self.started_at: Optional[datetime] = None
        self.completed_at: Optional[datetime] = None

    @abstractmethod
    def run(self) -> Any:
        """Execute the task. Override in subclass."""
        pass

    def update_progress(self, progress: float, message: Optional[str] = None) -> None:
        """Update task progress (0.0 to 1.0)."""
        self.progress = min(1.0, max(0.0, progress))
        if message:
            self.message = message

    def get_status(self) -> TaskStatusResponse:
        """Get current task status."""
        return TaskStatusResponse(
            task_id=self.task_id,
            status=self.status,
            progress=self.progress,
            message=self.message,
            result=self.result,
            error=self.error,
        )


@dataclass
class TaskQueue:
    """Simple in-memory task queue with thread pool execution.

    Design notes:
    - Tasks are executed in a thread pool
    - Status is stored in memory (lost on restart)
    - Interface matches Celery patterns for easy migration

    To migrate to Celery:
    1. Replace submit() with celery_task.delay()
    2. Replace get_status() with AsyncResult
    3. Remove start()/stop() lifecycle
    """

    max_workers: int = 4
    _executor: Optional[ThreadPoolExecutor] = field(default=None, init=False)
    _tasks: dict[str, Task] = field(default_factory=dict, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)
    _running: bool = field(default=False, init=False)

    def start(self) -> None:
        """Start the task queue."""
        self._executor = ThreadPoolExecutor(max_workers=self.max_workers)
        self._running = True
        logger.info("Task queue started", max_workers=self.max_workers)

    def stop(self) -> None:
        """Stop the task queue gracefully."""
        self._running = False
        if self._executor:
            self._executor.shutdown(wait=True)
            self._executor = None
        logger.info("Task queue stopped")

    def submit(self, task: Task) -> str:
        """Submit a task for execution. Returns task ID."""
        if not self._running:
            raise RuntimeError("Task queue not running")

        with self._lock:
            self._tasks[task.task_id] = task

        logger.info("Task submitted", task_id=task.task_id, task_type=type(task).__name__)

        # Submit to thread pool
        self._executor.submit(self._execute_task, task)

        return task.task_id

    def _execute_task(self, task: Task) -> None:
        """Execute a task in the thread pool."""
        task.status = TaskStatus.RUNNING
        task.started_at = datetime.utcnow()

        try:
            logger.info("Task started", task_id=task.task_id)
            result = task.run()
            task.result = result
            task.status = TaskStatus.COMPLETED
            task.progress = 1.0
            logger.info("Task completed", task_id=task.task_id)
        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error = str(e)
            logger.error("Task failed", task_id=task.task_id, error=str(e))
        finally:
            task.completed_at = datetime.utcnow()

    def get_status(self, task_id: str) -> Optional[TaskStatusResponse]:
        """Get task status by ID."""
        with self._lock:
            task = self._tasks.get(task_id)
            if task:
                return task.get_status()
            return None

    def get_task(self, task_id: str) -> Optional[Task]:
        """Get task instance by ID."""
        with self._lock:
            return self._tasks.get(task_id)

    def cleanup_completed(self, max_age_seconds: int = 3600) -> int:
        """Remove completed tasks older than max_age. Returns count removed."""
        cutoff = datetime.utcnow().timestamp() - max_age_seconds
        removed = 0

        with self._lock:
            to_remove = []
            for task_id, task in self._tasks.items():
                if task.completed_at and task.completed_at.timestamp() < cutoff:
                    to_remove.append(task_id)

            for task_id in to_remove:
                del self._tasks[task_id]
                removed += 1

        if removed:
            logger.info("Cleaned up old tasks", count=removed)

        return removed


# Global task queue instance
task_queue = TaskQueue()


