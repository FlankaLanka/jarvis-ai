"""
Jarvis Voice Assistant - Task Manager Service

Manages async tasks with progress tracking and interrupt support.
"""

import asyncio
from typing import Dict, Any, Optional, Callable, Awaitable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import uuid

from loguru import logger

from app.services.interrupt import interrupt_handler


class TaskStatus(Enum):
    """Status of a managed task."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Task:
    """Represents a managed async task."""
    id: str
    session_id: str
    name: str
    status: TaskStatus = TaskStatus.PENDING
    progress: float = 0.0
    result: Any = None
    error: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    _asyncio_task: Optional[asyncio.Task] = field(default=None, repr=False)


class TaskManager:
    """
    Manages async tasks with progress tracking.
    
    Features:
    1. Task creation and tracking
    2. Progress updates
    3. Interrupt integration
    4. Long-task notifications
    """
    
    def __init__(self, long_task_threshold: float = 3.0):
        """
        Initialize the task manager.
        
        Args:
            long_task_threshold: Seconds before a task is considered "long"
        """
        self._tasks: Dict[str, Task] = {}
        self._long_task_threshold = long_task_threshold
        self._progress_callbacks: Dict[str, Callable] = {}
    
    async def run_task(
        self,
        session_id: str,
        name: str,
        coroutine: Awaitable[Any],
        on_progress: Optional[Callable[[float], None]] = None
    ) -> Task:
        """
        Run an async task with management.
        
        Args:
            session_id: The session this task belongs to
            name: Human-readable task name
            coroutine: The async function to run
            on_progress: Optional callback for progress updates
            
        Returns:
            The completed Task object
        """
        task_id = str(uuid.uuid4())[:8]
        
        task = Task(
            id=task_id,
            session_id=session_id,
            name=name,
        )
        
        self._tasks[task_id] = task
        
        if on_progress:
            self._progress_callbacks[task_id] = on_progress
        
        try:
            task.status = TaskStatus.RUNNING
            task.started_at = datetime.utcnow()
            
            logger.debug(f"Starting task {task_id}: {name}")
            
            # Run with interrupt checking
            task.result = await self._run_with_interrupt_check(
                task_id, session_id, coroutine
            )
            
            task.status = TaskStatus.COMPLETED
            task.progress = 1.0
            
        except asyncio.CancelledError:
            task.status = TaskStatus.CANCELLED
            logger.info(f"Task {task_id} was cancelled")
            
        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error = str(e)
            logger.error(f"Task {task_id} failed: {e}")
            
        finally:
            task.completed_at = datetime.utcnow()
            self._progress_callbacks.pop(task_id, None)
        
        return task
    
    async def _run_with_interrupt_check(
        self,
        task_id: str,
        session_id: str,
        coroutine: Awaitable[Any]
    ) -> Any:
        """Run a coroutine with periodic interrupt checks."""
        # Create the actual task
        asyncio_task = asyncio.create_task(coroutine)
        self._tasks[task_id]._asyncio_task = asyncio_task
        
        start_time = datetime.utcnow()
        notified_long_task = False
        
        while not asyncio_task.done():
            # Check for interrupt
            if interrupt_handler.is_interrupted(session_id):
                asyncio_task.cancel()
                raise asyncio.CancelledError()
            
            # Check if this is a long-running task
            elapsed = (datetime.utcnow() - start_time).total_seconds()
            if elapsed > self._long_task_threshold and not notified_long_task:
                notified_long_task = True
                logger.info(f"Task {task_id} is taking longer than expected")
                # Trigger long-task callback if registered
                callback = self._progress_callbacks.get(task_id)
                if callback:
                    callback(-1)  # -1 indicates "still working"
            
            # Brief sleep to avoid busy waiting
            await asyncio.sleep(0.1)
        
        return asyncio_task.result()
    
    def update_progress(self, task_id: str, progress: float):
        """
        Update task progress.
        
        Args:
            task_id: The task ID
            progress: Progress value (0.0 to 1.0)
        """
        if task_id in self._tasks:
            self._tasks[task_id].progress = min(1.0, max(0.0, progress))
            
            callback = self._progress_callbacks.get(task_id)
            if callback:
                callback(progress)
    
    def get_task(self, task_id: str) -> Optional[Task]:
        """Get a task by ID."""
        return self._tasks.get(task_id)
    
    def get_session_tasks(self, session_id: str) -> list[Task]:
        """Get all tasks for a session."""
        return [t for t in self._tasks.values() if t.session_id == session_id]
    
    def cancel_task(self, task_id: str) -> bool:
        """
        Cancel a running task.
        
        Args:
            task_id: The task ID
            
        Returns:
            True if cancelled, False if not found/already done
        """
        task = self._tasks.get(task_id)
        
        if not task or task.status != TaskStatus.RUNNING:
            return False
        
        if task._asyncio_task:
            task._asyncio_task.cancel()
        
        return True
    
    def cancel_session_tasks(self, session_id: str):
        """Cancel all tasks for a session."""
        for task in self.get_session_tasks(session_id):
            if task.status == TaskStatus.RUNNING:
                self.cancel_task(task.id)
    
    def cleanup_old_tasks(self, max_age_minutes: int = 30):
        """Remove old completed/failed tasks."""
        now = datetime.utcnow()
        to_remove = []
        
        for task_id, task in self._tasks.items():
            if task.completed_at:
                age = (now - task.completed_at).total_seconds() / 60
                if age > max_age_minutes:
                    to_remove.append(task_id)
        
        for task_id in to_remove:
            del self._tasks[task_id]
        
        if to_remove:
            logger.debug(f"Cleaned up {len(to_remove)} old tasks")
    
    def is_long_running(self, task_id: str) -> bool:
        """Check if a task is long-running."""
        task = self._tasks.get(task_id)
        
        if not task or not task.started_at:
            return False
        
        elapsed = (datetime.utcnow() - task.started_at).total_seconds()
        return elapsed > self._long_task_threshold


# Global task manager instance
task_manager = TaskManager()

