import asyncio
import logging
from typing import Dict, List, Any, Optional, Callable, Coroutine
from datetime import datetime, timedelta
import time

logger = logging.getLogger("ai-model-server")

class TaskScheduler:
    """Service for scheduling and managing background tasks"""
    
    def __init__(self):
        self.tasks: Dict[str, Dict[str, Any]] = {}
        self.periodic_tasks: Dict[str, Dict[str, Any]] = {}
        self.running = False
        self.scheduler_task = None
    
    async def start(self):
        """Start the task scheduler"""
        if self.running:
            return
        
        self.running = True
        self.scheduler_task = asyncio.create_task(self._scheduler_loop())
        logger.info("Task scheduler started")
    
    async def stop(self):
        """Stop the task scheduler"""
        if not self.running:
            return
        
        self.running = False
        if self.scheduler_task:
            self.scheduler_task.cancel()
            try:
                await self.scheduler_task
            except asyncio.CancelledError:
                pass
        
        # Cancel all running tasks
        for task_id, task_info in list(self.tasks.items()):
            if task_info["task"].done() is False:
                task_info["task"].cancel()
        
        logger.info("Task scheduler stopped")
    
    async def _scheduler_loop(self):
        """Main scheduler loop for periodic tasks"""
        while self.running:
            now = datetime.now()
            
            # Check periodic tasks
            for task_id, task_info in list(self.periodic_tasks.items()):
                if now >= task_info["next_run"]:
                    # Schedule the next run
                    task_info["next_run"] = now + task_info["interval"]
                    
                    # Run the task
                    self.schedule_task(
                        task_id=f"{task_id}_{int(time.time())}",
                        coroutine=task_info["coroutine"](*task_info["args"], **task_info["kwargs"]),
                        description=task_info["description"]
                    )
            
            # Clean up completed tasks
            self._cleanup_tasks()
            
            # Sleep for a short time
            await asyncio.sleep(1)
    
    def _cleanup_tasks(self):
        """Clean up completed tasks"""
        for task_id in list(self.tasks.keys()):
            task_info = self.tasks[task_id]
            if task_info["task"].done():
                # Check for exceptions
                if task_info["task"].exception():
                    logger.error(f"Task {task_id} failed with exception: {task_info['task'].exception()}")
                else:
                    logger.info(f"Task {task_id} completed successfully")
                
                # Remove the task
                del self.tasks[task_id]
    
    def schedule_task(self, task_id: str, coroutine: Coroutine, description: str = "") -> str:
        """Schedule a one-time task"""
        if task_id in self.tasks:
            logger.warning(f"Task {task_id} already exists, overwriting")
        
        task = asyncio.create_task(coroutine)
        self.tasks[task_id] = {
            "task": task,
            "description": description,
            "start_time": datetime.now()
        }
        
        logger.info(f"Scheduled task {task_id}: {description}")
        return task_id
    
    def schedule_periodic_task(self, task_id: str, interval: timedelta, coroutine_factory: Callable, 
                              args: tuple = (), kwargs: dict = None, description: str = "") -> str:
        """Schedule a periodic task"""
        if kwargs is None:
            kwargs = {}
        
        if task_id in self.periodic_tasks:
            logger.warning(f"Periodic task {task_id} already exists, overwriting")
        
        self.periodic_tasks[task_id] = {
            "interval": interval,
            "next_run": datetime.now(),
            "coroutine": coroutine_factory,
            "args": args,
            "kwargs": kwargs,
            "description": description
        }
        
        logger.info(f"Scheduled periodic task {task_id}: {description} (interval: {interval})")
        return task_id
    
    def cancel_task(self, task_id: str) -> bool:
        """Cancel a scheduled task"""
        if task_id in self.tasks:
            task_info = self.tasks[task_id]
            if not task_info["task"].done():
                task_info["task"].cancel()
            del self.tasks[task_id]
            logger.info(f"Cancelled task {task_id}")
            return True
        elif task_id in self.periodic_tasks:
            del self.periodic_tasks[task_id]
            logger.info(f"Cancelled periodic task {task_id}")
            return True
        else:
            logger.warning(f"Task {task_id} not found")
            return False
    
    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get the status of a task"""
        if task_id in self.tasks:
            task_info = self.tasks[task_id]
            task = task_info["task"]
            
            status = {
                "task_id": task_id,
                "description": task_info["description"],
                "start_time": task_info["start_time"].isoformat(),
                "status": "running"
            }
            
            if task.done():
                if task.exception():
                    status["status"] = "failed"
                    status["error"] = str(task.exception())
                else:
                    status["status"] = "completed"
                    try:
                        status["result"] = task.result()
                    except Exception:
                        status["result"] = None
            
            return status
        elif task_id in self.periodic_tasks:
            task_info = self.periodic_tasks[task_id]
            
            return {
                "task_id": task_id,
                "description": task_info["description"],
                "interval": str(task_info["interval"]),
                "next_run": task_info["next_run"].isoformat(),
                "status": "scheduled"
            }
        
        return None
    
    def list_tasks(self) -> Dict[str, List[Dict[str, Any]]]:
        """List all tasks"""
        one_time_tasks = []
        for task_id, task_info in self.tasks.items():
            task = task_info["task"]
            status = "running"
            if task.done():
                status = "completed" if not task.exception() else "failed"
            
            one_time_tasks.append({
                "task_id": task_id,
                "description": task_info["description"],
                "start_time": task_info["start_time"].isoformat(),
                "status": status
            })
        
        periodic_tasks = []
        for task_id, task_info in self.periodic_tasks.items():
            periodic_tasks.append({
                "task_id": task_id,
                "description": task_info["description"],
                "interval": str(task_info["interval"]),
                "next_run": task_info["next_run"].isoformat()
            })
        
        return {
            "one_time_tasks": one_time_tasks,
            "periodic_tasks": periodic_tasks
        }