# -*- coding: utf-8 -*-
"""异步任务管理器 - 使用 asyncio + ThreadPoolExecutor"""

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Callable, Any

from sqlalchemy.orm import Session

from app.database import SessionLocal
from app.models import Task

logger = logging.getLogger(__name__)

# 全局线程池
_executor = ThreadPoolExecutor(max_workers=4)

# 进行中的任务缓存 {task_id: future}
_running_tasks: Dict[int, asyncio.Future] = {}


async def submit_task(task_id: int, func: Callable, **kwargs) -> None:
    """提交一个后台任务到线程池"""
    loop = asyncio.get_event_loop()
    future = loop.run_in_executor(_executor, _run_task_wrapper, task_id, func, kwargs)
    _running_tasks[task_id] = future

    # 添加完成回调
    def _on_done(fut):
        _running_tasks.pop(task_id, None)
    future.add_done_callback(_on_done)


def _run_task_wrapper(task_id: int, func: Callable, kwargs: dict) -> Any:
    """在线程中执行任务，自动更新数据库状态"""
    db = SessionLocal()
    try:
        # 标记为运行中
        task = db.query(Task).filter(Task.id == task_id).first()
        if task:
            task.status = 'running'
            task.progress = 0.0
            db.commit()

        # 传入 progress_callback 以便任务函数汇报进度
        def progress_callback(progress: float, result_data: dict = None):
            t = db.query(Task).filter(Task.id == task_id).first()
            if t:
                t.progress = min(progress, 100.0)
                if result_data:
                    t.result = result_data
                db.commit()

        result = func(db=db, task_id=task_id, progress_callback=progress_callback, **kwargs)

        # 标记完成
        task = db.query(Task).filter(Task.id == task_id).first()
        if task:
            task.status = 'completed'
            task.progress = 100.0
            if result:
                task.result = result
            db.commit()

        return result

    except Exception as e:
        logger.exception(f"Task {task_id} failed")
        task = db.query(Task).filter(Task.id == task_id).first()
        if task:
            task.status = 'failed'
            task.error_msg = str(e)
            db.commit()
        raise
    finally:
        db.close()


def get_task_status(task_id: int) -> bool:
    """检查任务是否仍在运行"""
    return task_id in _running_tasks
