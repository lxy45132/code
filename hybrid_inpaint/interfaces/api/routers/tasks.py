"""GET /tasks/{task_id}：worker 模式单图任务状态/结果查询。"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from hybrid_inpaint.bootstrap import Container
from hybrid_inpaint.interfaces.api.dependencies import get_container
from hybrid_inpaint.interfaces.api.schemas.batch import TaskStatus

router = APIRouter(tags=["tasks"])


@router.get("/tasks/{task_id}", response_model=TaskStatus)
def task_status(
    task_id: str,
    request: Request,
    container: Container = Depends(get_container),
) -> TaskStatus:
    result = request.app.state.celery.AsyncResult(task_id)
    payload = result.result if result.successful() else None
    return TaskStatus(
        task_id=task_id,
        status=result.status,
        result=payload if isinstance(payload, dict) else None,
    )
