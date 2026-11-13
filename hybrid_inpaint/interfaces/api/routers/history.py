"""历史：分页列表 / CSV 导出 / 详情 / 删除。

注意路由顺序：/history/export 必须先于 /history/{job_id} 注册。
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, Query
from fastapi.responses import FileResponse, Response
from starlette.background import BackgroundTask

from hybrid_inpaint.bootstrap import Container
from hybrid_inpaint.domain.services import validate_job_id
from hybrid_inpaint.interfaces.api.dependencies import get_container
from hybrid_inpaint.interfaces.api.schemas.history import HistoryItem, HistoryPage

router = APIRouter(tags=["history"])


@router.get("/history", response_model=HistoryPage)
def list_history(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    container: Container = Depends(get_container),
) -> HistoryPage:
    records = container.history_service.list(limit, offset)
    return HistoryPage(
        items=[HistoryItem.from_record(record) for record in records],
        limit=limit,
        offset=offset,
    )


@router.get("/history/export")
def export_history(container: Container = Depends(get_container)) -> FileResponse:
    directory = tempfile.mkdtemp(prefix="hybrid-export-")
    target = Path(directory) / "history.csv"
    container.history_service.export_csv(target)
    return FileResponse(
        target,
        media_type="text/csv",
        filename="history.csv",
        background=BackgroundTask(shutil.rmtree, directory, ignore_errors=True),
    )


@router.get("/history/{job_id}", response_model=HistoryItem)
def get_history(
    job_id: str, container: Container = Depends(get_container)
) -> HistoryItem:
    validate_job_id(job_id)
    return HistoryItem.from_record(container.history_service.get(job_id))


@router.delete("/history/{job_id}", status_code=204)
def delete_history(
    job_id: str,
    delete_files: bool = Query(False),
    container: Container = Depends(get_container),
) -> Response:
    validate_job_id(job_id)
    container.history_service.delete(job_id, delete_files=delete_files)
    return Response(status_code=204)
