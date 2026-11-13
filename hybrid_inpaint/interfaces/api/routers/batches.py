"""批量任务路由：创建（chunk 投递）/ 状态 / 取消 / report / SSE 事件流。

进度链路（AD-11）：MySQL/SQLite 权威状态 → Redis Stream → SSE。
SSE 支持 Last-Event-ID 断线回放；15s 无事件发 keep-alive 注释行。
"""

from __future__ import annotations

import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from celery import chord, group
from fastapi import APIRouter, Depends, File, Form, Request, UploadFile
from fastapi.responses import FileResponse, StreamingResponse

from hybrid_inpaint.bootstrap import Container
from hybrid_inpaint.domain.entities import BatchItemState, BatchJobState
from hybrid_inpaint.domain.errors import (
    ModelUnavailableError,
    NotFoundError,
    ResourceLimitError,
    ValidationError,
)
from hybrid_inpaint.domain.services import validate_job_id
from hybrid_inpaint.interfaces.api.dependencies import get_container
from hybrid_inpaint.interfaces.api.schemas.batch import (
    BatchCreated,
    BatchDetail,
    CancelResponse,
)
from hybrid_inpaint.interfaces.api.tasks.batch_chunk import batch_chunk, batch_stream
from hybrid_inpaint.interfaces.api.tasks.batch_finalize import batch_finalize

router = APIRouter(tags=["batches"])

_STREAM_ID_PATTERN = re.compile(r"^\d+-\d+$")
_SSE_BLOCK_MS = 15_000


@router.post("/batches", response_model=BatchCreated, status_code=202)
async def create_batch(
    request: Request,
    images: list[UploadFile] = File(...),
    masks: list[UploadFile] = File(...),
    note: str = Form(""),
    container: Container = Depends(get_container),
) -> BatchCreated:
    if not images or len(images) != len(masks):
        raise ValidationError("images and masks must be non-empty and equal in count")
    max_items = container.config.system.batch.max_items
    if len(images) > max_items:
        raise ResourceLimitError(f"batch exceeds the {max_items}-item limit")

    batch_id = uuid4().hex
    upload_dir = container.config.system.output_dir.parent / "tmp" / batch_id
    upload_dir.mkdir(parents=True, exist_ok=True)
    max_bytes = container.limits.max_file_mb * 1024 * 1024
    items: list[BatchItemState] = []
    try:
        for index, (image, mask) in enumerate(zip(images, masks)):
            image_bytes = await image.read()
            mask_bytes = await mask.read()
            if len(image_bytes) > max_bytes or len(mask_bytes) > max_bytes:
                raise ResourceLimitError("uploaded file exceeds the size limit")
            (upload_dir / f"{index:03d}.image").write_bytes(image_bytes)
            (upload_dir / f"{index:03d}.mask").write_bytes(mask_bytes)
            items.append(
                BatchItemState(
                    batch_id=batch_id,
                    item_index=index,
                    image_name=Path(image.filename or f"{index}.png").name,
                    status="pending",
                )
            )
    except Exception:
        shutil.rmtree(upload_dir, ignore_errors=True)
        raise

    batch = BatchJobState(
        batch_id=batch_id,
        status="queued",
        total=len(items),
        done=0,
        failed=0,
        created_at=datetime.now(timezone.utc),
        note=note or None,
    )
    container.batch_state.create(batch, items)

    chunk_size = container.config.system.inference.chunk_size
    chunks = [
        list(range(start, min(start + chunk_size, len(items))))
        for start in range(0, len(items), chunk_size)
    ]
    try:
        chord(
            group(batch_chunk.s(batch_id, indexes) for indexes in chunks),
            batch_finalize.s(batch_id),
        ).apply_async()
    except Exception as exc:
        raise ModelUnavailableError(
            "task broker unavailable", detail=str(exc)
        ) from exc
    return BatchCreated(
        batch_id=batch_id,
        status="queued",
        total=len(items),
        events_url=f"/api/v1/batches/{batch_id}/events",
    )


@router.get("/batches/{batch_id}", response_model=BatchDetail)
def get_batch(
    batch_id: str, container: Container = Depends(get_container)
) -> BatchDetail:
    validate_job_id(batch_id)
    batch = container.batch_state.get(batch_id)
    if batch is None:
        raise NotFoundError(f"batch not found: {batch_id}")
    return BatchDetail.from_state(batch, container.batch_state.items(batch_id))


@router.post("/batches/{batch_id}/cancel", response_model=CancelResponse)
def cancel_batch(
    batch_id: str, container: Container = Depends(get_container)
) -> CancelResponse:
    validate_job_id(batch_id)
    batch = container.batch_state.get(batch_id)
    if batch is None:
        raise NotFoundError(f"batch not found: {batch_id}")
    container.cancellation.cancel(batch_id)
    return CancelResponse(
        batch_id=batch_id, status=batch.status, cancel_requested=True
    )


@router.get("/batches/{batch_id}/report")
def get_report(
    batch_id: str, container: Container = Depends(get_container)
) -> FileResponse:
    validate_job_id(batch_id)
    batch = container.batch_state.get(batch_id)
    if batch is None:
        raise NotFoundError(f"batch not found: {batch_id}")
    if not batch.report_path or not Path(batch.report_path).is_file():
        raise NotFoundError("batch report is not ready")
    return FileResponse(
        batch.report_path,
        media_type="text/csv",
        filename=f"batch_{batch_id}.csv",
    )


@router.get("/batches/{batch_id}/events")
async def batch_events(
    batch_id: str, request: Request, container: Container = Depends(get_container)
) -> StreamingResponse:
    validate_job_id(batch_id)
    if container.batch_state.get(batch_id) is None:
        raise NotFoundError(f"batch not found: {batch_id}")
    last_event_id = request.headers.get("last-event-id", "")
    cursor = last_event_id if _STREAM_ID_PATTERN.fullmatch(last_event_id) else "0-0"
    stream = batch_stream(batch_id)
    aredis = request.app.state.aredis

    async def event_stream():
        nonlocal cursor
        while True:
            if await request.is_disconnected():
                return
            entries = await aredis.xread({stream: cursor}, count=100, block=_SSE_BLOCK_MS)
            if not entries:
                yield ": ping\n\n"  # keep-alive，防代理超时断连
                continue
            for _key, messages in entries:
                for entry_id, fields in messages:
                    cursor = entry_id
                    yield (
                        f"id: {entry_id}\n"
                        f"event: {fields['type']}\n"
                        f"data: {fields['data']}\n\n"
                    )
                    if fields["type"] == "finished":
                        return

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # 禁 nginx 缓冲
            "Connection": "keep-alive",
        },
    )
