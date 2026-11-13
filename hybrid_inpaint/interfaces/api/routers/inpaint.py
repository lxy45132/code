"""POST /inpaint：单图修复（multipart image/mask/note）。

inline 模式同步返回结果；阻塞用例经 run_in_threadpool 执行，
不占事件循环（GPU 串行由 SerializedEngine 兜底，FIFO 排队）。
worker 模式落盘上传后投 inpaint_one 任务，202 返回 task_id（03 文档 §4）。
"""

from __future__ import annotations

from uuid import uuid4

from fastapi import APIRouter, Depends, File, Form, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse

from hybrid_inpaint.bootstrap import Container
from hybrid_inpaint.domain.entities import InpaintRequest
from hybrid_inpaint.domain.errors import ModelUnavailableError, ResourceLimitError
from hybrid_inpaint.interfaces.api.caching import bump_history_cache
from hybrid_inpaint.interfaces.api.dependencies import get_container
from hybrid_inpaint.interfaces.api.schemas.inpaint import InpaintResponse
from hybrid_inpaint.interfaces.api.tasks.inpaint_one import inpaint_one

router = APIRouter(tags=["inpaint"])


@router.post("/inpaint", response_model=None)
async def inpaint(
    request: Request,
    image: UploadFile = File(...),
    mask: UploadFile = File(...),
    note: str = Form(""),
    container: Container = Depends(get_container),
):
    image_bytes = await image.read()
    mask_bytes = await mask.read()
    max_bytes = container.limits.max_file_mb * 1024 * 1024
    if len(image_bytes) > max_bytes or len(mask_bytes) > max_bytes:
        raise ResourceLimitError("uploaded file exceeds the size limit")

    if container.config.system.inference.mode == "worker":
        directory = (
            container.config.system.output_dir.parent / "tmp" / "single" / uuid4().hex
        )
        directory.mkdir(parents=True, exist_ok=True)
        image_path = directory / "image"
        mask_path = directory / "mask"
        image_path.write_bytes(image_bytes)
        mask_path.write_bytes(mask_bytes)
        try:
            async_result = inpaint_one.apply_async(
                args=[str(image_path), str(mask_path), note]
            )
        except Exception as exc:
            raise ModelUnavailableError(
                "task broker unavailable", detail=str(exc)
            ) from exc
        return JSONResponse(
            status_code=202,
            content={
                "task_id": async_result.id,
                "status": "queued",
                "task_url": f"/api/v1/tasks/{async_result.id}",
            },
        )

    decoded = container.codec.decode(image_bytes, container.limits)
    decoded_mask = container.codec.decode_mask(
        mask_bytes, height=decoded.height, width=decoded.width
    )
    inpaint_request = InpaintRequest(decoded, decoded_mask, mode="web", note=note)
    result = await run_in_threadpool(container.inpaint.execute, inpaint_request)
    bump_history_cache(container)
    return InpaintResponse(
        job_id=result.job_id,
        output_url=f"/api/v1/outputs/{result.job_id}",
        latency_ms=result.latency_ms,
        model_version=result.model_version,
        mask_ratio=result.mask_ratio,
        input_size=(decoded.width, decoded.height),
        warnings=list(result.warnings),
    )
