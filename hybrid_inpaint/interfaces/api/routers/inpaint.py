"""POST /inpaint：单图修复（multipart image/mask/note）。

inline 模式同步返回结果；阻塞用例经 run_in_threadpool 执行，
不占事件循环（GPU 串行由 SerializedEngine 兜底，FIFO 排队）。
worker 模式（队列化）在 C7d 落地。
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, File, Form, UploadFile
from fastapi.concurrency import run_in_threadpool

from hybrid_inpaint.bootstrap import Container
from hybrid_inpaint.domain.entities import InpaintRequest
from hybrid_inpaint.domain.errors import ModelUnavailableError, ResourceLimitError
from hybrid_inpaint.interfaces.api.dependencies import get_container
from hybrid_inpaint.interfaces.api.schemas.inpaint import InpaintResponse

router = APIRouter(tags=["inpaint"])


@router.post("/inpaint", response_model=InpaintResponse)
async def inpaint(
    image: UploadFile = File(...),
    mask: UploadFile = File(...),
    note: str = Form(""),
    container: Container = Depends(get_container),
) -> InpaintResponse:
    if container.config.system.inference.mode == "worker":
        raise ModelUnavailableError("worker inference mode is not available yet")

    image_bytes = await image.read()
    mask_bytes = await mask.read()
    if len(mask_bytes) > container.limits.max_file_mb * 1024 * 1024:
        raise ResourceLimitError("mask file exceeds the size limit")

    decoded = container.codec.decode(image_bytes, container.limits)
    decoded_mask = container.codec.decode_mask(
        mask_bytes, height=decoded.height, width=decoded.width
    )
    request = InpaintRequest(decoded, decoded_mask, mode="web", note=note)
    result = await run_in_threadpool(container.inpaint.execute, request)
    return InpaintResponse(
        job_id=result.job_id,
        output_url=f"/api/v1/outputs/{result.job_id}",
        latency_ms=result.latency_ms,
        model_version=result.model_version,
        mask_ratio=result.mask_ratio,
        input_size=(decoded.width, decoded.height),
        warnings=list(result.warnings),
    )
