"""worker 模式单图任务（docs/03 §4）：api 落盘上传 → 投队列 → 轮询取结果。

输入错误不重试（直接返回 failed 结果体）；瞬时错误指数退避。
路径必须位于 outputs/tmp 下（防 broker 注入任意文件读取）。
"""

from __future__ import annotations

import logging
from pathlib import Path

from hybrid_inpaint.domain.entities import InpaintRequest
from hybrid_inpaint.domain.errors import ResourceLimitError, ValidationError
from hybrid_inpaint.interfaces.api.celery_app import celery_app
from hybrid_inpaint.interfaces.api.tasks.common import get_worker_container

logger = logging.getLogger("hybrid_inpaint.tasks")


def _check_upload_path(container, raw: str) -> Path:
    path = Path(raw).resolve()
    tmp_root = (container.config.system.output_dir.parent / "tmp").resolve()
    if not path.is_relative_to(tmp_root):
        raise ValidationError("task input path must stay under outputs/tmp")
    return path


@celery_app.task(bind=True, name="hybrid.inpaint_one")
def inpaint_one(self, image_path: str, mask_path: str, note: str = "") -> dict:
    container = get_worker_container()
    try:
        image_file = _check_upload_path(container, image_path)
        mask_file = _check_upload_path(container, mask_path)
        image = container.codec.decode(image_file.read_bytes(), container.limits)
        mask = container.codec.decode_mask(
            mask_file.read_bytes(), height=image.height, width=image.width
        )
        result = container.inpaint.execute(
            InpaintRequest(image, mask, mode="web", note=note)
        )
    except (ValidationError, ResourceLimitError) as exc:
        return {"status": "failed", "error_code": exc.code, "message": exc.user_message}
    except Exception as exc:
        limit = container.config.system.celery.retries
        if self.request.retries >= limit:
            logger.error("inpaint_one exhausted retries: %s", exc)
            return {
                "status": "failed",
                "error_code": getattr(exc, "code", "internal_error"),
                "message": str(exc),
            }
        raise self.retry(exc=exc, countdown=2**self.request.retries, max_retries=limit)
    return {
        "status": "ok",
        "job_id": result.job_id,
        "output_url": f"/api/v1/outputs/{result.job_id}",
        "latency_ms": result.latency_ms,
        "model_version": result.model_version,
        "mask_ratio": result.mask_ratio,
    }
