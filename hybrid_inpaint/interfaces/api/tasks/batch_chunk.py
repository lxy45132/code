"""chunk 级批量任务（docs/03 §5）。

拓扑：chord(group(batch_chunk...), batch_finalize)。
- 幂等：重试时跳过已终态（ok/failed/cancelled/skipped）的 item；
- 取消：执行前与项间检查 Redis 标志，未执行项落 cancelled；
- 分类重试：输入错误（ValidationError/ResourceLimitError）不重试直接
  failed；瞬时错误指数退避（2^retries 秒），上限 config.celery.retries；
- 凑批：chunk 内有效项一次 execute_many（引擎按尺寸分组 batch 前向）。
"""

from __future__ import annotations

import logging
from pathlib import Path

from hybrid_inpaint.bootstrap import Container
from hybrid_inpaint.domain.entities import InpaintRequest
from hybrid_inpaint.domain.errors import ResourceLimitError, ValidationError
from hybrid_inpaint.interfaces.api.celery_app import celery_app
from hybrid_inpaint.interfaces.api.tasks.common import get_worker_container

logger = logging.getLogger("hybrid_inpaint.tasks")

_TERMINAL = ("ok", "failed", "cancelled", "skipped")


def batch_upload_dir(container: Container, batch_id: str) -> Path:
    return container.config.system.output_dir.parent / "tmp" / batch_id


def batch_stream(batch_id: str) -> str:
    return f"hybrid:batch:{batch_id}"


def _record_ok(container, batch_id, index, result, stream) -> None:
    container.batch_state.record_item(
        batch_id,
        index,
        status="ok",
        job_id=result.job_id,
        latency_ms=result.latency_ms,
        mask_ratio=result.mask_ratio,
    )
    container.events.publish(
        stream,
        "item_done",
        {"index": index, "job_id": result.job_id, "latency_ms": result.latency_ms},
    )


def _record_failed(container, batch_id, index, error_code, stream) -> None:
    container.batch_state.record_item(
        batch_id, index, status="failed", error_code=error_code
    )
    container.events.publish(
        stream, "item_failed", {"index": index, "error_code": error_code}
    )


def _run_chunk(batch_id: str, indexes: list[int]) -> dict:
    container = get_worker_container()
    state = container.batch_state
    batch = state.get(batch_id)
    if batch is None:
        return {"batch_id": batch_id, "processed": 0}
    token = container.cancellation.token(batch_id)
    items = {item.item_index: item for item in state.items(batch_id)}
    stream = batch_stream(batch_id)
    upload_dir = batch_upload_dir(container, batch_id)

    requests: list[tuple[int, InpaintRequest]] = []
    for index in indexes:
        item = items[index]
        if item.status in _TERMINAL:  # 幂等：重试跳过已终态
            continue
        if token.cancelled:
            state.record_item(batch_id, index, status="cancelled")
            container.events.publish(stream, "cancelled", {"index": index})
            continue
        container.events.publish(
            stream, "item_started", {"index": index, "image": item.image_name}
        )
        try:
            image_bytes = (upload_dir / f"{index:03d}.image").read_bytes()
            mask_bytes = (upload_dir / f"{index:03d}.mask").read_bytes()
            image = container.codec.decode(image_bytes, container.limits)
            mask = container.codec.decode_mask(
                mask_bytes, height=image.height, width=image.width
            )
            requests.append(
                (
                    index,
                    InpaintRequest(
                        image, mask, mode="web", note=batch.note or "", batch_id=batch_id
                    ),
                )
            )
        except (ValidationError, ResourceLimitError) as exc:
            _record_failed(container, batch_id, index, exc.code, stream)
        except OSError as exc:
            logger.warning("batch upload read failed batch=%s index=%s: %s", batch_id, index, exc)
            _record_failed(container, batch_id, index, "internal_error", stream)

    if not requests:
        return {"batch_id": batch_id, "processed": 0}
    if token.cancelled:
        for index, _ in requests:
            state.record_item(batch_id, index, status="cancelled")
            container.events.publish(stream, "cancelled", {"index": index})
        return {"batch_id": batch_id, "processed": 0}

    try:
        results = container.inpaint.execute_many([request for _, request in requests])
    except (ValidationError, ResourceLimitError):
        # 整批校验失败（如某项全洞）→ 退化逐张隔离坏项；瞬时错误向上抛触发重试
        for index, request in requests:
            try:
                result = container.inpaint.execute(request)
            except (ValidationError, ResourceLimitError) as exc:
                _record_failed(container, batch_id, index, exc.code, stream)
                continue
            _record_ok(container, batch_id, index, result, stream)
        return {"batch_id": batch_id, "processed": len(requests)}

    for (index, _), result in zip(requests, results):
        _record_ok(container, batch_id, index, result, stream)
    return {"batch_id": batch_id, "processed": len(requests)}


def _fail_pending(batch_id: str, indexes: list[int], exc: Exception) -> None:
    """重试耗尽：仍处 pending 的项落 failed（终态项不动）。"""
    container = get_worker_container()
    state = container.batch_state
    stream = batch_stream(batch_id)
    error_code = getattr(exc, "code", "internal_error")
    items = {item.item_index: item for item in state.items(batch_id)}
    for index in indexes:
        if items[index].status not in _TERMINAL:
            _record_failed(container, batch_id, index, error_code, stream)


@celery_app.task(bind=True, name="hybrid.batch_chunk")
def batch_chunk(self, batch_id: str, indexes: list[int]) -> dict:
    try:
        return _run_chunk(batch_id, indexes)
    except Exception as exc:
        limit = get_worker_container().config.system.celery.retries
        if self.request.retries >= limit:
            logger.error("batch_chunk exhausted retries batch=%s: %s", batch_id, exc)
            _fail_pending(batch_id, indexes, exc)
            return {"batch_id": batch_id, "processed": 0, "failed": True}
        raise self.retry(exc=exc, countdown=2**self.request.retries, max_retries=limit)
