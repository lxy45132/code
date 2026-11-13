"""批量任务与队列任务 DTO。"""

from __future__ import annotations

from pydantic import BaseModel

from hybrid_inpaint.domain.entities import BatchItemState, BatchJobState


class BatchItemDTO(BaseModel):
    item_index: int
    image_name: str
    status: str
    job_id: str | None
    error_code: str | None
    latency_ms: float | None
    mask_ratio: float | None
    output_url: str | None

    @classmethod
    def from_state(cls, item: BatchItemState) -> "BatchItemDTO":
        return cls(
            item_index=item.item_index,
            image_name=item.image_name,
            status=item.status,
            job_id=item.job_id,
            error_code=item.error_code,
            latency_ms=item.latency_ms,
            mask_ratio=item.mask_ratio,
            output_url=(
                f"/api/v1/outputs/{item.job_id}" if item.job_id else None
            ),
        )


class BatchDetail(BaseModel):
    batch_id: str
    status: str
    total: int
    done: int
    failed: int
    created_at: str
    finished_at: str | None
    note: str | None
    report_url: str | None
    items: list[BatchItemDTO]

    @classmethod
    def from_state(
        cls, batch: BatchJobState, items: list[BatchItemState]
    ) -> "BatchDetail":
        return cls(
            batch_id=batch.batch_id,
            status=batch.status,
            total=batch.total,
            done=batch.done,
            failed=batch.failed,
            created_at=batch.created_at.isoformat(),
            finished_at=(
                batch.finished_at.isoformat() if batch.finished_at else None
            ),
            note=batch.note,
            report_url=(
                f"/api/v1/batches/{batch.batch_id}/report"
                if batch.report_path
                else None
            ),
            items=[BatchItemDTO.from_state(item) for item in items],
        )


class BatchCreated(BaseModel):
    batch_id: str
    status: str
    total: int
    events_url: str


class CancelResponse(BaseModel):
    batch_id: str
    status: str
    cancel_requested: bool


class TaskStatus(BaseModel):
    task_id: str
    status: str  # PENDING / STARTED / RETRY / SUCCESS / FAILURE
    result: dict | None = None
