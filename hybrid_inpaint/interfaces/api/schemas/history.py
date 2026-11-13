"""历史列表/详情 DTO。"""

from __future__ import annotations

from pydantic import BaseModel

from hybrid_inpaint.domain.entities import HistoryRecord


class HistoryItem(BaseModel):
    job_id: str
    created_at: str  # ISO 8601
    mode: str
    batch_id: str | None
    output_url: str
    model_version: str
    width: int
    height: int
    mask_ratio: float
    latency_ms: float
    note: str | None

    @classmethod
    def from_record(cls, record: HistoryRecord) -> "HistoryItem":
        return cls(
            job_id=record.job_id,
            created_at=record.created_at.isoformat(),
            mode=record.mode,
            batch_id=record.batch_id,
            output_url=f"/api/v1/outputs/{record.job_id}",
            model_version=record.model_version,
            width=record.width,
            height=record.height,
            mask_ratio=record.mask_ratio,
            latency_ms=record.latency_ms,
            note=record.note,
        )


class HistoryPage(BaseModel):
    items: list[HistoryItem]
    limit: int
    offset: int
