"""SqlAlchemyBatchRepository: BatchStateRepository 端口的 SQLAlchemy 适配器。

MySQL/SQLite 通用（同一套 ORM）。事务边界 = 单 item（record_item 内
更新项 + 重算计数 + queued→running 一次提交）；单 worker 串行消费，
无并发写竞争。report_path 与 history 一致存相对路径。
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import Engine, func, select
from sqlalchemy.orm import Session

from hybrid_inpaint.domain.entities import BatchItemState, BatchJobState
from hybrid_inpaint.domain.errors import NotFoundError, ValidationError
from hybrid_inpaint.domain.services import validate_job_id
from hybrid_inpaint.infrastructure.sqlalchemy_models import BatchItemRow, BatchJobRow

_TERMINAL_ITEM = ("ok", "failed", "cancelled", "skipped")


class SqlAlchemyBatchRepository:
    def __init__(self, engine: Engine, paths_base: Path) -> None:
        self._engine = engine
        self._base = Path(paths_base)

    def _to_relative(self, path: str) -> str:
        candidate = Path(path)
        try:
            return candidate.relative_to(self._base).as_posix()
        except ValueError:
            return candidate.as_posix()

    def _to_absolute(self, stored: str) -> str:
        candidate = Path(stored)
        return str(candidate if candidate.is_absolute() else self._base / candidate)

    def _row_to_batch(self, row: BatchJobRow) -> BatchJobState:
        return BatchJobState(
            batch_id=row.batch_id,
            status=row.status,
            total=row.total,
            done=row.done,
            failed=row.failed,
            created_at=datetime.fromisoformat(row.created_at),
            finished_at=(
                datetime.fromisoformat(row.finished_at) if row.finished_at else None
            ),
            report_path=(
                self._to_absolute(row.report_path) if row.report_path else None
            ),
            note=row.note,
        )

    def _row_to_item(self, row: BatchItemRow) -> BatchItemState:
        return BatchItemState(
            batch_id=row.batch_id,
            item_index=row.item_index,
            image_name=row.image_name,
            status=row.status,
            job_id=row.job_id,
            error_code=row.error_code,
            latency_ms=row.latency_ms,
            mask_ratio=row.mask_ratio,
        )

    def create(self, batch: BatchJobState, items: Sequence[BatchItemState]) -> None:
        validate_job_id(batch.batch_id)
        if len(items) != batch.total:
            raise ValidationError("items count must equal batch.total")
        job_row = BatchJobRow(
            batch_id=batch.batch_id,
            status=batch.status,
            total=batch.total,
            done=batch.done,
            failed=batch.failed,
            created_at=batch.created_at.isoformat(),
            finished_at=(
                batch.finished_at.isoformat() if batch.finished_at else None
            ),
            report_path=(
                self._to_relative(batch.report_path) if batch.report_path else None
            ),
            note=batch.note,
        )
        item_rows = [
            BatchItemRow(
                batch_id=item.batch_id,
                item_index=item.item_index,
                image_name=item.image_name,
                status=item.status,
                job_id=item.job_id,
                error_code=item.error_code,
                latency_ms=item.latency_ms,
                mask_ratio=item.mask_ratio,
            )
            for item in items
        ]
        with Session(self._engine) as session, session.begin():
            session.add(job_row)
            session.add_all(item_rows)

    def get(self, batch_id: str) -> BatchJobState | None:
        with Session(self._engine) as session:
            row = session.scalar(
                select(BatchJobRow).where(BatchJobRow.batch_id == batch_id)
            )
        return self._row_to_batch(row) if row is not None else None

    def items(self, batch_id: str) -> list[BatchItemState]:
        statement = (
            select(BatchItemRow)
            .where(BatchItemRow.batch_id == batch_id)
            .order_by(BatchItemRow.item_index)
        )
        with Session(self._engine) as session:
            rows = session.scalars(statement).all()
        return [self._row_to_item(row) for row in rows]

    def record_item(
        self,
        batch_id: str,
        item_index: int,
        *,
        status: str,
        job_id: str | None = None,
        error_code: str | None = None,
        latency_ms: float | None = None,
        mask_ratio: float | None = None,
    ) -> None:
        if status not in _TERMINAL_ITEM:
            raise ValidationError(f"invalid terminal item status: {status}")
        with Session(self._engine) as session, session.begin():
            item = session.scalar(
                select(BatchItemRow).where(
                    BatchItemRow.batch_id == batch_id,
                    BatchItemRow.item_index == item_index,
                )
            )
            if item is None:
                raise NotFoundError(f"batch item not found: {batch_id}#{item_index}")
            item.status = status
            item.job_id = job_id
            item.error_code = error_code
            item.latency_ms = latency_ms
            item.mask_ratio = mask_ratio

            done = session.scalar(
                select(func.count())
                .select_from(BatchItemRow)
                .where(BatchItemRow.batch_id == batch_id, BatchItemRow.status == "ok")
            )
            failed = session.scalar(
                select(func.count())
                .select_from(BatchItemRow)
                .where(
                    BatchItemRow.batch_id == batch_id,
                    BatchItemRow.status == "failed",
                )
            )
            batch = session.scalar(
                select(BatchJobRow).where(BatchJobRow.batch_id == batch_id)
            )
            if batch is None:
                raise NotFoundError(f"batch not found: {batch_id}")
            batch.done = int(done or 0)
            batch.failed = int(failed or 0)
            if batch.status == "queued":
                batch.status = "running"

    def finish(
        self, batch_id: str, *, status: str, report_path: str | None = None
    ) -> None:
        with Session(self._engine) as session, session.begin():
            batch = session.scalar(
                select(BatchJobRow).where(BatchJobRow.batch_id == batch_id)
            )
            if batch is None:
                raise NotFoundError(f"batch not found: {batch_id}")
            batch.status = status
            batch.finished_at = datetime.now(timezone.utc).isoformat()
            if report_path is not None:
                batch.report_path = self._to_relative(report_path)
