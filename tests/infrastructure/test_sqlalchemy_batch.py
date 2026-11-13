"""SqlAlchemyBatchRepository 契约测试（sqlite 引擎，MySQL 同套 ORM）。"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest
from sqlalchemy import create_engine

from hybrid_inpaint.domain.entities import BatchItemState, BatchJobState
from hybrid_inpaint.domain.errors import NotFoundError, ValidationError
from hybrid_inpaint.infrastructure.sqlalchemy_batch import SqlAlchemyBatchRepository
from hybrid_inpaint.infrastructure.sqlalchemy_history import ensure_schema


def _repo(tmp_path: Path) -> SqlAlchemyBatchRepository:
    engine = create_engine(f"sqlite:///{(tmp_path / 'batch.db').as_posix()}")
    ensure_schema(engine)
    return SqlAlchemyBatchRepository(engine, paths_base=tmp_path)


def _batch(batch_id: str, total: int = 2) -> BatchJobState:
    return BatchJobState(
        batch_id=batch_id,
        status="queued",
        total=total,
        done=0,
        failed=0,
        created_at=datetime(2026, 11, 13, 12, 0, tzinfo=timezone.utc),
        note="契约测试",
    )


def _items(batch_id: str, total: int = 2) -> list[BatchItemState]:
    return [
        BatchItemState(
            batch_id=batch_id, item_index=i, image_name=f"{i}.png", status="pending"
        )
        for i in range(total)
    ]


def test_create_get_items_roundtrip(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    batch_id = format(1, "032x")
    repo.create(_batch(batch_id), _items(batch_id))

    loaded = repo.get(batch_id)
    assert loaded is not None
    assert loaded.status == "queued"
    assert loaded.total == 2
    assert loaded.note == "契约测试"
    assert loaded.created_at == datetime(2026, 11, 13, 12, 0, tzinfo=timezone.utc)

    items = repo.items(batch_id)
    assert [i.item_index for i in items] == [0, 1]
    assert all(i.status == "pending" for i in items)
    assert repo.get(format(2, "032x")) is None


def test_record_item_updates_counters_and_status(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    batch_id = format(1, "032x")
    repo.create(_batch(batch_id), _items(batch_id))

    repo.record_item(batch_id, 0, status="ok", job_id=format(9, "032x"), latency_ms=5.0, mask_ratio=0.2)
    batch = repo.get(batch_id)
    assert batch.status == "running"  # queued → running
    assert batch.done == 1 and batch.failed == 0

    repo.record_item(batch_id, 1, status="failed", error_code="validation_error")
    batch = repo.get(batch_id)
    assert batch.done == 1 and batch.failed == 1
    assert batch.status == "running"  # 终态由 finish 决定

    item = repo.items(batch_id)[0]
    assert item.job_id == format(9, "032x")
    assert item.latency_ms == 5.0


def test_finish_stores_relative_report_path(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    batch_id = format(1, "032x")
    repo.create(_batch(batch_id), _items(batch_id))
    report = tmp_path / "batches" / batch_id / "report.csv"
    repo.finish(batch_id, status="succeeded", report_path=str(report))

    batch = repo.get(batch_id)
    assert batch.status == "succeeded"
    assert batch.finished_at is not None
    assert batch.report_path == str(report)  # 读回绝对路径

    from sqlalchemy import select

    from hybrid_inpaint.infrastructure.sqlalchemy_models import BatchJobRow

    with repo._engine.connect() as connection:  # noqa: SLF001 - 契约检查
        stored = connection.scalar(select(BatchJobRow.report_path))
    assert stored == f"batches/{batch_id}/report.csv"  # 库里相对路径


def test_invalid_inputs_are_rejected(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    batch_id = format(1, "032x")
    with pytest.raises(ValidationError):
        repo.create(_batch("../evil"), _items("../evil"))
    with pytest.raises(ValidationError):
        repo.create(_batch(batch_id, total=3), _items(batch_id, total=2))
    repo.create(_batch(batch_id), _items(batch_id))
    with pytest.raises(ValidationError):
        repo.record_item(batch_id, 0, status="pending")  # 非终态
    with pytest.raises(NotFoundError):
        repo.record_item(batch_id, 99, status="ok")
    with pytest.raises(NotFoundError):
        repo.finish(format(2, "032x"), status="succeeded")
