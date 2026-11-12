from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from hybrid_inpaint.application.history import HistoryService
from hybrid_inpaint.domain.entities import HistoryRecord
from hybrid_inpaint.domain.errors import NotFoundError
from tests.application.fakes import InMemoryArtifacts, InMemoryHistory


def _service(tmp_path: Path):
    history = InMemoryHistory()
    artifacts = InMemoryArtifacts(tmp_path)
    return HistoryService(history, artifacts), history, artifacts


def _record(job_id: str) -> HistoryRecord:
    return HistoryRecord(
        job_id=job_id,
        created_at=datetime(2026, 11, 12, tzinfo=timezone.utc),
        mode="cli",
        batch_id=None,
        input_path="in.png",
        mask_path="mask.png",
        output_path="out.png",
        model_version="v",
        width=4,
        height=4,
        mask_ratio=0.25,
        latency_ms=1.0,
        note=None,
    )


def test_get_missing_raises_not_found(tmp_path: Path) -> None:
    service, _, _ = _service(tmp_path)
    with pytest.raises(NotFoundError):
        service.get("missing")


def test_delete_removes_row_and_optionally_files(tmp_path: Path) -> None:
    service, history, artifacts = _service(tmp_path)
    history.add(_record(format(1, "032x")))
    artifacts.save(format(1, "032x"), None, None, None)  # type: ignore[arg-type]

    service.delete(format(1, "032x"), delete_files=True)

    assert history.records == {}
    assert artifacts.deleted == [format(1, "032x")]


def test_delete_default_keeps_files(tmp_path: Path) -> None:
    service, history, artifacts = _service(tmp_path)
    history.add(_record(format(2, "032x")))
    service.delete(format(2, "032x"))
    assert artifacts.deleted == []


def test_list_and_export_delegate(tmp_path: Path) -> None:
    service, history, _ = _service(tmp_path)
    history.add(_record(format(3, "032x")))
    assert len(service.list(limit=10, offset=0)) == 1
    target = tmp_path / "history.csv"
    service.export_csv(target)
    assert target.is_file()