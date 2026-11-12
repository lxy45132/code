from __future__ import annotations

import csv
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pytest

from hybrid_inpaint.domain.entities import HistoryRecord
from hybrid_inpaint.domain.errors import ValidationError
from hybrid_inpaint.infrastructure.sqlite_history import SqliteHistoryRepository


def _record(
    tmp_path: Path, job_id: str, minute: int = 0, note: str | None = None
) -> HistoryRecord:
    return HistoryRecord(
        job_id=job_id,
        created_at=datetime(2026, 11, 12, 12, minute, tzinfo=timezone.utc),
        mode="cli",
        batch_id=None,
        input_path=str(tmp_path / "history" / job_id / "input.png"),
        mask_path=str(tmp_path / "history" / job_id / "mask.png"),
        output_path=str(tmp_path / "history" / job_id / "output.png"),
        model_version="v1",
        width=8,
        height=8,
        mask_ratio=0.25,
        latency_ms=12.5,
        note=note,
    )


def _repo(tmp_path: Path) -> SqliteHistoryRepository:
    return SqliteHistoryRepository(tmp_path / "history.sqlite", paths_base=tmp_path)


def test_add_get_roundtrip_and_relative_storage(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    record = _record(tmp_path, format(1, "032x"))
    repo.add(record)

    loaded = repo.get(record.job_id)
    assert loaded is not None
    assert loaded.created_at == record.created_at
    assert loaded.output_path == record.output_path  # 拼回绝对路径

    conn = sqlite3.connect(tmp_path / "history.sqlite")
    stored = conn.execute("SELECT output_path FROM jobs").fetchone()[0]
    conn.close()
    assert stored == f"history/{record.job_id}/output.png"  # 库里是相对路径


def test_list_order_and_pagination(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    repo.add(_record(tmp_path, format(1, "032x"), minute=0))
    repo.add(_record(tmp_path, format(2, "032x"), minute=1))
    repo.add(_record(tmp_path, format(3, "032x"), minute=2))

    page = repo.list(limit=2, offset=0)
    assert [r.job_id for r in page] == [format(3, "032x"), format(2, "032x")]
    assert repo.list(limit=2, offset=2)[0].job_id == format(1, "032x")


def test_delete_and_missing(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    repo.add(_record(tmp_path, format(1, "032x")))
    repo.delete(format(1, "032x"))
    assert repo.get(format(1, "032x")) is None
    repo.delete(format(1, "032x"))  # 幂等


def test_invalid_and_duplicate_job_id(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    with pytest.raises(ValidationError):
        repo.add(_record(tmp_path, "../evil"))
    repo.add(_record(tmp_path, format(1, "032x")))
    with pytest.raises(ValidationError):
        repo.add(_record(tmp_path, format(1, "032x")))


def test_sql_injection_note_is_stored_safely(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    evil = "'); DROP TABLE jobs; --"
    repo.add(_record(tmp_path, format(1, "032x"), note=evil))
    assert repo.get(format(1, "032x")).note == evil
    assert repo.list(10, 0)  # 表仍在


def test_export_csv_columns(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    repo.add(_record(tmp_path, format(1, "032x")))
    target = tmp_path / "export.csv"
    repo.export_csv(target)
    with target.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    assert rows[0] == [
        "job_id", "created_at", "mode", "input", "mask", "output",
        "model_version", "width", "height", "mask_ratio", "latency_ms", "note",
    ]
    assert len(rows) == 2


def test_persistence_across_instances(tmp_path: Path) -> None:
    _repo(tmp_path).add(_record(tmp_path, format(1, "032x")))
    assert _repo(tmp_path).get(format(1, "032x")) is not None