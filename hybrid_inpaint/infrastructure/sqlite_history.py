"""SqliteHistoryRepository: WAL, parameterized SQL, relative path storage."""

from __future__ import annotations

import csv
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Iterator

from hybrid_inpaint.domain.entities import HistoryRecord
from hybrid_inpaint.domain.errors import InternalError, ValidationError
from hybrid_inpaint.domain.services import validate_job_id

SCHEMA_VERSION = 1

_CSV_COLUMNS = (
    "job_id", "created_at", "mode", "input", "mask", "output",
    "model_version", "width", "height", "mask_ratio", "latency_ms", "note",
)

_SELECT = (
    "SELECT job_id, created_at, mode, batch_id, input_path, mask_path, "
    "output_path, model_version, image_width, image_height, mask_ratio, "
    "latency_ms, note FROM jobs"
)


class SqliteHistoryRepository:
    def __init__(self, db_path: Path, paths_base: Path) -> None:
        self._db_path = Path(db_path)
        self._base = Path(paths_base)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS schema_meta (version INTEGER NOT NULL)"
            )
            conn.execute(
                """CREATE TABLE IF NOT EXISTS jobs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id TEXT NOT NULL UNIQUE,
                    created_at TEXT NOT NULL,
                    mode TEXT NOT NULL CHECK(mode IN ('web','gui','cli')),
                    batch_id TEXT,
                    input_path TEXT NOT NULL,
                    mask_path TEXT NOT NULL,
                    output_path TEXT NOT NULL,
                    model_version TEXT NOT NULL,
                    image_width INTEGER NOT NULL,
                    image_height INTEGER NOT NULL,
                    mask_ratio REAL NOT NULL,
                    latency_ms REAL NOT NULL,
                    note TEXT
                )"""
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_jobs_created_at ON jobs(created_at DESC)"
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_jobs_batch ON jobs(batch_id)")
            row = conn.execute("SELECT version FROM schema_meta").fetchone()
            if row is None:
                conn.execute(
                    "INSERT INTO schema_meta (version) VALUES (?)", (SCHEMA_VERSION,)
                )
            elif row[0] != SCHEMA_VERSION:
                raise InternalError(f"unsupported history schema version: {row[0]}")

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self._db_path)
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=5000")
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _to_relative(self, path: str) -> str:
        candidate = Path(path)
        try:
            return candidate.relative_to(self._base).as_posix()
        except ValueError:
            return candidate.as_posix()

    def _to_absolute(self, stored: str) -> str:
        candidate = Path(stored)
        return str(candidate if candidate.is_absolute() else self._base / candidate)

    def _row_to_record(self, row: tuple) -> HistoryRecord:
        return HistoryRecord(
            job_id=row[0],
            created_at=datetime.fromisoformat(row[1]),
            mode=row[2],
            batch_id=row[3],
            input_path=self._to_absolute(row[4]),
            mask_path=self._to_absolute(row[5]),
            output_path=self._to_absolute(row[6]),
            model_version=row[7],
            width=row[8],
            height=row[9],
            mask_ratio=row[10],
            latency_ms=row[11],
            note=row[12],
        )

    def add(self, record: HistoryRecord) -> None:
        validate_job_id(record.job_id)
        try:
            with self._connect() as conn:
                conn.execute(
                    "INSERT INTO jobs (job_id, created_at, mode, batch_id, input_path, "
                    "mask_path, output_path, model_version, image_width, image_height, "
                    "mask_ratio, latency_ms, note) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        record.job_id,
                        record.created_at.isoformat(),
                        record.mode,
                        record.batch_id,
                        self._to_relative(record.input_path),
                        self._to_relative(record.mask_path),
                        self._to_relative(record.output_path),
                        record.model_version,
                        record.width,
                        record.height,
                        record.mask_ratio,
                        record.latency_ms,
                        record.note,
                    ),
                )
        except sqlite3.IntegrityError as exc:
            raise ValidationError(
                f"duplicate or invalid job_id: {record.job_id}"
            ) from exc

    def list(self, limit: int, offset: int) -> list[HistoryRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                _SELECT + " ORDER BY created_at DESC, id DESC LIMIT ? OFFSET ?",
                (limit, offset),
            ).fetchall()
        return [self._row_to_record(row) for row in rows]

    def get(self, job_id: str) -> HistoryRecord | None:
        with self._connect() as conn:
            row = conn.execute(_SELECT + " WHERE job_id = ?", (job_id,)).fetchone()
        return self._row_to_record(row) if row is not None else None

    def delete(self, job_id: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM jobs WHERE job_id = ?", (job_id,))

    def export_csv(self, path: Path) -> None:
        with self._connect() as conn:
            rows = conn.execute(
                _SELECT + " ORDER BY created_at DESC, id DESC"
            ).fetchall()
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(_CSV_COLUMNS)
            for row in rows:
                record = self._row_to_record(row)
                writer.writerow(
                    [
                        record.job_id,
                        record.created_at.isoformat(),
                        record.mode,
                        record.input_path,
                        record.mask_path,
                        record.output_path,
                        record.model_version,
                        record.width,
                        record.height,
                        record.mask_ratio,
                        record.latency_ms,
                        record.note if record.note is not None else "",
                    ]
                )