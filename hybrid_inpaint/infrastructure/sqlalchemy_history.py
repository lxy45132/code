"""SqlAlchemyHistoryRepository: MySQL/SQLAlchemy adapter for HistoryRepository.

端口语义与 SqliteHistoryRepository 完全一致（02 文档 §11.1）：相对路径
存储、参数化查询、CSV 导出、UUID job_id 校验。建表由 Alembic 迁移负责
（见 alembic/versions/0001_initial.py），本适配器不做 DDL。
"""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

from sqlalchemy import Engine, create_engine, delete, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from hybrid_inpaint.domain.entities import HistoryRecord
from hybrid_inpaint.domain.errors import InternalError, ValidationError
from hybrid_inpaint.domain.services import validate_job_id
from hybrid_inpaint.infrastructure.sqlalchemy_models import JobRow

CSV_COLUMNS = (
    "job_id", "created_at", "mode", "input", "mask", "output",
    "model_version", "width", "height", "mask_ratio", "latency_ms", "note",
)

# 生产惯例连接池参数（03 文档 §4）。
POOL_SIZE = 10
MAX_OVERFLOW = 20
POOL_RECYCLE = 3600


def mysql_url(
    host: str, port: int, user: str, password: str, database: str
) -> str:
    """Build a pymysql URL with a percent-escaped password."""
    from urllib.parse import quote_plus

    safe_password = quote_plus(password)
    return f"mysql+pymysql://{user}:{safe_password}@{host}:{port}/{database}?charset=utf8mb4"


def build_mysql_engine(
    host: str, port: int, user: str, password: str, database: str
) -> Engine:
    return create_engine(
        mysql_url(host, port, user, password, database),
        pool_size=POOL_SIZE,
        max_overflow=MAX_OVERFLOW,
        pool_pre_ping=True,
        pool_recycle=POOL_RECYCLE,
    )


class SqlAlchemyHistoryRepository:
    """Implements the HistoryRepository port on any SQLAlchemy engine."""

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

    def _row_to_record(self, row: JobRow) -> HistoryRecord:
        return HistoryRecord(
            job_id=row.job_id,
            created_at=datetime.fromisoformat(row.created_at),
            mode=row.mode,
            batch_id=row.batch_id,
            input_path=self._to_absolute(row.input_path),
            mask_path=self._to_absolute(row.mask_path),
            output_path=self._to_absolute(row.output_path),
            model_version=row.model_version,
            width=row.image_width,
            height=row.image_height,
            mask_ratio=row.mask_ratio,
            latency_ms=row.latency_ms,
            note=row.note,
        )

    def add(self, record: HistoryRecord) -> None:
        validate_job_id(record.job_id)
        row = JobRow(
            job_id=record.job_id,
            created_at=record.created_at.isoformat(),
            mode=record.mode,
            batch_id=record.batch_id,
            input_path=self._to_relative(record.input_path),
            mask_path=self._to_relative(record.mask_path),
            output_path=self._to_relative(record.output_path),
            model_version=record.model_version,
            image_width=record.width,
            image_height=record.height,
            mask_ratio=record.mask_ratio,
            latency_ms=record.latency_ms,
            note=record.note,
        )
        try:
            with Session(self._engine) as session, session.begin():
                session.add(row)
        except IntegrityError as exc:
            raise ValidationError(
                f"duplicate or invalid job_id: {record.job_id}"
            ) from exc

    def list(self, limit: int, offset: int) -> list[HistoryRecord]:
        statement = (
            select(JobRow)
            .order_by(JobRow.created_at.desc(), JobRow.id.desc())
            .limit(limit)
            .offset(offset)
        )
        with Session(self._engine) as session:
            rows = session.scalars(statement).all()
        return [self._row_to_record(row) for row in rows]

    def get(self, job_id: str) -> HistoryRecord | None:
        with Session(self._engine) as session:
            row = session.scalar(select(JobRow).where(JobRow.job_id == job_id))
        return self._row_to_record(row) if row is not None else None

    def delete(self, job_id: str) -> None:
        with Session(self._engine) as session, session.begin():
            session.execute(delete(JobRow).where(JobRow.job_id == job_id))

    def export_csv(self, path: Path) -> None:
        statement = select(JobRow).order_by(JobRow.created_at.desc(), JobRow.id.desc())
        with Session(self._engine) as session:
            rows = session.scalars(statement).all()
        records = [self._row_to_record(row) for row in rows]
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(CSV_COLUMNS)
            for record in records:
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


def ensure_schema(engine: Engine) -> None:
    """Create all tables on the given engine (tests / db init convenience).

    生产入口走 Alembic（``alembic upgrade head``）；此函数仅供测试与
    ``db init`` 的 SQLite 后端使用，避免在适配器中隐藏 DDL。
    """
    from hybrid_inpaint.infrastructure.sqlalchemy_models import Base

    try:
        Base.metadata.create_all(engine)
    except InternalError:
        raise
    except Exception as exc:  # pragma: no cover - dialect/IO errors
        raise InternalError(f"schema creation failed: {exc}") from exc
