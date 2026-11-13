"""SQLAlchemy 2.0 ORM models for the MySQL history/batch schema.

表结构与 02 文档 §11.1 的 SQLite 版一致，另加 C7 批量任务表
(batch_jobs / batch_items)。时间戳存 ISO-8601 字符串，与 SQLite 版
行为对齐（排序语义一致、无时区解析歧义）。建表由 Alembic 迁移负责，
代码运行时不做 DDL。
"""

from __future__ import annotations

from sqlalchemy import Float, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Declarative base for all ORM models."""


class JobRow(Base):
    """Single inpaint job history record (02 文档 §11.1)."""

    __tablename__ = "jobs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    job_id: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    created_at: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    mode: Mapped[str] = mapped_column(String(8), nullable=False)
    batch_id: Mapped[str | None] = mapped_column(String(64), index=True)
    input_path: Mapped[str] = mapped_column(Text, nullable=False)
    mask_path: Mapped[str] = mapped_column(Text, nullable=False)
    output_path: Mapped[str] = mapped_column(Text, nullable=False)
    model_version: Mapped[str] = mapped_column(String(64), nullable=False)
    image_width: Mapped[int] = mapped_column(Integer, nullable=False)
    image_height: Mapped[int] = mapped_column(Integer, nullable=False)
    mask_ratio: Mapped[float] = mapped_column(Float, nullable=False)
    latency_ms: Mapped[float] = mapped_column(Float, nullable=False)
    note: Mapped[str | None] = mapped_column(Text)


class BatchJobRow(Base):
    """Web batch job authoritative state (03 文档 §6)."""

    __tablename__ = "batch_jobs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    batch_id: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    total: Mapped[int] = mapped_column(Integer, nullable=False)
    done: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    failed: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    created_at: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    finished_at: Mapped[str | None] = mapped_column(String(32))
    report_path: Mapped[str | None] = mapped_column(Text)
    note: Mapped[str | None] = mapped_column(Text)


class BatchItemRow(Base):
    """Per-item result of a web batch job (03 文档 §6)."""

    __tablename__ = "batch_items"
    __table_args__ = (
        UniqueConstraint("batch_id", "item_index", name="uq_batch_item_index"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    batch_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    item_index: Mapped[int] = mapped_column(Integer, nullable=False)
    image_name: Mapped[str] = mapped_column(String(255), nullable=False)
    job_id: Mapped[str | None] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    error_code: Mapped[str | None] = mapped_column(String(32))
    latency_ms: Mapped[float | None] = mapped_column(Float)
    mask_ratio: Mapped[float | None] = mapped_column(Float)
