from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

import pytest

from hybrid_inpaint.domain.entities import HistoryRecord
from hybrid_inpaint.infrastructure.sqlalchemy_history import (
    SqlAlchemyHistoryRepository,
    ensure_schema,
)
from hybrid_inpaint.infrastructure.sqlalchemy_models import Base

pytestmark = [
    pytest.mark.mysql,
    pytest.mark.skipif(
        not os.environ.get("HYBRID_INPAINT_TEST_MYSQL_URL"),
        reason="set HYBRID_INPAINT_TEST_MYSQL_URL to run MySQL integration tests",
    ),
]


@pytest.fixture()
def repo() -> SqlAlchemyHistoryRepository:
    from sqlalchemy import create_engine

    url = os.environ["HYBRID_INPAINT_TEST_MYSQL_URL"]
    engine = create_engine(url, pool_pre_ping=True)
    Base.metadata.drop_all(engine)
    ensure_schema(engine)
    return SqlAlchemyHistoryRepository(engine, paths_base=Path("outputs"))


def _record(job_id: str) -> HistoryRecord:
    return HistoryRecord(
        job_id=job_id,
        created_at=datetime(2026, 11, 13, 12, 0, tzinfo=timezone.utc),
        mode="web",
        batch_id=None,
        input_path="outputs/history/input.png",
        mask_path="outputs/history/mask.png",
        output_path="outputs/history/output.png",
        model_version="v1",
        width=8,
        height=8,
        mask_ratio=0.25,
        latency_ms=12.5,
        note="mysql integration",
    )


def test_mysql_roundtrip(repo: SqlAlchemyHistoryRepository) -> None:
    repo.add(_record(format(1, "032x")))
    loaded = repo.get(format(1, "032x"))
    assert loaded is not None
    assert loaded.note == "mysql integration"
    assert loaded.created_at == datetime(2026, 11, 13, 12, 0, tzinfo=timezone.utc)


def test_mysql_duplicate_is_rejected(repo: SqlAlchemyHistoryRepository) -> None:
    from hybrid_inpaint.domain.errors import ValidationError

    repo.add(_record(format(1, "032x")))
    with pytest.raises(ValidationError):
        repo.add(_record(format(1, "032x")))
