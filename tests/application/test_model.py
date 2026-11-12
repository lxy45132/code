from __future__ import annotations

from pathlib import Path

import pytest

from hybrid_inpaint.application.model import ModelService
from hybrid_inpaint.domain.entities import ModelSpec
from hybrid_inpaint.domain.errors import InternalError, ModelUnavailableError
from tests.application.fakes import FakeEngine


class FakeRepo:
    def __init__(self, engine=None, error: Exception | None = None) -> None:
        self.engine = engine or FakeEngine()
        self.error = error
        self.loads = 0

    def resolve(self, model_id: str) -> ModelSpec:
        return ModelSpec(model_id, Path("fake.pt"))

    def load(self, spec: ModelSpec):
        self.loads += 1
        if self.error is not None:
            raise self.error
        return self.engine


def test_load_warms_up_and_reports_health() -> None:
    repo = FakeRepo()
    service = ModelService(repo)
    engine = service.load("main")
    assert engine.warmups == 1
    health = service.health()
    assert health["ready"] is True
    assert health["model_version"] == engine.version
    assert health["device"] == "fake"


def test_load_same_id_is_idempotent() -> None:
    repo = FakeRepo()
    service = ModelService(repo)
    first = service.load("main")
    second = service.load("main")
    assert first is second
    assert repo.loads == 1


def test_hot_swap_is_rejected() -> None:
    service = ModelService(FakeRepo())
    service.load("main")
    with pytest.raises(InternalError):
        service.load("other")


def test_load_failure_records_last_error() -> None:
    service = ModelService(FakeRepo(error=RuntimeError("bad weights")))
    with pytest.raises(ModelUnavailableError):
        service.load("main")
    health = service.health()
    assert health["ready"] is False
    assert "bad weights" in health["last_error"]


def test_preloaded_engine_skips_repository() -> None:
    engine = FakeEngine()
    service = ModelService(None, engine=engine, model_id="main")
    assert service.load("main") is engine
    assert service.health()["ready"] is True