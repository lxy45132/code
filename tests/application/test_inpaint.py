from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from hybrid_inpaint.application.inpaint import InpaintImage
from hybrid_inpaint.domain.entities import Image, InpaintRequest, Limits, Mask
from hybrid_inpaint.domain.errors import (
    InternalError,
    ResourceLimitError,
    ValidationError,
)
from tests.application.fakes import (
    FakeEngine,
    FixedClock,
    InMemoryArtifacts,
    InMemoryHistory,
    SeqIds,
)


def _setup(tmp_path: Path, limits: Limits | None = None):
    engine = FakeEngine()
    artifacts = InMemoryArtifacts(tmp_path / "artifacts")
    history = InMemoryHistory()
    use_case = InpaintImage(
        engine, artifacts, history, FixedClock(), SeqIds(), limits or Limits()
    )
    return use_case, engine, artifacts, history


def _request() -> InpaintRequest:
    image = Image(np.arange(48, dtype=np.uint8).reshape(4, 4, 3))
    mask = np.zeros((4, 4), dtype=np.bool_)
    mask[1:3, 1:3] = True
    return InpaintRequest(image, Mask(mask), mode="cli")


def test_normal_inpaint_preserves_outside_and_records(tmp_path: Path) -> None:
    use_case, engine, artifacts, history = _setup(tmp_path)
    request = _request()

    result = use_case.execute(request)

    assert engine.calls == 1
    assert np.array_equal(
        result.image.data[~request.mask.data], request.image.data[~request.mask.data]
    )
    assert result.model_version == engine.version
    assert result.mask_ratio == pytest.approx(0.25)
    assert result.job_id in artifacts.saved
    record = history.get(result.job_id)
    assert record is not None
    assert record.mode == "cli"
    assert record.width == 4 and record.height == 4
    assert record.output_path == str(artifacts.saved[result.job_id].output)


def test_zero_hole_returns_identity_without_engine(tmp_path: Path) -> None:
    use_case, engine, artifacts, history = _setup(tmp_path)
    image = Image(np.arange(48, dtype=np.uint8).reshape(4, 4, 3))
    request = InpaintRequest(image, Mask(np.zeros((4, 4), dtype=np.bool_)))

    result = use_case.execute(request)

    assert engine.calls == 0
    assert np.array_equal(result.image.data, image.data)
    assert result.mask_ratio == 0.0
    assert history.get(result.job_id) is not None  # 零洞仍写历史


def test_all_hole_is_rejected(tmp_path: Path) -> None:
    use_case, engine, _, history = _setup(tmp_path)
    image = Image(np.zeros((4, 4, 3), dtype=np.uint8))
    request = InpaintRequest(image, Mask(np.ones((4, 4), dtype=np.bool_)))

    with pytest.raises(ValidationError):
        use_case.execute(request)
    assert engine.calls == 0
    assert history.records == {}


def test_shape_mismatch_is_rejected(tmp_path: Path) -> None:
    use_case, _, _, _ = _setup(tmp_path)
    image = Image(np.zeros((4, 4, 3), dtype=np.uint8))
    # 实体构造即拒绝尺寸不一致；用例层另有同名校验作为纵深防御
    with pytest.raises(ValueError):
        InpaintRequest(image, Mask(np.zeros((3, 4), dtype=np.bool_)))


def test_pixel_limit_is_enforced(tmp_path: Path) -> None:
    use_case, _, _, _ = _setup(tmp_path, Limits(max_pixels=4))
    with pytest.raises(ResourceLimitError):
        use_case.execute(_request())


def test_history_failure_cleans_artifacts(tmp_path: Path) -> None:
    use_case, _, artifacts, history = _setup(tmp_path)
    history.raise_on_add = RuntimeError("db down")

    with pytest.raises(InternalError):
        use_case.execute(_request())
    assert artifacts.saved == {}
    assert len(artifacts.deleted) == 1