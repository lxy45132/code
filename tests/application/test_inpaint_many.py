"""InpaintImage.execute_many 用例测试（FakeEngine，无 GPU）。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from hybrid_inpaint.application.inpaint import InpaintImage
from hybrid_inpaint.domain.entities import Image, InpaintRequest, Limits, Mask
from hybrid_inpaint.domain.errors import ValidationError
from hybrid_inpaint.infrastructure.fs_artifacts import FileSystemArtifactStore
from hybrid_inpaint.infrastructure.pillow_codec import PillowImageCodec
from tests.application.fakes import FakeEngine, FixedClock, InMemoryHistory, SeqIds


def _use_case(tmp_path: Path) -> tuple[InpaintImage, InMemoryHistory, FakeEngine]:
    engine = FakeEngine()
    history = InMemoryHistory()
    codec = PillowImageCodec()
    artifacts = FileSystemArtifactStore(tmp_path / "history", codec)
    use_case = InpaintImage(
        engine, artifacts, history, FixedClock(), SeqIds(), Limits()
    )
    return use_case, history, engine


def _request(size: int = 8, hole: bool = True, note: str = "") -> InpaintRequest:
    image = Image(np.full((size, size, 3), 100, dtype=np.uint8))
    mask = np.zeros((size, size), dtype=np.bool_)
    if hole:
        mask[2 : size - 2, 2 : size - 2] = True
    return InpaintRequest(image, Mask(mask), mode="web", note=note)


def test_execute_many_mixed_sizes_and_zero_mask(tmp_path: Path) -> None:
    use_case, history, engine = _use_case(tmp_path)
    requests = [_request(8), _request(12), _request(8, hole=False)]

    results = use_case.execute_many(requests)

    assert len(results) == 3
    assert len({r.job_id for r in results}) == 3
    assert results[2].mask_ratio == 0.0
    # 零洞项不进引擎：只有 2 个活动项
    assert engine.calls == 2
    for result in results:
        assert history.get(result.job_id) is not None


def test_execute_many_matches_single_execute(tmp_path: Path) -> None:
    use_case, _, _ = _use_case(tmp_path)
    requests = [_request(8), _request(8)]
    batch_results = use_case.execute_many(requests)
    single_results = [use_case.execute(request) for request in requests]
    for batch, single in zip(batch_results, single_results):
        assert np.array_equal(batch.image.data, single.image.data)


def test_execute_many_validates_all_before_running(tmp_path: Path) -> None:
    use_case, history, engine = _use_case(tmp_path)
    bad = InpaintRequest(
        Image(np.zeros((8, 8, 3), dtype=np.uint8)),
        Mask(np.ones((8, 8), dtype=np.bool_)),  # 全洞拒绝
        mode="web",
    )
    with pytest.raises(ValidationError):
        use_case.execute_many([_request(8), bad])
    assert engine.calls == 0
    assert history.list(10, 0) == []


def test_execute_many_empty(tmp_path: Path) -> None:
    use_case, _, _ = _use_case(tmp_path)
    assert use_case.execute_many([]) == []
