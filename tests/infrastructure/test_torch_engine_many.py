"""TorchInpaintEngine.inpaint_many 凑批测试（tiny checkpoint，CPU）。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from hybrid_inpaint.domain.errors import ValidationError
from hybrid_inpaint.infrastructure.torch_engine import SerializedEngine
from tests.infrastructure.test_torch_engine import _engine


def _image(size: int = 16, value: int = 90) -> np.ndarray:
    return np.full((size, size, 3), value, dtype=np.uint8)


def _mask(size: int = 16) -> np.ndarray:
    mask = np.zeros((size, size), dtype=np.bool_)
    mask[4 : size - 4, 4 : size - 4] = True
    return mask


def test_many_same_size_matches_single(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    images = [_image(16, 80), _image(16, 120), _image(16, 160)]
    masks = [_mask(16) for _ in images]

    batched = engine.inpaint_many(images, masks)
    singles = [engine.inpaint(image, mask) for image, mask in zip(images, masks)]

    assert len(batched) == 3
    for batch, single in zip(batched, singles):
        assert batch.shape == (16, 16, 3)
        assert np.allclose(batch, single, atol=1e-5)


def test_many_mixed_sizes_degrades_per_image(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    images = [_image(16), _image(24), _image(16)]
    masks = [_mask(16), _mask(24), _mask(16)]

    results = engine.inpaint_many(images, masks)

    assert [r.shape[:2] for r in results] == [(16, 16), (24, 24), (16, 16)]
    for result, image, mask in zip(results, images, masks):
        assert np.allclose(result, engine.inpaint(image, mask), atol=1e-5)


def test_many_rejects_mismatched_counts(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    with pytest.raises(ValidationError):
        engine.inpaint_many([_image()], [_mask(), _mask()])


def test_serialized_engine_many_locks_once(tmp_path: Path) -> None:
    engine = SerializedEngine(_engine(tmp_path), timeout_s=5.0)
    results = engine.inpaint_many([_image(), _image()], [_mask(), _mask()])
    assert len(results) == 2
