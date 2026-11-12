from __future__ import annotations

import numpy as np
import pytest

from hybrid_inpaint.domain.services import composite


def test_composite_keeps_outside_pixels_and_fills_hole() -> None:
    image = np.arange(48, dtype=np.uint8).reshape(4, 4, 3)
    mask = np.zeros((4, 4), dtype=np.bool_)
    mask[1:3, 1:3] = True
    generated = np.full((4, 4, 3), 0.5, dtype=np.float32)

    output = composite(image, mask, generated)

    assert output.dtype == np.uint8
    assert np.array_equal(output[~mask], image[~mask])
    assert np.all(output[mask] == 128)


def test_composite_clips_generated_range() -> None:
    image = np.zeros((2, 2, 3), dtype=np.uint8)
    mask = np.ones((2, 2), dtype=np.bool_)
    generated = np.full((2, 2, 3), 2.0, dtype=np.float32)
    assert np.all(composite(image, mask, generated) == 255)


def test_composite_rejects_shape_mismatch() -> None:
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    mask = np.zeros((3, 4), dtype=np.bool_)
    generated = np.zeros((4, 4, 3), dtype=np.float32)
    with pytest.raises(ValueError):
        composite(image, mask, generated)