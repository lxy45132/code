from __future__ import annotations

import numpy as np
import pytest

from hybrid_inpaint.domain.entities import Image, Mask, Sample


def test_image_accepts_uint8_rgb() -> None:
    image = Image(np.zeros((4, 5, 3), dtype=np.uint8))
    assert image.height == 4
    assert image.width == 5


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int32])
def test_image_rejects_non_uint8(dtype: np.dtype) -> None:
    with pytest.raises(ValueError, match="uint8"):
        Image(np.zeros((4, 5, 3), dtype=dtype))


def test_image_rejects_grayscale_and_bad_rank() -> None:
    with pytest.raises(ValueError, match="H×W×3"):
        Image(np.zeros((4, 5), dtype=np.uint8))
    with pytest.raises(ValueError, match="H×W×3"):
        Image(np.zeros((4, 5, 1), dtype=np.uint8))


def test_mask_ratio_and_holes() -> None:
    mask = Mask(np.array([[True, False], [False, False]], dtype=np.bool_))
    assert mask.ratio == pytest.approx(0.25)
    assert mask.holes == 1


def test_mask_rejects_non_bool() -> None:
    with pytest.raises(ValueError, match="bool"):
        Mask(np.zeros((4, 4), dtype=np.uint8))


def test_sample_rejects_shape_mismatch() -> None:
    image = Image(np.zeros((4, 5, 3), dtype=np.uint8))
    mask = Mask(np.zeros((4, 4), dtype=np.bool_))
    with pytest.raises(ValueError, match="same height and width"):
        Sample(sample_id="x", image=image, mask=mask)
