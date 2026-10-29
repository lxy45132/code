from __future__ import annotations

import numpy as np
import pytest

from hybrid_inpaint.domain.entities import Mask
from hybrid_inpaint.domain.services import mask_area_bucket, validate_mask


def test_empty_mask_is_rejected_by_default() -> None:
    with pytest.raises(ValueError, match="no holes"):
        validate_mask(Mask(np.zeros((4, 4), dtype=np.bool_)))


def test_all_hole_mask_is_rejected_by_default() -> None:
    with pytest.raises(ValueError, match="whole image"):
        validate_mask(Mask(np.ones((4, 4), dtype=np.bool_)))


def test_allow_empty_and_all() -> None:
    validate_mask(Mask(np.zeros((4, 4), dtype=np.bool_)), allow_empty=True)
    validate_mask(Mask(np.ones((4, 4), dtype=np.bool_)), allow_all=True)


def test_normal_mask_passes() -> None:
    mask = np.zeros((4, 4), dtype=np.bool_)
    mask[0, 0] = True
    validate_mask(Mask(mask))


@pytest.mark.parametrize(
    ("ratio", "expected"),
    [
        (0.0, "0.0-0.1"),
        (0.05, "0.0-0.1"),
        (0.1, "0.1-0.2"),
        (0.35, "0.3-0.4"),
        (0.4, "0.4-0.6"),
        (0.6, "0.4-0.6"),
        (0.61, None),
    ],
)
def test_mask_area_bucket(ratio: float, expected: str | None) -> None:
    assert mask_area_bucket(ratio) == expected
