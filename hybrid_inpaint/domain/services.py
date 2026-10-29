"""Pure domain policies and helpers.

These functions do not touch the filesystem, torch, or any web framework, so
they are cheap to unit-test.
"""

from __future__ import annotations

from collections.abc import Sequence

from hybrid_inpaint.domain.entities import Mask

DEFAULT_MASK_BUCKETS: tuple[tuple[float, float], ...] = (
    (0.0, 0.1),
    (0.1, 0.2),
    (0.2, 0.3),
    (0.3, 0.4),
    (0.4, 0.6),
)


def validate_mask(
    mask: Mask,
    *,
    allow_empty: bool = False,
    allow_all: bool = False,
) -> None:
    """Validate an inpainting mask; raise ``ValueError`` on policy violation."""
    ratio = mask.ratio
    if ratio < 0.0 or ratio > 1.0:
        raise ValueError("mask ratio must be in [0, 1]")
    if not allow_empty and ratio == 0.0:
        raise ValueError("mask has no holes")
    if not allow_all and ratio == 1.0:
        raise ValueError("mask covers the whole image")


def mask_area_bucket(
    ratio: float,
    buckets: Sequence[tuple[float, float]] = DEFAULT_MASK_BUCKETS,
) -> str | None:
    """Return the bucket label containing ``ratio``, or ``None``.

    The last bucket includes its upper bound (e.g. exactly 0.6 belongs to
    ``0.4-0.6``); earlier buckets are right-open.
    """
    if not 0.0 <= ratio <= 1.0:
        raise ValueError("ratio must be in [0, 1]")
    for index, (start, end) in enumerate(buckets):
        last = index == len(buckets) - 1
        if start <= ratio < end or (last and ratio == end):
            return f"{start:.1f}-{end:.1f}"
    return None
