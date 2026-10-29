"""Synthetic mask generation and mask geometry helpers.

The training pipeline will eventually consume a published irregular-mask set;
this module provides reproducible synthetic masks for C1/C2 smoke tests and
for generating large-hole stress masks.
"""

from __future__ import annotations

import numpy as np

from hybrid_inpaint.domain.services import DEFAULT_MASK_BUCKETS, mask_area_bucket

MASK_MODES = ("rect", "brush")


def _validate_ratio(ratio: float) -> float:
    ratio = float(ratio)
    if not np.isfinite(ratio) or not 0.0 < ratio < 1.0:
        raise ValueError("mask ratio must be finite and in (0, 1)")
    return ratio


def make_rect_mask(size: int, ratio: float, rng: np.random.Generator) -> np.ndarray:
    """Return a rectangular mask whose area is close to ``ratio * size²``."""
    ratio = _validate_ratio(ratio)
    if size < 1:
        raise ValueError("size must be >= 1")

    target = ratio * size * size
    aspect = float(rng.uniform(0.5, 2.0))
    width = int(round(np.sqrt(target * aspect)))
    width = min(size, max(1, width))
    height = int(round(target / max(width, 1)))
    height = min(size, max(1, height))

    # If clamping made the rectangle too small, grow the other side.
    if width * height < target * 0.8:
        height = min(size, max(height, int(round(target / width))))
    if width * height < target * 0.8:
        width = min(size, max(width, int(round(target / height))))

    x0 = int(rng.integers(0, size - width + 1))
    y0 = int(rng.integers(0, size - height + 1))
    mask = np.zeros((size, size), dtype=np.bool_)
    mask[y0 : y0 + height, x0 : x0 + width] = True
    return mask


def make_brush_mask(size: int, ratio: float, rng: np.random.Generator) -> np.ndarray:
    """Return a stroke-like mask: a thick random polyline plus round caps."""
    _validate_ratio(ratio)
    if size < 1:
        raise ValueError("size must be >= 1")

    vertices = int(rng.integers(4, 13))
    points = rng.integers(0, size, size=(vertices, 2))
    width = max(1, int(round(float(rng.uniform(0.02, 0.12)) * size)))
    radius_sq = (width / 2.0) ** 2

    yy, xx = np.mgrid[0:size, 0:size]
    mask = np.zeros((size, size), dtype=np.bool_)

    for index in range(vertices - 1):
        x1, y1 = int(points[index, 0]), int(points[index, 1])
        x2, y2 = int(points[index + 1, 0]), int(points[index + 1, 1])
        dx, dy = x2 - x1, y2 - y1
        segment_sq = dx * dx + dy * dy
        if segment_sq == 0:
            continue
        t = ((xx - x1) * dx + (yy - y1) * dy) / segment_sq
        t = np.clip(t, 0.0, 1.0)
        px = x1 + t * dx
        py = y1 + t * dy
        mask |= (xx - px) ** 2 + (yy - py) ** 2 <= radius_sq

    if not mask.any():
        cx, cy = (int(v) for v in rng.integers(0, size, size=2))
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= max(1, width // 2) ** 2
    return mask


def make_mask(
    size: int,
    ratio: float,
    rng: np.random.Generator,
    mode: str = "rect",
) -> np.ndarray:
    """Generate a mask in ``mode``; ``rect`` or ``brush``."""
    if mode == "rect":
        return make_rect_mask(size, ratio, rng)
    if mode == "brush":
        return make_brush_mask(size, ratio, rng)
    raise ValueError(f"unknown mask mode: {mode!r}; expected one of {MASK_MODES}")


def resize_mask_nearest(mask: np.ndarray, height: int, width: int) -> np.ndarray:
    """Resize a 2D bool mask with nearest-neighbour sampling."""
    if mask.ndim != 2:
        raise ValueError("mask must be 2D")
    if height < 1 or width < 1:
        raise ValueError("height and width must be >= 1")
    if mask.shape == (height, width):
        return mask.copy()

    rows = np.linspace(0, mask.shape[0] - 1, height).round().astype(np.int64)
    cols = np.linspace(0, mask.shape[1] - 1, width).round().astype(np.int64)
    return mask[np.ix_(rows, cols)].astype(np.bool_, copy=True)


def mask_bucket_label(
    ratio: float,
    buckets: tuple[tuple[float, float], ...] = DEFAULT_MASK_BUCKETS,
) -> str | None:
    return mask_area_bucket(ratio, buckets)
