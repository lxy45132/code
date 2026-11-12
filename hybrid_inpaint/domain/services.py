"""Pure domain policies and helpers.

These functions do not touch the filesystem, torch, or any web framework, so
they are cheap to unit-test.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from hybrid_inpaint.domain.entities import Mask
from hybrid_inpaint.domain.errors import PostconditionError, ValidationError

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


JOB_ID_PATTERN = re.compile(r"[0-9a-f]{32}")
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp")


def validate_job_id(job_id: str) -> None:
    """job_id 必须是 uuid4().hex；拒绝 ``..``/绝对路径等注入。"""
    if not JOB_ID_PATTERN.fullmatch(job_id):
        raise ValidationError("job_id must be a 32-character lowercase hex UUID")


def composite(
    image: np.ndarray, mask: np.ndarray, generated: np.ndarray
) -> np.ndarray:
    """uint8 HWC + bool HW + float32 [0,1] HWC -> uint8 HWC。

    洞外逐像素等于输入；不相等抛 ``PostconditionError``。
    """
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must be uint8 HxWx3")
    if mask.dtype != np.bool_ or mask.shape != image.shape[:2]:
        raise ValueError("mask must be bool and match image spatial size")
    if generated.shape != image.shape:
        raise ValueError("generated must match image shape")
    hole = np.clip(np.rint(generated * 255.0), 0, 255).astype(np.uint8)
    output = image.copy()
    output[mask] = hole[mask]
    if not np.array_equal(output[~mask], image[~mask]):
        raise PostconditionError("composite changed pixels outside the hole")
    return output


def pair_files(
    input_dir: Path, mask_dir: Path, *, recursive: bool = False
) -> list[tuple[Path, Path | None]]:
    """按相对主文件名配对图像/掩码；缺掩码为 ``None``；按相对路径排序。"""
    pattern = "**/*" if recursive else "*"
    pairs: list[tuple[Path, Path | None]] = []
    for image_path in sorted(input_dir.glob(pattern)):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        relative = image_path.relative_to(input_dir)
        mask_path: Path | None = None
        for ext in IMAGE_EXTENSIONS:
            candidate = mask_dir / relative.with_suffix(ext)
            if candidate.is_file():
                mask_path = candidate
                break
        pairs.append((image_path, mask_path))
    return pairs
