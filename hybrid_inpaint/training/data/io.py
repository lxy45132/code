"""Image and mask loading for the data pipeline.

Training-time I/O is intentionally small: RGB conversion, optional resize,
binary mask thresholding and nearest-neighbour mask resize.  The system layer
will have its own stricter ``PillowImageCodec`` with limits and error mapping.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image as PILImage
from PIL import ImageOps

from hybrid_inpaint.domain.entities import Image, Mask
from hybrid_inpaint.training.data.masks import resize_mask_nearest


def load_image(
    path: str | Path,
    *,
    height: int | None = None,
    width: int | None = None,
) -> Image:
    """Load an image as RGB uint8, applying EXIF orientation."""
    if (height is None) != (width is None):
        raise ValueError("height and width must be provided together")
    with PILImage.open(path) as raw:
        image = ImageOps.exif_transpose(raw).convert("RGB")
        if height is not None and width is not None:
            image = image.resize((width, height), PILImage.Resampling.BILINEAR)
        array = np.asarray(image, dtype=np.uint8)
    return Image(array)


def load_mask(
    path: str | Path,
    *,
    height: int | None = None,
    width: int | None = None,
    threshold: int = 0,
) -> Mask:
    """Load a mask as bool (``> threshold`` is a hole), optionally resizing."""
    if (height is None) != (width is None):
        raise ValueError("height and width must be provided together")
    if not 0 <= threshold <= 255:
        raise ValueError("threshold must be in [0, 255]")

    with PILImage.open(path) as raw:
        array = np.asarray(raw.convert("L"), dtype=np.uint8)
    mask = array > threshold

    if height is not None and width is not None and mask.shape != (height, width):
        mask = resize_mask_nearest(mask, height, width)
    return Mask(mask)
