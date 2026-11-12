"""PillowImageCodec: bounded decoding, EXIF, RGB conversion, lossless PNG."""

from __future__ import annotations

import io

import numpy as np
from PIL import Image as PILImage
from PIL import ImageOps

from hybrid_inpaint.domain.entities import Image, Limits, Mask
from hybrid_inpaint.domain.errors import ResourceLimitError, ValidationError


class PillowImageCodec:
    def decode(self, data: bytes, limits: Limits) -> Image:
        if len(data) > limits.max_file_mb * 1024 * 1024:
            raise ResourceLimitError("image file exceeds the size limit")
        try:
            with PILImage.open(io.BytesIO(data)) as raw:
                width, height = raw.size  # header 预检，不解压全图
                if max(width, height) > limits.max_side:
                    raise ResourceLimitError("image side exceeds the maximum")
                if width * height > limits.max_pixels:
                    raise ResourceLimitError("image exceeds the pixel budget")
                image = ImageOps.exif_transpose(raw).convert("RGB")
                array = np.asarray(image, dtype=np.uint8)
        except ResourceLimitError:
            raise
        except Exception as exc:
            raise ValidationError("cannot decode image", detail=str(exc)) from exc
        return Image(array)

    def decode_mask(self, data: bytes, *, height: int, width: int) -> Mask:
        try:
            with PILImage.open(io.BytesIO(data)) as raw:
                array = np.asarray(raw.convert("L"), dtype=np.uint8)
        except Exception as exc:
            raise ValidationError("cannot decode mask", detail=str(exc)) from exc
        mask = array > 0
        if mask.shape != (height, width):
            resized = PILImage.fromarray(mask.astype(np.uint8) * 255).resize(
                (width, height), PILImage.Resampling.NEAREST
            )
            mask = np.asarray(resized) > 0
        return Mask(mask)

    def encode_png(self, image: Image) -> bytes:
        buf = io.BytesIO()
        PILImage.fromarray(image.data).save(buf, format="PNG")
        return buf.getvalue()

    def encode_mask_png(self, mask: Mask) -> bytes:
        buf = io.BytesIO()
        PILImage.fromarray(mask.to_uint8()).save(buf, format="PNG")
        return buf.getvalue()