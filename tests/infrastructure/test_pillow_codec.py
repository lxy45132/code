from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image as PILImage

from hybrid_inpaint.domain.entities import Image, Limits, Mask
from hybrid_inpaint.domain.errors import ResourceLimitError, ValidationError
from hybrid_inpaint.infrastructure.pillow_codec import PillowImageCodec


def _png_bytes(array: np.ndarray, mode: str = "RGB") -> bytes:
    buf = io.BytesIO()
    PILImage.fromarray(array, mode=mode).save(buf, format="PNG")
    return buf.getvalue()


def test_decode_grayscale_and_rgba_to_rgb() -> None:
    codec = PillowImageCodec()
    gray = codec.decode(_png_bytes(np.full((8, 8), 128, np.uint8), mode="L"), Limits())
    assert gray.data.shape == (8, 8, 3)
    rgba = codec.decode(
        _png_bytes(np.full((8, 8, 4), 200, np.uint8), mode="RGBA"), Limits()
    )
    assert rgba.data.shape == (8, 8, 3)


def test_decode_exif_orientation() -> None:
    array = np.zeros((8, 4, 3), dtype=np.uint8)
    array[:, :2] = 255
    exif = PILImage.Exif()
    exif[274] = 6  # Orientation: rotate 90 CW
    buf = io.BytesIO()
    PILImage.fromarray(array).save(buf, format="JPEG", exif=exif)
    decoded = PillowImageCodec().decode(buf.getvalue(), Limits())
    assert decoded.data.shape == (4, 8, 3)


def test_decode_corrupt_raises_validation() -> None:
    with pytest.raises(ValidationError):
        PillowImageCodec().decode(b"not an image", Limits())


def test_decode_file_size_limit() -> None:
    with pytest.raises(ResourceLimitError):
        PillowImageCodec().decode(b"x" * (2 * 1024 * 1024), Limits(max_file_mb=1))


def test_decode_side_and_pixel_limits() -> None:
    data = _png_bytes(np.zeros((16, 16, 3), np.uint8))
    with pytest.raises(ResourceLimitError):
        PillowImageCodec().decode(data, Limits(max_side=8))
    with pytest.raises(ResourceLimitError):
        PillowImageCodec().decode(data, Limits(max_pixels=100))


def test_encode_png_roundtrip() -> None:
    codec = PillowImageCodec()
    image = Image(np.arange(48, dtype=np.uint8).reshape(4, 4, 3))
    decoded = codec.decode(codec.encode_png(image), Limits())
    assert np.array_equal(decoded.data, image.data)


def test_decode_mask_threshold_and_resize() -> None:
    codec = PillowImageCodec()
    array = np.zeros((8, 8), dtype=np.uint8)
    array[2:6, 2:6] = 200
    mask = codec.decode_mask(_png_bytes(array, mode="L"), height=8, width=8)
    assert mask.data.dtype == np.bool_
    assert mask.holes == 16
    resized = codec.decode_mask(_png_bytes(array, mode="L"), height=4, width=4)
    assert resized.data.shape == (4, 4)
    assert resized.holes == 4


def test_encode_mask_png_roundtrip() -> None:
    codec = PillowImageCodec()
    mask = Mask(np.eye(4, dtype=np.bool_))
    decoded = codec.decode_mask(codec.encode_mask_png(mask), height=4, width=4)
    assert np.array_equal(decoded.data, mask.data)