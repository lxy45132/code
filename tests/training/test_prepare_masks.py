from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from hybrid_inpaint.training.data.prepare import (
    _preprocess_val_mask,
    _square_around_hole,
)


def test_square_around_hole_keeps_off_centre_hole() -> None:
    hole = np.zeros((640, 960), dtype=bool)
    hole[300:340, 20:60] = True
    cropped = _square_around_hole(hole)
    assert cropped.shape == (640, 640)
    assert cropped.any()


def test_preprocess_val_mask_survives_resizing(tmp_path: Path) -> None:
    array = np.full((640, 960), 255, dtype=np.uint8)
    array[300:340, 20:60] = 0  # hole near the left edge, outside a centre crop
    source = tmp_path / "mask.png"
    Image.fromarray(array, mode="L").save(source)
    output = tmp_path / "processed.png"

    info = _preprocess_val_mask(source, output)

    assert output.is_file()
    assert info["area_256"] > 0.0
    assert info["area_256"] < 1.0
    assert info["width_256"] == 256
    assert info["height_256"] == 256
    assert len(info["processed_sha256"]) == 64
