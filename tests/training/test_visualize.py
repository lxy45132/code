from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image as PILImage

from hybrid_inpaint.domain.entities import Image, Mask, Sample
from hybrid_inpaint.models.unet import UNet
from hybrid_inpaint.training.engine.visualize import save_val_samples


def _samples(count: int = 3, size: int = 16) -> list[Sample]:
    samples = []
    for index in range(count):
        image = np.full((size, size, 3), 30 + index * 10, dtype=np.uint8)
        mask = np.zeros((size, size), dtype=np.bool_)
        mask[4:12, 4:12] = True
        samples.append(Sample(sample_id=f"s{index}", image=Image(image), mask=Mask(mask)))
    return samples


def _model() -> UNet:
    return UNet(in_channels=4, out_channels=3, channels=(4, 8, 16, 32), norm="instance")


def test_save_val_samples_writes_three_column_grid(tmp_path: Path) -> None:
    target = tmp_path / "samples" / "val_000001.png"

    path = save_val_samples(
        _model(), _samples(), device=torch.device("cpu"), count=2, out_path=target
    )

    assert path == target
    assert target.is_file()
    with PILImage.open(target) as grid:
        assert grid.size == (3 * 16, 2 * 16)  # 宽=3列×W，高=N行×H


def test_zero_count_writes_nothing(tmp_path: Path) -> None:
    target = tmp_path / "none.png"

    result = save_val_samples(
        _model(), _samples(), device=torch.device("cpu"), count=0, out_path=target
    )

    assert result is None
    assert not target.exists()