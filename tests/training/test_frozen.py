from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from hybrid_inpaint.training.data.frozen import FrozenInpaintDataset
from hybrid_inpaint.training.data.io import load_image, load_mask
from hybrid_inpaint.training.data.synthetic import save_frozen_set


def test_load_image_and_mask_round_trip(tmp_path: Path) -> None:
    from PIL import Image as PILImage

    image_path = tmp_path / "image.png"
    mask_path = tmp_path / "mask.png"
    PILImage.new("RGB", (8, 6), color=(10, 20, 30)).save(image_path)
    mask = np.zeros((6, 8), dtype=np.uint8)
    mask[1:3, 2:5] = 255
    PILImage.fromarray(mask, mode="L").save(mask_path)

    image = load_image(image_path)
    loaded_mask = load_mask(mask_path)

    assert image.data.shape == (6, 8, 3)
    assert loaded_mask.data.shape == (6, 8)
    assert loaded_mask.data.sum() == 6


def test_load_mask_resizes_nearest(tmp_path: Path) -> None:
    from PIL import Image as PILImage

    mask = np.zeros((4, 4), dtype=np.uint8)
    mask[0:2, 0:2] = 255
    mask_path = tmp_path / "mask.png"
    PILImage.fromarray(mask, mode="L").save(mask_path)

    resized = load_mask(mask_path, height=8, width=8)
    assert resized.data.shape == (8, 8)
    assert resized.data[0:4, 0:4].all()
    assert not resized.data[4:, :].any()


def test_frozen_dataset_round_trip(tmp_path: Path) -> None:
    root = tmp_path / "frozen"
    save_frozen_set(root, size=64, count=3, seed=2026)
    dataset = FrozenInpaintDataset(root)

    assert len(dataset) == 3
    sample = dataset[0]
    assert sample.sample_id == "frozen-0000"
    assert sample.image.data.shape == (64, 64, 3)
    assert sample.mask.data.shape == (64, 64)
    assert sample.meta["frozen"] is True
    assert len(sample.meta["image_sha256"]) == 64


def test_frozen_dataset_missing_manifest(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        FrozenInpaintDataset(tmp_path)
