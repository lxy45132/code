from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from hybrid_inpaint.training.data.manifest import make_entry, write_manifest
from hybrid_inpaint.training.data.manifest_dataset import (
    FixedPairDataset,
    ManifestInpaintDataset,
)


def _rgb(path: Path, size: tuple[int, int], color: tuple[int, int, int]) -> Path:
    Image.new("RGB", size, color=color).save(path)
    return path


def _train_mask(path: Path) -> Path:
    array = np.full((640, 960), 255, dtype=np.uint8)
    array[300:340, 450:500] = 0
    Image.fromarray(array, mode="L").save(path)
    return path


def _fixed_mask(path: Path) -> Path:
    array = np.zeros((256, 256), dtype=np.uint8)
    array[100:150, 100:160] = 255
    Image.fromarray(array, mode="L").save(path)
    return path


def _image_manifest(tmp_path: Path, files: list[Path], split: str) -> Path:
    entries = [
        make_entry(
            path,
            image_id=f"{split}-{index}",
            split=split,
            source="unit-test",
            license="test-only",
        )
        for index, path in enumerate(files)
    ]
    path = tmp_path / f"{split}.jsonl"
    write_manifest(
        path,
        entries,
        kind="images",
        dataset="unit",
        split=split,
        seed=2026,
        source="unit-test",
        license="test-only",
    )
    return path


def _mask_manifest(
    tmp_path: Path,
    files: list[Path],
    *,
    split: str,
    processed: bool,
) -> Path:
    entries = []
    for index, path in enumerate(files):
        entry = make_entry(
            path,
            image_id=f"mask-{split}-{index}",
            split=split,
            source="unit-test",
            license="test-only",
        )
        if processed:
            entry["processed_path"] = entry["relative_path"]
            entry["processed_sha256"] = entry["sha256"]
            entry["area_256"] = 0.02
        entries.append(entry)
    out = tmp_path / f"mask-{split}.jsonl"
    write_manifest(
        out,
        entries,
        kind="masks",
        dataset="unit",
        split=split,
        seed=2026,
        source="unit-test",
        license="test-only",
    )
    return out


def test_training_dataset_is_deterministic_and_shaped(tmp_path: Path) -> None:
    images = [
        _rgb(tmp_path / f"img-{index}.png", (256, 256), (index * 30, 40, 60))
        for index in range(3)
    ]
    masks = [_train_mask(tmp_path / f"mask-{index}.png") for index in range(2)]
    image_manifest = _image_manifest(tmp_path, images, "train")
    mask_manifest = _mask_manifest(tmp_path, masks, split="train", processed=False)

    dataset = ManifestInpaintDataset(
        image_manifest,
        mask_manifest,
        dataset="places2",
        image_size=64,
        seed=2026,
        augment=True,
    )
    first = dataset[0]
    second = dataset[0]

    assert first.image.data.shape == (64, 64, 3)
    assert first.mask.data.shape == (64, 64)
    assert first.mask.data.dtype == np.bool_
    assert 0.0 < first.mask.ratio < 1.0
    assert np.array_equal(first.image.data, second.image.data)
    assert np.array_equal(first.mask.data, second.mask.data)


def test_validation_dataset_uses_fixed_masks(tmp_path: Path) -> None:
    images = [
        _rgb(tmp_path / f"val-img-{index}.png", (256, 256), (20, index * 20, 40))
        for index in range(3)
    ]
    masks = [_fixed_mask(tmp_path / f"val-mask-{index}.png") for index in range(2)]
    image_manifest = _image_manifest(tmp_path, images, "val")
    mask_manifest = _mask_manifest(tmp_path, masks, split="val", processed=True)

    dataset = ManifestInpaintDataset(
        image_manifest,
        mask_manifest,
        dataset="places2",
        image_size=64,
        seed=2026,
        augment=False,
    )
    sample = dataset[0]
    assert sample.image.data.shape == (64, 64, 3)
    assert sample.mask.data.shape == (64, 64)
    assert 0.0 < sample.mask.ratio < 1.0


def test_celeba_training_uses_short_side_crop(tmp_path: Path) -> None:
    image = _rgb(tmp_path / "face.png", (320, 400), (100, 120, 140))
    mask = _train_mask(tmp_path / "mask.png")
    image_manifest = _image_manifest(tmp_path, [image], "train")
    mask_manifest = _mask_manifest(tmp_path, [mask], split="train", processed=False)

    dataset = ManifestInpaintDataset(
        image_manifest,
        mask_manifest,
        dataset="celeba_hq",
        image_size=64,
        seed=2026,
        augment=True,
    )
    sample = dataset[0]
    assert sample.image.data.shape == (64, 64, 3)
    assert sample.mask.data.shape == (64, 64)


def test_fixed_pair_dataset(tmp_path: Path) -> None:
    image = _rgb(tmp_path / "test.png", (256, 256), (10, 20, 30))
    mask = _fixed_mask(tmp_path / "test-mask.png")
    image_entry = make_entry(
        image,
        image_id="test-image",
        split="test",
        source="unit-test",
        license="test-only",
    )
    mask_entry = make_entry(
        mask,
        image_id="test-mask",
        split="test",
        source="unit-test",
        license="test-only",
    )
    mask_entry["processed_path"] = mask_entry["relative_path"]
    pair = {
        **image_entry,
        "image_id": "test-image--m0",
        "image_source_id": "test-image",
        "mask_relative_path": mask_entry["processed_path"],
        "mask_sha256": mask_entry["sha256"],
        "mask_source_id": "test-mask",
        "bucket": 0,
        "bucket_range": "[0,10%)",
    }
    pair_manifest = tmp_path / "pairs.jsonl"
    write_manifest(
        pair_manifest,
        [pair],
        kind="pairs",
        dataset="places365",
        split="test",
        seed=2026,
        source="unit-test",
        license="test-only",
    )

    dataset = FixedPairDataset(
        pair_manifest,
        dataset="places2",
        image_size=64,
    )
    sample = dataset[0]
    assert sample.image.data.shape == (64, 64, 3)
    assert sample.mask.data.shape == (64, 64)
    assert sample.meta["bucket"] == 0
    assert 0.0 < sample.mask.ratio < 1.0
