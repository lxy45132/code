from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from hybrid_inpaint.config import AppConfig, DataConfig, LossConfig, TrainConfig
from hybrid_inpaint.training.data.factory import build_training_datasets
from hybrid_inpaint.training.data.manifest import make_entry, write_manifest


def _rgb(path: Path, size: tuple[int, int], value: int) -> Path:
    Image.new("RGB", size, color=(value, value, value)).save(path)
    return path


def _train_mask(path: Path) -> Path:
    array = np.full((640, 960), 255, dtype=np.uint8)
    array[300:340, 450:500] = 0
    Image.fromarray(array, mode="L").save(path)
    return path


def _val_mask(path: Path) -> Path:
    array = np.zeros((256, 256), dtype=np.uint8)
    array[100:150, 100:160] = 255
    Image.fromarray(array, mode="L").save(path)
    return path


def _write_images(tmp_path: Path, files: list[Path], name: str, split: str) -> Path:
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
    out = tmp_path / name
    write_manifest(
        out,
        entries,
        kind="images",
        dataset="unit",
        split=split,
        seed=2026,
        source="unit-test",
        license="test-only",
    )
    return out


def _write_masks(
    tmp_path: Path,
    files: list[Path],
    name: str,
    split: str,
    *,
    processed: bool,
) -> Path:
    entries = []
    for index, path in enumerate(files):
        entry = make_entry(
            path,
            image_id=f"mask-{index}",
            split=split,
            source="unit-test",
            license="test-only",
        )
        if processed:
            entry["processed_path"] = entry["relative_path"]
            entry["processed_sha256"] = entry["sha256"]
            entry["area_256"] = 0.02
        entries.append(entry)
    out = tmp_path / name
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


def test_places_manifest_factory(tmp_path: Path) -> None:
    train_images = [
        _rgb(tmp_path / f"train-{index}.png", (256, 256), 20 + index)
        for index in range(2)
    ]
    val_images = [_rgb(tmp_path / "val-0.png", (256, 256), 90)]
    train_masks = [_train_mask(tmp_path / f"train-mask-{index}.png") for index in range(2)]
    val_masks = [_val_mask(tmp_path / "val-mask-0.png")]

    _write_images(tmp_path, train_images, "places365_train.jsonl", "train")
    _write_images(tmp_path, val_images, "places365_val.jsonl", "val")
    _write_masks(tmp_path, train_masks, "masks_train.jsonl", "train", processed=False)
    _write_masks(tmp_path, val_masks, "masks_val.jsonl", "val", processed=True)

    config = AppConfig(
        experiment_id="factory_test",
        seed=2026,
        data=DataConfig(
            dataset="places2",
            image_size=64,
            manifest_dir=tmp_path,
            processed_dir=tmp_path,
            manifest_train_limit=2,
            manifest_val_limit=1,
        ),
        train=TrainConfig(
            iterations=1,
            batch_size=1,
            device="cpu",
            output_dir=tmp_path / "out",
            val_every=1,
            log_every=1,
            loss=LossConfig(perceptual=0.0),
        ),
    )

    train, val, identity = build_training_datasets(config)
    assert len(train) == 2
    assert len(val) == 1
    sample = train[0]
    assert sample.image.data.shape == (64, 64, 3)
    assert sample.mask.data.shape == (64, 64)
    assert identity["dataset"] == "places2"
    assert identity["train_samples"] == 2
    assert len(identity["train_manifest"]) == 64
