from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from hybrid_inpaint.training.data.masks import (
    make_brush_mask,
    make_rect_mask,
    resize_mask_nearest,
)
from hybrid_inpaint.training.data.synthetic import (
    ARCHETYPES,
    SyntheticInpaintDataset,
    frozen_samples,
    generate_synthetic_image,
    save_frozen_set,
)


@pytest.mark.parametrize("archetype", ARCHETYPES)
def test_archetype_shapes_and_metadata(archetype: str) -> None:
    rng = np.random.default_rng(1234)
    image, meta = generate_synthetic_image(archetype, 64, rng)

    assert image.shape == (64, 64, 3)
    assert image.dtype == np.uint8
    assert image.min() >= 0 and image.max() <= 255
    assert meta["archetype"] == archetype

    if archetype in ("face", "portrait"):
        assert meta["has_face"] is True
        assert meta["scene_type"] is None
        assert meta["symmetry_axis_x"] > 0
        assert len(meta["face_center"]) == 2
    else:
        assert meta["has_face"] is False
        assert meta["scene_type"] == archetype
        assert meta["horizon_y"] > 0


def test_generation_is_deterministic() -> None:
    first, first_meta = generate_synthetic_image("face", 64, np.random.default_rng(7))
    second, second_meta = generate_synthetic_image("face", 64, np.random.default_rng(7))

    assert np.array_equal(first, second)
    assert first_meta == second_meta


@pytest.mark.parametrize("archetype", ARCHETYPES)
def test_images_have_structure(archetype: str) -> None:
    image, _ = generate_synthetic_image(archetype, 128, np.random.default_rng(3))
    assert image.std() > 2.0
    assert len(np.unique(image.reshape(-1, 3), axis=0)) > 10


def test_rect_mask_ratio_is_close_to_target() -> None:
    mask = make_rect_mask(128, 0.25, np.random.default_rng(0))
    assert mask.dtype == np.bool_
    assert 0.18 <= mask.mean() <= 0.32


def test_brush_mask_is_non_empty() -> None:
    mask = make_brush_mask(128, 0.2, np.random.default_rng(1))
    assert mask.dtype == np.bool_
    assert mask.any()
    assert not mask.all()


def test_resize_mask_nearest_preserves_binary_values() -> None:
    mask = np.zeros((4, 4), dtype=np.bool_)
    mask[0:2, 0:2] = True
    resized = resize_mask_nearest(mask, 8, 8)

    assert resized.shape == (8, 8)
    assert resized.dtype == np.bool_
    assert resized[0:4, 0:4].all()
    assert not resized[4:, :].any()


def test_dataset_is_reproducible() -> None:
    first = SyntheticInpaintDataset(size=64, length=8, seed=2026)[3]
    second = SyntheticInpaintDataset(size=64, length=8, seed=2026)[3]

    assert first.sample_id == second.sample_id
    assert np.array_equal(first.image.data, second.image.data)
    assert np.array_equal(first.mask.data, second.mask.data)
    assert first.meta == second.meta


def test_dataset_rejects_invalid_size() -> None:
    with pytest.raises(ValueError, match="multiple of 16"):
        SyntheticInpaintDataset(size=65, length=4)


def test_frozen_samples_are_stable() -> None:
    first = frozen_samples(size=64, count=4, seed=2026)
    second = frozen_samples(size=64, count=4, seed=2026)

    assert [sample.sample_id for sample in first] == [
        "frozen-0000",
        "frozen-0001",
        "frozen-0002",
        "frozen-0003",
    ]
    for left, right in zip(first, second):
        assert np.array_equal(left.image.data, right.image.data)
        assert np.array_equal(left.mask.data, right.mask.data)


def test_save_frozen_set_writes_manifest(tmp_path: Path) -> None:
    manifest = save_frozen_set(tmp_path / "frozen", size=64, count=4, seed=2026)

    assert manifest["count"] == 4
    manifest_path = tmp_path / "frozen" / "frozen.json"
    assert manifest_path.exists()

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert len(payload["samples"]) == 4
    for sample in payload["samples"]:
        assert (tmp_path / "frozen" / sample["image"]).exists()
        assert (tmp_path / "frozen" / sample["mask"]).exists()
        assert len(sample["image_sha256"]) == 64
        assert len(sample["mask_sha256"]) == 64
