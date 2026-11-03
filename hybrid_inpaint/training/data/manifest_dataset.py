"""Datasets backed by frozen JSONL manifests.

Training reads a fixed image manifest and a pool of NVIDIA training masks.
The mask is selected and augmented deterministically from the sample index, so
the same ``(seed, index)`` always gives the same training pair.  Validation and
test read the frozen 256x256 masks produced by :mod:`prepare`.

Image policy:

* Places365: 256x256 source images are used natively (random crop only when
  ``image_size`` is smaller);
* CelebA-HQ: short side to ``image_size + 30`` then random crop for training;
  short side to ``image_size`` then centre crop for validation/test.

Masks:

* training masks (960x640, dark/gray = hole) are thresholded at 0.6, randomly
  flipped/rotated/cropped, resized nearest to the output size and dilated with
  the official 9-49 px recipe scaled to the output resolution;
* validation/test masks are pre-binarised 256x256 PNGs (white = hole).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image, ImageFilter

from hybrid_inpaint.domain.entities import Image as DomainImage
from hybrid_inpaint.domain.entities import Mask, Sample
from hybrid_inpaint.training.data.manifest import (
    PROJECT_ROOT,
    read_manifest,
    resolve_relative,
)

TRAIN_MASK_DARK_THRESHOLD = 153
TEST_MASK_WHITE_THRESHOLD = 127


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else PROJECT_ROOT / value


def load_manifest_entries(path: str | Path) -> list[dict[str, Any]]:
    """Read a manifest and attach resolved source paths."""
    entries = []
    for entry in read_manifest(_resolve(path)):
        item = dict(entry)
        item["path"] = resolve_relative(item["relative_path"])
        if "processed_path" in item:
            item["processed_path_full"] = resolve_relative(item["processed_path"])
        if "mask_relative_path" in item:
            item["mask_path"] = resolve_relative(item["mask_relative_path"])
        entries.append(item)
    return entries


def _resize_short_side(image: Image.Image, target: int) -> Image.Image:
    width, height = image.size
    scale = target / min(width, height)
    return image.resize(
        (max(1, round(width * scale)), max(1, round(height * scale))),
        Image.BICUBIC,
    )


def _random_crop(image: Image.Image, size: int, rng: np.random.Generator) -> Image.Image:
    width, height = image.size
    if width < size or height < size:
        image = _resize_short_side(image, size)
        width, height = image.size
    left = int(rng.integers(0, width - size + 1))
    top = int(rng.integers(0, height - size + 1))
    return image.crop((left, top, left + size, top + size))


def _center_crop(image: Image.Image, size: int) -> Image.Image:
    width, height = image.size
    if width < size or height < size:
        image = _resize_short_side(image, size)
        width, height = image.size
    left = (width - size) // 2
    top = (height - size) // 2
    return image.crop((left, top, left + size, top + size))


def _color_jitter(
    image: Image.Image,
    rng: np.random.Generator,
) -> Image.Image:
    array = np.asarray(image, dtype=np.float32)
    brightness = float(rng.uniform(0.9, 1.1))
    contrast = float(rng.uniform(0.9, 1.1))
    array = array * brightness
    mean = float(array.mean())
    array = (array - mean) * contrast + mean
    return Image.fromarray(np.clip(array, 0, 255).astype(np.uint8), mode="RGB")


def prepare_image(
    image: Image.Image,
    *,
    dataset: str,
    image_size: int,
    training: bool,
    rng: np.random.Generator | None = None,
) -> tuple[Image.Image, bool]:
    """Resize/crop/jitter an image; return ``(image, flipped)``."""
    if dataset == "celeba_hq" and training:
        image = _resize_short_side(image, image_size + 30)
        assert rng is not None
        image = _random_crop(image, image_size, rng)
    elif dataset == "celeba_hq":
        image = _resize_short_side(image, image_size)
        image = _center_crop(image, image_size)
    else:
        if image.size != (image_size, image_size):
            if training:
                assert rng is not None
                image = _resize_short_side(image, image_size + 30)
                image = _random_crop(image, image_size, rng)
            else:
                image = _resize_short_side(image, image_size)
                image = _center_crop(image, image_size)

    flipped = False
    if training:
        assert rng is not None
        image = _color_jitter(image, rng)
        if bool(rng.integers(0, 2)):
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            flipped = True
    return image, flipped


def _crop_square(mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    height, width = mask.shape
    side = min(height, width)
    top = int(rng.integers(0, height - side + 1))
    left = int(rng.integers(0, width - side + 1))
    return mask[top : top + side, left : left + side]


def prepare_training_mask(
    path: str | Path,
    *,
    image_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Binarise/augment one NVIDIA training mask to ``image_size x image_size``."""
    with Image.open(path) as image:
        hole = np.asarray(image.convert("L")) < TRAIN_MASK_DARK_THRESHOLD

    hole = hole[:, ::-1] if bool(rng.integers(0, 2)) else hole
    hole = np.rot90(hole, int(rng.integers(0, 4)))
    hole = _crop_square(hole, rng)

    mask_image = Image.fromarray((hole.astype(np.uint8) * 255), mode="L").resize(
        (image_size, image_size), Image.NEAREST
    )
    low = max(1, int(round(image_size * 0.008)))  # ~9 px at 960 input, scaled
    high = max(low + 1, int(round(image_size * 0.05)))  # ~49 px at 960 input
    radius = int(rng.integers(low, high + 1))
    if radius > 0:
        mask_image = mask_image.filter(ImageFilter.MaxFilter(2 * radius + 1))
    return np.asarray(mask_image) > 127


def prepare_fixed_mask(path: str | Path, *, image_size: int) -> np.ndarray:
    """Load a preprocessed white-hole mask and resize it if necessary."""
    with Image.open(path) as image:
        array = np.asarray(image.convert("L")) > TEST_MASK_WHITE_THRESHOLD
    if array.shape != (image_size, image_size):
        resized = Image.fromarray((array.astype(np.uint8) * 255), mode="L").resize(
            (image_size, image_size), Image.NEAREST
        )
        array = np.asarray(resized) > 127
    return array


class ManifestInpaintDataset:
    """Training/validation dataset over frozen image and mask manifests."""

    def __init__(
        self,
        image_manifest: str | Path,
        mask_manifest: str | Path,
        *,
        dataset: str,
        image_size: int = 256,
        seed: int = 2026,
        augment: bool = True,
        limit: int | None = None,
    ) -> None:
        if dataset not in ("places2", "celeba_hq"):
            raise ValueError("dataset must be places2 or celeba_hq")
        if image_size < 32:
            raise ValueError("image_size must be >= 32")
        if limit is not None and limit < 1:
            raise ValueError("limit must be >= 1 or None")

        self.images = load_manifest_entries(image_manifest)
        self.masks = load_manifest_entries(mask_manifest)
        if not self.images or not self.masks:
            raise ValueError("image and mask manifests must not be empty")
        if limit is not None:
            self.images = self.images[:limit]
        self.dataset = dataset
        self.image_size = image_size
        self.seed = seed
        self.augment = augment
        self.mask_key = "relative_path" if augment else "processed_path"

    def __len__(self) -> int:
        return len(self.images)

    def _sample_mask(self, index: int, rng: np.random.Generator) -> tuple[np.ndarray, dict[str, Any]]:
        if self.augment:
            for _ in range(12):
                mask_entry = self.masks[int(rng.integers(0, len(self.masks)))]
                hole = prepare_training_mask(
                    mask_entry["path"], image_size=self.image_size, rng=rng
                )
                area = float(hole.mean())
                if 0.0 < area < 1.0:
                    return hole, mask_entry
        else:
            mask_entry = self.masks[index % len(self.masks)]
            hole = prepare_fixed_mask(
                mask_entry["processed_path_full"], image_size=self.image_size
            )
            return hole, mask_entry
        raise RuntimeError("could not generate a non-empty, non-full training mask")

    def __getitem__(self, index: int) -> Sample:
        if not 0 <= index < len(self.images):
            raise IndexError(index)
        image_entry = self.images[index]
        rng = np.random.default_rng(
            np.random.SeedSequence([self.seed, 0xDA, index])
        )

        with Image.open(image_entry["path"]) as image:
            image = image.convert("RGB")
            image, flipped = prepare_image(
                image,
                dataset=self.dataset,
                image_size=self.image_size,
                training=self.augment,
                rng=rng,
            )

        hole, mask_entry = self._sample_mask(index, rng)
        if flipped:
            hole = hole[:, ::-1]

        sample_meta: dict[str, Any] = {
            "dataset": self.dataset,
            "split": image_entry.get("split"),
            "image_id": image_entry.get("image_id"),
            "mask_id": mask_entry.get("image_id"),
            "category_id": image_entry.get("category_id"),
            "category": image_entry.get("category"),
            "mask_bucket": mask_entry.get("bucket"),
            "mask_area": float(hole.mean()),
        }
        return Sample(
            sample_id=str(image_entry.get("image_id", f"sample-{index}")),
            image=DomainImage(np.asarray(image, dtype=np.uint8)),
            mask=Mask(hole.astype(np.bool_, copy=False)),
            meta=sample_meta,
        )


class FixedPairDataset:
    """Fixed test pairs produced by ``prepare.build_test_pairs``."""

    def __init__(
        self,
        pair_manifest: str | Path,
        *,
        dataset: str,
        image_size: int = 256,
        limit: int | None = None,
    ) -> None:
        if dataset not in ("places2", "celeba_hq"):
            raise ValueError("dataset must be places2 or celeba_hq")
        if limit is not None and limit < 1:
            raise ValueError("limit must be >= 1 or None")
        self.entries = load_manifest_entries(pair_manifest)
        if not self.entries:
            raise ValueError("pair manifest must not be empty")
        if limit is not None:
            self.entries = self.entries[:limit]
        self.dataset = dataset
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> Sample:
        if not 0 <= index < len(self.entries):
            raise IndexError(index)
        entry = self.entries[index]
        with Image.open(entry["path"]) as image:
            image = image.convert("RGB")
            image, _ = prepare_image(
                image,
                dataset=self.dataset,
                image_size=self.image_size,
                training=False,
            )
        hole = prepare_fixed_mask(entry["mask_path"], image_size=self.image_size)
        return Sample(
            sample_id=str(entry["image_id"]),
            image=DomainImage(np.asarray(image, dtype=np.uint8)),
            mask=Mask(hole.astype(np.bool_, copy=False)),
            meta={
                "dataset": self.dataset,
                "split": "test",
                "image_id": entry.get("image_source_id"),
                "mask_id": entry.get("mask_source_id"),
                "bucket": entry.get("bucket"),
                "bucket_range": entry.get("bucket_range"),
                "category_id": entry.get("category_id"),
                "category": entry.get("category"),
                "mask_area": float(hole.mean()),
            },
        )
