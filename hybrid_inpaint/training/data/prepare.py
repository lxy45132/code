"""Build the frozen T0 manifests for Places365, CelebA-HQ and masks.

This module reads the official file lists and raw directories, samples the
fixed quotas, hashes every file and writes JSONL manifests plus hashed
summaries.  It is intentionally deterministic: the same inputs and seed yield
the same manifest SHA256.

Layout expected under ``data/raw``::

    places365/filelists/{places365_train_standard,places365_val,places365_test,categories_places365}.txt
    places365/archives/b/data_256/b/<category>/...
    places365/archives/val_256/val_256/*.jpg
    places365/archives/test_256/test_256/*.jpg
    face/CelebAHQ-1024x1024/*.jpg
    mask/irregular_mask/disocclusion_img_mask/*.png
    mask/test_mask/mask/testing_mask_dataset/*.png

Test masks are preprocessed to 256x256 (nearest neighbour) under
``data/processed/masks/test_256/bucket_<i>/``; validation masks are
preprocessed from the training masks under ``data/processed/masks/val_256/``.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterator

import numpy as np
from PIL import Image, ImageFilter

from hybrid_inpaint.training.data.manifest import (
    PROJECT_ROOT,
    build_entries,
    read_manifest,
    relative_path,
    sha256_file,
    write_manifest,
)

PLACES_CLASSES = (
    "badlands",
    "bamboo forest",
    "beach",
    "botanical garden",
    "butte",
    "bridge",
    "building facade",
    "balcony/exterior",
    "barn",
    "boathouse",
    "boardwalk",
    "bazaar/outdoor",
    "bus station/indoor",
    "bedroom",
    "bathroom",
    "bookstore",
    "bowling alley",
    "ballroom",
    "bakery/shop",
    "bar",
)
PLACES_TRAIN_PER_CLASS = 2500
PLACES_VAL_PER_CLASS = 100
PLACES_TEST_COUNT = 2000
PLACES_SEED_TAG = 7001

CELEBA_TRAIN = 25500
CELEBA_VAL = 1500
CELEBA_TEST = 3000
CELEBA_SEED_TAG = 8001

MASK_BUCKETS: tuple[tuple[float, float, str], ...] = (
    (0.0, 0.1, "[0,10%)"),
    (0.1, 0.2, "[10,20%)"),
    (0.2, 0.3, "[20,30%)"),
    (0.3, 0.4, "[30,40%)"),
    (0.4, 0.6001, "[40,60%]"),
)
TRAIN_MASK_DARK_THRESHOLD = 153  # gray/black are holes in the NVIDIA training masks
TEST_MASK_WHITE_THRESHOLD = 127  # white pixels are holes in the official test masks
VAL_MASK_COUNT = 2000
MASK_SEED_TAG = 9001

PLACES_SOURCE = "Places365-Standard"
PLACES_LICENSE = "Places2 terms: non-commercial research/education, no redistribution"
CELEBA_SOURCE = "CelebA-HQ"
CELEBA_LICENSE = "CelebA/CelebA-HQ non-commercial research terms"
TRAIN_MASK_SOURCE = "NVIDIA Irregular Mask Dataset (training)"
TRAIN_MASK_LICENSE = "NVIDIA Irregular Mask Dataset: research only"
TEST_MASK_SOURCE = "NVIDIA Irregular Mask Dataset (testing)"
TEST_MASK_LICENSE = "CC BY-NC-SA 4.0"


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else PROJECT_ROOT / value


def _read_places_list(path: Path) -> Iterator[tuple[str, int | None]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.split()
            if not parts:
                continue
            yield parts[0], (int(parts[1]) if len(parts) > 1 else None)


def _category_lookup(categories_file: Path) -> dict[str, tuple[int, str]]:
    """Map a display name such as ``bamboo forest`` to ``(id, 'b/bamboo_forest')``."""
    lookup: dict[str, tuple[int, str]] = {}
    with categories_file.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().rsplit(" ", 1)
            if len(parts) != 2:
                continue
            category_path = parts[0].lstrip("/")
            label = int(parts[1])
            segments = category_path.split("/")[1:]
            display = "/".join(segment.replace("_", " ") for segment in segments)
            lookup[display] = (label, category_path)
    return lookup


def _places_rel_to_path(base: Path, relative: str) -> Path:
    return base / relative.lstrip("/")


def prepare_places(
    raw_dir: str | Path,
    manifest_dir: str | Path,
    *,
    seed: int = 2026,
) -> dict[str, Any]:
    raw = _resolve(raw_dir)
    manifests = _resolve(manifest_dir)
    filelists = raw / "places365" / "filelists"
    lookup = _category_lookup(filelists / "categories_places365.txt")

    missing_classes = [name for name in PLACES_CLASSES if name not in lookup]
    if missing_classes:
        raise ValueError(f"unknown Places365 classes: {missing_classes}")
    selected = {name: lookup[name] for name in PLACES_CLASSES}
    id_to_name = {label: name for name, (label, _) in selected.items()}
    wanted = set(id_to_name)

    train_base = raw / "places365" / "archives" / "b" / "data_256"
    val_base = raw / "places365" / "archives" / "val_256" / "val_256"
    test_base = raw / "places365" / "archives" / "test_256" / "test_256"

    # ---- train: exactly 2,500 images per selected class ---------------------
    train_by_id: dict[int, list[str]] = {label: [] for label in wanted}
    for relative, label in _read_places_list(filelists / "places365_train_standard.txt"):
        if label in train_by_id:
            train_by_id[label].append(relative)

    train_specs: list[dict[str, Any]] = []
    for label in sorted(train_by_id):
        relative_paths = sorted(train_by_id[label])
        if len(relative_paths) < PLACES_TRAIN_PER_CLASS:
            raise ValueError(
                f"class {id_to_name[label]!r} only has {len(relative_paths)} train images"
            )
        rng = np.random.default_rng(np.random.SeedSequence([seed, 100 + label]))
        chosen = rng.choice(
            len(relative_paths), size=PLACES_TRAIN_PER_CLASS, replace=False
        )
        for position in sorted(int(item) for item in chosen):
            relative = relative_paths[position]
            train_specs.append(
                {
                    "path": _places_rel_to_path(train_base, relative),
                    "image_id": f"places365-train-{label:03d}-{Path(relative).stem}",
                    "split": "train",
                    "source": PLACES_SOURCE,
                    "license": PLACES_LICENSE,
                    "width": 256,
                    "height": 256,
                    "category_id": label,
                    "category": id_to_name[label],
                }
            )

    train_entries = build_entries(
        train_specs, progress_every=5000, label="places train"
    )
    train_summary = write_manifest(
        manifests / "places365_train.jsonl",
        train_entries,
        kind="images",
        dataset="places365",
        split="train",
        seed=seed,
        source=PLACES_SOURCE,
        license=PLACES_LICENSE,
        extra={
            "classes": [
                {"id": label, "name": name}
                for name, (label, _) in sorted(selected.items(), key=lambda item: item[1][0])
            ],
            "train_per_class": PLACES_TRAIN_PER_CLASS,
        },
    )

    # ---- validation: all 100 official val images for each selected class ----
    val_by_id: dict[int, list[str]] = {label: [] for label in wanted}
    for relative, label in _read_places_list(filelists / "places365_val.txt"):
        if label in val_by_id:
            val_by_id[label].append(relative)

    val_specs: list[dict[str, Any]] = []
    for label in sorted(val_by_id):
        relative_paths = sorted(val_by_id[label])
        if not relative_paths:
            raise ValueError(f"class {id_to_name[label]!r} has no validation images")
        for relative in relative_paths:
            val_specs.append(
                {
                    "path": val_base / relative,
                    "image_id": f"places365-val-{Path(relative).stem}",
                    "split": "val",
                    "source": PLACES_SOURCE,
                    "license": PLACES_LICENSE,
                    "width": 256,
                    "height": 256,
                    "category_id": label,
                    "category": id_to_name[label],
                }
            )

    val_entries = build_entries(val_specs, progress_every=2000, label="places val")
    val_summary = write_manifest(
        manifests / "places365_val.jsonl",
        val_entries,
        kind="images",
        dataset="places365",
        split="val",
        seed=seed,
        source=PLACES_SOURCE,
        license=PLACES_LICENSE,
        extra={"val_per_class": PLACES_VAL_PER_CLASS},
    )

    # ---- test: fixed random subset; official test labels are unavailable ----
    test_names = [
        relative
        for relative, _ in _read_places_list(filelists / "places365_test.txt")
    ]
    rng = np.random.default_rng(np.random.SeedSequence([seed, PLACES_SEED_TAG]))
    chosen = rng.choice(
        len(test_names), size=min(PLACES_TEST_COUNT, len(test_names)), replace=False
    )
    test_specs: list[dict[str, Any]] = []
    for position in sorted(int(item) for item in chosen):
        relative = test_names[position]
        test_specs.append(
            {
                "path": test_base / relative,
                "image_id": f"places365-test-{Path(relative).stem}",
                "split": "test",
                "source": PLACES_SOURCE,
                "license": PLACES_LICENSE,
                "width": 256,
                "height": 256,
                "category_id": None,
                "category": None,
            }
        )

    test_entries = build_entries(test_specs, progress_every=2000, label="places test")
    test_summary = write_manifest(
        manifests / "places365_test.jsonl",
        test_entries,
        kind="images",
        dataset="places365",
        split="test",
        seed=seed,
        source=PLACES_SOURCE,
        license=PLACES_LICENSE,
        extra={
            "test_count": len(test_entries),
            "test_labels": "not released by Places365; class-wise tables use val",
        },
    )

    return {
        "train": train_summary,
        "val": val_summary,
        "test": test_summary,
    }


def prepare_celeba(
    raw_dir: str | Path,
    manifest_dir: str | Path,
    *,
    seed: int = 2026,
) -> dict[str, Any]:
    raw = _resolve(raw_dir)
    manifests = _resolve(manifest_dir)
    image_dir = raw / "face" / "CelebAHQ-1024x1024"
    files = sorted(image_dir.glob("*.jpg"))
    if len(files) != CELEBA_TRAIN + CELEBA_VAL + CELEBA_TEST:
        raise ValueError(
            f"expected {CELEBA_TRAIN + CELEBA_VAL + CELEBA_TEST} CelebA-HQ images, "
            f"found {len(files)}"
        )

    rng = np.random.default_rng(np.random.SeedSequence([seed, CELEBA_SEED_TAG]))
    permutation = rng.permutation(len(files))
    split_indices = {
        "test": permutation[:CELEBA_TEST],
        "val": permutation[CELEBA_TEST : CELEBA_TEST + CELEBA_VAL],
        "train": permutation[CELEBA_TEST + CELEBA_VAL :],
    }

    summaries: dict[str, Any] = {}
    for split, indices in split_indices.items():
        specs: list[dict[str, Any]] = []
        for position in sorted(int(item) for item in indices):
            path = files[position]
            specs.append(
                {
                    "path": path,
                    "image_id": f"celeba-{split}-{path.stem}",
                    "split": split,
                    "source": CELEBA_SOURCE,
                    "license": CELEBA_LICENSE,
                    "width": 1024,
                    "height": 1024,
                    "original_id": path.stem,
                }
            )
        entries = build_entries(
            specs, progress_every=5000, label=f"celeba {split}"
        )
        summaries[split] = write_manifest(
            manifests / f"celeba_hq_{split}.jsonl",
            entries,
            kind="images",
            dataset="celeba_hq",
            split=split,
            seed=seed,
            source=CELEBA_SOURCE,
            license=CELEBA_LICENSE,
            extra={
                "split_method": "seeded permutation, frozen before training",
                "counts": {
                    "train": CELEBA_TRAIN,
                    "val": CELEBA_VAL,
                    "test": CELEBA_TEST,
                },
            },
        )
    return summaries


def _mask_bucket_index(area: float) -> int:
    for index, (low, high, _) in enumerate(MASK_BUCKETS):
        if low <= area < high:
            return index
    return len(MASK_BUCKETS) - 1


def _square_around_hole(array: np.ndarray) -> np.ndarray:
    """Crop the largest square that keeps the hole centroid inside."""
    height, width = array.shape
    side = min(height, width)
    rows, columns = np.nonzero(array)
    if rows.size == 0:
        raise ValueError("validation mask has no hole before cropping")
    center_row = int(rows.mean())
    center_column = int(columns.mean())
    top = min(max(0, center_row - side // 2), height - side)
    left = min(max(0, center_column - side // 2), width - side)
    cropped = array[top : top + side, left : left + side]
    if not cropped.any():
        raise ValueError("validation mask became empty after cropping")
    return cropped


def _train_mask_info(path: Path) -> dict[str, Any]:
    with Image.open(path) as image:
        array = np.asarray(image.convert("L"))
        width, height = image.size
    hole = array < TRAIN_MASK_DARK_THRESHOLD
    return {
        "path": path,
        "width": width,
        "height": height,
        "area": float(hole.mean()),
        "sha256": sha256_file(path),
    }


def _preprocess_val_mask(path: Path, out_path: Path) -> dict[str, Any]:
    with Image.open(path) as image:
        hole = np.asarray(image.convert("L")) < TRAIN_MASK_DARK_THRESHOLD
    hole = _square_around_hole(hole)
    processed = Image.fromarray((hole.astype(np.uint8) * 255), mode="L").resize(
        (256, 256), Image.NEAREST
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    processed.save(out_path)
    area = np.asarray(processed) > 127
    if not area.any():
        # Very small holes can disappear under nearest-neighbour downsampling;
        # one 3x3 max-filter step keeps the validation region non-empty.
        processed = processed.filter(ImageFilter.MaxFilter(3))
        processed.save(out_path)
        area = np.asarray(processed) > 127
    if not area.any():
        raise ValueError(f"validation mask {path} became empty after resizing")
    return {
        "processed_path": relative_path(out_path),
        "processed_sha256": sha256_file(out_path),
        "area_256": float(area.mean()),
        "width_256": 256,
        "height_256": 256,
    }


def prepare_masks(
    raw_dir: str | Path,
    manifest_dir: str | Path,
    processed_dir: str | Path,
    *,
    seed: int = 2026,
) -> dict[str, Any]:
    raw = _resolve(raw_dir)
    manifests = _resolve(manifest_dir)
    processed = _resolve(processed_dir)

    train_mask_dir = raw / "mask" / "irregular_mask" / "disocclusion_img_mask"
    test_mask_dir = raw / "mask" / "test_mask" / "mask" / "testing_mask_dataset"
    train_files = sorted(train_mask_dir.glob("*.png"))
    test_files = sorted(test_mask_dir.glob("*.png"))
    if not train_files or not test_files:
        raise FileNotFoundError(
            f"mask directories are empty: {train_mask_dir}, {test_mask_dir}"
        )

    # ---- training/validation masks -----------------------------------------
    with ThreadPoolExecutor(max_workers=16) as executor:
        infos = []
        for index, info in enumerate(
            executor.map(_train_mask_info, train_files), 1
        ):
            infos.append(info)
            if index % 5000 == 0:
                print(f"  train masks: {index}/{len(train_files)}", flush=True)

    rng = np.random.default_rng(np.random.SeedSequence([seed, MASK_SEED_TAG]))
    eligible = [
        position
        for position, info in enumerate(infos)
        if 0.005 <= info["area"] < 0.6
    ]
    if len(eligible) < VAL_MASK_COUNT:
        raise ValueError(
            f"only {len(eligible)} masks with area in (0, 0.6); "
            f"need {VAL_MASK_COUNT} validation masks"
        )
    val_positions = set(
        int(item)
        for item in rng.choice(
            eligible, size=VAL_MASK_COUNT, replace=False
        )
    )

    train_entries: list[dict[str, Any]] = []
    val_entries: list[dict[str, Any]] = []
    for position, info in enumerate(infos):
        mask_id = f"mask-{info['path'].stem}"
        base = {
            "image_id": mask_id,
            "relative_path": relative_path(info["path"]),
            "sha256": info["sha256"],
            "width": info["width"],
            "height": info["height"],
            "source": TRAIN_MASK_SOURCE,
            "license": TRAIN_MASK_LICENSE,
            "area": info["area"],
        }
        if position in val_positions:
            out_path = (
                processed / "masks" / "val_256" / f"{info['path'].stem}.png"
            )
            extra = _preprocess_val_mask(info["path"], out_path)
            val_entries.append({**base, **extra, "split": "val"})
        else:
            train_entries.append({**base, "split": "train"})

    train_mask_summary = write_manifest(
        manifests / "masks_train.jsonl",
        train_entries,
        kind="masks",
        dataset="nvidia_irregular",
        split="train",
        seed=seed,
        source=TRAIN_MASK_SOURCE,
        license=TRAIN_MASK_LICENSE,
        extra={
            "threshold_dark": TRAIN_MASK_DARK_THRESHOLD,
            "note": "dark/gray pixels are holes; augment with dilation/rotation/crop",
        },
    )
    val_mask_summary = write_manifest(
        manifests / "masks_val.jsonl",
        val_entries,
        kind="masks",
        dataset="nvidia_irregular",
        split="val",
        seed=seed,
        source=TRAIN_MASK_SOURCE,
        license=TRAIN_MASK_LICENSE,
        extra={
            "count": len(val_entries),
            "preprocess": "square crop around hole centroid -> nearest 256x256",
            "area_256_min": min(entry["area_256"] for entry in val_entries),
            "area_256_max": max(entry["area_256"] for entry in val_entries),
        },
    )

    # ---- official test masks, bucketed by final 256x256 area ----------------
    test_entries: list[dict[str, Any]] = []
    bucket_counts = [0] * len(MASK_BUCKETS)
    for index, path in enumerate(test_files, 1):
        with Image.open(path) as image:
            hole = np.asarray(image.convert("L")) > TEST_MASK_WHITE_THRESHOLD
        area_512 = float(hole.mean())
        processed_image = Image.fromarray(
            (hole.astype(np.uint8) * 255), mode="L"
        ).resize((256, 256), Image.NEAREST)
        area_256 = float(np.asarray(processed_image).mean() / 255.0)
        bucket = _mask_bucket_index(area_256)
        out_path = (
            processed
            / "masks"
            / "test_256"
            / f"bucket_{bucket}"
            / f"{path.stem}.png"
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        processed_image.save(out_path)
        bucket_counts[bucket] += 1
        test_entries.append(
            {
                "image_id": f"test-mask-{path.stem}",
                "relative_path": relative_path(path),
                "sha256": sha256_file(path),
                "width": 512,
                "height": 512,
                "split": "test",
                "source": TEST_MASK_SOURCE,
                "license": TEST_MASK_LICENSE,
                "area_512": area_512,
                "area_256": area_256,
                "bucket": bucket,
                "bucket_range": MASK_BUCKETS[bucket][2],
                "processed_path": relative_path(out_path),
                "processed_sha256": sha256_file(out_path),
                "width_256": 256,
                "height_256": 256,
            }
        )
        if index % 2000 == 0:
            print(f"  test masks: {index}/{len(test_files)}", flush=True)

    test_mask_summary = write_manifest(
        manifests / "masks_test.jsonl",
        test_entries,
        kind="masks",
        dataset="nvidia_irregular",
        split="test",
        seed=seed,
        source=TEST_MASK_SOURCE,
        license=TEST_MASK_LICENSE,
        extra={
            "threshold_white": TEST_MASK_WHITE_THRESHOLD,
            "buckets": [
                {"index": index, "range": label, "count": bucket_counts[index]}
                for index, (_, _, label) in enumerate(MASK_BUCKETS)
            ],
            "preprocess": "nearest 512x512 -> 256x256, white=hole",
        },
    )

    return {
        "train": train_mask_summary,
        "val": val_mask_summary,
        "test": test_mask_summary,
        "bucket_counts": bucket_counts,
    }


def build_test_pairs(
    image_manifest: str | Path,
    mask_manifest: str | Path,
    output_path: str | Path,
    *,
    dataset: str,
    seed: int = 2026,
    pairs_per_image: int = 5,
) -> dict[str, Any]:
    """Pair every test image with one fixed mask from each area bucket."""
    images = read_manifest(_resolve(image_manifest))
    masks = read_manifest(_resolve(mask_manifest))
    by_bucket: dict[int, list[dict[str, Any]]] = {
        index: [] for index in range(len(MASK_BUCKETS))
    }
    for mask in masks:
        by_bucket[int(mask["bucket"])].append(mask)
    empty = [index for index, pool in by_bucket.items() if not pool]
    if empty:
        raise ValueError(f"test mask buckets are empty: {empty}")
    if pairs_per_image != len(MASK_BUCKETS):
        raise ValueError("pairs_per_image must equal the number of mask buckets")

    dataset_tag = 11001 if dataset == "places365" else 22002
    pairs: list[dict[str, Any]] = []
    for image_index, image in enumerate(images):
        for bucket in range(len(MASK_BUCKETS)):
            pool = by_bucket[bucket]
            rng = np.random.default_rng(
                np.random.SeedSequence([seed, dataset_tag, image_index, bucket])
            )
            mask = pool[int(rng.integers(0, len(pool)))]
            pairs.append(
                {
                    "image_id": f"{image['image_id']}--m{bucket}",
                    "relative_path": image["relative_path"],
                    "sha256": image["sha256"],
                    "width": image["width"],
                    "height": image["height"],
                    "split": "test",
                    "source": image["source"],
                    "license": image["license"],
                    "image_source_id": image["image_id"],
                    "mask_relative_path": mask["processed_path"],
                    "mask_sha256": mask["processed_sha256"],
                    "mask_source_id": mask["image_id"],
                    "mask_area_256": mask["area_256"],
                    "bucket": bucket,
                    "bucket_range": MASK_BUCKETS[bucket][2],
                }
            )

    return write_manifest(
        _resolve(output_path),
        pairs,
        kind="pairs",
        dataset=dataset,
        split="test",
        seed=seed,
        source=images[0]["source"] if images else "unknown",
        license=images[0]["license"] if images else "unknown",
        extra={
            "pairs_per_image": pairs_per_image,
            "images": len(images),
            "masks": len(masks),
            "buckets": [label for _, _, label in MASK_BUCKETS],
        },
    )


def prepare_all(
    raw_dir: str | Path = "data/raw",
    manifest_dir: str | Path = "data/manifests",
    processed_dir: str | Path = "data/processed",
    *,
    seed: int = 2026,
) -> dict[str, Any]:
    print("[prepare] Places365", flush=True)
    places = prepare_places(raw_dir, manifest_dir, seed=seed)
    print("[prepare] CelebA-HQ", flush=True)
    celeba = prepare_celeba(raw_dir, manifest_dir, seed=seed)
    print("[prepare] masks", flush=True)
    masks = prepare_masks(raw_dir, manifest_dir, processed_dir, seed=seed)
    manifests = _resolve(manifest_dir)

    print("[prepare] test pairs", flush=True)
    places_pairs = build_test_pairs(
        manifests / "places365_test.jsonl",
        manifests / "masks_test.jsonl",
        manifests / "places365_test_pairs.jsonl",
        dataset="places365",
        seed=seed,
    )
    celeba_pairs = build_test_pairs(
        manifests / "celeba_hq_test.jsonl",
        manifests / "masks_test.jsonl",
        manifests / "celeba_hq_test_pairs.jsonl",
        dataset="celeba_hq",
        seed=seed,
    )

    return {
        "places": places,
        "celeba": celeba,
        "masks": masks,
        "places_pairs": places_pairs,
        "celeba_pairs": celeba_pairs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--manifest-dir", default="data/manifests")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    summaries = prepare_all(
        args.raw_dir,
        args.manifest_dir,
        args.processed_dir,
        seed=args.seed,
    )
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
