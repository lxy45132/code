from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from hybrid_inpaint.training.data.manifest import (
    make_entry,
    read_manifest,
    read_summary,
    sha256_file,
    write_manifest,
)


def _image(path: Path, color: tuple[int, int, int] = (10, 20, 30)) -> Path:
    Image.new("RGB", (8, 6), color=color).save(path)
    return path


def test_make_entry_and_manifest_round_trip(tmp_path: Path) -> None:
    image = _image(tmp_path / "a.png")
    entry = make_entry(
        image,
        image_id="a",
        split="train",
        source="unit-test",
        license="test-only",
        category_id=3,
    )
    manifest = tmp_path / "images.jsonl"
    summary = write_manifest(
        manifest,
        [entry],
        kind="images",
        dataset="unit",
        split="train",
        seed=2026,
        source="unit-test",
        license="test-only",
    )

    loaded = read_manifest(manifest)
    assert loaded[0]["image_id"] == "a"
    assert loaded[0]["sha256"] == sha256_file(image)
    assert loaded[0]["width"] == 8
    assert loaded[0]["height"] == 6
    assert loaded[0]["category_id"] == 3
    assert summary["count"] == 1
    assert summary["sha256"] == sha256_file(manifest)
    assert read_summary(manifest)["sha256"] == summary["sha256"]


def test_manifest_is_deterministic(tmp_path: Path) -> None:
    entries = [
        make_entry(
            _image(tmp_path / f"{index}.png"),
            image_id=f"image-{index}",
            split="train",
            source="unit-test",
            license="test-only",
        )
        for index in range(3)
    ]
    first = write_manifest(
        tmp_path / "a.jsonl",
        list(reversed(entries)),
        kind="images",
        dataset="unit",
        split="train",
        seed=2026,
        source="unit-test",
        license="test-only",
    )
    second = write_manifest(
        tmp_path / "b.jsonl",
        entries,
        kind="images",
        dataset="unit",
        split="train",
        seed=2026,
        source="unit-test",
        license="test-only",
    )
    assert first["sha256"] == second["sha256"]
    assert [item["image_id"] for item in read_manifest(tmp_path / "a.jsonl")] == [
        "image-0",
        "image-1",
        "image-2",
    ]


def test_missing_source_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        make_entry(
            tmp_path / "missing.png",
            image_id="missing",
            split="train",
            source="unit-test",
            license="test-only",
        )
