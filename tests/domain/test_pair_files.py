from __future__ import annotations

from pathlib import Path

from hybrid_inpaint.domain.services import pair_files


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x")


def test_pairs_by_relative_name_and_reports_missing(tmp_path: Path) -> None:
    inputs, masks = tmp_path / "in", tmp_path / "mask"
    _touch(inputs / "a.png")
    _touch(inputs / "b.jpg")
    _touch(masks / "a.png")

    pairs = pair_files(inputs, masks)

    assert [(p[0].name, p[1].name if p[1] else None) for p in pairs] == [
        ("a.png", "a.png"),
        ("b.jpg", None),
    ]


def test_recursive_and_extension_filter(tmp_path: Path) -> None:
    inputs, masks = tmp_path / "in", tmp_path / "mask"
    _touch(inputs / "sub" / "c.jpeg")
    _touch(inputs / "notes.txt")
    _touch(masks / "sub" / "c.bmp")

    assert pair_files(inputs, masks) == []
    pairs = pair_files(inputs, masks, recursive=True)
    assert len(pairs) == 1
    assert pairs[0][1] == masks / "sub" / "c.bmp"