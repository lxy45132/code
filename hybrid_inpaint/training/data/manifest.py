"""JSONL dataset manifests with per-file hashes and a hashed summary.

Every image/mask entry records the SHA256 of the file, pixel dimensions, split
and source/license.  Manifests are sorted by ``image_id`` before writing, so
the same inputs and seed always produce the same file hash.  Paths are stored
relative to the project root and resolved by the dataset readers.

The manifest itself is not committed (it lives under ``data/``), but its
SHA256 is written to a neighbouring ``*.summary.json`` and can be copied into
``docs/PROGRESS.md`` / handoffs for the thesis record.
"""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MANIFEST_SCHEMA_VERSION = 1


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative_path(path: str | Path, root: str | Path = PROJECT_ROOT) -> str:
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(Path(root).resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def resolve_relative(relative: str, root: str | Path = PROJECT_ROOT) -> Path:
    path = Path(relative)
    return path if path.is_absolute() else Path(root) / path


def image_size(path: str | Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return int(image.width), int(image.height)


def make_entry(
    path: str | Path,
    *,
    image_id: str,
    split: str,
    source: str,
    license: str,
    root: str | Path = PROJECT_ROOT,
    width: int | None = None,
    height: int | None = None,
    **meta: Any,
) -> dict[str, Any]:
    """Build one manifest entry, hashing the file and reading its size.

    ``width``/``height`` may be supplied for homogeneous datasets to avoid an
    extra image-header read; otherwise they are read with Pillow.
    """
    source_path = Path(path)
    if not source_path.is_file():
        raise FileNotFoundError(f"manifest source not found: {source_path}")
    if width is None or height is None:
        width, height = image_size(source_path)
    entry: dict[str, Any] = {
        "image_id": image_id,
        "relative_path": relative_path(source_path, root),
        "sha256": sha256_file(source_path),
        "width": width,
        "height": height,
        "split": split,
        "source": source,
        "license": license,
    }
    entry.update(meta)
    return entry


def build_entries(
    specs: Sequence[Mapping[str, Any]],
    *,
    max_workers: int = 16,
    progress_every: int = 2000,
    label: str = "entries",
) -> list[dict[str, Any]]:
    """Hash/build entries in parallel; ``specs`` are kwargs for make_entry."""
    if not specs:
        return []

    def _build(spec: Mapping[str, Any]) -> dict[str, Any]:
        return make_entry(**dict(spec))

    entries: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for index, entry in enumerate(executor.map(_build, specs), 1):
            entries.append(entry)
            if index % progress_every == 0:
                print(f"  {label}: {index}/{len(specs)}", flush=True)
    return entries


def write_manifest(
    path: str | Path,
    entries: Iterable[Mapping[str, Any]],
    *,
    kind: str,
    dataset: str,
    split: str,
    seed: int,
    source: str,
    license: str,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Write sorted JSONL entries and a hashed summary beside them."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted((dict(entry) for entry in entries), key=lambda item: item["image_id"])

    with target.open("w", encoding="utf-8", newline="\n") as handle:
        for entry in ordered:
            handle.write(json.dumps(entry, ensure_ascii=False, sort_keys=True))
            handle.write("\n")

    summary: dict[str, Any] = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "kind": kind,
        "dataset": dataset,
        "split": split,
        "seed": int(seed),
        "count": len(ordered),
        "sha256": sha256_file(target),
        "source": source,
        "license": license,
    }
    if extra:
        summary.update(extra)

    summary_path = target.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return summary


def read_manifest(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_summary(path: str | Path) -> dict[str, Any]:
    manifest = Path(path)
    return json.loads(manifest.with_suffix(".summary.json").read_text(encoding="utf-8"))


def summary_path(manifest: str | Path) -> Path:
    return Path(manifest).with_suffix(".summary.json")
