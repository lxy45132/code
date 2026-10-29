"""Read a frozen synthetic set written by ``save_frozen_set``.

Freezing the images and masks makes validation/test observations independent
of future changes to the procedural generator.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from hybrid_inpaint.domain.entities import Sample
from hybrid_inpaint.training.data.io import load_image, load_mask


class FrozenInpaintDataset:
    """A small, immutable image/mask set described by ``frozen.json``."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        manifest_path = self.root / "frozen.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"frozen manifest not found: {manifest_path}")

        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if payload.get("contract_version") != 1:
            raise ValueError("unsupported frozen manifest contract_version")
        samples = payload.get("samples")
        if not isinstance(samples, list) or not samples:
            raise ValueError("frozen manifest must contain a non-empty samples list")
        self.manifest: dict[str, Any] = payload
        self._samples: list[dict[str, Any]] = samples

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> Sample:
        entry = self._samples[index]
        image = load_image(self.root / entry["image"])
        mask = load_mask(
            self.root / entry["mask"],
            height=image.height,
            width=image.width,
        )
        meta = {
            **entry.get("meta", {}),
            "frozen": True,
            "image_sha256": entry.get("image_sha256"),
            "mask_sha256": entry.get("mask_sha256"),
        }
        return Sample(
            sample_id=entry["sample_id"],
            image=image,
            mask=mask,
            meta=meta,
        )
