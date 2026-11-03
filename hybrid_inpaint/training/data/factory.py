"""Build training/validation datasets from a resolved config.

C2 only has the procedural synthetic dataset.  CelebA-HQ and Places2 adapters
land at T0 together with the real split/mask manifests; until then they fail
loudly rather than silently falling back to synthetic data.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Sequence

from hybrid_inpaint.config import AppConfig
from hybrid_inpaint.domain.entities import Sample
from hybrid_inpaint.training.data.manifest import PROJECT_ROOT, read_summary
from hybrid_inpaint.training.data.manifest_dataset import ManifestInpaintDataset
from hybrid_inpaint.training.data.synthetic import SyntheticInpaintDataset

_VALIDATION_SEED_OFFSET = 7919


def _fingerprint_dataset(dataset: Sequence[Sample]) -> str:
    """SHA256 over sample ids, images and masks (construction datasets only)."""
    digest = hashlib.sha256()
    for index in range(len(dataset)):
        sample = dataset[index]
        digest.update(sample.sample_id.encode("utf-8"))
        digest.update(sample.image.data.tobytes())
        digest.update(sample.mask.data.tobytes())
    return digest.hexdigest()


def build_training_datasets(
    config: AppConfig,
) -> tuple[Sequence[Sample], Sequence[Sample], dict[str, Any]]:
    """Return ``(train, val, data_identity)`` for the configured dataset."""
    if config.data.dataset == "synthetic":
        train = SyntheticInpaintDataset(
            size=config.data.image_size,
            length=config.data.train_samples,
            seed=config.seed,
        )
        val = SyntheticInpaintDataset(
            size=config.data.image_size,
            length=config.data.val_samples,
            seed=config.seed + _VALIDATION_SEED_OFFSET,
        )
        identity = {
            "dataset": "synthetic",
            "image_size": config.data.image_size,
            "seed": config.seed,
            "train_samples": len(train),
            "val_samples": len(val),
            "train_fingerprint": _fingerprint_dataset(train),
            "val_fingerprint": _fingerprint_dataset(val),
        }
        return train, val, identity

    if config.data.dataset in ("places2", "celeba_hq"):
        prefix = "places365" if config.data.dataset == "places2" else "celeba_hq"
        manifest_dir = Path(config.data.manifest_dir)
        if not manifest_dir.is_absolute():
            manifest_dir = PROJECT_ROOT / manifest_dir
        train_manifest = manifest_dir / f"{prefix}_train.jsonl"
        val_manifest = manifest_dir / f"{prefix}_val.jsonl"
        train_mask_manifest = manifest_dir / "masks_train.jsonl"
        val_mask_manifest = manifest_dir / "masks_val.jsonl"

        missing = [
            path
            for path in (
                train_manifest,
                val_manifest,
                train_mask_manifest,
                val_mask_manifest,
            )
            if not path.is_file()
        ]
        if missing:
            raise FileNotFoundError(
                "missing frozen manifests: "
                + ", ".join(str(path) for path in missing)
                + "; run `hybrid-inpaint data prepare` first"
            )

        train = ManifestInpaintDataset(
            train_manifest,
            train_mask_manifest,
            dataset=config.data.dataset,
            image_size=config.data.image_size,
            seed=config.seed,
            augment=True,
            limit=config.data.manifest_train_limit,
        )
        val = ManifestInpaintDataset(
            val_manifest,
            val_mask_manifest,
            dataset=config.data.dataset,
            image_size=config.data.image_size,
            seed=config.seed,
            augment=False,
            limit=config.data.manifest_val_limit,
        )
        identity = {
            "dataset": config.data.dataset,
            "image_size": config.data.image_size,
            "seed": config.seed,
            "train_samples": len(train),
            "val_samples": len(val),
            "train_manifest": read_summary(train_manifest)["sha256"],
            "val_manifest": read_summary(val_manifest)["sha256"],
            "train_mask_manifest": read_summary(train_mask_manifest)["sha256"],
            "val_mask_manifest": read_summary(val_mask_manifest)["sha256"],
        }
        return train, val, identity

    raise ValueError(f"unsupported dataset: {config.data.dataset!r}")
