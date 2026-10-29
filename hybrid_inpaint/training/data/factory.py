"""Build training/validation datasets from a resolved config.

C2 only has the procedural synthetic dataset.  CelebA-HQ and Places2 adapters
land at T0 together with the real split/mask manifests; until then they fail
loudly rather than silently falling back to synthetic data.
"""

from __future__ import annotations

import hashlib
from typing import Any, Sequence

from hybrid_inpaint.config import AppConfig
from hybrid_inpaint.domain.entities import Sample
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

    raise NotImplementedError(
        f"dataset {config.data.dataset!r} is not available at construction time; "
        "CelebA-HQ/Places2 adapters and frozen splits land at T0. "
        "Use data.dataset: synthetic for C2."
    )
