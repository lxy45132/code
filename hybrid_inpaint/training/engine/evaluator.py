"""Checkpoint evaluation over frozen image/mask pair manifests.

The evaluator loads the *checkpoint's* stored model configuration (not the
current YAML model section), then runs the fixed pairs from
``*_test_pairs.jsonl`` / ``*_val_pairs.jsonl``.

Aggregation follows the evaluation contract: metrics are computed per image
and mask, averaged over the five masks of each image, then summarised over
images (mean ± population std).  Per-mask-bucket and per-category tables use
the same per-image values.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from hybrid_inpaint.config import AppConfig
from hybrid_inpaint.models.generator import build_generator
from hybrid_inpaint.models.unet import compose_output
from hybrid_inpaint.training.data.manifest_dataset import FixedPairDataset
from hybrid_inpaint.training.engine.checkpoint import load_checkpoint
from hybrid_inpaint.training.engine.validator import collate_samples
from hybrid_inpaint.training.metrics.image import evaluate_image

METRIC_KEYS = ("psnr_hole", "psnr_full", "ssim", "mae_hole", "mae_full")


@dataclass(frozen=True)
class EvaluationResult:
    rows: tuple[dict[str, Any], ...]
    summary: dict[str, Any]


def _resolve_device(name: str) -> torch.device:
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")
    if device.type == "mps":
        available = getattr(torch.backends, "mps", None)
        if available is None or not available.is_available():
            raise RuntimeError("MPS was requested but is not available")
    return device


def _to_hwc(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().to(torch.float32).cpu().permute(0, 2, 3, 1).numpy()


def _stats(values: Sequence[float]) -> dict[str, Any] | None:
    if not values:
        return None
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "std": float(array.std()),
        "count": int(array.size),
    }


def _aggregate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    metric_keys = list(METRIC_KEYS)
    if any("lpips" in row for row in rows):
        metric_keys.append("lpips")

    per_image: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    image_category: dict[str, str | None] = {}
    for row in rows:
        image_id = str(row["image_id"])
        image_category.setdefault(image_id, row.get("category"))
        for key in metric_keys:
            if key in row and row[key] is not None:
                per_image[image_id][key].append(float(row[key]))

    image_metrics = {
        image_id: {
            key: float(np.mean(values))
            for key, values in metrics.items()
            if values
        }
        for image_id, metrics in per_image.items()
    }

    summary: dict[str, Any] = {
        "images": len(image_metrics),
        "pairs": len(rows),
        "metrics": {
            key: _stats(
                [
                    metrics[key]
                    for metrics in image_metrics.values()
                    if key in metrics
                ]
            )
            for key in metric_keys
        },
    }

    buckets = sorted(
        {int(row["bucket"]) for row in rows if row.get("bucket") is not None}
    )
    by_bucket: dict[str, Any] = {}
    for bucket in buckets:
        bucket_rows = [row for row in rows if row.get("bucket") == bucket]
        bucket_range = bucket_rows[0].get("bucket_range") if bucket_rows else None
        by_bucket[str(bucket)] = {
            "bucket_range": bucket_range,
            "images": len({str(row["image_id"]) for row in bucket_rows}),
            "pairs": len(bucket_rows),
            "metrics": {
                key: _stats(
                    [float(row[key]) for row in bucket_rows if key in row]
                )
                for key in metric_keys
            },
        }
    summary["by_bucket"] = by_bucket

    categories: dict[str, list[str]] = defaultdict(list)
    for image_id, category in image_category.items():
        if category:
            categories[str(category)].append(image_id)
    if categories:
        summary["by_category"] = {
            category: {
                "images": len(image_ids),
                "metrics": {
                    key: _stats(
                        [
                            image_metrics[image_id][key]
                            for image_id in image_ids
                            if key in image_metrics[image_id]
                        ]
                    )
                    for key in metric_keys
                },
            }
            for category, image_ids in sorted(categories.items())
        }
    return summary


def evaluate_checkpoint(
    config: AppConfig,
    checkpoint: str | Path,
    pair_manifest: str | Path,
    *,
    dataset: str,
    device: str | None = None,
    limit: int | None = None,
    batch_size: int | None = None,
    lpips_metric: Any | None = None,
) -> EvaluationResult:
    """Evaluate one checkpoint on a frozen pair manifest."""
    if dataset not in ("places2", "celeba_hq"):
        raise ValueError("dataset must be places2 or celeba_hq")

    payload = load_checkpoint(
        checkpoint,
        expected_config=None,
        map_location="cpu",
        restore_rng=False,
    )
    stored_config = AppConfig.model_validate(payload["config"])
    model = build_generator(stored_config.model)
    model.load_state_dict(payload["model"], strict=True)

    resolved_device = _resolve_device(device or config.system.device)
    model = model.to(resolved_device).eval()

    dataset_obj = FixedPairDataset(
        pair_manifest,
        dataset=dataset,
        image_size=config.data.image_size,
        limit=limit,
    )

    if lpips_metric is None and config.eval.compute_lpips:
        from hybrid_inpaint.training.metrics.perceptual import LPIPSMetric

        lpips_metric = LPIPSMetric(device=resolved_device, net="alex")

    batch_size = batch_size or config.eval.batch_size
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")

    amp_enabled = (
        config.train.precision != "fp32"
        and resolved_device.type in ("cuda", "cpu")
    )
    amp_dtype = (
        torch.bfloat16 if config.train.precision == "bf16" else torch.float16
    )

    rows: list[dict[str, Any]] = []
    with torch.inference_mode():
        for start in range(0, len(dataset_obj), batch_size):
            samples = [
                dataset_obj[index]
                for index in range(
                    start, min(start + batch_size, len(dataset_obj))
                )
            ]
            images, masks = collate_samples(samples)
            images = images.to(resolved_device)
            masks = masks.to(resolved_device)
            observed = images * (1.0 - masks)

            with torch.autocast(
                device_type=resolved_device.type,
                dtype=amp_dtype,
                enabled=amp_enabled,
            ):
                generated = model(torch.cat((observed, masks), dim=1))
            generated = generated.float()
            composed = compose_output(observed, generated, masks)

            prediction = _to_hwc(composed)
            target = _to_hwc(images)
            hole_masks = masks.detach().to(torch.bool).cpu().numpy()[:, 0]

            for index, sample in enumerate(samples):
                metrics = evaluate_image(
                    prediction[index], target[index], hole_masks[index]
                )
                if lpips_metric is not None:
                    metrics["lpips"] = float(
                        lpips_metric(
                            composed[index : index + 1],
                            images[index : index + 1],
                        )
                    )
                rows.append(
                    {
                        "image_id": sample.meta.get("image_id"),
                        "category_id": sample.meta.get("category_id"),
                        "category": sample.meta.get("category"),
                        "bucket": sample.meta.get("bucket"),
                        "bucket_range": sample.meta.get("bucket_range"),
                        "mask_area": sample.meta.get("mask_area"),
                        "mask_id": sample.meta.get("mask_id"),
                        **metrics,
                    }
                )

    return EvaluationResult(rows=tuple(rows), summary=_aggregate(rows))


def write_evaluation(
    result: EvaluationResult,
    output_dir: str | Path,
    *,
    checkpoint: str | Path,
    dataset: str,
    image_size: int,
) -> tuple[Path, Path]:
    """Write per-pair CSV and summary JSON; return their paths."""
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)

    rows = list(result.rows)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    pairs_path = target / "pairs.csv"
    with pairs_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        **result.summary,
        "checkpoint": str(checkpoint),
        "dataset": dataset,
        "image_size": int(image_size),
    }
    summary_path = target / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return pairs_path, summary_path
