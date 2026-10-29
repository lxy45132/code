"""Deterministic validation over a fixed image/mask set.

The validator consumes the same ``Sample`` sequences as training but never
augments or shuffles.  Losses are computed in FP32 on the composed output;
image metrics convert each sample to HWC NumPy and use the shared metric
implementation.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import torch

from hybrid_inpaint.domain.entities import Sample
from hybrid_inpaint.models.unet import compose_output
from hybrid_inpaint.training.losses.pixel import PixelLoss
from hybrid_inpaint.training.metrics.image import evaluate_image


def collate_samples(samples: Sequence[Sample]) -> tuple[torch.Tensor, torch.Tensor]:
    """Stack domain samples into ``(images B3HW, masks B1HW)`` float32 tensors."""
    if not samples:
        raise ValueError("cannot collate an empty sample list")
    images = np.stack(
        [np.asarray(sample.image.data, dtype=np.float32) / 255.0 for sample in samples]
    )
    masks = np.stack(
        [np.asarray(sample.mask.data, dtype=np.float32) for sample in samples]
    )[:, None, :, :]
    return (
        torch.from_numpy(images).permute(0, 3, 1, 2).contiguous(),
        torch.from_numpy(masks).contiguous(),
    )


def _to_hwc(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().to(torch.float32).cpu().permute(0, 2, 3, 1).numpy()


@torch.inference_mode()
def validate(
    model: torch.nn.Module,
    dataset: Sequence[Sample],
    *,
    device: torch.device,
    batch_size: int,
    loss_fn: PixelLoss,
) -> dict[str, Any]:
    """Return averaged validation metrics over ``dataset``."""
    if len(dataset) < 1:
        raise ValueError("validation dataset must not be empty")
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")

    model.eval()
    metric_keys = ("psnr_hole", "psnr_full", "mae_hole", "mae_full", "ssim")
    totals = {key: 0.0 for key in metric_keys}
    loss_totals = {"hole": 0.0, "valid": 0.0, "tv": 0.0, "total": 0.0}
    count = 0

    for start in range(0, len(dataset), batch_size):
        batch = [dataset[index] for index in range(start, min(start + batch_size, len(dataset)))]
        images, masks = collate_samples(batch)
        images = images.to(device)
        masks = masks.to(device)

        observed = images * (1.0 - masks)
        generated = model(torch.cat((observed, masks), dim=1))
        composed = compose_output(observed, generated, masks)
        losses = loss_fn(composed, generated, images, masks)

        for key in loss_totals:
            loss_totals[key] += float(losses[key].item()) * len(batch)

        pred_hwc = _to_hwc(composed)
        target_hwc = _to_hwc(images)
        mask_hw = masks.detach().to(torch.bool).cpu().numpy()[:, 0]
        for index in range(len(batch)):
            metrics = evaluate_image(pred_hwc[index], target_hwc[index], mask_hw[index])
            for key in metric_keys:
                totals[key] += metrics[key]
            count += 1

    result: dict[str, Any] = {key: totals[key] / count for key in metric_keys}
    result.update({f"loss_{key}": value / count for key, value in loss_totals.items()})
    result["samples"] = count
    return result
