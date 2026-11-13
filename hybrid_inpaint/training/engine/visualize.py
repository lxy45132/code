"""Validation comparison grids: original | mask | inpainted, one row per sample.

Pure inference + PNG dump; never touches the training RNG or gradients.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image as PILImage

from hybrid_inpaint.domain.entities import Sample
from hybrid_inpaint.models.unet import compose_output
from hybrid_inpaint.training.engine.validator import collate_samples


@torch.inference_mode()
def save_val_samples(
    model: torch.nn.Module,
    dataset: Sequence[Sample],
    *,
    device: torch.device,
    count: int,
    out_path: Path,
) -> Path | None:
    """Save a ``(count rows x 3 cols)`` uint8 grid; ``None`` when disabled/empty."""
    if count <= 0 or len(dataset) < 1:
        return None
    # 数据集可能只支持整数索引（如 SyntheticInpaintDataset），不用切片
    batch = [dataset[index] for index in range(min(count, len(dataset)))]
    images, masks = collate_samples(batch)
    images = images.to(device)
    masks = masks.to(device)
    model.eval()
    observed = images * (1.0 - masks)
    generated = model(torch.cat((observed, masks), dim=1))
    composed = compose_output(observed, generated, masks)

    target = np.clip(np.rint(images.cpu().numpy() * 255.0), 0, 255).astype(np.uint8)
    pred = np.clip(
        np.rint(composed.float().cpu().numpy() * 255.0), 0, 255
    ).astype(np.uint8)
    mask_rgb = np.repeat(
        np.clip(np.rint(masks.cpu().numpy() * 255.0), 0, 255).astype(np.uint8),
        3,
        axis=1,
    )
    rows = [
        np.concatenate(
            [
                target[index].transpose(1, 2, 0),
                mask_rgb[index].transpose(1, 2, 0),
                pred[index].transpose(1, 2, 0),
            ],
            axis=1,
        )
        for index in range(len(batch))
    ]
    grid = np.concatenate(rows, axis=0)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    PILImage.fromarray(grid).save(out_path, format="PNG")
    return out_path