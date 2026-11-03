"""Overlapping-tile inference for high-resolution evaluation.

The generator is fully convolutional; for 512/1024 inputs we crop overlapping
tiles, run the model on each tile that contains a hole, and blend the generated
hole content with a Hann window.  Pixels outside the hole are copied from the
original image and never blended.  This keeps peak memory proportional to the
tile size and makes 512/1024 evaluation possible on smaller GPUs.
"""

from __future__ import annotations

import numpy as np
import torch

from hybrid_inpaint.models.padding import pad_to_multiple


def _positions(total: int, tile: int, step: int) -> list[int]:
    if total <= tile:
        return [0]
    positions = list(range(0, total - tile + 1, step))
    if positions[-1] != total - tile:
        positions.append(total - tile)
    return positions


def _window(height: int, width: int) -> np.ndarray:
    rows = np.hanning(height) if height > 1 else np.ones(1, dtype=np.float32)
    columns = np.hanning(width) if width > 1 else np.ones(1, dtype=np.float32)
    window = np.outer(rows, columns).astype(np.float32)
    return np.maximum(window, 1e-3)


@torch.inference_mode()
def inpaint_tiled(
    model: torch.nn.Module,
    image: np.ndarray,
    mask: np.ndarray,
    *,
    device: torch.device,
    tile_size: int = 512,
    tile_overlap: int = 64,
    pad_multiple: int = 16,
    amp_dtype: torch.dtype | None = None,
    amp_enabled: bool = False,
) -> np.ndarray:
    """Return an HWC float32 composed image; ``image`` is ``[0,1]``, mask bool."""
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must be HxWx3")
    if mask.shape != image.shape[:2]:
        raise ValueError("mask spatial size must match image")
    if tile_size < 1 or tile_overlap < 0 or tile_overlap >= tile_size:
        raise ValueError("require tile_size >= 1 and 0 <= tile_overlap < tile_size")
    if tile_size % pad_multiple:
        raise ValueError("tile_size must be a multiple of pad_multiple")

    height, width = image.shape[:2]
    image_tensor = (
        torch.from_numpy(np.ascontiguousarray(image, dtype=np.float32))
        .permute(2, 0, 1)
        .unsqueeze(0)
    )
    mask_tensor = (
        torch.from_numpy(np.ascontiguousarray(mask, dtype=np.float32))
        .unsqueeze(0)
        .unsqueeze(0)
    )
    padded_image, (original_height, original_width) = pad_to_multiple(
        image_tensor, pad_multiple, mode="reflect"
    )
    padded_mask, _ = pad_to_multiple(mask_tensor, pad_multiple, mode="replicate")
    padded_image = padded_image.to(device)
    padded_mask = padded_mask.to(device)
    padded_height = int(padded_image.shape[-2])
    padded_width = int(padded_image.shape[-1])

    image_np = padded_image[0].permute(1, 2, 0).cpu().numpy()
    mask_np = padded_mask[0, 0].cpu().numpy() > 0.5

    tile_height = min(tile_size, padded_height)
    tile_width = min(tile_size, padded_width)
    step = max(1, tile_size - tile_overlap)
    rows = _positions(padded_height, tile_height, step)
    columns = _positions(padded_width, tile_width, step)

    accumulated = np.zeros((padded_height, padded_width, 3), dtype=np.float32)
    weights = np.zeros((padded_height, padded_width), dtype=np.float32)

    for top in rows:
        for left in columns:
            bottom = min(top + tile_height, padded_height)
            right = min(left + tile_width, padded_width)
            tile_mask = mask_np[top:bottom, left:right]
            if not tile_mask.any():
                continue

            tile_image = padded_image[:, :, top:bottom, left:right]
            tile_mask_tensor = padded_mask[:, :, top:bottom, left:right]
            observed = tile_image * (1.0 - tile_mask_tensor)
            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype or torch.bfloat16,
                enabled=amp_enabled,
            ):
                generated = model(torch.cat((observed, tile_mask_tensor), dim=1))
            generated = generated.float()
            composed = observed * (1.0 - tile_mask_tensor) + generated * tile_mask_tensor
            composed_np = composed[0].permute(1, 2, 0).cpu().numpy()

            window = _window(bottom - top, right - left)
            selected = tile_mask
            accumulated[top:bottom, left:right] += (
                composed_np * window[:, :, None] * selected[:, :, None]
            )
            weights[top:bottom, left:right] += window * selected

    hole = mask_np
    missing = hole & (weights <= 0)
    if missing.any():
        raise RuntimeError("tiled inference left some hole pixels uncovered")
    result = image_np.copy()
    hole_weight = np.maximum(weights[hole], 1e-8)
    result[hole] = accumulated[hole] / hole_weight[:, None]
    result = np.clip(result, 0.0, 1.0)
    # pad_to_multiple splits padding evenly; remove the same top/left offset.
    pad_top = (padded_height - original_height) // 2
    pad_left = (padded_width - original_width) // 2
    result = result[
        pad_top : pad_top + original_height,
        pad_left : pad_left + original_width,
    ]
    return result.astype(np.float32, copy=False)
