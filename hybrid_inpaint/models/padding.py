"""Padding helpers for non-aligned inference sizes.

The generator needs both sides divisible by 8; the 16-multiple contract also
leaves room for the C3 ViT patch embedding.  Non-aligned user images are
padded, processed, then cropped back to their original size.  Padded pixels
must never contribute to metrics.
"""

from __future__ import annotations

import torch
from torch.nn import functional as F

PAD_MODES = ("reflect", "replicate", "constant")


def pad_to_multiple(
    tensor: torch.Tensor,
    multiple: int = 16,
    *,
    mode: str = "reflect",
    value: float = 0.0,
) -> tuple[torch.Tensor, tuple[int, int]]:
    """Pad the last two dimensions to a multiple of ``multiple``.

    Padding is split as evenly as possible (extra pixel on the right/bottom).
    ``reflect`` falls back to ``replicate`` when a side is too small to reflect.
    Returns ``(padded, (original_height, original_width))``.
    """
    if tensor.ndim < 2:
        raise ValueError("tensor must have at least 2 dimensions")
    if multiple < 1:
        raise ValueError("multiple must be >= 1")
    if mode not in PAD_MODES:
        raise ValueError(f"mode must be one of {PAD_MODES}")

    height, width = int(tensor.shape[-2]), int(tensor.shape[-1])
    pad_h = (-height) % multiple
    pad_w = (-width) % multiple
    if pad_h == 0 and pad_w == 0:
        return tensor, (height, width)

    if mode == "reflect" and (pad_h >= height or pad_w >= width):
        mode = "replicate"
    if mode == "replicate" and (height < 1 or width < 1):
        mode = "constant"

    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left
    padding = (left, right, top, bottom)
    if mode == "constant":
        padded = F.pad(tensor, padding, mode=mode, value=value)
    else:
        padded = F.pad(tensor, padding, mode=mode)
    return padded, (height, width)


def crop_to_size(tensor: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Crop the last two dimensions to ``height x width``."""
    if height < 1 or width < 1:
        raise ValueError("height and width must be >= 1")
    if tensor.shape[-2] < height or tensor.shape[-1] < width:
        raise ValueError(
            f"cannot crop {tuple(tensor.shape[-2:])} to {(height, width)}"
        )
    return tensor[..., :height, :width]
