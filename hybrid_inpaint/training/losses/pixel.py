"""Pixel, valid-region and total-variation losses.

All masked reductions normalise by the number of selected elements (hole
pixels times RGB channels).  This keeps ``L_hole``/``L_valid`` comparable to
the reported MAE and to the MSE used by PSNR.  Empty regions return exactly
zero without producing NaNs.

Interpretation note (recorded in ``docs/01_模型与训练方案.md`` §5.1):
``L_valid`` is evaluated on the *raw generator output* over visible pixels.
After composition ``x_out = (1-M)x + M x_gen`` the visible region is identical
to the target, so computing ``L_valid`` on ``x_out`` would always be zero and
could not influence training.
"""

from __future__ import annotations

from typing import Mapping

import torch
from torch import nn

LOSS_KEYS = ("hole", "valid", "tv", "total")


def _check_inputs(prediction: torch.Tensor, target: torch.Tensor) -> None:
    if prediction.shape != target.shape:
        raise ValueError(
            f"prediction shape {tuple(prediction.shape)} does not match "
            f"target shape {tuple(target.shape)}"
        )
    if prediction.ndim != 4:
        raise ValueError("prediction/target must be BCHW tensors")


def _check_mask(mask: torch.Tensor, prediction: torch.Tensor) -> torch.Tensor:
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    if mask.ndim != 4 or mask.shape[0] != prediction.shape[0]:
        raise ValueError("mask must be B1HW or BHW and match the batch size")
    if mask.shape[1] not in (1, prediction.shape[1]):
        raise ValueError("mask must have 1 channel or the same channels as prediction")
    if mask.shape[-2:] != prediction.shape[-2:]:
        raise ValueError("mask spatial size must match prediction")
    # Expand to all channels so ``mask.sum()`` counts selected scalar elements;
    # this is what makes the reduction a per-channel mean over hole pixels.
    return mask.expand_as(prediction).to(dtype=prediction.dtype)


def masked_l1(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Mean absolute error over ``mask == 1`` elements; 0 when empty."""
    _check_inputs(prediction, target)
    mask = _check_mask(mask, prediction)
    difference = (prediction - target).abs() * mask
    # mask.sum() counts selected scalar elements, so an empty region gives 0/1=0.
    return difference.sum() / mask.sum().clamp_min(1.0)


def tv_loss(image: torch.Tensor) -> torch.Tensor:
    """Mean total variation ``mean(|dx| + |dy|)`` over the image."""
    if image.ndim != 4:
        raise ValueError("image must be a BCHW tensor")
    zero = image.sum() * 0.0
    if image.shape[-2] < 2 and image.shape[-1] < 2:
        return zero

    terms = []
    if image.shape[-1] >= 2:
        terms.append((image[..., :, 1:] - image[..., :, :-1]).abs().mean())
    if image.shape[-2] >= 2:
        terms.append((image[..., 1:, :] - image[..., :-1, :]).abs().mean())
    return torch.stack(terms).sum()


class PixelLoss(nn.Module):
    """Weighted ``L_hole + L_valid + L_tv`` for stage-one training."""

    def __init__(self, hole: float = 1.0, valid: float = 0.5, tv: float = 0.05) -> None:
        super().__init__()
        for name, weight in (("hole", hole), ("valid", valid), ("tv", tv)):
            if not isinstance(weight, (int, float)) or weight < 0:
                raise ValueError(f"{name} weight must be a finite number >= 0")
        self.weights = {"hole": float(hole), "valid": float(valid), "tv": float(tv)}

    def forward(
        self,
        composed: torch.Tensor,
        generated: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Return the individual losses and their weighted total.

        ``composed`` is used for the hole term (identical to the generated
        tensor inside the hole) and for TV (the contract defines TV on the
        composed output).  ``generated`` is used for the valid-region term.
        """
        _check_inputs(composed, target)
        _check_inputs(generated, target)
        mask = _check_mask(mask, target)

        hole = masked_l1(composed, target, mask)
        valid = masked_l1(generated, target, 1.0 - mask)
        tv = tv_loss(composed)
        total = (
            self.weights["hole"] * hole
            + self.weights["valid"] * valid
            + self.weights["tv"] * tv
        )
        return {"hole": hole, "valid": valid, "tv": tv, "total": total}

    def weights_dict(self) -> Mapping[str, float]:
        return dict(self.weights)
