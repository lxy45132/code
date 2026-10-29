"""Hinge adversarial losses.

``L_D = mean(relu(1 - D(x))) + mean(relu(1 + D(x_out)))``
``L_G_adv = -mean(D(x_out))``

No R1 regularisation and no EMA are applied by default (see the model/training
contract §5.3).
"""

from __future__ import annotations

import torch
from torch.nn import functional as F


def _check_logits(logits: torch.Tensor, name: str) -> None:
    if logits.ndim < 1:
        raise ValueError(f"{name} must have at least one dimension")
    if logits.numel() == 0:
        raise ValueError(f"{name} must not be empty")


def hinge_d_loss(real_logits: torch.Tensor, fake_logits: torch.Tensor) -> torch.Tensor:
    """Discriminator hinge loss; lower is better for D."""
    _check_logits(real_logits, "real_logits")
    _check_logits(fake_logits, "fake_logits")
    return F.relu(1.0 - real_logits).mean() + F.relu(1.0 + fake_logits).mean()


def hinge_g_loss(fake_logits: torch.Tensor) -> torch.Tensor:
    """Generator adversarial loss; gradients flow through ``fake_logits``."""
    _check_logits(fake_logits, "fake_logits")
    return -fake_logits.mean()
