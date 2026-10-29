"""LPIPS wrapper used by stage-two validation.

The ``lpips`` package is an optional training dependency.  Importing this
module is cheap; constructing :class:`LPIPSMetric` loads the loss network.
Inputs are ``[0,1]`` BCHW tensors; LPIPS expects ``[-1,1]``.
"""

from __future__ import annotations

import torch
from torch import nn


class LPIPSMetric:
    """Whole-image LPIPS (AlexNet by default), lower is better."""

    def __init__(
        self,
        model: nn.Module | None = None,
        *,
        device: str | torch.device = "cpu",
        net: str = "alex",
    ) -> None:
        if model is None:
            try:
                from lpips import LPIPS
            except ImportError as exc:  # pragma: no cover - depends on extras
                raise ImportError(
                    "LPIPS requires the 'lpips' package (uv sync --extra train)"
                ) from exc
            model = LPIPS(net=net, verbose=False).eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)

        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.net = net

    @torch.inference_mode()
    def __call__(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> float:
        if prediction.shape != target.shape:
            raise ValueError("prediction and target shapes must match")
        if prediction.ndim != 4:
            raise ValueError("prediction and target must be BCHW tensors")

        prediction = prediction.to(self.device, dtype=torch.float32).clamp(0.0, 1.0)
        target = target.to(self.device, dtype=torch.float32).clamp(0.0, 1.0)
        value = self.model(prediction * 2.0 - 1.0, target * 2.0 - 1.0)
        return float(value.mean())
