from __future__ import annotations

import pytest
import torch
from torch import nn

from hybrid_inpaint.training.metrics.perceptual import LPIPSMetric


class _FakeLPIPS(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.last_inputs: tuple[torch.Tensor, torch.Tensor] | None = None
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        self.last_inputs = (prediction.detach(), target.detach())
        value = (prediction - target).abs().mean()
        return value.view(1, 1, 1, 1)


def test_lpips_inputs_are_mapped_to_minus_one_one() -> None:
    model = _FakeLPIPS()
    metric = LPIPSMetric(model=model)
    prediction = torch.zeros(1, 3, 8, 8)
    target = torch.ones(1, 3, 8, 8)

    value = metric(prediction, target)
    assert model.last_inputs is not None
    assert torch.allclose(model.last_inputs[0], torch.full_like(prediction, -1.0))
    assert torch.allclose(model.last_inputs[1], torch.full_like(target, 1.0))
    assert value == pytest.approx(2.0)


def test_lpips_clamps_out_of_range_inputs() -> None:
    model = _FakeLPIPS()
    metric = LPIPSMetric(model=model)
    metric(torch.full((1, 3, 8, 8), 2.0), torch.full((1, 3, 8, 8), -1.0))
    assert model.last_inputs is not None
    assert float(model.last_inputs[0].max()) == 1.0
    assert float(model.last_inputs[1].min()) == -1.0


def test_lpips_model_is_frozen() -> None:
    model = _FakeLPIPS()
    LPIPSMetric(model=model)
    assert not model.training
    assert not model.scale.requires_grad


def test_shape_mismatch_is_rejected() -> None:
    metric = LPIPSMetric(model=_FakeLPIPS())
    with pytest.raises(ValueError, match="shapes must match"):
        metric(torch.zeros(1, 3, 8, 8), torch.zeros(1, 3, 9, 9))
