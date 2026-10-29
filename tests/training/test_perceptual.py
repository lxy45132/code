from __future__ import annotations

import pytest
import torch
from torch import nn

from hybrid_inpaint.training.losses.perceptual import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    PerceptualLoss,
)


class _FakeExtractor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.last_input: torch.Tensor | None = None

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self.last_input = inputs.detach()
        return inputs * 2.0, inputs * 3.0


def test_loss_matches_hand_calculation() -> None:
    extractor = _FakeExtractor()
    loss = PerceptualLoss(feature_extractor=extractor)
    prediction = torch.rand(1, 3, 8, 8)
    target = torch.rand(1, 3, 8, 8)

    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
    pred_n = (prediction - mean) / std
    target_n = (target - mean) / std
    expected = (2.0 * (pred_n - target_n)).abs().mean() + (
        3.0 * (pred_n - target_n)
    ).abs().mean()

    assert loss(prediction, target).item() == pytest.approx(expected.item())


def test_inputs_are_normalised_with_imagenet_statistics() -> None:
    extractor = _FakeExtractor()
    loss = PerceptualLoss(feature_extractor=extractor)
    image = torch.ones(1, 3, 8, 8)
    loss(image, image)

    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
    assert extractor.last_input is not None
    assert torch.allclose(extractor.last_input, (image - mean) / std)


def test_extractor_stays_frozen_and_in_eval_mode() -> None:
    extractor = _FakeExtractor()
    loss = PerceptualLoss(feature_extractor=extractor)
    loss.train()

    assert not loss.feature_extractor.training
    assert not extractor.scale.requires_grad


def test_shape_mismatch_is_rejected() -> None:
    loss = PerceptualLoss(feature_extractor=_FakeExtractor())
    with pytest.raises(ValueError, match="shapes must match"):
        loss(torch.zeros(1, 3, 8, 8), torch.zeros(1, 3, 9, 9))


def test_loss_is_differentiable() -> None:
    loss = PerceptualLoss(feature_extractor=_FakeExtractor())
    prediction = torch.rand(1, 3, 8, 8, requires_grad=True)
    target = torch.rand(1, 3, 8, 8)
    loss(prediction, target).backward()
    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()
