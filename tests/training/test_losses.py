from __future__ import annotations

import pytest
import torch

from hybrid_inpaint.training.losses.pixel import (
    PixelLoss,
    masked_l1,
    tv_loss,
)


def test_masked_l1_hand_value() -> None:
    prediction = torch.zeros(1, 3, 2, 2)
    target = torch.ones(1, 3, 2, 2)
    mask = torch.ones(1, 1, 2, 2)
    assert masked_l1(prediction, target, mask).item() == pytest.approx(1.0)


def test_masked_l1_selects_only_hole() -> None:
    prediction = torch.zeros(1, 3, 2, 2)
    target = torch.ones(1, 3, 2, 2)
    mask = torch.zeros(1, 1, 2, 2)
    mask[..., 0:1, 0:1] = 1.0
    assert masked_l1(prediction, target, mask).item() == pytest.approx(1.0)


def test_masked_l1_empty_region_is_zero() -> None:
    prediction = torch.rand(1, 3, 4, 4, requires_grad=True)
    target = torch.rand(1, 3, 4, 4)
    mask = torch.zeros(1, 1, 4, 4)
    value = masked_l1(prediction, target, mask)
    assert value.item() == 0.0
    assert value.requires_grad


def test_masked_l1_accepts_hw_mask() -> None:
    prediction = torch.zeros(2, 3, 4, 4)
    target = torch.ones(2, 3, 4, 4)
    mask = torch.ones(2, 4, 4)
    assert masked_l1(prediction, target, mask).item() == pytest.approx(1.0)


def test_tv_loss_is_zero_on_constant_image() -> None:
    image = torch.full((1, 3, 8, 8), 0.4)
    assert tv_loss(image).item() == pytest.approx(0.0)


def test_tv_loss_hand_value_for_horizontal_ramp() -> None:
    image = torch.tensor([[[[0.0, 0.1, 0.2]]]])
    assert tv_loss(image).item() == pytest.approx(0.1)


def test_tv_loss_handles_degenerate_axes() -> None:
    assert tv_loss(torch.rand(1, 3, 1, 1)).item() == pytest.approx(0.0)
    assert tv_loss(torch.rand(1, 3, 1, 4)).item() > 0.0


def test_valid_loss_uses_raw_generator_output() -> None:
    target = torch.ones(1, 3, 4, 4)
    generated = torch.full((1, 3, 4, 4), 0.5)
    mask = torch.zeros(1, 1, 4, 4)
    mask[..., 0, 0] = 1.0
    observed = target * (1.0 - mask)
    composed = observed * (1.0 - mask) + generated * mask

    loss = PixelLoss(hole=1.0, valid=0.5, tv=0.0)(composed, generated, target, mask)
    # On the composed output the valid region equals the target exactly, so
    # the valid term must be taken from the raw generator output instead.
    assert loss["valid"].item() == pytest.approx(0.5)
    assert loss["hole"].item() == pytest.approx(0.5)
    assert loss["total"].item() == pytest.approx(0.5 + 0.5 * 0.5)


def test_pixel_loss_total_applies_weights() -> None:
    loss = PixelLoss(hole=1.0, valid=0.5, tv=0.05)
    composed = generated = target = torch.zeros(1, 3, 4, 4)
    mask = torch.ones(1, 1, 4, 4)
    result = loss(composed, generated, target, mask)
    assert result["total"].item() == pytest.approx(0.0)


def test_pixel_loss_is_differentiable() -> None:
    generated = torch.rand(1, 3, 8, 8, requires_grad=True)
    target = torch.rand(1, 3, 8, 8)
    mask = torch.zeros(1, 1, 8, 8)
    mask[..., 2:6, 2:6] = 1.0
    composed = target * (1.0 - mask) + generated * mask
    result = PixelLoss()(composed, generated, target, mask)
    result["total"].backward()
    assert generated.grad is not None
    assert torch.isfinite(generated.grad).all()


def test_shape_mismatch_is_rejected() -> None:
    with pytest.raises(ValueError, match="does not match"):
        masked_l1(torch.zeros(1, 3, 4, 4), torch.zeros(1, 3, 5, 5), torch.ones(1, 1, 4, 4))


def test_mask_size_mismatch_is_rejected() -> None:
    with pytest.raises(ValueError, match="spatial"):
        masked_l1(torch.zeros(1, 3, 4, 4), torch.zeros(1, 3, 4, 4), torch.ones(1, 1, 5, 5))
