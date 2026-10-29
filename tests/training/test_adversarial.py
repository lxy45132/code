from __future__ import annotations

import pytest
import torch

from hybrid_inpaint.training.losses.adversarial import hinge_d_loss, hinge_g_loss


def test_hinge_d_loss_hand_value() -> None:
    real = torch.tensor([2.0, 0.0])
    fake = torch.tensor([-2.0, 0.0])
    # real: relu(1-2)=0, relu(1-0)=1 -> mean 0.5
    # fake: relu(1-2)=0, relu(1+0)=1 -> mean 0.5
    assert hinge_d_loss(real, fake).item() == pytest.approx(1.0)


def test_hinge_d_loss_penalises_confident_wrong_predictions() -> None:
    real = torch.tensor([-0.5])
    fake = torch.tensor([0.5])
    assert hinge_d_loss(real, fake).item() == pytest.approx(3.0)


def test_hinge_g_loss_hand_value() -> None:
    assert hinge_g_loss(torch.tensor([1.0, 3.0])).item() == pytest.approx(-2.0)


def test_hinge_g_loss_gradient_direction() -> None:
    logits = torch.tensor([0.0, 1.0, 2.0], requires_grad=True)
    hinge_g_loss(logits).backward()
    assert logits.grad is not None
    assert torch.allclose(logits.grad, torch.full((3,), -1.0 / 3.0))


def test_empty_logits_are_rejected() -> None:
    with pytest.raises(ValueError, match="empty"):
        hinge_g_loss(torch.empty(0))
