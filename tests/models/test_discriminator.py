from __future__ import annotations

import pytest
import torch

from hybrid_inpaint.models.discriminator import PatchDiscriminator


def test_output_is_patch_map() -> None:
    discriminator = PatchDiscriminator()
    outputs = discriminator(torch.randn(2, 3, 64, 64))
    # 64 -> 32 -> 16 -> 8 -> 7 -> 6
    assert outputs.shape == (2, 1, 6, 6)


def test_minimum_input_size_is_enforced() -> None:
    discriminator = PatchDiscriminator()
    with pytest.raises(ValueError, match="at least 32"):
        discriminator(torch.randn(1, 3, 16, 16))


def test_wrong_channel_count_is_rejected() -> None:
    discriminator = PatchDiscriminator()
    with pytest.raises(ValueError, match="input channels"):
        discriminator(torch.randn(1, 4, 64, 64))


def test_all_convolutions_are_spectral_normalised() -> None:
    discriminator = PatchDiscriminator()
    keys = discriminator.state_dict().keys()
    original_weights = [key for key in keys if key.endswith("parametrizations.weight.original")]
    assert len(original_weights) == 5


def test_gradients_are_finite() -> None:
    discriminator = PatchDiscriminator()
    discriminator(torch.randn(1, 3, 64, 64)).mean().backward()
    for parameter in discriminator.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


def test_parameter_count_is_positive() -> None:
    assert PatchDiscriminator().num_parameters() > 0
