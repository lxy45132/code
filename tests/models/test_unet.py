from __future__ import annotations

import pytest
import torch
from torch import nn

from hybrid_inpaint.models.padding import crop_to_size, pad_to_multiple
from hybrid_inpaint.models.unet import UNet, compose_output

TINY = dict(in_channels=4, out_channels=3, channels=(4, 8, 16, 32))


def test_output_shape_and_sigmoid_range() -> None:
    model = UNet(**TINY)
    outputs = model(torch.randn(2, 4, 32, 32))
    assert outputs.shape == (2, 3, 32, 32)
    assert float(outputs.min()) >= 0.0
    assert float(outputs.max()) <= 1.0


def test_accepts_non_square_input() -> None:
    model = UNet(**TINY)
    assert model(torch.randn(1, 4, 32, 48)).shape == (1, 3, 32, 48)


def test_rejects_non_aligned_input() -> None:
    model = UNet(**TINY)
    with pytest.raises(ValueError, match="divisible by 8"):
        model(torch.randn(1, 4, 30, 32))


def test_pad_forward_crop_round_trip() -> None:
    model = UNet(**TINY).eval()
    inputs = torch.randn(1, 4, 51, 71)
    padded, (height, width) = pad_to_multiple(inputs, 16)
    assert padded.shape[-2:] == (64, 80)
    with torch.no_grad():
        outputs = model(padded)
    restored = crop_to_size(outputs, height, width)
    assert restored.shape == (1, 3, 51, 71)


def test_all_parameters_receive_finite_gradients() -> None:
    model = UNet(**TINY)
    outputs = model(torch.randn(1, 4, 32, 32))
    outputs.mean().backward()
    for parameter in model.parameters():
        if not parameter.requires_grad:
            continue
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


def test_bottleneck_hook_sees_encoder3_features() -> None:
    seen: list[tuple[int, ...]] = []

    class Recorder(nn.Identity):
        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            seen.append(tuple(inputs.shape))
            return inputs

    model = UNet(**TINY, bottleneck=Recorder())
    model(torch.randn(1, 4, 32, 32))
    assert seen == [(1, 32, 4, 4)]


def test_compose_output_keeps_observed_pixels_exact() -> None:
    observed = torch.rand(1, 3, 8, 8)
    generated = torch.rand(1, 3, 8, 8)
    mask = torch.zeros(1, 1, 8, 8)
    mask[..., 2:5, 3:6] = 1.0

    composed = compose_output(observed, generated, mask)
    outside = mask.expand_as(composed) == 0
    assert torch.equal(composed[outside], observed[outside])
    inside = mask.expand_as(composed) == 1
    assert torch.equal(composed[inside], generated[inside])


def test_parameter_count_is_positive() -> None:
    assert UNet(**TINY).num_parameters() > 0


def test_rejects_wrong_channel_count() -> None:
    with pytest.raises(ValueError, match="exactly 4"):
        UNet(channels=(8, 16, 32))
