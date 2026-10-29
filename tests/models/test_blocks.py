from __future__ import annotations

import pytest
import torch
from torch import nn

from hybrid_inpaint.models.blocks import (
    Downsample,
    ResidualBlock,
    UpsampleFuse,
    make_norm,
    make_residual_stack,
)


def test_make_norm_choices() -> None:
    assert isinstance(make_norm("instance", 8), nn.InstanceNorm2d)
    assert isinstance(make_norm("batch", 8), nn.BatchNorm2d)
    assert isinstance(make_norm("none", 8), nn.Identity)
    with pytest.raises(ValueError, match="unknown norm"):
        make_norm("layer", 8)


def test_residual_block_preserves_shape() -> None:
    block = ResidualBlock(8, 8)
    inputs = torch.randn(2, 8, 12, 12)
    outputs = block(inputs)
    assert outputs.shape == inputs.shape


def test_residual_block_projects_channels() -> None:
    block = ResidualBlock(8, 16)
    outputs = block(torch.randn(1, 8, 10, 10))
    assert outputs.shape == (1, 16, 10, 10)


def test_residual_stack_zero_blocks_is_identity() -> None:
    stack = make_residual_stack(8, 0)
    inputs = torch.randn(1, 8, 8, 8)
    assert torch.equal(stack(inputs), inputs)


def test_downsample_halves_resolution_and_sets_width() -> None:
    down = Downsample(8, 16)
    outputs = down(torch.randn(1, 8, 16, 12))
    assert outputs.shape == (1, 16, 8, 6)


def test_upsample_fuse_uses_skip_spatial_size() -> None:
    up = UpsampleFuse(16, 8, 8)
    outputs = up(torch.randn(1, 16, 4, 4), torch.randn(1, 8, 9, 7))
    assert outputs.shape == (1, 8, 9, 7)


def test_residual_block_gradients_are_finite() -> None:
    block = ResidualBlock(6, 6)
    outputs = block(torch.randn(2, 6, 8, 8, requires_grad=True))
    outputs.mean().backward()
    for parameter in block.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
