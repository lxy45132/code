"""Building blocks for the convolutional generator.

The block shapes follow the model/training contract:

* residual block: ``Conv3x3 -> norm -> ReLU -> Conv3x3 -> norm -> (+ x)``
* downsampling: ``Conv3x3, stride=2, padding=1``
* upsampling: bilinear ``align_corners=False`` -> ``Conv3x3`` -> concat skip
  -> ``1x1`` channel fusion

Normalisation is InstanceNorm by default (``affine=True``); BatchNorm and no
normalisation are kept for ablations.  A trailing ReLU is intentionally *not*
applied after the residual add, matching the written contract.
"""

from __future__ import annotations

from typing import Type

import torch
from torch import nn
from torch.nn import functional as F

NORM_KINDS = ("instance", "batch", "none")


def make_norm(kind: str, channels: int) -> nn.Module:
    """Return a normalisation layer for ``channels`` feature maps."""
    if channels < 1:
        raise ValueError("channels must be >= 1")
    if kind == "instance":
        return nn.InstanceNorm2d(channels, affine=True, track_running_stats=False)
    if kind == "batch":
        return nn.BatchNorm2d(channels)
    if kind == "none":
        return nn.Identity()
    raise ValueError(f"unknown norm: {kind!r}; expected one of {NORM_KINDS}")


class ResidualBlock(nn.Module):
    """Two 3x3 convolutions with a projection shortcut."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int | None = None,
        *,
        norm: str = "instance",
        activation: Type[nn.Module] = nn.ReLU,
    ) -> None:
        super().__init__()
        out_channels = in_channels if out_channels is None else out_channels
        if out_channels < 1:
            raise ValueError("out_channels must be >= 1")

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = make_norm(norm, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = make_norm(norm, out_channels)
        self.activation = activation(inplace=True)
        self.shortcut: nn.Module = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, kernel_size=1)
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(inputs)
        hidden = self.activation(self.norm1(self.conv1(inputs)))
        hidden = self.norm2(self.conv2(hidden))
        return hidden + residual


def make_residual_stack(
    channels: int,
    blocks: int,
    *,
    norm: str = "instance",
    activation: Type[nn.Module] = nn.ReLU,
) -> nn.Sequential:
    """Return ``blocks`` residual blocks at a constant width."""
    if blocks < 0:
        raise ValueError("blocks must be >= 0")
    if blocks == 0:
        return nn.Identity()
    return nn.Sequential(
        *[
            ResidualBlock(channels, channels, norm=norm, activation=activation)
            for _ in range(blocks)
        ]
    )


class Downsample(nn.Module):
    """Strided 3x3 convolution used by the encoder."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.conv(inputs)


class UpsampleFuse(nn.Module):
    """Bilinear upsample, 3x3 projection, skip fusion and residual refinement.

    The skip tensor's spatial size is used as the interpolation target, which
    keeps odd resolutions exact and avoids off-by-one errors.
    """

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        *,
        norm: str = "instance",
        blocks: int = 2,
        activation: Type[nn.Module] = nn.ReLU,
    ) -> None:
        super().__init__()
        self.project = nn.Conv2d(in_channels, skip_channels, kernel_size=3, padding=1)
        self.fuse = nn.Conv2d(skip_channels * 2, out_channels, kernel_size=1)
        self.blocks = make_residual_stack(
            out_channels,
            blocks,
            norm=norm,
            activation=activation,
        )

    def forward(self, inputs: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        hidden = F.interpolate(
            inputs,
            size=skip.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        hidden = self.project(hidden)
        hidden = torch.cat((hidden, skip), dim=1)
        hidden = self.fuse(hidden)
        return self.blocks(hidden)
