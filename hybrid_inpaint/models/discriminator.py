"""70x70 PatchGAN discriminator.

Layer layout from ``docs/01_模型与训练方案.md`` §4:

====================  ==========================================
layer                 configuration
====================  ==========================================
1                     Conv4x4 s2 3->64, LeakyReLU(0.2), no norm
2                     Conv4x4 s2 64->128, InstanceNorm, LeakyReLU
3                     Conv4x4 s2 128->256, InstanceNorm, LeakyReLU
4                     Conv4x4 s1 256->512, InstanceNorm, LeakyReLU
5                     Conv4x4 s1 512->1
====================  ==========================================

Every convolution is spectral-normalised.  The output is a patch-level
real/fake map; losses average over patches rather than pooling globally.
The discriminator is always run in FP32 by the trainer: mixed-precision
spectral-norm power iterations are explicitly forbidden by the contract.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn.utils.parametrizations import spectral_norm

MIN_INPUT_SIZE = 32
RECEPTIVE_FIELD = 70


def _conv(
    in_channels: int,
    out_channels: int,
    *,
    stride: int,
    norm: bool,
) -> nn.Sequential:
    layers: list[nn.Module] = [
        spectral_norm(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=4,
                stride=stride,
                padding=1,
                bias=True,
            )
        )
    ]
    if norm:
        layers.append(nn.InstanceNorm2d(out_channels, affine=True))
    layers.append(nn.LeakyReLU(0.2, inplace=True))
    return nn.Sequential(*layers)


class PatchDiscriminator(nn.Module):
    """PatchGAN judging 70x70 overlapping patches."""

    def __init__(self, in_channels: int = 3) -> None:
        super().__init__()
        if in_channels < 1:
            raise ValueError("in_channels must be >= 1")

        self.in_channels = in_channels
        self.features = nn.Sequential(
            _conv(in_channels, 64, stride=2, norm=False),
            _conv(64, 128, stride=2, norm=True),
            _conv(128, 256, stride=2, norm=True),
            _conv(256, 512, stride=1, norm=True),
            spectral_norm(
                nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1, bias=True)
            ),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 4:
            raise ValueError("input must be a BCHW tensor")
        if inputs.shape[1] != self.in_channels:
            raise ValueError(
                f"expected {self.in_channels} input channels, got {inputs.shape[1]}"
            )
        height, width = inputs.shape[-2:]
        if min(height, width) < MIN_INPUT_SIZE:
            raise ValueError(
                f"discriminator input must be at least {MIN_INPUT_SIZE}x{MIN_INPUT_SIZE}"
            )
        return self.features(inputs)

    def num_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
