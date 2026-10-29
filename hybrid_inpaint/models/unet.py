"""4-scale U-Net generator (B0 baseline).

Layout for a 256x256 input with the default widths ``(64, 128, 256, 512)``::

    stem      256x256x64
    E0        256x256x64   2 residual blocks
    E1        128x128x128  stride-2 conv + 2 residual blocks
    E2         64x64x256   stride-2 conv + 2 residual blocks
    E3         32x32x512   stride-2 conv + 2 residual blocks + bottleneck hook
    D2         64x64x256   upsample + skip E2 + 1x1 fuse + 2 residual blocks
    D1        128x128x128  upsample + skip E1 + 1x1 fuse + 2 residual blocks
    D0        256x256x64   upsample + skip E0 + 1x1 fuse + 2 residual blocks
    head      256x256x3    3x3 conv + sigmoid

The model consumes the 4-channel ``[z, M]`` tensor and predicts hole contents
only.  Blending with the observed image happens outside the model.  The
``bottleneck`` argument is the seam for the C3 ViT; it defaults to identity and
must preserve shape and width.
"""

from __future__ import annotations

import torch
from torch import nn

from hybrid_inpaint.models.blocks import (
    Downsample,
    UpsampleFuse,
    make_residual_stack,
)

REQUIRED_DOWNSCALES = 3
SIZE_MULTIPLE = 2**REQUIRED_DOWNSCALES


class UNet(nn.Module):
    """U-Net with 4 scales and 3 downsampling stages."""

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 3,
        channels: tuple[int, int, int, int] = (64, 128, 256, 512),
        *,
        norm: str = "instance",
        residual_blocks: int = 2,
        bottleneck: nn.Module | None = None,
    ) -> None:
        super().__init__()
        if len(channels) != 4:
            raise ValueError("channels must contain exactly 4 widths")
        if any(width < 1 for width in channels):
            raise ValueError("all channel widths must be >= 1")
        if residual_blocks < 0:
            raise ValueError("residual_blocks must be >= 0")

        c0, c1, c2, c3 = channels
        self.channels = channels

        self.stem = nn.Conv2d(in_channels, c0, kernel_size=3, padding=1)

        self.encoder0 = make_residual_stack(c0, residual_blocks, norm=norm)
        self.down1 = Downsample(c0, c1)
        self.encoder1 = make_residual_stack(c1, residual_blocks, norm=norm)
        self.down2 = Downsample(c1, c2)
        self.encoder2 = make_residual_stack(c2, residual_blocks, norm=norm)
        self.down3 = Downsample(c2, c3)
        self.encoder3 = make_residual_stack(c3, residual_blocks, norm=norm)

        # Seam for the C3 bottleneck ViT.  Identity keeps B0 pure convolutional.
        self.bottleneck: nn.Module = bottleneck if bottleneck is not None else nn.Identity()

        self.decoder2 = UpsampleFuse(c3, c2, c2, norm=norm, blocks=residual_blocks)
        self.decoder1 = UpsampleFuse(c2, c1, c1, norm=norm, blocks=residual_blocks)
        self.decoder0 = UpsampleFuse(c1, c0, c0, norm=norm, blocks=residual_blocks)

        self.head = nn.Conv2d(c0, out_channels, kernel_size=3, padding=1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 4:
            raise ValueError("input must be a BCHW tensor")
        height, width = inputs.shape[-2:]
        if height % SIZE_MULTIPLE or width % SIZE_MULTIPLE:
            raise ValueError(
                f"input height/width must be divisible by {SIZE_MULTIPLE}; "
                "pad with hybrid_inpaint.models.padding.pad_to_multiple"
            )

        e0 = self.encoder0(self.stem(inputs))
        e1 = self.encoder1(self.down1(e0))
        e2 = self.encoder2(self.down2(e1))
        e3 = self.encoder3(self.down3(e2))
        hidden = self.bottleneck(e3)
        hidden = self.decoder2(hidden, e2)
        hidden = self.decoder1(hidden, e1)
        hidden = self.decoder0(hidden, e0)
        return torch.sigmoid(self.head(hidden))

    def num_parameters(self) -> int:
        """Total parameter count (the number reported before training)."""
        return sum(parameter.numel() for parameter in self.parameters())


def compose_output(
    observed: torch.Tensor,
    generated: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Blend generated hole content back into the observed image.

    ``observed`` is expected to already have holes zeroed (``z``), but the
    expression works for any base image.  Pixels outside the hole are returned
    bit-for-bit from ``observed`` (since ``mask == 0`` multiplies them away).
    """
    return observed * (1.0 - mask) + generated * mask
