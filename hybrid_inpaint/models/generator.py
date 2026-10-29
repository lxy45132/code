"""Generator factory: U-Net plus an optional bottleneck ViT.

``vit_depth == 0`` returns exactly the C2 B0 architecture (no extra module,
identical state-dict keys).  Positive depths insert :class:`BottleneckViT` at
the E3 seam without changing widths or spatial scales.
"""

from __future__ import annotations

from hybrid_inpaint.config import ModelConfig
from hybrid_inpaint.models.unet import UNet
from hybrid_inpaint.models.vit import BottleneckViT


def build_generator(config: ModelConfig) -> UNet:
    """Build the generator described by ``config``."""
    bottleneck: BottleneckViT | None = None
    if config.vit_depth > 0:
        bottleneck = BottleneckViT(
            dim=config.channels[-1],
            depth=config.vit_depth,
            heads=config.vit_heads,
            patch_size=config.vit_patch,
            mlp_ratio=config.vit_mlp_ratio,
            dropout=config.dropout,
        )

    return UNet(
        in_channels=config.in_channels,
        out_channels=config.out_channels,
        channels=config.channels,
        norm=config.norm,
        bottleneck=bottleneck,
    )
