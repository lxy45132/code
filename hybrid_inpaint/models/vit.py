"""Bottleneck Vision Transformer for the U-Net E3 stage.

Contract (``docs/01_模型与训练方案.md`` §3.2):

* patch embedding: ``Conv2d(C, C, kernel=stride=patch)``; patch=2 for the
  default 256 input, yielding 16x16 = 256 tokens at width 512;
* learnable 2D positional encoding initialised on a 16x16 grid and bicubically
  interpolated when the token grid changes;
* pre-LN Transformer blocks: LN -> multi-head self-attention -> residual,
  LN -> MLP (ratio 4, GELU) -> residual;
* output tokens are reshaped and transposed-convolved back to the E3 spatial
  size, then added to E3 as a residual.

Attention uses ``F.scaled_dot_product_attention`` with the standard
``1/sqrt(head_dim)`` temperature.  A final LayerNorm is applied after the last
block (standard pre-LN practice); it does not change shape or width.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

DEFAULT_BASE_GRID = (16, 16)
ALLOWED_PATCH_SIZES = (1, 2, 4)
LAYER_NORM_EPS = 1e-6


class MultiHeadSelfAttention(nn.Module):
    """Pre-projection QKV attention with standard scaled dot products."""

    def __init__(self, dim: int, heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        if dim < 1 or heads < 1:
            raise ValueError("dim and heads must be >= 1")
        if dim % heads:
            raise ValueError("dim must be divisible by heads")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)
        self.attn_dropout = float(dropout)
        self.proj_dropout = nn.Dropout(dropout)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        batch, count, dim = tokens.shape
        if dim != self.dim:
            raise ValueError(
                f"expected token width {self.dim}, got {dim}"
            )

        qkv = self.qkv(tokens)
        qkv = qkv.reshape(batch, count, 3, self.heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # 3, B, heads, N, head_dim
        query, key, value = qkv[0], qkv[1], qkv[2]

        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=self.attn_dropout if self.training else 0.0,
        )
        attended = attended.transpose(1, 2).reshape(batch, count, dim)
        return self.proj_dropout(self.proj(attended))


class TransformerBlock(nn.Module):
    """Pre-LN Transformer block with a GELU MLP."""

    def __init__(
        self,
        dim: int,
        heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if not math.isfinite(mlp_ratio) or mlp_ratio <= 0:
            raise ValueError("mlp_ratio must be finite and > 0")
        hidden = max(1, int(round(dim * mlp_ratio)))

        self.norm1 = nn.LayerNorm(dim, eps=LAYER_NORM_EPS)
        self.attention = MultiHeadSelfAttention(dim, heads, dropout)
        self.norm2 = nn.LayerNorm(dim, eps=LAYER_NORM_EPS)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        tokens = tokens + self.attention(self.norm1(tokens))
        tokens = tokens + self.mlp(self.norm2(tokens))
        return tokens


class BottleneckViT(nn.Module):
    """ViT that maps E3 features to E3 features via a token bottleneck."""

    def __init__(
        self,
        dim: int = 512,
        depth: int = 4,
        heads: int = 8,
        patch_size: int = 2,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        base_grid: tuple[int, int] = DEFAULT_BASE_GRID,
    ) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError(
                "BottleneckViT needs depth >= 1; use vit_depth=0 to disable the ViT"
            )
        if patch_size not in ALLOWED_PATCH_SIZES:
            raise ValueError(f"patch_size must be one of {ALLOWED_PATCH_SIZES}")
        if dim % heads:
            raise ValueError("dim must be divisible by heads")
        base_h, base_w = base_grid
        if base_h < 1 or base_w < 1:
            raise ValueError("base_grid dimensions must be >= 1")

        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.patch_size = patch_size
        self.mlp_ratio = mlp_ratio
        self.base_grid = (base_h, base_w)

        self.patch_embed = nn.Conv2d(
            dim,
            dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.pos_embed = nn.Parameter(
            torch.zeros(1, base_h * base_w, dim)
        )
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(dim, heads, mlp_ratio=mlp_ratio, dropout=dropout)
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(dim, eps=LAYER_NORM_EPS)
        self.deconv = nn.ConvTranspose2d(
            dim,
            dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def _positional_tokens(self, height: int, width: int) -> torch.Tensor:
        if (height, width) == self.base_grid:
            return self.pos_embed
        base_h, base_w = self.base_grid
        grid = self.pos_embed.reshape(1, base_h, base_w, self.dim)
        grid = grid.permute(0, 3, 1, 2)
        mode = "bicubic" if min(height, width) >= 2 else "bilinear"
        grid = F.interpolate(
            grid,
            size=(height, width),
            mode=mode,
            align_corners=False,
        )
        return grid.permute(0, 2, 3, 1).reshape(1, height * width, self.dim)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 4:
            raise ValueError("features must be a BCHW tensor")
        channels, height, width = features.shape[1:]
        if channels != self.dim:
            raise ValueError(
                f"expected {self.dim} feature channels, got {channels}"
            )
        if height % self.patch_size or width % self.patch_size:
            raise ValueError(
                f"E3 spatial size {(height, width)} must be divisible by "
                f"vit_patch={self.patch_size}; pad the input image to "
                f"{8 * self.patch_size} alignment"
            )

        patch = self.patch_embed(features)
        token_h, token_w = patch.shape[-2:]
        tokens = patch.flatten(2).transpose(1, 2)
        tokens = self.dropout(tokens + self._positional_tokens(token_h, token_w))

        for block in self.blocks:
            tokens = block(tokens)
        tokens = self.norm(tokens)

        hidden = tokens.transpose(1, 2).reshape(
            tokens.shape[0], self.dim, token_h, token_w
        )
        hidden = self.deconv(hidden)
        if hidden.shape[-2:] != (height, width):
            raise RuntimeError(
                f"ViT output size {tuple(hidden.shape[-2:])} does not match "
                f"input {(height, width)}"
            )
        return features + hidden

    def num_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
