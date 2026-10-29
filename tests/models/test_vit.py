from __future__ import annotations

import pytest
import torch

from hybrid_inpaint.models.vit import (
    BottleneckViT,
    MultiHeadSelfAttention,
    TransformerBlock,
)


def _vit(**overrides: object) -> BottleneckViT:
    kwargs: dict[str, object] = dict(
        dim=32,
        depth=2,
        heads=4,
        patch_size=2,
        mlp_ratio=2.0,
        dropout=0.0,
    )
    kwargs.update(overrides)
    return BottleneckViT(**kwargs)


def test_vit_preserves_feature_shape() -> None:
    features = torch.randn(2, 32, 8, 8)
    outputs = _vit()(features)
    assert outputs.shape == features.shape
    assert torch.isfinite(outputs).all()


def test_vit_is_residual() -> None:
    vit = _vit()
    features = torch.zeros(1, 32, 8, 8)
    # With a zero input the residual keeps the deconv branch, so the output is
    # generally non-zero but must stay finite and shaped correctly.
    outputs = vit(features)
    assert outputs.shape == features.shape
    assert torch.isfinite(outputs).all()


def test_positional_encoding_interpolates_to_token_grid() -> None:
    vit = _vit()
    assert vit.pos_embed.shape == (1, 16 * 16, 32)
    for spatial in ((8, 8), (12, 12), (4, 10)):
        outputs = vit(torch.randn(1, 32, *spatial))
        assert outputs.shape == (1, 32, *spatial)


def test_rejects_patch_misalignment() -> None:
    vit = _vit()
    with pytest.raises(ValueError, match="divisible by vit_patch=2"):
        vit(torch.randn(1, 32, 5, 8))


def test_rejects_wrong_feature_width() -> None:
    vit = _vit()
    with pytest.raises(ValueError, match="feature channels"):
        vit(torch.randn(1, 16, 8, 8))


def test_zero_depth_is_rejected() -> None:
    with pytest.raises(ValueError, match="vit_depth=0"):
        _vit(depth=0)


def test_heads_must_divide_dim() -> None:
    with pytest.raises(ValueError, match="divisible by heads"):
        _vit(dim=30, heads=4)


def test_block_count_matches_depth() -> None:
    assert len(_vit(depth=4).blocks) == 4


def test_attention_and_block_preserve_shape() -> None:
    tokens = torch.randn(2, 5, 32)
    assert MultiHeadSelfAttention(32, 4)(tokens).shape == tokens.shape
    assert TransformerBlock(32, 4, mlp_ratio=2.0)(tokens).shape == tokens.shape


def test_all_vit_parameters_receive_finite_gradients() -> None:
    vit = _vit()
    outputs = vit(torch.randn(1, 32, 8, 8))
    outputs.mean().backward()
    for parameter in vit.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


def test_attention_numerical_gradient_check() -> None:
    torch.manual_seed(0)
    attention = MultiHeadSelfAttention(4, 2).double()
    tokens = torch.randn(1, 3, 4, dtype=torch.double, requires_grad=True)
    assert torch.autograd.gradcheck(
        attention,
        (tokens,),
        eps=1e-6,
        atol=1e-4,
        rtol=1e-3,
    )


def test_dropout_is_active_only_in_training_mode() -> None:
    vit = _vit(dropout=0.5)
    features = torch.randn(1, 32, 8, 8)
    vit.train()
    first = vit(features)
    second = vit(features)
    assert not torch.allclose(first, second)
    vit.eval()
    assert torch.allclose(vit(features), vit(features))
