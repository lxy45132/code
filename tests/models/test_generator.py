from __future__ import annotations

import pytest
import torch

from hybrid_inpaint.config import ModelConfig
from hybrid_inpaint.models.generator import build_generator
from hybrid_inpaint.models.unet import UNet


def _config(**overrides: object) -> ModelConfig:
    values: dict[str, object] = dict(
        channels=(4, 8, 16, 32),
        vit_heads=4,
        vit_depth=0,
        vit_patch=2,
        vit_mlp_ratio=2.0,
    )
    values.update(overrides)
    return ModelConfig(**values)


def test_zero_depth_is_exactly_b0() -> None:
    generator = build_generator(_config(vit_depth=0))
    reference = UNet(in_channels=4, out_channels=3, channels=(4, 8, 16, 32))

    assert set(generator.state_dict()) == set(reference.state_dict())
    assert not any(key.startswith("bottleneck.") for key in generator.state_dict())

    reference.load_state_dict(generator.state_dict(), strict=True)
    generator.eval()
    reference.eval()
    inputs = torch.randn(2, 4, 32, 32)
    with torch.no_grad():
        assert torch.allclose(generator(inputs), reference(inputs), atol=1e-6)


def test_vit_path_adds_bottleneck_and_preserves_output_shape() -> None:
    generator = build_generator(_config(vit_depth=2))
    assert any(key.startswith("bottleneck.") for key in generator.state_dict())
    outputs = generator(torch.randn(1, 4, 32, 32))
    assert outputs.shape == (1, 3, 32, 32)
    assert torch.isfinite(outputs).all()


def test_vit_increases_parameter_count() -> None:
    b0 = build_generator(_config(vit_depth=0)).num_parameters()
    b1 = build_generator(_config(vit_depth=2)).num_parameters()
    assert b1 > b0


def test_patch_alignment_error_is_reported() -> None:
    generator = build_generator(_config(vit_depth=1, vit_heads=1, vit_patch=4))
    # 16x16 input -> E3 is 2x2, which is not divisible by patch 4.
    with pytest.raises(ValueError, match="vit_patch=4"):
        generator(torch.randn(1, 4, 16, 16))


def test_vit_depth_four_builds_four_blocks() -> None:
    generator = build_generator(_config(vit_depth=4))
    bottleneck = generator.bottleneck
    assert hasattr(bottleneck, "blocks")
    assert len(bottleneck.blocks) == 4
