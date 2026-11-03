from __future__ import annotations

import numpy as np
import torch
from torch import nn

from hybrid_inpaint.training.engine.tiling import inpaint_tiled


class _ConstantModel(nn.Module):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.full(
            (inputs.shape[0], 3, inputs.shape[2], inputs.shape[3]),
            0.5,
            device=inputs.device,
        )


def test_tiled_inference_keeps_outside_and_fills_hole() -> None:
    rng = np.random.default_rng(0)
    image = rng.random((40, 50, 3), dtype=np.float32)
    mask = np.zeros((40, 50), dtype=bool)
    mask[10:25, 15:35] = True

    output = inpaint_tiled(
        _ConstantModel(),
        image,
        mask,
        device=torch.device("cpu"),
        tile_size=32,
        tile_overlap=8,
    )

    assert output.shape == image.shape
    assert np.allclose(output[~mask], image[~mask], atol=1e-6)
    assert np.allclose(output[mask], 0.5, atol=1e-4)


def test_tiled_inference_rejects_bad_tile_configuration() -> None:
    image = np.zeros((32, 32, 3), dtype=np.float32)
    mask = np.ones((32, 32), dtype=bool)
    try:
        inpaint_tiled(
            _ConstantModel(),
            image,
            mask,
            device=torch.device("cpu"),
            tile_size=30,
            tile_overlap=8,
        )
    except ValueError as exc:
        assert "multiple" in str(exc)
    else:
        raise AssertionError("expected ValueError")
