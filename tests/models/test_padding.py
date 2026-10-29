from __future__ import annotations

import pytest
import torch

from hybrid_inpaint.models.padding import crop_to_size, pad_to_multiple


def test_aligned_tensor_is_returned_unchanged() -> None:
    inputs = torch.randn(1, 3, 32, 48)
    padded, size = pad_to_multiple(inputs, 16)
    assert padded is inputs
    assert size == (32, 48)


def test_padding_is_split_evenly() -> None:
    padded, size = pad_to_multiple(torch.randn(1, 3, 50, 70), 16)
    assert size == (50, 70)
    assert padded.shape == (1, 3, 64, 80)
    assert padded[..., 7:57, 5:75].shape == (1, 3, 50, 70)


def test_reflect_falls_back_when_side_is_too_small() -> None:
    padded, size = pad_to_multiple(torch.randn(1, 3, 8, 3), 16)
    assert size == (8, 3)
    assert padded.shape == (1, 3, 16, 16)


def test_constant_padding_uses_value() -> None:
    padded, _ = pad_to_multiple(torch.ones(1, 1, 3, 3), 4, mode="constant", value=7.0)
    assert padded.shape == (1, 1, 4, 4)
    assert float(padded[0, 0, 0, 3]) == 7.0


def test_crop_to_size() -> None:
    inputs = torch.arange(2 * 3 * 4 * 5).reshape(2, 3, 4, 5)
    cropped = crop_to_size(inputs, 3, 4)
    assert cropped.shape == (2, 3, 3, 4)
    assert torch.equal(cropped, inputs[..., :3, :4])


def test_crop_rejects_too_large_request() -> None:
    with pytest.raises(ValueError, match="cannot crop"):
        crop_to_size(torch.zeros(1, 3, 4, 4), 5, 4)


def test_invalid_mode_is_rejected() -> None:
    with pytest.raises(ValueError, match="mode"):
        pad_to_multiple(torch.zeros(1, 3, 5, 5), 16, mode="nearest")
