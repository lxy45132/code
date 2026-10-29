from __future__ import annotations

import math

import numpy as np
import pytest

from hybrid_inpaint.training.metrics.image import (
    evaluate_image,
    mae,
    mse,
    psnr,
    ssim,
)


def _constant(value: float, size: int = 16) -> np.ndarray:
    return np.full((size, size, 3), value, dtype=np.float64)


def test_psnr_hand_calculation() -> None:
    target = _constant(0.0)
    prediction = _constant(0.5)
    # MSE = 0.25 -> 10*log10(1/0.25) = 6.0206 dB
    assert psnr(prediction, target) == pytest.approx(10.0 * math.log10(4.0))


def test_psnr_masked_ignores_visible_pixels() -> None:
    target = _constant(0.0)
    prediction = _constant(0.0)
    prediction[:8, :8] = 0.5
    mask = np.zeros((16, 16), dtype=bool)
    mask[:8, :8] = True
    assert psnr(prediction, target, mask) == pytest.approx(10.0 * math.log10(4.0))
    # The full-image PSNR is better because only a quarter of pixels changed.
    assert psnr(prediction, target) > psnr(prediction, target, mask)


def test_psnr_identical_images_is_capped() -> None:
    target = _constant(0.3)
    assert psnr(target, target.copy()) == pytest.approx(120.0)


def test_mae_hand_calculation() -> None:
    target = _constant(0.0)
    prediction = _constant(0.25)
    assert mae(prediction, target) == pytest.approx(0.25)


def test_mse_and_mae_are_masked() -> None:
    target = _constant(0.0)
    prediction = _constant(0.0)
    prediction[0, 0] = 1.0
    mask = np.zeros((16, 16), dtype=bool)
    mask[0, 0] = True
    assert mae(prediction, target, mask) == pytest.approx(1.0)
    assert mse(prediction, target, mask) == pytest.approx(1.0)


def test_ssim_identical_is_one() -> None:
    image = np.random.default_rng(0).random((16, 16, 3))
    assert ssim(image, image.copy()) == pytest.approx(1.0)


def test_ssim_degraded_is_lower() -> None:
    target = np.zeros((16, 16, 3))
    prediction = target.copy()
    prediction[4:12, 4:12] = 1.0
    value = ssim(prediction, target)
    assert 0.0 <= value < 1.0


def test_evaluate_image_keys_and_values() -> None:
    target = _constant(0.0)
    prediction = _constant(0.0)
    mask = np.zeros((16, 16), dtype=bool)
    mask[4:12, 4:12] = True
    prediction[mask] = 0.5
    metrics = evaluate_image(prediction, target, mask)
    assert set(metrics) == {"psnr_hole", "psnr_full", "mae_hole", "mae_full", "ssim"}
    assert metrics["psnr_hole"] == pytest.approx(10.0 * math.log10(4.0))
    assert metrics["mae_hole"] == pytest.approx(0.5)
    assert metrics["ssim"] < 1.0


def test_empty_mask_is_rejected() -> None:
    with pytest.raises(ValueError, match="no pixels"):
        mae(_constant(1.0), _constant(0.0), np.zeros((16, 16), dtype=bool))


def test_shape_mismatch_is_rejected() -> None:
    with pytest.raises(ValueError, match="shapes must match"):
        mae(np.zeros((16, 16, 3)), np.zeros((8, 8, 3)))


def test_non_finite_values_are_rejected() -> None:
    prediction = _constant(0.0)
    prediction[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        psnr(prediction, _constant(0.0))
