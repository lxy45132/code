"""Image quality metrics for validation and evaluation.

Inputs are HWC float arrays in ``[0, 1]`` (RGB), masks are HW boolean arrays
with ``True`` meaning "hole".  Masked metrics select the hole pixels *and* all
three channels, matching :mod:`hybrid_inpaint.training.losses.pixel`.

SSIM uses the fixed contract parameters (scikit-image, Gaussian weights,
sigma=1.5, win_size=11, no sample covariance).  The implementation and
parameters are reported in the evaluation tables and must not be swapped
silently.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from skimage.metrics import structural_similarity

PSNR_EPS = 1e-12
SSIM_KWARGS: dict[str, Any] = {
    "channel_axis": -1,
    "data_range": 1.0,
    "gaussian_weights": True,
    "sigma": 1.5,
    "win_size": 11,
    "use_sample_covariance": False,
}


def _as_image(value: np.ndarray | Any, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"{name} must have shape H×W×3")
    if array.shape[0] < 1 or array.shape[1] < 1:
        raise ValueError(f"{name} must be non-empty")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")
    return array


def _as_mask(mask: np.ndarray | Any, image: np.ndarray) -> np.ndarray:
    array = np.asarray(mask)
    if array.ndim != 2 or array.shape != image.shape[:2]:
        raise ValueError("mask must be H×W and match the image")
    return array.astype(bool, copy=False)


def _select(image: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    if mask is None:
        return image.reshape(-1)
    selected = image[mask]
    if selected.size == 0:
        raise ValueError("mask selects no pixels")
    return selected.reshape(-1)


def mse(
    prediction: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray | None = None,
) -> float:
    """Mean squared error over all values or over a masked region."""
    pred = _as_image(prediction, "prediction")
    truth = _as_image(target, "target")
    if pred.shape != truth.shape:
        raise ValueError("prediction and target shapes must match")
    selected_mask = None if mask is None else _as_mask(mask, truth)
    difference = (_select(pred, selected_mask) - _select(truth, selected_mask)) ** 2
    return float(difference.mean())


def mae(
    prediction: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray | None = None,
) -> float:
    """Mean absolute error over all values or over a masked region."""
    pred = _as_image(prediction, "prediction")
    truth = _as_image(target, "target")
    if pred.shape != truth.shape:
        raise ValueError("prediction and target shapes must match")
    selected_mask = None if mask is None else _as_mask(mask, truth)
    difference = np.abs(_select(pred, selected_mask) - _select(truth, selected_mask))
    return float(difference.mean())


def psnr(
    prediction: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray | None = None,
    *,
    data_range: float = 1.0,
) -> float:
    """PSNR in dB.

    ``MSE`` is clamped to ``1e-12`` before the logarithm, so identical images
    report 120 dB instead of infinity.  Zero-error samples are counted by the
    caller when needed.
    """
    if data_range <= 0:
        raise ValueError("data_range must be > 0")
    error = mse(prediction, target, mask)
    return float(10.0 * math.log10((data_range**2) / max(error, PSNR_EPS)))


def ssim(
    prediction: np.ndarray,
    target: np.ndarray,
    *,
    data_range: float = 1.0,
) -> float:
    """Whole-image SSIM with the contract parameters."""
    pred = _as_image(prediction, "prediction")
    truth = _as_image(target, "target")
    if pred.shape != truth.shape:
        raise ValueError("prediction and target shapes must match")
    if min(pred.shape[:2]) < SSIM_KWARGS["win_size"]:
        raise ValueError(
            f"SSIM needs both sides >= {SSIM_KWARGS['win_size']} pixels"
        )
    return float(
        structural_similarity(
            pred,
            truth,
            data_range=data_range,
            **{key: value for key, value in SSIM_KWARGS.items() if key != "data_range"},
        )
    )


def evaluate_image(
    prediction: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
) -> dict[str, float]:
    """Compute the C2 validation metric set for one image/mask pair."""
    pred = _as_image(prediction, "prediction")
    truth = _as_image(target, "target")
    hole = _as_mask(mask, truth)
    return {
        "psnr_hole": psnr(pred, truth, hole),
        "psnr_full": psnr(pred, truth),
        "mae_hole": mae(pred, truth, hole),
        "mae_full": mae(pred, truth),
        "ssim": ssim(pred, truth),
    }
