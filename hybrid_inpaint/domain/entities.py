"""Core domain entities shared by data, training, inference and the system.

Boundary images are uint8 RGB ``H×W×3``; masks are bool ``H×W`` with
``True`` meaning "hole".  The entities validate their own shape/dtype but do
not impose policy such as "no empty mask" — that belongs to the use case
(``domain.services.validate_mask``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Image:
    """An RGB uint8 image in ``H×W×3`` layout."""

    data: np.ndarray

    def __post_init__(self) -> None:
        if not isinstance(self.data, np.ndarray):
            raise TypeError("Image.data must be a numpy array")
        if self.data.ndim != 3 or self.data.shape[2] != 3:
            raise ValueError("Image.data must have shape H×W×3")
        if self.data.dtype != np.uint8:
            raise ValueError("Image.data must be uint8")
        if self.data.shape[0] < 1 or self.data.shape[1] < 1:
            raise ValueError("Image.data must have positive height and width")

    @property
    def height(self) -> int:
        return int(self.data.shape[0])

    @property
    def width(self) -> int:
        return int(self.data.shape[1])

    def copy(self) -> "Image":
        return Image(self.data.copy())


@dataclass(frozen=True)
class Mask:
    """A binary mask in ``H×W`` layout; ``True`` means "hole"."""

    data: np.ndarray

    def __post_init__(self) -> None:
        if not isinstance(self.data, np.ndarray):
            raise TypeError("Mask.data must be a numpy array")
        if self.data.ndim != 2:
            raise ValueError("Mask.data must have shape H×W")
        if self.data.dtype != np.bool_:
            raise ValueError("Mask.data must be bool")
        if self.data.shape[0] < 1 or self.data.shape[1] < 1:
            raise ValueError("Mask.data must have positive height and width")

    @property
    def ratio(self) -> float:
        return float(self.data.mean())

    @property
    def holes(self) -> int:
        return int(self.data.sum())

    def copy(self) -> "Mask":
        return Mask(self.data.copy())

    def to_uint8(self) -> np.ndarray:
        """Return a 0/255 single-channel array for PNG export."""
        return (self.data.astype(np.uint8) * 255)


@dataclass(frozen=True)
class Sample:
    """One image/mask pair with reproducibility metadata."""

    sample_id: str
    image: Image
    mask: Mask
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.sample_id:
            raise ValueError("sample_id must not be empty")
        if self.image.height != self.mask.data.shape[0] or self.image.width != self.mask.data.shape[1]:
            raise ValueError("image and mask must have the same height and width")
