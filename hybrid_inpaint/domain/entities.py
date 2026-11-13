"""Core domain entities shared by data, training, inference and the system.

Boundary images are uint8 RGB ``H×W×3``; masks are bool ``H×W`` with
``True`` meaning "hole".  The entities validate their own shape/dtype but do
not impose policy such as "no empty mask" — that belongs to the use case
(``domain.services.validate_mask``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

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


@dataclass(frozen=True)
class Limits:
    """Input resource budgets enforced by codecs and use cases."""

    max_file_mb: int = 20
    max_side: int = 4096
    max_pixels: int = 16_777_216


@dataclass(frozen=True)
class InpaintRequest:
    image: Image
    mask: Mask
    mode: Literal["web", "gui", "cli"] = "cli"
    note: str = ""
    batch_id: str | None = None

    def __post_init__(self) -> None:
        if self.image.data.shape[:2] != self.mask.data.shape:
            raise ValueError("image and mask must have the same height and width")
        if self.mode not in ("web", "gui", "cli"):
            raise ValueError("mode must be web, gui or cli")


@dataclass(frozen=True)
class InpaintResult:
    job_id: str
    image: Image
    latency_ms: float
    model_version: str
    mask_ratio: float
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class ArtifactPaths:
    input: Path
    mask: Path
    output: Path


@dataclass(frozen=True)
class HistoryRecord:
    job_id: str
    created_at: datetime
    mode: str
    batch_id: str | None
    input_path: str
    mask_path: str
    output_path: str
    model_version: str
    width: int
    height: int
    mask_ratio: float
    latency_ms: float
    note: str | None


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    checkpoint: Path
    sha256: str | None = None


@dataclass(frozen=True)
class BatchRequest:
    input_dir: Path
    mask_dir: Path
    output_dir: Path
    recursive: bool = False
    overwrite: bool = False
    limit: int | None = None


@dataclass(frozen=True)
class BatchItem:
    relative_path: str
    image_path: Path
    mask_path: Path | None


@dataclass(frozen=True)
class BatchResult:
    relative_path: str
    mask_path: str | None
    status: Literal[
        "ok", "missing_mask", "invalid_image", "invalid_mask", "skipped_exists", "error"
    ]
    output_path: str | None = None
    latency_ms: float | None = None
    mask_ratio: float | None = None
    error_code: str | None = None
    error: str | None = None
    job_id: str | None = None


@dataclass(frozen=True)
class BatchReport:
    batch_id: str
    results: tuple[BatchResult, ...]
    cancelled: bool = False

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def ok(self) -> int:
        return sum(1 for r in self.results if r.status == "ok")

    @property
    def failed(self) -> int:
        return sum(
            1
            for r in self.results
            if r.status in ("missing_mask", "invalid_image", "invalid_mask", "error")
        )


@dataclass(frozen=True)
class BatchJobState:
    """Web 批量任务权威状态（MySQL/SQLite，03 文档 §6）。"""

    batch_id: str
    status: Literal["queued", "running", "succeeded", "failed", "cancelled"]
    total: int
    done: int
    failed: int
    created_at: datetime
    finished_at: datetime | None = None
    report_path: str | None = None
    note: str | None = None


@dataclass(frozen=True)
class BatchItemState:
    """批量单项状态；pending 为初始态，其余为终态。"""

    batch_id: str
    item_index: int
    image_name: str
    status: Literal["pending", "ok", "failed", "cancelled", "skipped"]
    job_id: str | None = None
    error_code: str | None = None
    latency_ms: float | None = None
    mask_ratio: float | None = None
