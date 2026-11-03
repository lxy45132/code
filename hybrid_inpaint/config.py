"""Strict configuration contract for training and system entry points.

The config is a pydantic v2 model tree with ``extra="forbid"``.  Relative
paths are resolved against the directory containing the YAML file, so the
same config works from any working directory.
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

CONTRACT_VERSION = 1


class ConfigError(Exception):
    """Raised when a config file cannot be read or validated."""


class StrictModel(BaseModel):
    """Base model: unknown keys are errors, assignment is re-validated."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class ModelConfig(StrictModel):
    name: str = "hybrid_unet_vit"
    in_channels: int = 4
    out_channels: int = 3
    channels: tuple[int, int, int, int] = (64, 128, 256, 512)
    vit_depth: int = 4
    vit_heads: int = 8
    vit_patch: int = 2
    vit_mlp_ratio: float = 4.0
    norm: Literal["instance", "batch", "none"] = "instance"
    dropout: float = 0.0

    @field_validator("in_channels", "out_channels", "vit_heads")
    @classmethod
    def _positive_int(cls, value: int) -> int:
        if value < 1:
            raise ValueError("must be >= 1")
        return value

    @field_validator("channels")
    @classmethod
    def _four_channels(cls, value: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
        if len(value) != 4:
            raise ValueError("channels must contain exactly 4 values")
        if any(channel < 1 for channel in value):
            raise ValueError("channels must all be >= 1")
        return value

    @field_validator("vit_depth")
    @classmethod
    def _non_negative_depth(cls, value: int) -> int:
        if value < 0:
            raise ValueError("vit_depth must be >= 0")
        return value

    @field_validator("vit_patch")
    @classmethod
    def _patch_size(cls, value: int) -> int:
        if value not in (1, 2, 4):
            raise ValueError("vit_patch must be one of 1, 2, 4")
        return value

    @field_validator("vit_mlp_ratio")
    @classmethod
    def _positive_ratio(cls, value: float) -> float:
        if not math.isfinite(value) or value <= 0:
            raise ValueError("vit_mlp_ratio must be a finite positive number")
        return value

    @field_validator("dropout")
    @classmethod
    def _valid_dropout(cls, value: float) -> float:
        if not math.isfinite(value) or not 0.0 <= value < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        return value

    @model_validator(mode="after")
    def _heads_divide_channels(self) -> "ModelConfig":
        if any(channel % self.vit_heads for channel in self.channels):
            raise ValueError("every channel must be divisible by vit_heads")
        return self


class DataConfig(StrictModel):
    dataset: Literal["celeba_hq", "places2", "synthetic"] = "celeba_hq"
    root: Path = Path("data/celeba_hq")
    split_file: Path | None = None
    mask_dir: Path = Path("data/masks")
    image_size: int = 256
    num_workers: int = 0
    train_split: str = "train"
    val_split: str = "val"
    test_split: str = "test"
    # Construction-time synthetic dataset sizes (ignored by real datasets).
    train_samples: int = 256
    val_samples: int = 16
    # Frozen manifests for real datasets (places2 / celeba_hq).
    manifest_dir: Path = Path("data/manifests")
    processed_dir: Path = Path("data/processed")
    manifest_train_limit: int | None = None
    manifest_val_limit: int | None = None

    @field_validator("root", "mask_dir", "split_file", "manifest_dir", "processed_dir")
    @classmethod
    def _non_empty_path(cls, value: Path | None) -> Path | None:
        if value is not None and not str(value).strip():
            raise ValueError("path must not be empty")
        return value

    @field_validator("image_size")
    @classmethod
    def _aligned_image_size(cls, value: int) -> int:
        if value < 16 or value % 16 != 0:
            raise ValueError("image_size must be >= 16 and a multiple of 16")
        return value

    @field_validator("num_workers")
    @classmethod
    def _non_negative_workers(cls, value: int) -> int:
        if value < 0:
            raise ValueError("num_workers must be >= 0")
        return value

    @field_validator("train_samples", "val_samples")
    @classmethod
    def _positive_samples(cls, value: int) -> int:
        if value < 1:
            raise ValueError("sample counts must be >= 1")
        return value

    @field_validator("manifest_train_limit", "manifest_val_limit")
    @classmethod
    def _positive_manifest_limits(cls, value: int | None) -> int | None:
        if value is not None and value < 1:
            raise ValueError("manifest limits must be >= 1 or null")
        return value


class LossConfig(StrictModel):
    l1_hole: float = 1.0
    l1_valid: float = 0.5
    perceptual: float = 0.1
    tv: float = 0.05
    adversarial: float = 0.1

    @field_validator("*")
    @classmethod
    def _non_negative_finite(cls, value: float) -> float:
        if not math.isfinite(value) or value < 0:
            raise ValueError("loss weights must be finite and >= 0")
        return value


class TrainConfig(StrictModel):
    stage: Literal["stage1", "stage2"] = "stage1"
    iterations: int = 100_000
    batch_size: int = 16
    lr: float = 1e-4
    betas: tuple[float, float] = (0.5, 0.999)
    weight_decay: float = 0.0
    warmup_steps: int = 1_000
    min_lr: float = 1e-6
    grad_clip: float = 1.0
    precision: Literal["fp32", "bf16", "fp16"] = "bf16"
    val_every: int = 5_000
    val_batch_size: int = 8
    patience: int = 10
    log_every: int = 10
    seed: int = 2026
    device: Literal["cuda", "cpu", "mps"] = "cuda"
    output_dir: Path = Path("outputs")
    loss: LossConfig = Field(default_factory=LossConfig)

    @field_validator(
        "iterations",
        "batch_size",
        "warmup_steps",
        "val_every",
        "val_batch_size",
        "patience",
        "log_every",
    )
    @classmethod
    def _positive_int(cls, value: int) -> int:
        if value < 1:
            raise ValueError("must be >= 1")
        return value

    @field_validator("seed")
    @classmethod
    def _non_negative_seed(cls, value: int) -> int:
        if value < 0:
            raise ValueError("seed must be >= 0")
        return value

    @field_validator("lr", "min_lr", "grad_clip")
    @classmethod
    def _positive_finite(cls, value: float) -> float:
        if not math.isfinite(value) or value <= 0:
            raise ValueError("must be a finite positive number")
        return value

    @field_validator("weight_decay")
    @classmethod
    def _non_negative_finite(cls, value: float) -> float:
        if not math.isfinite(value) or value < 0:
            raise ValueError("weight_decay must be finite and >= 0")
        return value

    @field_validator("output_dir")
    @classmethod
    def _non_empty_output_dir(cls, value: Path) -> Path:
        if not str(value).strip():
            raise ValueError("output_dir must not be empty")
        return value

    @field_validator("betas")
    @classmethod
    def _valid_betas(cls, value: tuple[float, float]) -> tuple[float, float]:
        if len(value) != 2:
            raise ValueError("betas must contain exactly 2 values")
        if any(not 0.0 < beta < 1.0 for beta in value):
            raise ValueError("betas must be in (0, 1)")
        return value

    @model_validator(mode="after")
    def _lr_order(self) -> "TrainConfig":
        if self.min_lr > self.lr:
            raise ValueError("min_lr must be <= lr")
        return self


class EvalConfig(StrictModel):
    test_split: str = "test"
    masks_per_image: int = 5
    batch_size: int = 8
    resolutions: tuple[int, ...] = (256,)
    mask_area_buckets: tuple[tuple[float, float], ...] = (
        (0.0, 0.1),
        (0.1, 0.2),
        (0.2, 0.3),
        (0.3, 0.4),
        (0.4, 0.6),
    )
    compute_lpips: bool = True
    compute_fid: bool = False

    @field_validator("masks_per_image", "batch_size")
    @classmethod
    def _positive_masks(cls, value: int) -> int:
        if value < 1:
            raise ValueError("masks_per_image and batch_size must be >= 1")
        return value

    @field_validator("resolutions")
    @classmethod
    def _valid_resolutions(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        if not value:
            raise ValueError("resolutions must not be empty")
        if any(size < 16 or size % 16 != 0 for size in value):
            raise ValueError("each resolution must be >= 16 and a multiple of 16")
        if len(set(value)) != len(value):
            raise ValueError("resolutions must not contain duplicates")
        return value

    @field_validator("mask_area_buckets")
    @classmethod
    def _valid_buckets(
        cls, value: tuple[tuple[float, float], ...]
    ) -> tuple[tuple[float, float], ...]:
        if not value:
            raise ValueError("mask_area_buckets must not be empty")
        previous_end = 0.0
        for start, end in value:
            if not (math.isfinite(start) and math.isfinite(end)):
                raise ValueError("mask bucket bounds must be finite")
            if not 0.0 <= start < end <= 1.0:
                raise ValueError("mask buckets must satisfy 0 <= start < end <= 1")
            if start < previous_end:
                raise ValueError("mask buckets must be sorted and non-overlapping")
            previous_end = end
        return value


class LimitsConfig(StrictModel):
    max_file_mb: int = 20
    max_side: int = 4096
    max_pixels: int = 16_777_216

    @field_validator("max_file_mb", "max_side", "max_pixels")
    @classmethod
    def _positive_int(cls, value: int) -> int:
        if value < 1:
            raise ValueError("must be >= 1")
        return value


class WebConfig(StrictModel):
    host: str = "127.0.0.1"
    port: int = 8000
    mount_gradio: bool = True

    @field_validator("port")
    @classmethod
    def _valid_port(cls, value: int) -> int:
        if not 1 <= value <= 65535:
            raise ValueError("port must be in [1, 65535]")
        return value


class TilingConfig(StrictModel):
    tile_size: int = 512
    tile_overlap: int = 64

    @field_validator("tile_size", "tile_overlap")
    @classmethod
    def _non_negative_int(cls, value: int) -> int:
        if value < 0:
            raise ValueError("must be >= 0")
        return value

    @model_validator(mode="after")
    def _overlap_smaller(self) -> "TilingConfig":
        if self.tile_size < 1:
            raise ValueError("tile_size must be >= 1")
        if self.tile_overlap >= self.tile_size:
            raise ValueError("tile_overlap must be smaller than tile_size")
        return self


class SystemConfig(StrictModel):
    model_id: str = "main"
    checkpoint: Path = Path("outputs/main/seed_2026/checkpoints/best_psnr.pt")
    device: Literal["cuda", "cpu", "mps"] = "cuda"
    pad_multiple: int = 16
    tiling: TilingConfig = Field(default_factory=TilingConfig)
    history_db: Path = Path("outputs/history.sqlite")
    output_dir: Path = Path("outputs/history")
    limits: LimitsConfig = Field(default_factory=LimitsConfig)
    web: WebConfig = Field(default_factory=WebConfig)
    busy_timeout_s: float = 2.0

    @field_validator("model_id")
    @classmethod
    def _non_empty_model_id(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("model_id must not be empty")
        return value

    @field_validator("checkpoint", "history_db", "output_dir")
    @classmethod
    def _non_empty_path(cls, value: Path) -> Path:
        if not str(value).strip():
            raise ValueError("path must not be empty")
        return value

    @field_validator("pad_multiple")
    @classmethod
    def _valid_pad_multiple(cls, value: int) -> int:
        if value not in (8, 16, 32):
            raise ValueError("pad_multiple must be one of 8, 16, 32")
        return value

    @field_validator("busy_timeout_s")
    @classmethod
    def _non_negative_timeout(cls, value: float) -> float:
        if not math.isfinite(value) or value < 0:
            raise ValueError("busy_timeout_s must be finite and >= 0")
        return value


class AppConfig(StrictModel):
    contract_version: int = CONTRACT_VERSION
    experiment_id: str = "base"
    seed: int = 2026
    model: ModelConfig = Field(default_factory=ModelConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    train: TrainConfig = Field(default_factory=TrainConfig)
    eval: EvalConfig = Field(default_factory=EvalConfig)
    system: SystemConfig = Field(default_factory=SystemConfig)

    @field_validator("contract_version")
    @classmethod
    def _version_one(cls, value: int) -> int:
        if value != CONTRACT_VERSION:
            raise ValueError(f"contract_version must be {CONTRACT_VERSION}")
        return value

    @field_validator("experiment_id")
    @classmethod
    def _valid_experiment_id(cls, value: str) -> str:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", value):
            raise ValueError("experiment_id may only contain letters, digits, '_' and '-'")
        return value

    @field_validator("seed")
    @classmethod
    def _non_negative_seed(cls, value: int) -> int:
        if value < 0:
            raise ValueError("seed must be >= 0")
        return value


_PATH_FIELDS: tuple[tuple[str, str], ...] = (
    ("data", "root"),
    ("data", "split_file"),
    ("data", "mask_dir"),
    ("data", "manifest_dir"),
    ("data", "processed_dir"),
    ("train", "output_dir"),
    ("system", "checkpoint"),
    ("system", "history_db"),
    ("system", "output_dir"),
)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Return a new mapping with ``override`` merged into ``base`` recursively."""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _resolve_path(base_dir: Path, value: Path | None) -> Path | None:
    if value is None:
        return None
    return value if value.is_absolute() else (base_dir / value).resolve()


def _format_validation_error(error: ValidationError) -> str:
    lines = []
    for item in error.errors():
        location = ".".join(str(part) for part in item["loc"])
        lines.append(f"{location}: {item['msg']}")
    return "; ".join(lines)


def load_config(path: str | Path, overrides: dict[str, Any] | None = None) -> AppConfig:
    """Load, validate and path-resolve a YAML config.

    Relative paths in the config are resolved against the YAML file's parent
    directory.  Unknown keys, wrong types, non-finite numbers and invalid
    ranges are rejected with :class:`ConfigError`.
    """
    config_path = Path(path)
    if not config_path.is_file():
        raise ConfigError(f"config file not found: {config_path}")

    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"YAML parse error: {exc}") from exc

    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ConfigError("config root must be a mapping")

    if overrides:
        raw = _deep_merge(raw, overrides)

    try:
        config = AppConfig.model_validate(raw)
    except ValidationError as exc:
        raise ConfigError(_format_validation_error(exc)) from exc

    base_dir = config_path.parent
    for section_name, field_name in _PATH_FIELDS:
        # Only anchor paths that were actually written in the YAML.  A default
        # path in a partial config (e.g. configs/system.yaml) has no YAML
        # directory to be relative to and would otherwise point at a
        # nonsensical location such as configs/data/celeba_hq.
        section_raw = raw.get(section_name)
        if not isinstance(section_raw, dict) or field_name not in section_raw:
            continue
        section = getattr(config, section_name)
        value = getattr(section, field_name)
        setattr(section, field_name, _resolve_path(base_dir, value))

    return config
