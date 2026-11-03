"""Checkpoint format, config fingerprints and RNG snapshots.

Checkpoints are loaded with ``weights_only=True``.  Everything stored is a
plain primitive container or tensor: NumPy's RNG state is converted to a list
of integers before saving, and Python/CUDA RNG states are already built from
tensors/tuples.  No custom classes are pickled.

A checkpoint is only accepted for resume when its schema and contract versions
match and its config fingerprint matches the current training-relevant config.
Iteration budget, validation cadence, logging cadence, device and output
directory are deliberately excluded from the fingerprint: extending a run or
moving its output does not change the optimisation problem.
"""

from __future__ import annotations

import hashlib
import json
import platform
import random
import subprocess
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from hybrid_inpaint.config import CONTRACT_VERSION, AppConfig

SCHEMA_VERSION = 2

_TRAIN_FINGERPRINT_KEYS = (
    "stage",
    "batch_size",
    "lr",
    "betas",
    "weight_decay",
    "warmup_steps",
    "min_lr",
    "grad_clip",
    "precision",
    "seed",
    "loss",
)


class CheckpointError(Exception):
    """Raised when a checkpoint is unreadable or incompatible."""


def canonical_config(config: AppConfig) -> dict[str, Any]:
    """Return the fingerprint-relevant part of a resolved config."""
    payload = config.model_dump(mode="json")
    train = payload.get("train", {})
    return {
        "contract_version": payload["contract_version"],
        "experiment_id": payload["experiment_id"],
        "seed": payload["seed"],
        "model": payload["model"],
        "data": payload["data"],
        "train": {key: train[key] for key in _TRAIN_FINGERPRINT_KEYS if key in train},
    }


def config_fingerprint(config: AppConfig) -> str:
    """Stable SHA256 over :func:`canonical_config`."""
    encoded = json.dumps(
        canonical_config(config),
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _check_stage1_to_stage2_init(
    payload: Mapping[str, Any],
    config: AppConfig,
) -> None:
    """Validate a stage-one checkpoint used to initialise stage two.

    The full config fingerprint includes ``train.stage`` and the loss weights,
    so it cannot match across the stage transition.  The parts that must stay
    identical are the model, dataset and seed.
    """
    stored = payload.get("config")
    if not isinstance(stored, Mapping):
        raise CheckpointError("checkpoint has no config snapshot")

    current = config.model_dump(mode="json")
    for key in ("contract_version", "seed", "model", "data"):
        if stored.get(key) != current.get(key):
            raise CheckpointError(
                f"stage-one checkpoint {key!r} does not match the stage-two config"
            )


def check_checkpoint_compatibility(
    payload: Mapping[str, Any],
    config: AppConfig,
    *,
    allow_stage_init: bool = False,
    data_identity: Mapping[str, Any] | None = None,
) -> str:
    """Return ``"resume"`` or ``"init"``; raise on incompatible checkpoints.

    ``"init"`` means a stage-one checkpoint initialises a stage-two run: the
    generator weights are loaded but iteration/optimiser state starts fresh.
    When ``data_identity`` is supplied (manifest hashes), it must match the
    stored identity exactly; this prevents silently resuming on a different
    frozen split.
    """
    stage = payload.get("stage")
    if stage == config.train.stage:
        if payload.get("config_fingerprint") != config_fingerprint(config):
            raise CheckpointError(
                "config fingerprint mismatch; refusing to resume "
                f"(stored={payload.get('config_fingerprint')!r}, "
                f"current={config_fingerprint(config)!r})"
            )
        mode = "resume"
    elif allow_stage_init and stage == "stage1" and config.train.stage == "stage2":
        _check_stage1_to_stage2_init(payload, config)
        mode = "init"
    else:
        raise CheckpointError(
            f"checkpoint stage {stage!r} cannot be used for {config.train.stage!r}"
        )

    if data_identity is not None:
        stored = payload.get("data_identity") or {}
        if dict(stored) != dict(data_identity):
            raise CheckpointError(
                "data identity mismatch; refreshing the frozen manifests changes "
                "the data and is not allowed in this checkpoint"
            )
    return mode


def capture_rng_state() -> dict[str, Any]:
    """Capture Python, NumPy, Torch and CUDA RNG states safely."""
    numpy_state = np.random.get_state()
    cuda_states = (
        [state.cpu() for state in torch.cuda.get_rng_state_all()]
        if torch.cuda.is_available()
        else []
    )
    return {
        "python": random.getstate(),
        "numpy": {
            "name": str(numpy_state[0]),
            "keys": np.asarray(numpy_state[1], dtype=np.uint32).tolist(),
            "pos": int(numpy_state[2]),
            "has_gauss": int(numpy_state[3]),
            "cached_gaussian": float(numpy_state[4]),
        },
        "torch": torch.get_rng_state(),
        "cuda": cuda_states,
    }


def restore_rng_state(state: Mapping[str, Any]) -> None:
    """Restore RNG states produced by :func:`capture_rng_state`."""
    random.setstate(state["python"])

    numpy_state = state["numpy"]
    np.random.set_state(
        (
            numpy_state["name"],
            np.asarray(numpy_state["keys"], dtype=np.uint32),
            int(numpy_state["pos"]),
            int(numpy_state["has_gauss"]),
            float(numpy_state["cached_gaussian"]),
        )
    )

    torch.set_rng_state(state["torch"])
    cuda_states = state.get("cuda") or []
    if cuda_states and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(cuda_states)


def environment_info() -> dict[str, Any]:
    """Versions recorded with every run and checkpoint."""
    info: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": str(np.__version__),
        # torch.__version__ is a TorchVersion instance, which the
        # weights_only unpickler rejects unless allowlisted; store a plain str.
        "torch": str(torch.__version__),
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        info["cuda"] = torch.version.cuda
        info["gpu"] = torch.cuda.get_device_name(0)
    return info


def git_info(root: str | Path | None = None) -> dict[str, Any]:
    """Best-effort git commit/dirty marker; never raises."""
    if root is None:
        root = Path(__file__).resolve().parents[3]
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return {"commit": "unknown", "dirty": None}

    if commit.returncode != 0:
        return {"commit": "unknown", "dirty": None}
    return {
        "commit": commit.stdout.strip(),
        "dirty": bool(status.stdout.strip()) if status.returncode == 0 else None,
    }


def save_checkpoint(
    path: str | Path,
    *,
    config: AppConfig,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler | None,
    iteration: int,
    best_psnr_hole: float,
    history: list[dict[str, Any]],
    patience_count: int = 0,
    data_identity: Mapping[str, Any] | None = None,
    discriminator: torch.nn.Module | None = None,
    optimizer_d: torch.optim.Optimizer | None = None,
    scheduler_d: torch.optim.lr_scheduler.LRScheduler | None = None,
    scaler_d: torch.amp.GradScaler | None = None,
    best_lpips: float | None = None,
) -> Path:
    """Write a self-describing, ``weights_only``-loadable checkpoint.

    Stage-one runs leave the discriminator fields as ``None``; stage-two runs
    store ``model_d``/``optimizer_d``/``scheduler_d``/``scaler_d`` alongside the
    generator state.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    best_metric: dict[str, float] = {"psnr_hole": float(best_psnr_hole)}
    if best_lpips is not None:
        best_metric["lpips"] = float(best_lpips)

    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "contract_version": CONTRACT_VERSION,
        "experiment_id": config.experiment_id,
        "seed": config.seed,
        "config": config.model_dump(mode="json"),
        "config_fingerprint": config_fingerprint(config),
        "iteration": int(iteration),
        "stage": config.train.stage,
        "best_metric": best_metric,
        "patience_count": int(patience_count),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "history": [dict(entry) for entry in history],
        "rng": capture_rng_state(),
        "data_identity": dict(data_identity or {}),
        "environment": environment_info(),
        "git": git_info(),
    }
    if discriminator is not None:
        payload["model_d"] = discriminator.state_dict()
    if optimizer_d is not None:
        payload["optimizer_d"] = optimizer_d.state_dict()
    if scheduler_d is not None:
        payload["scheduler_d"] = scheduler_d.state_dict()
    if scaler_d is not None:
        payload["scaler_d"] = scaler_d.state_dict()

    torch.save(payload, target)
    return target


def load_checkpoint(
    path: str | Path,
    *,
    expected_config: AppConfig | None = None,
    map_location: str | torch.device = "cpu",
    restore_rng: bool = True,
) -> dict[str, Any]:
    """Load and validate a checkpoint.

    Raises :class:`CheckpointError` for schema/contract/fingerprint mismatches.
    """
    source = Path(path)
    if not source.is_file():
        raise CheckpointError(f"checkpoint not found: {source}")

    try:
        payload = torch.load(source, map_location=map_location, weights_only=True)
    except Exception as exc:  # pickle/format/IO errors become one domain error
        raise CheckpointError(f"cannot read checkpoint {source}: {exc}") from exc

    if not isinstance(payload, dict):
        raise CheckpointError("checkpoint root must be a mapping")

    if payload.get("schema_version") != SCHEMA_VERSION:
        raise CheckpointError(
            f"unsupported checkpoint schema: {payload.get('schema_version')!r} "
            f"(expected {SCHEMA_VERSION})"
        )
    if payload.get("contract_version") != CONTRACT_VERSION:
        raise CheckpointError(
            f"contract version mismatch: {payload.get('contract_version')!r} "
            f"(expected {CONTRACT_VERSION})"
        )

    if expected_config is not None:
        expected = config_fingerprint(expected_config)
        stored = payload.get("config_fingerprint")
        if stored != expected:
            raise CheckpointError(
                "config fingerprint mismatch; refusing to resume "
                f"(stored={stored!r}, current={expected!r})"
            )

    if restore_rng:
        try:
            restore_rng_state(payload["rng"])
        except (KeyError, TypeError, ValueError) as exc:
            raise CheckpointError(f"invalid RNG state in checkpoint: {exc}") from exc

    return payload
