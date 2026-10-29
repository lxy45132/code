"""Single-stage (stage-one) training loop.

Scope of C2: pure U-Net B0, pixel losses, validation, last/best checkpoints and
exact resume.  The bottleneck ViT (C3), perceptual loss (C4) and PatchGAN
(C4) are explicit extension points; configuring them before those steps land
raises instead of being silently ignored.

Batch composition is deterministic per iteration: iteration ``k`` draws its
indices from ``SeedSequence([seed, 0xB2, k])``.  A resumed run therefore sees
exactly the same batches as an uninterrupted one without having to serialise a
DataLoader iterator.  This is sufficient for the C1 synthetic dataset; real
image datasets will introduce worker-based loaders at T0.
"""

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch

from hybrid_inpaint.config import AppConfig
from hybrid_inpaint.domain.entities import Sample
from hybrid_inpaint.models.generator import build_generator
from hybrid_inpaint.models.unet import compose_output
from hybrid_inpaint.training.engine.checkpoint import (
    config_fingerprint,
    environment_info,
    git_info,
    load_checkpoint,
    save_checkpoint,
)
from hybrid_inpaint.training.engine.validator import collate_samples, validate
from hybrid_inpaint.training.losses.pixel import PixelLoss

PerceptualFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class TrainResult:
    """Summary of a completed or early-stopped run."""

    run_dir: Path
    start_iteration: int
    final_iteration: int
    best_psnr_hole: float
    stopped_early: bool
    history: tuple[dict[str, Any], ...]


def make_lr_lambda(
    *,
    warmup_steps: int,
    iterations: int,
    min_lr_ratio: float,
) -> Callable[[int], float]:
    """Linear warmup followed by cosine decay to ``min_lr_ratio``."""
    if iterations < 1:
        raise ValueError("iterations must be >= 1")
    if warmup_steps < 0:
        raise ValueError("warmup_steps must be >= 0")
    if not 0.0 <= min_lr_ratio <= 1.0:
        raise ValueError("min_lr_ratio must be in [0, 1]")

    def schedule(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        remaining = iterations - warmup_steps
        if remaining <= 0:
            return 1.0
        progress = min(max((step - warmup_steps) / remaining, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return schedule


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_device(name: str) -> torch.device:
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")
    if device.type == "mps":
        available = getattr(torch.backends, "mps", None)
        if available is None or not available.is_available():
            raise RuntimeError("MPS was requested but is not available")
    return device


class Trainer:
    """Stage-one trainer with deterministic batches and resumable checkpoints."""

    def __init__(
        self,
        config: AppConfig,
        train_dataset: Sequence[Sample],
        val_dataset: Sequence[Sample],
        *,
        data_identity: Mapping[str, Any] | None = None,
        run_dir: str | Path | None = None,
        device: str | None = None,
        resume_from: str | Path | None = None,
        perceptual: PerceptualFn | None = None,
    ) -> None:
        if config.train.stage != "stage1":
            raise NotImplementedError(
                "stage-two adversarial training lands in C4; set train.stage: stage1"
            )
        if config.model.in_channels != 4 or config.model.out_channels != 3:
            raise ValueError("the C2 contract requires in_channels=4 and out_channels=3")
        if len(train_dataset) < 1 or len(val_dataset) < 1:
            raise ValueError("train and validation datasets must not be empty")
        if config.train.loss.perceptual > 0 and perceptual is None:
            raise ValueError(
                "perceptual weight is non-zero but no perceptual loss is wired yet "
                "(C4); set train.loss.perceptual: 0.0"
            )

        _seed_everything(config.train.seed)

        self.config = config
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.device = _resolve_device(device or config.train.device)
        self.data_identity = dict(data_identity or {})

        if run_dir is None:
            run_dir = (
                Path(config.train.output_dir)
                / config.experiment_id
                / f"seed_{config.seed}"
            )
        self.run_dir = Path(run_dir)
        self.checkpoint_dir = self.run_dir / "checkpoints"

        self.model = build_generator(config.model).to(self.device)
        self.parameter_count = self.model.num_parameters()

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.train.lr,
            betas=config.train.betas,
            weight_decay=config.train.weight_decay,
        )
        self._lr_lambda = make_lr_lambda(
            warmup_steps=config.train.warmup_steps,
            iterations=config.train.iterations,
            min_lr_ratio=config.train.min_lr / config.train.lr,
        )
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            self._lr_lambda,
        )

        self.precision = config.train.precision
        self._amp_dtype = (
            torch.bfloat16 if self.precision == "bf16" else torch.float16
        )
        self._autocast_enabled = (
            self.precision != "fp32" and self.device.type in ("cuda", "cpu")
        )
        self.scaler: torch.amp.GradScaler | None = None
        if self.precision == "fp16":
            self.scaler = torch.amp.GradScaler(
                "cuda", enabled=self.device.type == "cuda"
            )

        self.loss_fn = PixelLoss(
            hole=config.train.loss.l1_hole,
            valid=config.train.loss.l1_valid,
            tv=config.train.loss.tv,
        )
        self.perceptual = perceptual
        self.perceptual_weight = config.train.loss.perceptual

        self.start_iteration = 0
        self.best_psnr_hole = float("-inf")
        self.history: list[dict[str, Any]] = []
        self._patience_count = 0

        if resume_from is not None:
            self._resume(Path(resume_from))

        if self.start_iteration >= config.train.iterations:
            raise ValueError(
                "checkpoint iteration is already at/after train.iterations; "
                "increase train.iterations to continue"
            )

    # ------------------------------------------------------------------ setup
    def _resume(self, path: Path) -> None:
        payload = load_checkpoint(
            path,
            expected_config=self.config,
            map_location="cpu",
            restore_rng=True,
        )
        self.model.load_state_dict(payload["model"], strict=True)
        self.optimizer.load_state_dict(payload["optimizer"])
        self.scheduler.load_state_dict(payload["scheduler"])
        if self.scaler is not None and payload.get("scaler"):
            self.scaler.load_state_dict(payload["scaler"])
        self.start_iteration = int(payload["iteration"])

        # Re-derive the LR for the first resumed step.  When the iteration
        # budget was extended, the checkpoint's stored LR follows the old
        # cosine schedule; recomputing keeps the resumed run identical to an
        # uninterrupted run under the new budget.
        self.scheduler.last_epoch = self.start_iteration
        for group, base_lr in zip(self.optimizer.param_groups, self.scheduler.base_lrs):
            group["lr"] = base_lr * self._lr_lambda(self.start_iteration)
        self.best_psnr_hole = float(payload["best_metric"]["psnr_hole"])
        self.history = [dict(entry) for entry in payload.get("history", [])]
        self._patience_count = int(payload.get("patience_count", 0))

    # --------------------------------------------------------------- training
    def _batch_indices(self, iteration: int) -> list[int]:
        length = len(self.train_dataset)
        batch_size = self.config.train.batch_size
        rng = np.random.default_rng(
            np.random.SeedSequence([self.config.train.seed, 0xB2, iteration])
        )
        indices = rng.choice(
            length,
            size=batch_size,
            replace=batch_size > length,
        )
        return [int(index) for index in indices]

    def _train_step(self, iteration: int) -> dict[str, float]:
        samples = [self.train_dataset[index] for index in self._batch_indices(iteration)]
        images, masks = collate_samples(samples)
        images = images.to(self.device)
        masks = masks.to(self.device)

        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type=self.device.type,
            dtype=self._amp_dtype,
            enabled=self._autocast_enabled,
        ):
            observed = images * (1.0 - masks)
            generated = self.model(torch.cat((observed, masks), dim=1))
            composed = compose_output(observed, generated, masks)
            losses = self.loss_fn(composed, generated, images, masks)
            total = losses["total"]
            if self.perceptual is not None and self.perceptual_weight > 0:
                total = total + self.perceptual_weight * self.perceptual(composed, images)

        if not bool(torch.isfinite(total).all()):
            raise FloatingPointError(
                f"non-finite training loss at iteration {iteration}: {float(total)}"
            )

        if self.scaler is not None and self.scaler.is_enabled():
            self.scaler.scale(total).backward()
            self.scaler.unscale_(self.optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.train.grad_clip
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            total.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.train.grad_clip
            )
            self.optimizer.step()

        if not bool(torch.isfinite(grad_norm).all()):
            raise FloatingPointError(
                f"non-finite gradient norm at iteration {iteration}: {float(grad_norm)}"
            )

        self.scheduler.step()

        result = {
            key: float(value.detach())
            for key, value in losses.items()
        }
        result["total"] = float(total.detach())
        result["grad_norm"] = float(grad_norm)
        result["lr"] = float(self.optimizer.param_groups[0]["lr"])
        return result

    # -------------------------------------------------------------- persistence
    def _append_jsonl(self, name: str, entry: Mapping[str, Any]) -> None:
        with (self.run_dir / name).open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def _write_json(self, name: str, payload: Mapping[str, Any]) -> None:
        (self.run_dir / name).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _save(self, name: str, iteration: int) -> Path:
        return save_checkpoint(
            self.checkpoint_dir / name,
            config=self.config,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            iteration=iteration,
            best_psnr_hole=self.best_psnr_hole,
            history=self.history,
            patience_count=self._patience_count,
            data_identity=self.data_identity,
        )

    # -------------------------------------------------------------------- run
    def fit(self, until: int | None = None) -> TrainResult:
        """Train according to ``config.train`` and return the final summary.

        ``until`` stops early without changing the LR schedule or config
        fingerprint, which is how resume-equivalence is exercised.  The normal
        entry point (CLI) leaves it unset.
        """
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "train.jsonl").touch()
        (self.run_dir / "val.jsonl").touch()

        self._write_json("config.resolved.json", self.config.model_dump(mode="json"))
        self._write_json(
            "environment.json",
            {
                "environment": environment_info(),
                "git": git_info(),
                "device": str(self.device),
                "parameters": self.parameter_count,
                "config_fingerprint": config_fingerprint(self.config),
                "data_identity": self.data_identity,
                "start_iteration": self.start_iteration,
            },
        )

        total = self.config.train.iterations
        if until is not None:
            if until < self.start_iteration:
                raise ValueError(
                    f"until={until} is before the resume iteration {self.start_iteration}"
                )
            total = min(total, until)
        val_every = self.config.train.val_every
        log_every = self.config.train.log_every
        patience = self.config.train.patience

        print(
            f"[train] device={self.device} params={self.parameter_count:,} "
            f"iters={self.start_iteration}->{total} "
            f"batch={self.config.train.batch_size} precision={self.precision}",
            flush=True,
        )

        stopped_early = False
        no_improvement = self._patience_count
        final_iteration = self.start_iteration

        for iteration in range(self.start_iteration, total):
            started = time.perf_counter()
            metrics = self._train_step(iteration)
            entry = {"iteration": iteration, **metrics}
            self._append_jsonl("train.jsonl", entry)

            if (iteration + 1) % log_every == 0 or iteration == total - 1:
                elapsed = time.perf_counter() - started
                print(
                    f"[train] iter {iteration + 1}/{total} "
                    f"loss={metrics['total']:.6f} hole={metrics['hole']:.6f} "
                    f"valid={metrics['valid']:.6f} tv={metrics['tv']:.6f} "
                    f"lr={metrics['lr']:.2e} grad={metrics['grad_norm']:.3f} "
                    f"({elapsed * 1000:.0f} ms)",
                    flush=True,
                )

            final_iteration = iteration + 1

            if final_iteration % val_every == 0 or final_iteration == total:
                validation = validate(
                    self.model,
                    self.val_dataset,
                    device=self.device,
                    batch_size=self.config.train.val_batch_size,
                    loss_fn=self.loss_fn,
                )
                validation["iteration"] = final_iteration
                self.history.append(validation)
                self._append_jsonl("val.jsonl", validation)

                print(
                    f"[val] iter {final_iteration}/{total} "
                    f"psnr_hole={validation['psnr_hole']:.3f} "
                    f"psnr_full={validation['psnr_full']:.3f} "
                    f"ssim={validation['ssim']:.4f} "
                    f"mae_hole={validation['mae_hole']:.4f} "
                    f"loss={validation['loss_total']:.6f}",
                    flush=True,
                )

                improved = validation["psnr_hole"] > self.best_psnr_hole + 1e-9
                if improved:
                    self.best_psnr_hole = validation["psnr_hole"]
                    no_improvement = 0
                else:
                    no_improvement += 1
                self._patience_count = no_improvement
                self._save("last.pt", final_iteration)
                if improved:
                    self._save("best.pt", final_iteration)

                if no_improvement >= patience:
                    stopped_early = True
                    print(
                        f"[train] early stop after {no_improvement} validations "
                        f"without improvement",
                        flush=True,
                    )
                    break

        if final_iteration == self.start_iteration:
            # Defensive: a zero-length run never wrote a checkpoint.
            self._save("last.pt", final_iteration)

        return TrainResult(
            run_dir=self.run_dir,
            start_iteration=self.start_iteration,
            final_iteration=final_iteration,
            best_psnr_hole=self.best_psnr_hole,
            stopped_early=stopped_early,
            history=tuple(self.history),
        )
