"""TorchInpaintEngine: checkpoint -> inference; SerializedEngine: single-GPU lock.

引擎只做张量推理，不合成、不存盘、不碰 SQLite。小图直接前向；
大图复用 training.engine.tiling.inpaint_tiled（返回已合成图，用例只消费
洞内区域，洞外恒等由 domain.services.composite 的后置校验保证）。
"""

from __future__ import annotations

import hashlib
import threading
from pathlib import Path

import numpy as np
import torch

from hybrid_inpaint.config import AppConfig
from hybrid_inpaint.domain.errors import (
    BusyError,
    ModelUnavailableError,
    ValidationError,
)
from hybrid_inpaint.domain.ports import InpaintEngine
from hybrid_inpaint.models.generator import build_generator
from hybrid_inpaint.models.padding import pad_to_multiple
from hybrid_inpaint.training.engine.checkpoint import CheckpointError, load_checkpoint
from hybrid_inpaint.training.engine.tiling import inpaint_tiled


def _resolve_device(name: str) -> torch.device:
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ModelUnavailableError("CUDA requested but not available")
    return device


class TorchInpaintEngine:
    def __init__(
        self,
        checkpoint: str | Path,
        *,
        device: str = "cuda",
        pad_multiple: int = 16,
        tile_size: int = 512,
        tile_overlap: int = 64,
    ) -> None:
        source = Path(checkpoint)
        try:
            payload = load_checkpoint(
                source, expected_config=None, map_location="cpu", restore_rng=False
            )
            stored = AppConfig.model_validate(payload["config"])
            model = build_generator(stored.model)
            model.load_state_dict(payload["model"], strict=True)
        except (CheckpointError, KeyError, ValueError, RuntimeError) as exc:
            raise ModelUnavailableError(
                f"cannot load checkpoint {source}", detail=str(exc)
            ) from exc
        self._device = _resolve_device(device)
        self._model = model.to(self._device).eval()
        self._pad = pad_multiple
        self._tile_size = tile_size
        self._tile_overlap = tile_overlap
        self._amp_enabled = stored.train.precision != "fp32" and self._device.type in (
            "cuda",
            "cpu",
        )
        self._amp_dtype = (
            torch.bfloat16 if stored.train.precision == "bf16" else torch.float16
        )
        digest = hashlib.sha256(source.read_bytes()).hexdigest()[:12]
        self._version = f"{digest}-{str(payload['config_fingerprint'])[:8]}"
        self._warnings: tuple[str, ...] = ()

    @property
    def version(self) -> str:
        return self._version

    @property
    def device(self) -> str:
        return str(self._device)

    @property
    def last_warnings(self) -> tuple[str, ...]:
        return self._warnings

    def warmup(self) -> None:
        dummy = torch.zeros(1, 4, 256, 256, device=self._device)
        with torch.inference_mode():
            self._model(dummy)

    def inpaint(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
            raise ValidationError("engine expects uint8 HxWx3 image")
        if mask.shape != image.shape[:2] or mask.dtype != np.bool_:
            raise ValidationError("engine expects bool mask matching image size")
        self._warnings = ()
        x = image.astype(np.float32) / 255.0
        if max(x.shape[:2]) > self._tile_size:
            self._warnings = (
                f"tiled inference tile_size={self._tile_size} overlap={self._tile_overlap}",
            )
            return inpaint_tiled(
                self._model,
                x,
                mask,
                device=self._device,
                tile_size=self._tile_size,
                tile_overlap=self._tile_overlap,
                pad_multiple=self._pad,
                amp_dtype=self._amp_dtype,
                amp_enabled=self._amp_enabled,
            )
        return self._forward_stack([image], [mask])[0]

    def inpaint_many(
        self, images: list[np.ndarray], masks: list[np.ndarray]
    ) -> list[np.ndarray]:
        """GPU 凑批（dynamic batching，03 文档 §4）。

        同尺寸项 stack 成 batch 维一次前向；异尺寸/超大图退化为逐张；
        显存不足自动降批（二分），单张仍 OOM 则向上抛。
        """
        if len(images) != len(masks):
            raise ValidationError("images and masks count differ")
        results: list[np.ndarray | None] = [None] * len(images)
        groups: dict[tuple[int, int], list[int]] = {}
        for index, image in enumerate(images):
            if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
                raise ValidationError("engine expects uint8 HxWx3 image")
            if masks[index].shape != image.shape[:2] or masks[index].dtype != np.bool_:
                raise ValidationError("engine expects bool mask matching image size")
            groups.setdefault(image.shape[:2], []).append(index)
        self._warnings = ()
        for shape, indexes in groups.items():
            if len(indexes) == 1 or max(shape) > self._tile_size:
                for index in indexes:
                    results[index] = self.inpaint(images[index], masks[index])
                continue
            self._inpaint_group(indexes, images, masks, results)
        return [result for result in results if result is not None]

    def _inpaint_group(
        self,
        indexes: list[int],
        images: list[np.ndarray],
        masks: list[np.ndarray],
        results: list[np.ndarray | None],
    ) -> None:
        try:
            outputs = self._forward_stack(
                [images[i] for i in indexes], [masks[i] for i in indexes]
            )
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if len(indexes) == 1:
                raise
            middle = len(indexes) // 2
            self._inpaint_group(indexes[:middle], images, masks, results)
            self._inpaint_group(indexes[middle:], images, masks, results)
            return
        for index, output in zip(indexes, outputs):
            results[index] = output

    def _forward_stack(
        self, images: list[np.ndarray], masks: list[np.ndarray]
    ) -> list[np.ndarray]:
        """同尺寸 batch 维前向：pad → 模型 → 合成 → 裁回。"""
        x = np.stack([image.astype(np.float32) / 255.0 for image in images])
        m = np.stack([mask.astype(np.float32) for mask in masks])
        image_tensor = (
            torch.from_numpy(np.ascontiguousarray(x)).permute(0, 3, 1, 2)
        )
        mask_tensor = torch.from_numpy(np.ascontiguousarray(m)).unsqueeze(1)
        padded, (height, width) = pad_to_multiple(
            image_tensor, self._pad, mode="reflect"
        )
        padded_mask, _ = pad_to_multiple(mask_tensor, self._pad, mode="replicate")
        padded = padded.to(self._device)
        padded_mask = padded_mask.to(self._device)
        observed = padded * (1.0 - padded_mask)
        with torch.inference_mode(), torch.autocast(
            device_type=self._device.type,
            dtype=self._amp_dtype,
            enabled=self._amp_enabled,
        ):
            generated = self._model(torch.cat((observed, padded_mask), dim=1))
        generated = generated.float()
        composed = observed * (1.0 - padded_mask) + generated * padded_mask
        out = composed.permute(0, 2, 3, 1).cpu().numpy()
        pad_top = (int(padded.shape[-2]) - height) // 2
        pad_left = (int(padded.shape[-1]) - width) // 2
        out = out[:, pad_top : pad_top + height, pad_left : pad_left + width, :]
        return [
            np.clip(item, 0.0, 1.0).astype(np.float32, copy=False) for item in out
        ]


class SerializedEngine:
    """单 GPU 串行锁；超时抛 BusyError（默认等待 busy_timeout_s）。"""

    def __init__(self, engine: InpaintEngine, *, timeout_s: float = 2.0) -> None:
        self._engine = engine
        self._timeout = timeout_s
        self._lock = threading.Lock()

    @property
    def version(self) -> str:
        return self._engine.version

    @property
    def device(self) -> str:
        return getattr(self._engine, "device", "unknown")

    @property
    def last_warnings(self) -> tuple[str, ...]:
        return self._engine.last_warnings

    def warmup(self) -> None:
        if not self._lock.acquire(timeout=self._timeout):
            raise BusyError("inference engine is busy")
        try:
            self._engine.warmup()
        finally:
            self._lock.release()

    def inpaint(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        if not self._lock.acquire(timeout=self._timeout):
            raise BusyError("inference engine is busy")
        try:
            return self._engine.inpaint(image, mask)
        finally:
            self._lock.release()

    def inpaint_many(
        self, images: list[np.ndarray], masks: list[np.ndarray]
    ) -> list[np.ndarray]:
        """整批一次锁；timeout_s=-1 时 FIFO 无限等待（03 文档 §4）。"""
        if not self._lock.acquire(timeout=self._timeout):
            raise BusyError("inference engine is busy")
        try:
            return self._engine.inpaint_many(images, masks)
        finally:
            self._lock.release()