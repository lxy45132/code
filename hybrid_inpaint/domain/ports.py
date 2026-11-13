"""Port protocols implemented by infrastructure adapters.

domain 只依赖标准库与 numpy；用例通过这些端口做推理/编解码/存储，
不直接 import torch、PIL、sqlite3。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from hybrid_inpaint.domain.entities import (
    ArtifactPaths,
    BatchItem,
    BatchItemState,
    BatchJobState,
    BatchReport,
    BatchResult,
    HistoryRecord,
    Image,
    Limits,
    Mask,
    ModelSpec,
)


class InpaintEngine(Protocol):
    @property
    def version(self) -> str: ...
    @property
    def last_warnings(self) -> tuple[str, ...]: ...
    def warmup(self) -> None: ...
    def inpaint(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray: ...
    def inpaint_many(
        self, images: Sequence[np.ndarray], masks: Sequence[np.ndarray]
    ) -> list[np.ndarray]:
        """GPU 凑批入口；实现可退化为逐张循环（03 文档 §4）。"""
        ...


class ImageCodec(Protocol):
    def decode(self, data: bytes, limits: Limits) -> Image: ...
    def decode_mask(self, data: bytes, *, height: int, width: int) -> Mask: ...
    def encode_png(self, image: Image) -> bytes: ...
    def encode_mask_png(self, mask: Mask) -> bytes: ...


class ArtifactStore(Protocol):
    def save(self, job_id: str, image: Image, mask: Mask, output: Image) -> ArtifactPaths: ...
    def read_output(self, job_id: str) -> bytes: ...
    def delete(self, job_id: str) -> None: ...


class HistoryRepository(Protocol):
    def add(self, record: HistoryRecord) -> None: ...
    def list(self, limit: int, offset: int) -> list[HistoryRecord]: ...
    def get(self, job_id: str) -> HistoryRecord | None: ...
    def delete(self, job_id: str) -> None: ...
    def export_csv(self, path: Path) -> None: ...


class ModelRepository(Protocol):
    def resolve(self, model_id: str) -> ModelSpec: ...
    def load(self, spec: ModelSpec) -> InpaintEngine: ...


class BatchFileStore(Protocol):
    def read(self, path: Path) -> bytes: ...
    def exists(self, path: Path) -> bool: ...
    def write(self, path: Path, data: bytes) -> None: ...


class ProgressReporter(Protocol):
    def start(self, total: int) -> None: ...
    def item(self, index: int, item: BatchItem, result: BatchResult) -> None: ...
    def finish(self, report: BatchReport) -> None: ...


class CancellationToken(Protocol):
    @property
    def cancelled(self) -> bool: ...


class CancelFlag(Protocol):
    """取消标志写侧；读侧复用 CancellationToken。

    API 进程置标志，worker 完成当前项即停（Redis 实现：独立 key + TTL）。
    """

    def set(self) -> None: ...


class EventBus(Protocol):
    """跨进程进度事件（Redis Stream 实现：条目 id 单调，订阅者断线可回放）。"""

    def publish(self, stream: str, event_type: str, data: Mapping[str, Any]) -> None: ...


class Cache(Protocol):
    """短 TTL 查询缓存（history/models），写操作后失效。"""

    def get(self, key: str) -> str | None: ...
    def set(self, key: str, value: str, ttl_s: int) -> None: ...
    def delete(self, key: str) -> None: ...


class BatchStateRepository(Protocol):
    """批量任务权威状态（AD-11：DB 保证不丢，Stream 保证及时）。"""

    def create(self, batch: BatchJobState, items: Sequence[BatchItemState]) -> None: ...
    def get(self, batch_id: str) -> BatchJobState | None: ...
    def items(self, batch_id: str) -> list[BatchItemState]: ...
    def record_item(
        self,
        batch_id: str,
        item_index: int,
        *,
        status: str,
        job_id: str | None = None,
        error_code: str | None = None,
        latency_ms: float | None = None,
        mask_ratio: float | None = None,
    ) -> None:
        """单 item 事务：更新项 + 重算 done/failed + queued→running。"""
        ...

    def finish(
        self, batch_id: str, *, status: str, report_path: str | None = None
    ) -> None: ...


class BatchCancellation(Protocol):
    """取消标志的工厂端口：api 进程 cancel，worker 进程 token 轮询。"""

    def cancel(self, batch_id: str) -> None: ...
    def token(self, batch_id: str) -> CancellationToken: ...


class Clock(Protocol):
    def now(self) -> datetime: ...


class IdGenerator(Protocol):
    def new_id(self) -> str: ...