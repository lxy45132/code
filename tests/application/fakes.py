"""Fake adapters: application tests run without GPU/torch/PIL/sqlite."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from hybrid_inpaint.domain.entities import ArtifactPaths, HistoryRecord, Image, Mask


class FakeEngine:
    device = "fake"

    def __init__(self) -> None:
        self.calls = 0
        self.warmups = 0
        self._warnings: tuple[str, ...] = ()

    @property
    def version(self) -> str:
        return "fake00000000-00000000"

    @property
    def last_warnings(self) -> tuple[str, ...]:
        return self._warnings

    def warmup(self) -> None:
        self.warmups += 1

    def inpaint(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        self.calls += 1
        arr = image.astype(np.float32) / 255.0
        visible = ~mask
        fill = (
            arr[visible].mean(axis=0)
            if visible.any()
            else np.full(3, 0.5, dtype=np.float32)
        )
        out = arr.copy()
        out[mask] = fill
        return out


class InMemoryHistory:
    def __init__(self) -> None:
        self.records: dict[str, HistoryRecord] = {}
        self.raise_on_add: Exception | None = None

    def add(self, record: HistoryRecord) -> None:
        if self.raise_on_add is not None:
            raise self.raise_on_add
        self.records[record.job_id] = record

    def list(self, limit: int, offset: int) -> list[HistoryRecord]:
        ordered = sorted(
            self.records.values(), key=lambda r: r.created_at, reverse=True
        )
        return ordered[offset : offset + limit]

    def get(self, job_id: str) -> HistoryRecord | None:
        return self.records.get(job_id)

    def delete(self, job_id: str) -> None:
        self.records.pop(job_id, None)

    def export_csv(self, path: Path) -> None:
        with Path(path).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["job_id", "created_at", "mode"])
            for record in self.list(1_000_000, 0):
                writer.writerow(
                    [record.job_id, record.created_at.isoformat(), record.mode]
                )


class InMemoryArtifacts:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.saved: dict[str, ArtifactPaths] = {}
        self.deleted: list[str] = []

    def save(
        self, job_id: str, image: Image, mask: Mask, output: Image
    ) -> ArtifactPaths:
        directory = self.root / job_id
        paths = ArtifactPaths(
            directory / "input.png", directory / "mask.png", directory / "output.png"
        )
        self.saved[job_id] = paths
        return paths

    def read_output(self, job_id: str) -> bytes:
        return b"fake-png"

    def delete(self, job_id: str) -> None:
        self.saved.pop(job_id, None)
        self.deleted.append(job_id)


class FixedClock:
    def __init__(self, moment: datetime | None = None) -> None:
        self.moment = moment or datetime(2026, 11, 12, 12, 0, tzinfo=timezone.utc)

    def now(self) -> datetime:
        return self.moment


class SeqIds:
    """32 位小写 hex，满足 job_id UUID 正则。"""

    def __init__(self) -> None:
        self.counter = 0

    def new_id(self) -> str:
        self.counter += 1
        return format(self.counter, "032x")


class CollectingReporter:
    def __init__(self) -> None:
        self.total: int | None = None
        self.items: list[tuple[int, str, str]] = []
        self.report = None

    def start(self, total: int) -> None:
        self.total = total

    def item(self, index: int, item, result) -> None:
        self.items.append((index, item.relative_path, result.status))

    def finish(self, report) -> None:
        self.report = report


class FlagToken:
    def __init__(self, cancel_after: int) -> None:
        self.cancel_after = cancel_after
        self.checks = 0

    @property
    def cancelled(self) -> bool:
        self.checks += 1
        return self.checks > self.cancel_after