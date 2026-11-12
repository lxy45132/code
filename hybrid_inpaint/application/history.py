"""HistoryService: thin orchestration over repository + artifact ports."""

from __future__ import annotations

from pathlib import Path

from hybrid_inpaint.domain.entities import HistoryRecord
from hybrid_inpaint.domain.errors import NotFoundError
from hybrid_inpaint.domain.ports import ArtifactStore, HistoryRepository


class HistoryService:
    def __init__(self, history: HistoryRepository, artifacts: ArtifactStore) -> None:
        self._history = history
        self._artifacts = artifacts

    def list(self, limit: int = 50, offset: int = 0) -> list[HistoryRecord]:
        return self._history.list(limit, offset)

    def get(self, job_id: str) -> HistoryRecord:
        record = self._history.get(job_id)
        if record is None:
            raise NotFoundError(f"history record not found: {job_id}")
        return record

    def delete(self, job_id: str, *, delete_files: bool = False) -> None:
        self.get(job_id)
        self._history.delete(job_id)
        if delete_files:
            self._artifacts.delete(job_id)

    def export_csv(self, path: Path) -> None:
        self._history.export_csv(path)