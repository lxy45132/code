"""Atomic filesystem helpers for batch I/O."""

from __future__ import annotations

import os
from pathlib import Path


def atomic_write(path: Path, data: bytes) -> None:
    """先写 .tmp 再 os.replace，避免半文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


class FileSystemBatchFiles:
    def read(self, path: Path) -> bytes:
        return path.read_bytes()

    def exists(self, path: Path) -> bool:
        return path.is_file()

    def write(self, path: Path, data: bytes) -> None:
        atomic_write(path, data)