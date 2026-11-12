"""Stdlib logging: console + rotating file; no image content ever logged."""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path


def setup_logging(
    log_path: Path | None = None, level: int = logging.INFO
) -> logging.Logger:
    logger = logging.getLogger("hybrid_inpaint")
    logger.setLevel(level)
    if logger.handlers:  # 幂等：重复装配不叠加 handler
        return logger
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(console)
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            log_path, maxBytes=5_000_000, backupCount=3, encoding="utf-8"
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger