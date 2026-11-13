"""chord 回调：聚合批量结果 → report.csv → 终态 + finished 事件。

状态判定：有 cancelled → cancelled；否则有 failed → failed；否则 succeeded。
report.csv 列口径与 CLI 批量一致（02 文档 §11.2）。
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path

from hybrid_inpaint.domain.entities import BatchItemState
from hybrid_inpaint.interfaces.api.celery_app import celery_app
from hybrid_inpaint.interfaces.api.tasks.batch_chunk import batch_stream
from hybrid_inpaint.interfaces.api.tasks.common import get_worker_container

logger = logging.getLogger("hybrid_inpaint.tasks")

REPORT_COLUMNS = (
    "relative_path", "mask_path", "status", "output_path",
    "latency_ms", "mask_ratio", "error",
)


def _write_report(container, batch_id: str, items: list[BatchItemState]) -> Path:
    target = (
        container.config.system.output_dir.parent / "batches" / batch_id / "report.csv"
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(REPORT_COLUMNS)
        for item in items:
            output_path = ""
            if item.job_id:
                record = container.history.get(item.job_id)
                if record is not None:
                    output_path = record.output_path
            writer.writerow(
                [
                    item.image_name,
                    f"{item.item_index:03d}.mask",
                    item.status,
                    output_path,
                    item.latency_ms if item.latency_ms is not None else "",
                    item.mask_ratio if item.mask_ratio is not None else "",
                    item.error_code or "",
                ]
            )
    return target


@celery_app.task(name="hybrid.batch_finalize")
def batch_finalize(results: list, batch_id: str) -> dict:
    container = get_worker_container()
    state = container.batch_state
    batch = state.get(batch_id)
    if batch is None:
        return {"batch_id": batch_id, "status": "missing"}
    items = state.items(batch_id)
    cancelled = any(item.status == "cancelled" for item in items)
    failed = sum(1 for item in items if item.status == "failed")
    done = sum(1 for item in items if item.status == "ok")
    status = "cancelled" if cancelled else ("failed" if failed else "succeeded")

    report_path = _write_report(container, batch_id, items)
    state.finish(batch_id, status=status, report_path=str(report_path))
    container.events.publish(
        batch_stream(batch_id),
        "finished",
        {"status": status, "total": batch.total, "done": done, "failed": failed},
    )
    logger.info(
        "batch finished batch=%s status=%s done=%s failed=%s", batch_id, status, done, failed
    )
    return {"batch_id": batch_id, "status": status, "done": done, "failed": failed}
