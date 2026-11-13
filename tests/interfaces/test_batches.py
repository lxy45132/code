"""C7d：批量任务状态机 + SSE + worker 模式测试。

Celery 用 task_always_eager 在进程内执行（无需 broker）；
事件/缓存/取消用 fakeredis（同步与异步共享 FakeServer，SSE 可回放）。
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image as PILImage

from hybrid_inpaint.bootstrap import build_container
from hybrid_inpaint.domain.entities import BatchItemState, BatchJobState
from hybrid_inpaint.infrastructure.celery_cancel import RedisBatchCancellation
from hybrid_inpaint.infrastructure.redis_events import RedisCache, RedisEventBus
from hybrid_inpaint.interfaces.api.app import create_app
from hybrid_inpaint.interfaces.api.celery_app import celery_app
from hybrid_inpaint.interfaces.api.tasks.batch_chunk import batch_chunk
from hybrid_inpaint.interfaces.api.tasks.batch_finalize import batch_finalize
from hybrid_inpaint.interfaces.api.tasks.common import set_worker_container
from tests.application.fakes import FakeEngine

fakeredis = pytest.importorskip("fakeredis")


def _config(tmp_path: Path, extra: str = "") -> Path:
    path = tmp_path / "system.yaml"
    lines = [
        "contract_version: 1",
        "experiment_id: batches_test",
        "system:",
        "  model_id: tiny",
        f"  checkpoint: {(tmp_path / 'unused.pt').as_posix()}",
        "  device: cpu",
        f"  history_db: {(tmp_path / 'history.sqlite').as_posix()}",
        f"  output_dir: {(tmp_path / 'history').as_posix()}",
        "  limits:",
        "    max_side: 16",
    ]
    if extra:
        lines.append(extra)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _png(size: int = 8, color: tuple = (120, 130, 140)) -> bytes:
    buf = io.BytesIO()
    PILImage.new("RGB", (size, size), color).save(buf, format="PNG")
    return buf.getvalue()


def _mask_png(size: int = 8) -> bytes:
    array = np.zeros((size, size), dtype=np.uint8)
    array[2 : size - 2, 2 : size - 2] = 255
    buf = io.BytesIO()
    PILImage.fromarray(array).save(buf, format="PNG")
    return buf.getvalue()


def _batch_files(count: int = 3) -> list[tuple]:
    files = []
    for index in range(count):
        size = 8 if index % 2 == 0 else 12
        files.append(("images", (f"{index}.png", _png(size), "image/png")))
        files.append(("masks", (f"{index}.png", _mask_png(size), "image/png")))
    return files


@pytest.fixture()
def make_env(tmp_path: Path, monkeypatch):
    clients: list[TestClient] = []

    def _make(extra: str = "", engine=None):
        server = fakeredis.FakeServer()
        sync_client = fakeredis.FakeStrictRedis(server=server, decode_responses=True)
        container = build_container(
            _config(tmp_path, extra),
            engine=engine or FakeEngine(),
            events=RedisEventBus(sync_client),
            cache=RedisCache(sync_client),
            cancellation=RedisBatchCancellation(sync_client),
        )
        set_worker_container(container)
        monkeypatch.setattr(celery_app.conf, "task_always_eager", True)
        monkeypatch.setattr(celery_app.conf, "task_eager_propagates", True)
        monkeypatch.setattr(celery_app.conf, "task_store_eager_result", True)
        monkeypatch.setattr(celery_app.conf, "result_backend", "cache+memory://")
        client = TestClient(create_app(container))
        client.__enter__()
        client.app.state.aredis = fakeredis.aioredis.FakeRedis(
            server=server, decode_responses=True
        )
        clients.append(client)
        return client, container

    yield _make
    for client in clients:
        client.__exit__(None, None, None)
    set_worker_container(None)


def _read_sse(response) -> list[dict]:
    events: list[dict] = []
    current: dict = {}
    for line in response.iter_lines():
        if not line:
            if current:
                events.append(current)
                current = {}
            continue
        if line.startswith(":"):
            continue
        key, _, value = line.partition(": ")
        current[key] = value
    if current:
        events.append(current)
    return events


def test_batch_full_flow(make_env) -> None:
    client, container = make_env()
    response = client.post("/api/v1/batches", files=_batch_files(3), data={"note": "批量冒烟"})
    assert response.status_code == 202
    created = response.json()
    batch_id = created["batch_id"]
    assert created["total"] == 3
    assert created["events_url"] == f"/api/v1/batches/{batch_id}/events"

    detail = client.get(f"/api/v1/batches/{batch_id}").json()
    assert detail["status"] == "succeeded"  # eager 同步完成
    assert detail["done"] == 3 and detail["failed"] == 0
    assert detail["finished_at"] is not None
    assert len(detail["items"]) == 3
    for item in detail["items"]:
        assert item["status"] == "ok"
        assert item["job_id"] and item["output_url"]
        output = client.get(item["output_url"])
        assert output.status_code == 200
        assert output.content.startswith(b"\x89PNG")

    # 历史记录带 batch_id
    page = client.get("/api/v1/history", params={"limit": 10}).json()
    assert len(page["items"]) == 3

    report = client.get(f"/api/v1/batches/{batch_id}/report")
    assert report.status_code == 200
    lines = report.text.splitlines()
    assert lines[0] == "relative_path,mask_path,status,output_path,latency_ms,mask_ratio,error"
    assert len(lines) == 4


def test_batch_sse_events_and_replay(make_env) -> None:
    client, _ = make_env()
    batch_id = client.post("/api/v1/batches", files=_batch_files(2)).json()["batch_id"]

    with client.stream("GET", f"/api/v1/batches/{batch_id}/events") as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        assert response.headers["x-accel-buffering"] == "no"
        events = _read_sse(response)

    types = [event["event"] for event in events]
    assert types.count("item_started") == 2
    assert types.count("item_done") == 2
    assert types[-1] == "finished"
    assert json.loads(events[-1]["data"])["status"] == "succeeded"
    assert all("id" in event for event in events)

    # Last-Event-ID 断线回放：从第一个事件之后重连
    cursor = events[0]["id"]
    with client.stream(
        "GET",
        f"/api/v1/batches/{batch_id}/events",
        headers={"Last-Event-ID": cursor},
    ) as response:
        replayed = _read_sse(response)
    assert len(replayed) == len(events) - 1
    assert replayed[0]["id"] == events[1]["id"]


def test_batch_invalid_item_isolated(make_env) -> None:
    client, _ = make_env()
    files = _batch_files(2)
    files[0] = ("images", ("bad.png", b"not a png", "image/png"))
    batch_id = client.post("/api/v1/batches", files=files).json()["batch_id"]

    detail = client.get(f"/api/v1/batches/{batch_id}").json()
    assert detail["status"] == "failed"
    assert detail["done"] == 1 and detail["failed"] == 1
    statuses = {item["item_index"]: item for item in detail["items"]}
    assert statuses[0]["status"] == "failed"
    assert statuses[0]["error_code"] == "validation_error"
    assert statuses[1]["status"] == "ok"


def test_batch_validation_errors(make_env) -> None:
    client, _ = make_env("  batch:\n    max_items: 2")
    mismatched = client.post(
        "/api/v1/batches",
        files=[("images", ("a.png", _png(), "image/png"))],
    )
    assert mismatched.status_code == 422
    too_many = client.post("/api/v1/batches", files=_batch_files(3))
    assert too_many.status_code == 413
    assert too_many.json()["code"] == "resource_limit"


def test_batch_not_found_and_report_not_ready(make_env) -> None:
    client, container = make_env()
    missing = format(9, "032x")
    assert client.get(f"/api/v1/batches/{missing}").status_code == 404
    assert client.post(f"/api/v1/batches/{missing}/cancel").status_code == 404
    assert client.get(f"/api/v1/batches/{missing}/events").status_code == 404
    assert client.get("/api/v1/batches/../evil").status_code in (404, 422)

    # 只建状态不 finalize → report 未就绪
    batch_id = format(8, "032x")
    container.batch_state.create(
        BatchJobState(
            batch_id=batch_id, status="queued", total=1, done=0, failed=0,
            created_at=__import__("datetime").datetime.now(__import__("datetime").timezone.utc),
        ),
        [BatchItemState(batch_id=batch_id, item_index=0, image_name="a.png", status="pending")],
    )
    assert client.get(f"/api/v1/batches/{batch_id}/report").status_code == 404


def test_cancel_marks_pending_items(make_env) -> None:
    client, container = make_env()
    batch_id = format(5, "032x")
    upload_dir = container.config.system.output_dir.parent / "tmp" / batch_id
    upload_dir.mkdir(parents=True)
    for index in range(2):
        (upload_dir / f"{index:03d}.image").write_bytes(_png())
        (upload_dir / f"{index:03d}.mask").write_bytes(_mask_png())
    from datetime import datetime, timezone

    container.batch_state.create(
        BatchJobState(
            batch_id=batch_id, status="queued", total=2, done=0, failed=0,
            created_at=datetime.now(timezone.utc),
        ),
        [
            BatchItemState(batch_id=batch_id, item_index=i, image_name=f"{i}.png", status="pending")
            for i in range(2)
        ],
    )
    container.cancellation.cancel(batch_id)

    batch_chunk.apply(args=[batch_id, [0, 1]])
    batch_finalize.apply(args=[[], batch_id])

    detail = client.get(f"/api/v1/batches/{batch_id}").json()
    assert detail["status"] == "cancelled"
    assert all(item["status"] == "cancelled" for item in detail["items"])

    # API 取消端点：已终态批次也可置标志（幂等），未知批次 404
    response = client.post(f"/api/v1/batches/{batch_id}/cancel")
    assert response.status_code == 200
    assert response.json()["cancel_requested"] is True


def test_chunk_idempotent_on_rerun(make_env) -> None:
    client, container = make_env()
    batch_id = client.post("/api/v1/batches", files=_batch_files(2)).json()["batch_id"]
    before = len(container.history.list(100, 0))

    batch_chunk.apply(args=[batch_id, [0, 1]])  # 重跑：终态项全部跳过

    assert len(container.history.list(100, 0)) == before
    detail = client.get(f"/api/v1/batches/{batch_id}").json()
    assert detail["done"] == 2


def test_transient_error_exhausts_retries(make_env) -> None:
    class BrokenEngine(FakeEngine):
        def inpaint(self, image, mask):
            raise RuntimeError("gpu exploded")

    client, _ = make_env("  celery:\n    retries: 0", engine=BrokenEngine())
    batch_id = client.post("/api/v1/batches", files=_batch_files(2)).json()["batch_id"]

    detail = client.get(f"/api/v1/batches/{batch_id}").json()
    assert detail["status"] == "failed"
    assert detail["failed"] == 2
    assert all(item["error_code"] == "internal_error" for item in detail["items"])


def test_broker_unavailable_maps_to_503(make_env, monkeypatch) -> None:
    client, container = make_env()

    def broken_chord(*args, **kwargs):
        raise RuntimeError("broker down")

    monkeypatch.setattr(
        "hybrid_inpaint.interfaces.api.routers.batches.chord", broken_chord
    )
    response = client.post("/api/v1/batches", files=_batch_files(1))
    assert response.status_code == 503
    assert response.json()["code"] == "model_unavailable"


def test_worker_mode_api_skips_engine_loading(tmp_path: Path) -> None:
    # 生产装配路径：worker 模式 api 进程不加载模型（不注入 engine）
    container = build_container(_config(tmp_path, "  inference:\n    mode: worker"))
    assert container.engine is None
    assert container.inpaint is None
    client = TestClient(create_app(container))
    health = client.get("/api/v1/health").json()
    assert health["model"]["ready"] is False
    assert health["inference_mode"] == "worker"


def test_worker_mode_single_via_queue(make_env) -> None:
    client, container = make_env("  inference:\n    mode: worker")
    health = client.get("/api/v1/health").json()
    assert health["inference_mode"] == "worker"

    response = client.post(
        "/api/v1/inpaint",
        files={
            "image": ("a.png", _png(), "image/png"),
            "mask": ("m.png", _mask_png(), "image/png"),
        },
    )
    assert response.status_code == 202
    task_id = response.json()["task_id"]

    status = client.get(f"/api/v1/tasks/{task_id}").json()
    assert status["status"] == "SUCCESS"
    assert status["result"]["status"] == "ok"
    job_id = status["result"]["job_id"]
    assert client.get(f"/api/v1/outputs/{job_id}").content.startswith(b"\x89PNG")
