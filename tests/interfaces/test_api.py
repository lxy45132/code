"""C7c：FastAPI 骨架 TestClient 测试（fake engine + 真实 sqlite/codec/artifacts）。

覆盖：错误映射 422/413/404/503/500、统一错误体、RequestID、CORS、
API Key、单图全流程、零洞恒等、历史分页/导出/删除、并发 FIFO。
"""

from __future__ import annotations

import io
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image as PILImage

from hybrid_inpaint.bootstrap import build_container
from hybrid_inpaint.domain.errors import BusyError
from hybrid_inpaint.interfaces.api.app import create_app
from tests.application.fakes import (
    FakeEngine,
    InMemoryBatchCancellation,
    InMemoryCache,
    InMemoryEventBus,
)


def _config(tmp_path: Path, extra: str = "") -> Path:
    path = tmp_path / "system.yaml"
    lines = [
        "contract_version: 1",
        "experiment_id: api_test",
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


def _mask_png(size: int = 8, hole: bool = True) -> bytes:
    array = np.zeros((size, size), dtype=np.uint8)
    if hole:
        array[2:6, 2:6] = 255
    buf = io.BytesIO()
    PILImage.fromarray(array).save(buf, format="PNG")
    return buf.getvalue()


def _files(size: int = 8, hole: bool = True) -> dict:
    return {
        "image": ("image.png", _png(size), "image/png"),
        "mask": ("mask.png", _mask_png(size, hole), "image/png"),
    }


@pytest.fixture()
def container(tmp_path: Path):
    return build_container(
        _config(tmp_path),
        engine=FakeEngine(),
        events=InMemoryEventBus(),
        cache=InMemoryCache(),
        cancellation=InMemoryBatchCancellation(),
    )


@pytest.fixture()
def client(container) -> TestClient:
    return TestClient(create_app(container))


def _post_inpaint(client: TestClient, **kwargs) -> object:
    return client.post("/api/v1/inpaint", files=_files(), **kwargs)


def test_health_and_request_id(client: TestClient) -> None:
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model"]["ready"] is True
    assert body["model"]["device"] == "fake"
    assert body["storage_backend"] == "sqlite"
    assert body["inference_mode"] == "inline"
    assert len(response.headers["X-Request-ID"]) == 32


def test_request_id_echo_and_regeneration(client: TestClient) -> None:
    echoed = client.get("/api/v1/health", headers={"X-Request-ID": "abc-123"})
    assert echoed.headers["X-Request-ID"] == "abc-123"
    regenerated = client.get("/api/v1/health", headers={"X-Request-ID": "bad id\n"})
    assert len(regenerated.headers["X-Request-ID"]) == 32


def test_models_endpoint(client: TestClient) -> None:
    response = client.get("/api/v1/models")
    assert response.status_code == 200
    assert response.json()["model_id"] == "tiny"


def test_inpaint_roundtrip(client: TestClient) -> None:
    response = _post_inpaint(client, data={"note": "web 冒烟"})
    assert response.status_code == 200
    body = response.json()
    assert len(body["job_id"]) == 32
    assert body["output_url"] == f"/api/v1/outputs/{body['job_id']}"
    assert body["input_size"] == [8, 8]
    assert 0.0 < body["mask_ratio"] < 1.0

    output = client.get(body["output_url"])
    assert output.status_code == 200
    assert output.headers["content-type"] == "image/png"
    assert output.content.startswith(b"\x89PNG")

    detail = client.get(f"/api/v1/history/{body['job_id']}").json()
    assert detail["note"] == "web 冒烟"
    assert detail["mode"] == "web"


def test_inpaint_zero_mask_is_identity(client: TestClient) -> None:
    response = client.post("/api/v1/inpaint", files=_files(hole=False))
    assert response.status_code == 200
    assert response.json()["mask_ratio"] == 0.0
    output = client.get(response.json()["output_url"])
    decoded = np.asarray(PILImage.open(io.BytesIO(output.content)).convert("RGB"))
    expected = np.asarray(PILImage.open(io.BytesIO(_png())).convert("RGB"))
    assert np.array_equal(decoded, expected)


def test_inpaint_invalid_image_is_422(client: TestClient) -> None:
    response = client.post(
        "/api/v1/inpaint",
        files={
            "image": ("image.png", b"not a png", "image/png"),
            "mask": ("mask.png", _mask_png(), "image/png"),
        },
    )
    assert response.status_code == 422
    assert response.json()["code"] == "validation_error"


def test_inpaint_oversize_is_413(client: TestClient) -> None:
    response = client.post("/api/v1/inpaint", files=_files(size=32))  # max_side=16
    assert response.status_code == 413
    assert response.json()["code"] == "resource_limit"


def test_inpaint_missing_mask_is_422(client: TestClient) -> None:
    response = client.post(
        "/api/v1/inpaint", files={"image": ("image.png", _png(), "image/png")}
    )
    assert response.status_code == 422
    body = response.json()
    assert body["code"] == "validation_error"
    assert "detail" in body


def test_outputs_not_found_and_invalid_id(client: TestClient) -> None:
    missing = client.get(f"/api/v1/outputs/{format(9, '032x')}")
    assert missing.status_code == 404
    assert missing.json()["code"] == "not_found"
    invalid = client.get("/api/v1/outputs/../evil")
    assert invalid.status_code in (404, 422)


def test_history_pagination_and_order(client: TestClient) -> None:
    job_ids = [_post_inpaint(client).json()["job_id"] for _ in range(3)]
    page = client.get("/api/v1/history", params={"limit": 2, "offset": 0}).json()
    assert page["limit"] == 2 and page["offset"] == 0
    assert len(page["items"]) == 2
    rest = client.get("/api/v1/history", params={"limit": 2, "offset": 2}).json()
    assert len(rest["items"]) == 1
    listed = [item["job_id"] for item in page["items"] + rest["items"]]
    assert set(listed) == set(job_ids)


def test_history_invalid_limit_is_422(client: TestClient) -> None:
    response = client.get("/api/v1/history", params={"limit": 0})
    assert response.status_code == 422
    assert response.json()["code"] == "validation_error"


def test_history_delete_then_404(client: TestClient) -> None:
    job_id = _post_inpaint(client).json()["job_id"]
    assert client.delete(f"/api/v1/history/{job_id}").status_code == 204
    assert client.get(f"/api/v1/history/{job_id}").status_code == 404
    assert client.delete(f"/api/v1/history/{job_id}").status_code == 404


def test_history_delete_files_removes_artifacts(client: TestClient) -> None:
    job_id = _post_inpaint(client).json()["job_id"]
    response = client.delete(
        f"/api/v1/history/{job_id}", params={"delete_files": "true"}
    )
    assert response.status_code == 204
    assert client.get(f"/api/v1/outputs/{job_id}").status_code == 404


def test_history_export_csv_columns(client: TestClient) -> None:
    _post_inpaint(client)
    response = client.get("/api/v1/history/export")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/csv")
    header = response.text.splitlines()[0]
    assert header == (
        "job_id,created_at,mode,input,mask,output,model_version,"
        "width,height,mask_ratio,latency_ms,note"
    )


def test_busy_maps_to_503(client: TestClient, container, monkeypatch) -> None:
    def raise_busy(request):
        raise BusyError("inference engine is busy")

    monkeypatch.setattr(container.inpaint, "execute", raise_busy)
    response = _post_inpaint(client)
    assert response.status_code == 503
    assert response.json()["code"] == "busy"


def test_unhandled_error_maps_to_500(container, monkeypatch) -> None:
    def raise_runtime(request):
        raise RuntimeError("boom")

    monkeypatch.setattr(container.inpaint, "execute", raise_runtime)
    client = TestClient(create_app(container), raise_server_exceptions=False)
    response = _post_inpaint(client)
    assert response.status_code == 500
    assert response.json() == {
        "code": "internal_error",
        "message": "internal server error",
    }


def test_api_key_enforcement(tmp_path: Path) -> None:
    container = build_container(
        _config(tmp_path, "  api:\n    api_key: secret"),
        engine=FakeEngine(),
        cache=InMemoryCache(),
    )
    client = TestClient(create_app(container))
    assert client.get("/api/v1/models").status_code == 401
    assert (
        client.get("/api/v1/models", headers={"X-API-Key": "wrong"}).status_code == 401
    )
    assert (
        client.get("/api/v1/models", headers={"X-API-Key": "secret"}).status_code == 200
    )
    assert client.get("/api/v1/health").status_code == 200  # 探针免鉴权


def test_cors_preflight(client: TestClient) -> None:
    response = client.options(
        "/api/v1/inpaint",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://localhost:5173"


def test_concurrent_inpaint_all_complete(client: TestClient) -> None:
    with ThreadPoolExecutor(max_workers=4) as pool:
        responses = list(pool.map(lambda _: _post_inpaint(client), range(4)))
    assert all(response.status_code == 200 for response in responses)
    job_ids = {response.json()["job_id"] for response in responses}
    assert len(job_ids) == 4
    page = client.get("/api/v1/history", params={"limit": 10}).json()
    assert len(page["items"]) == 4


_DIST = Path(__file__).resolve().parents[2] / "frontend" / "dist"


@pytest.mark.skipif(not _DIST.is_dir(), reason="frontend/dist not built")
def test_spa_hosting_and_fallback(client: TestClient) -> None:
    root = client.get("/")
    assert root.status_code == 200
    assert 'id="app"' in root.text

    route = client.get("/inpaint")  # Vue Router history 模式 fallback
    assert route.status_code == 200
    assert 'id="app"' in route.text

    unknown_api = client.get("/api/v1/nonexistent")
    assert unknown_api.status_code == 404
    assert unknown_api.json()["code"] == "not_found"

    assets = list((_DIST / "assets").glob("*.js"))
    assert assets, "dist/assets 应有构建产物"
    asset = client.get(f"/assets/{assets[0].name}")
    assert asset.status_code == 200
