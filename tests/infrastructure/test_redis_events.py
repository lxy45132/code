"""EventBus/Cache/CancelToken 端口契约测试。

同一套断言跑两套实现（InMemory fake 与 Redis 适配器，后者用 fakeredis
模拟协议层），保证 fake 语义与真实适配器一致——Redis 就绪后另有
``test_redis_integration.py`` 对真实服务端复测。
"""

from __future__ import annotations

import json

import pytest

from hybrid_inpaint.domain.errors import ValidationError
from hybrid_inpaint.infrastructure.redis_events import RedisCache, RedisEventBus

from tests.application.fakes import InMemoryCache, InMemoryCancelToken, InMemoryEventBus


@pytest.fixture(params=["memory", "redis"])
def event_bus(request) -> object:
    if request.param == "memory":
        return InMemoryEventBus(maxlen=5)
    fakeredis = pytest.importorskip("fakeredis")
    return RedisEventBus(fakeredis.FakeStrictRedis(decode_responses=True), maxlen=5)


@pytest.fixture(params=["memory", "redis"])
def cache(request) -> object:
    if request.param == "memory":
        return InMemoryCache()
    fakeredis = pytest.importorskip("fakeredis")
    return RedisCache(fakeredis.FakeStrictRedis(decode_responses=True))


@pytest.fixture(params=["memory", "redis"])
def cancel_token(request) -> object:
    if request.param == "memory":
        return InMemoryCancelToken()
    fakeredis = pytest.importorskip("fakeredis")
    from hybrid_inpaint.infrastructure.celery_cancel import RedisCancellationToken

    return RedisCancellationToken(
        fakeredis.FakeStrictRedis(decode_responses=True), format(7, "032x")
    )


def _entries(bus: object, stream: str) -> list[tuple[str, dict]]:
    """归一化读取：返回 (event_type, data) 序列。"""
    if isinstance(bus, InMemoryEventBus):
        return list(bus.events.get(stream, []))
    entries = bus._client.xrange(stream)  # noqa: SLF001 - 契约检查
    return [(fields["type"], json.loads(fields["data"])) for _, fields in entries]


def test_publish_keeps_order_and_payload(event_bus: object) -> None:
    stream = "hybrid:batch:00aa"
    event_bus.publish(stream, "item_started", {"index": 0, "image": "a.png"})
    event_bus.publish(stream, "item_done", {"index": 0, "latency_ms": 12.5, "ok": True})
    event_bus.publish(stream, "finished", {"done": 2, "失败": "中文也保真"})

    entries = _entries(event_bus, stream)
    assert [e[0] for e in entries] == ["item_started", "item_done", "finished"]
    assert entries[0][1] == {"index": 0, "image": "a.png"}
    assert entries[1][1] == {"latency_ms": 12.5, "ok": True, "index": 0}
    assert entries[2][1]["失败"] == "中文也保真"


def test_publish_trims_to_maxlen(event_bus: object) -> None:
    stream = "hybrid:batch:00bb"
    for index in range(8):
        event_bus.publish(stream, "progress", {"index": index})

    entries = _entries(event_bus, stream)
    assert len(entries) == 5
    assert [e[1]["index"] for e in entries] == [3, 4, 5, 6, 7]  # 最旧被裁


def test_cache_roundtrip_overwrite_and_delete(cache: object) -> None:
    cache.set("hybrid:cache:models", "v1", ttl_s=5)
    assert cache.get("hybrid:cache:models") == "v1"

    cache.set("hybrid:cache:models", "v2", ttl_s=5)  # 覆盖
    assert cache.get("hybrid:cache:models") == "v2"

    cache.delete("hybrid:cache:models")
    assert cache.get("hybrid:cache:models") is None
    cache.delete("hybrid:cache:models")  # 幂等
    assert cache.get("hybrid:cache:missing") is None


def test_cancel_token_set_then_cancelled(cancel_token: object) -> None:
    assert cancel_token.cancelled is False
    cancel_token.set()
    assert cancel_token.cancelled is True
    cancel_token.set()  # 幂等
    assert cancel_token.cancelled is True


def test_redis_bus_rejects_invalid_stream_and_empty_type() -> None:
    fakeredis = pytest.importorskip("fakeredis")
    bus = RedisEventBus(fakeredis.FakeStrictRedis(decode_responses=True))
    with pytest.raises(ValidationError):
        bus.publish("hybrid:batch:00cc\nINJECT", "progress", {})
    with pytest.raises(ValidationError):
        bus.publish("hybrid:batch:00cc", "", {})
    with pytest.raises(ValidationError):
        bus.publish("", "progress", {})


def test_redis_cache_rejects_bad_key_and_ttl() -> None:
    fakeredis = pytest.importorskip("fakeredis")
    cache = RedisCache(fakeredis.FakeStrictRedis(decode_responses=True))
    with pytest.raises(ValidationError):
        cache.set("bad key", "v", ttl_s=5)
    with pytest.raises(ValidationError):
        cache.set("hybrid:cache:models", "v", ttl_s=0)
    with pytest.raises(ValidationError):
        cache.get("")


def test_redis_cancel_token_rejects_bad_batch_id() -> None:
    fakeredis = pytest.importorskip("fakeredis")
    from hybrid_inpaint.infrastructure.celery_cancel import RedisCancellationToken

    with pytest.raises(ValidationError):
        RedisCancellationToken(fakeredis.FakeStrictRedis(decode_responses=True), "../evil")
