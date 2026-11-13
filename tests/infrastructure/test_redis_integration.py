"""对真实 Redis 服务端的集成测试（默认 skip）。

启用方式：set HYBRID_INPAINT_TEST_REDIS_URL=redis://127.0.0.1:6379/15
（建议用独立 db；测试用 uuid key 隔离并逐个清理，不 flush 整库）。
"""

from __future__ import annotations

import os
import time

import pytest

pytestmark = [
    pytest.mark.redis,
    pytest.mark.skipif(
        not os.environ.get("HYBRID_INPAINT_TEST_REDIS_URL"),
        reason="set HYBRID_INPAINT_TEST_REDIS_URL to run Redis integration tests",
    ),
]


@pytest.fixture()
def client():
    import redis

    return redis.Redis.from_url(
        os.environ["HYBRID_INPAINT_TEST_REDIS_URL"], decode_responses=True
    )


def test_event_bus_stream_roundtrip(client) -> None:
    from hybrid_inpaint.infrastructure.redis_events import RedisEventBus

    stream = f"hybrid:batch:{format(2026, '032x')}"
    bus = RedisEventBus(client)
    bus.publish(stream, "item_started", {"index": 0})
    bus.publish(stream, "item_done", {"index": 0, "latency_ms": 1.5})

    entries = client.xrange(stream)
    assert [fields["type"] for _, fields in entries] == ["item_started", "item_done"]
    assert "1.5" in entries[1][1]["data"]
    client.delete(stream)


def test_cache_ttl_expires(client) -> None:
    from hybrid_inpaint.infrastructure.redis_events import RedisCache

    key = f"hybrid:cache:{format(2027, '032x')}"
    cache = RedisCache(client)
    cache.set(key, "v", ttl_s=1)
    assert cache.get(key) == "v"
    time.sleep(1.1)
    assert cache.get(key) is None


def test_cancel_token_roundtrip_across_instances(client) -> None:
    from hybrid_inpaint.infrastructure.celery_cancel import RedisCancellationToken

    batch_id = format(2028, "032x")
    writer = RedisCancellationToken(client, batch_id)
    reader = RedisCancellationToken(client, batch_id)  # 独立实例模拟跨进程
    assert reader.cancelled is False
    writer.set()
    assert reader.cancelled is True
    client.delete(writer.key)


def test_ping(client) -> None:
    assert client.ping() is True
