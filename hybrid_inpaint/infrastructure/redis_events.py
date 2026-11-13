"""EventBus/Cache 端口的 Redis 适配器（Stream 事件 + TTL 缓存）。

进度链路（docs/03 §7）：MySQL 权威状态 → Redis Stream 跨进程事件 → SSE 推送。
Stream 相对 Pub/Sub 的优势：条目 id 单调、订阅者掉线可凭 Last-Event-ID 回放。
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import Any

import redis

from hybrid_inpaint.domain.errors import ValidationError

STREAM_MAXLEN = 10_000
CACHE_TTL_S = 5

# 可见 ASCII 且无空白/控制字符：内部 key（hybrid:batch:<uuid> 等）足够且防注入。
_KEY_PATTERN = re.compile(r"^[\x21-\x7e]+$")


def build_redis_client(url: str) -> redis.Redis:
    """业务库连接（decode_responses=True：事件/缓存均为字符串语义）。

    短超时：Redis 不可达时快速失败，不拖慢主流程（03 文档 §13）。
    """
    return redis.Redis.from_url(
        url,
        decode_responses=True,
        socket_connect_timeout=1.0,
        socket_timeout=1.0,
    )


def _validate_key(name: str, what: str) -> None:
    if not _KEY_PATTERN.fullmatch(name):
        raise ValidationError(f"invalid {what}: {name!r}")


class RedisEventBus:
    """EventBus 端口的 Redis Stream 实现。

    XADD 条目 id 即单调 seq；MAXLEN 兜底防超长批次撑爆 Redis；
    SSE 端点用 XREAD block 消费（C7d 落地）。
    """

    def __init__(self, client: redis.Redis, *, maxlen: int = STREAM_MAXLEN) -> None:
        self._client = client
        self._maxlen = maxlen

    def publish(self, stream: str, event_type: str, data: Mapping[str, Any]) -> None:
        _validate_key(stream, "stream name")
        if not event_type:
            raise ValidationError("event_type must not be empty")
        payload = json.dumps(dict(data), ensure_ascii=False, separators=(",", ":"))
        self._client.xadd(
            stream,
            {"type": event_type, "data": payload},
            maxlen=self._maxlen,
            approximate=False,
        )


class RedisCache:
    """Cache 端口的 Redis 实现：GET/SET EX/DEL，写操作后 DEL 失效。"""

    def __init__(self, client: redis.Redis) -> None:
        self._client = client

    def get(self, key: str) -> str | None:
        _validate_key(key, "cache key")
        return self._client.get(key)

    def set(self, key: str, value: str, ttl_s: int) -> None:
        _validate_key(key, "cache key")
        if ttl_s <= 0:
            raise ValidationError("ttl_s must be positive")
        self._client.set(key, value, ex=ttl_s)

    def delete(self, key: str) -> None:
        _validate_key(key, "cache key")
        self._client.delete(key)
