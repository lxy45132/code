"""CancellationToken 端口的 Redis 实现（读侧 cancelled，写侧 CancelFlag）。

POST /batches/{id}/cancel 由 api 进程置标志；worker 在 chunk 执行前与项间
轮询检查，完成当前项即停（docs/03 §4 取消设计）。
"""

from __future__ import annotations

import redis

from hybrid_inpaint.domain.services import validate_job_id

CANCEL_TTL_S = 86_400


class RedisCancellationToken:
    """key: hybrid:batch:{batch_id}:cancel；TTL 防孤儿标志长期滞留。"""

    def __init__(
        self, client: redis.Redis, batch_id: str, *, ttl_s: int = CANCEL_TTL_S
    ) -> None:
        validate_job_id(batch_id)  # 拒绝路径穿越/非法字符，防 key 注入
        self._client = client
        self._key = f"hybrid:batch:{batch_id}:cancel"
        self._ttl_s = ttl_s

    @property
    def key(self) -> str:
        return self._key

    def set(self) -> None:
        self._client.set(self._key, "1", ex=self._ttl_s)

    @property
    def cancelled(self) -> bool:
        return bool(self._client.exists(self._key))


class RedisBatchCancellation:
    """BatchCancellation 端口的 Redis 实现（跨进程共享同一 key）。"""

    def __init__(self, client: redis.Redis, *, ttl_s: int = CANCEL_TTL_S) -> None:
        self._client = client
        self._ttl_s = ttl_s

    def cancel(self, batch_id: str) -> None:
        RedisCancellationToken(self._client, batch_id, ttl_s=self._ttl_s).set()

    def token(self, batch_id: str) -> RedisCancellationToken:
        return RedisCancellationToken(self._client, batch_id, ttl_s=self._ttl_s)
