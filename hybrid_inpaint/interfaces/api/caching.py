"""查询缓存助手（docs/03 §7：history/models TTL 5s，写后失效）。

Redis 故障不影响主流程：缓存读写失败一律静默降级到 DB。
历史列表用 generation key 失效（写操作 bump gen，旧条目靠 TTL 自然过期）。
"""

from __future__ import annotations

import time

from hybrid_inpaint.bootstrap import Container
from hybrid_inpaint.interfaces.api.schemas.history import HistoryPage

CACHE_TTL_S = 5
_GEN_KEY = "hybrid:cache:history:gen"
_MODELS_KEY = "hybrid:cache:models"

# 熔断：Redis 故障后 5s 内不再尝试，避免每请求重复连接超时。
_DOWN_COOLDOWN_S = 5.0
_down_until = 0.0


def _available() -> bool:
    return time.monotonic() >= _down_until


def _trip() -> None:
    global _down_until
    _down_until = time.monotonic() + _DOWN_COOLDOWN_S


def _safe_get(container: Container, key: str) -> str | None:
    if not _available():
        return None
    try:
        return container.cache.get(key)
    except Exception:
        _trip()
        return None


def _safe_set(container: Container, key: str, value: str, ttl_s: int = CACHE_TTL_S) -> None:
    if not _available():
        return
    try:
        container.cache.set(key, value, ttl_s=ttl_s)
    except Exception:
        _trip()


def bump_history_cache(container: Container) -> None:
    generation = int(_safe_get(container, _GEN_KEY) or "0")
    _safe_set(container, _GEN_KEY, str(generation + 1), ttl_s=86_400)


def get_history_page(
    container: Container, limit: int, offset: int
) -> HistoryPage | None:
    generation = _safe_get(container, _GEN_KEY) or "0"
    cached = _safe_get(container, f"hybrid:cache:history:{generation}:{limit}:{offset}")
    if cached is None:
        return None
    try:
        return HistoryPage.model_validate_json(cached)
    except Exception:
        return None


def set_history_page(container: Container, page: HistoryPage) -> None:
    generation = _safe_get(container, _GEN_KEY) or "0"
    _safe_set(
        container,
        f"hybrid:cache:history:{generation}:{page.limit}:{page.offset}",
        page.model_dump_json(),
    )


def get_models_cache(container: Container) -> str | None:
    return _safe_get(container, _MODELS_KEY)


def set_models_cache(container: Container, payload: str) -> None:
    _safe_set(container, _MODELS_KEY, payload)
