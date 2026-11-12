"""Injectable time and id sources."""

from __future__ import annotations

import uuid
from datetime import datetime


class SystemClock:
    def now(self) -> datetime:
        return datetime.now().astimezone()


class UuidGenerator:
    def new_id(self) -> str:
        return uuid.uuid4().hex