"""CursorStore implementations for DomynRelayChannel (and similar transports).

``InMemoryCursorStore`` is the default: the Domyn relay's own retained window
is short (``MESSAGE_TTL_SECONDS`` defaults to 120s / ``MESSAGE_WINDOW_MAX_COUNT``
to 1000), so a cursor surviving a process
restart only helps for restarts *within* that same short window; outside it,
the relay has nothing left to replay anyway and a fresh start is just as
correct. In-memory keeps the mechanism zero-config and dependency-free for
that common case.

``FileCursorStore`` is an opt-in alternative for deployments that configure a
longer retained window (or want best-effort resume across quick restarts
without relying on ``SeenWindow`` alone) and are fine with local-disk state.
For multi-instance / distributed deployments, implement ``CursorStore``
against a shared backend (Redis, a database, …) instead — the Protocol makes
no assumption about where the cursor lives.
"""

import asyncio
import json
from pathlib import Path
from typing import cast


class InMemoryCursorStore:
    """Default ``CursorStore``: a plain dict, scoped to this process's lifetime.

    Formalizes "no persistence beyond the current run" as a real ``CursorStore``
    implementation (rather than a special-cased ``None``), so ``DomynRelayChannel``
    always has exactly one code path for reading/writing its cursor.
    """

    def __init__(self) -> None:
        self._data: dict[str, str] = {}

    async def load(self, key: str) -> str | None:
        return self._data.get(key)

    async def save(self, key: str, event_id: str) -> None:
        self._data[key] = event_id


class FileCursorStore:
    """Persist cursors to a local JSON file, one entry per key.

    Writes rewrite the whole file (via a temp-file + atomic rename) under an
    ``asyncio.Lock``. This is deliberately simple, not high-throughput —
    ``save()`` is expected to be called at most once per accepted event, not
    from a hot loop.
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._lock = asyncio.Lock()

    async def load(self, key: str) -> str | None:
        async with self._lock:
            value = self._read().get(key)
            return value if isinstance(value, str) else None

    async def save(self, key: str, event_id: str) -> None:
        async with self._lock:
            data = self._read()
            data[key] = event_id
            self._write(data)

    def _read(self) -> dict[str, str]:
        if not self._path.exists():
            return {}
        try:
            return cast(dict[str, str], json.loads(self._path.read_text()))
        except (json.JSONDecodeError, OSError):
            return {}

    def _write(self, data: dict[str, str]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(self._path.suffix + ".tmp")
        tmp.write_text(json.dumps(data))
        tmp.replace(self._path)  # atomic on POSIX
