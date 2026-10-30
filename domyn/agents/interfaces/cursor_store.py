"""CursorStore Protocol — persists a channel's resume cursor across process restarts."""

from typing import Protocol


class CursorStore(Protocol):
    """Persist the last-seen ``event_id`` (resume cursor) for a channel, keyed by *key*.

    A reconnecting channel (e.g. ``DomynRelayChannel``) sends this cursor back to the
    transport so it replays only the forward delta instead of its whole retained
    window. Held in memory, the cursor is lost on every process restart — a
    ``CursorStore`` gives it a durable home so a restarted worker resumes exactly
    where it left off instead of re-triggering a full-window replay.

    *key* scopes the cursor to one logical connection (e.g. a relay ``channel_id``)
    so a single store instance can be shared across multiple channels/workers.

    Implementations should be cheap to call: ``save`` may be invoked on every
    accepted event, not just at connect/disconnect boundaries.
    """

    async def load(self, key: str) -> str | None:
        """Return the last saved cursor for *key*, or ``None`` if none is stored."""
        ...

    async def save(self, key: str, event_id: str) -> None:
        """Persist *event_id* as the latest cursor for *key*."""
        ...
