"""Relay helpers shared by the integrations: dedup of events seen on a connection, resume headers."""

import threading
import time
from collections import OrderedDict
from collections.abc import Callable


def resume_headers(base: dict[str, str], last_event_id: str | None) -> dict[str, str]:
    """Return ``base`` plus the ``last-event-id`` cursor when set; never mutates ``base``."""
    if not last_event_id:
        return dict(base)
    return {**base, "last-event-id": last_event_id}


# Mirror the relay's retained-window bounds (1000 events, 120 s) so a client remembers
# an event_id at least as long as the relay can replay that event (or an echo of our own
# outbound) back to us. Remembering longer only wastes memory; shorter risks a missed
# dedup, so these are a floor that matches the relay.
SEEN_WINDOW_MAX_COUNT = 1000
SEEN_WINDOW_TTL_SECONDS = 120.0


class SeenWindow:
    """Bounded, TTL-expiring dedup of ``event_id``s we've sent or received.

    One instance per relay client; survives reconnects. Both the send path
    (:meth:`add`) and the receive path (:meth:`is_new`) share it, so an event we
    sent (replayed back as an echo) and an event we already processed (replayed on
    reconnect) are deduped by the same structure.

    Bounded by ``max_count`` entries and ``ttl_seconds`` age, pruned oldest-first —
    mirroring the relay's own retained-window prune so the client's dedup memory
    covers exactly the window the relay can replay from, without growing unbounded
    over the process lifetime.
    """

    def __init__(
        self,
        max_count: int = SEEN_WINDOW_MAX_COUNT,
        ttl_seconds: float = SEEN_WINDOW_TTL_SECONDS,
        now: Callable[[], float] = time.monotonic,
    ) -> None:
        # id -> expiry deadline. Constant ttl ⇒ insertion order == expiry order,
        # so the oldest (earliest-expiring) entry is always at the front.
        self._entries: OrderedDict[str, float] = OrderedDict()
        self._max_count = max_count
        self._ttl = ttl_seconds
        self._now = now
        # send and receive paths may run on different event loops/threads
        # (e.g. adapter.call_tool sends off the gateway loop); guard the
        # multi-step prune so concurrent access can't corrupt the OrderedDict.
        self._lock = threading.Lock()

    def _prune(self, now: float) -> None:
        while self._entries:
            oldest = next(iter(self._entries))
            if self._entries[oldest] <= now:
                self._entries.popitem(last=False)
            else:
                break
        while len(self._entries) > self._max_count:
            self._entries.popitem(last=False)

    def add(self, event_id: str | None) -> None:
        """Record an id we sent so its relay echo is later dropped. No-op if falsy."""
        if not event_id:
            return
        with self._lock:
            now = self._now()
            # setdefault: don't extend the deadline / reorder if already present
            # (e.g. the cancel path reuses an inbound AGENT_START event_id).
            self._entries.setdefault(event_id, now + self._ttl)
            self._prune(now)

    def is_new(self, event_id: str | None) -> bool:
        """Return True if *event_id* is new (recording it); False if already seen.

        A falsy ``event_id`` is always new — it cannot be deduped, so it is never
        dropped.
        """
        if not event_id:
            return True
        with self._lock:
            now = self._now()
            self._prune(now)
            if event_id in self._entries:
                return False
            self._entries[event_id] = now + self._ttl
            self._prune(now)
            return True
