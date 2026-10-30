"""DomynRelayChannel — outbound WebSocket transport to the Domyn relay.

Owns:
  - One persistent ``asyncio.Queue`` for outbound payloads (survives reconnects).
  - One drain task per connection that pulls from the queue and calls ``ws.send``.
  - An automatic reconnect loop with full-jitter exponential backoff.

``receive()`` is an async generator that:
  1. Opens a WebSocket connection (reconnecting on failure with backoff), sending
     the ``last-event-id`` resume cursor so the relay only replays events after it.
  2. Starts a drain task for the current connection.
  3. Parses each inbound frame as a ``RelayMessage`` / ``BaseEvent``.
  4. Yields ``(conversation_id, event)`` for every routable event type
     (AGENT_START, TOOL_FEEDBACK, TOOL_END, TOOL_ERROR, AGENT_END).
  5. On disconnect, cancels the drain task and reconnects.

``send_event`` is best-effort (``put_nowait``; drops when the queue is full).
``send_event_priority`` is blocking (``await put``; used for TOOL_START and AGENT_END).

The outbound queue persists across reconnects — items enqueued on the old
connection are drained by the new connection's drain task.

Two complementary, non-redundant defenses against the relay replaying old
frames on reconnect (a missing/stale cursor means the WHOLE retained window
— up to ``MESSAGE_WINDOW_MAX_COUNT`` / ``MESSAGE_TTL_SECONDS`` — gets replayed,
including our own past sends):

  - The resume cursor (a ``CursorStore``, ``InMemoryCursorStore`` by default):
    tells the relay "skip everything up to here" so a normal reconnect gets
    only the forward delta. Advanced from BOTH inbound events we accept AND
    events we send (the relay's per-channel offset is one shared sequence
    across both directions). In-memory by default because the relay's own
    retained window is short (``MESSAGE_TTL_SECONDS`` defaults to 120s) — a
    cursor surviving a process restart only matters for restarts *within*
    that same short window; a ``FileCursorStore`` (or a custom ``CursorStore``)
    can be passed in for that case, see ``domyn.agents.channels.cursor_store``.
  - ``SeenWindow`` (``_seen``): dedups by ``event_id`` regardless of the
    cursor. Still needed because the cursor alone can't cover every case: a
    cursor event that's aged out of the relay's own retained window before we
    reconnect (whether the cursor is in-memory or persisted) falls back to a
    full replay on the relay's side — ``_seen`` is what actually drops the
    resulting duplicates.
"""

import asyncio
import contextlib
import random
from collections.abc import AsyncGenerator
from typing import Any

from domyn.agents.channels.cursor_store import InMemoryCursorStore
from domyn.agents.core import BaseEvent, ExecutionEventType, RelayMessage
from domyn.agents.integrations.utils import SeenWindow, resume_headers
from domyn.agents.interfaces import CursorStore
from domyn.agents.logger import get_logger

logger = get_logger()

_ROUTABLE = frozenset(
    {
        ExecutionEventType.AGENT_START,
        ExecutionEventType.TOOL_FEEDBACK,
        ExecutionEventType.TOOL_END,
        ExecutionEventType.TOOL_ERROR,
        ExecutionEventType.AGENT_END,
    }
)

_RECONNECT_BASE_S = 0.5
_RECONNECT_CAP_S = 30.0


def _backoff(attempt: int) -> float:
    """Full-jitter exponential backoff: uniform(0, min(cap, base * 2^attempt))."""
    cap = min(_RECONNECT_CAP_S, _RECONNECT_BASE_S * (2**attempt))
    return float(cap * random.random())


def _parse_frame(raw: str | bytes) -> BaseEvent | None:
    """Parse a relay wire frame into a BaseEvent.

    Tries the ``RelayMessage`` envelope first (production relay), then raw
    ``BaseEvent`` JSON (dev / test).  Returns ``None`` on failure.
    """
    try:
        return RelayMessage.model_validate_json(raw).payload
    except Exception:
        pass
    try:
        return BaseEvent.model_validate_json(raw)
    except Exception:
        return None


class DomynRelayChannel:
    """Outbound WebSocket channel that speaks the Domyn relay protocol.

    Parameters
    ----------
    ws_url:
        Full WebSocket URL, e.g. ``wss://relay.example.com/ws``.
    headers:
        HTTP headers sent on every connect attempt (``api-key``, ``channel-id``,
        ``space-id``, …).
    queue_maxsize:
        Maximum items in the outbound queue.  Best-effort sends are dropped when
        the queue is full; priority sends block until space is available.
    cursor_store:
        ``CursorStore`` used to read/write the resume cursor. Defaults to a
        fresh ``InMemoryCursorStore()`` (cursor lives only for this instance's
        lifetime — a full restart starts with no cursor, i.e. the relay
        replays its whole retained window on the next connect). Pass a
        ``FileCursorStore`` (or a custom ``CursorStore``) to survive quick
        process restarts within the relay's own retention window.
    cursor_key:
        Key under which the cursor is saved/loaded in *cursor_store*. Defaults to
        ``headers["channel-id"]`` (falling back to *ws_url*) so one store can be
        shared across multiple channels/workers.
    """

    def __init__(
        self,
        ws_url: str,
        headers: dict[str, str],
        queue_maxsize: int = 10_000,
        cursor_store: CursorStore | None = None,
        cursor_key: str | None = None,
    ) -> None:
        self._ws_url = ws_url
        self._headers = headers
        self._outbound: asyncio.Queue[str] = asyncio.Queue(maxsize=queue_maxsize)
        self._stop = asyncio.Event()
        # Dedup for frames we've sent (relay echoes) or already processed (relay
        # replays on reconnect). Without this, our own AGENT_END completion frame
        # can come back through receive() and be misrouted to adapter.on_cancel(),
        # cancelling a newer in-flight invocation for the same conversation_id —
        # or a replayed AGENT_START can silently re-run an old request.
        self._seen = SeenWindow()
        # Resume cursor: sent as the `last-event-id` handshake header on every
        # (re)connect so the relay replays only the forward delta instead of
        # the whole retained window. Read fresh from
        # *cursor_store* on every (re)connect attempt (not cached in an instance
        # attribute) so the store is the single source of truth. Advanced from
        # BOTH directions (inbound events we accept AND events we send —
        # send_event* calls _advance_cursor too) since the relay's per-channel
        # offset is a single shared sequence across both directions; without
        # covering sends, a long burst of outbound-only traffic (e.g. several
        # TOOL_START/TOOL_UPDATE in one long turn with no inbound event in
        # between) would leave the cursor stale and rely entirely on ``_seen``
        # to filter the replay. ``_seen`` remains defense-in-depth for what the
        # cursor can't cover: a cursor event evicted from the relay's own
        # retained window (TTL/count) before we could reconnect.
        self._cursor_store: CursorStore = cursor_store or InMemoryCursorStore()
        self._cursor_key = cursor_key or headers.get("channel-id") or ws_url

        self.events_sent: int = 0
        self.events_dropped: int = 0

    # ------------------------------------------------------------------
    # InputChannel contract
    # ------------------------------------------------------------------

    async def send_event(self, conversation_id: str, event: BaseEvent) -> None:
        """Best-effort enqueue.  Increments ``events_dropped`` on queue full."""
        self._seen.add(event.event_id)
        await self._advance_cursor(event.event_id)
        try:
            self._outbound.put_nowait(event.model_dump_json())
        except asyncio.QueueFull:
            self.events_dropped += 1
            logger.warning(
                "DomynRelayChannel: outbound queue full; dropped %s",
                event.event_type.value,
            )

    async def send_event_priority(self, conversation_id: str, event: BaseEvent) -> None:
        """Blocking enqueue — never drops.  Use for TOOL_START and AGENT_END."""
        self._seen.add(event.event_id)
        await self._advance_cursor(event.event_id)
        await self._outbound.put(event.model_dump_json())

    async def receive(self) -> AsyncGenerator[tuple[str, BaseEvent], None]:
        """Connect to the relay, yield inbound events, reconnect on disconnect."""
        try:
            import websockets  # websockets is a core dependency; this guard is defensive
        except ImportError as exc:
            raise ImportError(
                "websockets is required for DomynRelayChannel. Run: pip install websockets"
            ) from exc

        attempt = 0
        while not self._stop.is_set():
            try:
                cursor = await self._cursor_store.load(self._cursor_key)
                headers = resume_headers(self._headers, cursor)
                async with websockets.connect(
                    self._ws_url,
                    additional_headers=headers,
                ) as ws:
                    logger.info("DomynRelayChannel: connected to %s", self._ws_url)
                    attempt = 0
                    drain_task = asyncio.create_task(self._drain_loop(ws), name="relay-ws-drain")
                    try:
                        async for raw in ws:
                            event = _parse_frame(raw)
                            if event is None:
                                logger.debug(
                                    "DomynRelayChannel: dropping unparseable frame %.200s",
                                    str(raw)[:200],
                                )
                                continue
                            if not self._seen.is_new(event.event_id):
                                logger.debug(
                                    "DomynRelayChannel: dropping duplicate/own event %s",
                                    event.event_id,
                                )
                                continue
                            # Advance the resume cursor for every non-duplicate frame we
                            # observe (not just routable ones) so a reconnect never
                            # re-fetches anything already seen, routable or not.
                            await self._advance_cursor(event.event_id)
                            if event.event_type not in _ROUTABLE:
                                logger.debug(
                                    "DomynRelayChannel: ignoring %s",
                                    event.event_type.value,
                                )
                                continue
                            conv_id = event.conversation_id or event.event_id or ""
                            yield conv_id, event
                    finally:
                        drain_task.cancel()
                        with contextlib.suppress(asyncio.CancelledError, Exception):
                            await drain_task

                code = getattr(ws, "close_code", None)
                reason = getattr(ws, "close_reason", None)
                logger.info(
                    "DomynRelayChannel: connection closed (code=%s reason=%r)",
                    code,
                    reason,
                )

            except OSError as exc:
                logger.warning("DomynRelayChannel: connection failed: %s", exc)
            except Exception as exc:
                if "InvalidStatus" in type(exc).__name__:
                    logger.error("DomynRelayChannel: server rejected connection: %s", exc)
                else:
                    logger.warning(
                        "DomynRelayChannel: unexpected error: %s",
                        exc,
                        exc_info=True,
                    )

            if self._stop.is_set():
                break

            delay = _backoff(attempt)
            attempt += 1
            logger.info(
                "DomynRelayChannel: reconnecting in %.1fs (attempt %d)…",
                delay,
                attempt,
            )
            try:
                await asyncio.sleep(delay)
            except asyncio.CancelledError:
                break

    async def close(self) -> None:
        """Signal the receive loop to stop after the current connection closes."""
        self._stop.set()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _advance_cursor(self, event_id: str | None) -> None:
        """Advance the resume cursor by persisting *event_id* to ``_cursor_store``."""
        if not event_id:
            return
        try:
            await self._cursor_store.save(self._cursor_key, event_id)
        except Exception:
            logger.warning("DomynRelayChannel: cursor_store.save failed", exc_info=True)

    async def _drain_loop(self, ws: Any) -> None:
        """Drain the outbound queue to the given WebSocket connection."""
        while True:
            try:
                payload = await self._outbound.get()
            except asyncio.CancelledError:
                return
            try:
                await ws.send(payload)
                self.events_sent += 1
                logger.debug("DomynRelayChannel: sent %.200s", payload[:200])
            except asyncio.CancelledError:
                self._outbound.task_done()
                return
            except Exception:
                self.events_dropped += 1
                logger.warning("DomynRelayChannel: drain send failed", exc_info=True)
            finally:
                self._outbound.task_done()

    # ------------------------------------------------------------------
    # Helpers (used by CLI / expose command)
    # ------------------------------------------------------------------

    @staticmethod
    def from_env(
        *,
        api_key: str,
        channel_id: str,
        space_id: str,
        base_url: str,
        configuration_id: str | None = None,
        cursor_store: CursorStore | None = None,
    ) -> "DomynRelayChannel":
        """Construct a channel from the standard Domyn relay environment values.

        Where to connect and with which headers is the platform SDK's to say
        (it needs ``domyn-platform-sdk``, installed next to domyn-agents).
        """
        from domyn.agents.integrations.domyn_platform_sdk import PlatformSettings, relay_connection

        ws_url, headers = relay_connection(
            PlatformSettings(
                api_key=api_key,
                space_id=space_id,
                channel_id=channel_id,
                base_url=base_url,
                configuration_id=configuration_id,
            )
        )
        return DomynRelayChannel(
            ws_url=ws_url, headers=headers, cursor_store=cursor_store, cursor_key=channel_id
        )
