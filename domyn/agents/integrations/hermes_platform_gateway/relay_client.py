"""Domyn relay WebSocket client.

Pure transport: framing, connect, send_event, receive-loop dispatch.
No business logic — adapter.py owns event routing.
"""

import asyncio
import contextlib
import random
from collections.abc import Awaitable, Callable
from typing import Any

from domyn.agents.integrations.utils import SeenWindow, resume_headers
from domyn.agents.logger import get_logger

logger = get_logger()


def _backoff_delay(attempt: int, *, rng: Callable[[], float] = random.random) -> float:
    """Full-jitter exponential backoff.

    delay = min(30, 0.5 * 2**attempt) * (0.5 + 0.5 * rng())
    """
    base = min(30.0, 0.5 * (2**attempt))
    return base * (0.5 + 0.5 * rng())


class DomynRelayClient:
    """Async WebSocket client for the Domyn relay.

    Owns the connect/reconnect loop and exposes ``send_event`` for outbound
    frames. Inbound frames are passed to a caller-supplied async callback.
    """

    def __init__(
        self,
        ws_url: str,
        headers: dict[str, str],
        on_event: Callable[[Any, dict[str, Any], str | None], Awaitable[None]] | None = None,
    ) -> None:
        self._ws_url = ws_url
        self._headers = headers
        self._on_event = on_event
        self._ws: Any = None
        self._task: asyncio.Task | None = None
        self._stop = asyncio.Event()
        # Single dedup window over every event_id we send or receive. The relay
        # retains events in a window and replays them on (re)connect, so both our
        # own outbound events (echoes) and already-processed inbound events can come
        # back; recording sends and checking receives against one bounded, TTL'd set
        # drops them uniformly. Survives reconnects and cannot grow unbounded.
        self._seen = SeenWindow()

    async def send_event(self, event: Any) -> None:
        from domyn.agents.core import RelayMessage

        if self._ws is None:
            raise RuntimeError("DomynRelayClient: not connected")

        relay_msg = RelayMessage(payload=event)
        msg = relay_msg.model_dump_json()

        # Log only non-sensitive metadata at INFO level to avoid exposing
        # tool parameters, thoughts, content, and other sensitive fields in
        # centralized production logs.
        safe_info = {
            "event_type": str(relay_msg.payload.event_type),
            "event_id": relay_msg.payload.event_id,
            "author": relay_msg.payload.author,
            "conversation_id": relay_msg.payload.conversation_id,
            "interaction_id": relay_msg.payload.interaction_id,
        }
        logger.info("DomynRelayClient: outbound event %s", safe_info)
        # Full payload available at DEBUG level for local troubleshooting
        logger.debug("DomynRelayClient: outbound payload %s", msg[:1000])
        self._seen.add(relay_msg.payload.event_id)
        await self._ws.send(msg)

    async def _consume(self, ws: Any, cursor: dict[str, str | None]) -> None:
        """Iterate frames, parse RelayMessage, dispatch to on_event.

        ``cursor`` is the persistent resume-cursor holder owned by
        ``_connect_loop``; it is advanced in place here as events are processed.
        Dedup uses the shared ``self._seen`` window, which persists across
        reconnects and also holds ids of frames we sent, so replayed duplicates and
        our own echoes are dropped by the same check.
        """
        from domyn.agents.core import RelayMessage

        async for raw in ws:
            try:
                msg = RelayMessage.model_validate_json(raw)
            except Exception as exc:
                # Raw frame (truncated) only at DEBUG so timeouts caused by
                # silently-dropped TOOL_END frames stay diagnosable without
                # leaking content. The platform's schema sometimes diverges
                # from domyn-agents' ToolAction (e.g. missing/extra fields).
                logger.warning(
                    "DomynRelayClient: dropping malformed frame: %s (len=%d)",
                    type(exc).__name__,
                    len(str(raw)),
                )
                logger.debug(
                    "DomynRelayClient: malformed frame detail: %s | raw=%s",
                    str(exc)[:200],
                    str(raw)[:500],
                )
                continue
            if self._on_event is None:
                continue
            # Skip anything already sent or processed and replayed back to us
            # (read-only replay is at-least-once; the seen window + cursor make it
            # exactly-once, and it also covers echoes of our own outbound events).
            event_id = getattr(msg.payload, "event_id", None)
            if not self._seen.is_new(event_id):
                logger.debug("DomynRelayClient: dropping duplicate/own event %s", event_id)
                continue
            if event_id:
                cursor["last"] = event_id
            try:
                # Forward the envelope ``meta`` (W3C ``traceparent`` + platform
                # tags) alongside the payload so the adapter can trace-join and
                # tag its logs.
                #
                # ``run_config`` carries the turn's ``config_id``, which the
                # adapter needs to reconcile pod settings against the (possibly
                # newly activated) configuration. Only that one field is lifted
                # out and passed on: ``run_config`` as a whole is PII, so it must
                # not travel further into the plugin than this boundary.
                config_id = (msg.run_config or {}).get("config_id")
                await self._on_event(msg.payload, msg.meta, config_id)
            except Exception as exc:
                logger.warning("DomynRelayClient: on_event raised: %s", type(exc).__name__)
                logger.debug("DomynRelayClient: on_event detail", exc_info=True)

    async def connect(self) -> None:
        """Spawn the connect loop task. Returns immediately."""
        if self._task is not None and not self._task.done():
            return
        self._stop.clear()
        self._task = asyncio.create_task(self._connect_loop())

    async def disconnect(self) -> None:
        """Stop the connect loop and close the active socket."""
        self._stop.set()
        ws, self._ws = self._ws, None
        if ws is not None:
            with contextlib.suppress(Exception):
                await ws.close()
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._task
            self._task = None

    async def _connect_loop(self) -> None:
        import websockets

        # One cursor holder across the whole reconnect loop so state survives
        # transient reconnects; dedup uses the shared ``self._seen`` window.
        cursor: dict[str, str | None] = {"last": None}

        attempt = 0
        while not self._stop.is_set():
            try:
                # Resume from the last processed event so the relay replays only the
                # events missed while disconnected (built per attempt).
                headers = resume_headers(self._headers, cursor["last"])
                async with websockets.connect(self._ws_url, additional_headers=headers) as ws:
                    self._ws = ws
                    attempt = 0
                    await self._consume(ws, cursor)
            except Exception as exc:
                logger.warning("DomynRelayClient: connection error - %s", type(exc).__name__)
                logger.debug("DomynRelayClient: connection detail: %s", exc)
            finally:
                self._ws = None

            if self._stop.is_set():
                break

            delay = _backoff_delay(attempt)
            logger.debug("DomynRelayClient: reconnecting in %.1fs", delay)
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=delay)
                break  # stop requested during sleep
            except TimeoutError:
                pass
            attempt = min(attempt + 1, 6)
