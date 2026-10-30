"""WebSocket-based RunnerTransport for DelegateAgent.

Thin wire layer: connects to a WebSocket relay, sends an ``AGENT_START`` to
kick off the external agent, and streams every event it receives back to the
caller with ``author`` normalised to *agent_name*.

**Reconnect-per-boundary behaviour**

The connection is opened fresh for each *boundary* event (any non-partial
event).  Partial / token-level ``RESPONSE`` events are yielded from the same
connection without reconnecting.  This prevents a long-lived socket from
accumulating stale events from concurrent conversations.

When the runner executes a local tool it calls ``send_result()`` which stores
the ``TOOL_END`` event.  The next ``execute()`` iteration reconnects, sends the
pending event, and then waits for the next boundary.

``send_result()`` lets the caller push ``TOOL_END`` / ``LLM_END`` results back
to the external agent while the stream is active.  All higher-level logic —
tool routing, event-type conversion — belongs in the ``DelegateAgent`` that
consumes this transport.

Wire it up::

    from domyn.agents.runner.websocket_transport import WebSocketTransport
    from domyn.agents.agents import DelegateAgent

    transport = WebSocketTransport(ws_url="ws://localhost:8765")
    agent = DelegateAgent(
        name="MathAgent",
        description="...",
        transport=transport,
    )

The caller is responsible for adding the ``DelegateAgent`` to its parent's
``sub_agents`` list (or using it as the entrypoint of a ``Runner``).
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from typing import Any, ClassVar

import websockets
import websockets.asyncio.client as ws_client

from domyn.agents.core import BaseEvent, ExecutionEventType, InteractionContext, Part
from domyn.agents.logger import get_logger
from domyn.agents.runner.transport import RunnerTransport


def _is_streaming_event(event: BaseEvent) -> bool:
    """Return True for events that should NOT trigger a reconnect.

    Partial RESPONSE tokens are already streaming by definition.

    LLM_START / LLM_END are fire-and-forget observability events emitted by
    external agents (e.g. LangGraph) between tool calls.  They never require a
    runner response.
    """
    if event.event_type == ExecutionEventType.RESPONSE and event.is_partial is True:
        return True
    return event.event_type in (ExecutionEventType.LLM_START, ExecutionEventType.LLM_END)


def _is_terminal_event(event: BaseEvent) -> bool:
    """Return True when the full delegate session is over."""
    if event.event_type == ExecutionEventType.AGENT_END:
        return True
    return event.event_type == ExecutionEventType.RESPONSE and not event.is_partial


class WebSocketTransport(RunnerTransport):
    """Bidirectional WebSocket transport for ``DelegateAgent``.

    Parameters
    ----------
    type:
        ``"websocket"`` — used as the ``transport.type`` discriminator in configs.
    ws_url:
        ``ws://`` or ``wss://`` URL of the relay server.
    headers:
        Extra HTTP headers sent on the WebSocket handshake (e.g.
        ``{"channel-id": "abc", "Authorization": "Bearer …"}``).
    timeout:
        Seconds to wait for the next message; on expiry the delegation ends with
        an ``AGENT_END`` (``error_code='delegate_timeout'``).
    ping_interval:
        Interval in seconds between WebSocket keep-alive pings.  ``None``
        disables pings.
    """

    type: ClassVar[str] = "websocket"

    @classmethod
    def from_config(cls, config: dict) -> WebSocketTransport:
        return cls(
            ws_url=config.get("url", ""),
            headers=config.get("headers"),
            timeout=float(config.get("timeout", 60.0)),
            ping_interval=config.get("ping_interval", 20.0),
        )

    def __init__(
        self,
        ws_url: str,
        headers: dict[str, str] | None = None,
        timeout: float = 60.0,
        ping_interval: float | None = 20.0,
    ) -> None:
        self._ws_url = ws_url
        self._headers = headers or {}
        self._timeout = timeout
        self._ping_interval = ping_interval
        # Per-conversation pending queues keyed by conversation_id.
        # This prevents cross-conversation data leakage in concurrent scenarios.
        self._pending_queues: dict[str, asyncio.Queue[BaseEvent]] = {}
        # Conversations whose remote run this transport started and that have not
        # ended yet, with the agent's name: what cancel() may stop.
        self._running: dict[str, str] = {}

    # ------------------------------------------------------------------
    # RunnerTransport interface
    # ------------------------------------------------------------------

    async def execute(
        self,
        agent_name: str,
        context: InteractionContext,
        run_config: dict[str, Any] | None = None,
    ) -> AsyncGenerator[BaseEvent, None]:
        """Stream events for *agent_name* from the remote WebSocket endpoint.

        Opens a fresh connection for each boundary event so no socket is held
        open between turns.  Yields every received event with ``author``
        normalised to *agent_name*.  Terminates after ``AGENT_END`` or a final
        ``RESPONSE``.

        ``run_config`` (visibility-filtered by the caller) rides in the
        ``RelayMessage`` envelope ``meta``, kept out of the ``BaseEvent`` payload.
        Held as a local rather than instance state so concurrent conversations
        sharing this transport never see each other's config.

        Replays are filtered on two axes: events belonging to another turn are dropped
        (cross-turn, stateless — see ``RunnerTransport._is_current_turn``), and every
        event surviving that is deduped by ``event_id`` (intra-turn, since a mid-turn
        reconnect replays events carrying this turn's id). ``seen`` is a local for the
        lifetime of this call, seeded from ``interaction_history`` so a fresh-transport
        resume still dedups replays.
        """
        # Local dedup state (no instance attribute, no cleanup needed).
        seen: set[str] = self._seed_seen(context)
        turn_id: str = context.turn_id or ""

        agent_start = self._build_outbound_event(agent_name, context)

        connect_kwargs: dict[str, Any] = {
            "open_timeout": self._timeout,
        }
        if self._ping_interval is not None:
            connect_kwargs["ping_interval"] = self._ping_interval

        # Resume cursor: sent as the `last-event-id` handshake header on every
        # (re)connect so the relay replays only forward events after it. Local to this
        # execute() call. With log replay on reconnect the connected-but-unread loss
        # during a local tool is recovered, giving WS parity with SSE.
        last_event_id: str | None = None

        # First connection sends AGENT_START unless send_result() was called before
        # execute() (HITL resume), in which case there's a pending event for this
        # specific conversation.
        conversation_id = context.conversation_id
        outbound: BaseEvent | None = None
        pending_queue = self._pending_queues.get(conversation_id)
        if pending_queue is not None and not pending_queue.empty():
            outbound = await pending_queue.get()
            if pending_queue.empty():
                del self._pending_queues[conversation_id]
        if outbound is None:
            outbound = agent_start

        # Tracks depth of remote sub-agent calls so that intermediate AGENT_END
        # events (sub-agent completions) are not mistaken for the root session end.
        # Persists across reconnects because a reconnect mid-nesting should resume
        # tracking where it left off.
        _remote_nesting = 0

        while True:
            get_logger().info(f"[WS transport] Connecting to {self._ws_url}")
            boundary_event: BaseEvent | None = None

            # Per-connection handshake headers: opt into read-only forward-log replay
            # by identifying the conversation and the resume cursor.
            headers = dict(self._headers)
            if conversation_id:
                headers["conversation-id"] = conversation_id
            if last_event_id:
                headers["last-event-id"] = last_event_id

            try:
                async with ws_client.connect(
                    self._ws_url, additional_headers=headers, **connect_kwargs
                ) as ws:  # type: ignore[arg-type]
                    if outbound is not None:
                        get_logger().debug(
                            f"[WS transport] Sending {outbound.event_type.value} "
                            f"for agent '{agent_name}'"
                        )
                        # Record our own outbound event so the relay replaying it back
                        # to us on reconnect is deduped rather than re-processed.
                        if outbound.event_id:
                            seen.add(outbound.event_id)
                        await ws.send(self._wrap_outbound(outbound, run_config))
                        if conversation_id:
                            self._running[conversation_id] = agent_name

                    async for event in self._relay(ws, agent_name):
                        # Drop events from another turn before the nesting logic, so a
                        # replayed AGENT_CALL/AGENT_END never double-counts nesting. The
                        # relay's retained log is keyed by channel and shared by every
                        # turn (and, on the WS route, by every conversation) of that
                        # channel, and it replays the whole window on a cursor-less
                        # connect — which is every first connect, since the transport is
                        # rebuilt per turn.
                        if not self._is_current_turn(event, turn_id):
                            get_logger().debug(
                                f"[WS transport] Skipping {event.event_type.value} from turn "
                                f"{event.turn_id} (expected {turn_id})"
                            )
                            continue
                        # Dedup by event_id before the nesting logic so replayed
                        # AGENT_CALL/AGENT_END never double-count nesting.
                        if not self._mark_seen(seen, event):
                            get_logger().debug(
                                f"[WS transport] Dropping duplicate event {event.event_id}"
                            )
                            continue
                        if event.event_id:
                            last_event_id = event.event_id
                        etype = event.event_type
                        if etype == ExecutionEventType.AGENT_CALL:
                            if _remote_nesting == 0:
                                event = event.model_copy(update={"author": agent_name})
                            _remote_nesting += 1
                            yield event
                            continue
                        if etype == ExecutionEventType.AGENT_END and _remote_nesting > 0:
                            _remote_nesting -= 1
                            yield event
                            continue
                        if (
                            etype == ExecutionEventType.RESPONSE
                            and not event.is_partial
                            and _remote_nesting > 0
                        ):
                            # Final response from a remote sub-agent: the root agent is still
                            # running, so yield for observability but don't break/reconnect.
                            yield event
                            continue
                        # Root-level event (nesting == 0): normalise author to agent_name.
                        # Sub-agent events already continued above with their original author.
                        event = event.model_copy(update={"author": agent_name})
                        yield event
                        if not _is_streaming_event(event):
                            boundary_event = event
                            break  # close connection, surface event to caller
            except TimeoutError:
                error_msg = (
                    f"The delegated agent '{agent_name}' could not be reached "
                    f"(connection timeout to {self._ws_url}). "
                    "The connection has been closed. Please check the agent's availability and connection."
                )
                get_logger().warning(f"[WS transport] Connect timeout to {self._ws_url} — stopping")
                yield BaseEvent(
                    event_type=ExecutionEventType.AGENT_END,
                    author=agent_name,
                    error_code="delegate_timeout",
                    error_message=error_msg,
                    content=[Part(text=error_msg)],
                )
                break
            except websockets.exceptions.ConnectionClosed as exc:
                get_logger().info(
                    f"[WS transport] Connection closed unexpectedly: {type(exc).__name__}"
                )
                get_logger().debug(f"[WS transport] Connection closed detail: {exc}")
                break

            if boundary_event is None:
                get_logger().warning("[WS transport] Connection closed without a boundary event")
                break

            if _is_terminal_event(boundary_event):
                get_logger().info(f"[WS transport] Session complete for agent '{agent_name}'")
                self._running.pop(conversation_id, None)
                break
            # Non-terminal boundary: pick up the queued result for this conversation.
            pending_queue = self._pending_queues.get(conversation_id)
            if pending_queue is not None and not pending_queue.empty():
                outbound = await pending_queue.get()
                if pending_queue.empty():
                    del self._pending_queues[conversation_id]
            else:
                outbound = None

    async def send_result(self, event: BaseEvent, conversation_id: str | None = None) -> None:
        """Store a result event to be sent on the next reconnect.

        The event is queued in a conversation-specific buffer keyed by conversation_id
        to prevent cross-conversation leakage. If conversation_id is None, fails
        """
        conv_id = conversation_id
        if conv_id is None:
            raise ValueError("conversation_id is required")
        if event.conversation_id != conv_id:
            event = event.model_copy(update={"conversation_id": conv_id})
        if conv_id not in self._pending_queues:
            self._pending_queues[conv_id] = asyncio.Queue[BaseEvent]()
        await self._pending_queues[conv_id].put(event)
        get_logger().debug(
            f"[WS transport] Queued {event.event_type.value} for conversation {conv_id}"
        )

    async def cancel(self, conversation_id: str) -> None:
        """Send an AGENT_END cancellation signal to the remote agent via the relay.

        The same signal as ``SSETransport.cancel``, on a connection of its own as
        every boundary is. Nothing is sent for a conversation whose remote run this
        transport did not start, or that has already ended.
        """
        agent_name = self._running.pop(conversation_id, None)
        if agent_name is None:
            get_logger().debug(
                f"[WS transport] Skipping cancel — no remote run in progress (conv={conversation_id})"
            )
            return
        # A result queued for the next reconnect would restart what is being stopped.
        self._pending_queues.pop(conversation_id, None)
        get_logger().info(
            f"[WS transport] Sending cancellation signal to relay (conv={conversation_id})"
        )
        cancel_event = BaseEvent(
            event_type=ExecutionEventType.AGENT_END,
            author=agent_name,
            error_code="user_cancelled",
            error_message="Execution cancelled by user",
            conversation_id=conversation_id,
        )
        headers = {**self._headers, "conversation-id": conversation_id}
        try:
            async with ws_client.connect(
                self._ws_url, additional_headers=headers, open_timeout=min(self._timeout, 5.0)
            ) as ws:
                await ws.send(cancel_event.model_dump_json())
        except Exception as exc:
            get_logger().warning(
                f"[WS transport] Failed to send cancellation signal: {type(exc).__name__}"
            )
            get_logger().debug(f"[WS transport] Cancellation signal detail: {exc}")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _relay(
        self,
        ws: ws_client.ClientConnection,
        agent_name: str,
    ) -> AsyncGenerator[BaseEvent, None]:
        """Receive events from the WebSocket and yield them verbatim.

        Author normalisation happens in execute().
        """
        while True:
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=self._timeout)
            except TimeoutError:
                error_msg = (
                    f"The delegated agent '{agent_name}' stopped responding "
                    f"(no message received within {self._timeout}s). "
                    "The connection has been closed. Please check the agent's availability and connection."
                )
                get_logger().warning("[WS transport] Timeout waiting for next event — closing")
                yield BaseEvent(
                    event_type=ExecutionEventType.AGENT_END,
                    author=agent_name,
                    error_code="delegate_timeout",
                    error_message=error_msg,
                    content=[Part(text=error_msg)],
                )
                break
            except websockets.exceptions.ConnectionClosed as exc:
                get_logger().info(f"[WS transport] Connection closed: {type(exc).__name__}")
                get_logger().debug(f"[WS transport] Connection closed detail: {exc}")
                break

            try:
                event = BaseEvent.model_validate_json(raw)
            except Exception:
                get_logger().warning(f"[WS transport] Could not parse event (len={len(raw)})")
                get_logger().debug(f"[WS transport] Unparseable event: {raw!r}")
                continue

            get_logger().debug(f"[WS transport] Received {event.event_type.value}")
            yield event

            # Yield all events; the outer execute() loop decides when to break.
