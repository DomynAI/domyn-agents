"""SSE-based RunnerTransport for DelegateAgent.

Communicates with a relay over two HTTP primitives:

* **Receive** — ``POST {url}?channelId=…&spaceId=…&conversationId=…`` opens a
  Server-Sent Events stream.  The SSE connection stays open and streams all
  events as they arrive.  It is only closed and reopened when the runner calls
  ``send_result`` (i.e. a local tool was executed and the result must be
  forwarded to the relay).  Observational events (LLM_START,
  LLM_END, remote-side TOOL_START/TOOL_END) never trigger a close, so events
  already in the TCP receive buffer are never lost.

* **Send** — the next SSE open carries the result event as its POST body,
  forwarding it to the relay's WS clients in the same request that re-opens
  the subscription.

Wire it up::

    from domyn.agents.runner.sse_transport import SSETransport
    from domyn.agents.agents import DelegateAgent

    transport = SSETransport(
        url="https://relay.example.com/v1/messages",
        channel_id="abc123",
        space_id="space-uuid",
    )
    agent = DelegateAgent(name="Expert", description="…", transport=transport)

Or via config dict (as produced by the config loader)::

    transport:
      type: sse
      url: https://relay.example.com/v1/messages
      headers:
        channel-id: abc123
        space-id: space-uuid
      timeout: 60.0
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncGenerator
from typing import Any, ClassVar

import httpx
from httpx_sse import aconnect_sse

from domyn.agents.core import BaseEvent, ExecutionEventType, InteractionContext, Part, RelayMessage
from domyn.agents.logger import get_logger
from domyn.agents.runner.transport import RunnerTransport
from domyn.agents.tracing import inject_trace_headers


def _is_terminal_event(event: BaseEvent) -> bool:
    """Return True when the full delegate session is over."""
    if event.event_type == ExecutionEventType.AGENT_END:
        return True
    return event.event_type == ExecutionEventType.RESPONSE and not event.is_partial


class SSETransport(RunnerTransport):
    """SSE transport for ``DelegateAgent`` with close-on-send semantics.

    One SSE connection is kept open and streams all events from the relay.
    The connection is only closed (and immediately reopened) when
    ``send_result`` is called — that is, when the runner has executed a local
    tool and needs to forward the result to the remote agent.  Purely
    observational events (LLM_START/LLM_END, remote tool execution) never
    trigger a close, avoiding TCP-buffer event loss.

    Parameters
    ----------
    url:
        URL of the relay endpoint.
    channel_id:
        Relay channel identifier (``channelId`` query param).
    space_id:
        Relay space / tenant identifier (``spaceId`` query param).
    meta:
        Extra HTTP headers sent on every request.
    timeout:
        Connect / write timeout in seconds.  SSE read is unbounded.
    """

    type: ClassVar[str] = "sse"

    @classmethod
    def from_config(cls, config: dict) -> SSETransport:
        headers: dict[str, str] = config.get("headers") or {}
        return cls(
            url=config.get("url", ""),
            channel_id=headers.get("channel-id", ""),
            space_id=headers.get("space-id", ""),
            meta=config.get("meta") or (headers if headers else None),
            timeout=float(config.get("timeout", 60.0)),
        )

    def __init__(
        self,
        url: str,
        channel_id: str,
        space_id: str,
        meta: dict[str, str] | None = None,
        timeout: float = 60.0,
    ) -> None:
        self._url = url
        self._channel_id = channel_id
        self._space_id = space_id
        self._meta = meta or {}
        self._timeout = timeout
        # Per-session state, reset in execute().
        self._send_trigger: asyncio.Event | None = None
        self._pending_send: BaseEvent | None = None
        # The httpx client backing the currently-open receive SSE. Held so
        # pause_receiving() can close it mid-flight to release the subscription.
        self._active_client: httpx.AsyncClient | None = None
        # True between pause_receiving() and the reconnect triggered by
        # send_result(): the receive SSE is intentionally disconnected while a
        # local tool runs so the relay buffers events for replay on reconnect.
        self._paused: bool = False

    # ------------------------------------------------------------------
    # RunnerTransport interface
    # ------------------------------------------------------------------

    async def execute(
        self,
        agent_name: str,
        context: InteractionContext,
        run_config: dict[str, Any] | None = None,
    ) -> AsyncGenerator[BaseEvent, None]:
        """Stream events from the relay for *agent_name*.

        Opens an SSE connection with AGENT_START as the POST body. Each
        connection uses its own httpx client (held in ``self._active_client``) so
        it can be released mid-flight. The connection is closed and re-opened
        when either:

        * ``send_result`` is called by the runner (a local tool was executed) →
          reopen with the result as the next POST body; or
        * ``pause_receiving`` is called at the START of a local tool → close the
          receive SSE NOW so the relay buffers events while the (blocking) tool
          runs, then reopen on ``send_result`` so the relay replays them.

        A terminal event (AGENT_END or final RESPONSE) ends the session.

        Replays are filtered on two axes: ``_open_sse`` drops events from other turns
        (cross-turn, stateless — see ``RunnerTransport._is_current_turn``), and every
        event surviving that is deduped by ``event_id`` here (intra-turn, since a
        mid-turn reconnect replays events carrying this turn's id). ``seen`` and
        ``last_event_id`` are locals for the lifetime of this call; ``seen`` is seeded
        from ``interaction_history`` so a fresh-transport resume still dedups replays.
        """
        conversation_id: str = context.conversation_id or ""

        # Local dedup state (no instance attribute, no cleanup needed).
        seen: set[str] = self._seed_seen(context)
        last_event_id: str | None = None

        agent_start = self._build_outbound_event(agent_name, context)

        self._send_trigger = asyncio.Event()
        self._session_started = False
        self._agent_name = agent_name
        self._paused = False
        # Preserve a TOOL_END pre-set by send_result() before execute() was called
        # (HITL resume path). Fall back to AGENT_START for a normal fresh start.
        self._pending_send = self._pending_send or agent_start

        get_logger().info(
            f"[SSE transport] Opening session for agent '{agent_name}' (conv={conversation_id})"
        )

        sse_timeout = httpx.Timeout(
            connect=self._timeout,
            read=self._timeout,
            write=self._timeout,
            pool=self._timeout,
        )
        turn_id: str = context.turn_id or ""

        # Tracks depth of remote sub-agent calls so intermediate AGENT_END
        # events (sub-agent completions) are not mistaken for the root session end.
        # Persists across SSE reconnects (a reconnect mid-nesting resumes tracking).
        _remote_nesting = 0

        while True:
            self._send_trigger.clear()
            # Record our own outbound event so the relay replaying it back to us on
            # a later (re)connect is deduped rather than processed as an inbound event.
            if self._pending_send is not None and self._pending_send.event_id:
                seen.add(self._pending_send.event_id)
            # Fresh client per connection so pause_receiving() can close THIS
            # connection without breaking a subsequent reopen.
            client = httpx.AsyncClient(headers=self._meta, timeout=sse_timeout)
            self._active_client = client
            try:
                async with client:
                    async for event in self._open_sse(
                        client,
                        agent_name,
                        conversation_id,
                        turn_id=turn_id,
                        body=self._pending_send,
                        run_config=run_config,
                        last_event_id=last_event_id,
                    ):
                        self._pending_send = None

                        # Dedup by event_id BEFORE the nesting logic so replayed
                        # AGENT_CALL/AGENT_END never double-count nesting. Events from
                        # other turns were already dropped in _open_sse.
                        if not self._mark_seen(seen, event):
                            get_logger().debug(
                                f"[SSE transport] Dropping duplicate event {event.event_id}"
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
                        # A non-partial RESPONSE from inside a nested remote sub-agent
                        # would be misidentified as the root session's terminal event.
                        # Suppress it; the root's final RESPONSE arrives only after
                        # _remote_nesting returns to 0.
                        if (
                            etype == ExecutionEventType.RESPONSE
                            and not event.is_partial
                            and _remote_nesting > 0
                        ):
                            continue
                        # Events from inside a nested remote sub-agent (nesting > 0)
                        # keep their original author for observability.
                        if _remote_nesting == 0:
                            # Root-level event (nesting == 0): normalise author to agent_name.
                            event = event.model_copy(update={"author": agent_name})

                        yield event

                        if _is_terminal_event(event):
                            get_logger().info(
                                f"[SSE transport] Session complete for agent '{agent_name}'"
                            )
                            return

                        # Close and reopen only when send_result was called.
                        if self._send_trigger.is_set():
                            break
            finally:
                self._active_client = None

            if self._paused:
                # pause_receiving() closed the receive SSE so the relay buffers
                # events while the local tool runs. Wait for the tool result, then
                # reopen — the relay replays the buffered events on the new
                # subscription.
                get_logger().debug(
                    "[SSE transport] paused for local tool; awaiting result to reconnect"
                )
                await self._send_trigger.wait()
                self._paused = False

            # Reopen iff send_result fired this iteration (either mid-stream, or
            # after the pause wait above). The trigger is cleared at the top of
            # the next iteration, so is_set() means "there's a result to forward".
            if not self._send_trigger.is_set():
                # SSE stream ended cleanly without a terminal event and no result
                # to forward.
                get_logger().warning(
                    f"[SSE transport] SSE stream ended without terminal event for '{agent_name}'"
                )
                return

            # send_result was called — loop back with the result as body.
            get_logger().debug(
                f"[SSE transport] Reopening SSE to send "
                f"{self._pending_send.event_type.value if self._pending_send else '?'} "
                f"for agent '{agent_name}'"
            )

    async def send_result(self, event: BaseEvent, conversation_id: str | None = None) -> None:
        """Signal that a local tool result is ready to be forwarded.

        Sets ``_pending_send`` and fires the trigger so ``execute`` closes
        the current SSE and reopens with the result as the POST body.
        """
        get_logger().debug(f"[SSE transport] send_result {event.event_type.value}")
        self._pending_send = event
        if self._send_trigger is not None:
            self._send_trigger.set()

    async def pause_receiving(self) -> None:
        """Release the receive SSE so the relay buffers events while we're not reading.

        Called when the runner is about to block executing a local tool (e.g. a
        slow search) and will stop consuming this transport's events for the
        tool's duration. If the SSE stayed
        connected-but-unread, the relay would write events (like the remote
        agent's wrap-up ``RESPONSE``) into a socket we never read — counted as
        delivered, so NOT buffered for replay — and they would be discarded when
        we later close to forward the result. By closing NOW, the relay sees zero
        subscribers for the channel and BUFFERS subsequent events;
        :meth:`send_result` then reopens and the relay replays the buffered events
        on the new subscription. ``execute`` treats the
        resulting connection close as a pause (see ``self._paused``) rather than a
        stream end.
        """
        self._paused = True
        client = self._active_client
        if client is not None:
            get_logger().debug(
                "[SSE transport] pause_receiving: closing receive SSE so the relay buffers events"
            )
            with contextlib.suppress(Exception):
                await client.aclose()

    async def cancel(self, conversation_id: str) -> None:
        """Send an AGENT_END cancellation signal to the remote agent via the relay."""
        if not getattr(self, "_session_started", False):
            get_logger().debug(
                f"[SSE transport] Skipping cancel — session never started (conv={conversation_id})"
            )
            return
        get_logger().info(
            f"[SSE transport] Sending cancellation signal to relay (conv={conversation_id})"
        )
        cancel_event = BaseEvent(
            event_type=ExecutionEventType.AGENT_END,
            author=getattr(self, "_agent_name", "user"),
            error_code="user_cancelled",
            error_message="Execution cancelled by user",
            conversation_id=conversation_id,
        )
        relay_msg = RelayMessage(payload=cancel_event)
        params: dict[str, str] = {
            "channelId": self._channel_id,
            "spaceId": self._space_id,
            "conversationId": conversation_id,
        }
        try:
            async with (
                httpx.AsyncClient(
                    headers=self._meta,
                    timeout=httpx.Timeout(connect=5.0, write=5.0, read=None, pool=5.0),
                ) as client,
                client.stream(
                    "POST",
                    self._url,
                    params=params,
                    content=relay_msg.model_dump_json(),
                    headers={"Content-Type": "application/json"},
                ) as response,
            ):
                if response.status_code >= 400:
                    get_logger().warning(
                        f"[SSE transport] Cancellation signal rejected by relay: HTTP {response.status_code}"
                    )
        except Exception as exc:
            get_logger().warning(
                f"[SSE transport] Failed to send cancellation signal: {type(exc).__name__}"
            )
            get_logger().debug(f"[SSE transport] Cancellation signal detail: {exc}", exc_info=True)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _open_sse(
        self,
        client: httpx.AsyncClient,
        agent_name: str,
        conversation_id: str,
        turn_id: str = "",
        body: BaseEvent | None = None,
        run_config: dict[str, Any] | None = None,
        last_event_id: str | None = None,
    ) -> AsyncGenerator[BaseEvent, None]:
        """Open one SSE connection, optionally forwarding *body* to the relay.

        The POST body is wrapped in a ``RelayMessage`` whose ``run_config`` field
        carries the visibility-filtered ``run_config`` (kept out of the ``BaseEvent``
        payload).
        ``last_event_id`` is the caller's resume cursor for the ``Last-Event-ID``
        header; the caller owns dedup, so this connection yields every parsed event.
        """
        params: dict[str, str] = {
            "channelId": self._channel_id,
            "spaceId": self._space_id,
            "conversationId": conversation_id,
        }

        get_logger().debug(
            f"[SSE transport] Connecting to {self._url} "
            f"(body={body.event_type.value if body else None}, conv={conversation_id})"
        )

        headers: dict[str, str] = {}
        # Resume cursor: the relay replays only events after this id, trimming the
        # replay. Absent (the first connect of every turn, since the transport is
        # rebuilt per turn) the relay replays its whole retained window for this
        # conversation — correctness then rests entirely on the turn filter + dedup
        # below, not on the cursor.
        if last_event_id:
            headers["Last-Event-ID"] = last_event_id

        kwargs: dict[str, Any] = {"params": params}
        if body is not None:
            kwargs["content"] = self._wrap_outbound(body, run_config)
            headers["Content-Type"] = "application/json"
            headers = inject_trace_headers(headers)
        if headers:
            kwargs["headers"] = headers

        try:
            async with aconnect_sse(
                client,
                "POST",
                self._url,
                **kwargs,
            ) as event_source:
                self._session_started = True  # POST accepted by relay
                async for sse_event in event_source.aiter_sse():
                    if not sse_event.data or sse_event.data.strip() == "":
                        continue  # heartbeat / keep-alive

                    try:
                        body = RelayMessage.model_validate_json(sse_event.data)
                        event = body.payload
                        _ = body.meta
                    except Exception:
                        get_logger().warning(
                            f"[SSE transport] Could not parse SSE event (len={len(sse_event.data)})"
                        )
                        get_logger().debug(
                            f"[SSE transport] Unparseable SSE event: {sse_event.data!r}"
                        )
                        continue

                    if (
                        conversation_id
                        and event.conversation_id
                        and event.conversation_id != conversation_id
                    ):
                        get_logger().debug(
                            f"[SSE transport] Skipping event for conversation "
                            f"{event.conversation_id} (expected {conversation_id})"
                        )
                        continue

                    # Drop every event belonging to another turn, not just terminal ones.
                    # The relay's retained log is keyed by channel — deterministic per
                    # (conversation, agent), so shared by every turn of the conversation —
                    # and a connect without a resume cursor replays the whole window. Since
                    # the transport is rebuilt per turn it has no cursor to send on its
                    # first connect, so a new turn's stream routinely opens with the
                    # previous turn's events. Terminal ones used to be the only visible
                    # symptom (a stale AGENT_END ended the new session early); the rest
                    # surfaced as duplicated events on the next user message.
                    #
                    # This also drops a background delegation that completed after its own
                    # turn had closed: it belongs to that turn, not to this one.
                    if not self._is_current_turn(event, turn_id):
                        get_logger().debug(
                            f"[SSE transport] Skipping {event.event_type.value} from turn "
                            f"{event.turn_id} (expected {turn_id})"
                        )
                        continue

                    get_logger().debug(f"[SSE transport] Received {event.event_type.value}")
                    yield event

        except httpx.TimeoutException:
            error_msg = (
                f"The delegated agent '{agent_name}' did not respond within the timeout period. "
                "The connection has been closed. Please check the agent's availability and connection."
            )
            get_logger().warning(
                f"[SSE transport] Timeout waiting for SSE event — closing connection for '{agent_name}'"
            )
            yield BaseEvent(
                event_type=ExecutionEventType.AGENT_END,
                author=agent_name,
                error_code="delegate_timeout",
                error_message=error_msg,
                content=[Part(text=error_msg)],
            )
        except httpx.HTTPStatusError as exc:
            get_logger().error(
                f"[SSE transport] HTTP error on SSE stream: {exc.response.status_code}"
            )
            get_logger().debug(
                f"[SSE transport] HTTP error body on SSE stream: {exc.response.text}"
            )
            raise
        except Exception as exc:
            if self._paused:
                # pause_receiving() closed this connection on purpose so the
                # relay buffers events during the local tool. End cleanly; execute()
                # reopens after the result and the relay replays the buffer.
                get_logger().debug(
                    f"[SSE transport] receive SSE closed for local-tool pause: {exc!r}"
                )
                return
            get_logger().error(
                f"[SSE transport] Unexpected error on SSE stream: {type(exc).__name__}"
            )
            get_logger().debug(f"[SSE transport] Unexpected error on SSE stream: {exc}")
            raise
