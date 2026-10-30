"""Tests for DomynRelayChannel.

Covered:
1. receive() yields AGENT_START events from the mock WebSocket.
2. send_event() enqueues payloads (best-effort).
3. send_event_priority() enqueues payloads (blocking).
4. Frames that are not routable are ignored.
5. Malformed frames are silently skipped.
6. close() stops the receive loop.
7. from_env() takes the relay URL and headers from the platform seam.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from typing import Any

import pytest

from domyn.agents.channels import DomynRelayChannel, InMemoryCursorStore
from domyn.agents.core import BaseEvent, ExecutionEventType, Part, RelayMessage

# ---------------------------------------------------------------------------
# Mock WebSocket
# ---------------------------------------------------------------------------


class _MockWS:
    """Minimal mock websocket that delivers a preset sequence of frames."""

    def __init__(self, frames: list[str | bytes], *, close_code: int = 1000) -> None:
        self._frames = list(frames)
        self._idx = 0
        self.close_code = close_code
        self.sent: list[str] = []

    def __aiter__(self) -> _MockWS:
        return self

    async def __anext__(self) -> str:
        if self._idx >= len(self._frames):
            raise StopAsyncIteration
        frame = self._frames[self._idx]
        self._idx += 1
        return frame  # type: ignore[return-value]

    async def send(self, payload: str) -> None:
        self.sent.append(payload)


def _relay_frame(event: BaseEvent) -> str:
    return RelayMessage(payload=event).model_dump_json()


def _agent_start_event(conversation_id: str = "conv-1") -> BaseEvent:
    return BaseEvent(
        event_type=ExecutionEventType.AGENT_START,
        author="platform",
        event_id=str(uuid.uuid4()),
        conversation_id=conversation_id,
    )


# ---------------------------------------------------------------------------
# Helpers: monkey-patch websockets.connect
# ---------------------------------------------------------------------------


class _FakeConnect:
    """Context manager that returns a _MockWS."""

    def __init__(self, ws: _MockWS) -> None:
        self._ws = ws

    async def __aenter__(self) -> _MockWS:
        return self._ws

    async def __aexit__(self, *_: Any) -> None:
        pass


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_from_env_connects_where_the_platform_says(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[Any] = []

    def relay_connection(settings: Any) -> tuple[str, dict[str, str]]:
        seen.append(settings)
        return "wss://relay.example.com/ws", {"channel-id": "c1", "api-key": "k"}

    monkeypatch.setattr(
        "domyn.agents.integrations.domyn_platform_sdk.relay_connection", relay_connection
    )

    channel = DomynRelayChannel.from_env(
        api_key="k",
        channel_id="c1",
        space_id="s1",
        base_url="relay.example.com",
        configuration_id="cfg",
    )

    assert channel._ws_url == "wss://relay.example.com/ws"
    assert channel._headers == {"channel-id": "c1", "api-key": "k"}
    assert (seen[0].channel_id, seen[0].space_id, seen[0].configuration_id) == ("c1", "s1", "cfg")
    assert seen[0].base_url == "relay.example.com"


@pytest.mark.asyncio
async def test_receive_yields_agent_start(monkeypatch: pytest.MonkeyPatch) -> None:
    """receive() must yield (conv_id, event) for AGENT_START frames."""
    event = _agent_start_event("my-conv")
    ws = _MockWS([_relay_frame(event)])

    channel = DomynRelayChannel("wss://relay.example.com/relay/v1/ws", {})

    import websockets

    import domyn.agents.channels.domyn_relay as mod

    monkeypatch.setattr(websockets, "connect", lambda *a, **kw: _FakeConnect(ws))

    results: list[tuple[str, BaseEvent]] = []
    async for conv_id, ev in channel.receive():
        results.append((conv_id, ev))
        await channel.close()  # stop after first event

    assert len(results) == 1
    conv_id, ev = results[0]
    assert conv_id == "my-conv"
    assert ev.event_type == ExecutionEventType.AGENT_START


@pytest.mark.asyncio
async def test_receive_yields_tool_feedback(monkeypatch: pytest.MonkeyPatch) -> None:
    """receive() must yield TOOL_FEEDBACK frames — the human's approval reply.

    Regression: TOOL_FEEDBACK was missing from _ROUTABLE, so an approval sent
    back from the platform was silently dropped and a gated tool's turn could
    never be resumed over this channel.
    """
    import websockets

    feedback_event = BaseEvent(
        event_type=ExecutionEventType.TOOL_FEEDBACK,
        author="user",
        conversation_id="conv-fb",
    )
    ws = _MockWS([_relay_frame(feedback_event)])

    channel = DomynRelayChannel("wss://x/relay/v1/ws", {})
    monkeypatch.setattr(websockets, "connect", lambda *a, **kw: _FakeConnect(ws))

    results: list[tuple[str, BaseEvent]] = []
    async for conv_id, ev in channel.receive():
        results.append((conv_id, ev))
        await channel.close()

    assert len(results) == 1
    conv_id, ev = results[0]
    assert conv_id == "conv-fb"
    assert ev.event_type == ExecutionEventType.TOOL_FEEDBACK


@pytest.mark.asyncio
async def test_receive_ignores_non_routable(monkeypatch: pytest.MonkeyPatch) -> None:
    """RESPONSE / LLM_START events must not be yielded by receive()."""
    import websockets

    response_event = BaseEvent(
        event_type=ExecutionEventType.RESPONSE,
        author="agent",
        conversation_id="conv-x",
        content=[Part(text="hello")],
    )
    agent_start = _agent_start_event("conv-x")
    ws = _MockWS([_relay_frame(response_event), _relay_frame(agent_start)])

    channel = DomynRelayChannel("wss://x/relay/v1/ws", {})
    monkeypatch.setattr(websockets, "connect", lambda *a, **kw: _FakeConnect(ws))

    results: list[tuple[str, BaseEvent]] = []
    async for conv_id, ev in channel.receive():
        results.append((conv_id, ev))
        await channel.close()

    # Only AGENT_START should be yielded
    assert all(ev.event_type == ExecutionEventType.AGENT_START for _, ev in results)


@pytest.mark.asyncio
async def test_receive_skips_malformed_frame(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unparseable frames must be silently skipped."""
    import websockets

    good = _relay_frame(_agent_start_event("conv-ok"))
    ws = _MockWS(["{not json at all}", good])

    channel = DomynRelayChannel("wss://x/relay/v1/ws", {})
    monkeypatch.setattr(websockets, "connect", lambda *a, **kw: _FakeConnect(ws))

    results: list[tuple[str, BaseEvent]] = []
    async for conv_id, ev in channel.receive():
        results.append((conv_id, ev))
        await channel.close()

    assert len(results) == 1
    assert results[0][0] == "conv-ok"


@pytest.mark.asyncio
async def test_send_event_best_effort() -> None:
    """send_event() must enqueue without blocking."""
    channel = DomynRelayChannel("wss://x/relay/v1/ws", {}, queue_maxsize=10)
    event = _agent_start_event()
    await channel.send_event("conv-1", event)
    assert channel._outbound.qsize() == 1


@pytest.mark.asyncio
async def test_send_event_drops_when_queue_full() -> None:
    """send_event() increments events_dropped instead of blocking when queue is full."""
    channel = DomynRelayChannel("wss://x/relay/v1/ws", {}, queue_maxsize=1)
    event = _agent_start_event()
    # Fill the queue
    await channel.send_event("c", event)
    assert channel.events_dropped == 0
    # This one should be dropped
    await channel.send_event("c", event)
    assert channel.events_dropped == 1


@pytest.mark.asyncio
async def test_send_event_priority_enqueues() -> None:
    """send_event_priority() must enqueue the payload."""
    channel = DomynRelayChannel("wss://x/relay/v1/ws", {})
    event = _agent_start_event()
    await channel.send_event_priority("conv-1", event)
    assert channel._outbound.qsize() == 1


# ---------------------------------------------------------------------------
# Resume cursor (last-event-id) — trims what the relay replays on reconnect
# ---------------------------------------------------------------------------


class _RecordingConnect:
    """Monkeypatch target for ``websockets.connect``.

    Records the ``additional_headers`` passed on every call and hands back a
    preset ``_MockWS`` per connection attempt, so a test can assert on the
    resume cursor sent on the SECOND (reconnect) call.
    """

    def __init__(self, ws_sequence: list[_MockWS]) -> None:
        self._ws_sequence = list(ws_sequence)
        self.headers_per_call: list[dict[str, str]] = []

    def __call__(self, *args: Any, **kwargs: Any) -> _FakeConnect:
        self.headers_per_call.append(dict(kwargs.get("additional_headers") or {}))
        ws = self._ws_sequence.pop(0)
        return _FakeConnect(ws)


@pytest.mark.asyncio
async def test_receive_sends_last_event_id_on_reconnect(monkeypatch: pytest.MonkeyPatch) -> None:
    """The SECOND (reconnect) connect() call must carry `last-event-id` set to the
    most recently seen event's event_id, so the relay replays only the forward
    delta instead of its whole retained window.
    """
    import websockets

    ev1 = _agent_start_event("conv-x")
    ev2 = _agent_start_event("conv-x")
    ws1 = _MockWS([_relay_frame(ev1)])  # one frame, then EOF -> triggers reconnect
    ws2 = _MockWS([_relay_frame(ev2)])

    connect = _RecordingConnect([ws1, ws2])
    monkeypatch.setattr(websockets, "connect", connect)

    channel = DomynRelayChannel("wss://x/relay/v1/ws", {"api-key": "k"})

    results: list[tuple[str, BaseEvent]] = []
    async for conv_id, ev in channel.receive():
        results.append((conv_id, ev))
        if len(connect.headers_per_call) >= 2:
            await channel.close()

    assert len(connect.headers_per_call) == 2
    # First connect: no prior cursor.
    assert "last-event-id" not in connect.headers_per_call[0]
    # Original headers must still be present alongside the cursor.
    assert connect.headers_per_call[1]["api-key"] == "k"
    # Reconnect: cursor advanced to the last event seen on the first connection.
    assert connect.headers_per_call[1]["last-event-id"] == ev1.event_id
    assert [ev.event_id for _, ev in results] == [ev1.event_id, ev2.event_id]


@pytest.mark.asyncio
async def test_receive_advances_cursor_on_non_routable_events(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The resume cursor must advance even for non-routable frames (e.g. RESPONSE),
    not just AGENT_START/TOOL_END/TOOL_ERROR/AGENT_END, so a reconnect never
    re-fetches anything already observed regardless of type."""
    import websockets

    non_routable = BaseEvent(
        event_type=ExecutionEventType.RESPONSE,
        author="agent",
        conversation_id="conv-y",
        content=[Part(text="hello")],
    )
    ev2 = _agent_start_event("conv-y")
    ws1 = _MockWS([_relay_frame(non_routable)])  # ignored, but must still advance the cursor
    ws2 = _MockWS([_relay_frame(ev2)])

    connect = _RecordingConnect([ws1, ws2])
    monkeypatch.setattr(websockets, "connect", connect)

    channel = DomynRelayChannel("wss://x/relay/v1/ws", {})

    async for _conv_id, _ev in channel.receive():
        await channel.close()  # stop as soon as anything is yielded (ev2)

    assert connect.headers_per_call[1]["last-event-id"] == non_routable.event_id


@pytest.mark.asyncio
async def test_receive_does_not_reprocess_stale_replay_after_cancel_race(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression for the exact bug this fix targets: a stale replay of an
    already-completed AGENT_END must not be misrouted as a fresh inbound event
    (AgentServer routes inbound AGENT_END to adapter.on_cancel(), which would
    cancel a newer in-flight invocation for the same conversation_id if this
    frame were allowed through a second time)."""
    import websockets

    agent_end = BaseEvent(
        event_type=ExecutionEventType.AGENT_END,
        author="agent",
        conversation_id="conv-z",
    )
    agent_start_2 = _agent_start_event("conv-z")
    # Same AGENT_END delivered twice within one connection (e.g. the relay's own
    # replay-on-reconnect landing mid-stream) must be yielded only once.
    ws = _MockWS([_relay_frame(agent_end), _relay_frame(agent_end), _relay_frame(agent_start_2)])

    connect = _RecordingConnect([ws])
    monkeypatch.setattr(websockets, "connect", connect)

    channel = DomynRelayChannel("wss://x/relay/v1/ws", {})

    results: list[tuple[str, BaseEvent]] = []
    async for conv_id, ev in channel.receive():
        results.append((conv_id, ev))
        await channel.close()

    assert [ev.event_type for _, ev in results] == [
        ExecutionEventType.AGENT_END,
        ExecutionEventType.AGENT_START,
    ]


@pytest.mark.asyncio
async def test_default_cursor_store_is_in_memory() -> None:
    """With no cursor_store passed, DomynRelayChannel must default to an
    InMemoryCursorStore (not None, not a FileCursorStore) — zero-config,
    dependency-free, matching the relay's own short (minutes-long) retained
    window where disk persistence across restarts buys little."""
    channel = DomynRelayChannel("wss://x/relay/v1/ws", {})
    assert isinstance(channel._cursor_store, InMemoryCursorStore)


@pytest.mark.asyncio
async def test_send_event_advances_cursor() -> None:
    """Outbound sends must advance the resume cursor too (not just inbound
    events), since the relay's per-channel offset is one shared sequence
    across both directions — otherwise a long outbound-only burst (e.g. many
    TOOL_START/TOOL_UPDATE in one turn) would leave the cursor stale."""
    channel = DomynRelayChannel("wss://x/relay/v1/ws", {})
    event = _agent_start_event("conv-1")
    await channel.send_event("conv-1", event)
    assert await channel._cursor_store.load(channel._cursor_key) == event.event_id


@pytest.mark.asyncio
async def test_send_event_priority_advances_cursor() -> None:
    channel = DomynRelayChannel("wss://x/relay/v1/ws", {})
    event = _agent_start_event("conv-1")
    await channel.send_event_priority("conv-1", event)
    assert await channel._cursor_store.load(channel._cursor_key) == event.event_id


class _FakeCursorStore:
    """Minimal in-memory CursorStore double for testing DomynRelayChannel's wiring."""

    def __init__(self, initial: dict[str, str] | None = None) -> None:
        self.data: dict[str, str] = dict(initial or {})
        self.saved: list[tuple[str, str]] = []

    async def load(self, key: str) -> str | None:
        return self.data.get(key)

    async def save(self, key: str, event_id: str) -> None:
        self.data[key] = event_id
        self.saved.append((key, event_id))


@pytest.mark.asyncio
async def test_send_event_persists_cursor_to_store() -> None:
    store = _FakeCursorStore()
    channel = DomynRelayChannel("wss://x/relay/v1/ws", {}, cursor_store=store, cursor_key="chan-1")
    event = _agent_start_event("conv-1")
    await channel.send_event_priority("conv-1", event)
    assert store.data["chan-1"] == event.event_id


@pytest.mark.asyncio
async def test_receive_loads_persisted_cursor_and_sends_it_on_first_connect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A cursor persisted by a PREVIOUS process (loaded via cursor_store) must be
    sent as `last-event-id` on the very FIRST connect of a fresh channel
    instance — this is what lets a restarted worker resume instead of
    re-triggering a full-window replay."""
    import websockets

    store = _FakeCursorStore({"chan-1": "evt-from-previous-process"})
    ws = _MockWS([_relay_frame(_agent_start_event("conv-1"))])
    connect = _RecordingConnect([ws])
    monkeypatch.setattr(websockets, "connect", connect)

    channel = DomynRelayChannel(
        "wss://x/relay/v1/ws", {"api-key": "k"}, cursor_store=store, cursor_key="chan-1"
    )

    async for _conv_id, _ev in channel.receive():
        await channel.close()

    assert connect.headers_per_call[0]["last-event-id"] == "evt-from-previous-process"


@pytest.mark.asyncio
async def test_receive_persists_cursor_as_events_arrive(monkeypatch: pytest.MonkeyPatch) -> None:
    import websockets

    store = _FakeCursorStore()
    ev = _agent_start_event("conv-1")
    ws = _MockWS([_relay_frame(ev)])
    connect = _RecordingConnect([ws])
    monkeypatch.setattr(websockets, "connect", connect)

    channel = DomynRelayChannel("wss://x/relay/v1/ws", {}, cursor_store=store, cursor_key="chan-1")

    async for _conv_id, _ev in channel.receive():
        await channel.close()

    assert store.data["chan-1"] == ev.event_id
