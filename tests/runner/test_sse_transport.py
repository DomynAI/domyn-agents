"""Tests for SSETransport."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from unittest.mock import MagicMock, patch

import pytest

from domyn.agents.core import (
    BaseEvent,
    ExecutionEventType,
    InteractionContext,
    Part,
    RelayMessage,
    ToolAction,
)
from domyn.agents.runner.sse_transport import (
    SSETransport,
    _is_terminal_event,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_event(
    event_type: ExecutionEventType,
    *,
    is_partial: bool = False,
    author: str = "ext-agent",
    content: list[Part] | None = None,
) -> BaseEvent:
    return BaseEvent(
        event_type=event_type,
        author=author,
        content=content or [Part(text="hello")],
        is_partial=is_partial,
    )


def make_context(conversation_id: str = "conv-123") -> InteractionContext:
    return InteractionContext(
        conversation_id=conversation_id,
        user_id="user-1",
        agent_input=BaseEvent(
            event_type=ExecutionEventType.USER_INPUT,
            author="user",
            content=[Part(text="do the thing")],
        ),
    )


def make_transport(**kwargs) -> SSETransport:
    return SSETransport(
        url="https://relay.example.com/v1/messages",
        channel_id="ch-abc",
        space_id="sp-xyz",
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Unit: helper predicates
# ---------------------------------------------------------------------------


class TestPredicates:
    def test_agent_end_is_terminal(self):
        ev = make_event(ExecutionEventType.AGENT_END)
        assert _is_terminal_event(ev) is True

    def test_final_response_is_terminal(self):
        ev = make_event(ExecutionEventType.RESPONSE, is_partial=False)
        assert _is_terminal_event(ev) is True

    def test_partial_response_is_not_terminal(self):
        ev = make_event(ExecutionEventType.RESPONSE, is_partial=True)
        assert _is_terminal_event(ev) is False

    def test_tool_start_is_not_terminal(self):
        ev = make_event(ExecutionEventType.TOOL_START)
        assert _is_terminal_event(ev) is False


# ---------------------------------------------------------------------------
# Unit: build from transport registry
# ---------------------------------------------------------------------------


class TestTransportRegistry:
    def test_build_sse_from_config(self):
        from domyn.agents.runner.transport import build_transport

        transport = build_transport(
            {
                "type": "sse",
                "url": "https://relay.example.com/v1/messages",
                "headers": {"channel-id": "ch-1", "space-id": "sp-1"},
                "timeout": 30.0,
            }
        )
        assert isinstance(transport, SSETransport)
        assert transport._url == "https://relay.example.com/v1/messages"
        assert transport._channel_id == "ch-1"
        assert transport._space_id == "sp-1"
        assert transport._timeout == 30.0

    def test_subclass_self_registers_and_builds_from_config(self):
        """A RunnerTransport subclass declared in external code self-registers
        via __init_subclass__ and is immediately resolvable by build_transport
        without any manual register_transport() call."""
        from domyn.agents.runner.transport import RunnerTransport, build_transport

        # Simulate what a service would define when importing domyn.agents.
        class CustomSSETransport(SSETransport):
            type = "custom_sse"

            extra_header: str = ""

            @classmethod
            def from_config(cls, config: dict) -> CustomSSETransport:  # type: ignore[override]
                inst = cls(
                    url=config.get("url", ""),
                    channel_id=config.get("channel_id", ""),
                    space_id=config.get("space_id", ""),
                )
                inst.extra_header = config.get("extra_header", "")
                return inst

        # The subclass must be in the registry as soon as it's defined.
        assert "custom_sse" in RunnerTransport._registry
        assert RunnerTransport._registry["custom_sse"] is CustomSSETransport

        # build_transport must resolve it from a plain dict (as the YAML
        # loader would produce).
        transport = build_transport(
            {
                "type": "custom_sse",
                "url": "https://relay.example.com/v1/messages",
                "channel_id": "ch-custom",
                "space_id": "sp-custom",
                "extra_header": "Bearer tok",
            }
        )

        assert isinstance(transport, CustomSSETransport)
        assert transport._url == "https://relay.example.com/v1/messages"
        assert transport.extra_header == "Bearer tok"

        # Clean up so this test does not pollute other tests.
        RunnerTransport._registry.pop("custom_sse", None)


# ---------------------------------------------------------------------------
# Integration: execute() flow with mocked HTTP
# ---------------------------------------------------------------------------


class _FakeSSEStream:
    """Async context manager that yields faked SSE events."""

    def __init__(self, events: list[BaseEvent]) -> None:
        self._events = events

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    async def aiter_sse(self) -> AsyncGenerator:
        for ev in self._events:
            m = MagicMock()
            m.data = RelayMessage(payload=ev).model_dump_json()
            yield m


@pytest.mark.asyncio
async def test_execute_single_final_response():
    """A single final RESPONSE ends the generator after one SSE open."""
    transport = make_transport()
    context = make_context()

    final = make_event(ExecutionEventType.RESPONSE, is_partial=False)
    captured_kwargs: list[dict] = []

    def fake_connect(client, method, url, **kwargs):
        captured_kwargs.append(kwargs)
        return _FakeSSEStream([final])

    with patch("domyn.agents.runner.sse_transport.aconnect_sse", side_effect=fake_connect):
        events = [ev async for ev in transport.execute("Agent", context)]

    assert len(events) == 1
    assert events[0].event_type == ExecutionEventType.RESPONSE
    assert not events[0].is_partial
    assert len(captured_kwargs) == 1
    assert captured_kwargs[0]["params"]["channelId"] == "ch-abc"
    assert captured_kwargs[0]["params"]["spaceId"] == "sp-xyz"
    posted = BaseEvent.model_validate_json(captured_kwargs[0]["content"])
    assert posted.event_type == ExecutionEventType.AGENT_START


@pytest.mark.asyncio
async def test_execute_agent_end_terminates():
    """AGENT_END terminates the generator."""
    transport = make_transport()
    context = make_context()

    agent_end = make_event(ExecutionEventType.AGENT_END)

    with patch(
        "domyn.agents.runner.sse_transport.aconnect_sse",
        return_value=_FakeSSEStream([agent_end]),
    ):
        events = [ev async for ev in transport.execute("Agent", context)]

    assert len(events) == 1
    assert events[0].event_type == ExecutionEventType.AGENT_END


@pytest.mark.asyncio
async def test_execute_partial_responses_then_final():
    """Partial RESPONSE events are yielded without closing; final closes."""
    transport = make_transport()
    context = make_context()

    partial1 = make_event(ExecutionEventType.RESPONSE, is_partial=True)
    partial2 = make_event(ExecutionEventType.RESPONSE, is_partial=True)
    final = make_event(ExecutionEventType.RESPONSE, is_partial=False)

    call_count = 0

    def fake_connect(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return _FakeSSEStream([partial1, partial2, final])

    with patch("domyn.agents.runner.sse_transport.aconnect_sse", side_effect=fake_connect):
        events = [ev async for ev in transport.execute("Agent", context)]

    assert [e.is_partial for e in events] == [True, True, False]
    # All from a single SSE connection — only one aconnect_sse call.
    assert call_count == 1


@pytest.mark.asyncio
async def test_execute_tool_start_then_final_response():
    """TOOL_START is a non-terminal boundary; second SSE open delivers RESPONSE."""
    transport = make_transport()
    context = make_context()

    tool_start = BaseEvent(
        event_type=ExecutionEventType.TOOL_START,
        author="ext",
        action=ToolAction(name="my_tool", parameters={}),
    )
    final = make_event(ExecutionEventType.RESPONSE, is_partial=False)

    sse_calls: list[list[BaseEvent]] = [[tool_start], [final]]
    call_idx = 0
    captured_kwargs: list[dict] = []

    def fake_connect(client, method, url, **kwargs):
        nonlocal call_idx
        captured_kwargs.append(kwargs)
        stream = _FakeSSEStream(sse_calls[call_idx])
        call_idx += 1
        return stream

    with patch("domyn.agents.runner.sse_transport.aconnect_sse", side_effect=fake_connect):
        gen = transport.execute("Agent", context)

        # First event: TOOL_START
        ev1 = await gen.__anext__()
        assert ev1.event_type == ExecutionEventType.TOOL_START

        # Simulate runner sending TOOL_END back
        tool_end = make_event(ExecutionEventType.TOOL_END)
        await transport.send_result(tool_end)

        # Second event: RESPONSE (from second SSE connection)
        ev2 = await gen.__anext__()
        assert ev2.event_type == ExecutionEventType.RESPONSE

        with pytest.raises(StopAsyncIteration):
            await gen.__anext__()

    # Two SSE connections opened
    assert call_idx == 2
    first_body = BaseEvent.model_validate_json(captured_kwargs[0]["content"])
    second_body = BaseEvent.model_validate_json(captured_kwargs[1]["content"])
    assert first_body.event_type == ExecutionEventType.AGENT_START
    assert second_body.event_type == ExecutionEventType.TOOL_END


@pytest.mark.asyncio
async def test_pause_receiving_releases_connection_and_reopens_on_result():
    """pause_receiving() closes the receive SSE (so the relay buffers events
    while the local tool runs); execute() stays disconnected until send_result,
    then reopens with the result — delivering events the relay replays on the
    new connection (modeled here as the reopen stream)."""
    transport = make_transport()
    context = make_context()

    tool_start = BaseEvent(
        event_type=ExecutionEventType.TOOL_START,
        author="ext",
        action=ToolAction(name="web_search", parameters={}, call_id="c1"),
    )
    # What the relay buffered during the tool and replays on reopen:
    buffered = make_event(
        ExecutionEventType.RESPONSE, is_partial=True, content=[Part(text="waiting for subagents")]
    )
    final = make_event(ExecutionEventType.RESPONSE, is_partial=False, content=[Part(text="done")])

    opened_bodies: list[BaseEvent | None] = []

    def fake_connect(client, method, url, **kwargs):
        body = BaseEvent.model_validate_json(kwargs["content"]) if kwargs.get("content") else None
        opened_bodies.append(body)
        if body is not None and body.event_type == ExecutionEventType.AGENT_START:
            return _FakeSSEStream([tool_start])
        # Reopen carrying the TOOL_END result → relay replays buffered wrap-up, then final.
        return _FakeSSEStream([buffered, final])

    with patch("domyn.agents.runner.sse_transport.aconnect_sse", side_effect=fake_connect):
        gen = transport.execute("Agent", context)

        ev1 = await gen.__anext__()
        assert ev1.event_type == ExecutionEventType.TOOL_START

        # Runner begins the local tool → transport releases the receive connection.
        active_client = transport._active_client
        await transport.pause_receiving()
        assert transport._paused is True
        assert active_client is not None and active_client.is_closed

        # Runner forwards the tool result → transport reopens.
        tool_end = make_event(ExecutionEventType.TOOL_END)
        await transport.send_result(tool_end)

        rest = [ev async for ev in gen]

    types = [(e.event_type, e.is_partial) for e in rest]
    # The buffered wrap-up (partial) and the final response both arrive after reopen.
    assert (ExecutionEventType.RESPONSE, True) in types
    assert (ExecutionEventType.RESPONSE, False) in types
    # First open was AGENT_START; the reopen carried the TOOL_END result as its body.
    assert opened_bodies[0].event_type == ExecutionEventType.AGENT_START
    assert opened_bodies[1].event_type == ExecutionEventType.TOOL_END


@pytest.mark.asyncio
async def test_execute_skips_empty_sse_lines():
    """Heartbeat / keep-alive empty data lines are ignored."""
    transport = make_transport()
    context = make_context()

    final = make_event(ExecutionEventType.RESPONSE, is_partial=False)

    class StreamWithHeartbeats:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def aiter_sse(self):
            # heartbeat
            m = MagicMock()
            m.data = ""
            yield m
            # real event
            m2 = MagicMock()
            m2.data = RelayMessage(payload=final).model_dump_json()
            yield m2

    with patch(
        "domyn.agents.runner.sse_transport.aconnect_sse",
        return_value=StreamWithHeartbeats(),
    ):
        events = [ev async for ev in transport.execute("Agent", context)]

    assert len(events) == 1


@pytest.mark.asyncio
async def test_execute_ignores_malformed_sse_data():
    """Unparseable SSE lines are skipped, not raised."""
    transport = make_transport()
    context = make_context()

    final = make_event(ExecutionEventType.RESPONSE, is_partial=False)

    class StreamWithGarbage:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def aiter_sse(self):
            m = MagicMock()
            m.data = "not json {{{"
            yield m
            m2 = MagicMock()
            m2.data = RelayMessage(payload=final).model_dump_json()
            yield m2

    with patch(
        "domyn.agents.runner.sse_transport.aconnect_sse",
        return_value=StreamWithGarbage(),
    ):
        events = [ev async for ev in transport.execute("Agent", context)]

    assert len(events) == 1


@pytest.mark.asyncio
async def test_execute_normalises_author():
    """author is overwritten to agent_name on every received event."""
    transport = make_transport()
    context = make_context()

    ev = make_event(ExecutionEventType.RESPONSE, is_partial=False, author="some-other-author")

    with patch(
        "domyn.agents.runner.sse_transport.aconnect_sse",
        return_value=_FakeSSEStream([ev]),
    ):
        events = [e async for e in transport.execute("MyAgent", context)]

    assert events[0].author == "MyAgent"


@pytest.mark.asyncio
async def test_execute_includes_conversation_id_in_sse_params():
    """conversation_id from context is passed as a query param."""
    transport = make_transport()
    context = make_context(conversation_id="conv-999")

    final = make_event(ExecutionEventType.RESPONSE, is_partial=False)
    captured_params: list[dict] = []

    def fake_connect(client, method, url, *, params=None, **kwargs):
        captured_params.append(dict(params or {}))
        return _FakeSSEStream([final])

    with patch("domyn.agents.runner.sse_transport.aconnect_sse", side_effect=fake_connect):
        [_ async for _ in transport.execute("Agent", context)]

    assert captured_params[0]["conversationId"] == "conv-999"
    assert captured_params[0]["channelId"] == "ch-abc"
    assert captured_params[0]["spaceId"] == "sp-xyz"


@pytest.mark.asyncio
async def test_send_result_posts_event():
    """send_result() stores pending event for the next SSE reopen."""
    transport = make_transport()
    tool_end = make_event(ExecutionEventType.TOOL_END)

    await transport.send_result(tool_end)

    assert transport._pending_send == tool_end
    assert transport._send_trigger is None


@pytest.mark.asyncio
async def test_execute_sse_stream_ends_without_boundary():
    """If the SSE stream closes without any event, the generator ends cleanly."""
    transport = make_transport()
    context = make_context()

    with patch(
        "domyn.agents.runner.sse_transport.aconnect_sse",
        return_value=_FakeSSEStream([]),  # empty stream
    ):
        events = [ev async for ev in transport.execute("Agent", context)]

    assert events == []


@pytest.mark.asyncio
async def test_execute_sse_timeout_yields_agent_end():
    """An SSE connect/read timeout yields a synthetic AGENT_END with an error message."""
    import httpx

    transport = make_transport()
    context = make_context()

    class TimeoutStream:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def aiter_sse(self):
            raise httpx.TimeoutException("timed out")
            yield  # make it a generator

    with patch(
        "domyn.agents.runner.sse_transport.aconnect_sse",
        return_value=TimeoutStream(),
    ):
        events = [ev async for ev in transport.execute("MyAgent", context)]

    assert len(events) == 1
    ev = events[0]
    assert ev.event_type == ExecutionEventType.AGENT_END
    assert ev.author == "MyAgent"
    assert ev.content
    assert ev.error_code == "delegate_timeout"
    assert "MyAgent" in ev.content[0].text
    assert "timeout" in ev.content[0].text.lower()


# ---------------------------------------------------------------------------
# cancel()
# ---------------------------------------------------------------------------


class TestCancel:
    @pytest.mark.asyncio
    async def test_cancel_posts_agent_end_to_relay(self):
        """cancel() sends an AGENT_END with error_code=user_cancelled to the relay."""
        from unittest.mock import AsyncMock, patch

        import httpx

        transport = make_transport()
        transport._session_started = True
        transport._agent_name = "ext-agent"

        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200

        mock_stream_cm = AsyncMock()
        mock_stream_cm.__aenter__ = AsyncMock(return_value=mock_response)
        mock_stream_cm.__aexit__ = AsyncMock(return_value=False)

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.stream = MagicMock(return_value=mock_stream_cm)

        with patch("domyn.agents.runner.sse_transport.httpx.AsyncClient", return_value=mock_client):
            await transport.cancel("conv-999")

        mock_client.stream.assert_called_once()
        call_kwargs = mock_client.stream.call_args

        # Correct method and relay URL
        assert call_kwargs.args[0] == "POST"
        assert call_kwargs.args[1] == "https://relay.example.com/v1/messages"

        # Correct query params
        params = call_kwargs.kwargs["params"]
        assert params["channelId"] == "ch-abc"
        assert params["spaceId"] == "sp-xyz"
        assert params["conversationId"] == "conv-999"

        # Body is a valid RelayMessage with AGENT_END / user_cancelled
        body_json = call_kwargs.kwargs["content"]
        msg = RelayMessage.model_validate_json(body_json)
        assert msg.payload.event_type == ExecutionEventType.AGENT_END
        assert msg.payload.error_code == "user_cancelled"
        assert msg.payload.conversation_id == "conv-999"

    @pytest.mark.asyncio
    async def test_cancel_skipped_when_session_never_started(self):
        """cancel() must be a no-op if execute() was never entered."""
        from unittest.mock import AsyncMock, patch

        transport = make_transport()
        # _session_started defaults to False / not set

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.stream = MagicMock()

        with patch("domyn.agents.runner.sse_transport.httpx.AsyncClient", return_value=mock_client):
            await transport.cancel("conv-no-session")

        mock_client.stream.assert_not_called()

    @pytest.mark.asyncio
    async def test_cancel_swallows_http_errors(self):
        """cancel() must not raise even if the relay POST fails."""
        from unittest.mock import AsyncMock, patch

        import httpx

        transport = make_transport()
        transport._session_started = True
        transport._agent_name = "ext-agent"

        mock_stream_cm = AsyncMock()
        mock_stream_cm.__aenter__ = AsyncMock(side_effect=httpx.ConnectError("refused"))
        mock_stream_cm.__aexit__ = AsyncMock(return_value=False)

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.stream = MagicMock(return_value=mock_stream_cm)

        with patch("domyn.agents.runner.sse_transport.httpx.AsyncClient", return_value=mock_client):
            # Should not raise
            await transport.cancel("conv-error")


# ---------------------------------------------------------------------------
# event_id deduplication + resume cursor (at-least-once relay -> once to runner)
# ---------------------------------------------------------------------------


def _evt(
    event_type: ExecutionEventType,
    event_id: str,
    *,
    is_partial: bool = False,
    author: str = "ext-agent",
    turn_id: str | None = None,
) -> BaseEvent:
    return BaseEvent(
        event_type=event_type,
        author=author,
        event_id=event_id,
        content=[Part(text="x")],
        is_partial=is_partial,
        turn_id=turn_id,
    )


@pytest.mark.asyncio
async def test_execute_dedups_duplicate_event_ids_in_one_stream():
    """A replayed event (same event_id) is dropped before reaching the runner."""
    transport = make_transport()
    context = make_context()

    p1 = _evt(ExecutionEventType.RESPONSE, "evt-1", is_partial=True)
    p1_dup = _evt(ExecutionEventType.RESPONSE, "evt-1", is_partial=True)
    final = _evt(ExecutionEventType.RESPONSE, "evt-2", is_partial=False)

    with patch(
        "domyn.agents.runner.sse_transport.aconnect_sse",
        return_value=_FakeSSEStream([p1, p1_dup, final]),
    ):
        events = [ev async for ev in transport.execute("Agent", context)]

    assert [e.event_id for e in events] == ["evt-1", "evt-2"]


@pytest.mark.asyncio
async def test_execute_seeds_dedup_from_interaction_history():
    """An event whose id is already in interaction_history is dropped on replay.

    Simulates a fresh-transport resume: the runner rebuilds interaction_history from
    the durable store, so a relay replay of a pre-restart event is deduped.
    """
    transport = make_transport()
    context = make_context()
    context.interaction_history.append(_evt(ExecutionEventType.RESPONSE, "hist-1", is_partial=True))

    replay = _evt(ExecutionEventType.RESPONSE, "hist-1", is_partial=True)
    final = _evt(ExecutionEventType.RESPONSE, "fresh-1", is_partial=False)

    with patch(
        "domyn.agents.runner.sse_transport.aconnect_sse",
        return_value=_FakeSSEStream([replay, final]),
    ):
        events = [ev async for ev in transport.execute("Agent", context)]

    assert [e.event_id for e in events] == ["fresh-1"]


@pytest.mark.asyncio
async def test_execute_dedup_state_is_local_per_call():
    """Dedup state is local to each execute(); a later call does not inherit seen ids.

    This is the behavioral guarantee of holding `seen` as a local rather than a
    transport attribute: reusing an event_id in a subsequent execute() is NOT dropped.
    """
    transport = make_transport()

    with patch(
        "domyn.agents.runner.sse_transport.aconnect_sse",
        return_value=_FakeSSEStream([_evt(ExecutionEventType.RESPONSE, "evt-1", is_partial=False)]),
    ):
        first = [ev async for ev in transport.execute("Agent", make_context("conv-1"))]
    assert [e.event_id for e in first] == ["evt-1"]

    # A fresh execute() reusing the same event_id must still deliver it (state was local).
    with patch(
        "domyn.agents.runner.sse_transport.aconnect_sse",
        return_value=_FakeSSEStream([_evt(ExecutionEventType.RESPONSE, "evt-1", is_partial=False)]),
    ):
        second = [ev async for ev in transport.execute("Agent", make_context("conv-2"))]
    assert [e.event_id for e in second] == ["evt-1"]


@pytest.mark.asyncio
async def test_execute_sends_last_event_id_cursor_on_reopen():
    """On reopen (after send_result) the transport sends Last-Event-ID = last seen id."""
    transport = make_transport()
    context = make_context()

    p1 = _evt(ExecutionEventType.RESPONSE, "evt-1", is_partial=True)
    tool_start = BaseEvent(
        event_type=ExecutionEventType.TOOL_START,
        author="ext",
        event_id="evt-ts",
        action=ToolAction(name="my_tool", parameters={}),
    )
    final = _evt(ExecutionEventType.RESPONSE, "evt-final", is_partial=False)

    sse_calls = [[p1, tool_start], [final]]
    call_idx = 0
    captured_kwargs: list[dict] = []

    def fake_connect(client, method, url, **kwargs):
        nonlocal call_idx
        captured_kwargs.append(kwargs)
        stream = _FakeSSEStream(sse_calls[call_idx])
        call_idx += 1
        return stream

    with patch("domyn.agents.runner.sse_transport.aconnect_sse", side_effect=fake_connect):
        gen = transport.execute("Agent", context)
        await gen.__anext__()  # partial evt-1
        await gen.__anext__()  # tool_start evt-ts
        await transport.send_result(make_event(ExecutionEventType.TOOL_END))
        await gen.__anext__()  # final (from reopened connection)
        with pytest.raises(StopAsyncIteration):
            await gen.__anext__()

    # First open carries no cursor; the reopen carries the last seen event id.
    assert "Last-Event-ID" not in (captured_kwargs[0].get("headers") or {})
    assert captured_kwargs[1]["headers"]["Last-Event-ID"] == "evt-ts"


@pytest.mark.asyncio
async def test_execute_dedups_replayed_events_across_reopen():
    """Events replayed by the relay on the reopened connection are deduped."""
    transport = make_transport()
    context = make_context()

    p1 = _evt(ExecutionEventType.RESPONSE, "evt-1", is_partial=True)
    tool_start = BaseEvent(
        event_type=ExecutionEventType.TOOL_START,
        author="ext",
        event_id="evt-ts",
        action=ToolAction(name="my_tool", parameters={}),
    )
    # Relay replays evt-1 (already seen) plus a fresh final on the reopened stream.
    p1_replay = _evt(ExecutionEventType.RESPONSE, "evt-1", is_partial=True)
    final = _evt(ExecutionEventType.RESPONSE, "evt-final", is_partial=False)

    sse_calls = [[p1, tool_start], [p1_replay, final]]
    call_idx = 0

    def fake_connect(client, method, url, **kwargs):
        nonlocal call_idx
        stream = _FakeSSEStream(sse_calls[call_idx])
        call_idx += 1
        return stream

    with patch("domyn.agents.runner.sse_transport.aconnect_sse", side_effect=fake_connect):
        gen = transport.execute("Agent", context)
        collected = []
        collected.append(await gen.__anext__())  # evt-1
        collected.append(await gen.__anext__())  # evt-ts
        await transport.send_result(make_event(ExecutionEventType.TOOL_END))
        collected.append(await gen.__anext__())  # evt-final (evt-1 replay dropped)
        with pytest.raises(StopAsyncIteration):
            await gen.__anext__()

    assert [e.event_id for e in collected] == ["evt-1", "evt-ts", "evt-final"]


# ---------------------------------------------------------------------------
# Cross-turn replay filtering
#
# The relay's retained log is keyed by channel — deterministic per (conversation,
# agent), so shared by every turn — and it replays the whole window on a connect
# without a resume cursor. The transport is rebuilt per turn, so its first connect
# never has a cursor: a new turn's stream routinely opens with the previous turn's
# events. They are identified by turn_id, which every producer stamps on its
# outbound events.
# ---------------------------------------------------------------------------


def _turn_context(conversation_id: str = "conv-123", turn_id: str = "turn-2") -> InteractionContext:
    context = make_context(conversation_id)
    context.turn_id = turn_id
    return context


@pytest.mark.asyncio
async def test_execute_drops_events_from_another_turn():
    """A replayed event carrying a previous turn's id never reaches the runner."""
    transport = make_transport()
    context = _turn_context(turn_id="turn-2")

    stale = _evt(ExecutionEventType.RESPONSE, "old-1", is_partial=True, turn_id="turn-1")
    mine = _evt(ExecutionEventType.RESPONSE, "new-1", is_partial=True, turn_id="turn-2")
    final = _evt(ExecutionEventType.RESPONSE, "new-2", is_partial=False, turn_id="turn-2")

    with patch(
        "domyn.agents.runner.sse_transport.aconnect_sse",
        return_value=_FakeSSEStream([stale, mine, final]),
    ):
        events = [ev async for ev in transport.execute("Agent", context)]

    assert [e.event_id for e in events] == ["new-1", "new-2"]


@pytest.mark.asyncio
async def test_execute_drops_terminal_event_from_another_turn():
    """A stale terminal event must not end the current turn's session early."""
    transport = make_transport()
    context = _turn_context(turn_id="turn-2")

    stale_end = _evt(ExecutionEventType.AGENT_END, "old-end", turn_id="turn-1")
    mine = _evt(ExecutionEventType.RESPONSE, "new-1", is_partial=True, turn_id="turn-2")
    final = _evt(ExecutionEventType.RESPONSE, "new-2", is_partial=False, turn_id="turn-2")

    with patch(
        "domyn.agents.runner.sse_transport.aconnect_sse",
        return_value=_FakeSSEStream([stale_end, mine, final]),
    ):
        events = [ev async for ev in transport.execute("Agent", context)]

    assert [e.event_id for e in events] == ["new-1", "new-2"]


@pytest.mark.asyncio
async def test_execute_drops_foreign_agent_call_before_nesting_is_counted():
    """A replayed AGENT_CALL from another turn must not skew remote-nesting depth.

    If it were counted, the current turn's final RESPONSE would be treated as a
    nested sub-agent's result and suppressed instead of ending the session.
    """
    transport = make_transport()
    context = _turn_context(turn_id="turn-2")

    stale_call = _evt(ExecutionEventType.AGENT_CALL, "old-call", turn_id="turn-1")
    final = _evt(ExecutionEventType.RESPONSE, "new-final", is_partial=False, turn_id="turn-2")

    with patch(
        "domyn.agents.runner.sse_transport.aconnect_sse",
        return_value=_FakeSSEStream([stale_call, final]),
    ):
        events = [ev async for ev in transport.execute("Agent", context)]

    assert [e.event_id for e in events] == ["new-final"]


@pytest.mark.asyncio
async def test_execute_delivers_events_without_turn_id():
    """Fail-open: a producer that doesn't stamp turn_id is unaffected by the filter."""
    transport = make_transport()
    context = _turn_context(turn_id="turn-2")

    no_turn = _evt(ExecutionEventType.RESPONSE, "anon-1", is_partial=True)
    final = _evt(ExecutionEventType.RESPONSE, "anon-2", is_partial=False)

    with patch(
        "domyn.agents.runner.sse_transport.aconnect_sse",
        return_value=_FakeSSEStream([no_turn, final]),
    ):
        events = [ev async for ev in transport.execute("Agent", context)]

    assert [e.event_id for e in events] == ["anon-1", "anon-2"]


@pytest.mark.asyncio
async def test_execute_delivers_all_turns_when_context_has_no_turn_id():
    """Fail-open the other way: no turn_id on the context means nothing to compare."""
    transport = make_transport()
    context = make_context()  # turn_id unset

    other = _evt(ExecutionEventType.RESPONSE, "x-1", is_partial=True, turn_id="turn-9")
    final = _evt(ExecutionEventType.RESPONSE, "x-2", is_partial=False, turn_id="turn-9")

    with patch(
        "domyn.agents.runner.sse_transport.aconnect_sse",
        return_value=_FakeSSEStream([other, final]),
    ):
        events = [ev async for ev in transport.execute("Agent", context)]

    assert [e.event_id for e in events] == ["x-1", "x-2"]


@pytest.mark.asyncio
async def test_execute_still_dedups_within_the_current_turn():
    """The turn filter and the event_id seen-set are complementary, not redundant.

    A duplicate carrying the CURRENT turn_id passes the turn filter and must still be
    dropped by the seen-set (this is the mid-turn reconnect case).
    """
    transport = make_transport()
    context = _turn_context(turn_id="turn-2")

    p1 = _evt(ExecutionEventType.RESPONSE, "evt-1", is_partial=True, turn_id="turn-2")
    p1_dup = _evt(ExecutionEventType.RESPONSE, "evt-1", is_partial=True, turn_id="turn-2")
    final = _evt(ExecutionEventType.RESPONSE, "evt-2", is_partial=False, turn_id="turn-2")

    with patch(
        "domyn.agents.runner.sse_transport.aconnect_sse",
        return_value=_FakeSSEStream([p1, p1_dup, final]),
    ):
        events = [ev async for ev in transport.execute("Agent", context)]

    assert [e.event_id for e in events] == ["evt-1", "evt-2"]
