"""Metadata propagation across the delegate transport hop.

When the runner hands a task to a remote agent, the transport builds the event
it sends over the wire from ``context.agent_input``. These tests pin down that
the ``metadata`` carried by the original AGENT_START (e.g. ``run_config``,
caller-supplied ``user_info``) — and a TOOL_FEEDBACK resume — is preserved in
the event the remote delegate receives, rather than being dropped when the
outbound event is constructed.

Coverage:
* ``RunnerTransport._build_outbound_event`` — the shared builder used by every
  transport (so both SSE and WebSocket inherit the behaviour).
* ``SSETransport.execute`` — asserts the POSTed body keeps the metadata.
* ``WebSocketTransport.execute`` — asserts the sent frame keeps the metadata.
"""

import contextlib
from collections.abc import AsyncGenerator
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from domyn.agents.core import (
    AgentAction,
    BaseEvent,
    ExecutionEventType,
    InteractionContext,
    Part,
    RelayMessage,
    VisibilityScope,
)
from domyn.agents.runner import RunnerTransport
from domyn.agents.runner.sse_transport import SSETransport
from domyn.agents.runner.websocket_transport import WebSocketTransport

# Representative metadata an upstream runner attaches to an AGENT_START:
# run_config (always merged in by _handle_agent_transition) plus caller info.
SAMPLE_METADATA: dict[str, Any] = {
    "run_config": {"trace": True, "lang": "en"},
    "user_info": {"user_id": "u-42", "tenant": "acme", "roles": ["admin"]},
}


def _ctx(agent_input: BaseEvent) -> InteractionContext:
    return InteractionContext(
        conversation_id="conv-1",
        user_id="u-42",
        agent_input=agent_input,
        interaction_id="int-1",
        turn_id="turn-1",
    )


def _agent_start_with_metadata(author: str = "Orchestrator") -> BaseEvent:
    return BaseEvent(
        event_type=ExecutionEventType.AGENT_START,
        author=author,
        content=[Part(text="summarise the report")],
        action=AgentAction(name="Delegate", parameters={"task": "summarise the report"}),
        conversation_id="conv-orig",
        interaction_id="int-orig",
        turn_id="turn-orig",
        metadata=dict(SAMPLE_METADATA),
    )


def _tool_feedback_with_metadata() -> BaseEvent:
    return BaseEvent(
        event_type=ExecutionEventType.TOOL_FEEDBACK,
        author="user",
        content=[Part(text="use 7 as the tax rate")],
        conversation_id="conv-orig",
        interaction_id="int-orig",
        turn_id="turn-orig",
        metadata=dict(SAMPLE_METADATA),
    )


class _StubTransport(RunnerTransport):
    """Minimal concrete transport to exercise the shared builder directly.

    Deliberately leaves ``type`` unset so it does not self-register.
    """

    async def execute(self, agent_name: str, context: InteractionContext):  # type: ignore[override]
        yield BaseEvent(event_type=ExecutionEventType.AGENT_END, author=agent_name)

    async def send_result(self, event: BaseEvent, conversation_id: str | None = None) -> None:
        return None


# ---------------------------------------------------------------------------
# Shared builder: RunnerTransport._build_outbound_event
# ---------------------------------------------------------------------------


def test_build_outbound_event_preserves_agent_start_metadata() -> None:
    """An AGENT_START's metadata survives; author and ids are normalised to the
    delegating context (so the remote is addressed correctly)."""
    transport = _StubTransport()
    context = _ctx(_agent_start_with_metadata())

    outbound = transport._build_outbound_event("Delegate", context)

    assert outbound.event_type == ExecutionEventType.AGENT_START
    # Metadata carried verbatim — this is the property under test.
    assert outbound.metadata == SAMPLE_METADATA
    # Identity normalised to the current delegation context.
    assert outbound.author == "Delegate"
    assert outbound.conversation_id == "conv-1"
    assert outbound.interaction_id == "int-1"
    assert outbound.turn_id == "turn-1"
    # Payload preserved.
    assert outbound.action is not None and outbound.action.name == "Delegate"
    assert any("summarise the report" in (p.text or "") for p in outbound.content)


def test_build_outbound_event_forwards_tool_feedback_verbatim() -> None:
    """A TOOL_FEEDBACK resume is forwarded as-is (same event_type) with metadata
    intact so the remote knows it is feedback rather than a fresh start."""
    transport = _StubTransport()
    feedback = _tool_feedback_with_metadata()
    context = _ctx(feedback)

    outbound = transport._build_outbound_event("Delegate", context)

    assert outbound is feedback  # forwarded verbatim
    assert outbound.event_type == ExecutionEventType.TOOL_FEEDBACK
    assert outbound.metadata == SAMPLE_METADATA


def test_build_outbound_event_user_input_fallback_builds_minimal() -> None:
    """Fallback: when agent_input is a USER_INPUT (e.g. driving a transport
    directly, outside a Runner — the runner itself sets agent_input to the built
    AGENT_START), a minimal AGENT_START is built with empty metadata."""
    transport = _StubTransport()
    user_input = BaseEvent(
        event_type=ExecutionEventType.USER_INPUT,
        author="user",
        content=[Part(text="hello there")],
    )
    outbound = transport._build_outbound_event("Delegate", _ctx(user_input))
    assert outbound.event_type == ExecutionEventType.AGENT_START
    assert outbound.metadata == {}
    assert any("hello there" in (p.text or "") for p in outbound.content)


# ---------------------------------------------------------------------------
# SSE transport: metadata reaches the POSTed body
# ---------------------------------------------------------------------------


class _SSEStream:
    """Async context manager yielding a fixed list of BaseEvents as SSE frames."""

    def __init__(self, events: list[BaseEvent]) -> None:
        self._events = events

    async def __aenter__(self) -> "_SSEStream":
        return self

    async def __aexit__(self, *_: Any) -> None:
        pass

    async def aiter_sse(self) -> AsyncGenerator[MagicMock, None]:
        for ev in self._events:
            m = MagicMock()
            m.data = RelayMessage(payload=ev).model_dump_json()
            yield m


class _RecordingRelay:
    """Records every POSTed outbound event and replies with a terminal RESPONSE."""

    def __init__(self) -> None:
        self.posted: list[BaseEvent] = []
        self.posted_raw: list[str] = []

    def sse_connect(self, *_: Any, **kwargs: Any) -> _SSEStream:
        if content := kwargs.get("content"):
            self.posted_raw.append(content)
            # Recorded as a bare event only when no envelope is used; envelope
            # frames are asserted via ``posted_raw`` in the run_config tests.
            with contextlib.suppress(Exception):
                self.posted.append(BaseEvent.model_validate_json(content))
        return _SSEStream(
            [
                BaseEvent(
                    event_type=ExecutionEventType.RESPONSE,
                    author="ext",
                    content=[Part(text="done")],
                    is_partial=False,
                )
            ]
        )


def _make_sse_transport() -> SSETransport:
    return SSETransport(
        url="https://relay.example.com/v1/messages",
        channel_id="ch-test",
        space_id="sp-test",
    )


@pytest.mark.asyncio
async def test_sse_execute_posts_agent_start_with_metadata() -> None:
    relay = _RecordingRelay()
    transport = _make_sse_transport()
    context = _ctx(_agent_start_with_metadata())

    with patch("domyn.agents.runner.sse_transport.aconnect_sse", side_effect=relay.sse_connect):
        async for _ in transport.execute("Delegate", context):
            pass

    assert relay.posted, "expected an event POSTed to the relay"
    start = relay.posted[0]
    assert start.event_type == ExecutionEventType.AGENT_START
    assert start.metadata == SAMPLE_METADATA


@pytest.mark.asyncio
async def test_sse_execute_posts_tool_feedback_with_metadata() -> None:
    relay = _RecordingRelay()
    transport = _make_sse_transport()
    context = _ctx(_tool_feedback_with_metadata())

    with patch("domyn.agents.runner.sse_transport.aconnect_sse", side_effect=relay.sse_connect):
        async for _ in transport.execute("Delegate", context):
            pass

    assert relay.posted, "expected an event POSTed to the relay"
    feedback = relay.posted[0]
    assert feedback.event_type == ExecutionEventType.TOOL_FEEDBACK
    assert feedback.metadata == SAMPLE_METADATA


@pytest.mark.asyncio
async def test_runner_root_delegate_forwards_user_input_metadata_not_run_config() -> None:
    """End-to-end: a ROOT DelegateAgent run.

    The AGENT_START inherits the USER_INPUT's metadata, so caller metadata
    (e.g. user_info) reaches the relay — while run_config (the tool channel, which
    carries secrets like auth tokens) is NOT copied onto the event and never egresses.
    """
    from domyn.agents import Runner
    from domyn.agents.agents import DelegateAgent

    relay = _RecordingRelay()
    transport = _make_sse_transport()
    agent = DelegateAgent(name="RemoteAgent", description="answers", transport=transport)

    user_input = BaseEvent(
        event_type=ExecutionEventType.USER_INPUT,
        author="user",
        content=[Part(text="what is the answer?")],
        metadata={"user_info": {"user_id": "u-42"}},
    )

    runner = Runner()
    with patch("domyn.agents.runner.sse_transport.aconnect_sse", side_effect=relay.sse_connect):
        async for _ in runner.run(
            application_name="app",
            user_id="u1",
            root=agent,
            user_input=user_input,
            conversation_id="conv-e2e",
            # A secret in run_config (the tool channel) must NOT reach the delegate.
            run_config={"x-domyn-authorization": "Bearer SECRET", "trace": True},
        ):
            pass

    agent_starts = [p for p in relay.posted if p.event_type == ExecutionEventType.AGENT_START]
    assert agent_starts, "expected an AGENT_START POSTed to the relay"
    start = agent_starts[0]
    # user_input metadata propagates...
    assert start.metadata.get("user_info") == {"user_id": "u-42"}, start.metadata
    # ...but run_config (incl. the secret) does not.
    assert "x-domyn-authorization" not in start.metadata, start.metadata
    assert "trace" not in start.metadata, start.metadata


# ---------------------------------------------------------------------------
# WebSocket transport: metadata reaches the sent frame
# ---------------------------------------------------------------------------


async def _run_ws_execute(context: InteractionContext) -> list[BaseEvent]:
    """Drive WebSocketTransport.execute once and return the frames it sent."""
    transport = WebSocketTransport(ws_url="ws://test")
    sent: list[BaseEvent] = []

    async def _terminal_relay(ws: Any, agent_name: str) -> AsyncGenerator[BaseEvent, None]:
        yield BaseEvent(
            event_type=ExecutionEventType.AGENT_END,
            author="remote",
            content=[Part(text="done")],
        )

    mock_ws = AsyncMock()
    mock_ws.send = AsyncMock(
        side_effect=lambda msg: sent.append(BaseEvent.model_validate_json(msg))
    )
    mock_ws.__aenter__ = AsyncMock(return_value=mock_ws)
    mock_ws.__aexit__ = AsyncMock(return_value=False)

    with (
        patch.object(transport, "_relay", _terminal_relay),
        patch("websockets.asyncio.client.connect", return_value=mock_ws),
    ):
        gen = transport.execute("Delegate", context)
        async for _ in gen:
            pass
        await gen.aclose()

    return sent


@pytest.mark.asyncio
async def test_ws_execute_sends_agent_start_with_metadata() -> None:
    sent = await _run_ws_execute(_ctx(_agent_start_with_metadata()))

    assert sent, "expected a frame sent over the WebSocket"
    start = sent[0]
    assert start.event_type == ExecutionEventType.AGENT_START
    assert start.author == "Delegate"
    assert start.metadata == SAMPLE_METADATA


@pytest.mark.asyncio
async def test_ws_execute_sends_tool_feedback_with_metadata() -> None:
    sent = await _run_ws_execute(_ctx(_tool_feedback_with_metadata()))

    assert sent, "expected a frame sent over the WebSocket"
    feedback = sent[0]
    assert feedback.event_type == ExecutionEventType.TOOL_FEEDBACK
    assert feedback.metadata == SAMPLE_METADATA


# ---------------------------------------------------------------------------
# run_config rides in the RelayMessage envelope run_config field — never in the event
# ---------------------------------------------------------------------------

FILTERED_RUN_CONFIG: dict[str, Any] = {"user_id": "u-42", "space_id": "s-9"}


def test_wrap_outbound_raw_when_no_run_config() -> None:
    """Absent run_config → the event is serialized raw (pre-existing wire format)."""
    transport = _StubTransport()
    event = _agent_start_with_metadata()

    wire = transport._wrap_outbound(event, None)

    # Parses as a bare BaseEvent, not a RelayMessage envelope.
    parsed = BaseEvent.model_validate_json(wire)
    assert parsed.event_type == ExecutionEventType.AGENT_START
    assert "meta" not in wire or '"payload"' not in wire


def test_wrap_outbound_envelopes_run_config_field() -> None:
    """Present run_config → RelayMessage.run_config carries it; event untouched."""
    transport = _StubTransport()
    event = _agent_start_with_metadata()

    wire = transport._wrap_outbound(event, FILTERED_RUN_CONFIG)

    envelope = RelayMessage.model_validate_json(wire)
    # Our filtered run_config lives in the envelope's run_config field...
    assert envelope.run_config == FILTERED_RUN_CONFIG
    # ...and the event payload is left untouched (its own metadata is unchanged).
    assert envelope.payload.metadata == SAMPLE_METADATA
    assert not hasattr(envelope.payload, "run_config")


@pytest.mark.asyncio
async def test_sse_execute_envelopes_run_config_field() -> None:
    """SSE POST body is a RelayMessage whose run_config field carries the config."""
    relay = _RecordingRelay()
    transport = _make_sse_transport()
    context = _ctx(_agent_start_with_metadata())

    with patch("domyn.agents.runner.sse_transport.aconnect_sse", side_effect=relay.sse_connect):
        async for _ in transport.execute("Delegate", context, run_config=FILTERED_RUN_CONFIG):
            pass

    assert relay.posted_raw, "expected an event POSTed to the relay"
    envelope = RelayMessage.model_validate_json(relay.posted_raw[0])
    assert envelope.run_config == FILTERED_RUN_CONFIG
    assert envelope.payload.event_type == ExecutionEventType.AGENT_START
    assert envelope.payload.metadata == SAMPLE_METADATA


async def _run_ws_execute_raw(
    context: InteractionContext, run_config: dict[str, Any] | None
) -> list[str]:
    """Drive WebSocketTransport.execute once and return the raw frames it sent."""
    transport = WebSocketTransport(ws_url="ws://test")
    sent: list[str] = []

    async def _terminal_relay(ws: Any, agent_name: str) -> AsyncGenerator[BaseEvent, None]:
        yield BaseEvent(
            event_type=ExecutionEventType.AGENT_END,
            author="remote",
            content=[Part(text="done")],
        )

    mock_ws = AsyncMock()
    mock_ws.send = AsyncMock(side_effect=lambda msg: sent.append(msg))
    mock_ws.__aenter__ = AsyncMock(return_value=mock_ws)
    mock_ws.__aexit__ = AsyncMock(return_value=False)

    with (
        patch.object(transport, "_relay", _terminal_relay),
        patch("websockets.asyncio.client.connect", return_value=mock_ws),
    ):
        gen = transport.execute("Delegate", context, run_config=run_config)
        async for _ in gen:
            pass
        await gen.aclose()

    return sent


@pytest.mark.asyncio
async def test_ws_execute_envelopes_run_config_field() -> None:
    sent = await _run_ws_execute_raw(_ctx(_agent_start_with_metadata()), FILTERED_RUN_CONFIG)

    assert sent, "expected a frame sent over the WebSocket"
    envelope = RelayMessage.model_validate_json(sent[0])
    assert envelope.run_config == FILTERED_RUN_CONFIG
    assert envelope.payload.event_type == ExecutionEventType.AGENT_START
    assert envelope.payload.author == "Delegate"
    assert envelope.payload.metadata == SAMPLE_METADATA


@pytest.mark.asyncio
async def test_ws_execute_no_envelope_without_run_config() -> None:
    """Default (no run_config) keeps the raw BaseEvent wire format unchanged."""
    sent = await _run_ws_execute_raw(_ctx(_agent_start_with_metadata()), None)

    assert sent
    start = BaseEvent.model_validate_json(sent[0])  # parses as a bare event
    assert start.event_type == ExecutionEventType.AGENT_START


# ---------------------------------------------------------------------------
# DelegateAgent visibility gates what reaches the transport
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("visibility", "expect_run_config"),
    [
        (VisibilityScope.ISOLATED, False),
        (VisibilityScope.SCOPED, False),
        (VisibilityScope.TRUSTED, True),
        (VisibilityScope.GLOBAL, True),
    ],
)
@pytest.mark.asyncio
async def test_delegate_agent_visibility_gates_run_config(
    visibility: VisibilityScope, expect_run_config: bool
) -> None:
    """A DelegateAgent forwards run_config to its transport only for TRUSTED/GLOBAL."""
    from domyn.agents.agents import DelegateAgent

    captured: dict[str, Any] = {}

    class _CapturingTransport(RunnerTransport):
        async def execute(  # type: ignore[override]
            self, agent_name: str, context: InteractionContext, run_config=None
        ):
            captured["run_config"] = run_config
            yield BaseEvent(event_type=ExecutionEventType.AGENT_END, author=agent_name)

        async def send_result(self, event: BaseEvent, conversation_id: str | None = None) -> None:
            return None

    agent = DelegateAgent(
        name="Remote",
        description="d",
        transport=_CapturingTransport(),
        visibility=visibility,
    )
    context = _ctx(_agent_start_with_metadata())
    context.run_config = {"user_id": "u-42", "token": "secret"}

    async for _ in agent.execute(context):
        pass

    if expect_run_config:
        assert captured["run_config"] == {"user_id": "u-42", "token": "secret"}
    else:
        # Denied scopes don't pass the kwarg at all, so the transport sees its default.
        assert not captured["run_config"]
