"""Integration tests for the full Runner → DelegateAgent → SSETransport flow.

The HTTP layer (POST send + SSE receive) is mocked via a stateful relay
simulator so we exercise the real Runner, DelegateAgent, and SSETransport
code paths without needing a live server.
"""

import asyncio
import logging
from collections.abc import AsyncGenerator
from typing import Any
from unittest.mock import MagicMock, patch

import httpx
import pytest

from domyn.agents import Runner
from domyn.agents.agents import Agent, DelegateAgent, HandoffScope
from domyn.agents.core import (
    AgentAction,
    BaseEvent,
    ExecutionEventType,
    InteractionContext,
    Part,
    RelayMessage,
    ToolAction,
)
from domyn.agents.hooks import FunctionHook, HookPhase, HookType
from domyn.agents.llm import BaseLLMProvider, LLMResponse
from domyn.agents.planner_strategy import ToolUsePlannerStrategy
from domyn.agents.runner.sse_transport import SSETransport
from domyn.agents.tools import FunctionTool

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_transport() -> SSETransport:
    return SSETransport(
        url="https://relay.example.com/v1/messages",
        channel_id="ch-test",
        space_id="sp-test",
    )


def _user_input(text: str = "hello") -> BaseEvent:
    return BaseEvent(
        event_type=ExecutionEventType.USER_INPUT,
        author="user",
        content=[Part(text=text)],
    )


class _SSEStream:
    """Async context manager that yields a fixed list of BaseEvents as SSE."""

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


class _RelaySimulator:
    """Stateful mock that drives one scripted conversation.

    *script* is a list of "rounds".  Each round is a list of BaseEvents the
    relay will push over SSE when the client opens the next connection.
    ``sse_connect`` records every outbound event (the POST body) and advances
    to the next round after a non-AGENT_START POST (i.e. after TOOL_END / LLM_END).
    """

    def __init__(self, rounds: list[list[BaseEvent]]) -> None:
        self._rounds = list(rounds)
        self._round_idx = 0
        self.posted: list[BaseEvent] = []

    def sse_connect(self, *_: Any, **kwargs: Any) -> _SSEStream:
        # Outbound events are sent as POST body on SSE open.
        if content := kwargs.get("content"):
            event = BaseEvent.model_validate_json(content)
            self.posted.append(event)
            if event.event_type != ExecutionEventType.AGENT_START:
                self._round_idx += 1

        events = self._rounds[self._round_idx] if self._round_idx < len(self._rounds) else []
        return _SSEStream(events)


# ---------------------------------------------------------------------------
# Test 1: simple delegation — remote agent responds with a plain RESPONSE
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sse_delegate_simple_response():
    """DelegateAgent (root) receives a plain RESPONSE from remote — surfaced to caller."""
    relay = _RelaySimulator(
        rounds=[
            [
                BaseEvent(
                    event_type=ExecutionEventType.RESPONSE,
                    author="ext",
                    content=[Part(text="42 is the answer")],
                    is_partial=False,
                )
            ]
        ]
    )

    transport = _make_transport()
    agent = DelegateAgent(
        name="RemoteAgent",
        description="answers questions",
        transport=transport,
    )

    runner = Runner()
    with patch("domyn.agents.runner.sse_transport.aconnect_sse", side_effect=relay.sse_connect):
        events = [
            e
            async for e in runner.run(
                application_name="app",
                user_id="u1",
                root=agent,
                user_input=_user_input("what is the answer?"),
                conversation_id="conv-1",
            )
        ]

    # AGENT_START was POSTed
    assert relay.posted[0].event_type == ExecutionEventType.AGENT_START

    # Runner surfaces a final RESPONSE
    responses = [
        e for e in events if e.event_type == ExecutionEventType.RESPONSE and not e.is_partial
    ]
    assert responses, "Expected at least one final RESPONSE"
    text = " ".join(p.text or "" for p in responses[-1].content if p.text)
    assert "42" in text


# ---------------------------------------------------------------------------
# Test 2: remote agent signals completion via AGENT_END (with content)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sse_delegate_agent_end_promoted_to_response():
    """AGENT_END with content on the root path is promoted to a final RESPONSE."""
    relay = _RelaySimulator(
        rounds=[
            [
                BaseEvent(
                    event_type=ExecutionEventType.AGENT_END,
                    author="ext",
                    content=[Part(text="done via AGENT_END")],
                )
            ]
        ]
    )

    transport = _make_transport()
    agent = DelegateAgent(name="RemoteAgent", description="desc", transport=transport)

    runner = Runner()
    with patch("domyn.agents.runner.sse_transport.aconnect_sse", side_effect=relay.sse_connect):
        events = [
            e
            async for e in runner.run(
                application_name="app",
                user_id="u1",
                root=agent,
                user_input=_user_input("go"),
                conversation_id="conv-2",
            )
        ]

    responses = [
        e for e in events if e.event_type == ExecutionEventType.RESPONSE and not e.is_partial
    ]
    assert responses
    text = " ".join(p.text or "" for p in responses[-1].content if p.text)
    assert "done via AGENT_END" in text


# ---------------------------------------------------------------------------
# Test 3: local tool round-trip
#   remote → TOOL_START (local tool) → runner executes → TOOL_END sent back
#   remote → RESPONSE
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sse_delegate_local_tool_roundtrip():
    """Remote agent calls a local tool; runner executes it and POSTs TOOL_END back."""

    def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    local_tool = FunctionTool(func=add)

    relay = _RelaySimulator(
        rounds=[
            # Round 0: after AGENT_START → remote asks for local tool
            [
                BaseEvent(
                    event_type=ExecutionEventType.TOOL_START,
                    author="ext",
                    action=ToolAction(name="add", parameters={"a": 3, "b": 4}, call_id="c1"),
                )
            ],
            # Round 1: after TOOL_END → remote sends final response
            [
                BaseEvent(
                    event_type=ExecutionEventType.RESPONSE,
                    author="ext",
                    content=[Part(text="add(3,4)=7")],
                    is_partial=False,
                )
            ],
        ]
    )

    transport = _make_transport()
    agent = DelegateAgent(
        name="RemoteAgent",
        description="desc",
        transport=transport,
        tools=[local_tool],
    )

    runner = Runner()
    with patch("domyn.agents.runner.sse_transport.aconnect_sse", side_effect=relay.sse_connect):
        events = [
            e
            async for e in runner.run(
                application_name="app",
                user_id="u1",
                root=agent,
                user_input=_user_input("add 3 and 4"),
                conversation_id="conv-3",
            )
        ]

    event_types = [e.event_type for e in events]

    # Runner observed TOOL_START and TOOL_END
    assert ExecutionEventType.TOOL_START in event_types
    assert ExecutionEventType.TOOL_END in event_types

    # TOOL_END was POSTed back with the correct result
    tool_end_posts = [p for p in relay.posted if p.event_type == ExecutionEventType.TOOL_END]
    assert tool_end_posts, "Expected TOOL_END to be POSTed to relay"
    assert tool_end_posts[0].action.observation == 7

    # Final RESPONSE surfaces
    responses = [
        e for e in events if e.event_type == ExecutionEventType.RESPONSE and not e.is_partial
    ]
    assert responses
    assert "7" in " ".join(p.text or "" for p in responses[-1].content if p.text)


# ---------------------------------------------------------------------------
# Test 3b: local tool round-trip fires on_agent_event hooks on TOOL_END
#
# Regression test: the runner used to execute a delegate's known tool
# locally, POST the result back, and yield TOOL_END for observability --
# without ever routing it through on_event_hooks.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sse_delegate_local_tool_roundtrip_fires_tool_end_hook():
    """on_agent_event hook with TOOL_END trigger fires for a locally-executed
    known tool, and the result is still POSTed back to the relay."""

    def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    local_tool = FunctionTool(func=add)

    fired_event_types: list[ExecutionEventType] = []

    def _observer(event: BaseEvent, context: Any) -> None:
        fired_event_types.append(event.event_type)

    hook = FunctionHook(
        fun=_observer,
        trigger=HookPhase.ON_AGENT_EVENT,
        type=HookType.LISTENER,
        on_event_triggers=[ExecutionEventType.TOOL_END],
    )

    relay = _RelaySimulator(
        rounds=[
            [
                BaseEvent(
                    event_type=ExecutionEventType.TOOL_START,
                    author="ext",
                    action=ToolAction(name="add", parameters={"a": 3, "b": 4}, call_id="c1"),
                )
            ],
            [
                BaseEvent(
                    event_type=ExecutionEventType.RESPONSE,
                    author="ext",
                    content=[Part(text="add(3,4)=7")],
                    is_partial=False,
                )
            ],
        ]
    )

    transport = _make_transport()
    agent = DelegateAgent(
        name="RemoteAgent",
        description="desc",
        transport=transport,
        tools=[local_tool],
        hooks=[hook],
    )

    runner = Runner()
    with patch("domyn.agents.runner.sse_transport.aconnect_sse", side_effect=relay.sse_connect):
        events = [
            e
            async for e in runner.run(
                application_name="app",
                user_id="u1",
                root=agent,
                user_input=_user_input("add 3 and 4"),
                conversation_id="conv-tool-end-hook",
            )
        ]

    assert ExecutionEventType.TOOL_END in fired_event_types, (
        f"on_agent_event hook did not fire on TOOL_END; "
        f"fired={fired_event_types}, events={[e.event_type for e in events]}"
    )

    # The fix must not break the existing round-trip: the tool result still
    # reaches the relay regardless of hook dispatch.
    tool_end_posts = [p for p in relay.posted if p.event_type == ExecutionEventType.TOOL_END]
    assert tool_end_posts, "Expected TOOL_END to be POSTed to relay"
    assert tool_end_posts[0].action.observation == 7


# ---------------------------------------------------------------------------
# Test 4: streaming partial responses followed by final
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sse_delegate_streaming_response():
    """Partial RESPONSE tokens arrive over the same SSE connection; final closes it."""
    relay = _RelaySimulator(
        rounds=[
            [
                BaseEvent(
                    event_type=ExecutionEventType.RESPONSE,
                    author="ext",
                    content=[Part(text="hel")],
                    is_partial=True,
                ),
                BaseEvent(
                    event_type=ExecutionEventType.RESPONSE,
                    author="ext",
                    content=[Part(text="lo")],
                    is_partial=True,
                ),
                BaseEvent(
                    event_type=ExecutionEventType.RESPONSE,
                    author="ext",
                    content=[Part(text="hello")],
                    is_partial=False,
                ),
            ]
        ]
    )

    transport = _make_transport()
    agent = DelegateAgent(name="StreamAgent", description="streams", transport=transport)

    runner = Runner()
    with patch("domyn.agents.runner.sse_transport.aconnect_sse", side_effect=relay.sse_connect):
        events = [
            e
            async for e in runner.run(
                application_name="app",
                user_id="u1",
                root=agent,
                user_input=_user_input("say hello"),
                conversation_id="conv-4",
            )
        ]

    partial = [e for e in events if e.event_type == ExecutionEventType.RESPONSE and e.is_partial]
    final = [e for e in events if e.event_type == ExecutionEventType.RESPONSE and not e.is_partial]
    assert len(partial) == 2
    assert final


# ---------------------------------------------------------------------------
# Test 5: subagent delegation
#   Root Agent hands off to DelegateAgent; expects AGENT_END back.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sse_delegate_as_subagent():
    """DelegateAgent used as a sub-agent: Runner receives AGENT_END from it."""
    from domyn.agents.llm import BaseLLMProvider, LLMMessage, LLMResponse, StructuredOutputType

    relay = _RelaySimulator(
        rounds=[
            [
                BaseEvent(
                    event_type=ExecutionEventType.RESPONSE,
                    author="ext",
                    content=[Part(text="subtask complete")],
                    is_partial=False,
                )
            ]
        ]
    )

    transport = _make_transport()
    delegate = DelegateAgent(
        name="Specialist",
        description="handles specialist tasks",
        transport=transport,
    )

    class MockLLM(BaseLLMProvider):
        type: str = "mock"
        name: str = "mock"
        model_name: str = "mock"
        supported_structured_outputs: None = None
        _call: int = 0

        async def generate(self, messages: Any, **kwargs: Any) -> LLMResponse:
            self._call += 1
            if self._call == 1:
                # First call: return tool-call-style AGENT_START
                return LLMResponse(
                    content="",
                    tool_calls=[
                        {
                            "id": "tc1",
                            "type": "function",
                            "function": {
                                "name": "Specialist",
                                "arguments": '{"task":"do the subtask"}',
                            },
                        }
                    ],
                    generated_tokens=5,
                    execution_time=0.0,
                )
            # Second call (after AGENT_END): final answer
            return LLMResponse(
                content="all done",
                tool_calls=[],
                generated_tokens=3,
                execution_time=0.0,
            )

        async def generate_streaming(
            self, messages: Any, **kwargs: Any
        ) -> AsyncGenerator[LLMResponse, None]:
            resp = await self.generate(messages, **kwargs)
            yield resp

    root = Agent(
        name="Root",
        description="root agent",
        instruction="you are root",
        llm_provider=MockLLM(),
        sub_agents=[delegate],
    )

    runner = Runner()
    with patch("domyn.agents.runner.sse_transport.aconnect_sse", side_effect=relay.sse_connect):
        events = [
            e
            async for e in runner.run(
                application_name="app",
                user_id="u1",
                root=root,
                user_input=_user_input("do something"),
                conversation_id="conv-5",
            )
        ]

    event_types = [e.event_type for e in events]
    # Delegate's AGENT_END should have been emitted
    assert ExecutionEventType.AGENT_END in event_types


# ---------------------------------------------------------------------------
# Test 6: timeout surfaces error message to user
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sse_delegate_timeout_surfaces_error():
    """SSE timeout yields a synthetic AGENT_END that becomes an error RESPONSE."""

    class TimeoutStream:
        async def __aenter__(self) -> "TimeoutStream":
            return self

        async def __aexit__(self, *_: Any) -> None:
            pass

        async def aiter_sse(self) -> AsyncGenerator[Any, None]:
            raise httpx.TimeoutException("read timeout")
            yield  # make it an async generator

    transport = _make_transport()
    agent = DelegateAgent(name="SlowAgent", description="slow", transport=transport)

    runner = Runner()
    with patch("domyn.agents.runner.sse_transport.aconnect_sse", return_value=TimeoutStream()):
        events = [
            e
            async for e in runner.run(
                application_name="app",
                user_id="u1",
                root=agent,
                user_input=_user_input("do something slow"),
                conversation_id="conv-6",
            )
        ]

    # The synthetic AGENT_END is promoted to RESPONSE on the root path
    responses = [
        e for e in events if e.event_type == ExecutionEventType.RESPONSE and not e.is_partial
    ]
    assert responses, "Expected error RESPONSE"
    text = " ".join(p.text or "" for p in responses[-1].content if p.text)
    assert "SlowAgent" in text
    assert "timeout" in text.lower()


# ---------------------------------------------------------------------------
# Test 7: remote sub-agent AGENT_END does not prematurely terminate session
#   Remote runner: AGENT_CALL → sub-agent AGENT_END → root RESPONSE
#   DelegateAgent (root) must stream through without treating the inner
#   AGENT_END as its own completion.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sse_delegate_sub_agent_end_does_not_terminate_session():
    """Remote sub-agent AGENT_END is forwarded for observability; session continues."""
    relay = _RelaySimulator(
        rounds=[
            [
                # Remote root calls a sub-agent
                BaseEvent(
                    event_type=ExecutionEventType.AGENT_CALL,
                    author="RemoteRoot",
                    content=[Part(text="do subtask")],
                ),
                # Remote sub-agent completes
                BaseEvent(
                    event_type=ExecutionEventType.AGENT_END,
                    author="RemoteSub",
                    content=[Part(text="subtask done")],
                ),
                # Remote root produces the final response
                BaseEvent(
                    event_type=ExecutionEventType.RESPONSE,
                    author="RemoteRoot",
                    content=[Part(text="all done")],
                    is_partial=False,
                ),
            ]
        ]
    )

    transport = _make_transport()
    agent = DelegateAgent(name="RemoteAgent", description="desc", transport=transport)

    runner = Runner()
    with patch("domyn.agents.runner.sse_transport.aconnect_sse", side_effect=relay.sse_connect):
        events = [
            e
            async for e in runner.run(
                application_name="app",
                user_id="u1",
                root=agent,
                user_input=_user_input("do something"),
                conversation_id="conv-7",
            )
        ]

    # Sub-agent AGENT_END is observable but keeps the sub-agent as author
    sub_agent_ends = [
        e
        for e in events
        if e.event_type == ExecutionEventType.AGENT_END and e.author == "RemoteSub"
    ]
    assert sub_agent_ends, "Expected forwarded sub-agent AGENT_END"

    # The session must NOT terminate at the sub-agent AGENT_END —
    # the final RESPONSE from the remote root must still arrive.
    final_responses = [
        e for e in events if e.event_type == ExecutionEventType.RESPONSE and not e.is_partial
    ]
    assert final_responses, "Expected final RESPONSE after sub-agent AGENT_END"
    text = " ".join(p.text or "" for p in final_responses[-1].content if p.text)
    assert "all done" in text


# ---------------------------------------------------------------------------
# Test 8: realistic Domyn-to-Domyn sub-agent flow
#   Remote Domyn runner emits AGENT_CALL → sub-agent RESPONSE → AGENT_END
#   → root RESPONSE (matching the actual remote runner event sequence).
#   The sub-agent RESPONSE must NOT terminate the SSE session.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sse_delegate_subagent_response_before_agent_end():
    """Sub-agent final RESPONSE (before AGENT_END) does not terminate the SSE session."""
    relay = _RelaySimulator(
        rounds=[
            [
                # Remote root delegates to sub-agent
                BaseEvent(
                    event_type=ExecutionEventType.AGENT_CALL,
                    author="RemoteRoot",
                    content=[Part(text="do subtask")],
                ),
                # Sub-agent emits its final RESPONSE (Domyn runner behaviour)
                BaseEvent(
                    event_type=ExecutionEventType.RESPONSE,
                    author="RemoteSub",
                    content=[Part(text="subtask result")],
                    is_partial=False,
                ),
                # Runner synthesises AGENT_END after the sub-agent RESPONSE
                BaseEvent(
                    event_type=ExecutionEventType.AGENT_END,
                    author="RemoteSub",
                    content=[Part(text="subtask result")],
                ),
                # Remote root resumes and produces its final RESPONSE
                BaseEvent(
                    event_type=ExecutionEventType.RESPONSE,
                    author="RemoteRoot",
                    content=[Part(text="root done")],
                    is_partial=False,
                ),
            ]
        ]
    )

    transport = _make_transport()
    agent = DelegateAgent(name="RemoteAgent", description="desc", transport=transport)

    runner = Runner()
    with patch("domyn.agents.runner.sse_transport.aconnect_sse", side_effect=relay.sse_connect):
        events = [
            e
            async for e in runner.run(
                application_name="app",
                user_id="u1",
                root=agent,
                user_input=_user_input("do something"),
                conversation_id="conv-8",
            )
        ]

    # The sub-agent's intermediate RESPONSE must be suppressed — it is an
    # internal event of the remote runner and should not surface as the
    # root session's final answer.
    sub_agent_responses = [
        e
        for e in events
        if e.event_type == ExecutionEventType.RESPONSE
        and e.content
        and any("subtask result" in (p.text or "") for p in e.content)
    ]
    assert not sub_agent_responses, "Sub-agent intermediate RESPONSE must be suppressed"

    # The root's final RESPONSE must still arrive.
    final_responses = [
        e for e in events if e.event_type == ExecutionEventType.RESPONSE and not e.is_partial
    ]
    assert final_responses, "Expected root final RESPONSE"
    text = " ".join(p.text or "" for p in final_responses[-1].content if p.text)
    assert "root done" in text


# ---------------------------------------------------------------------------
# Test 9: author normalisation — root vs. nested events
#   Events at nesting==0 are normalised to the local DelegateAgent name.
#   Events at nesting>0 keep their original remote author.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sse_delegate_normalises_root_events_preserves_nested_authors():
    """Root-level events (nesting==0) are normalised; nested events keep their original author."""
    relay = _RelaySimulator(
        rounds=[
            [
                # Remote root delegates to a sub-agent (nesting: 0 → 1)
                BaseEvent(
                    event_type=ExecutionEventType.AGENT_CALL,
                    author="RemoteRoot",
                    content=[Part(text="do subtask")],
                ),
                # Observational events emitted while the sub-agent is running (nesting=1)
                BaseEvent(
                    event_type=ExecutionEventType.LLM_START,
                    author="RemoteSub",
                ),
                BaseEvent(
                    event_type=ExecutionEventType.TOOL_START,
                    author="RemoteSub",
                    action=ToolAction(name="noop", parameters={}, call_id="c1"),
                ),
                # Sub-agent ends (nesting: 1 → 0)
                BaseEvent(
                    event_type=ExecutionEventType.AGENT_END,
                    author="RemoteSub",
                    content=[Part(text="subtask done")],
                ),
                # Back in the remote root (nesting=0)
                BaseEvent(
                    event_type=ExecutionEventType.LLM_START,
                    author="RemoteRoot",
                ),
                BaseEvent(
                    event_type=ExecutionEventType.RESPONSE,
                    author="RemoteRoot",
                    content=[Part(text="all done")],
                    is_partial=False,
                ),
            ]
        ]
    )

    transport = _make_transport()
    agent = DelegateAgent(name="LocalDelegate", description="desc", transport=transport)

    runner = Runner()
    with patch("domyn.agents.runner.sse_transport.aconnect_sse", side_effect=relay.sse_connect):
        events = [
            e
            async for e in runner.run(
                application_name="app",
                user_id="u1",
                root=agent,
                user_input=_user_input("do something"),
                conversation_id="conv-9",
            )
        ]

    # Root-level events (nesting==0 when received) must be normalised to the
    # DelegateAgent name: AGENT_CALL, the root LLM_START, and the final RESPONSE.
    root_events = [
        e
        for e in events
        if e.event_type
        in (
            ExecutionEventType.AGENT_CALL,
            ExecutionEventType.RESPONSE,
        )
        or (e.event_type == ExecutionEventType.LLM_START and e.author == "LocalDelegate")
    ]
    assert all(e.author == "LocalDelegate" for e in root_events), (
        "Root-level events must be normalised to 'LocalDelegate'"
    )

    # Nested events (nesting>0 when received) must keep their original remote author.
    nested_llm = [
        e
        for e in events
        if e.event_type == ExecutionEventType.LLM_START and e.author == "RemoteSub"
    ]
    nested_tool = [
        e
        for e in events
        if e.event_type == ExecutionEventType.TOOL_START and e.author == "RemoteSub"
    ]
    nested_end = [
        e
        for e in events
        if e.event_type == ExecutionEventType.AGENT_END and e.author == "RemoteSub"
    ]
    assert nested_llm, "Nested LLM_START must keep original author 'RemoteSub'"
    assert nested_tool, "Nested TOOL_START must keep original author 'RemoteSub'"
    assert nested_end, "Nested AGENT_END must keep original author 'RemoteSub'"


# ---------------------------------------------------------------------------
# Test 9-10: relayed handoff is gated by the delegate's LOCAL handoff_scope
#   A remote runner (behind a sub-agent DelegateAgent) relays an AGENT_CALL
#   targeting a LOCALLY-known sibling. The full Runner → DelegateAgent →
#   SSETransport path (incl. RelayMessage JSON round-trip of action.name) must
#   enforce the delegate's local scope before transitioning into that agent.
# ---------------------------------------------------------------------------


class _OnceRelay:
    """Push a scripted SSE round on the first connection, nothing afterwards.

    Prevents the delegate from re-emitting the relayed call when it is re-opened
    after a local transition completes (which would loop).
    """

    def __init__(self, events: list[BaseEvent]) -> None:
        self._events = events
        self._opened = False

    def sse_connect(self, *_: Any, **kwargs: Any) -> _SSEStream:
        if self._opened:
            return _SSEStream([])
        self._opened = True
        return _SSEStream(self._events)


class _RootHandoffLLM(BaseLLMProvider):
    """Hand off to ``SubDelegate`` on the first call, then answer (tool_use planner)."""

    type: str = "mock"
    name: str = "mock"
    model_name: str = "mock"
    supported_structured_outputs: None = None
    _call: int = 0

    async def generate(self, messages: Any, **kwargs: Any) -> LLMResponse:
        self._call += 1
        if self._call == 1:
            return LLMResponse(
                content="",
                tool_calls=[
                    {
                        "id": "tc1",
                        "type": "function",
                        "function": {"name": "SubDelegate", "arguments": '{"task":"go"}'},
                    }
                ],
                generated_tokens=5,
                execution_time=0.0,
            )
        return LLMResponse(
            content="root done", tool_calls=[], generated_tokens=3, execution_time=0.0
        )

    async def generate_streaming(
        self, messages: Any, **kwargs: Any
    ) -> AsyncGenerator[LLMResponse, None]:
        yield await self.generate(messages, **kwargs)


class _FinalLLM(BaseLLMProvider):
    """Answer in a single turn, no tool calls."""

    type: str = "mock"
    name: str = "mock"
    model_name: str = "mock"
    supported_structured_outputs: None = None

    async def generate(self, messages: Any, **kwargs: Any) -> LLMResponse:
        return LLMResponse(
            content="local ran", tool_calls=[], generated_tokens=2, execution_time=0.0
        )

    async def generate_streaming(
        self, messages: Any, **kwargs: Any
    ) -> AsyncGenerator[LLMResponse, None]:
        yield await self.generate(messages, **kwargs)


def _sse_relay_tree(delegate_scope: HandoffScope) -> tuple[Agent, _OnceRelay]:
    """Root → [SubDelegate (SSE), LocalTarget]; the relay pushes AGENT_CALL→LocalTarget.

    Root and LocalTarget use the real ``tool_use`` planner (native tool calling,
    as in production), so the local handoff and LocalTarget's turn go through
    the real parse path. Only the delegate leg crosses the real SSETransport +
    RelayMessage round-trip the fix guards.
    """
    delegate = DelegateAgent(
        name="SubDelegate",
        description="sub delegate",
        transport=_make_transport(),
        handoff_scope=delegate_scope,
    )
    local_target = Agent(
        name="LocalTarget",
        description="local",
        instruction="be helpful",
        llm_provider=_FinalLLM(),
        planner=ToolUsePlannerStrategy(),
    )
    root = Agent(
        name="Root",
        description="root agent",
        instruction="you are root",
        llm_provider=_RootHandoffLLM(),
        planner=ToolUsePlannerStrategy(),
        sub_agents=[delegate, local_target],
    )
    relay = _OnceRelay(
        [
            BaseEvent(
                event_type=ExecutionEventType.AGENT_CALL,
                author="RemoteRoot",
                content=[],
                action=AgentAction(
                    name="LocalTarget",
                    parameters={"task": "sensitive work"},
                    thought="delegating to local",
                ),
            )
        ]
    )
    return root, relay


@pytest.mark.asyncio
async def test_sse_relayed_handoff_off_scope_local_agent_denied(caplog):
    """EXPLICIT delegate: relayed AGENT_CALL to a local sibling is denied through SSE."""
    root, relay = _sse_relay_tree(HandoffScope.EXPLICIT)
    runner = Runner()
    with (
        patch("domyn.agents.runner.sse_transport.aconnect_sse", side_effect=relay.sse_connect),
        caplog.at_level(logging.WARNING, logger="domyn.agents"),
    ):
        events = [
            e
            async for e in runner.run(
                application_name="app",
                user_id="u1",
                root=root,
                user_input=_user_input("go"),
                conversation_id="conv-relay-sse-deny",
            )
        ]

    local_starts = [
        e
        for e in events
        if e.event_type == ExecutionEventType.AGENT_START and e.author == "LocalTarget"
    ]
    assert not local_starts, "off-scope relayed AGENT_CALL must not transition into the local agent"
    assert "denied" in caplog.text and "LocalTarget" in caplog.text, (
        "the denial must be audited via a warning log"
    )


@pytest.mark.asyncio
async def test_sse_relayed_handoff_in_scope_local_agent_allowed():
    """GLOBAL delegate: relayed AGENT_CALL resolves the tree, reaching the local sibling via SSE."""
    root, relay = _sse_relay_tree(HandoffScope.GLOBAL)
    runner = Runner()
    with patch("domyn.agents.runner.sse_transport.aconnect_sse", side_effect=relay.sse_connect):
        events = [
            e
            async for e in runner.run(
                application_name="app",
                user_id="u1",
                root=root,
                user_input=_user_input("go"),
                conversation_id="conv-relay-sse-allow",
            )
        ]

    local_starts = [
        e
        for e in events
        if e.event_type == ExecutionEventType.AGENT_START and e.author == "LocalTarget"
    ]
    assert local_starts, "in-scope relayed AGENT_CALL must transition into the local agent via SSE"
