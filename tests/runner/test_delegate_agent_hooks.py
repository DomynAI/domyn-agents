"""Tests to verify hook execution for DelegateAgent instances.

Covers:
- pre_agent_call hooks fire when DelegateAgent is the root agent
- pre_agent_call hooks fire when DelegateAgent is a sub-agent
- post_agent_call hooks fire when DelegateAgent is the root agent
- post_agent_call hooks fire when DelegateAgent is a sub-agent
- on_agent_event hooks fire on AGENT_START for a sub-agent DelegateAgent
- on_agent_event hooks fire on proxied AGENT_START events from the remote

TOOL_END-on-known-tool coverage lives in test_sse_delegate_flow.py, next to the
plain (non-hook) local-tool round-trip test -- this module's module-level
`from __future__ import annotations` breaks FunctionTool's signature
introspection for a locally-defined function, which that module doesn't have.
"""

from collections.abc import AsyncGenerator
from typing import Any

import pytest

from domyn.agents import Runner
from domyn.agents.agents import Agent, DelegateAgent
from domyn.agents.core import AgentAction, BaseEvent, ExecutionEventType, InteractionContext, Part
from domyn.agents.hooks import FunctionHook, HookPhase, HookType
from domyn.agents.llm import BaseLLMProvider, LLMMessage, LLMResponse, StructuredOutputType
from domyn.agents.runner import RunnerTransport

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _user_input(text: str = "hello") -> BaseEvent:
    return BaseEvent(
        event_type=ExecutionEventType.USER_INPUT,
        author="user",
        content=[Part(text=text)],
    )


class _MockTransport(RunnerTransport):
    """Simple in-process transport that yields a fixed sequence of events."""

    def __init__(self, events: list[BaseEvent]) -> None:
        self._events = events

    async def execute(
        self, agent_name: str, context: InteractionContext
    ) -> AsyncGenerator[BaseEvent, None]:
        for event in self._events:
            yield event

    async def send_result(self, event: BaseEvent, conversation_id: str | None = None) -> None:
        pass


def _make_listener_hook(phase: HookPhase, called: list[str], label: str) -> FunctionHook:
    """Create a listener hook that appends *label* to *called* when it fires."""

    def _hook(context: Any, agent: Any) -> None:
        called.append(label)

    return FunctionHook(fun=_hook, trigger=phase, type=HookType.LISTENER)


def _make_on_event_listener_hook(called: list[ExecutionEventType]) -> FunctionHook:
    """Create an on_agent_event listener that records which event types fire."""

    def _hook(event: BaseEvent, context: Any) -> None:
        called.append(event.event_type)

    return FunctionHook(
        fun=_hook,
        trigger=HookPhase.ON_AGENT_EVENT,
        type=HookType.LISTENER,
        on_event_triggers=[
            ExecutionEventType.AGENT_START,
            ExecutionEventType.AGENT_END,
        ],
    )


class _MockLLMProvider(BaseLLMProvider):
    type: str = "mock"
    name: str = "mock"
    model_name: str = "mock"
    supported_structured_outputs: None = None

    async def generate(self, messages: LLMMessage, **kwargs: Any) -> LLMResponse:
        return LLMResponse(content="", raw_response=None)

    async def generate_streaming(
        self, messages: LLMMessage, **kwargs: Any
    ) -> AsyncGenerator[LLMResponse, None]:
        yield LLMResponse(content="", raw_response=None)


def _make_parent_agent(delegate: DelegateAgent, execute_events: list[BaseEvent]) -> Agent:
    """Return a parent Agent whose execute() yields *execute_events* on first call, then stops."""
    llm = _MockLLMProvider(name="mock", model_name="mock")

    parent = Agent(
        name="ParentAgent",
        description="parent",
        instruction="be helpful",
        llm_provider=llm,
        sub_agents=[delegate],
    )

    # Stateful: first call yields scripted events; subsequent calls yield nothing.
    _events = list(execute_events)
    _call_count = [0]

    async def _fake_execute(context: InteractionContext) -> AsyncGenerator[BaseEvent, None]:
        if _call_count[0] == 0:
            _call_count[0] += 1
            for ev in _events:
                yield ev
        # second call (resumed after delegate returns) → yield nothing → StopAsyncIteration

    parent._execute = _fake_execute  # type: ignore[method-assign]

    return parent


# ---------------------------------------------------------------------------
# Tests: root DelegateAgent
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pre_hook_fires_for_root_delegate():
    """pre_agent_call hook fires when DelegateAgent is the root/entrypoint agent."""
    called: list[str] = []
    hook = _make_listener_hook(HookPhase.PRE_AGENT_CALL, called, "pre")

    transport = _MockTransport(
        [
            BaseEvent(
                event_type=ExecutionEventType.RESPONSE,
                author="ext",
                content=[Part(text="done")],
                is_partial=False,
            )
        ]
    )
    agent = DelegateAgent(
        name="RemoteRoot",
        description="root delegate",
        transport=transport,
        hooks=[hook],
    )

    runner = Runner()
    events = [
        e
        async for e in runner.run(
            application_name="app",
            user_id="u1",
            root=agent,
            user_input=_user_input("hello"),
            conversation_id="conv-root-pre",
        )
    ]

    assert "pre" in called, (
        f"pre_agent_call hook not called; events={[e.event_type for e in events]}"
    )


@pytest.mark.asyncio
async def test_post_hook_fires_for_root_delegate():
    """post_agent_call hook fires when DelegateAgent is the root/entrypoint agent."""
    called: list[str] = []
    hook = _make_listener_hook(HookPhase.POST_AGENT_CALL, called, "post")

    transport = _MockTransport(
        [
            BaseEvent(
                event_type=ExecutionEventType.RESPONSE,
                author="ext",
                content=[Part(text="done")],
                is_partial=False,
            )
        ]
    )
    agent = DelegateAgent(
        name="RemoteRoot",
        description="root delegate",
        transport=transport,
        hooks=[hook],
    )

    runner = Runner()
    events = [
        e
        async for e in runner.run(
            application_name="app",
            user_id="u1",
            root=agent,
            user_input=_user_input("hello"),
            conversation_id="conv-root-post",
        )
    ]

    assert "post" in called, (
        f"post_agent_call hook not called; events={[e.event_type for e in events]}"
    )


# ---------------------------------------------------------------------------
# Tests: sub-agent DelegateAgent (called via AGENT_CALL from parent)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pre_hook_fires_for_subagent_delegate():
    """pre_agent_call hook fires when DelegateAgent is a sub-agent."""
    called: list[str] = []
    hook = _make_listener_hook(HookPhase.PRE_AGENT_CALL, called, "pre")

    transport = _MockTransport(
        [
            BaseEvent(
                event_type=ExecutionEventType.RESPONSE,
                author="ext",
                content=[Part(text="sub done")],
                is_partial=False,
            )
        ]
    )
    delegate = DelegateAgent(
        name="SubDelegate",
        description="sub delegate",
        transport=transport,
        hooks=[hook],
    )

    # Parent agent emits AGENT_CALL → delegate, then stops
    parent_events = [
        BaseEvent(
            event_type=ExecutionEventType.AGENT_CALL,
            author="ParentAgent",
            content=[],
            action=AgentAction(
                name="SubDelegate",
                parameters={"task": "do something"},
                thought="calling sub",
            ),
        ),
    ]
    parent = _make_parent_agent(delegate, parent_events)

    runner = Runner()
    events = [
        e
        async for e in runner.run(
            application_name="app",
            user_id="u1",
            root=parent,
            user_input=_user_input("go"),
            conversation_id="conv-sub-pre",
        )
    ]

    assert "pre" in called, (
        f"pre_agent_call hook not called; events={[e.event_type for e in events]}"
    )


@pytest.mark.asyncio
async def test_post_hook_fires_for_subagent_delegate():
    """post_agent_call hook fires when DelegateAgent is a sub-agent."""
    called: list[str] = []
    hook = _make_listener_hook(HookPhase.POST_AGENT_CALL, called, "post")

    transport = _MockTransport(
        [
            BaseEvent(
                event_type=ExecutionEventType.RESPONSE,
                author="ext",
                content=[Part(text="sub done")],
                is_partial=False,
            )
        ]
    )
    delegate = DelegateAgent(
        name="SubDelegate",
        description="sub delegate",
        transport=transport,
        hooks=[hook],
    )

    parent_events = [
        BaseEvent(
            event_type=ExecutionEventType.AGENT_CALL,
            author="ParentAgent",
            content=[],
            action=AgentAction(
                name="SubDelegate",
                parameters={"task": "do something"},
                thought="calling sub",
            ),
        ),
    ]
    parent = _make_parent_agent(delegate, parent_events)

    runner = Runner()
    events = [
        e
        async for e in runner.run(
            application_name="app",
            user_id="u1",
            root=parent,
            user_input=_user_input("go"),
            conversation_id="conv-sub-post",
        )
    ]

    assert "post" in called, (
        f"post_agent_call hook not called; events={[e.event_type for e in events]}"
    )


# ---------------------------------------------------------------------------
# Tests: on_agent_event hooks on AGENT_START / AGENT_END
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_on_event_hook_fires_on_agent_start_for_subagent_delegate():
    """on_agent_event hook with AGENT_START trigger fires when sub-agent DelegateAgent starts."""
    fired_event_types: list[ExecutionEventType] = []
    hook = _make_on_event_listener_hook(fired_event_types)

    transport = _MockTransport(
        [
            BaseEvent(
                event_type=ExecutionEventType.RESPONSE,
                author="ext",
                content=[Part(text="sub done")],
                is_partial=False,
            )
        ]
    )
    delegate = DelegateAgent(
        name="SubDelegate",
        description="sub delegate",
        transport=transport,
        hooks=[hook],
    )

    parent_events = [
        BaseEvent(
            event_type=ExecutionEventType.AGENT_CALL,
            author="ParentAgent",
            content=[],
            action=AgentAction(
                name="SubDelegate",
                parameters={"task": "do something"},
                thought="calling sub",
            ),
        ),
    ]
    parent = _make_parent_agent(delegate, parent_events)

    runner = Runner()
    events = [
        e
        async for e in runner.run(
            application_name="app",
            user_id="u1",
            root=parent,
            user_input=_user_input("go"),
            conversation_id="conv-sub-on-event-start",
        )
    ]

    assert ExecutionEventType.AGENT_START in fired_event_types, (
        f"on_agent_event hook did not fire on AGENT_START; "
        f"fired={fired_event_types}, events={[e.event_type for e in events]}"
    )


@pytest.mark.asyncio
async def test_on_event_hook_fires_on_agent_end_for_subagent_delegate():
    """on_agent_event hook with AGENT_END trigger fires when sub-agent DelegateAgent ends."""
    fired_event_types: list[ExecutionEventType] = []
    hook = _make_on_event_listener_hook(fired_event_types)

    transport = _MockTransport(
        [
            BaseEvent(
                event_type=ExecutionEventType.RESPONSE,
                author="ext",
                content=[Part(text="sub done")],
                is_partial=False,
            )
        ]
    )
    delegate = DelegateAgent(
        name="SubDelegate",
        description="sub delegate",
        transport=transport,
        hooks=[hook],
    )

    parent_events = [
        BaseEvent(
            event_type=ExecutionEventType.AGENT_CALL,
            author="ParentAgent",
            content=[],
            action=AgentAction(
                name="SubDelegate",
                parameters={"task": "do something"},
                thought="calling sub",
            ),
        ),
    ]
    parent = _make_parent_agent(delegate, parent_events)

    runner = Runner()
    events = [
        e
        async for e in runner.run(
            application_name="app",
            user_id="u1",
            root=parent,
            user_input=_user_input("go"),
            conversation_id="conv-sub-on-event-end",
        )
    ]

    assert ExecutionEventType.AGENT_END in fired_event_types, (
        f"on_agent_event hook did not fire on AGENT_END; "
        f"fired={fired_event_types}, events={[e.event_type for e in events]}"
    )


@pytest.mark.asyncio
async def test_on_event_hook_fires_on_proxied_agent_start_from_remote():
    """on_agent_event hooks fire on AGENT_START events proxied from the remote agent."""
    fired_event_types: list[ExecutionEventType] = []
    hook = _make_on_event_listener_hook(fired_event_types)

    # Remote sends back an AGENT_START (it's delegating to one of its own sub-agents)
    # followed by a final RESPONSE.
    transport = _MockTransport(
        [
            BaseEvent(
                event_type=ExecutionEventType.AGENT_START,
                author="RemoteSubAgent",
                content=[Part(text="starting sub")],
                action=AgentAction(name="RemoteSubAgent", parameters={}),
            ),
            BaseEvent(
                event_type=ExecutionEventType.RESPONSE,
                author="ext",
                content=[Part(text="done")],
                is_partial=False,
            ),
        ]
    )
    agent = DelegateAgent(
        name="RemoteRoot",
        description="root delegate",
        transport=transport,
        hooks=[hook],
    )

    runner = Runner()
    events = [
        e
        async for e in runner.run(
            application_name="app",
            user_id="u1",
            root=agent,
            user_input=_user_input("hello"),
            conversation_id="conv-proxied-agent-start",
        )
    ]

    # The proxied AGENT_START from the remote should also trigger the on_event hook
    assert fired_event_types.count(ExecutionEventType.AGENT_START) >= 1, (
        f"on_agent_event hook did not fire on proxied AGENT_START; "
        f"fired={fired_event_types}, events={[e.event_type for e in events]}"
    )


# ---------------------------------------------------------------------------
# Test: hook-modified AGENT_START is forwarded to the transport
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_hook_modified_agent_start_propagates_to_transport():
    """An on_agent_event updater hook that modifies AGENT_START content is seen by the transport."""
    received_agent_start: list[BaseEvent] = []

    class _CapturingTransport(RunnerTransport):
        async def execute(
            self, agent_name: str, context: InteractionContext
        ) -> AsyncGenerator[BaseEvent, None]:
            # Record the agent_input the transport sees (runner's AGENT_START after hooks)
            if context.agent_input is not None:
                received_agent_start.append(context.agent_input)
            yield BaseEvent(
                event_type=ExecutionEventType.RESPONSE,
                author="ext",
                content=[Part(text="done")],
                is_partial=False,
            )

        async def send_result(self, event: BaseEvent, conversation_id: str | None = None) -> None:
            pass

    def _modifying_hook(event: BaseEvent, context: Any) -> dict:
        # Return a modified event via the on_agent_event updater protocol
        modified = event.model_copy(update={"content": [Part(text="hook-modified")]})
        return {"modified_event": modified}

    hook = FunctionHook(
        fun=_modifying_hook,
        trigger=HookPhase.ON_AGENT_EVENT,
        type=HookType.UPDATER,
        on_event_triggers=[ExecutionEventType.AGENT_START],
    )

    delegate = DelegateAgent(
        name="SubDelegate",
        description="sub delegate",
        transport=_CapturingTransport(),
        hooks=[hook],
    )

    parent_events = [
        BaseEvent(
            event_type=ExecutionEventType.AGENT_CALL,
            author="ParentAgent",
            content=[],
            action=AgentAction(
                name="SubDelegate",
                parameters={"task": "original task"},
                thought="calling sub",
            ),
        ),
    ]
    parent = _make_parent_agent(delegate, parent_events)

    runner = Runner()
    [
        e
        async for e in runner.run(
            application_name="app",
            user_id="u1",
            root=parent,
            user_input=_user_input("go"),
            conversation_id="conv-hook-modified-agent-start",
        )
    ]

    assert received_agent_start, "Transport never received an AGENT_START in context.agent_input"
    text = " ".join(p.text or "" for p in received_agent_start[0].content if p.text)
    assert text == "hook-modified", f"Transport saw unmodified AGENT_START content; got={text!r}"


# ---------------------------------------------------------------------------
# Test: on_agent_event hook with on_event_triggers=[AGENT_START, USER_INPUT]
# on a delegate sub-agent fires on AGENT_START but NOT on USER_INPUT.
#
# USER_INPUT is processed at the root-agent level before the sub-agent is
# even on the stack, so the sub-agent's hooks never see it.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_on_event_hook_with_agent_start_and_user_input_triggers_on_subagent_delegate():
    """on_event hook with [AGENT_START, USER_INPUT] triggers fires on AGENT_START.

    The hook is registered on a DelegateAgent that is called as a sub-agent.
    USER_INPUT arrives at the root agent level, so the sub-agent's hook does NOT
    fire on it.  AGENT_START is created by the runner for the sub-agent transition
    and the hook fires on that.
    """
    fired: list[ExecutionEventType] = []

    def _observer(event: BaseEvent, context: Any) -> None:
        fired.append(event.event_type)

    hook = FunctionHook(
        fun=_observer,
        trigger=HookPhase.ON_AGENT_EVENT,
        type=HookType.LISTENER,
        on_event_triggers=[ExecutionEventType.AGENT_START, ExecutionEventType.USER_INPUT],
    )

    transport = _MockTransport(
        [
            BaseEvent(
                event_type=ExecutionEventType.RESPONSE,
                author="ext",
                content=[Part(text="done")],
                is_partial=False,
            )
        ]
    )
    delegate = DelegateAgent(
        name="SubDelegate",
        description="sub delegate",
        transport=transport,
        hooks=[hook],
    )

    parent_events = [
        BaseEvent(
            event_type=ExecutionEventType.AGENT_CALL,
            author="ParentAgent",
            content=[],
            action=AgentAction(
                name="SubDelegate",
                parameters={"task": "do something"},
                thought="calling sub",
            ),
        ),
    ]
    parent = _make_parent_agent(delegate, parent_events)

    runner = Runner()
    [
        e
        async for e in runner.run(
            application_name="app",
            user_id="u1",
            root=parent,
            user_input=_user_input("original user message"),
            conversation_id="conv-agent-start-and-user-input",
        )
    ]

    # Fires exactly once on AGENT_START; USER_INPUT never reaches the sub-agent's hooks.
    assert fired == [ExecutionEventType.AGENT_START], (
        f"Expected exactly one AGENT_START firing; fired={fired}"
    )


# ---------------------------------------------------------------------------
# Test: on_event_triggers=[RESPONSE] on a delegate sub-agent
#
# DelegateAgent.execute() converts the remote's final RESPONSE to AGENT_END
# before yielding it to the runner.  The on_event_hooks loop therefore never
# sees a RESPONSE event — use AGENT_END to observe the sub-agent finishing.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_on_event_hook_response_trigger_does_not_fire_for_subagent_delegate():
    """Hook with [RESPONSE] on a sub-agent DelegateAgent never fires.

    DelegateAgent converts the remote RESPONSE → AGENT_END before the runner
    sees it, so a RESPONSE trigger is silently skipped.
    Use [AGENT_END] instead to react to the sub-agent finishing.
    """
    fired: list[ExecutionEventType] = []

    def _observer(event: BaseEvent, context: Any) -> None:
        fired.append(event.event_type)

    hook = FunctionHook(
        fun=_observer,
        trigger=HookPhase.ON_AGENT_EVENT,
        type=HookType.LISTENER,
        on_event_triggers=[ExecutionEventType.RESPONSE],
    )

    transport = _MockTransport(
        [
            BaseEvent(
                event_type=ExecutionEventType.RESPONSE,
                author="ext",
                content=[Part(text="done")],
                is_partial=False,
            )
        ]
    )
    delegate = DelegateAgent(
        name="SubDelegate",
        description="sub delegate",
        transport=transport,
        hooks=[hook],
    )

    parent_events = [
        BaseEvent(
            event_type=ExecutionEventType.AGENT_CALL,
            author="ParentAgent",
            content=[],
            action=AgentAction(
                name="SubDelegate",
                parameters={"task": "do something"},
                thought="calling sub",
            ),
        ),
    ]
    parent = _make_parent_agent(delegate, parent_events)

    runner = Runner()
    [
        e
        async for e in runner.run(
            application_name="app",
            user_id="u1",
            root=parent,
            user_input=_user_input("go"),
            conversation_id="conv-response-trigger-subagent",
        )
    ]

    assert fired == [], (
        f"Expected RESPONSE hook to never fire for sub-agent delegate; fired={fired}"
    )
