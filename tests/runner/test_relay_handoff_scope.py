"""Relay ``handoff_scope`` enforcement.

A relayed AGENT_CALL (emitted by a remote runner behind a ``DelegateAgent``)
that targets a locally-known agent must be gated by the delegate's own LOCAL
``handoff_scope`` before the runner transitions into that local agent. The
remote runner only gated the call against the REMOTE scope, so without a local
check a delegate could reach any locally-known agent regardless of its scope.
"""

import logging
from collections.abc import AsyncGenerator
from typing import Any

import pytest

from domyn.agents import Runner
from domyn.agents.agents import Agent, DelegateAgent, HandoffScope
from domyn.agents.core import AgentAction, BaseEvent, ExecutionEventType, InteractionContext, Part
from domyn.agents.llm import BaseLLMProvider, LLMMessage, LLMResponse
from domyn.agents.runner import RunnerTransport


def _user_input(text: str = "go") -> BaseEvent:
    return BaseEvent(
        event_type=ExecutionEventType.USER_INPUT, author="user", content=[Part(text=text)]
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


class _OnceTransport(RunnerTransport):
    """Yield a fixed sequence on the first ``execute()`` call, nothing afterwards."""

    def __init__(self, events: list[BaseEvent]) -> None:
        self._events = list(events)
        self._used = False

    async def execute(
        self, agent_name: str, context: InteractionContext
    ) -> AsyncGenerator[BaseEvent, None]:
        if not self._used:
            self._used = True
            for event in self._events:
                yield event

    async def send_result(self, event: BaseEvent, conversation_id: str | None = None) -> None:
        pass


def _scripted_execute(agent: Agent, events: list[BaseEvent]) -> None:
    """Patch ``agent._execute`` to yield *events* on the first call, then nothing."""
    remaining = list(events)
    calls = [0]

    async def _fake_execute(context: InteractionContext) -> AsyncGenerator[BaseEvent, None]:
        if calls[0] == 0:
            calls[0] += 1
            for ev in remaining:
                yield ev

    agent._execute = _fake_execute  # type: ignore[method-assign]


def _build_tree(
    delegate_scope: HandoffScope, relayed_target: str = "LocalTarget"
) -> tuple[Agent, list[str]]:
    """Build ``Parent(entrypoint) → [SubDelegate, LocalTarget]``.

    ``SubDelegate`` relays an AGENT_CALL targeting *relayed_target* (the
    locally-known ``LocalTarget`` by default; pass an unknown name to exercise
    the genuinely-remote path). Returns ``(parent, executed)`` where *executed*
    records whether ``LocalTarget``'s body actually ran.
    """
    llm = _MockLLMProvider(name="mock", model_name="mock")

    delegate = DelegateAgent(
        name="SubDelegate",
        description="sub delegate",
        transport=_OnceTransport(
            [
                # Remote runner relays a handoff to a LOCALLY-known agent.
                BaseEvent(
                    event_type=ExecutionEventType.AGENT_CALL,
                    author="RemoteRoot",
                    content=[],
                    action=AgentAction(
                        name=relayed_target,
                        parameters={"task": "sensitive work"},
                        thought="delegating to local",
                    ),
                ),
                # Completion consumed only when the relayed call is denied locally.
                BaseEvent(
                    event_type=ExecutionEventType.RESPONSE,
                    author="RemoteRoot",
                    content=[Part(text="remote done")],
                    is_partial=False,
                ),
            ]
        ),
        handoff_scope=delegate_scope,
    )

    executed: list[str] = []
    local_target = Agent(
        name="LocalTarget",
        description="local",
        instruction="be helpful",
        llm_provider=llm,
    )

    async def _local_execute(context: InteractionContext) -> AsyncGenerator[BaseEvent, None]:
        executed.append("ran")
        yield BaseEvent(
            event_type=ExecutionEventType.RESPONSE,
            author="LocalTarget",
            content=[Part(text="local ran")],
            is_partial=False,
        )

    local_target._execute = _local_execute  # type: ignore[method-assign]

    parent = Agent(
        name="ParentAgent",
        description="parent",
        instruction="be helpful",
        llm_provider=llm,
        sub_agents=[delegate, local_target],
    )
    _scripted_execute(
        parent,
        [
            BaseEvent(
                event_type=ExecutionEventType.AGENT_CALL,
                author="ParentAgent",
                content=[],
                action=AgentAction(
                    name="SubDelegate", parameters={"task": "go"}, thought="call delegate"
                ),
            )
        ],
    )
    return parent, executed


async def _run(parent: Agent, conversation_id: str) -> list[BaseEvent]:
    runner = Runner()
    return [
        e
        async for e in runner.run(
            application_name="app",
            user_id="u1",
            root=parent,
            user_input=_user_input(),
            conversation_id=conversation_id,
        )
    ]


@pytest.mark.asyncio
async def test_relayed_handoff_to_off_scope_local_agent_is_denied(caplog):
    """An EXPLICIT delegate (empty ``sub_agents``) must not reach a local sibling via relay."""
    parent, executed = _build_tree(HandoffScope.EXPLICIT)
    with caplog.at_level(logging.WARNING, logger="domyn.agents"):
        events = await _run(parent, "conv-relay-deny")

    local_starts = [
        e
        for e in events
        if e.event_type == ExecutionEventType.AGENT_START and e.author == "LocalTarget"
    ]
    assert not local_starts, "off-scope relayed AGENT_CALL must not transition into the local agent"
    assert executed == [], "off-scope local agent body must not run"
    assert "denied" in caplog.text and "LocalTarget" in caplog.text, (
        "an off-scope denial of a locally-known target must be audited via a warning log"
    )


@pytest.mark.asyncio
async def test_relayed_handoff_to_in_scope_local_agent_is_allowed(caplog):
    """A GLOBAL delegate resolves the whole tree, so the local sibling is reachable."""
    parent, executed = _build_tree(HandoffScope.GLOBAL)
    with caplog.at_level(logging.WARNING, logger="domyn.agents"):
        events = await _run(parent, "conv-relay-allow")

    local_starts = [
        e
        for e in events
        if e.event_type == ExecutionEventType.AGENT_START and e.author == "LocalTarget"
    ]
    assert local_starts, "in-scope relayed AGENT_CALL must transition into the local agent"
    assert executed == ["ran"], "in-scope local agent body must run"
    assert "denied" not in caplog.text, "an in-scope handoff must not log a denial"


@pytest.mark.asyncio
async def test_relayed_handoff_to_unknown_agent_is_silent(caplog):
    """An unknown relayed target is handled on the remote side — no transition, no denial log."""
    parent, executed = _build_tree(HandoffScope.GLOBAL, relayed_target="GhostAgent")
    with caplog.at_level(logging.WARNING, logger="domyn.agents"):
        events = await _run(parent, "conv-relay-unknown")

    transitions = [e for e in events if e.event_type == ExecutionEventType.AGENT_START]
    assert all(e.author != "GhostAgent" for e in transitions), (
        "an unknown relayed target must not transition locally"
    )
    assert executed == [], "no local agent body must run for an unknown remote target"
    assert "denied" not in caplog.text, (
        "an unknown remote target is not a local denial and must not be audited"
    )
