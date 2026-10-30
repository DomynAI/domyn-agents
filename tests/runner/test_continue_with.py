"""Tests for HookResult.continue_with — post_agent_call hooks that keep a turn going.

The primitive exists so "you are not done, keep going" policies (completion
verifiers, constrained retries, quality gates) are hooks rather than core
features. These tests pin the contract, above all the limits the runner — not the
hook — enforces.

Agent execution is stubbed by replacing ``agent._execute`` (the pattern used by the
other runner flow tests), so the tests exercise the runner's transition logic
rather than a planner or a provider.
"""

from collections.abc import AsyncGenerator
from inspect import Parameter, Signature
from typing import Any
from unittest.mock import AsyncMock

import pytest

from domyn.agents import BaseEvent, ExecutionEventType, Part, Runner
from domyn.agents.agents import Agent
from domyn.agents.hooks import FunctionHook, HookPhase, HookType
from domyn.agents.llm import BaseLLMProvider

_PARAMS = [
    Parameter("agent", Parameter.POSITIONAL_OR_KEYWORD),
    Parameter("context", Parameter.POSITIONAL_OR_KEYWORD),
]


def _hook(
    name: str, side_effect: Any, phase: HookPhase = HookPhase.POST_AGENT_CALL
) -> FunctionHook:
    fun = AsyncMock(side_effect=side_effect)
    fun.__signature__ = Signature(_PARAMS)
    return FunctionHook(name=name, fun=fun, trigger=phase, type=HookType.UPDATER)


def _ending_execute() -> Any:
    """Stub agent execution: one answer, then AGENT_END. Re-runnable.

    Unlike the single-shot helper in test_flows, this restarts on every call — a
    continued turn executes the agent again.
    """

    async def _gen(*args: Any, **kwargs: Any) -> AsyncGenerator[BaseEvent, None]:
        yield BaseEvent(
            event_type=ExecutionEventType.RESPONSE,
            author="MainAgent",
            content=[Part(text="partial answer")],
            is_partial=True,
        )
        yield BaseEvent(event_type=ExecutionEventType.AGENT_END, author="MainAgent", content=[])

    return _gen


def _agent(provider: BaseLLMProvider, hooks: list[FunctionHook]) -> Agent:
    agent = Agent(
        name="MainAgent",
        description="main agent",
        instruction="main instruction",
        sub_agents=[],
        llm_provider=provider,
        hooks=hooks,
    )
    agent._execute = _ending_execute()
    return agent


def _continue_event(text: str = "again") -> BaseEvent:
    return BaseEvent(
        event_type=ExecutionEventType.USER_INPUT,
        author="verifier",
        content=[Part(text=text)],
    )


async def _run(
    runner: Runner, agent: Agent, text: str = "hello", conv: str = "c1"
) -> list[BaseEvent]:
    return [
        event
        async for event in runner.run(
            application_name="test",
            user_id="u",
            user_input=BaseEvent(
                event_type=ExecutionEventType.USER_INPUT,
                author="user",
                content=[Part(text=text)],
            ),
            root=agent,
            conversation_id=conv,
        )
    ]


def _continuations(events: list[BaseEvent]) -> list[BaseEvent]:
    return [e for e in events if e.author == "verifier"]


class TestUnusedIsInvisible:
    @pytest.mark.asyncio
    async def test_hook_not_using_it_changes_nothing(self, llm_provider: BaseLLMProvider) -> None:
        """With no hook using continue_with, behaviour is identical to before."""
        agent = _agent(
            llm_provider, [_hook("Plain", lambda agent, context: {"new_context": context})]
        )
        events = await _run(Runner(), agent)
        assert _continuations(events) == []
        assert any(e.event_type is ExecutionEventType.AGENT_END for e in events)


class TestContinuation:
    @pytest.mark.asyncio
    async def test_reenters_the_agent_once(self, llm_provider: BaseLLMProvider) -> None:
        seen = {"n": 0}

        def once(agent: Any, context: Any) -> dict[str, Any]:
            seen["n"] += 1
            if seen["n"] == 1:
                return {"new_context": context, "continue_with": _continue_event()}
            return {"new_context": context}

        agent = _agent(llm_provider, [_hook("Once", once)])
        events = await _run(Runner(), agent)

        assert len(_continuations(events)) == 1
        # The post hook ran again after the continuation, proving the agent was
        # actually re-entered rather than the event merely being emitted.
        assert seen["n"] == 2

    @pytest.mark.asyncio
    async def test_reason_and_metadata_reach_history(self, llm_provider: BaseLLMProvider) -> None:
        """The model must be able to read *why* it is continuing."""
        state = {"done": False}

        def once(agent: Any, context: Any) -> dict[str, Any]:
            if state["done"]:
                return {"new_context": context}
            state["done"] = True
            return {
                "new_context": context,
                "continue_with": _continue_event("you announced work you did not do"),
            }

        agent = _agent(llm_provider, [_hook("Reason", once)])
        events = await _run(Runner(), agent)

        first = _continuations(events)[0]
        assert "did not do" in " ".join(p.text or "" for p in first.content)
        assert first.metadata.get("continuation") is True
        assert first.metadata.get("continuation_index") == 1
        assert first.metadata.get("requested_by")


class TestRunnerEnforcedLimits:
    """The cap belongs to the runner: a hook cannot be its own limit."""

    @staticmethod
    def _always(agent: Any, context: Any) -> dict[str, Any]:
        return {"new_context": context, "continue_with": _continue_event()}

    @pytest.mark.asyncio
    async def test_always_continuing_hook_is_capped(self, llm_provider: BaseLLMProvider) -> None:
        agent = _agent(llm_provider, [_hook("Always", self._always)])
        events = await _run(Runner(max_turn_continuations=2), agent)
        assert len(_continuations(events)) == 2

    @pytest.mark.asyncio
    async def test_cap_is_reported_not_silent(self, llm_provider: BaseLLMProvider) -> None:
        """A turn that quietly stopped continuing looks identical to one that
        finished, so the cap must surface as an event."""
        agent = _agent(llm_provider, [_hook("Always", self._always)])
        events = await _run(Runner(max_turn_continuations=1), agent)
        assert any(e.error_code == "ContinuationCapReached" for e in events)

    @pytest.mark.asyncio
    async def test_break_execution_wins(self, llm_provider: BaseLLMProvider) -> None:
        """Two contradictory instructions from one hook resolve to stopping."""

        def both(agent: Any, context: Any) -> dict[str, Any]:
            return {
                "new_context": context,
                "break_execution": True,
                "continue_with": _continue_event(),
            }

        agent = _agent(llm_provider, [_hook("Both", both)])
        events = await _run(Runner(), agent)
        assert _continuations(events) == []

    @pytest.mark.asyncio
    async def test_ignored_outside_post_agent_call(self, llm_provider: BaseLLMProvider) -> None:
        """Only post_agent_call may continue a turn; elsewhere it is meaningless."""
        agent = _agent(llm_provider, [_hook("Pre", self._always, HookPhase.PRE_AGENT_CALL)])
        events = await _run(Runner(), agent)
        assert _continuations(events) == []

    @pytest.mark.asyncio
    async def test_allowance_is_per_turn_not_per_agent(self, llm_provider: BaseLLMProvider) -> None:
        """One Agent instance is shared across conversations, so the counter cannot
        live on it: the second caller would inherit an exhausted allowance."""
        agent = _agent(llm_provider, [_hook("Always", self._always)])
        runner = Runner(max_turn_continuations=1)

        first = await _run(runner, agent, "first", conv="conv-a")
        second = await _run(runner, agent, "second", conv="conv-b")

        assert len(_continuations(first)) == 1
        assert len(_continuations(second)) == 1
        assert not hasattr(agent, "continue_count")
