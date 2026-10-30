"""Mid-turn steering: a message that arrives while the agent is already working.

Steering is ``push_parts`` with ``event_type=USER_INPUT`` and ``timing=SCHEDULED``
— there is no separate method, and ``_steer`` below is the two arguments spelled
out. The turn takes the message in at its next tool-cycle boundary, or at the
answer it was about to end on, which it then continues rather than ending.

Three properties, and between them most of what is tested here:

- **Scheduled, not racing.** The message lands where the planner is about to
  rebuild its messages. ``TestTheTwoTimings`` pins what the other timing does
  instead, and ``TestAPartialAnswerIsNotABoundary`` the one point that looks like a
  boundary and is not.
- **Delivered, or said not to be.** With no turn running there is nowhere to put it
  and nothing is written on the caller's behalf (``TestWhenNoTurnIsRunning``). A turn
  stops accepting the moment it knows it is ending rather than at teardown, so
  "accepted" never means "accepted and then discarded"
  (``TestTheTurnStopsAcceptingOnceItIsEnding``).
- **Not taking a message is not consuming it.** A message a sub-agent may not see
  waits for the parent. That is the one place this could silently lose one:
  ``test_a_message_a_sub_agent_may_not_see_keeps_waiting``.

No sleeps anywhere: the tool being executed is what steers, so the message provably
arrives while the turn is in flight.
"""

# No `from __future__ import annotations`: builds real agents and tools.
import asyncio
from typing import Any

import pytest
from pydantic import Field

from domyn.agents import Agent
from domyn.agents.conversation_manager import ConversationInfo, InMemoryConversationManager
from domyn.agents.core import (
    METADATA_EXTERNAL_PART_AGENT_VISIBLE,
    METADATA_STEERING,
    BaseEvent,
    ExecutionEventType,
    InteractionContext,
    Part,
)
from domyn.agents.llm import (
    BaseLLMProvider,
    LLMMessage,
    LLMResponse,
    LLMToolCall,
    ToolCallFunction,
)
from domyn.agents.planner_strategy import PlannerStrategyFactory
from domyn.agents.runner import (
    ContextBuilder,
    InjectionsQueue,
    PushTiming,
    Runner,
    RunnerState,
)

APP = "steering_app"
USER = "user_1"
STEER = "actually, only the python files"

PLANNERS = ["tool_use", "tool_use_thought"]


class _ToolThenAnswer(BaseLLMProvider):
    """Calls ``look`` ``rounds`` times, then answers, recording what it was shown.

    ``seen[n]`` is the rendered content of every message of the n-th call, which is
    how a test asks "did the model see this, and on which call".
    """

    type: str = "tool-then-answer"
    name: str = "tool-then-answer"
    model_name: str = "tool-then-answer"
    supported_structured_outputs: list[Any] | None = None
    rounds: int = 2
    calls: int = 0
    seen: list[list[str]] = Field(default_factory=list)
    shown: list[list[LLMMessage]] = Field(default_factory=list)

    async def generate(self, messages: list[LLMMessage], **kwargs: Any) -> LLMResponse:
        self.calls += 1
        self.seen.append([str(getattr(m, "content", "") or "") for m in messages])
        self.shown.append(list(messages))
        if self.calls <= self.rounds:
            return LLMResponse(
                content="",
                generated_tokens=1,
                execution_time=0.0,
                usage={},
                tool_calls=[
                    LLMToolCall(
                        id=f"call_{self.calls}",
                        type="function",
                        function=ToolCallFunction(name="look", arguments='{"path": "a.txt"}'),
                    )
                ],
            )
        return LLMResponse(content="Done.", generated_tokens=1, execution_time=0.0, usage={})

    async def generate_streaming(self, messages: list[LLMMessage], **kwargs: Any) -> Any:
        raise NotImplementedError
        yield  # pragma: no cover

    def saw(self, call: int, text: str) -> bool:
        return any(text in content for content in self.seen[call])

    def occurrences(self, call: int, text: str) -> int:
        return sum(content.count(text) for content in self.seen[call])


async def _steer(runner: Runner, conversation_id: str, text: str = STEER, **metadata: Any) -> Any:
    """Steering, spelled out: a user message held for the turn's next boundary.

    There is no `steer()` method — this is all it ever was, and writing it here
    keeps the two arguments that make it steering visible in the tests rather than
    hidden behind a wrapper.
    """
    return await runner.push_parts(
        APP,
        conversation_id,
        [Part(text=text)],
        author="user",
        event_type=ExecutionEventType.USER_INPUT,
        timing=PushTiming.SCHEDULED,
        metadata=metadata or None,
    )


def _agent(provider: BaseLLMProvider, tool: Any, planner: str = "tool_use") -> Agent:
    return Agent(
        name="MainAgent",
        description="the agent the user is talking to",
        instruction="do as asked",
        llm_provider=provider,
        planner=PlannerStrategyFactory.from_config({"type": planner}, save_in_registry=False),
        tools=[tool],
    )


def _steering_tool(steerer: Runner, conversation_id: str, times: int = 1, **kwargs: Any) -> Any:
    """A ``look`` tool that steers the turn it is running inside, ``times`` times."""
    from domyn.agents.tools import FunctionTool

    remaining = {"n": times}

    async def look(path: str) -> str:
        """Look at a path.

        Args:
            path: what to look at.
        """
        if remaining["n"] > 0:
            remaining["n"] -= 1
            await _steer(steerer, conversation_id, **kwargs)
        return "looked"

    return FunctionTool(func=look)


async def _run(runner: Runner, agent: Agent, conversation_id: str) -> list[BaseEvent]:
    collected: list[BaseEvent] = []
    async for event in runner.run(
        application_name=APP,
        user_id=USER,
        conversation_id=conversation_id,
        root=agent,
        user_input=BaseEvent(
            event_type=ExecutionEventType.USER_INPUT,
            author="user",
            content=[Part(text="read every file")],
        ),
    ):
        collected.append(event)
    return collected


class TestAMessageThatArrivesMidTurn:
    @pytest.mark.parametrize("planner", PLANNERS)
    @pytest.mark.asyncio
    async def test_the_next_llm_call_sees_it(self, planner: str) -> None:
        """The whole feature. Also parametrized over both planners because
        ``ToolUseWithThoughtPlannerStrategy.execute`` is a copy of its parent's loop
        rather than an extension, and has shipped a feature missing from one copy
        before. The absorb lives in the runner, so this should pass for free — this
        is what says so."""
        runner = Runner()
        conversation_id = f"c_next_call_{planner}"
        provider = _ToolThenAnswer(rounds=2)
        agent = _agent(provider, _steering_tool(runner, conversation_id), planner)

        await _run(runner, agent, conversation_id)

        assert provider.calls >= 2, "the fixture produced no second call to steer into"
        assert not provider.saw(0, STEER), "it cannot have been seen before it was sent"
        assert provider.saw(1, STEER)

    @pytest.mark.asyncio
    async def test_it_is_yielded_to_the_consumer(self) -> None:
        """Without this a steered message is invisible until it changes what the
        agent does, which is indistinguishable from it having been lost."""
        runner = Runner()
        conversation_id = "c_yielded"
        provider = _ToolThenAnswer(rounds=2)
        agent = _agent(provider, _steering_tool(runner, conversation_id))

        events = await _run(runner, agent, conversation_id)

        steered = [
            event
            for event in events
            if event.event_type is ExecutionEventType.USER_INPUT
            and any(part.text == STEER for part in event.content)
        ]
        assert len(steered) == 1

    @pytest.mark.asyncio
    async def test_it_is_absorbed_once(self) -> None:
        """Three tool cycles, one message. Every cycle after the first re-reads it
        from the store and has to recognise it as already taken in."""
        runner = Runner()
        conversation_id = "c_once"
        provider = _ToolThenAnswer(rounds=3)
        agent = _agent(provider, _steering_tool(runner, conversation_id, times=1))

        events = await _run(runner, agent, conversation_id)

        assert (
            sum(
                1
                for event in events
                if event.event_type is ExecutionEventType.USER_INPUT
                and any(part.text == STEER for part in event.content)
            )
            == 1
        )
        assert provider.occurrences(provider.calls - 1, STEER) == 1

    @pytest.mark.asyncio
    async def test_a_turn_nobody_steers_is_unchanged(self) -> None:
        """The feature has to be invisible when unused."""
        from domyn.agents.tools import FunctionTool

        async def look(path: str) -> str:
            """Look at a path.

            Args:
                path: what to look at.
            """
            return "looked"

        runner = Runner()
        provider = _ToolThenAnswer(rounds=2)
        agent = _agent(provider, FunctionTool(func=look))

        events = await _run(runner, agent, "c_quiet")

        assert not any(event.event_type is ExecutionEventType.USER_INPUT for event in events)
        assert not any(provider.saw(call, STEER) for call in range(provider.calls))


class TestTheTriggersAreASeam:
    """``steering_triggers`` is a class attribute so a deployment can move the
    check, not decoration — these assert it is actually consulted."""

    @pytest.mark.asyncio
    async def test_emptying_it_switches_the_feature_off(self) -> None:
        runner = Runner()
        runner.steering_triggers = ()
        conversation_id = "c_no_triggers"
        provider = _ToolThenAnswer(rounds=2)
        agent = _agent(provider, _steering_tool(runner, conversation_id))

        events = await _run(runner, agent, conversation_id)

        assert not provider.saw(1, STEER)
        assert not any(event.event_type is ExecutionEventType.USER_INPUT for event in events)

    @pytest.mark.asyncio
    async def test_a_subclass_can_narrow_it(self) -> None:
        """Dropping TOOL_ERROR alone must still leave the ordinary path working, or
        the attribute is all-or-nothing and not worth having."""

        class _OnlyToolEnd(Runner):
            steering_triggers = (ExecutionEventType.TOOL_END,)

        runner = _OnlyToolEnd()
        conversation_id = "c_narrow"
        provider = _ToolThenAnswer(rounds=2)
        agent = _agent(provider, _steering_tool(runner, conversation_id))

        await _run(runner, agent, conversation_id)

        assert provider.saw(1, STEER)


class TestAnotherRunnerHasNoTurnToSteer:
    @pytest.mark.asyncio
    async def test_a_second_runner_is_told_there_is_nothing_to_steer(self) -> None:
        """Delivery is to a turn this runner is executing, and to nothing else.

        Two ``Runner`` objects sharing a conversation store — the stand-in for two
        replicas. The message travels through the running turn, so the second runner
        has nowhere to put it and says so rather than writing something the first
        will never read. Synchronizing the queues across replicas is what would make
        this work.
        """
        store = InMemoryConversationManager()
        running = Runner(conversation_manager=store)
        elsewhere = Runner(conversation_manager=store)
        conversation_id = "c_two_runners"

        results: list[Any] = []

        from domyn.agents.tools import FunctionTool

        async def look(path: str) -> str:
            """Look at a path.

            Args:
                path: what to look at.
            """
            results.append(await _steer(elsewhere, conversation_id))
            return "looked"

        provider = _ToolThenAnswer(rounds=2)
        await _run(running, _agent(provider, FunctionTool(func=look)), conversation_id)

        assert results and all(not result.delivered for result in results)
        assert not provider.saw(1, STEER)


class TestWhenNoTurnIsRunning:
    @pytest.mark.asyncio
    async def test_it_reports_that_nothing_took_the_message(self) -> None:
        """``steer()`` delivers to a turn in flight, and promises nothing else.

        With none running there is nowhere to put the message, so it says so. It is
        deliberately *not* written to the conversation on the caller's behalf: a
        runner with no turn to feed has no business deciding what a conversation
        contains, and the caller — which knows why it was steering — owns whether to
        hold the message, retry it, or drop it.
        """
        store = InMemoryConversationManager()
        runner = Runner(conversation_manager=store)
        conversation_id = "c_waiting"
        await store.create_conversation(APP, USER, conversation_id)

        result = await _steer(runner, conversation_id)
        assert result.delivered is False

        conversation = await store.get_conversation(APP, USER, conversation_id)
        assert not [
            event
            for event in conversation.events_history
            if any(part.text == STEER for part in event.content)
        ], "nothing should have been written"

    @pytest.mark.asyncio
    async def test_the_next_turn_does_not_inherit_it(self) -> None:
        """The other half of the promise above, stated from the model's side."""
        store = InMemoryConversationManager()
        runner = Runner(conversation_manager=store)
        conversation_id = "c_not_inherited"
        await store.create_conversation(APP, USER, conversation_id)

        await _steer(runner, conversation_id)

        from domyn.agents.tools import FunctionTool

        async def look(path: str) -> str:
            """Look at a path.

            Args:
                path: what to look at.
            """
            return "looked"

        provider = _ToolThenAnswer(rounds=1)
        await _run(runner, _agent(provider, FunctionTool(func=look)), conversation_id)

        assert not provider.saw(0, STEER)

    @pytest.mark.asyncio
    async def test_an_unknown_conversation_is_reported_not_raised(self) -> None:
        result = await _steer(Runner(), "never_created")
        assert result.delivered is False


def _bare_agent(name: str) -> Agent:
    class _Silent(BaseLLMProvider):
        type: str = "silent"
        name: str = "silent"
        model_name: str = "silent"
        supported_structured_outputs: None = None

        async def generate(self, messages: Any, **kwargs: Any) -> Any:  # pragma: no cover
            raise NotImplementedError

        async def generate_streaming(self, messages: Any, **kwargs: Any) -> Any:  # pragma: no cover
            raise NotImplementedError
            yield

    return Agent(name=name, description="d", instruction="i", llm_provider=_Silent())


def _state(events_history: list[BaseEvent], current: Agent, entrypoint: Agent, interaction_id: str):
    return RunnerState(
        application_name=APP,
        user_id=USER,
        conversation_id="c_sub",
        conversation_info=ConversationInfo(
            application_name=APP,
            user_id=USER,
            conversation_id="c_sub",
            events_history=events_history,
        ),
        current_agent=current,
        available_agents=[current, entrypoint],
        entrypoint_agent=entrypoint,
        interaction_id=interaction_id,
    )


class TestSubAgentsOnlySeeItWhenTheDeploymentSaysSo:
    """Two enforcement points, because one of them alone is only half a rule.

    ``_absorb_pending`` governs a sub-agent that is *already running*: it does not
    look at the queue. A sub-agent that starts *after* the message was taken in
    builds its history from ``ContextBuilder.create_context`` instead, where a
    steering event carries the turn's own ``interaction_id`` and would otherwise walk
    straight in.

    Both read ``steering_reaches_subagents``, which is a property of the deployment
    rather than of the message — the queue is a plain FIFO precisely because nothing
    has to be decided per event.
    """

    async def _absorb(
        self, *, is_subagent: bool, reaches_subagents: bool = False
    ) -> tuple[Runner, list[str]]:
        """Steer into a turn, then absorb as the given kind of agent.

        An ``InjectionsQueue`` is registered by hand: without a turn to hand the
        message to there is nothing to steer and nothing to absorb.
        """
        store = InMemoryConversationManager()
        runner = Runner(conversation_manager=store)
        runner.steering_reaches_subagents = reaches_subagents
        await store.create_conversation(APP, USER, "c_sub")

        agent = _bare_agent("MainAgent")
        state = _state([], agent, agent, "i1")
        state.steering_reaches_subagents = reaches_subagents
        runner.injection_queue[(APP, "c_sub")] = InjectionsQueue(
            racing_queue=asyncio.Queue(),
            scheduled_queue=asyncio.Queue(),
            default_author=agent.name,
        )

        await _steer(runner, "c_sub")

        context = InteractionContext(
            conversation_id="c_sub",
            user_id=USER,
            agent_input=BaseEvent(event_type=ExecutionEventType.USER_INPUT, author="user"),
            is_subagent=is_subagent,
        )
        absorbed = await runner._absorb_pending(state, context)
        return runner, [part.text for event in absorbed for part in event.content if part.text]

    async def _absorbed(self, *, is_subagent: bool, reaches_subagents: bool = False) -> list[str]:
        _, texts = await self._absorb(is_subagent=is_subagent, reaches_subagents=reaches_subagents)
        return texts

    @pytest.mark.asyncio
    async def test_the_user_facing_agent_absorbs_it(self) -> None:
        assert await self._absorbed(is_subagent=False) == [STEER]

    @pytest.mark.asyncio
    async def test_a_sub_agent_does_not_by_default(self) -> None:
        assert await self._absorbed(is_subagent=True) == []

    @pytest.mark.asyncio
    async def test_the_setting_lets_it_through(self) -> None:
        assert await self._absorbed(is_subagent=True, reaches_subagents=True) == [STEER]

    @pytest.mark.asyncio
    async def test_a_message_a_sub_agent_may_not_see_keeps_waiting(self) -> None:
        """Declining to look must not mean consuming.

        Nothing else holds a copy — the message was never written anywhere on the way
        in — so a sub-agent that drained the queue and discarded what it may not have
        would lose it outright. It does not drain it at all, and the parent takes it
        in when control returns.
        """
        runner, absorbed = await self._absorb(is_subagent=True)
        assert absorbed == [], "the sub-agent must not take it"

        injections = runner.injection_queue[(APP, "c_sub")]
        assert injections.scheduled_queue.qsize() == 1, "it is still queued, not dropped"

        agent = _bare_agent("MainAgent")
        parent = InteractionContext(
            conversation_id="c_sub",
            user_id=USER,
            agent_input=BaseEvent(event_type=ExecutionEventType.USER_INPUT, author="user"),
            is_subagent=False,
        )
        taken = await runner._absorb_pending(_state([], agent, agent, "i1"), parent)
        assert [part.text for event in taken for part in event.content] == [STEER]
        assert injections.scheduled_queue.empty(), "and only then is it consumed"

    @staticmethod
    def _late_child_sees(reaches_subagents: bool) -> list[str]:
        """What a sub-agent started *after* the message finds in its history."""
        opening = BaseEvent(
            event_type=ExecutionEventType.USER_INPUT,
            author="user",
            content=[Part(text="read every file")],
            interaction_id="i1",
        )
        steering = BaseEvent(
            event_type=ExecutionEventType.USER_INPUT,
            author="user",
            content=[Part(text=STEER)],
            interaction_id="i1",
            metadata={METADATA_STEERING: True},
        )
        state = _state(
            [opening, steering], _bare_agent("ChildAgent"), _bare_agent("MainAgent"), "i1"
        )
        state.steering_reaches_subagents = reaches_subagents
        context = ContextBuilder.create_context(state=state, user_input=opening)
        return [part.text for event in context.conversation_history for part in event.content]

    @pytest.mark.asyncio
    async def test_a_sub_agent_started_afterwards_also_does_not_see_it(self) -> None:
        texts = self._late_child_sees(reaches_subagents=False)
        assert "read every file" in texts, "the turn's own question must still reach it"
        assert STEER not in texts

    @pytest.mark.asyncio
    async def test_the_setting_reaches_a_sub_agent_started_afterwards_too(self) -> None:
        assert STEER in self._late_child_sees(reaches_subagents=True)


class _SteerWhileAnswering(_ToolThenAnswer):
    """Steers from inside the model call that is about to produce the answer.

    The window the tool-based helper above cannot reach: past the last tool call
    there is no boundary left, so the message has to be written while the final
    generation is in flight. Doing it here is what makes that exact — the planner
    has already built the messages for this call, so the steer provably lands
    after them and before the turn can end. Still no sleeps.

    ``steer_now`` is an async callable rather than a ``Runner`` because the field
    is typed ``Any``: a pydantic model cannot hold a ``Runner`` without
    ``arbitrary_types_allowed``.
    """

    steer_now: Any = None
    steers: int = 1
    probe: Any = None
    probed: list[Any] = Field(default_factory=list)

    async def generate(self, messages: list[LLMMessage], **kwargs: Any) -> LLMResponse:
        if self.calls + 1 > self.rounds and self.steers > 0 and self.steer_now is not None:
            self.steers -= 1
            await self.steer_now()
        response = await super().generate(messages, **kwargs)
        if self.probe is not None:
            self.probed.append(await self.probe())
        return response


def _plain_look_tool() -> Any:
    from domyn.agents.tools import FunctionTool

    async def look(path: str) -> str:
        """Look at a path.

        Args:
            path: what to look at.
        """
        return "looked"

    return FunctionTool(func=look)


class TestAMessageThatArrivesDuringTheFinalCall:
    """The window with no boundary after it: the turn continues instead of ending."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("planner", PLANNERS)
    async def test_the_turn_continues_and_the_next_call_sees_it(self, planner: str) -> None:
        """`tool_use_thought.execute` is a copy of its parent's loop, so both."""
        runner = Runner()
        conversation_id = f"c_final_call_{planner}"
        provider = _SteerWhileAnswering(rounds=1)
        provider.steer_now = lambda: _steer(runner, conversation_id)
        agent = _agent(provider, _plain_look_tool(), planner=planner)

        await _run(runner, agent, conversation_id)

        # 1: the tool call. 2: the answer, during which the message arrives.
        # 3: the call that only exists because the turn did not end.
        assert provider.calls == 3
        assert provider.saw(2, STEER)

    @pytest.mark.asyncio
    async def test_the_answer_it_interrupted_is_still_in_history(self) -> None:
        """Nothing is rewound: the model reads its own answer, then the correction."""
        runner = Runner()
        conversation_id = "c_final_order"
        provider = _SteerWhileAnswering(rounds=1)
        provider.steer_now = lambda: _steer(runner, conversation_id)
        agent = _agent(provider, _plain_look_tool())

        await _run(runner, agent, conversation_id)

        last = provider.seen[2]
        assert any("Done." in content for content in last)
        answer_at = next(i for i, content in enumerate(last) if "Done." in content)
        steer_at = next(i for i, content in enumerate(last) if STEER in content)
        assert answer_at < steer_at

    @pytest.mark.asyncio
    async def test_the_answer_reaches_the_user_before_the_echo(self) -> None:
        """Order on the wire: the answer, then the message that follows it, then
        the answer to that. Reversed, the transcript reads as a reply to nothing."""
        runner = Runner()
        conversation_id = "c_final_wire_order"
        provider = _SteerWhileAnswering(rounds=1)
        provider.steer_now = lambda: _steer(runner, conversation_id)
        agent = _agent(provider, _plain_look_tool())

        events = await _run(runner, agent, conversation_id)
        kinds = [event.event_type for event in events]

        first_answer = kinds.index(ExecutionEventType.RESPONSE)
        echo = kinds.index(ExecutionEventType.USER_INPUT)
        second_answer = kinds.index(ExecutionEventType.RESPONSE, echo)
        assert first_answer < echo < second_answer

    @pytest.mark.asyncio
    async def test_it_keeps_going_while_messages_keep_arriving(self) -> None:
        runner = Runner()
        conversation_id = "c_final_twice"
        provider = _SteerWhileAnswering(rounds=1, steers=2)
        provider.steer_now = lambda: _steer(runner, conversation_id)
        agent = _agent(provider, _plain_look_tool())

        events = await _run(runner, agent, conversation_id)

        # tool, answer+steer, answer+steer, answer with nothing pending.
        assert provider.calls == 4
        assert sum(1 for e in events if e.event_type is ExecutionEventType.USER_INPUT) == 2

    @pytest.mark.asyncio
    async def test_it_is_absorbed_once(self) -> None:
        runner = Runner()
        conversation_id = "c_final_once"
        provider = _SteerWhileAnswering(rounds=1)
        provider.steer_now = lambda: _steer(runner, conversation_id)
        agent = _agent(provider, _plain_look_tool())

        events = await _run(runner, agent, conversation_id)

        assert sum(1 for e in events if e.event_type is ExecutionEventType.USER_INPUT) == 1
        assert provider.occurrences(provider.calls - 1, STEER) == 1

    @pytest.mark.asyncio
    async def test_the_turn_ends_once(self) -> None:
        """One AGENT_END, and it is last: the intermediate answer must not look
        like the end of the turn to anything downstream."""
        runner = Runner()
        conversation_id = "c_final_one_end"
        provider = _SteerWhileAnswering(rounds=1)
        provider.steer_now = lambda: _steer(runner, conversation_id)
        agent = _agent(provider, _plain_look_tool())

        events = await _run(runner, agent, conversation_id)
        ends = [i for i, e in enumerate(events) if e.event_type is ExecutionEventType.AGENT_END]

        assert len(ends) == 1
        assert ends[0] == len(events) - 1

    @pytest.mark.asyncio
    async def test_the_post_hooks_fire_once_for_the_whole_turn(self) -> None:
        """The agent has not ended at the intermediate answer, so POST_AGENT_CALL
        must not fire there — twice would double-write anything that remembers."""
        from inspect import Parameter, Signature
        from unittest.mock import AsyncMock

        from domyn.agents.hooks import FunctionHook, HookPhase, HookType

        calls = {"n": 0}

        async def count(agent: Any, context: Any) -> Any:
            calls["n"] += 1
            return None

        fun = AsyncMock(side_effect=count)
        fun.__signature__ = Signature(
            [
                Parameter("agent", Parameter.POSITIONAL_OR_KEYWORD),
                Parameter("context", Parameter.POSITIONAL_OR_KEYWORD),
            ]
        )
        hook = FunctionHook(
            name="Counter", fun=fun, trigger=HookPhase.POST_AGENT_CALL, type=HookType.UPDATER
        )

        runner = Runner()
        conversation_id = "c_final_post_hooks"
        provider = _SteerWhileAnswering(rounds=1)
        provider.steer_now = lambda: _steer(runner, conversation_id)
        agent = _agent(provider, _plain_look_tool())
        agent.hooks = [hook]

        await _run(runner, agent, conversation_id)

        assert provider.calls == 3, "the turn did not continue, so this proves nothing"
        assert calls["n"] == 1

    @pytest.mark.asyncio
    async def test_the_agents_stack_is_intact_while_it_continues(self) -> None:
        """The root's stack entry is what a later sub-agent return pops back to, so
        it cannot be cleared on an answer the turn goes on past."""
        store = InMemoryConversationManager()
        runner = Runner(conversation_manager=store)
        conversation_id = "c_final_stack"
        provider = _SteerWhileAnswering(rounds=1)
        provider.steer_now = lambda: _steer(runner, conversation_id)
        agent = _agent(provider, _plain_look_tool())

        async def stack() -> Any:
            return await store.get_last_agent_stack(
                application_name=APP, user_id=USER, conversation_id=conversation_id
            )

        provider.probe = stack

        await _run(runner, agent, conversation_id)

        # Third call: the continuation. The entry must still be there.
        assert provider.probed[2] is not None
        assert provider.probed[2].agent_name == "MainAgent"
        # And cleared by the time the turn is really over.
        assert (
            await store.get_last_agent_stack(
                application_name=APP, user_id=USER, conversation_id=conversation_id
            )
            is None
        )

    @pytest.mark.asyncio
    async def test_a_quiet_final_call_ends_the_turn_as_before(self) -> None:
        runner = Runner()
        provider = _ToolThenAnswer(rounds=1)
        agent = _agent(provider, _plain_look_tool())

        events = await _run(runner, agent, "c_final_quiet")

        assert provider.calls == 2
        assert not any(e.event_type is ExecutionEventType.USER_INPUT for e in events)
        assert events[-1].event_type is ExecutionEventType.AGENT_END


class TestTheResponseTriggerIsASeamToo:
    @pytest.mark.asyncio
    async def test_it_is_on_by_default(self) -> None:
        assert ExecutionEventType.RESPONSE in Runner.steering_triggers

    @pytest.mark.asyncio
    async def test_dropping_it_ends_the_turn_on_the_answer(self) -> None:
        """Without RESPONSE the message waits in the store for the next turn,
        which is the behaviour this feature replaces."""

        class _NoResponseTrigger(Runner):
            steering_triggers = (
                ExecutionEventType.TOOL_END,
                ExecutionEventType.TOOL_ERROR,
                ExecutionEventType.AGENT_END,
            )

        runner = _NoResponseTrigger()
        conversation_id = "c_final_off"
        provider = _SteerWhileAnswering(rounds=1)
        provider.steer_now = lambda: _steer(runner, conversation_id)
        agent = _agent(provider, _plain_look_tool())

        events = await _run(runner, agent, conversation_id)

        assert provider.calls == 2
        assert not any(e.event_type is ExecutionEventType.USER_INPUT for e in events)

    @pytest.mark.asyncio
    async def test_dropping_it_leaves_the_tool_boundary_working(self) -> None:
        class _NoResponseTrigger(Runner):
            steering_triggers = (ExecutionEventType.TOOL_END,)

        runner = _NoResponseTrigger()
        conversation_id = "c_final_off_tools"
        provider = _ToolThenAnswer(rounds=2)
        agent = _agent(provider, _steering_tool(runner, conversation_id))

        await _run(runner, agent, conversation_id)

        assert provider.saw(1, STEER)


class TestTheTurnStopsAcceptingOnceItIsEnding:
    @pytest.mark.asyncio
    async def test_a_push_after_the_last_chance_is_told_so(self) -> None:
        """The honesty this design rests on.

        Between deciding nothing is waiting and the turn actually finishing there are
        several suspension points — emptying the agents stack, the answer itself, the
        post-agent hooks, the synthesized AGENT_END. Content arriving in that window
        cannot be taken in by anything, so it has to be reported undelivered rather
        than accepted into a run that will discard it. The turn therefore stops
        accepting at the moment it knows it is ending, not at teardown.
        """
        runner = Runner()
        conversation_id = "c_too_late"
        provider = _ToolThenAnswer(rounds=1)
        agent = _agent(provider, _plain_look_tool())

        results: list[Any] = []

        async def push_on_every_event() -> None:
            async for event in runner.run(
                application_name=APP,
                user_id=USER,
                conversation_id=conversation_id,
                root=agent,
                user_input=BaseEvent(
                    event_type=ExecutionEventType.USER_INPUT,
                    author="user",
                    content=[Part(text="read every file")],
                ),
            ):
                if event.event_type is ExecutionEventType.RESPONSE and not event.is_partial:
                    # The answer is out; the turn is over bar the paperwork.
                    results.append(await _steer(runner, conversation_id))

        await push_on_every_event()

        assert results, "the turn produced no terminal answer to push after"
        assert all(not result.delivered for result in results)
        assert provider.calls == 2, "and nothing re-entered the turn"


class _RecordingStore(InMemoryConversationManager):
    """Logs when each store call starts and finishes, and suspends on the write.

    The suspend makes the write a real round trip rather than something that
    completes without ever yielding to the event loop, which is what a
    database-backed store does.
    """

    def __init__(self) -> None:
        super().__init__()
        self.log: list[str] = []

    async def update_conversation_state(self, *args: Any, **kwargs: Any) -> Any:
        self.log.append("write:start")
        await asyncio.sleep(0)
        result = await super().update_conversation_state(*args, **kwargs)
        self.log.append("write:end")
        return result


class TestABoundaryCostsNoStoreTraffic:
    """A message is handed to the turn, so a boundary that finds nothing waiting
    does no store work at all — which is what makes ``steering_triggers`` free to
    widen."""

    @pytest.mark.asyncio
    async def test_a_quiet_boundary_only_writes(self) -> None:
        """The state write is the only store call a quiet boundary makes."""
        store = _RecordingStore()
        runner = Runner(conversation_manager=store)
        provider = _ToolThenAnswer(rounds=1)
        agent = _agent(provider, _plain_look_tool())

        await _run(runner, agent, "c_no_read")

        assert set(store.log) == {"write:start", "write:end"}, store.log

    @pytest.mark.asyncio
    async def test_the_message_still_lands_at_the_boundary(self) -> None:
        """Losing the read must not cost the feature."""
        store = _RecordingStore()
        runner = Runner(conversation_manager=store)
        conversation_id = "c_no_read_finds"
        provider = _ToolThenAnswer(rounds=2)
        agent = _agent(provider, _steering_tool(runner, conversation_id))

        await _run(runner, agent, conversation_id)

        assert provider.saw(1, STEER)

    @pytest.mark.asyncio
    async def test_a_message_taken_in_is_persisted(self) -> None:
        """Nobody wrote it before it got here — ``steer()`` deliberately does not —
        so the turn that takes it in is what commits it. Without this the message
        would reach the model and then vanish from the conversation."""
        store = InMemoryConversationManager()
        runner = Runner(conversation_manager=store)
        conversation_id = "c_persisted"
        provider = _ToolThenAnswer(rounds=1)
        agent = _agent(provider, _steering_tool(runner, conversation_id))

        await _run(runner, agent, conversation_id)

        conversation = await store.get_conversation(APP, USER, conversation_id)
        steered = [
            event
            for event in conversation.events_history
            if any(part.text == STEER for part in event.content)
        ]
        assert len(steered) == 1
        assert steered[0].metadata.get(METADATA_STEERING) is True

    @staticmethod
    def _state(conversation_id: str) -> RunnerState:
        return RunnerState(
            application_name=APP,
            user_id=USER,
            conversation_id=conversation_id,
            conversation_info=ConversationInfo(
                application_name=APP, user_id=USER, conversation_id=conversation_id
            ),
            current_agent=_bare_agent("MainAgent"),
            available_agents=[],
            entrypoint_agent=_bare_agent("MainAgent"),
        )

    @pytest.mark.asyncio
    async def test_the_state_write_is_just_a_write(self) -> None:
        """The only store traffic a boundary makes on its own."""
        store = _RecordingStore()
        runner = Runner(conversation_manager=store)
        context = InteractionContext(
            conversation_id="c_write_only",
            user_id=USER,
            agent_input=BaseEvent(event_type=ExecutionEventType.USER_INPUT, author="user"),
        )

        assert await runner._write_conversation_state(self._state("c_write_only"), context) is None
        assert store.log == ["write:start", "write:end"]

    @pytest.mark.asyncio
    async def test_a_blip_on_that_write_is_swallowed(self) -> None:
        """State that did not land is re-derived next turn, so this one write is
        allowed to fail quietly — a turn that cannot reach its store still has an
        answer to finish. Tested directly rather than through a run, because the
        loop makes *other*, unguarded store writes: a store that is really down
        takes the turn out through those, and always has.
        """

        class _FailingWrite(InMemoryConversationManager):
            async def update_conversation_state(self, *args: Any, **kwargs: Any) -> Any:
                raise RuntimeError("store is down")

        runner = Runner(conversation_manager=_FailingWrite())
        context = InteractionContext(
            conversation_id="c_write_blip",
            user_id=USER,
            agent_input=BaseEvent(event_type=ExecutionEventType.USER_INPUT, author="user"),
        )

        assert await runner._write_conversation_state(self._state("c_write_blip"), context) is None


class _PreambleThenAnswer(_ToolThenAnswer):
    """Returns content *alongside* its tool call, which is what makes a partial.

    With streaming off, the planner has no way to have shown that text already, so
    it emits it as a partial ``RESPONSE`` before the tool events — the only way a
    partial answer reaches the bottom of the runner's loop in practice.
    """

    async def generate(self, messages: list[LLMMessage], **kwargs: Any) -> LLMResponse:
        response = await super().generate(messages, **kwargs)
        if response.tool_calls:
            response.content = "Let me look at that."
        return response


class TestAPartialAnswerIsNotABoundary:
    """A partial is followed by more by definition, so the next boundary feeds the
    same model call and catches a superset of what taking a message in here could."""

    @pytest.mark.asyncio
    async def test_the_preamble_takes_nothing_in(self) -> None:
        """Observable in the ordering: a message taken in at the partial would be
        echoed straight after it, before the tool ever ran."""
        runner = Runner()
        conversation_id = "c_preamble"
        provider = _PreambleThenAnswer(rounds=1)
        agent = _agent(provider, _steering_tool(runner, conversation_id))

        events = await _run(runner, agent, conversation_id)
        kinds = [event.event_type for event in events]

        partials = [
            event
            for event in events
            if event.event_type is ExecutionEventType.RESPONSE and event.is_partial
        ]
        assert partials, "no partial answer was emitted, so this proves nothing"

        echo = next(
            index
            for index, event in enumerate(events)
            if event.event_type is ExecutionEventType.USER_INPUT
            and any(part.text == STEER for part in event.content)
        )
        assert echo > kinds.index(ExecutionEventType.TOOL_END), (
            f"the partial took the message in instead of the tool boundary: {kinds}"
        )

    @pytest.mark.asyncio
    async def test_the_next_boundary_still_catches_the_message(self) -> None:
        """Skipping the partial must not lose anything — the read moments later has
        to find a message that arrived before it."""
        store = _RecordingStore()
        runner = Runner(conversation_manager=store)
        conversation_id = "c_preamble_catch"
        provider = _PreambleThenAnswer(rounds=1)
        agent = _agent(provider, _steering_tool(runner, conversation_id))

        await _run(runner, agent, conversation_id)

        assert provider.saw(provider.calls - 1, STEER)


class TestTheTwoTimings:
    """What each timing does with a user message, which is the whole difference.

    ``SCHEDULED`` is held for the loop, which takes it into the running agent's
    history at the next of ``steering_triggers`` — including the turn's final answer,
    which it then continues rather than ending. ``RACING`` streams and persists it
    and nothing more: it enters no history, so the model reads it on a later turn.
    """

    @staticmethod
    async def _push_racing(runner: Runner, conversation_id: str) -> Any:
        return await runner.push_parts(
            APP,
            conversation_id,
            [Part(text=STEER)],
            author="user",
            event_type=ExecutionEventType.USER_INPUT,
            timing=PushTiming.RACING,
        )

    @pytest.mark.asyncio
    async def test_racing_reaches_the_user_and_the_conversation(self) -> None:
        store = InMemoryConversationManager()
        runner = Runner(conversation_manager=store)
        conversation_id = "c_immediate"

        provider = _SteerWhileAnswering(rounds=1)
        provider.steer_now = lambda: self._push_racing(runner, conversation_id)
        agent = _agent(provider, _plain_look_tool())

        events = await _run(runner, agent, conversation_id)

        assert any(
            event.event_type is ExecutionEventType.USER_INPUT
            and any(part.text == STEER for part in event.content)
            for event in events
        ), "it is streamed to the consumer"
        conversation = await store.get_conversation(APP, USER, conversation_id)
        assert [
            event
            for event in conversation.events_history
            if any(part.text == STEER for part in event.content)
        ], "and persisted"

    @pytest.mark.asyncio
    async def test_racing_does_not_reach_the_model_this_turn(self) -> None:
        runner = Runner()
        conversation_id = "c_immediate_unread"

        provider = _SteerWhileAnswering(rounds=1)
        provider.steer_now = lambda: self._push_racing(runner, conversation_id)
        agent = _agent(provider, _plain_look_tool())

        await _run(runner, agent, conversation_id)

        assert provider.calls == 2, "the turn ends on the answer it was giving"
        assert not any(provider.saw(call, STEER) for call in range(provider.calls))

    @pytest.mark.asyncio
    async def test_scheduled_reaches_the_model_and_continues_the_turn(self) -> None:
        """The same push, one argument different. This is why steering uses it."""
        runner = Runner()
        conversation_id = "c_boundary"

        provider = _SteerWhileAnswering(rounds=1)
        provider.steer_now = lambda: _steer(runner, conversation_id)
        agent = _agent(provider, _plain_look_tool())

        await _run(runner, agent, conversation_id)

        assert provider.calls == 3
        assert provider.saw(2, STEER)

    @pytest.mark.asyncio
    async def test_display_content_is_racing_and_ends_nothing(self) -> None:
        """An EXTERNAL_PART is content to show, so it goes out where it is pushed and
        the turn carries on to the answer it was giving."""
        runner = Runner()
        conversation_id = "c_external_immediate"

        provider = _SteerWhileAnswering(rounds=1)

        async def push_now() -> None:
            await runner.push_parts(
                APP,
                conversation_id,
                [Part(text=STEER)],
                author="weather-tool",
                event_type=ExecutionEventType.EXTERNAL_PART,
                metadata={METADATA_EXTERNAL_PART_AGENT_VISIBLE: True},
            )

        provider.steer_now = push_now
        agent = _agent(provider, _plain_look_tool())

        events = await _run(runner, agent, conversation_id)

        assert provider.calls == 2
        assert any(event.event_type is ExecutionEventType.EXTERNAL_PART for event in events)


class TestTheCallerCanNameTheEvent:
    @pytest.mark.asyncio
    async def test_the_pushed_event_id_is_the_one_that_reaches_the_stream(self) -> None:
        """The id a caller records has to be the id it later sees, or an idempotency
        key is only an idempotency key on one of the two paths."""
        runner = Runner()
        conversation_id = "c_event_id"
        seen: list[BaseEvent] = []

        provider = _SteerWhileAnswering(rounds=1)
        result: list[Any] = []

        async def push_now() -> None:
            result.append(
                await runner.push_parts(
                    APP,
                    conversation_id,
                    [Part(text="from the tool")],
                    author="weather-tool",
                    event_type=ExecutionEventType.EXTERNAL_PART,
                    event_id="event_chosen_by_the_caller",
                )
            )

        provider.steer_now = push_now
        agent = _agent(provider, _plain_look_tool())

        async for event in runner.run(
            application_name=APP,
            user_id=USER,
            conversation_id=conversation_id,
            root=agent,
            user_input=BaseEvent(
                event_type=ExecutionEventType.USER_INPUT,
                author="user",
                content=[Part(text="read every file")],
            ),
        ):
            seen.append(event)

        assert result and result[0].event.event_id == "event_chosen_by_the_caller"
        assert [
            event.event_id for event in seen if event.event_type is ExecutionEventType.EXTERNAL_PART
        ] == ["event_chosen_by_the_caller"]


def _plain_look_tool_gated() -> Any:
    """``look`` behind an approval gate, steered from outside rather than within."""
    from domyn.agents.core import ApprovalType
    from domyn.agents.tools import FunctionTool

    async def look(path: str) -> str:
        """Look at a path.

        Args:
            path: what to look at.
        """
        return "looked"

    return FunctionTool(func=look, approval_type=ApprovalType.USER_CONFIRM)


def _gated_look_tool(steerer: Runner, conversation_id: str, times: int = 1) -> Any:
    """``look``, but behind an approval gate, steering the turn it runs inside.

    The steer happens in the tool body for the same reason as everywhere else here:
    no sleeps, and the message provably arrives while the turn is in flight. The
    gate is what makes this different — the *first* call is suspended before the
    body runs, so the message goes out from the resumed one.
    """
    from domyn.agents.core import ApprovalType
    from domyn.agents.tools import FunctionTool

    remaining = {"n": times}

    async def look(path: str) -> str:
        """Look at a path.

        Args:
            path: what to look at.
        """
        if remaining["n"] > 0:
            remaining["n"] -= 1
            await _steer(steerer, conversation_id)
        return "looked"

    return FunctionTool(func=look, approval_type=ApprovalType.USER_CONFIRM)


async def _run_answering_gates(
    runner: Runner,
    agent: Agent,
    conversation_id: str,
    steer_at: Any = None,
    stop_at_gate: bool = False,
) -> list[BaseEvent]:
    """Drive a turn the way a front-end does: every gate approved, in segments.

    This is the shape the bug lived in. A gated call ends ``run()`` on
    ``TOOL_NEEDS_FEEDBACK`` and the caller starts a *new* one with the
    ``TOOL_FEEDBACK``, so a turn with N approvals is N+1 generators — and the
    injection queue is registered per generator. ``steer_at`` is called with each
    event, for the case where the message arrives from outside the tool.
    """
    collected: list[BaseEvent] = []
    pending: BaseEvent | None = BaseEvent(
        event_type=ExecutionEventType.USER_INPUT,
        author="user",
        content=[Part(text="read every file")],
    )
    while pending is not None:
        needs_feedback: BaseEvent | None = None
        async for event in runner.run(
            application_name=APP,
            user_id=USER,
            conversation_id=conversation_id,
            root=agent,
            user_input=pending,
        ):
            collected.append(event)
            if steer_at is not None:
                await steer_at(event)
            if event.event_type is ExecutionEventType.TOOL_NEEDS_FEEDBACK:
                needs_feedback = event
        if needs_feedback is None:
            break
        if stop_at_gate:
            return collected
        action = needs_feedback.action.model_copy()
        action.observation = None
        pending = BaseEvent(
            event_type=ExecutionEventType.TOOL_FEEDBACK,
            author="user",
            action=action,
            metadata={"approved": True},
        )
    return collected


class TestAMessageThatArrivesBeforeAnApprovalGate:
    """The gate cuts the turn into several ``run()`` calls, so it has to be a
    boundary: the queue lives exactly as long as one of them.

    Worth stating plainly because the CLI's own configuration is the worst case: the
    ``files`` and ``shell`` bundles are gated, and ``/auto`` answers the gate rather
    than removing it, so *every* tool call ends a run. Before the gate was a
    boundary a mid-turn message reached the model on none of them — accepted,
    promised delivery, then dropped with the registration in ``run()``'s teardown,
    and resurfaced only as the next turn once the whole thing had finished.
    """

    @pytest.mark.asyncio
    @pytest.mark.parametrize("planner", PLANNERS)
    async def test_one_that_arrived_before_the_gate_is_taken_in_at_it(self, planner: str) -> None:
        """The reported bug: typed while the model was choosing the call.

        The run that accepted it ends at the gate, taking its queue with it, so the
        gate is the last chance to take it in. Missing that chance is what made the
        message reach the model only after the whole turn — which, for "actually,
        make it a txt", is after the docx had been written.
        """
        runner = Runner()
        conversation_id = f"c_gate_before_{planner}"
        provider = _ToolThenAnswer(rounds=2)
        agent = _agent(provider, _plain_look_tool_gated(), planner=planner)

        sent: list[Any] = []

        async def type_while_the_call_is_announced(event: BaseEvent) -> None:
            if event.event_type is ExecutionEventType.TOOL_START and not sent:
                sent.append(await _steer(runner, conversation_id))

        await _run_answering_gates(
            runner, agent, conversation_id, steer_at=type_while_the_call_is_announced
        )

        assert sent and sent[0].delivered, "accepted by the run that then hit the gate"
        assert provider.saw(1, STEER), "and read by the first call after the approval"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("planner", PLANNERS)
    async def test_the_resumed_run_takes_it_in(self, planner: str) -> None:
        runner = Runner()
        conversation_id = f"c_gate_survives_{planner}"
        provider = _ToolThenAnswer(rounds=2)
        agent = _agent(provider, _gated_look_tool(runner, conversation_id), planner=planner)

        await _run_answering_gates(runner, agent, conversation_id)

        # Call 0 asked for the tool. The message arrived while that call was
        # running, one generator ago — call 1 is the first one after it.
        assert provider.saw(1, STEER)

    @pytest.mark.asyncio
    async def test_it_lands_at_the_boundary_the_feedback_made(self) -> None:
        """Right after the approved call's result, not a tool call later.

        A resume executes the tool *before* the agent loop is entered, so its
        ``TOOL_END`` is the one boundary that is not inside the loop. Leaving it out
        cost a whole tool call: the planner had already been handed its messages.
        """
        runner = Runner()
        conversation_id = "c_gate_position"
        provider = _ToolThenAnswer(rounds=2)
        agent = _agent(provider, _gated_look_tool(runner, conversation_id))

        events = await _run_answering_gates(runner, agent, conversation_id)

        kinds = [
            event.event_type
            for event in events
            if event.event_type
            in (
                ExecutionEventType.TOOL_NEEDS_FEEDBACK,
                ExecutionEventType.TOOL_END,
                ExecutionEventType.USER_INPUT,
            )
        ]
        first_gate = kinds.index(ExecutionEventType.TOOL_NEEDS_FEEDBACK)
        assert kinds[first_gate + 1 : first_gate + 3] == [
            ExecutionEventType.TOOL_END,
            ExecutionEventType.USER_INPUT,
        ]

    @pytest.mark.asyncio
    async def test_it_is_absorbed_once(self) -> None:
        runner = Runner()
        conversation_id = "c_gate_once"
        provider = _ToolThenAnswer(rounds=2)
        agent = _agent(provider, _gated_look_tool(runner, conversation_id))

        events = await _run_answering_gates(runner, agent, conversation_id)

        echoes = [
            event
            for event in events
            if event.event_type is ExecutionEventType.USER_INPUT and event.author == "user"
        ]
        assert len(echoes) == 1
        assert provider.occurrences(provider.calls - 1, STEER) == 1

    @pytest.mark.asyncio
    async def test_nothing_is_held_once_it_has_been_taken_in(self) -> None:
        """A finished turn leaves no registration behind, suspended or otherwise."""
        runner = Runner()
        conversation_id = "c_gate_cleared"
        provider = _ToolThenAnswer(rounds=2)
        agent = _agent(provider, _gated_look_tool(runner, conversation_id))

        await _run_answering_gates(runner, agent, conversation_id)

        assert runner.injection_queue == {}

    @pytest.mark.asyncio
    async def test_a_message_the_user_abandoned_does_not_start_the_next_turn_twice(
        self,
    ) -> None:
        """A suspension nobody resumes hands the message back to the front-end.

        The front-end has its own copy — it is what it drew above the prompt — and
        sends it as an ordinary turn. So the next turn must *not* also find it
        parked, or the model reads the same sentence twice.
        """
        runner = Runner()
        conversation_id = "c_gate_abandoned"
        provider = _ToolThenAnswer(rounds=2)
        agent = _agent(provider, _gated_look_tool(runner, conversation_id))

        # The run ends at the gate on its own: the message is parked, nothing
        # resumes. Drained rather than broken out of, because a generator abandoned
        # mid-iteration is finalized by the garbage collector and its teardown is
        # then no longer part of what this is testing.
        await _run_answering_gates(runner, agent, conversation_id, stop_at_gate=True)

        # A fresh turn, carrying the text itself, as the front-end would send it.
        ordinary = _ToolThenAnswer(rounds=0)
        next_agent = _agent(ordinary, _plain_look_tool())
        second: list[BaseEvent] = []
        async for event in runner.run(
            application_name=APP,
            user_id=USER,
            conversation_id=conversation_id,
            root=next_agent,
            user_input=BaseEvent(
                event_type=ExecutionEventType.USER_INPUT, author="user", content=[Part(text=STEER)]
            ),
        ):
            second.append(event)

        # The turn's own input is rendered more than once by the context manager —
        # as the task and as the message — so the sentence is not what tells the two
        # apart. A parked one arrives stamped by ``push_parts``; the turn's does not.
        assert not any(event.metadata.get(METADATA_STEERING) for event in second)

    @pytest.mark.asyncio
    async def test_a_message_typed_during_the_pause_is_told_there_is_no_turn(self) -> None:
        """Between the gate and the answer to it, no generator is running.

        The gate is the run's last event and the queue goes with the run, so there is
        nowhere to put a message typed while the panel is up — and saying so is what
        lets the front-end keep it and send it as the next turn. Anything typed
        *before* the gate was taken in at it; this is the window after.
        """
        runner = Runner()
        conversation_id = "c_gate_paused_push"
        provider = _ToolThenAnswer(rounds=2)
        agent = _agent(provider, _plain_look_tool_gated())

        await _run_answering_gates(runner, agent, conversation_id, stop_at_gate=True)

        result = await _steer(runner, conversation_id)
        assert not result.delivered

    @pytest.mark.asyncio
    async def test_the_messages_stay_valid_around_the_tool_call(self) -> None:
        """Taken in at the gate, it lands between a tool call and its result.

        That ordering is the one native tool-calling APIs reject, and it is why this
        boundary is only safe with ``reorder_messages_around_tool_calls`` under it:
        every ``tool`` response still immediately follows the assistant message that
        asked for it, and the message is rendered after them.
        """
        runner = Runner()
        conversation_id = "c_gate_render"
        provider = _ToolThenAnswer(rounds=2)
        agent = _agent(provider, _plain_look_tool_gated())

        sent: list[Any] = []

        async def type_while_the_call_is_announced(event: BaseEvent) -> None:
            if event.event_type is ExecutionEventType.TOOL_START and not sent:
                sent.append(await _steer(runner, conversation_id))

        await _run_answering_gates(
            runner, agent, conversation_id, steer_at=type_while_the_call_is_announced
        )

        for messages in provider.shown:
            for index, message in enumerate(messages):
                if message.role != "assistant" or not message.tool_calls:
                    continue
                answers = messages[index + 1 : index + 1 + len(message.tool_calls)]
                assert [m.role for m in answers] == ["tool"] * len(message.tool_calls), (
                    "a message was left wedged between a tool call and its result"
                )
