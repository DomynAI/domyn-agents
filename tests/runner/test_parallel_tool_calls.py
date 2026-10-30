"""Several tools asked for in one model response.

- The calls that need no approval run at the same time; the turn waits for the
  slowest, not for their sum.
- A call that needs approval runs alone and in order: what comes before it has
  finished, what comes after it has not started.
- A pause for approval holds the calls after it: the answer runs them, before the
  model is asked again. A new message instead of the answer closes them, each with
  a result saying it did not run.
- Every call the model made has exactly one result in what it is shown next.
"""

# No `from __future__ import annotations`: builds real agents and tools.
import asyncio
import time
from typing import Any

import pytest
from pydantic import Field

from domyn.agents import Agent
from domyn.agents.core import (
    METADATA_TOOL_BATCH,
    ApprovalType,
    BaseEvent,
    ExecutionEventType,
    FeedbackAction,
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
from domyn.agents.runner import HELD_TOOL_CALLS_KEY, Runner
from domyn.agents.tools import FunctionTool

APP = "parallel_app"
USER = "user_1"
PLANNERS = ["tool_use", "tool_use_thought"]


class _Calls(BaseLLMProvider):
    """Asks for ``batch`` (name, arguments) in one response, then answers."""

    type: str = "calls"
    name: str = "calls"
    model_name: str = "calls"
    supported_structured_outputs: list[Any] | None = None
    batch: list[tuple[str, str]] = Field(default_factory=list)
    calls: int = 0
    shown: list[list[LLMMessage]] = Field(default_factory=list)

    async def generate(self, messages: list[LLMMessage], **kwargs: Any) -> LLMResponse:
        self.calls += 1
        self.shown.append(list(messages))
        if self.calls == 1:
            return LLMResponse(
                content="",
                generated_tokens=1,
                execution_time=0.0,
                usage={},
                tool_calls=[
                    LLMToolCall(
                        id=f"call_{i}",
                        type="function",
                        function=ToolCallFunction(name=name, arguments=arguments),
                    )
                    for i, (name, arguments) in enumerate(self.batch)
                ],
            )
        return LLMResponse(content="Done.", generated_tokens=1, execution_time=0.0, usage={})

    async def generate_streaming(self, messages: list[LLMMessage], **kwargs: Any) -> Any:
        raise NotImplementedError
        yield  # pragma: no cover

    def tool_messages(self, call: int) -> dict[str, str]:
        return {
            m.tool_call_id: str(m.content)
            for m in self.shown[call]
            if m.role == "tool" and m.tool_call_id
        }

    def asked(self, call: int) -> list[str]:
        return [tc.id for m in self.shown[call] if m.tool_calls for tc in m.tool_calls]


def _tools(log: list[str], delay: float = 0.3) -> list[FunctionTool]:
    async def slow_read(path: str) -> str:
        """Read a file slowly.

        Args:
            path: the file.
        """
        log.append(f"start {path}")
        await asyncio.sleep(delay)
        log.append(f"end {path}")
        return f"contents of {path}"

    async def write(path: str) -> str:
        """Write a file.

        Args:
            path: the file.
        """
        log.append(f"write {path}")
        return f"wrote {path}"

    return [
        FunctionTool(func=slow_read),
        FunctionTool(func=write, approval_type=ApprovalType.USER_CONFIRM),
    ]


def _agent(
    provider: BaseLLMProvider, tools: list[FunctionTool], planner: str, **options: Any
) -> Agent:
    return Agent(
        name="MainAgent",
        description="the agent",
        instruction="do as asked",
        llm_provider=provider,
        planner=PlannerStrategyFactory.from_config(
            {"type": planner, **options}, save_in_registry=False
        ),
        tools=tools,
    )


async def _run(
    runner: Runner, agent: Agent, conversation_id: str, user_input: BaseEvent
) -> list[BaseEvent]:
    return [
        event
        async for event in runner.run(
            application_name=APP,
            user_id=USER,
            conversation_id=conversation_id,
            root=agent,
            user_input=user_input,
        )
    ]


def _ask(text: str = "do it") -> BaseEvent:
    return BaseEvent(
        event_type=ExecutionEventType.USER_INPUT, author="user", content=[Part(text=text)]
    )


def _approve(name: str = "write", approve: bool = True) -> BaseEvent:
    return BaseEvent(
        event_type=ExecutionEventType.TOOL_FEEDBACK,
        author="user",
        action=FeedbackAction(
            name=name,
            parameters=[],
            approval_type=ApprovalType.USER_CONFIRM,
            approve=approve,
        ),
    )


def _results(events: list[BaseEvent]) -> list[str]:
    return [
        e.action.call_id
        for e in events
        if e.event_type in (ExecutionEventType.TOOL_END, ExecutionEventType.TOOL_ERROR)
    ]


def _reads(*paths: str) -> list[tuple[str, str]]:
    return [("slow_read", f'{{"path": "{p}"}}') for p in paths]


class TestCallsWithoutApprovalRunTogether:
    @pytest.mark.parametrize("planner", PLANNERS)
    @pytest.mark.asyncio
    async def test_three_slow_reads_take_the_time_of_one(self, planner: str) -> None:
        log: list[str] = []
        provider = _Calls(batch=_reads("a", "b", "c"))
        agent = _agent(provider, _tools(log), planner)

        started = time.monotonic()
        events = await _run(Runner(), agent, f"c_par_{planner}", _ask())
        elapsed = time.monotonic() - started

        assert elapsed < 0.75, f"took {elapsed:.2f}s: the reads ran one after another"
        assert log[:3] == ["start a", "start b", "start c"]
        assert sorted(_results(events)) == ["call_0", "call_1", "call_2"]
        # The model is asked again once, after all three: two requests in all, the
        # second with the three calls and the three results.
        assert provider.calls == 2
        assert provider.asked(1) == ["call_0", "call_1", "call_2"]
        results = provider.tool_messages(1)
        assert sorted(results) == ["call_0", "call_1", "call_2"]
        # Each after the `[event_id: ...]` header the context manager puts first.
        assert [results[f"call_{i}"].splitlines()[-1] for i in range(3)] == [
            "contents of a",
            "contents of b",
            "contents of c",
        ]
        assert events[-1].event_type == ExecutionEventType.AGENT_END

    @pytest.mark.asyncio
    async def test_the_starts_go_out_before_the_results(self) -> None:
        log: list[str] = []
        agent = _agent(_Calls(batch=_reads("a", "b")), _tools(log), "tool_use")
        events = await _run(Runner(), agent, "c_order", _ask())
        kinds = [
            e.event_type
            for e in events
            if e.event_type in (ExecutionEventType.TOOL_START, ExecutionEventType.TOOL_END)
        ]
        assert kinds == [
            ExecutionEventType.TOOL_START,
            ExecutionEventType.TOOL_START,
            ExecutionEventType.TOOL_END,
            ExecutionEventType.TOOL_END,
        ]

    @pytest.mark.parametrize("planner", PLANNERS)
    @pytest.mark.asyncio
    async def test_switched_off_on_the_strategy_they_run_one_after_another(
        self, planner: str
    ) -> None:
        log: list[str] = []
        agent = _agent(
            _Calls(batch=_reads("a", "b")),
            _tools(log, delay=0.05),
            planner,
            parallel_tool_calls=False,
        )
        events = await _run(Runner(), agent, f"c_off_{planner}", _ask())
        assert log == ["start a", "end a", "start b", "end b"]
        assert all(
            METADATA_TOOL_BATCH not in e.metadata
            for e in events
            if e.event_type == ExecutionEventType.TOOL_START
        )

    def test_it_is_on_by_default(self) -> None:
        for planner in PLANNERS:
            strategy = PlannerStrategyFactory.from_config({"type": planner}, save_in_registry=False)
            assert strategy.parallel_tool_calls is True

    @pytest.mark.asyncio
    async def test_the_limit_caps_how_many_run_at_once(self) -> None:
        log: list[str] = []
        agent = _agent(_Calls(batch=_reads("a", "b", "c")), _tools(log, delay=0.05), "tool_use")
        await _run(Runner(max_parallel_tool_calls=1), agent, "c_cap", _ask())
        assert log == ["start a", "end a", "start b", "end b", "start c", "end c"]

    @pytest.mark.asyncio
    async def test_the_planner_marks_the_batch(self) -> None:
        log: list[str] = []
        agent = _agent(_Calls(batch=_reads("a", "b")), _tools(log, delay=0.0), "tool_use")
        events = await _run(Runner(), agent, "c_mark", _ask())
        batches = [
            e.metadata.get(METADATA_TOOL_BATCH)
            for e in events
            if e.event_type == ExecutionEventType.TOOL_START
        ]
        assert [b["index"] for b in batches] == [0, 1]
        assert {b["size"] for b in batches} == {2} and len({b["id"] for b in batches}) == 1

    @pytest.mark.asyncio
    async def test_a_single_call_is_not_a_batch(self) -> None:
        log: list[str] = []
        agent = _agent(_Calls(batch=_reads("a")), _tools(log, delay=0.0), "tool_use")
        events = await _run(Runner(), agent, "c_one", _ask())
        start = next(e for e in events if e.event_type == ExecutionEventType.TOOL_START)
        assert METADATA_TOOL_BATCH not in start.metadata


class TestACallThatNeedsApproval:
    @pytest.mark.asyncio
    async def test_it_runs_alone_and_in_order(self) -> None:
        """read a, then write b (approved by the answer), then read c: the read
        after the write does not start before the write has run."""
        log: list[str] = []
        provider = _Calls(
            batch=[*_reads("a"), ("write", '{"path": "b"}'), *_reads("c")],
        )
        agent = _agent(provider, _tools(log, delay=0.0), "tool_use")
        runner = Runner()

        first = await _run(runner, agent, "c_gate", _ask())
        assert first[-1].event_type == ExecutionEventType.TOOL_NEEDS_FEEDBACK
        assert log == ["start a", "end a"], "nothing after the write may start before it"

        second = await _run(runner, agent, "c_gate", _approve())
        assert log == ["start a", "end a", "write b", "start c", "end c"]
        assert _results(second)[:2] == ["call_1", "call_2"]
        assert provider.calls == 2
        assert sorted(provider.tool_messages(1)) == ["call_0", "call_1", "call_2"]

    @pytest.mark.asyncio
    async def test_a_refusal_still_runs_the_calls_after_it(self) -> None:
        log: list[str] = []
        provider = _Calls(batch=[("write", '{"path": "b"}'), *_reads("c")])
        agent = _agent(provider, _tools(log, delay=0.0), "tool_use")
        runner = Runner()

        await _run(runner, agent, "c_refuse", _ask())
        second = await _run(runner, agent, "c_refuse", _approve(approve=False))
        assert log == ["start c", "end c"]
        assert _results(second)[:2] == ["call_0", "call_1"]
        assert "refused" in provider.tool_messages(1)["call_0"]

    @pytest.mark.asyncio
    async def test_the_answer_goes_to_the_call_that_was_waiting(self) -> None:
        """The latest TOOL_START is not the waiting one when a response asked for
        several: the answer must not run the wrong call."""
        log: list[str] = []
        provider = _Calls(batch=[("write", '{"path": "first"}'), *_reads("later")])
        agent = _agent(provider, _tools(log, delay=0.0), "tool_use")
        runner = Runner()
        await _run(runner, agent, "c_which", _ask())
        await _run(runner, agent, "c_which", _approve())
        assert log[0] == "write first"


class TestANewMessageInsteadOfTheAnswer:
    @pytest.mark.asyncio
    async def test_the_waiting_and_held_calls_are_closed(self) -> None:
        log: list[str] = []
        provider = _Calls(batch=[("write", '{"path": "b"}'), *_reads("c")])
        agent = _agent(provider, _tools(log, delay=0.0), "tool_use")
        runner = Runner()

        await _run(runner, agent, "c_moved_on", _ask())
        await _run(runner, agent, "c_moved_on", _ask("never mind, do something else"))

        assert log == [], "neither call may run"
        shown = provider.tool_messages(1)
        assert sorted(shown) == ["call_0", "call_1"]
        assert all("was not executed" in text for text in shown.values())
        assert "new message" in shown["call_0"]

    @pytest.mark.asyncio
    async def test_nothing_is_left_held(self) -> None:
        log: list[str] = []
        provider = _Calls(batch=[("write", '{"path": "b"}'), *_reads("c")])
        agent = _agent(provider, _tools(log, delay=0.0), "tool_use")
        runner = Runner()
        await _run(runner, agent, "c_clean", _ask())
        await _run(runner, agent, "c_clean", _ask("something else"))
        info = await runner.conversation_manager.get_conversation(APP, USER, "c_clean")
        assert HELD_TOOL_CALLS_KEY not in (info.state or {})
