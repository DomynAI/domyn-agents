"""Several sub-agents asked for in one response (`parallel_handoffs`).

- On, they run at the same time, each as a branch of the same conversation, and
  the caller resumes once all of them have ended, with every call answered and
  their results in the order they were called.
- A sub-agent that needs an approval stops there while the others finish; the
  questions are asked one at a time, each answer resumes the sub-agent that asked,
  and the caller resumes once the last has ended — after a restart too.
- A new message instead of the answer closes what had not ended, so the caller's
  next request still has a result for every call it made.
- Off, a response naming several runs them one after the other, each to its end;
  one naming a single sub-agent and nothing else is a hand-off, as it always was.
"""

# No `from __future__ import annotations`: builds real agents and tools.
import asyncio
import re
import sqlite3
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any

import pytest
from pydantic import Field

from domyn.agents.agents import Agent
from domyn.agents.agents.base import ContextVisibility
from domyn.agents.agents.delegate_agent import DelegateAgent
from domyn.agents.conversation_manager import (
    BaseConversationManager,
    SqliteConversationManager,
)
from domyn.agents.core import (
    METADATA_BRANCH,
    METADATA_TOOL_BATCH,
    ApprovalType,
    BaseEvent,
    ExecutionEventType,
    FeedbackAction,
    Part,
)
from domyn.agents.core.context import InteractionContext
from domyn.agents.hooks.base import HookPhase, HookResult, HookType
from domyn.agents.hooks.function_hook import FunctionHook
from domyn.agents.llm.base import (
    BaseLLMProvider,
    LLMMessage,
    LLMResponse,
    LLMToolCall,
    ToolCallFunction,
)
from domyn.agents.planner_strategy import HeuristicEstimator, PlannerStrategyFactory
from domyn.agents.planner_strategy.context_management.budget import CONTEXT_USAGE_METADATA_KEY
from domyn.agents.runner import RUNNING_TURN_KEY, Runner
from domyn.agents.runner.branches import WAITING_BRANCHES_KEY
from domyn.agents.runner.transport import RunnerTransport
from domyn.agents.tools.function_tool import FunctionTool

APP = "parallel_handoffs_app"
USER = "user_1"
PLANNERS = ["tool_use", "tool_use_thought"]


class _Script(BaseLLMProvider):
    """Answers with ``replies`` in turn, after ``delay`` seconds each; a reply that
    is a list of (name, arguments) is a response asking for those calls."""

    type: str = "script"
    name: str = "script"
    model_name: str = "script"
    supported_structured_outputs: list[Any] | None = None
    replies: list[Any] = Field(default_factory=list)
    delay: float = 0.0
    # Any, so the list is the caller's and not a copy: the agents write to one log.
    log: Any = Field(default_factory=list)
    label: str = ""
    shown: list[list[LLMMessage]] = Field(default_factory=list)

    async def generate(self, messages: list[LLMMessage], **kwargs: Any) -> LLMResponse:
        self.shown.append(list(messages))
        self.log.append(f"start {self.label}")
        await asyncio.sleep(self.delay)
        self.log.append(f"end {self.label}")
        turn = min(len(self.shown), len(self.replies))
        reply = self.replies[turn - 1]
        if isinstance(reply, list):
            return LLMResponse(
                content="",
                generated_tokens=1,
                execution_time=0.0,
                usage={},
                tool_calls=[
                    LLMToolCall(
                        # Unique across agents, as a provider's ids are.
                        id=f"{self.label}_{turn}_{i}",
                        type="function",
                        function=ToolCallFunction(name=name, arguments=arguments),
                    )
                    for i, (name, arguments) in enumerate(reply)
                ],
            )
        return LLMResponse(content=reply, generated_tokens=1, execution_time=0.0, usage={})

    async def generate_streaming(self, messages: list[LLMMessage], **kwargs: Any) -> Any:
        raise NotImplementedError
        yield  # pragma: no cover

    def calls(self, request: int) -> list[str]:
        return [tc.id for m in self.shown[request] if m.tool_calls for tc in m.tool_calls]

    def answered(self, request: int) -> list[str]:
        return [m.tool_call_id for m in self.shown[request] if m.role == "tool" and m.tool_call_id]


def _planner(planner: str, **options: Any) -> Any:
    return PlannerStrategyFactory.from_config({"type": planner, **options}, save_in_registry=False)


def _write_tool(log: list[str]) -> FunctionTool:
    async def write(path: str) -> str:
        """Write a file.

        Args:
            path: the file.
        """
        log.append(f"write {path}")
        return f"wrote {path}"

    return FunctionTool(func=write, approval_type=ApprovalType.USER_CONFIRM)


def _sub(
    name: str, planner: str, log: list[str], delay: float = 0.0, writes: bool = False
) -> tuple[Agent, _Script]:
    """A sub-agent that answers, or first asks to write a file (needing approval)."""
    replies: list[Any] = [f"{name} found it"]
    if writes:
        replies = [[("write", f'{{"path": "{name.lower()}.txt"}}')], f"{name} wrote it"]
    provider = _Script(label=name, log=log, delay=delay, replies=replies)
    agent = Agent(
        name=name,
        description=f"finds {name.lower()}",
        instruction="do the task",
        llm_provider=provider,
        planner=_planner(planner),
        tools=[_write_tool(log)] if writes else [],
    )
    return agent, provider


def _team(
    planner: str,
    log: list[str] | None = None,
    parallel: bool = True,
    delays: tuple[float, float] = (0.3, 0.3),
    writes: tuple[bool, bool] = (False, False),
) -> tuple[Agent, _Script, list[_Script]]:
    log = [] if log is None else log
    main = _Script(
        label="main",
        log=log,
        replies=[
            [("Alpha", '{"task": "find a"}'), ("Beta", '{"task": "find b"}')],
            "Done.",
        ],
    )
    subs = [
        _sub(name, planner, log, delay, write)
        for name, delay, write in zip(("Alpha", "Beta"), delays, writes, strict=True)
    ]
    root = Agent(
        name="MainAgent",
        description="the agent",
        instruction="delegate",
        llm_provider=main,
        planner=_planner(planner, parallel_handoffs=parallel),
        sub_agents=[agent for agent, _ in subs],
    )
    return root, main, [provider for _, provider in subs]


def _overlapped(log: list[str], first: str, second: str) -> bool:
    """Whether ``first`` and ``second`` both started before either of them ended.

    Read from the order of the log rather than from the clock: a time limit has
    to leave room for a busy CI runner, and the room it leaves is what sequential
    runs fit in (two 0.3 s waits, 0.55 s allowed, 0.65 s seen on a loaded runner).
    """
    started = max(log.index(f"start {first}"), log.index(f"start {second}"))
    ended = min(log.index(f"end {first}"), log.index(f"end {second}"))
    return started < ended


def _ask(text: str = "find a and b") -> BaseEvent:
    return BaseEvent(
        event_type=ExecutionEventType.USER_INPUT, author="user", content=[Part(text=text)]
    )


def _approve(approve: bool = True) -> BaseEvent:
    return BaseEvent(
        event_type=ExecutionEventType.TOOL_FEEDBACK,
        author="user",
        action=FeedbackAction(
            name="write",
            parameters=[],
            approval_type=ApprovalType.USER_CONFIRM,
            approve=approve,
        ),
    )


async def _run(
    runner: Runner, agent: Agent, conversation_id: str, user_input: BaseEvent | None = None
) -> list[BaseEvent]:
    return [
        event
        async for event in runner.run(
            application_name=APP,
            user_id=USER,
            conversation_id=conversation_id,
            root=agent,
            user_input=user_input or _ask(),
        )
    ]


def _of(events: list[BaseEvent], kind: ExecutionEventType) -> list[str]:
    """Who among the sub-agents had an event of ``kind``, in order."""
    return [e.author for e in events if e.event_type == kind and e.author != "MainAgent"]


def _answered_by_main(events: list[BaseEvent]) -> bool:
    """Whether the turn ended on the caller's answer (followed by its AGENT_END)."""
    answers = [e for e in events if e.event_type == ExecutionEventType.RESPONSE]
    return bool(answers) and answers[-1].author == "MainAgent" and not answers[-1].is_partial


async def _state(manager: BaseConversationManager, conversation_id: str) -> dict[str, Any]:
    info = await manager.get_conversation(APP, USER, conversation_id)
    assert info is not None
    return info.state or {}


class TestSubAgentsRunTogether:
    @pytest.mark.parametrize("planner", PLANNERS)
    @pytest.mark.asyncio
    async def test_two_slow_sub_agents_take_the_time_of_one(self, planner: str) -> None:
        log: list[str] = []
        root, main, _ = _team(planner, log)

        events = await _run(Runner(), root, f"c_together_{planner}")

        assert _overlapped(log, "Alpha", "Beta"), f"the sub-agents ran one after another: {log}"
        assert log[2:4] == ["start Alpha", "start Beta"]
        assert _of(events, ExecutionEventType.AGENT_START) == ["Alpha", "Beta"]
        assert sorted(_of(events, ExecutionEventType.AGENT_END)) == ["Alpha", "Beta"]
        # The caller is asked again once, after both, and answers.
        assert len(main.shown) == 2
        final = [e for e in events if e.event_type == ExecutionEventType.RESPONSE]
        assert final and final[-1].author == "MainAgent"

    @pytest.mark.parametrize("planner", PLANNERS)
    @pytest.mark.asyncio
    async def test_the_caller_sees_every_call_answered_in_call_order(self, planner: str) -> None:
        # Beta ends first; the caller still reads Alpha's result first.
        root, main, _ = _team(planner, delays=(0.3, 0.05))

        await _run(Runner(), root, f"c_order_{planner}")

        assert main.calls(1) == ["main_1_0", "main_1_1"]
        # Each call is followed by its own answer, as the wire protocol demands.
        request = main.shown[1]
        position = next(i for i, m in enumerate(request) if m.tool_calls)
        answered = [m.tool_call_id for m in request[position + 1 : position + 3]]
        assert answered == ["main_1_0", "main_1_1"]
        reports = [str(m.content) for m in request if "found it" in str(m.content or "")]
        assert [r.split("found it")[0].split()[-1] for r in reports] == ["Alpha", "Beta"]

    @pytest.mark.parametrize("planner", PLANNERS)
    @pytest.mark.asyncio
    async def test_a_sub_agent_does_not_see_its_sibling_at_work(self, planner: str) -> None:
        root, _, (alpha, beta) = _team(planner, delays=(0.0, 0.0))

        await _run(Runner(), root, f"c_hidden_{planner}")

        for provider, sibling in ((alpha, "Beta"), (beta, "Alpha")):
            seen = " ".join(str(m.content or "") for m in provider.shown[0])
            assert f"{sibling} found it" not in seen
            assert f"Agent: {sibling}" not in seen


class TestASubAgentWaitingForApproval:
    @pytest.mark.parametrize("planner", PLANNERS)
    @pytest.mark.asyncio
    async def test_the_other_finishes_and_the_run_ends_on_the_question(self, planner: str) -> None:
        log: list[str] = []
        root, main, _ = _team(planner, log, delays=(0.0, 0.0), writes=(True, False))
        runner = Runner()

        events = await _run(runner, root, f"c_wait_{planner}")

        assert events[-1].event_type == ExecutionEventType.TOOL_NEEDS_FEEDBACK
        assert events[-1].author == "Alpha"
        assert events[-1].metadata[METADATA_BRANCH] == "Alpha"
        assert _of(events, ExecutionEventType.AGENT_END) == ["Beta"]
        assert "write alpha.txt" not in log
        # The caller has not been asked again: one of its calls has no result yet.
        assert len(main.shown) == 1

    @pytest.mark.parametrize("planner", PLANNERS)
    @pytest.mark.asyncio
    async def test_the_answer_resumes_it_and_then_the_caller(self, planner: str) -> None:
        log: list[str] = []
        root, main, _ = _team(planner, log, delays=(0.0, 0.0), writes=(True, False))
        runner = Runner()
        conversation = f"c_resume_{planner}"
        await _run(runner, root, conversation)

        events = await _run(runner, root, conversation, _approve())

        assert "write alpha.txt" in log
        assert _of(events, ExecutionEventType.AGENT_END) == ["Alpha"]
        assert _answered_by_main(events)
        assert main.calls(1) == ["main_1_0", "main_1_1"]
        assert sorted(main.answered(1)) == ["main_1_0", "main_1_1"]
        assert not (await _state(runner.conversation_manager, conversation)).get(
            WAITING_BRANCHES_KEY
        )

    @pytest.mark.parametrize("planner", PLANNERS)
    @pytest.mark.asyncio
    async def test_two_questions_are_asked_one_at_a_time(self, planner: str) -> None:
        log: list[str] = []
        root, main, _ = _team(planner, log, delays=(0.0, 0.0), writes=(True, True))
        runner = Runner()
        conversation = f"c_two_{planner}"

        first = await _run(runner, root, conversation)
        second = await _run(runner, root, conversation, _approve())
        third = await _run(runner, root, conversation, _approve())

        assert first[-1].author == "Alpha"
        assert first[-1].event_type == ExecutionEventType.TOOL_NEEDS_FEEDBACK
        assert second[-1].author == "Beta"
        assert second[-1].event_type == ExecutionEventType.TOOL_NEEDS_FEEDBACK
        assert _answered_by_main(third)
        assert log.count("write alpha.txt") == log.count("write beta.txt") == 1
        assert len(main.shown) == 2

    @pytest.mark.parametrize("planner", PLANNERS)
    @pytest.mark.asyncio
    async def test_it_resumes_after_a_restart(self, planner: str, tmp_path: Path) -> None:
        log: list[str] = []
        path = str(tmp_path / "conversations.db")
        root, main, _ = _team(planner, log, delays=(0.0, 0.0), writes=(True, False))
        conversation = f"c_restart_{planner}"
        await _run(Runner(conversation_manager=SqliteConversationManager(path)), root, conversation)

        events = await _run(
            Runner(conversation_manager=SqliteConversationManager(path)),
            root,
            conversation,
            _approve(),
        )

        assert "write alpha.txt" in log
        assert _answered_by_main(events)
        assert sorted(main.answered(1)) == ["main_1_0", "main_1_1"]

    @pytest.mark.parametrize("planner", PLANNERS)
    @pytest.mark.asyncio
    async def test_every_question_two_levels_down_names_the_branch_the_user_saw(
        self, planner: str
    ) -> None:
        """Two sub-agents of a nested batch both ask: each question names the
        sub-agent the caller started, not only the first one asked. The others
        waiting were asked later, from the conversation's stack, where no branch
        is left to name them, and named the agent asking instead."""
        log: list[str] = []
        inner = [_sub("Y", planner, log, writes=True)[0], _sub("Z", planner, log, writes=True)[0]]
        x = Agent(
            name="X",
            description="splits its work",
            instruction="delegate",
            llm_provider=_Script(
                label="X",
                log=log,
                replies=[[("Y", '{"task": "y"}'), ("Z", '{"task": "z"}')], "X done"],
            ),
            planner=_planner(planner, parallel_handoffs=True),
            sub_agents=inner,
        )
        root = Agent(
            name="MainAgent",
            description="the agent",
            instruction="delegate",
            llm_provider=_Script(
                label="main",
                log=log,
                replies=[[("X", '{"task": "x"}'), ("Beta", '{"task": "b"}')], "Done."],
            ),
            planner=_planner(planner, parallel_handoffs=True),
            sub_agents=[x, _sub("Beta", planner, log)[0]],
        )
        runner = Runner()
        conversation = f"c_nested_names_{planner}"

        first = await _run(runner, root, conversation)
        second = await _run(runner, root, conversation, _approve())
        third = await _run(runner, root, conversation, _approve())

        asked = [(t[-1].author, t[-1].metadata.get(METADATA_BRANCH)) for t in (first, second)]
        assert sorted(asked) == [("Y", "X"), ("Z", "X")]
        assert _answered_by_main(third)

    @pytest.mark.parametrize("planner", PLANNERS)
    @pytest.mark.asyncio
    async def test_one_waiting_two_levels_down_is_resumed_and_returns_up(
        self, planner: str
    ) -> None:
        log: list[str] = []
        inner = [_sub("Y", planner, log, writes=True)[0], _sub("Z", planner, log)[0]]
        x_provider = _Script(
            label="X", log=log, replies=[[("Y", '{"task": "y"}'), ("Z", '{"task": "z"}')], "X done"]
        )
        x = Agent(
            name="X",
            description="splits its work",
            instruction="delegate",
            llm_provider=x_provider,
            planner=_planner(planner, parallel_handoffs=True),
            sub_agents=inner,
        )
        main = _Script(
            label="main",
            log=log,
            replies=[[("X", '{"task": "x"}'), ("Beta", '{"task": "b"}')], "Done."],
        )
        root = Agent(
            name="MainAgent",
            description="the agent",
            instruction="delegate",
            llm_provider=main,
            planner=_planner(planner, parallel_handoffs=True),
            sub_agents=[x, _sub("Beta", planner, log)[0]],
        )
        runner = Runner()
        conversation = f"c_nested_{planner}"

        first = await _run(runner, root, conversation)
        second = await _run(runner, root, conversation, _approve())

        assert first[-1].event_type == ExecutionEventType.TOOL_NEEDS_FEEDBACK
        assert first[-1].author == "Y"
        assert first[-1].metadata[METADATA_BRANCH] == "X"
        assert "write y.txt" in log
        assert _of(second, ExecutionEventType.AGENT_END) == ["Y", "X"]
        assert sorted(x_provider.answered(1)) == ["X_1_0", "X_1_1"]
        assert _answered_by_main(second)
        assert sorted(main.answered(1)) == ["main_1_0", "main_1_1"]

    @pytest.mark.parametrize("planner", PLANNERS)
    @pytest.mark.asyncio
    async def test_a_new_message_goes_to_the_caller_and_closes_it(self, planner: str) -> None:
        log: list[str] = []
        root, main, (alpha, _) = _team(planner, log, delays=(0.0, 0.0), writes=(True, False))
        runner = Runner()
        conversation = f"c_new_message_{planner}"
        await _run(runner, root, conversation)

        events = await _run(runner, root, conversation, _ask("never mind"))

        # The entrypoint is the one the message reaches: Alpha is not asked again,
        # its call to write is closed as not run, and it ends as interrupted.
        assert "write alpha.txt" not in log
        assert len(alpha.shown) == 1
        [end] = [
            e
            for e in events
            if e.event_type == ExecutionEventType.AGENT_END and e.author == "Alpha"
        ]
        assert end.error_code == "SubAgentInterrupted"
        assert "never mind" in " ".join(str(m.content or "") for m in main.shown[1])
        assert _answered_by_main(events)
        assert sorted(main.answered(1)) == ["main_1_0", "main_1_1"]


class _Remote(RunnerTransport):
    """An external agent that takes ``delay`` seconds and then ends with an answer."""

    def __init__(
        self, name: str, delay: float, log: list[str], interaction_id: str | None = None
    ) -> None:
        self._name = name
        self._delay = delay
        self._log = log
        self._interaction_id = interaction_id

    async def execute(
        self, agent_name: str, context: InteractionContext
    ) -> AsyncGenerator[BaseEvent, None]:
        self._log.append(f"start {self._name}")
        await asyncio.sleep(self._delay)
        self._log.append(f"end {self._name}")
        yield BaseEvent(
            event_type=ExecutionEventType.AGENT_END,
            author=self._name,
            content=[Part(text=f"{self._name} found it")],
            interaction_id=self._interaction_id,
        )

    async def send_result(self, event: BaseEvent, conversation_id: str | None = None) -> None:
        pass


def _slow_read(log: list[str], delay: float) -> FunctionTool:
    async def slow_read(path: str) -> str:
        """Read a file slowly.

        Args:
            path: the file.
        """
        log.append(f"start read {path}")
        await asyncio.sleep(delay)
        log.append(f"end read {path}")
        return f"contents of {path}"

    return FunctionTool(func=slow_read)


class TestWhatRunsAlongside:
    @pytest.mark.parametrize("planner", PLANNERS)
    @pytest.mark.asyncio
    async def test_a_tool_and_a_sub_agent_of_one_response_run_together(self, planner: str) -> None:
        log: list[str] = []
        alpha, _ = _sub("Alpha", planner, log, delay=0.3)
        main = _Script(
            label="main",
            log=log,
            replies=[
                [("slow_read", '{"path": "x"}'), ("Alpha", '{"task": "find a"}')],
                "Done.",
            ],
        )
        root = Agent(
            name="MainAgent",
            description="the agent",
            instruction="delegate",
            llm_provider=main,
            planner=_planner(planner, parallel_handoffs=True),
            sub_agents=[alpha],
            tools=[_slow_read(log, 0.3)],
        )

        events = await _run(Runner(), root, f"c_mixed_{planner}")

        assert _overlapped(log, "read x", "Alpha"), f"the tool and the sub-agent took turns: {log}"
        assert _answered_by_main(events)
        assert sorted(main.answered(1)) == ["main_1_0", "main_1_1"]

    @pytest.mark.parametrize("planner", PLANNERS)
    @pytest.mark.asyncio
    async def test_a_batch_ending_on_a_tool_runs_the_same(self, planner: str) -> None:
        log: list[str] = []
        alpha, _ = _sub("Alpha", planner, log, delay=0.3)
        main = _Script(
            label="main",
            log=log,
            replies=[
                [("Alpha", '{"task": "find a"}'), ("slow_read", '{"path": "x"}')],
                "Done.",
            ],
        )
        root = Agent(
            name="MainAgent",
            description="the agent",
            instruction="delegate",
            llm_provider=main,
            planner=_planner(planner, parallel_handoffs=True),
            sub_agents=[alpha],
            tools=[_slow_read(log, 0.3)],
        )

        events = await _run(Runner(), root, f"c_tool_last_{planner}")

        assert _overlapped(log, "Alpha", "read x"), f"the sub-agent and the tool took turns: {log}"
        assert _answered_by_main(events)
        assert sorted(main.answered(1)) == ["main_1_0", "main_1_1"]

    @pytest.mark.parametrize("planner", PLANNERS)
    @pytest.mark.asyncio
    async def test_a_call_needing_approval_after_a_waiting_sub_agent_runs_once_it_returns(
        self, planner: str
    ) -> None:
        log: list[str] = []
        alpha, _ = _sub("Alpha", planner, log, writes=True)
        main = _Script(
            label="main",
            log=log,
            replies=[
                [("Alpha", '{"task": "find a"}'), ("write", '{"path": "main.txt"}')],
                "Done.",
            ],
        )
        root = Agent(
            name="MainAgent",
            description="the agent",
            instruction="delegate",
            llm_provider=main,
            planner=_planner(planner, parallel_handoffs=True),
            sub_agents=[alpha],
            tools=[_write_tool(log)],
        )
        runner = Runner()
        conversation = f"c_held_after_{planner}"

        first = await _run(runner, root, conversation)
        second = await _run(runner, root, conversation, _approve())
        third = await _run(runner, root, conversation, _approve())

        assert first[-1].author == "Alpha"
        # Alpha's call ran and Alpha returned; then the caller's own call asks.
        assert second[-1].event_type == ExecutionEventType.TOOL_NEEDS_FEEDBACK
        assert second[-1].author == "MainAgent"
        assert log.index("write alpha.txt") < log.index("write main.txt")
        assert _answered_by_main(third)
        assert sorted(main.answered(1)) == ["main_1_0", "main_1_1"]
        assert not (await _state(runner.conversation_manager, conversation)).get(
            WAITING_BRANCHES_KEY
        )

    @pytest.mark.parametrize("planner", PLANNERS)
    @pytest.mark.asyncio
    async def test_a_sub_agent_after_a_call_needing_approval_waits_for_the_answer(
        self, planner: str
    ) -> None:
        log: list[str] = []
        alpha, _ = _sub("Alpha", planner, log)
        main = _Script(
            label="main",
            log=log,
            replies=[
                [("write", '{"path": "x"}'), ("Alpha", '{"task": "find a"}')],
                "Done.",
            ],
        )
        root = Agent(
            name="MainAgent",
            description="the agent",
            instruction="delegate",
            llm_provider=main,
            planner=_planner(planner, parallel_handoffs=True),
            sub_agents=[alpha],
            tools=[_write_tool(log)],
        )
        runner = Runner()
        conversation = f"c_gated_first_{planner}"

        first = await _run(runner, root, conversation)
        second = await _run(runner, root, conversation, _approve())

        assert first[-1].event_type == ExecutionEventType.TOOL_NEEDS_FEEDBACK
        assert _of(first, ExecutionEventType.AGENT_START) == []
        assert log.index("write x") < log.index("start Alpha")
        assert _of(second, ExecutionEventType.AGENT_END) == ["Alpha"]
        assert _answered_by_main(second)
        assert sorted(main.answered(1)) == ["main_1_0", "main_1_1"]

    @pytest.mark.parametrize("planner", PLANNERS)
    @pytest.mark.asyncio
    async def test_an_external_agent_is_a_branch_like_any_other(self, planner: str) -> None:
        log: list[str] = []
        alpha, _ = _sub("Alpha", planner, log, delay=0.3)
        remote = DelegateAgent(
            name="Remote", description="finds r", transport=_Remote("Remote", 0.3, log)
        )
        main = _Script(
            label="main",
            log=log,
            replies=[
                [("Alpha", '{"task": "find a"}'), ("Remote", '{"task": "find r"}')],
                "Done.",
            ],
        )
        root = Agent(
            name="MainAgent",
            description="the agent",
            instruction="delegate",
            llm_provider=main,
            planner=_planner(planner, parallel_handoffs=True),
            sub_agents=[alpha, remote],
        )

        events = await _run(Runner(), root, f"c_remote_{planner}")

        assert _overlapped(log, "Alpha", "Remote"), f"the agents took turns: {log}"
        assert sorted(_of(events, ExecutionEventType.AGENT_END)) == ["Alpha", "Remote"]
        assert _answered_by_main(events)
        said = " ".join(str(m.content or "") for m in main.shown[1])
        assert "Remote found it" in said and "Alpha found it" in said

    @pytest.mark.parametrize("planner", PLANNERS)
    @pytest.mark.asyncio
    async def test_an_external_agent_ending_under_ids_of_its_own_still_ends(
        self, planner: str
    ) -> None:
        # `DelegateAgent` puts the delegation's ids on what the remote runner sends,
        # which is what tells the branch that its agent ended.
        log: list[str] = []
        alpha, _ = _sub("Alpha", planner, log)
        remote = DelegateAgent(
            name="Remote",
            description="finds r",
            transport=_Remote("Remote", 0.0, log, interaction_id="the-remote-runner-s-own"),
        )
        main = _Script(
            label="main",
            log=log,
            replies=[[("Alpha", '{"task": "a"}'), ("Remote", '{"task": "r"}')], "Done."],
        )
        root = Agent(
            name="MainAgent",
            description="the agent",
            instruction="delegate",
            llm_provider=main,
            planner=_planner(planner, parallel_handoffs=True),
            sub_agents=[alpha, remote],
        )

        events = await _run(Runner(), root, f"c_remote_ids_{planner}")

        assert _answered_by_main(events)
        assert sorted(main.answered(1)) == ["main_1_0", "main_1_1"]

    @pytest.mark.parametrize("planner", PLANNERS)
    @pytest.mark.asyncio
    async def test_a_sub_agent_handing_off_in_turn_stays_in_its_branch(self, planner: str) -> None:
        log: list[str] = []
        helper, helper_provider = _sub("Helper", planner, log)
        alpha_provider = _Script(
            label="Alpha", log=log, replies=[[("Helper", '{"task": "help"}')], "Alpha found it"]
        )
        alpha = Agent(
            name="Alpha",
            description="finds a",
            instruction="ask your helper",
            llm_provider=alpha_provider,
            planner=_planner(planner),
            sub_agents=[helper],
        )
        main = _Script(
            label="main",
            log=log,
            replies=[[("Alpha", '{"task": "a"}'), ("Beta", '{"task": "b"}')], "Done."],
        )
        root = Agent(
            name="MainAgent",
            description="the agent",
            instruction="delegate",
            llm_provider=main,
            planner=_planner(planner, parallel_handoffs=True),
            sub_agents=[alpha, _sub("Beta", planner, log)[0]],
        )

        events = await _run(Runner(), root, f"c_inner_handoff_{planner}")

        # Helper returned to Alpha, not to the caller; Alpha was asked again.
        assert len(helper_provider.shown) == 1
        assert len(alpha_provider.shown) == 2
        assert _of(events, ExecutionEventType.AGENT_END).count("Alpha") == 1
        assert _answered_by_main(events)
        assert len(main.shown) == 2
        assert sorted(main.answered(1)) == ["main_1_0", "main_1_1"]


class _ByTask(_Script):
    """A sub-agent whose answer says which task it read; on "task one" it first
    asks to write, which needs approval."""

    async def generate(self, messages: list[LLMMessage], **kwargs: Any) -> LLMResponse:
        self.shown.append(list(messages))
        said = " ".join(str(m.content or "") for m in messages)
        if "task one" in said and not any(m.role == "tool" for m in messages):
            return LLMResponse(
                content="",
                generated_tokens=1,
                execution_time=0.0,
                usage={},
                tool_calls=[
                    LLMToolCall(
                        id=f"w{len(self.shown)}",
                        type="function",
                        function=ToolCallFunction(name="write", arguments='{"path": "x"}'),
                    )
                ],
            )
        task = "one" if "task one" in said else "two"
        return LLMResponse(
            content=f"did task {task}", generated_tokens=1, execution_time=0.0, usage={}
        )


class _SlowSecondReply(_Script):
    """Answers its first request at once and every later one after a while."""

    async def generate(self, messages: list[LLMMessage], **kwargs: Any) -> LLMResponse:
        if self.shown:
            await asyncio.sleep(0.6)
        return await super().generate(messages, **kwargs)


def _stopping_hook() -> FunctionHook:
    async def stop(context: Any, agent: Any) -> HookResult:
        return HookResult(success=True, break_execution=True, name="stop")

    return FunctionHook(
        name="stop", fun=stop, trigger=HookPhase.PRE_AGENT_CALL, type=HookType.LISTENER
    )


class TestCopiesAndNesting:
    @pytest.mark.parametrize("planner", PLANNERS)
    @pytest.mark.asyncio
    async def test_two_copies_of_one_agent_never_read_each_other(self, planner: str) -> None:
        log: list[str] = []
        provider = _ByTask(label="Alpha", log=log)
        alpha = Agent(
            name="Alpha",
            description="does tasks",
            instruction="do the task",
            llm_provider=provider,
            planner=_planner(planner),
            tools=[_write_tool(log)],
        )
        main = _Script(
            label="main",
            log=log,
            replies=[
                [("Alpha", '{"task": "task one"}'), ("Alpha", '{"task": "task two"}')],
                "Done.",
            ],
        )
        root = Agent(
            name="MainAgent",
            description="the agent",
            instruction="delegate",
            llm_provider=main,
            planner=_planner(planner, parallel_handoffs=True),
            sub_agents=[alpha],
        )
        runner = Runner()
        conversation = f"c_copies_{planner}"
        await _run(runner, root, conversation)

        events = await _run(runner, root, conversation, _approve())

        # The copy resumed after its approval reads its own task, not the other's,
        # nor the other's answer, and does the task it was given.
        resumed = " ".join(str(m.content or "") for m in provider.shown[-1])
        assert "task one" in resumed
        assert "task two" not in resumed
        assert "did task two" not in resumed
        answers = [
            " ".join(p.text for p in e.content if p.text)
            for e in events
            if e.event_type == ExecutionEventType.AGENT_END and e.author == "Alpha"
        ]
        assert answers and "did task one" in answers[-1]
        assert _answered_by_main(events)

    @pytest.mark.parametrize("planner", PLANNERS)
    @pytest.mark.asyncio
    async def test_a_branch_of_a_branch_does_not_see_its_caller_s_siblings_at_work(
        self, planner: str
    ) -> None:
        log: list[str] = []

        async def secret(path: str) -> str:
            """Read a secret.

            Args:
                path: the file.
            """
            return "BETA-SECRET"

        # Beta reads its secret at once and answers late: the secret is in the
        # conversation while Beta still runs, and X starts Y meanwhile.
        beta_provider = _SlowSecondReply(
            label="Beta", log=log, replies=[[("secret", '{"path": "x"}')], "Beta done"]
        )
        beta = Agent(
            name="Beta",
            description="b",
            instruction="do",
            llm_provider=beta_provider,
            planner=_planner(planner),
            tools=[FunctionTool(func=secret)],
        )
        y_provider = _Script(label="Y", log=log, replies=["Y done"])
        y = Agent(
            name="Y",
            description="y",
            instruction="do",
            llm_provider=y_provider,
            planner=_planner(planner),
            context_visibility=ContextVisibility.GLOBAL,
        )
        z, _ = _sub("Z", planner, log)
        x = Agent(
            name="X",
            description="x",
            instruction="delegate",
            llm_provider=_Script(
                label="X",
                log=log,
                delay=0.2,
                replies=[[("Y", '{"task": "y"}'), ("Z", '{"task": "z"}')], "X done"],
            ),
            planner=_planner(planner, parallel_handoffs=True),
            sub_agents=[y, z],
        )
        main = _Script(
            label="main",
            log=log,
            replies=[[("X", '{"task": "x"}'), ("Beta", '{"task": "b"}')], "Done."],
        )
        root = Agent(
            name="MainAgent",
            description="the agent",
            instruction="delegate",
            llm_provider=main,
            planner=_planner(planner, parallel_handoffs=True),
            sub_agents=[x, beta],
        )

        events = await _run(Runner(), root, f"c_nested_hidden_{planner}")

        assert _answered_by_main(events)
        assert not any("BETA-SECRET" in str(m.content or "") for m in y_provider.shown[0])

    @pytest.mark.parametrize("planner", PLANNERS)
    @pytest.mark.asyncio
    async def test_a_hook_stopping_one_stops_the_turn_though_another_waits(
        self, planner: str
    ) -> None:
        log: list[str] = []
        alpha, _ = _sub("Alpha", planner, log, writes=True)
        beta, _ = _sub("Beta", planner, log)
        beta.hooks = [_stopping_hook()]
        main = _Script(
            label="main",
            log=log,
            replies=[[("Alpha", '{"task": "a"}'), ("Beta", '{"task": "b"}')], "Done."],
        )
        root = Agent(
            name="MainAgent",
            description="the agent",
            instruction="delegate",
            llm_provider=main,
            planner=_planner(planner, parallel_handoffs=True),
            sub_agents=[alpha, beta],
        )
        runner = Runner()
        conversation = f"c_hook_stop_{planner}"

        events = await _run(runner, root, conversation)

        assert events[-1].event_type != ExecutionEventType.TOOL_NEEDS_FEEDBACK
        assert (
            await runner.conversation_manager.get_last_agent_stack(APP, USER, conversation) is None
        )
        assert len(main.shown) == 1


class TestWhenASubAgentDoesNotEnd:
    @pytest.mark.parametrize("planner", PLANNERS)
    @pytest.mark.asyncio
    async def test_one_that_fails_ends_with_the_error_and_the_other_finishes(
        self, planner: str
    ) -> None:
        root, main, (alpha, _) = _team(planner, delays=(0.0, 0.0))

        async def broken(messages: list[LLMMessage], **kwargs: Any) -> LLMResponse:
            raise RuntimeError("the endpoint is down")

        object.__setattr__(alpha, "generate", broken)

        events = await _run(Runner(), root, f"c_fail_{planner}")

        ends = [e for e in events if e.event_type == ExecutionEventType.AGENT_END]
        failed = [e for e in ends if e.author == "Alpha"]
        assert failed and failed[-1].error_code is not None
        assert "Beta" in [e.author for e in ends]
        assert _answered_by_main(events)
        assert sorted(main.answered(1)) == ["main_1_0", "main_1_1"]

    @pytest.mark.parametrize("planner", PLANNERS)
    @pytest.mark.asyncio
    async def test_a_batch_deeper_than_the_limit_does_not_start(self, planner: str) -> None:
        log: list[str] = []
        inner = [_sub(name, planner, log)[0] for name in ("A1", "A2")]
        alpha_provider = _Script(
            label="Alpha",
            log=log,
            replies=[[("A1", '{"task": "x"}'), ("A2", '{"task": "y"}')], "Alpha gave up"],
        )
        alpha = Agent(
            name="Alpha",
            description="finds a",
            instruction="delegate",
            llm_provider=alpha_provider,
            planner=_planner(planner, parallel_handoffs=True),
            sub_agents=inner,
        )
        main = _Script(
            label="main",
            log=log,
            replies=[[("Alpha", '{"task": "find a"}'), ("Beta", '{"task": "b"}')], "Done."],
        )
        beta, _ = _sub("Beta", planner, log)
        root = Agent(
            name="MainAgent",
            description="the agent",
            instruction="delegate",
            llm_provider=main,
            planner=_planner(planner, parallel_handoffs=True),
            sub_agents=[alpha, beta],
        )

        events = await _run(Runner(max_agent_depth=1), root, f"c_deep_{planner}")

        assert "A1" not in _of(events, ExecutionEventType.AGENT_START)
        said = " ".join(str(m.content or "") for m in alpha_provider.shown[1])
        assert "was not started" in said
        assert sorted(alpha_provider.answered(1)) == ["Alpha_1_0", "Alpha_1_1"]
        assert _answered_by_main(events)

    @pytest.mark.parametrize("planner", PLANNERS)
    @pytest.mark.asyncio
    async def test_after_a_turn_stopped_mid_batch_every_call_still_has_a_result(
        self, planner: str
    ) -> None:
        root, main, _ = _team(planner, delays=(5.0, 5.0))
        main.replies.append("Fine.")
        runner = Runner()
        conversation = f"c_stop_{planner}"

        run = runner.run(
            application_name=APP,
            user_id=USER,
            conversation_id=conversation,
            root=root,
            user_input=_ask(),
        )
        started = 0
        async for event in run:
            if event.event_type == ExecutionEventType.AGENT_START and event.author != "MainAgent":
                started += 1
                if started == 2:
                    break
        await run.aclose()

        # Stopped with the turn, not left running in the background: a consumer
        # walking away closes only the outermost generator.
        others = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
        assert not [t for t in others if "pump" in getattr(t.get_coro(), "__qualname__", "")]
        info = await runner.conversation_manager.get_conversation(APP, USER, conversation)
        assert info is not None
        interrupted = [
            e
            for e in info.events_history
            if e.event_type == ExecutionEventType.AGENT_END
            and e.error_code == "SubAgentInterrupted"
        ]
        assert sorted(e.author for e in interrupted) == ["Alpha", "Beta"]

        events = await _run(runner, root, conversation, _ask("stop that"))

        # As after a hand-off that was stopped: the message goes to the caller, and
        # each of its calls is answered by the start of its sub-agent. It also reads
        # that both were interrupted, rather than only that they were assigned.
        assert _answered_by_main(events)
        assert sorted(main.answered(len(main.shown) - 1)) == ["main_1_0", "main_1_1"]
        shown = " ".join(m.content or "" for m in main.shown[-1])
        assert shown.count("interrupted before it finished") == 2

    @pytest.mark.parametrize("planner", PLANNERS)
    @pytest.mark.asyncio
    async def test_a_turn_cancelled_mid_batch_interrupts_every_sub_agent(
        self, planner: str
    ) -> None:
        # The way the CLI stops a turn on Ctrl-C: the task reading it is cancelled.
        root, main, _ = _team(planner, delays=(5.0, 5.0))
        main.replies.append("Fine.")
        runner = Runner()
        conversation = f"c_cancel_{planner}"
        both_started = asyncio.Event()

        async def consume() -> None:
            started = 0
            async for event in runner.run(
                application_name=APP,
                user_id=USER,
                conversation_id=conversation,
                root=root,
                user_input=_ask(),
            ):
                if (
                    event.event_type == ExecutionEventType.AGENT_START
                    and event.author != "MainAgent"
                ):
                    started += 1
                    if started == 2:
                        both_started.set()

        reading = asyncio.ensure_future(consume())
        await asyncio.wait_for(both_started.wait(), timeout=2)
        reading.cancel()
        with pytest.raises(asyncio.CancelledError):
            await reading

        events = await _run(runner, root, conversation, _ask("stop that"))

        assert _answered_by_main(events)
        shown = " ".join(m.content or "" for m in main.shown[-1])
        assert shown.count("interrupted before it finished") == 2


def _snapshot(source: str, target: str) -> None:
    """What a killed process leaves on disk: the database as it is right now."""
    with sqlite3.connect(source) as src, sqlite3.connect(target) as dst:
        src.backup(dst)


async def _killed_at(
    root: Agent, conversation: str, tmp_path: Path, moment: Any
) -> SqliteConversationManager:
    """Run a turn until ``moment(event)``, take the database as a kill would leave
    it, and return a manager over that copy, as a restarted process would open it."""
    live = str(tmp_path / "live.db")
    left = str(tmp_path / "left.db")
    run = Runner(conversation_manager=SqliteConversationManager(live)).run(
        application_name=APP,
        user_id=USER,
        conversation_id=conversation,
        root=root,
        user_input=_ask(),
    )
    async for event in run:
        if moment(event):
            _snapshot(live, left)
            break
    await run.aclose()
    return SqliteConversationManager(left)


class TestAfterTheProcessDied:
    """A process killed mid-turn runs no ``finally``: the next turn, in a new
    process, closes what it left open."""

    @pytest.mark.parametrize("planner", PLANNERS)
    @pytest.mark.asyncio
    async def test_sub_agents_killed_mid_batch_are_read_as_interrupted(
        self, planner: str, tmp_path: Path
    ) -> None:
        root, main, _ = _team(planner, delays=(5.0, 5.0))
        main.replies.append("Fine.")
        conversation = f"c_killed_{planner}"
        started: list[str] = []

        def both_started(event: BaseEvent) -> bool:
            if event.event_type == ExecutionEventType.AGENT_START and event.author != "MainAgent":
                started.append(event.author or "")
            return len(started) == 2

        manager = await _killed_at(root, conversation, tmp_path, both_started)
        events = await _run(Runner(conversation_manager=manager), root, conversation, _ask("so?"))

        assert _answered_by_main(events)
        shown = " ".join(m.content or "" for m in main.shown[-1])
        assert shown.count("the process ended") == 2
        info = await manager.get_conversation(APP, USER, conversation)
        assert info is not None
        assert RUNNING_TURN_KEY not in (info.state or {})

    @pytest.mark.parametrize("planner", PLANNERS)
    @pytest.mark.asyncio
    async def test_a_tool_killed_while_it_ran_gets_a_result(
        self, planner: str, tmp_path: Path
    ) -> None:
        async def slow(path: str) -> str:
            """Take a while.

            Args:
                path: what to work on.
            """
            await asyncio.sleep(5)
            return "done"

        provider = _Script(label="solo", replies=[[("slow", '{"path": "x"}')], "Fine."])
        agent = Agent(
            name="Solo",
            description="works alone",
            instruction="work",
            llm_provider=provider,
            planner=_planner(planner),
            tools=[FunctionTool(func=slow)],
        )
        conversation = f"c_killed_tool_{planner}"

        manager = await _killed_at(
            agent,
            conversation,
            tmp_path,
            lambda e: e.event_type == ExecutionEventType.TOOL_START,
        )
        events = await _run(Runner(conversation_manager=manager), agent, conversation, _ask("so?"))

        # The call has its result, so the request is one a provider accepts.
        assert provider.calls(len(provider.shown) - 1) == ["solo_1_0"]
        assert provider.answered(len(provider.shown) - 1) == ["solo_1_0"]
        shown = " ".join(m.content or "" for m in provider.shown[-1])
        assert "may or may not have run" in shown
        answers = [e for e in events if e.event_type == ExecutionEventType.RESPONSE]
        assert answers and answers[-1].author == "Solo"

    @pytest.mark.parametrize("planner", PLANNERS)
    @pytest.mark.asyncio
    async def test_a_turn_that_ended_leaves_no_mark(self, planner: str, tmp_path: Path) -> None:
        manager = SqliteConversationManager(str(tmp_path / "c.db"))
        root, _, _ = _team(planner, delays=(0.0, 0.0), writes=(True, False))
        conversation = f"c_mark_{planner}"

        # Ended on a pause: nothing is lost, and the answer still resumes it.
        await _run(Runner(conversation_manager=manager), root, conversation)
        info = await manager.get_conversation(APP, USER, conversation)
        assert info is not None
        assert RUNNING_TURN_KEY not in (info.state or {})


async def _stored(manager: BaseConversationManager, conversation_id: str) -> list[BaseEvent]:
    info = await manager.get_conversation(APP, USER, conversation_id)
    assert info is not None
    return list(info.events_history)


class TestAMessageInsteadOfTheAnswer:
    """A new message where a sub-agent waits for an answer goes to the entrypoint,
    as every message does: what waited is closed, each call once, with a result
    saying it did not run, and each sub-agent with an AGENT_END saying it was
    interrupted. It used to reach the sub-agent on top of the stack, which then
    acted on it before its caller saw it."""

    @pytest.mark.parametrize("parallel", [False, True])
    @pytest.mark.parametrize("planner", PLANNERS)
    @pytest.mark.asyncio
    async def test_the_waiting_call_has_one_result(self, planner: str, parallel: bool) -> None:
        log: list[str] = []
        root, main, (alpha, _) = _team(
            planner, log, parallel=parallel, delays=(0.0, 0.0), writes=(True, False)
        )
        runner = Runner()
        conversation = f"c_one_result_{planner}_{parallel}"
        await _run(runner, root, conversation)

        await _run(runner, root, conversation, _ask("never mind"))

        results = [
            e
            for e in await _stored(runner.conversation_manager, conversation)
            if e.event_type in (ExecutionEventType.TOOL_END, ExecutionEventType.TOOL_ERROR)
            and getattr(e.action, "call_id", None) == "Alpha_1_0"
        ]
        assert len(results) == 1
        assert "was not executed" in str(results[0].action.observation)
        assert len(alpha.shown) == 1
        assert _every_call_answered(main, len(main.shown) - 1)

    @pytest.mark.parametrize("planner", PLANNERS)
    @pytest.mark.asyncio
    async def test_a_single_sub_agent_handed_off_to_is_closed_too(self, planner: str) -> None:
        log: list[str] = []
        root, main = _caller(planner, [ALPHA], parallel=False, log=log, alpha_writes=True)
        alpha = next(a for a in root.sub_agents if a.name == "Alpha")
        runner = Runner()
        conversation = f"c_single_{planner}"
        first = await _run(runner, root, conversation)
        assert first[-1].event_type == ExecutionEventType.TOOL_NEEDS_FEEDBACK

        events = await _run(runner, root, conversation, _ask("never mind"))

        assert len(alpha.llm_provider.shown) == 1
        assert _of(events, ExecutionEventType.AGENT_END) == ["Alpha"]
        assert "never mind" in " ".join(str(m.content or "") for m in main.shown[1])
        assert _answered_by_main(events)
        assert _every_call_answered(main, 1)

    @pytest.mark.parametrize("planner", PLANNERS)
    @pytest.mark.asyncio
    async def test_every_sub_agent_waiting_is_closed(self, planner: str) -> None:
        log: list[str] = []
        root, main, (alpha, beta) = _team(planner, log, delays=(0.0, 0.0), writes=(True, True))
        runner = Runner()
        conversation = f"c_every_waiting_{planner}"
        await _run(runner, root, conversation)

        events = await _run(runner, root, conversation, _ask("never mind"))

        assert sorted(_of(events, ExecutionEventType.AGENT_END)) == ["Alpha", "Beta"]
        assert len(alpha.shown) == len(beta.shown) == 1
        assert not [x for x in log if x.startswith("write")]
        assert _answered_by_main(events)
        assert sorted(main.answered(1)) == ["main_1_0", "main_1_1"]
        assert not (await _state(runner.conversation_manager, conversation)).get(
            WAITING_BRANCHES_KEY
        )


def _splitting(name: str, planner: str, log: list[str], subs: list[Agent]) -> Agent:
    """An agent that hands both of its sub-agents a task in one response."""
    calls = [(sub.name, f'{{"task": "{sub.name.lower()}"}}') for sub in subs]
    return Agent(
        name=name,
        description=f"splits {name.lower()}'s work",
        instruction="delegate",
        llm_provider=_Script(label=name, log=log, replies=[calls, f"{name} done"]),
        planner=_planner(planner, parallel_handoffs=True),
        sub_agents=subs,
    )


class TestAMessageWhereNestedBatchesWait:
    """Two batches two levels down, all four of their sub-agents waiting, and a new
    message: every one ends as interrupted and every call is closed. Only the
    batch on the conversation's stack used to be: a branch waiting beside it was
    ended alone, its own sub-agents left waiting, their calls without a result."""

    @pytest.mark.parametrize("planner", PLANNERS)
    @pytest.mark.asyncio
    async def test_nothing_is_left_waiting_or_without_a_result(self, planner: str) -> None:
        log: list[str] = []
        x = _splitting(
            "X",
            planner,
            log,
            [_sub("Y", planner, log, writes=True)[0], _sub("Z", planner, log, writes=True)[0]],
        )
        w = _splitting(
            "W",
            planner,
            log,
            [_sub("U", planner, log, writes=True)[0], _sub("V", planner, log, writes=True)[0]],
        )
        root = _splitting("MainAgent", planner, log, [x, w])
        runner = Runner()
        conversation = f"c_nested_message_{planner}"
        first = await _run(runner, root, conversation)
        assert first[-1].event_type == ExecutionEventType.TOOL_NEEDS_FEEDBACK

        events = await _run(runner, root, conversation, _ask("never mind"))

        assert sorted(_of(events, ExecutionEventType.AGENT_END)) == ["U", "V", "W", "X", "Y", "Z"]
        assert not [x for x in log if x.startswith("write")]
        assert _answered_by_main(events)
        stored = await _stored(runner.conversation_manager, conversation)
        made = {
            e.action.call_id
            for e in stored
            if e.event_type in (ExecutionEventType.TOOL_START, ExecutionEventType.AGENT_CALL)
        }
        closed = {
            e.action.call_id
            for e in stored
            if e.event_type
            in (
                ExecutionEventType.TOOL_END,
                ExecutionEventType.TOOL_ERROR,
                ExecutionEventType.AGENT_START,
            )
            and getattr(e.action, "call_id", None)
        }
        assert made <= closed, made - closed
        assert not (await _state(runner.conversation_manager, conversation)).get(
            WAITING_BRANCHES_KEY
        )


def _look_tool(log: list[str]) -> FunctionTool:
    async def look(what: str) -> str:
        """Look something up.

        Args:
            what: the thing.
        """
        log.append(f"look {what}")
        return f"looked at {what}"

    return FunctionTool(func=look)


def _caller(
    planner: str,
    calls: list[tuple[str, str]],
    parallel: bool,
    log: list[str],
    alpha_writes: bool = False,
) -> tuple[Agent, _Script]:
    """A caller with a tool of its own (`look`) whose first response makes ``calls``."""
    main = _Script(label="main", log=log, replies=[calls, "Done."])
    subs = [
        _sub("Alpha", planner, log, delay=0.2, writes=alpha_writes)[0],
        _sub("Beta", planner, log, delay=0.2)[0],
    ]
    root = Agent(
        name="MainAgent",
        description="the agent",
        instruction="delegate",
        llm_provider=main,
        planner=_planner(planner, parallel_handoffs=parallel),
        sub_agents=subs,
        tools=[_look_tool(log)],
    )
    return root, main


def _every_call_answered(main: _Script, request: int) -> bool:
    return sorted(main.calls(request)) == sorted(main.answered(request))


ALPHA = ("Alpha", '{"task": "find a"}')
BETA = ("Beta", '{"task": "find b"}')
LOOK = ("look", '{"what": "x"}')
NOPE = ("nope", "{}")


class TestNoCallOfAResponseIsDropped:
    """Every call of a response that names a sub-agent runs, and the caller's next
    request has a result for each: a call after a hand-off used to be dropped, with
    nothing telling the model it never ran."""

    @pytest.mark.parametrize("planner", PLANNERS)
    @pytest.mark.asyncio
    async def test_off_the_sub_agents_run_one_after_the_other(self, planner: str) -> None:
        log: list[str] = []
        root, main = _caller(planner, [ALPHA, BETA], parallel=False, log=log)

        events = await _run(Runner(), root, f"c_off_both_{planner}")

        assert _of(events, ExecutionEventType.AGENT_START) == ["Alpha", "Beta"]
        assert log.index("end Alpha") < log.index("start Beta")
        assert main.calls(1) == ["main_1_0", "main_1_1"]
        assert _every_call_answered(main, 1)
        assert _answered_by_main(events)

    @pytest.mark.parametrize("planner", PLANNERS)
    @pytest.mark.asyncio
    async def test_off_a_tool_after_a_sub_agent_runs_once_it_has_ended(self, planner: str) -> None:
        log: list[str] = []
        root, main = _caller(planner, [ALPHA, LOOK], parallel=False, log=log)

        await _run(Runner(), root, f"c_off_tool_{planner}")

        assert log.index("end Alpha") < log.index("look x")
        assert _every_call_answered(main, 1)

    @pytest.mark.parametrize("planner", PLANNERS)
    @pytest.mark.asyncio
    async def test_off_one_at_a_time_has_no_depth_limit(self, planner: str) -> None:
        root, main = _caller(planner, [ALPHA, BETA], parallel=False, log=[])

        events = await _run(Runner(max_agent_depth=0), root, f"c_off_depth_{planner}")

        assert _of(events, ExecutionEventType.AGENT_START) == ["Alpha", "Beta"]
        assert _every_call_answered(main, 1)

    @pytest.mark.parametrize("planner", PLANNERS)
    @pytest.mark.asyncio
    async def test_off_one_sub_agent_alone_is_a_hand_off_as_before(self, planner: str) -> None:
        root, main = _caller(planner, [ALPHA], parallel=False, log=[])

        events = await _run(Runner(), root, f"c_off_one_{planner}")

        [call] = [e for e in events if e.event_type == ExecutionEventType.AGENT_CALL]
        assert METADATA_TOOL_BATCH not in call.metadata
        assert _every_call_answered(main, 1)

    @pytest.mark.parametrize("planner", PLANNERS)
    @pytest.mark.asyncio
    async def test_off_the_calls_after_a_paused_sub_agent_wait_for_it(self, planner: str) -> None:
        log: list[str] = []
        root, main = _caller(planner, [ALPHA, BETA], parallel=False, log=log, alpha_writes=True)
        runner = Runner()
        conversation = f"c_off_pause_{planner}"

        first = await _run(runner, root, conversation)
        assert _of(first, ExecutionEventType.AGENT_START) == ["Alpha"]
        assert "start Beta" not in log

        second = await _run(runner, root, conversation, _approve())

        assert "write alpha.txt" in log
        assert log.index("write alpha.txt") < log.index("start Beta")
        assert _of(second, ExecutionEventType.AGENT_START) == ["Beta"]
        assert _every_call_answered(main, 1)
        assert _answered_by_main(second)

    @pytest.mark.parametrize("planner", PLANNERS)
    @pytest.mark.asyncio
    async def test_on_a_tool_it_lacks_does_not_split_the_sub_agents(self, planner: str) -> None:
        log: list[str] = []
        root, main = _caller(planner, [ALPHA, NOPE, BETA], parallel=True, log=log)

        events = await _run(Runner(), root, f"c_on_nope_{planner}")

        assert _of(events, ExecutionEventType.AGENT_START) == ["Alpha", "Beta"]
        assert log.index("start Beta") < log.index("end Alpha")
        assert _every_call_answered(main, 1)
        [result] = [m for m in main.shown[1] if m.tool_call_id == "main_1_1"]
        assert "not found" in str(result.content)

    @pytest.mark.parametrize("planner", PLANNERS)
    @pytest.mark.asyncio
    async def test_on_the_calls_after_the_sub_agents_run_with_them(self, planner: str) -> None:
        log: list[str] = []
        root, main = _caller(planner, [ALPHA, BETA, NOPE, LOOK], parallel=True, log=log)

        await _run(Runner(), root, f"c_on_after_{planner}")

        assert "look x" in log
        assert main.calls(1) == ["main_1_0", "main_1_1", "main_1_2", "main_1_3"]
        assert _every_call_answered(main, 1)

    @pytest.mark.parametrize("planner", PLANNERS)
    @pytest.mark.asyncio
    async def test_a_tool_it_lacks_among_tools_only_is_answered_as_before(
        self, planner: str
    ) -> None:
        log: list[str] = []
        root, main = _caller(planner, [LOOK, NOPE, ("look", '{"what": "y"}')], True, log)

        runner = Runner()
        conversation = f"c_tools_nope_{planner}"
        await _run(runner, root, conversation)

        info = await runner.conversation_manager.get_conversation(APP, USER, conversation)
        assert info is not None
        [nope] = [
            e
            for e in info.events_history
            if e.event_type == ExecutionEventType.TOOL_START and e.action.name == "nope"
        ]
        assert METADATA_TOOL_BATCH not in nope.metadata
        assert {"look x", "look y"} <= set(log)
        assert _every_call_answered(main, 1)


class TestItsDefault:
    """On by default for the tool-use planners; off where it cannot work, without
    refusing a configuration that never asked for it."""

    @pytest.mark.parametrize("planner", PLANNERS)
    def test_on_unless_asked_otherwise(self, planner: str) -> None:
        assert _planner(planner).parallel_handoffs is True
        assert _planner(planner, parallel_handoffs=False).parallel_handoffs is False

    def test_off_for_react(self) -> None:
        assert _planner("react").parallel_handoffs is False

    @pytest.mark.parametrize("planner", PLANNERS)
    def test_off_with_batches_off_when_not_asked_for(self, planner: str) -> None:
        assert _planner(planner, parallel_tool_calls=False).parallel_handoffs is False

    @pytest.mark.parametrize("planner", PLANNERS)
    @pytest.mark.asyncio
    async def test_an_agent_that_names_no_option_runs_them_together(self, planner: str) -> None:
        log: list[str] = []
        root, _, _ = _team(planner, log)
        root.planner = _planner(planner)

        events = await _run(Runner(), root, f"c_default_{planner}")

        calls = [e for e in events if e.event_type == ExecutionEventType.AGENT_CALL]
        assert all(METADATA_TOOL_BATCH in c.metadata for c in calls)
        assert not (calls[0].metadata[METADATA_TOOL_BATCH]).get("sequential")
        assert log.index("start Beta") < log.index("end Alpha")


class TestWhatRefusesIt:
    def test_react_with_parallel_handoffs_does_not_load(self) -> None:
        with pytest.raises(ValueError, match="parallel_handoffs"):
            _planner("react", parallel_handoffs=True)

    @pytest.mark.parametrize("planner", PLANNERS)
    def test_it_does_not_load_without_the_batch_it_needs(self, planner: str) -> None:
        with pytest.raises(ValueError, match="parallel_tool_calls"):
            _planner(planner, parallel_handoffs=True, parallel_tool_calls=False)


class _Counting(_Script):
    """A sub-agent whose provider counts each request as a multiple of the estimate,
    a different one per task, so a correction says which request it was measured on."""

    writes: bool = False

    async def generate(self, messages: list[LLMMessage], **kwargs: Any) -> LLMResponse:
        self.shown.append(list(messages))
        said = " ".join(str(m.content or "") for m in messages)
        factor = 3.0 if "task one" in said else 2.0 if "task two" in said else 1.5
        estimator = HeuristicEstimator()
        estimated = estimator.estimate_messages(messages) + estimator.estimate_tool_definitions(
            kwargs.get("tools") or []
        )
        usage = {"prompt_tokens": int(estimated * factor)}
        if self.writes and "task one" in said and not any(m.role == "tool" for m in messages):
            return LLMResponse(
                content="",
                generated_tokens=1,
                execution_time=0.0,
                usage=usage,
                tool_calls=[
                    LLMToolCall(
                        id=f"w{len(self.shown)}",
                        type="function",
                        function=ToolCallFunction(name="write", arguments='{"path": "x"}'),
                    )
                ],
            )
        return LLMResponse(content="done", generated_tokens=1, execution_time=0.0, usage=usage)


def _corrections(events: list[BaseEvent]) -> list[int]:
    """The measured correction each request of a copy of Alpha went out with."""
    return [
        e.metadata[CONTEXT_USAGE_METADATA_KEY]["correction_tokens"]
        for e in events
        if e.event_type == ExecutionEventType.LLM_START and e.author == "Alpha"
    ]


def _copies(
    planner: str, replies: list[Any], writes: bool = False, parallel: bool = False
) -> tuple[Agent, _Counting]:
    provider = _Counting(label="Alpha", writes=writes)
    alpha = Agent(
        name="Alpha",
        description="does tasks",
        instruction="do the task",
        llm_provider=provider,
        planner=_planner(planner),
        tools=[_write_tool([])] if writes else [],
    )
    root = Agent(
        name="MainAgent",
        description="the agent",
        instruction="delegate",
        llm_provider=_Script(label="main", replies=replies),
        planner=_planner(planner, parallel_handoffs=parallel),
        sub_agents=[alpha],
    )
    return root, provider


TWO_COPIES = [("Alpha", '{"task": "task one"}'), ("Alpha", '{"task": "task two"}')]


class TestCopiesCompactApart:
    """Two copies of one agent see different histories, so what compaction
    measured or decided on one must not be applied to the other."""

    @pytest.mark.parametrize("planner", PLANNERS)
    @pytest.mark.asyncio
    async def test_a_copy_is_not_measured_on_the_other_s_request(self, planner: str) -> None:
        root, _ = _copies(planner, [TWO_COPIES, "Done."])

        events = await _run(Runner(), root, f"c_measured_{planner}")

        # Alpha had never run: neither copy has a count to start from.
        assert _corrections(events) == [0, 0]

    @pytest.mark.parametrize("planner", PLANNERS)
    @pytest.mark.asyncio
    async def test_copies_start_from_what_the_agent_had_and_leave_it_as_it_was(
        self, planner: str
    ) -> None:
        root, _ = _copies(
            planner, [[("Alpha", '{"task": "prepare"}')], "Ready.", TWO_COPIES, "Done."]
        )
        runner = Runner()
        conversation = f"c_inherited_{planner}"
        await _run(runner, root, conversation)
        before = (await _state(runner.conversation_manager, conversation))["context_anchors"]
        anchor = before["Alpha"]
        measured = anchor["reported"] - anchor["estimated"]
        assert measured > 0

        events = await _run(runner, root, conversation)

        assert _corrections(events) == [measured, measured]
        after = (await _state(runner.conversation_manager, conversation))["context_anchors"]
        assert after["Alpha"] == anchor

    @pytest.mark.parametrize("planner", PLANNERS)
    @pytest.mark.asyncio
    async def test_a_copy_resumed_after_its_approval_is_measured_on_its_own_request(
        self, planner: str
    ) -> None:
        root, _ = _copies(planner, [TWO_COPIES, "Done."], writes=True, parallel=True)
        runner = Runner()
        conversation = f"c_resumed_{planner}"
        await _run(runner, root, conversation)
        anchors = (await _state(runner.conversation_manager, conversation))["context_anchors"]
        # Task one is counted at three times the estimate, task two at two.
        [own] = [a for a in anchors.values() if a["reported"] > 2.5 * a["estimated"]]

        events = await _run(runner, root, conversation, _approve())

        # The copy that waited goes on from its own count, not from the other's.
        assert _corrections(events) == [own["reported"] - own["estimated"]]


def _copy_keys(state: dict[str, Any]) -> set[str]:
    """The compaction records and anchors filed under a copy's own key."""
    return {
        key
        for table in ("compaction_records", "context_anchors")
        for key in state.get(table) or {}
        if "#copy:" in key
    }


class TestCopiesLeaveNothingBehind:
    """A copy's record and anchor are never read again once it ends (its key
    names its own interaction), so they are dropped then, not kept forever."""

    @pytest.mark.parametrize("planner", PLANNERS)
    @pytest.mark.asyncio
    async def test_copies_that_end_drop_their_keys(self, planner: str) -> None:
        root, _ = _copies(planner, [TWO_COPIES, "Done."], parallel=True)
        runner = Runner()
        conversation = f"c_dropped_{planner}"

        await _run(runner, root, conversation)

        assert _copy_keys(await _state(runner.conversation_manager, conversation)) == set()

    @pytest.mark.parametrize("planner", PLANNERS)
    @pytest.mark.asyncio
    async def test_a_copy_waiting_keeps_its_key_until_it_ends(self, planner: str) -> None:
        root, _ = _copies(planner, [TWO_COPIES, "Done."], writes=True, parallel=True)
        runner = Runner()
        conversation = f"c_kept_{planner}"

        await _run(runner, root, conversation)
        waiting = _copy_keys(await _state(runner.conversation_manager, conversation))
        events = await _run(runner, root, conversation, _approve())

        assert len(waiting) == 1
        assert _answered_by_main(events)
        assert _copy_keys(await _state(runner.conversation_manager, conversation)) == set()


class TestTheBranchesShareTheConversationState:
    @pytest.mark.parametrize("planner", PLANNERS)
    @pytest.mark.asyncio
    async def test_what_a_sub_agent_writes_there_is_kept(self, planner: str) -> None:
        """Each branch used to get its own copy of the state: a key it added was
        never seen by its siblings or its caller, and was lost."""
        beta = _Counting(label="Beta")
        root, _ = _copies(
            planner,
            [[("Alpha", '{"task": "a"}'), ("Beta", '{"task": "b"}')], "Done."],
            parallel=True,
        )
        root.sub_agents.append(
            Agent(
                name="Beta",
                description="does other tasks",
                instruction="do the task",
                llm_provider=beta,
                planner=_planner(planner),
            )
        )
        runner = Runner()
        conversation = f"c_shared_state_{planner}"
        await _run(runner, root, conversation)

        anchors = (await _state(runner.conversation_manager, conversation)).get("context_anchors")
        assert set(anchors or {}) == {"Alpha", "Beta"}


class _NoIds(_Script):
    """A provider whose tool calls carry no id, as some endpoints send them."""

    async def generate(self, messages: list[LLMMessage], **kwargs: Any) -> LLMResponse:
        response = await super().generate(messages, **kwargs)
        for call in response.tool_calls or []:
            call.id = ""
        return response


def _ids_sent(provider: _Script, request: int) -> list[str]:
    """Every call id the request carried, on the calls and on their results."""
    return provider.calls(request) + [
        m.tool_call_id or "" for m in provider.shown[request] if m.role == "tool"
    ]


def _well_formed(call_id: str) -> bool:
    # What the strictest providers accept: OpenAI at most 40 characters, Anthropic
    # letters, digits, `_` and `-` only.
    return 0 < len(call_id) <= 40 and re.fullmatch(r"[A-Za-z0-9_-]+", call_id) is not None


class TestCallsWithoutAnId:
    """A response whose calls have no id: each gets one, so none is lost, no call
    runs with another's arguments, and the provider is never sent an empty id."""

    @pytest.mark.parametrize("parallel", [True, False])
    @pytest.mark.parametrize("planner", PLANNERS)
    @pytest.mark.asyncio
    async def test_both_sub_agents_run_with_their_own_task(
        self, planner: str, parallel: bool
    ) -> None:
        log: list[str] = []
        root, _ = _caller(planner, [ALPHA, BETA], parallel=parallel, log=log)
        main = _NoIds(label="main", log=log, replies=[[ALPHA, BETA], "Done."])
        root.llm_provider = main

        events = await _run(Runner(), root, f"c_no_ids_{planner}_{parallel}")

        tasks = {
            e.author: e.action.parameters.get("task")
            for e in events
            if e.event_type == ExecutionEventType.AGENT_START and e.author != "MainAgent"
        }
        assert tasks == {"Alpha": "find a", "Beta": "find b"}
        assert _answered_by_main(events)
        assert _every_call_answered(main, 1)
        assert all(_well_formed(call_id) for call_id in _ids_sent(main, 1))

    @pytest.mark.parametrize("planner", PLANNERS)
    @pytest.mark.asyncio
    async def test_two_tools_keep_their_own_arguments(self, planner: str) -> None:
        log: list[str] = []
        root, _ = _caller(planner, [], parallel=True, log=log)
        main = _NoIds(
            label="main",
            log=log,
            replies=[[("look", '{"what": "x"}'), ("look", '{"what": "y"}')], "Done."],
        )
        root.llm_provider = main

        await _run(Runner(), root, f"c_no_ids_tools_{planner}")

        assert sorted(x for x in log if x.startswith("look")) == ["look x", "look y"]
        sent = _ids_sent(main, 1)
        assert len(set(main.calls(1))) == 2
        assert all(_well_formed(call_id) for call_id in sent)
        assert sorted(main.calls(1)) == sorted(main.answered(1))


def _ending(planner: str, first: list[tuple[str, str]], log: list[str]) -> tuple[Agent, _Script]:
    """An agent ending its step (``use_stop``) in a response that asks for ``first``."""
    main = _Script(label="main", log=log, replies=[first, "Done."])
    root = Agent(
        name="MainAgent",
        description="the agent",
        instruction="do it",
        llm_provider=main,
        planner=_planner(planner, use_stop=True),
        tools=[_look_tool(log)],
    )
    return root, main


class TestCallsAfterTheStepEnds:
    """Calls a response makes after the one that ends the step (STOP, a question to
    the user) do not run, but each gets a result saying so: the model is told,
    instead of believing they ran."""

    @pytest.mark.parametrize(
        "ender",
        [("STOP", '{"info": "done"}'), ("ASK_USER_MISSING_INFO", '{"text": "which one?"}')],
    )
    @pytest.mark.parametrize("planner", PLANNERS)
    @pytest.mark.asyncio
    async def test_they_do_not_run_and_the_model_is_told(
        self, planner: str, ender: tuple[str, str]
    ) -> None:
        log: list[str] = []
        root, main = _ending(planner, [ender, LOOK, LOOK], log)
        runner = Runner()
        conversation = f"c_after_{planner}_{ender[0]}"

        await _run(runner, root, conversation)
        await _run(runner, root, conversation, _ask("go on"))

        assert not [x for x in log if x.startswith("look")]
        results = [str(m.content) for m in main.shown[1] if m.role == "tool" and m.name == "look"]
        assert len(results) == 2
        assert all("was not executed" in r and ender[0] in r for r in results)
        assert _every_call_answered(main, 1)

    @pytest.mark.parametrize("planner", PLANNERS)
    @pytest.mark.asyncio
    async def test_the_calls_before_it_still_run(self, planner: str) -> None:
        log: list[str] = []
        root, main = _ending(planner, [LOOK, ("STOP", '{"info": "done"}'), LOOK], log)
        runner = Runner()
        conversation = f"c_before_{planner}"

        await _run(runner, root, conversation)
        await _run(runner, root, conversation, _ask("go on"))

        assert [x for x in log if x.startswith("look")] == ["look x"]
        assert _every_call_answered(main, 1)


class _Gauge:
    """How many tool calls run at once, at most."""

    def __init__(self) -> None:
        self.now = 0
        self.most = 0

    def tool(self, log: list[str]) -> FunctionTool:
        async def slow(what: str) -> str:
            """Take a while.

            Args:
                what: the thing.
            """
            self.now += 1
            self.most = max(self.most, self.now)
            log.append(f"slow {what}")
            await asyncio.sleep(0.05)
            self.now -= 1
            return what

        return FunctionTool(func=slow)


def _busy_team(planner: str, gauge: _Gauge, log: list[str]) -> Agent:
    """Two sub-agents run together, each asking for three slow calls at once."""

    def worker(name: str) -> Agent:
        calls = [("slow", f'{{"what": "{name.lower()}{i}"}}') for i in range(3)]
        return Agent(
            name=name,
            description=f"works on {name.lower()}",
            instruction="work",
            llm_provider=_Script(label=name, log=log, replies=[calls, f"{name} done"]),
            planner=_planner(planner),
            tools=[gauge.tool(log)],
        )

    return Agent(
        name="MainAgent",
        description="the agent",
        instruction="delegate",
        llm_provider=_Script(label="main", log=log, replies=[[ALPHA, BETA], "Done."]),
        planner=_planner(planner, parallel_handoffs=True),
        sub_agents=[worker("Alpha"), worker("Beta")],
    )


class TestOneLimitForTheTurn:
    """``max_parallel_tool_calls`` is the most tool calls running at once in the
    whole turn, sub-agents running together included; a sub-agent takes no place
    of its own, so one waiting for its calls never holds up another's."""

    @pytest.mark.parametrize("planner", PLANNERS)
    @pytest.mark.asyncio
    async def test_sub_agents_together_share_it(self, planner: str) -> None:
        gauge, log = _Gauge(), []
        root = _busy_team(planner, gauge, log)

        events = await _run(Runner(max_parallel_tool_calls=2), root, f"c_limit_{planner}")

        assert len([x for x in log if x.startswith("slow")]) == 6
        assert gauge.most == 2
        assert _answered_by_main(events)

    @pytest.mark.parametrize("planner", PLANNERS)
    @pytest.mark.asyncio
    async def test_with_one_place_the_sub_agents_still_run_together(self, planner: str) -> None:
        gauge, log = _Gauge(), []
        root = _busy_team(planner, gauge, log)

        events = await asyncio.wait_for(
            _run(Runner(max_parallel_tool_calls=1), root, f"c_one_place_{planner}"), timeout=10
        )

        assert gauge.most == 1
        # Beta is not held back until Alpha's calls are done: it holds no place.
        assert log.index("start Beta") < log.index("slow alpha2")
        assert _answered_by_main(events)


class TestTheRequestContextReachesEveryBranch:
    """What a host binds to the request — log fields, the trace a gateway started —
    lives in context variables, and a branch runs in a task of its own: asyncio
    copies the context into a task when it creates it, so every sub-agent running
    together sees the caller's, and the calls it makes carry the same trace."""

    @pytest.mark.parametrize("planner", PLANNERS)
    @pytest.mark.asyncio
    async def test_log_fields_and_trace_reach_the_tools_of_both(self, planner: str) -> None:
        # The `tracing` extra: without it there is no trace to follow.
        pytest.importorskip("opentelemetry.sdk.trace")
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider

        from domyn.agents.logger import bind_log_context, get_log_context

        seen: dict[str, tuple[dict[str, Any], int]] = {}

        def probe(owner: str) -> FunctionTool:
            async def probe_context(what: str) -> str:
                """Note what this call runs under.

                Args:
                    what: anything.
                """
                span = trace.get_current_span().get_span_context()
                seen[owner] = (dict(get_log_context()), span.trace_id)
                return "noted"

            probe_context.__name__ = probe_context.__qualname__ = f"probe_{owner.lower()}"
            return FunctionTool(func=probe_context)

        def worker(name: str) -> Agent:
            return Agent(
                name=name,
                description=f"works on {name.lower()}",
                instruction="work",
                llm_provider=_Script(
                    label=name,
                    replies=[[(f"probe_{name.lower()}", '{"what": "x"}')], f"{name} done"],
                ),
                planner=_planner(planner),
                tools=[probe(name)],
            )

        root = Agent(
            name="MainAgent",
            description="the agent",
            instruction="delegate",
            llm_provider=_Script(label="main", replies=[[ALPHA, BETA], "Done."]),
            planner=_planner(planner, parallel_handoffs=True),
            sub_agents=[worker("Alpha"), worker("Beta")],
        )
        tracer = TracerProvider().get_tracer("test")

        with (
            tracer.start_as_current_span("request") as request,
            bind_log_context(conversation_id="c_ctx", request_id="r1"),
        ):
            events = await _run(Runner(), root, f"c_ctx_{planner}")

        assert _answered_by_main(events)
        trace_id = request.get_span_context().trace_id
        for name in ("Alpha", "Beta"):
            fields, seen_trace = seen[name]
            assert fields["request_id"] == "r1"
            assert seen_trace == trace_id
