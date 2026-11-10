"""Several sub-agents asked for in one response (`parallel_handoffs`).

- On, they run at the same time, each as a branch of the same conversation, and
  the caller resumes once all of them have ended, with every call answered and
  their results in the order they were called.
- A sub-agent that needs an approval stops there while the others finish; the
  questions are asked one at a time, each answer resumes the sub-agent that asked,
  and the caller resumes once the last has ended — after a restart too.
- A new message instead of the answer closes what had not ended, so the caller's
  next request still has a result for every call it made.
- Off, a response naming several runs the first only, as a hand-off always did.
"""

# No `from __future__ import annotations`: builds real agents and tools.
import asyncio
import time
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
from domyn.agents.planner_strategy import PlannerStrategyFactory
from domyn.agents.runner import Runner
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

        started = time.monotonic()
        events = await _run(Runner(), root, f"c_together_{planner}")
        elapsed = time.monotonic() - started

        assert elapsed < 0.55, f"took {elapsed:.2f}s: the sub-agents ran one after another"
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
    async def test_a_new_message_goes_to_it_as_to_any_sub_agent_waiting(self, planner: str) -> None:
        log: list[str] = []
        root, main, (alpha, _) = _team(planner, log, delays=(0.0, 0.0), writes=(True, False))
        runner = Runner()
        conversation = f"c_new_message_{planner}"
        await _run(runner, root, conversation)

        events = await _run(runner, root, conversation, _ask("never mind"))

        # Alpha, on top of the stack, is the one the message reaches; its call to
        # write is closed as not run; it ends and the caller resumes.
        assert "write alpha.txt" not in log
        assert len(alpha.shown) == 2
        assert "never mind" in " ".join(str(m.content or "") for m in alpha.shown[1])
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

        started = time.monotonic()
        events = await _run(Runner(), root, f"c_mixed_{planner}")
        elapsed = time.monotonic() - started

        assert elapsed < 0.55, f"took {elapsed:.2f}s: the tool and the sub-agent took turns"
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

        started = time.monotonic()
        events = await _run(Runner(), root, f"c_tool_last_{planner}")
        elapsed = time.monotonic() - started

        assert elapsed < 0.55, f"took {elapsed:.2f}s: the sub-agent and the tool took turns"
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

        started = time.monotonic()
        events = await _run(Runner(), root, f"c_remote_{planner}")
        elapsed = time.monotonic() - started

        assert elapsed < 0.55, f"took {elapsed:.2f}s: the agents took turns"
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

        events = await _run(runner, root, conversation, _ask("stop that"))

        # As after a hand-off that was stopped: the message goes to the caller, and
        # each of its calls is answered by the start of its sub-agent.
        assert _answered_by_main(events)
        assert sorted(main.answered(len(main.shown) - 1)) == ["main_1_0", "main_1_1"]


class TestAMessageInsteadOfTheAnswer:
    """A sub-agent paused on its own call and given a new message instead of the
    answer: its next request closes that call once, with the result saying it did
    not run. Its new start used to close it too, so the call had two results."""

    @pytest.mark.parametrize("parallel", [False, True])
    @pytest.mark.parametrize("planner", PLANNERS)
    @pytest.mark.asyncio
    async def test_the_waiting_call_has_one_result(self, planner: str, parallel: bool) -> None:
        log: list[str] = []
        root, _, (alpha, _) = _team(
            planner, log, parallel=parallel, delays=(0.0, 0.0), writes=(True, False)
        )
        runner = Runner()
        conversation = f"c_one_result_{planner}_{parallel}"
        await _run(runner, root, conversation)

        await _run(runner, root, conversation, _ask("never mind"))

        assert alpha.answered(1) == ["Alpha_1_0"]
        request = alpha.shown[1]
        position = next(i for i, m in enumerate(request) if m.tool_calls)
        assert request[position + 1].tool_call_id == "Alpha_1_0"
        assert "was not executed" in str(request[position + 1].content)
        assert "never mind" in " ".join(str(m.content or "") for m in request)


class TestOffIsAsBefore:
    @pytest.mark.parametrize("planner", PLANNERS)
    @pytest.mark.asyncio
    async def test_only_the_first_sub_agent_runs(self, planner: str) -> None:
        root, _, _ = _team(planner, parallel=False)

        events = await _run(Runner(), root, f"c_off_{planner}")

        assert _of(events, ExecutionEventType.AGENT_START) == ["Alpha"]


class TestWhatRefusesIt:
    def test_react_with_parallel_handoffs_does_not_load(self) -> None:
        with pytest.raises(ValueError, match="parallel_handoffs"):
            _planner("react", parallel_handoffs=True)

    @pytest.mark.parametrize("planner", PLANNERS)
    def test_it_does_not_load_without_the_batch_it_needs(self, planner: str) -> None:
        with pytest.raises(ValueError, match="parallel_tool_calls"):
            _planner(planner, parallel_handoffs=True, parallel_tool_calls=False)
