"""The approval gate, driven through a real runner.

These exist because the interesting failures are not in the policy — that is
pure and easy to test — but in the wiring: a grant that is stored against the
proposed parameters instead of the approved ones, a denial that suspends a turn
nobody will ever resume, a gate that is present on one code path and missing on
the other.

One of those was real. Adding a second prompt to the approval flow turned it
into an unbounded loop, and the loop was only visible end-to-end: every unit
around it passed.
"""

# No `from __future__ import annotations`: FunctionTool introspects the tool
# signature at construction time and cannot resolve string annotations.
import json
from typing import Any

import pytest

from domyn.agents import Agent, Runner
from domyn.agents.approvals import (
    ApprovalPolicy,
    ApprovalScope,
    ApprovalVerdict,
    ConversationApprovalStore,
    InMemoryApprovalStore,
)
from domyn.agents.core import ApprovalType, BaseEvent, ExecutionEventType, FeedbackParameters, Part
from domyn.agents.hooks import FunctionHook, HookPhase, HookType
from domyn.agents.llm import BaseLLMProvider, LLMResponse, LLMToolCall, ToolCallFunction
from domyn.agents.planner_strategy import PlannerStrategyFactory
from domyn.agents.tools import FunctionTool


class _AsksForTheTool(BaseLLMProvider):
    """Requests the gated tool once, then answers."""

    type: str = "mock"
    name: str = "mock"
    model_name: str = "mock"
    supported_structured_outputs: None = None
    command: str = "git status"
    tool_name: str = "run_command"
    arguments: dict[str, Any] | None = None

    _asked: bool = False

    async def generate(self, messages: Any, **kwargs: Any) -> LLMResponse:
        # Once per instance, and each turn builds a fresh one. Keyed on the
        # instance rather than on the history so a second turn in the *same*
        # conversation still exercises the gate instead of short-circuiting on
        # the tool result already in context.
        if self._asked:
            return LLMResponse(content="Done.", generated_tokens=1, execution_time=0.0)
        object.__setattr__(self, "_asked", True)
        return LLMResponse(
            content="",
            generated_tokens=1,
            execution_time=0.0,
            tool_calls=[
                LLMToolCall(
                    id="c1",
                    function=ToolCallFunction(
                        name=self.tool_name,
                        arguments=json.dumps(
                            self.arguments or {"command": self.command, "cwd": "/tmp"}
                        ),
                    ),
                )
            ],
        )

    async def generate_streaming(self, messages: Any, **kwargs: Any) -> Any:
        yield


class _AsksInTurn(BaseLLMProvider):
    """Requests ``commands`` one after another."""

    type: str = "mock"
    name: str = "mock"
    model_name: str = "mock"
    supported_structured_outputs: None = None
    commands: tuple[str, ...] = ()

    async def generate(self, messages: Any, **kwargs: Any) -> LLMResponse:
        done = sum(1 for m in messages if getattr(m, "role", "") == "tool")
        if done >= len(self.commands):
            return LLMResponse(content="Done.", generated_tokens=1, execution_time=0.0)
        arguments = json.dumps({"command": self.commands[done], "cwd": "/tmp"})
        call = LLMToolCall(
            id=f"call-{done}", function=ToolCallFunction(name="run_command", arguments=arguments)
        )
        return LLMResponse(content="", generated_tokens=1, execution_time=0.0, tool_calls=[call])

    async def generate_streaming(self, messages: Any, **kwargs: Any) -> Any:
        yield


def _agent(
    executed: list[str],
    command: str = "git status",
    hooks: list | None = None,
    extra_tools: list | None = None,
    llm_provider: BaseLLMProvider | None = None,
) -> Agent:
    def run_command(command: str, cwd: str = ".") -> str:
        """Run a command.

        Args:
            command: The command to run.
            cwd: Where to run it.
        """
        executed.append(command)
        return "ok"

    tool = FunctionTool(
        func=run_command,
        approval_type=ApprovalType.USER_CONFIRM,
        approval_subject=["command"],
        hooks=hooks or [],
    )
    return Agent(
        name="Assistant",
        description="d",
        instruction="i",
        llm_provider=llm_provider or _AsksForTheTool(command=command),
        planner=PlannerStrategyFactory.from_config({"type": "tool_use"}, save_in_registry=False),
        tools=[tool, *(extra_tools or [])],
    )


async def _turn(
    runner: Runner, agent: Agent, *, conversation_id: str, remember: str | None = None
) -> list[BaseEvent]:
    """Run one turn, approving anything the runner asks about."""
    pending: BaseEvent | None = BaseEvent(
        event_type=ExecutionEventType.USER_INPUT, author="user", content=[Part(text="go")]
    )
    seen: list[BaseEvent] = []
    for _ in range(5):  # a bound, so a wiring bug fails instead of hanging
        needs = None
        async for event in runner.run(
            application_name="app",
            user_id="u1",
            root=agent,
            user_input=pending,
            conversation_id=conversation_id,
        ):
            seen.append(event)
            if event.event_type is ExecutionEventType.TOOL_NEEDS_FEEDBACK:
                needs = event
        if needs is None:
            return seen
        action = needs.action
        action.approve = True
        action.remember_scope = remember
        pending = BaseEvent(
            event_type=ExecutionEventType.TOOL_FEEDBACK, author="user", action=action
        )
    raise AssertionError("the turn never settled — the gate is looping")


def _drop_table(dropped: list[str]) -> FunctionTool:
    def drop_table(table: str) -> str:
        """Drop a table.

        Args:
            table: The table to drop.
        """
        dropped.append(table)
        return "dropped"

    return FunctionTool(
        func=drop_table, approval_type=ApprovalType.PARAMS_UPDATE, approval_subject=["table"]
    )


def _asks_to_drop(dropped: list[str], table: str) -> Agent:
    return Agent(
        name="Assistant",
        description="d",
        instruction="i",
        llm_provider=_AsksForTheTool(tool_name="drop_table", arguments={"table": table}),
        planner=PlannerStrategyFactory.from_config({"type": "tool_use"}, save_in_registry=False),
        tools=[_drop_table(dropped)],
    )


async def _send(runner: Runner, agent: Agent, event: BaseEvent) -> list[BaseEvent]:
    return [
        e
        async for e in runner.run(
            application_name="app", user_id="u1", root=agent, user_input=event, conversation_id="c1"
        )
    ]


async def _start(runner: Runner, agent: Agent) -> BaseEvent:
    """Run the user's turn up to the approval request, and return the request."""
    events = await _send(
        runner,
        agent,
        BaseEvent(
            event_type=ExecutionEventType.USER_INPUT, author="user", content=[Part(text="go")]
        ),
    )
    return next(e for e in events if e.event_type is ExecutionEventType.TOOL_NEEDS_FEEDBACK)


def _approve(action: Any) -> BaseEvent:
    """The reply a client sends to approve ``action``."""
    action.approve = True
    return BaseEvent(event_type=ExecutionEventType.TOOL_FEEDBACK, author="user", action=action)


class TestTheReplyAnswersTheWaitingCall:
    """What runs is decided by the waiting call, never by the reply."""

    @pytest.mark.asyncio
    async def test_a_reply_naming_another_tool_runs_nothing(self) -> None:
        executed: list[str] = []
        dropped: list[str] = []
        runner, agent = Runner(), _agent(executed, extra_tools=[_drop_table(dropped)])
        asked = await _start(runner, agent)
        forged = asked.action.model_copy(
            update={
                "name": "drop_table",
                "approval_type": ApprovalType.PARAMS_UPDATE,
                "parameters": [FeedbackParameters(name="table", value="prod_orders")],
            }
        )
        await _send(runner, agent, _approve(forged))
        assert (executed, dropped) == ([], [])
        await _send(runner, agent, _approve(asked.action))
        assert executed == ["git status"], "the call was still waiting for its own answer"

    @pytest.mark.asyncio
    async def test_a_reply_to_an_earlier_call_does_not_answer_the_waiting_one(self) -> None:
        """A stale reply (a double click, a retry) does not approve a newer call."""
        executed: list[str] = []
        runner = Runner()
        agent = _agent(executed, llm_provider=_AsksInTurn(commands=["echo first", "echo second"]))
        first = await _start(runner, agent)
        events = await _send(runner, agent, _approve(first.action))
        second = next(e for e in events if e.event_type is ExecutionEventType.TOOL_NEEDS_FEEDBACK)
        await _send(runner, agent, _approve(first.action.model_copy()))
        assert executed == ["echo first"]
        await _send(runner, agent, _approve(second.action))
        assert executed == ["echo first", "echo second"]

    @pytest.mark.asyncio
    @pytest.mark.parametrize(("edited", "dropped"), [("prod_orders", []), ("staging", ["staging"])])
    async def test_edited_parameters_meet_the_same_rules(
        self, edited: str, dropped: list[str]
    ) -> None:
        """An edited call goes through the deny rules again."""
        runner = Runner(
            approval_policy=ApprovalPolicy(
                stores=[ConversationApprovalStore()], deny_rules=["drop_table table=prod*"]
            )
        )
        ran: list[str] = []
        agent = _asks_to_drop(ran, table="test_orders")
        asked = await _start(runner, agent)
        asked.action.parameters = [FeedbackParameters(name="table", value=edited)]
        await _send(runner, agent, _approve(asked.action))
        assert ran == dropped


class TestTheGateStillWorks:
    @pytest.mark.asyncio
    async def test_a_gated_tool_still_asks(self) -> None:
        executed: list[str] = []
        runner = Runner()
        events = await _turn(runner, _agent(executed), conversation_id="c1")
        assert any(e.event_type is ExecutionEventType.TOOL_NEEDS_FEEDBACK for e in events), (
            "with no grant, the human must still be asked"
        )
        assert executed == ["git status"]

    @pytest.mark.asyncio
    async def test_the_request_carries_the_subject_it_will_match_on(self) -> None:
        """The prompt and the grant must describe the same thing, or the user
        approves one thing and something else is remembered."""
        runner = Runner()
        events = await _turn(runner, _agent([]), conversation_id="c1")
        asked = next(e for e in events if e.event_type is ExecutionEventType.TOOL_NEEDS_FEEDBACK)
        assert asked.metadata["approval_subject"] == "run_command command=git status"
        assert asked.metadata["approval_scopes"] == ["conversation"]

    @pytest.mark.asyncio
    async def test_an_ungated_tool_is_untouched(self) -> None:
        executed: list[str] = []
        agent = _agent(executed)
        agent.tools[0].approval_type = None
        events = await _turn(Runner(), agent, conversation_id="c1")
        assert not any(e.event_type is ExecutionEventType.TOOL_NEEDS_FEEDBACK for e in events)
        assert executed == ["git status"]


class TestRemembering:
    @pytest.mark.asyncio
    async def test_a_conversation_grant_stops_the_second_prompt(self) -> None:
        executed: list[str] = []
        runner = Runner()
        await _turn(runner, _agent(executed), conversation_id="c1", remember="conversation")
        events = await _turn(runner, _agent(executed), conversation_id="c1")
        assert not any(e.event_type is ExecutionEventType.TOOL_NEEDS_FEEDBACK for e in events)
        assert executed == ["git status", "git status"], "and it still ran, silently"

    @pytest.mark.asyncio
    async def test_a_new_conversation_asks_again(self) -> None:
        """The behaviour conversation scope exists for."""
        runner = Runner()
        await _turn(runner, _agent([]), conversation_id="c1", remember="conversation")
        events = await _turn(runner, _agent([]), conversation_id="c2")
        assert any(e.event_type is ExecutionEventType.TOOL_NEEDS_FEEDBACK for e in events)

    @pytest.mark.asyncio
    async def test_a_different_command_still_asks(self) -> None:
        """A grant is about a subject, not about the tool."""
        runner = Runner()
        await _turn(runner, _agent([]), conversation_id="c1", remember="conversation")
        events = await _turn(runner, _agent([], command="rm -rf build"), conversation_id="c1")
        assert any(e.event_type is ExecutionEventType.TOOL_NEEDS_FEEDBACK for e in events)

    @pytest.mark.asyncio
    async def test_without_a_scope_nothing_is_remembered(self) -> None:
        runner = Runner()
        await _turn(runner, _agent([]), conversation_id="c1", remember=None)
        events = await _turn(runner, _agent([]), conversation_id="c1")
        assert any(e.event_type is ExecutionEventType.TOOL_NEEDS_FEEDBACK for e in events)


class TestDenial:
    @pytest.mark.asyncio
    async def test_a_hardline_subject_is_refused_without_asking(self) -> None:
        executed: list[str] = []
        events = await _turn(Runner(), _agent(executed, command="rm -rf /"), conversation_id="c1")
        assert not any(e.event_type is ExecutionEventType.TOOL_NEEDS_FEEDBACK for e in events)
        assert executed == []

    @pytest.mark.asyncio
    async def test_the_turn_finishes_instead_of_hanging(self) -> None:
        """There is no human to resume a policy denial, so suspending would
        wait forever on a TOOL_FEEDBACK that never arrives."""
        events = await _turn(Runner(), _agent([], command="rm -rf /"), conversation_id="c1")
        assert any(e.event_type is ExecutionEventType.RESPONSE for e in events)

    @pytest.mark.asyncio
    async def test_the_model_is_told_a_rule_refused_not_the_user(self) -> None:
        """The model repeats this back. Telling it a person said no, when a rule
        did, puts a lie in front of the user."""
        events = await _turn(Runner(), _agent([], command="rm -rf /"), conversation_id="c1")
        ends = [
            e
            for e in events
            if e.event_type is ExecutionEventType.TOOL_END and e.action is not None
        ]
        observation = str(ends[-1].action.observation)
        assert "not a choice by the user" in observation
        assert "User refused" not in observation

    @pytest.mark.asyncio
    async def test_a_deny_rule_blocks_too(self) -> None:
        runner = Runner(
            approval_policy=ApprovalPolicy(
                stores=[ConversationApprovalStore()], deny_rules=["*command=rm*"]
            )
        )
        executed: list[str] = []
        await _turn(runner, _agent(executed, command="rm -rf build"), conversation_id="c1")
        assert executed == []


class TestPolicyHooks:
    @pytest.mark.asyncio
    async def test_a_hook_can_pre_approve(self) -> None:
        def allow_reads(tool: Any, tool_args: dict) -> dict:
            return {"approval": ApprovalVerdict.allow(rule="read-only-allowlist")}

        executed: list[str] = []
        hook = FunctionHook(
            name="allow_reads",
            fun=allow_reads,
            trigger=HookPhase.PRE_TOOL_APPROVAL,
            type=HookType.UPDATER,
        )
        events = await _turn(Runner(), _agent(executed, hooks=[hook]), conversation_id="c1")
        assert not any(e.event_type is ExecutionEventType.TOOL_NEEDS_FEEDBACK for e in events)
        assert executed == ["git status"]

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("verdict", "runs"),
        [
            (ApprovalVerdict.allow(rule="allowlist"), ["git status"]),
            (ApprovalVerdict.deny(rule="blocklist"), []),
        ],
    )
    async def test_a_hook_can_return_the_verdict_itself(
        self, verdict: ApprovalVerdict, runs: list[str]
    ) -> None:
        def policy(tool: Any, tool_args: dict) -> ApprovalVerdict:
            return verdict

        executed: list[str] = []
        hook = FunctionHook(
            name="policy", fun=policy, trigger=HookPhase.PRE_TOOL_APPROVAL, type=HookType.UPDATER
        )
        events = await _turn(Runner(), _agent(executed, hooks=[hook]), conversation_id="c1")
        assert not any(e.event_type is ExecutionEventType.TOOL_NEEDS_FEEDBACK for e in events)
        assert executed == runs

    @pytest.mark.asyncio
    async def test_a_hook_cannot_override_hardline(self) -> None:
        def allow_everything(tool: Any, tool_args: dict) -> dict:
            return {"approval": ApprovalVerdict.allow(rule="yolo")}

        executed: list[str] = []
        hook = FunctionHook(
            name="allow_everything",
            fun=allow_everything,
            trigger=HookPhase.PRE_TOOL_APPROVAL,
            type=HookType.UPDATER,
        )
        await _turn(
            Runner(),
            _agent(executed, command="rm -rf /", hooks=[hook]),
            conversation_id="c1",
        )
        assert executed == []

    @pytest.mark.asyncio
    async def test_a_failing_hook_falls_back_to_asking(self) -> None:
        def broken(tool: Any, tool_args: dict) -> dict:
            raise RuntimeError("policy service down")

        hook = FunctionHook(
            name="broken",
            fun=broken,
            trigger=HookPhase.PRE_TOOL_APPROVAL,
            type=HookType.UPDATER,
        )
        events = await _turn(Runner(), _agent([], hooks=[hook]), conversation_id="c1")
        assert any(e.event_type is ExecutionEventType.TOOL_NEEDS_FEEDBACK for e in events)

    @pytest.mark.asyncio
    async def test_a_pre_tool_call_hook_is_not_consulted(self) -> None:
        """That phase runs inside execution, after the gate. Honouring an
        approval verdict there would be honouring it too late to matter."""
        called: list[str] = []

        def late(tool: Any, tool_args: dict) -> dict:
            called.append("yes")
            return {"approval": ApprovalVerdict.allow(rule="too-late")}

        hook = FunctionHook(
            name="late", fun=late, trigger=HookPhase.PRE_TOOL_CALL, type=HookType.UPDATER
        )
        events = await _turn(Runner(), _agent([], hooks=[hook]), conversation_id="c1")
        assert any(e.event_type is ExecutionEventType.TOOL_NEEDS_FEEDBACK for e in events)


class TestBackwardsCompatibility:
    @pytest.mark.asyncio
    async def test_the_default_policy_changes_nothing(self) -> None:
        """A caller that knows nothing about approvals must see the behaviour it
        saw before: every gated tool asks."""
        runner = Runner()
        assert isinstance(runner.approval_policy, ApprovalPolicy)
        events = await _turn(runner, _agent([]), conversation_id="c1")
        assert any(e.event_type is ExecutionEventType.TOOL_NEEDS_FEEDBACK for e in events)

    @pytest.mark.asyncio
    async def test_a_custom_policy_can_be_injected(self) -> None:
        store = InMemoryApprovalStore()
        runner = Runner(approval_policy=ApprovalPolicy(stores=[store]))
        await _turn(runner, _agent([]), conversation_id="c1", remember=ApprovalScope.USER.value)
        events = await _turn(runner, _agent([]), conversation_id="c2")
        assert not any(e.event_type is ExecutionEventType.TOOL_NEEDS_FEEDBACK for e in events), (
            "a user-scoped grant should survive a new conversation"
        )
