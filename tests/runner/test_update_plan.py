"""An agent keeps a plan through the real runner and planners.

A scripted model writes a plan, works, and finishes it; what is checked is what
the model is sent: the tool offered only when asked for, and the plan at the end
of every request while a step is left to do.
"""

import json
from typing import Any

import pytest
from pydantic import PrivateAttr

from domyn.agents import Agent, BaseEvent, ExecutionEventType, Part, Runner
from domyn.agents.llm import BaseLLMProvider, LLMResponse, LLMToolCall, ToolCallFunction
from domyn.agents.planner_strategy import PlannerStrategyFactory
from domyn.agents.tools import FunctionTool
from domyn.agents.tools.update_plan import NO_PLAN_TEXT, PLANS_STATE_KEY

_PLAN = "<current_plan>"


def _call(name: str, call_id: str, **arguments: Any) -> LLMResponse:
    tool_call = LLMToolCall(
        id=call_id, function=ToolCallFunction(name=name, arguments=json.dumps(arguments))
    )
    return LLMResponse(content="", generated_tokens=1, execution_time=0.0, tool_calls=[tool_call])


def _text(text: str) -> LLMResponse:
    return LLMResponse(content=text, generated_tokens=1, execution_time=0.0)


class _ScriptedProvider(BaseLLMProvider):
    """Answers from a script, and records every request."""

    type: str = "mock"
    name: str = "mock"
    model_name: str = "mock"
    supported_structured_outputs: None = None
    _script: list[LLMResponse] = PrivateAttr()
    _requests: list[tuple[list[Any], list[str]]] = PrivateAttr(default_factory=list)

    def __init__(self, script: list[LLMResponse], **data: Any) -> None:
        super().__init__(**data)
        self._script = list(script)

    async def generate(self, messages: Any, tools: Any = None, **kwargs: Any) -> LLMResponse:
        self._requests.append((list(messages), [t["function"]["name"] for t in tools or []]))
        return self._script.pop(0)

    async def generate_streaming(self, messages: Any, **kwargs: Any) -> Any:
        yield


def work() -> str:
    """Do some work."""
    return "worked"


def _agent(provider: _ScriptedProvider, planner: str, plan_tool: bool) -> Agent:
    manager = {"tool_use": "tool_use", "tool_use_thought": "thought_tool_use"}[planner]
    config = {"type": planner, "context_manager": {"type": manager, "plan_tool": plan_tool}}
    return Agent(
        name="Worker",
        description="Works.",
        instruction="Use the tools.",
        llm_provider=provider,
        planner=PlannerStrategyFactory.from_config(config, save_in_registry=False),
        tools=[FunctionTool(func=work)],
        stream=False,
    )


async def _run(runner: Runner, agent: Agent) -> list[BaseEvent]:
    events = []
    async for event in runner.run(
        application_name="app",
        user_id="u",
        root=agent,
        user_input=BaseEvent(
            event_type=ExecutionEventType.USER_INPUT, author="user", content=[Part(text="go")]
        ),
        conversation_id="c1",
    ):
        events.append(event)
    return events


def _last(messages: list[Any]) -> str:
    return str(messages[-1].content or "")


@pytest.mark.parametrize("planner", ["tool_use", "tool_use_thought"])
@pytest.mark.asyncio
async def test_the_plan_is_shown_at_the_end_while_a_step_is_left(planner: str) -> None:
    provider = _ScriptedProvider(
        [
            _call(
                "update_plan",
                "p1",
                plan=[
                    {"step": "Work", "status": "in_progress"},
                    {"step": "Report", "status": "pending"},
                ],
            ),
            _call("work", "w1"),
            _call(
                "update_plan",
                "p2",
                plan=[
                    {"step": "Work", "status": "completed"},
                    {"step": "Report", "status": "completed"},
                ],
            ),
            _text("done"),
        ]
    )
    runner = Runner()

    await _run(runner, _agent(provider, planner, plan_tool=True))

    first, after_plan, after_work, after_finish = (messages for messages, _ in provider._requests)
    assert all("update_plan" in tools for _, tools in provider._requests)
    assert _last(first).endswith(NO_PLAN_TEXT), "before a plan, the place says there is none"
    assert _last(first).startswith("USER INPUT: go"), "joined to the user's own message"
    for messages in (after_plan, after_work):
        assert _PLAN in _last(messages), "the plan is the end of the request"
        assert sum(_PLAN in str(m.content) for m in messages) == 1, "and only there"
    assert "[>] 1. Work (in progress)" in _last(after_plan)
    assert not any(_PLAN in str(m.content) for m in after_finish), "a finished plan is not shown"


@pytest.mark.asyncio
async def test_the_plan_is_kept_in_the_conversation_state() -> None:
    provider = _ScriptedProvider(
        [
            _call("update_plan", "p1", plan=[{"step": "Work", "status": "in_progress"}]),
            _text("paused"),
        ]
    )
    runner = Runner()
    agent = _agent(provider, "tool_use_thought", plan_tool=True)

    events = await _run(runner, agent)

    ends = [e for e in events if e.event_type == ExecutionEventType.TOOL_END]
    assert ends and "Plan updated" in str(ends[0].action.observation)
    # The next turn of the same conversation starts with the plan at its end.
    provider._script = [_text("again")]
    await _run(runner, agent)
    assert "[>] 1. Work (in progress)" in _last(provider._requests[-1][0])


@pytest.mark.parametrize("planner", ["tool_use", "tool_use_thought"])
@pytest.mark.asyncio
async def test_off_by_default(planner: str) -> None:
    provider = _ScriptedProvider(
        [
            _call("update_plan", "p1", plan=[{"step": "Work", "status": "in_progress"}]),
            _text("done"),
        ]
    )
    runner = Runner()

    events = await _run(runner, _agent(provider, planner, plan_tool=False))

    assert all("update_plan" not in tools for _, tools in provider._requests)
    assert not any(_PLAN in str(m.content) for m in provider._requests[-1][0])
    stored = [
        e
        for e in events
        if e.event_type == ExecutionEventType.TOOL_END
        and PLANS_STATE_KEY in str(getattr(e.action, "observation", ""))
    ]
    assert not stored, "a call to a tool it was not offered stores nothing"
