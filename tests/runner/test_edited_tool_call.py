"""A call the user edited before approving is shown to the model as it ran, once.

The model used to see its initial call next to the result of the edited one, with
the explanation after it as a user message (tool_use) or not at all (react), and
made the call again, often with the arguments the user had replaced.
"""

# No `from __future__ import annotations`: builds real tools.
import json
from typing import Any

import pytest
from pydantic import PrivateAttr

from domyn.agents.agents import Agent
from domyn.agents.approvals import append_user_edit_note
from domyn.agents.core import (
    METADATA_INITIAL_PARAMETERS,
    ApprovalType,
    BaseEvent,
    ExecutionEventType,
    FeedbackParameters,
    Part,
    ToolAction,
)
from domyn.agents.core.context import InteractionContext
from domyn.agents.llm.base import (
    BaseLLMProvider,
    LLMMessage,
    LLMResponse,
    LLMToolCall,
    ToolCallFunction,
)
from domyn.agents.planner_strategy import PlannerStrategyFactory
from domyn.agents.planner_strategy.context_management.full_messages import FullMessageContext
from domyn.agents.planner_strategy.prompting.json_prompting import JSONMessagePrompting
from domyn.agents.planner_strategy.prompting.structural_tags_prompting import (
    StructuralTagMessagePrompting,
)
from domyn.agents.planner_strategy.react import ReactPlannerStrategy
from domyn.agents.runner import Runner
from domyn.agents.tools.base import ToolResult
from domyn.agents.tools.function_tool import FunctionTool

USER_EDIT_NOTE = "before approving, the user rejected the arguments you proposed"
REFUSAL = "User refused to execute the tool multiply"


class _ScriptedProvider(BaseLLMProvider):
    """Proposes multiply(3, 2), then answers; records every request."""

    type: str = "mock"
    name: str = "mock"
    model_name: str = "mock"
    supported_structured_outputs: None = None
    _requests: list[list[LLMMessage]] = PrivateAttr(default_factory=list)

    async def generate(self, messages: Any, tools: Any = None, **kwargs: Any) -> LLMResponse:
        self._requests.append(list(messages))
        if len(self._requests) == 1:
            arguments = {"a": 3, "b": 2, "reasoning": "the user asked for 3x2"}
            call = LLMToolCall(
                id="call_1",
                function=ToolCallFunction(name="multiply", arguments=json.dumps(arguments)),
            )
            return LLMResponse(
                content="", generated_tokens=1, execution_time=0.0, tool_calls=[call]
            )
        return LLMResponse(content="done", generated_tokens=1, execution_time=0.0)

    async def generate_streaming(self, messages: Any, **kwargs: Any) -> Any:
        yield


def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b


def _multiply_failing_above_3() -> Any:
    """A multiply that fails on the arguments the user puts in."""

    def multiply(a: int, b: int) -> int:
        """Multiply two numbers."""
        if a > 3:
            raise ValueError(f"a={a} is not allowed")
        return a * b

    return multiply


async def _run_and_answer_approval(
    planner: str, *, approve: bool, edit: bool, func: Any = multiply
) -> tuple[Runner, _ScriptedProvider]:
    """Run until multiply asks, then answer: approve or refuse, 3x2 edited to 4x5 or not."""
    provider = _ScriptedProvider()
    agent = Agent(
        name="Calc",
        description="Multiplies.",
        instruction="Use the tools.",
        llm_provider=provider,
        planner=PlannerStrategyFactory.from_config({"type": planner}, save_in_registry=False),
        tools=[FunctionTool(func=func, approval_type=ApprovalType.PARAMS_UPDATE)],
    )
    runner = Runner()
    kwargs: dict[str, Any] = {
        "application_name": "app",
        "user_id": "u",
        "root": agent,
        "conversation_id": "c1",
    }
    question = BaseEvent(
        event_type=ExecutionEventType.USER_INPUT, author="user", content=[Part(text="3x2?")]
    )
    events = [e async for e in runner.run(user_input=question, **kwargs)]
    assert events[-1].event_type == ExecutionEventType.TOOL_NEEDS_FEEDBACK
    feedback = events[-1].action
    feedback.approve = approve
    if edit:
        feedback.parameters = [
            FeedbackParameters(name="a", value=4, type="int"),
            FeedbackParameters(name="b", value=5, type="int"),
        ]
    reply = BaseEvent(event_type=ExecutionEventType.TOOL_FEEDBACK, author="user", action=feedback)
    _ = [e async for e in runner.run(user_input=reply, **kwargs)]
    return runner, provider


async def _stored_events(runner: Runner) -> list[BaseEvent]:
    conversation = await runner.conversation_manager.get_conversation("app", "u", "c1")
    assert conversation is not None
    return conversation.events_history


def _joined_contents(messages: list[LLMMessage]) -> str:
    return "\n".join(str(m.content or "") for m in messages)


@pytest.mark.parametrize("planner", ["tool_use", "tool_use_thought"])
@pytest.mark.asyncio
async def test_tool_use_shows_the_edited_call_as_it_ran(planner: str) -> None:
    _, provider = await _run_and_answer_approval(planner, approve=True, edit=True)
    request = provider._requests[-1]

    messages_with_tool_calls = [m for m in request if m.tool_calls]
    assert len(messages_with_tool_calls) == 1
    shown_arguments = json.loads(messages_with_tool_calls[0].tool_calls[0].function.arguments)
    assert {k: shown_arguments[k] for k in ("a", "b")} == {"a": 4, "b": 5}
    assert shown_arguments.get("reasoning") == "the user asked for 3x2", "the reasoning stays"
    # The result, with the note in it, ends the request: no user message after it.
    assert request[-1].role == "tool"
    assert "20" in str(request[-1].content)
    assert USER_EDIT_NOTE in str(request[-1].content)
    assert '"a": 3' in str(request[-1].content), "the note names the initial arguments"
    # Without it, a sub-agent went back to the value its task named.
    assert "override your task" in str(request[-1].content)


@pytest.mark.parametrize("planner", ["tool_use", "tool_use_thought"])
@pytest.mark.asyncio
async def test_tool_use_approval_adds_no_message(planner: str) -> None:
    _, provider = await _run_and_answer_approval(planner, approve=True, edit=False)
    request = provider._requests[-1]

    assert request[-1].role == "tool"
    assert str(request[-1].content).endswith("6")
    assert USER_EDIT_NOTE not in _joined_contents(request)
    assert all(m.content or m.tool_calls for m in request), "no empty message"


@pytest.mark.parametrize("planner", ["tool_use", "tool_use_thought"])
@pytest.mark.asyncio
async def test_tool_use_refusal_is_said_once(planner: str) -> None:
    _, provider = await _run_and_answer_approval(planner, approve=False, edit=False)
    request = provider._requests[-1]

    assert request[-1].role == "tool"
    assert _joined_contents(request).count(REFUSAL) == 1


@pytest.mark.parametrize("planner", ["tool_use", "tool_use_thought"])
@pytest.mark.asyncio
async def test_tool_use_shows_an_edited_call_that_failed_as_it_ran(planner: str) -> None:
    _, provider = await _run_and_answer_approval(
        planner, approve=True, edit=True, func=_multiply_failing_above_3()
    )
    request = provider._requests[-1]

    messages_with_tool_calls = [m for m in request if m.tool_calls]
    shown_arguments = json.loads(messages_with_tool_calls[0].tool_calls[0].function.arguments)
    assert {k: shown_arguments[k] for k in ("a", "b")} == {"a": 4, "b": 5}
    assert request[-1].role == "tool"
    assert "a=4 is not allowed" in str(request[-1].content)
    assert _joined_contents(request).count(USER_EDIT_NOTE) == 1
    assert "the error above" in str(request[-1].content)
    # Told only to start from "the user's values", models retried with the rejected ones.
    assert 'To retry, call exactly multiply({"a": 4, "b": 5})' in str(request[-1].content)
    assert "The call is done" not in _joined_contents(request)


def _multiply_returning_zero() -> Any:
    """A multiply whose result is 0."""

    def multiply(a: int, b: int) -> int:
        """Multiply two numbers."""
        return 0

    return multiply


@pytest.mark.parametrize("planner", ["tool_use", "tool_use_thought"])
@pytest.mark.asyncio
async def test_tool_use_keeps_a_zero_result_before_the_note(planner: str) -> None:
    _, provider = await _run_and_answer_approval(
        planner, approve=True, edit=True, func=_multiply_returning_zero()
    )
    request = provider._requests[-1]

    assert request[-1].role == "tool"
    # After the event id the model is shown, the result comes before the note.
    assert "]\n0\n\nNote: " + USER_EDIT_NOTE in str(request[-1].content)


@pytest.mark.asyncio
async def test_the_stored_result_is_the_tools_own() -> None:
    """The note is for the model: what is stored, and shown to users, is the result."""
    runner, _ = await _run_and_answer_approval("tool_use_thought", approve=True, edit=True)
    (tool_end,) = [
        e for e in await _stored_events(runner) if e.event_type == ExecutionEventType.TOOL_END
    ]

    assert tool_end.action.observation == 20
    assert tool_end.action.parameters == {"a": 4, "b": 5}
    assert tool_end.metadata[METADATA_INITIAL_PARAMETERS] == {"a": 3, "b": 2}
    assert tool_end.action.thought == "the user asked for 3x2"


@pytest.mark.asyncio
async def test_a_stored_error_names_the_initial_arguments() -> None:
    runner, _ = await _run_and_answer_approval(
        "tool_use_thought", approve=True, edit=True, func=_multiply_failing_above_3()
    )
    (tool_error,) = [
        e for e in await _stored_events(runner) if e.event_type == ExecutionEventType.TOOL_ERROR
    ]

    assert USER_EDIT_NOTE not in str(tool_error.action.observation)
    assert tool_error.metadata[METADATA_INITIAL_PARAMETERS] == {"a": 3, "b": 2}


async def _react_messages(
    runner: Runner, *, narrow: bool, prompting: Any = None
) -> list[LLMMessage]:
    """The stored conversation as a react agent is shown it."""
    events = await _stored_events(runner)
    planner = ReactPlannerStrategy(
        context_manager=FullMessageContext(
            narrow_non_visible_agents=narrow,
            name="test",
            prompting=prompting or StructuralTagMessagePrompting(),
        )
    )
    agent = Agent(
        name="Calc",
        description="Multiplies.",
        instruction="Use the tools.",
        llm_provider=_ScriptedProvider(),
        planner=planner,
        tools=[FunctionTool(func=multiply, approval_type=ApprovalType.PARAMS_UPDATE)],
    )
    context = InteractionContext(
        conversation_id="c1", user_id="u", agent_input=events[0], conversation_history=events
    )
    return await planner.context_manager.get_context(context, agent)


@pytest.mark.parametrize("narrow", [True, False])
@pytest.mark.asyncio
async def test_react_shows_the_edited_call_once_with_its_thought(narrow: bool) -> None:
    runner, _ = await _run_and_answer_approval("tool_use_thought", approve=True, edit=True)
    messages = await _react_messages(runner, narrow=narrow)

    rendered_calls = [m for m in messages[1:] if '"name": "multiply"' in str(m.content)]
    assert len(rendered_calls) == 1, "the call is shown once"
    assert '"a": 4, "b": 5' in str(rendered_calls[0].content)
    assert "<thought>the user asked for 3x2</thought>" in str(rendered_calls[0].content)
    assert _joined_contents(messages).count(USER_EDIT_NOTE) == 1


@pytest.mark.parametrize("narrow", [True, False])
@pytest.mark.asyncio
async def test_react_refusal_is_said_once(narrow: bool) -> None:
    runner, _ = await _run_and_answer_approval("tool_use_thought", approve=False, edit=False)
    messages = await _react_messages(runner, narrow=narrow)

    assert _joined_contents(messages).count(REFUSAL) == 1


@pytest.mark.parametrize("prompting", [StructuralTagMessagePrompting, JSONMessagePrompting])
@pytest.mark.asyncio
async def test_react_tells_an_edited_call_failed_with_the_users_arguments(prompting: Any) -> None:
    runner, _ = await _run_and_answer_approval(
        "tool_use_thought", approve=True, edit=True, func=_multiply_failing_above_3()
    )
    messages = await _react_messages(runner, narrow=True, prompting=prompting())

    assert _joined_contents(messages).count(USER_EDIT_NOTE) == 1
    assert "the error above" in _joined_contents(messages)


@pytest.mark.asyncio
async def test_react_json_prompting_adds_the_user_edit_note_once() -> None:
    runner, _ = await _run_and_answer_approval("tool_use_thought", approve=True, edit=True)
    messages = await _react_messages(runner, narrow=True, prompting=JSONMessagePrompting())

    assert _joined_contents(messages).count(USER_EDIT_NOTE) == 1
    assert "The call is done" in _joined_contents(messages)


@pytest.mark.asyncio
async def test_a_reused_call_id_shows_only_the_edited_call_as_it_ran() -> None:
    """Some providers repeat call ids across responses: an edit is tied to its own call."""
    runner, _ = await _run_and_answer_approval("tool_use_thought", approve=True, edit=True)
    later_events_with_same_call_id = [
        BaseEvent(
            event_type=event_type,
            event_id=f"later_{event_type.value}",
            author="Calc",
            action=ToolAction(
                call_id="call_1", name="multiply", parameters={"a": 3, "b": 2}, observation=6
            ),
        )
        for event_type in (ExecutionEventType.TOOL_START, ExecutionEventType.TOOL_END)
    ]
    events = [*await _stored_events(runner), *later_events_with_same_call_id]
    planner = PlannerStrategyFactory.from_config(
        {"type": "tool_use_thought"}, save_in_registry=False
    )
    agent = Agent(
        name="Calc",
        description="Multiplies.",
        instruction="Use the tools.",
        llm_provider=_ScriptedProvider(),
        planner=planner,
        tools=[FunctionTool(func=multiply, approval_type=ApprovalType.PARAMS_UPDATE)],
    )
    context = InteractionContext(
        conversation_id="c1", user_id="u", agent_input=events[0], conversation_history=events
    )
    messages = await planner.context_manager.get_context(context, agent)

    shown_call_arguments = [
        json.loads(m.tool_calls[0].function.arguments) for m in messages if m.tool_calls
    ]
    assert [{k: c[k] for k in ("a", "b")} for c in shown_call_arguments] == [
        {"a": 4, "b": 5},
        {"a": 3, "b": 2},
    ]


def _user_edited_tool_end(observation: Any) -> BaseEvent:
    return BaseEvent(
        event_type=ExecutionEventType.TOOL_END,
        author="Calc",
        action=ToolAction(name="multiply", parameters={"a": 0, "b": 5}, observation=observation),
        metadata={METADATA_INITIAL_PARAMETERS: {"a": 3, "b": 2}},
    )


@pytest.mark.parametrize("observation", [0, False, "  indented"])
def test_the_user_edit_note_keeps_every_result(observation: Any) -> None:
    rendered_result = append_user_edit_note(observation, _user_edited_tool_end(observation))

    assert rendered_result.startswith(f"{observation}\n\n")
    assert USER_EDIT_NOTE in rendered_result


@pytest.mark.parametrize("observation", [None, ""])
def test_a_missing_result_is_the_user_edit_note_alone(observation: Any) -> None:
    rendered_result = append_user_edit_note(observation, _user_edited_tool_end(observation))

    assert rendered_result.startswith("Note: ")


@pytest.mark.asyncio
@pytest.mark.parametrize("tool_value", ["x", ["a"], {"a": 1, "b": 2}])
async def test_a_tool_cannot_pass_its_call_off_as_edited(tool_value: Any) -> None:
    def multiply(a: int, b: int) -> ToolResult:
        """Multiply two numbers."""
        return ToolResult(
            success=True, data=a * b, metadata={METADATA_INITIAL_PARAMETERS: tool_value}
        )

    provider = _ScriptedProvider()
    agent = Agent(
        name="Calc",
        description="Multiplies.",
        instruction="Use the tools.",
        llm_provider=provider,
        planner=PlannerStrategyFactory.from_config(
            {"type": "tool_use_thought"}, save_in_registry=False
        ),
        tools=[FunctionTool(func=multiply)],
    )
    question = BaseEvent(
        event_type=ExecutionEventType.USER_INPUT, author="user", content=[Part(text="3x2?")]
    )
    events = [
        e
        async for e in Runner().run(
            application_name="app",
            user_id="u",
            root=agent,
            conversation_id="c1",
            user_input=question,
        )
    ]

    tool_end = next(e for e in events if e.event_type == ExecutionEventType.TOOL_END)
    assert METADATA_INITIAL_PARAMETERS not in tool_end.metadata
    assert USER_EDIT_NOTE not in _joined_contents(provider._requests[-1])
