"""A sub-agent called again in a later turn must be sent its new task.

Seen with the default planner: the second call's task was dropped from the
sub-agent's messages, so it repeated its previous answer without calling a tool.
"""

# No `from __future__ import annotations`: builds real tools.
import json
from typing import Any

import pytest
from pydantic import PrivateAttr

from domyn.agents import Agent, BaseEvent, ExecutionEventType, Part, Runner
from domyn.agents.agents import ContextVisibility
from domyn.agents.llm import BaseLLMProvider, LLMResponse, LLMToolCall, ToolCallFunction
from domyn.agents.planner_strategy import PlannerStrategyFactory
from domyn.agents.tools import FunctionTool


def _call(name: str, **arguments: Any) -> LLMResponse:
    tool_call = LLMToolCall(
        id=f"call_{name}_{json.dumps(arguments, sort_keys=True)}",
        function=ToolCallFunction(name=name, arguments=json.dumps(arguments)),
    )
    return LLMResponse(content="", generated_tokens=1, execution_time=0.0, tool_calls=[tool_call])


def _text(text: str) -> LLMResponse:
    return LLMResponse(content=text, generated_tokens=1, execution_time=0.0)


class _ScriptedProvider(BaseLLMProvider):
    """Answers from a fixed script and records the messages of every call."""

    type: str = "mock"
    name: str = "mock"
    model_name: str = "mock"
    supported_structured_outputs: None = None
    _script: list[LLMResponse] = PrivateAttr(default_factory=list)
    _seen: list[list[str]] = PrivateAttr(default_factory=list)

    def __init__(self, script: list[LLMResponse], **data: Any) -> None:
        super().__init__(**data)
        self._script = list(script)

    async def generate(self, messages: Any, **kwargs: Any) -> LLMResponse:
        self._seen.append([str(getattr(m, "content", "") or "") for m in messages])
        return self._script.pop(0)

    async def generate_streaming(self, messages: Any, **kwargs: Any) -> Any:
        yield


def get_order_status(order_id: str) -> str:
    """Return the current status of an order.

    Args:
        order_id: The order.
    """
    return f"{order_id}: shipped"


def _planner(planner: str, use_stop: bool = False) -> Any:
    return PlannerStrategyFactory.from_config(
        {"type": planner, "use_stop": use_stop}, save_in_registry=False
    )


@pytest.mark.parametrize("visibility", list(ContextVisibility))
@pytest.mark.parametrize("use_stop", [False, True])
@pytest.mark.parametrize("planner", ["tool_use", "tool_use_thought"])
@pytest.mark.asyncio
async def test_second_call_is_sent_the_new_task(
    planner: str, use_stop: bool, visibility: ContextVisibility
) -> None:
    def end(order: str) -> LLMResponse:
        info = f"Order {order} is shipped"
        return _call("STOP", info=info) if use_stop else _text(info)

    sub_provider = _ScriptedProvider(
        [
            _call("get_order_status", order_id="1234"),
            end("1234"),
            _call("get_order_status", order_id="5678"),
            end("5678"),
        ]
    )
    root_provider = _ScriptedProvider(
        [
            _call("OrderAgent", task="check order 1234"),
            _text("1234 is shipped"),
            _call("OrderAgent", task="check order 5678"),
            _text("5678 is shipped"),
        ]
    )
    sub = Agent(
        name="OrderAgent",
        description="Looks up orders.",
        instruction="Use the tool.",
        llm_provider=sub_provider,
        planner=_planner(planner, use_stop),
        context_visibility=visibility,
        tools=[FunctionTool(func=get_order_status)],
    )
    root = Agent(
        name="Support",
        description="Support.",
        instruction="Delegate orders to OrderAgent.",
        llm_provider=root_provider,
        planner=_planner(planner),
        sub_agents=[sub],
    )

    runner = Runner()
    for text in ("what about order 1234?", "and order 5678?"):
        async for _ in runner.run(
            application_name="app",
            user_id="u",
            root=root,
            user_input=BaseEvent(
                event_type=ExecutionEventType.USER_INPUT, author="user", content=[Part(text=text)]
            ),
            conversation_id="c1",
        ):
            pass

    assert len(sub_provider._seen) == 4, "the script must have run to the end"
    second_entry = " ".join(sub_provider._seen[2])
    assert second_entry.count("Task assigned: check order 5678") == 1
    # The root's first call of turn 2 has the request once.
    assert " ".join(root_provider._seen[2]).count("and order 5678?") == 1
