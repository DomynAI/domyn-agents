"""An agent reads back the part of a tool result it was not shown.

A scripted model follows the notes as a model would: the call the cut gives, then
the call each read ends with, until it reaches what the cut kept of the end. It
runs through the real runner and planners, so what is checked is what a model is
sent: that following the notes rebuilds the result exactly, every read reaching
the model whole.
"""

# No `from __future__ import annotations`: builds real tools.
import json
import re
from collections.abc import AsyncIterator
from typing import Any

import pytest
from pydantic import PrivateAttr

from domyn.agents import Agent, BaseEvent, ExecutionEventType, Part, Runner
from domyn.agents.llm import BaseLLMProvider, LLMResponse, LLMToolCall, ToolCallFunction
from domyn.agents.planner_strategy import PlannerStrategyFactory
from domyn.agents.tools import (
    FunctionTool,
    StreamableFunctionTool,
    StreamableToolResult,
)

ROWS = "".join(f"issue {i:05d}: {'x' * 90}\n" for i in range(600))
SHOWN = "".join(f"shown {i:05d}: {'y' * 90}\n" for i in range(300))

_CALL = re.compile(r'read_tool_result\(result_id="([^"]+)"(?:, offset=(\d+))?\)')
_SPAN = re.compile(r"characters ([\d,]+) to ([\d,]+) of ([\d,]+)")
# A cut: its note on a line of its own, inside the result.
_CUT = re.compile(r"\n\[(characters [^\[\]\n]*read_tool_result\([^\[\]\n]*)\]\n")
_HEADER = re.compile(r"^\[event_id: [^\]\n]+\]\n")


def _call(name: str, **arguments: Any) -> LLMResponse:
    tool_call = LLMToolCall(
        id=f"call_{name}_{arguments.get('offset', 0)}",
        function=ToolCallFunction(name=name, arguments=json.dumps(arguments)),
    )
    return LLMResponse(content="", generated_tokens=1, execution_time=0.0, tool_calls=[tool_call])


def _text(text: str) -> LLMResponse:
    return LLMResponse(content=text, generated_tokens=1, execution_time=0.0)


def _span(note: str) -> tuple[int, int, int]:
    found = _SPAN.search(note)
    assert found, f"no span in: {note}"
    start, end, total = (int(n.replace(",", "")) for n in found.groups())
    return start, end, total


class _Reader:
    """A model that makes its opening calls, then reads the first cut result it
    is shown back by following the notes, and keeps what it was shown."""

    def __init__(self, *opening: LLMResponse) -> None:
        self.opening = list(opening)
        self.head = self.tail = ""
        self.gap = (0, 0)
        self.total = 0
        self.read = ""
        self.done = False

    def __call__(self, messages: list[Any]) -> LLMResponse:
        if self.opening:
            return self.opening.pop(0)
        if not self.total:
            return self._follow_the_cut(messages)
        return self._follow_the_read(str(messages[-1].content))

    def _follow_the_cut(self, messages: list[Any]) -> LLMResponse:
        for message in messages:
            content = _HEADER.sub("", str(message.content or ""))
            cut = _CUT.search(content)
            if cut:
                break
        else:
            raise AssertionError("no cut result among the messages")
        head = content[: cut.start()]
        if head.startswith("Agent: "):  # another agent's result, told as narration
            head = head.split("Result: ", 1)[1]
        self.head, self.tail = head, content[cut.end() :]
        start, end, self.total = _span(cut.group(1))
        assert (start, end) == (len(self.head), self.total - len(self.tail)), (
            "the cut says which characters it leaves out"
        )
        self.gap = (start, end)
        return self._read(cut.group(1), expected=start)

    def _follow_the_read(self, content: str) -> LLMResponse:
        text, _, note = _HEADER.sub("", content).rpartition("\n\n[")
        start, end, total = _span(note)
        assert (start, total) == (self.gap[0] + len(self.read), self.total), (
            "a read goes on where the last one stopped"
        )
        assert len(text) == end - start, "a read reaches the model whole, not cut again"
        self.read += text
        if self.gap[0] + len(self.read) < self.gap[1]:
            return self._read(note, expected=end)
        self.done = True
        return _text("done")

    def _read(self, note: str, *, expected: int) -> LLMResponse:
        call = _CALL.search(note)
        assert call, f"no call to read on in: {note}"
        offset = int(call.group(2) or 0)
        assert offset == expected, "the call reads from the first character not shown"
        return _call("read_tool_result", result_id=call.group(1), offset=offset)

    def rebuilt(self) -> str:
        """The result as the model can put it back together."""
        return self.head + self.read[: self.gap[1] - self.gap[0]] + self.tail


class _ScriptedProvider(BaseLLMProvider):
    """Answers from a script, and records the tools offered with every request."""

    type: str = "mock"
    name: str = "mock"
    model_name: str = "mock"
    supported_structured_outputs: None = None
    _script: Any = PrivateAttr()
    _tools: list[list[str]] = PrivateAttr(default_factory=list)

    def __init__(self, script: Any, **data: Any) -> None:
        super().__init__(**data)
        self._script = script

    async def generate(self, messages: Any, tools: Any = None, **kwargs: Any) -> LLMResponse:
        self._tools.append([t["function"]["name"] for t in tools or []])
        if callable(self._script):
            return self._script(messages)
        return self._script.pop(0)

    async def generate_streaming(self, messages: Any, **kwargs: Any) -> Any:
        yield


def dump() -> str:
    """Return everything there is."""
    return ROWS


async def show_and_dump():  # type: ignore[no-untyped-def]
    """Show a report to the user, then return everything there is."""
    yield StreamableToolResult(data=Part(text=SHOWN))
    yield ROWS


def _planner(kind: str, budget: int) -> Any:
    manager = {"tool_use": "tool_use", "tool_use_thought": "thought_tool_use"}[kind]
    config = {"type": kind, "context_manager": {"type": manager, "max_tool_result_chars": budget}}
    return PlannerStrategyFactory.from_config(config, save_in_registry=False)


async def _run(agent: Agent) -> None:
    events: AsyncIterator[BaseEvent] = Runner().run(
        application_name="app",
        user_id="u",
        root=agent,
        user_input=BaseEvent(
            event_type=ExecutionEventType.USER_INPUT, author="user", content=[Part(text="go")]
        ),
        conversation_id="c1",
    )
    async for _ in events:
        pass


@pytest.mark.parametrize("budget", [20_000, 6_000])
@pytest.mark.parametrize("planner", ["tool_use", "tool_use_thought"])
@pytest.mark.parametrize("case", ["own result", "streamed", "another agent's"])
@pytest.mark.asyncio
async def test_following_the_notes_reads_the_result_back_exactly(
    case: str, planner: str, budget: int
) -> None:
    if case == "another agent's":
        reader = _Reader(_call("Child", task="dump it"))
        child = Agent(
            name="Child",
            description="Dumps.",
            instruction="Use the tools.",
            llm_provider=_ScriptedProvider([_call("dump"), _text("dumped")]),
            planner=_planner(planner, budget),
            tools=[FunctionTool(func=dump)],
        )
        tools: list[Any] = []
        sub_agents = [child]
    else:
        streamed = case == "streamed"
        reader = _Reader(_call("show_and_dump" if streamed else "dump"))
        tools = [
            StreamableFunctionTool(func=show_and_dump) if streamed else FunctionTool(func=dump)
        ]
        sub_agents = []
    provider = _ScriptedProvider(reader)
    root = Agent(
        name="Reader",
        description="Reads.",
        instruction="Use the tools.",
        llm_provider=provider,
        planner=_planner(planner, budget),
        tools=tools,
        sub_agents=sub_agents,
    )

    await _run(root)

    assert reader.done, "the model read up to the end the cut kept"
    rebuilt = reader.rebuilt()
    assert len(rebuilt) == reader.total
    if case == "streamed":
        # What the tool showed the user reaches the model inside its result, and
        # reads back with it.
        assert ROWS in rebuilt and SHOWN in rebuilt
    else:
        assert rebuilt == ROWS
    assert all("read_tool_result" in names for names in provider._tools)
