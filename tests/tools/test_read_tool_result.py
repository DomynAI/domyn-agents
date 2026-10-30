"""A tool result cut or dropped to save context is read back from its event."""

# No `from __future__ import annotations`: builds real tools.
import json
import re
from typing import Any
from unittest.mock import MagicMock

import pytest

from domyn.agents.core import (
    AgentDescription,
    BaseEvent,
    ExecutionEventType,
    InteractionContext,
    Part,
    ToolAction,
)
from domyn.agents.planner_strategy import ToolUseContextManager
from domyn.agents.planner_strategy.context_management.compaction import ToolResultPruner
from domyn.agents.tools import FunctionTool
from domyn.agents.tools.read_tool_result import READ_TOOL_RESULT, ToolResults, read_tool_result

# One line of JSON, as an MCP search returns it: no line to cut at.
ISSUES = json.dumps(
    {"issues": [{"key": f"KEY-{i}", "summary": "s" * 200, "labels": ["a"]} for i in range(130)]}
)


def _result(observation: Any, event_id: str = "event_r") -> BaseEvent:
    return BaseEvent(
        event_type=ExecutionEventType.TOOL_END,
        author="agent",
        event_id=event_id,
        action=ToolAction(call_id="call_r", name="search", parameters={}, observation=observation),
    )


def _read(observation: Any, budget: int = 20_000, **arguments: Any) -> str:
    event = _result(observation)
    results = ToolResults(events=lambda: [event], budget=budget)
    return read_tool_result(event.event_id, tool_results=results, **arguments)


def _continues_at(out: str) -> int | None:
    """The offset a read or a search says to go on from, if it says one."""
    found = re.search(r'offset=(\d+)(?:, find="(?:[^"\\]|\\.)*")?\)\]$', out)
    return int(found.group(1)) if found else None


@pytest.mark.parametrize("budget", [20_000, 3_000])
def test_following_the_notes_reads_the_whole_result_once(budget: int) -> None:
    offset, reads = 0, 0
    while True:
        out = _read(ISSUES, budget=budget, offset=offset)
        reads += 1
        # A read the context manager would cut again could not be read on from.
        assert len(out) <= budget
        resume = _continues_at(out)
        assert out.startswith(ISSUES[offset:resume])
        if resume is None:
            break
        assert resume > offset
        offset = resume
    assert reads > 1, "the result must span several reads"


def test_an_offset_sent_as_text_is_read_as_the_number() -> None:
    assert _read(ISSUES, offset="300") == _read(ISSUES, offset=300)


def test_a_search_finds_the_text_as_written() -> None:
    """Literally: a regular expression would read `["a"]` as one character."""
    find = '["a"]}, {"key": "KEY-12'
    out = _read(ISSUES, find=find)

    listed = re.findall(r"^offset=(\d+): (.*)$", out, re.MULTILINE)
    assert [int(offset) for offset, _ in listed] == [
        m.start() for m in re.finditer(re.escape(find), ISSUES)
    ]
    assert len(listed) == 11, "KEY-12 and KEY-120 to KEY-129"
    for _, around in listed:
        assert find in around, "each match is shown with its text"
        assert len(around) < 300, "a single line of JSON is not shown whole"


def test_a_long_list_of_matches_goes_on_from_the_offset_it_gives() -> None:
    offsets: list[int] = []
    start: int | None = 0
    while start is not None:
        out = _read(ISSUES, budget=2_000, find='"key"', offset=start)
        assert len(out) <= 2_000
        offsets += [int(n) for n in re.findall(r"^offset=(\d+):", out, re.MULTILINE)]
        start = _continues_at(out)
    assert offsets == [m.start() for m in re.finditer('"key"', ISSUES)]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "arguments",
    [
        {"result_id": "event_other"},
        {"result_id": "event_r", "offset": -1},
        {"result_id": "event_r", "offset": 101},
        {"result_id": "event_r", "offset": "ten"},
    ],
    ids=["not visible", "negative offset", "past the end", "not a number"],
)
async def test_a_call_it_cannot_serve_is_an_error(arguments: dict[str, Any]) -> None:
    """Never a silent answer: a negative offset would read the end, an id it
    cannot see would read nothing, and the model would take either for a result."""
    event = _result("x" * 100)
    results = ToolResults(events=lambda: [event], budget=20_000)

    result = await READ_TOOL_RESULT.execute(**arguments, tool_results=results)

    assert not result.success
    assert result.error


def _agent(tools: list[Any]) -> Any:
    agent = MagicMock()
    agent.name = "agent"
    agent.instruction = "Use the tools."
    agent.tools = tools
    agent.planner.use_stop = False
    agent.planner.forward_verbatim_enabled = False
    return agent


@pytest.mark.asyncio
async def test_an_agent_with_a_tool_of_that_name_keeps_its_own() -> None:
    """Two definitions with one name are a request providers reject, and a note
    giving the call would send the model to the other tool."""

    def read_tool_result(path: str) -> str:
        """Read a stored report."""
        return path

    own = FunctionTool(func=read_tool_result)
    agent = _agent([own])
    start = BaseEvent(
        event_type=ExecutionEventType.TOOL_START,
        author="agent",
        action=ToolAction(call_id="call_r", name="search", parameters={}),
    )
    context = InteractionContext(
        conversation_id="c",
        user_id="u",
        agent_input=BaseEvent(
            event_type=ExecutionEventType.USER_INPUT, author="user", content=[Part(text="go")]
        ),
        conversation_history=[start, _result("x\n" * 10_000)],
    )
    manager = ToolUseContextManager(max_tool_result_chars=2_000)

    definitions = manager.get_tool_definitions(agent, context)
    named = [d for d in definitions if d["function"]["name"] == "read_tool_result"]
    assert [d["function"]["description"] for d in named] == [
        own.tool_definition["function"]["description"]
    ]
    result = next(m for m in await manager.get_context(context, agent) if m.role == "tool")
    pruned, _ = ToolResultPruner().prune([result], over_chars=200)
    assert "read_tool_result" not in str(result.content) + str(pruned[0].content)


@pytest.mark.parametrize("hands_off", [False, True])
@pytest.mark.asyncio
async def test_an_agent_with_no_tools_gets_it_only_with_someone_to_hand_off_to(
    hands_off: bool,
) -> None:
    """With neither tools nor agents to hand off to, this one would be the only
    reason the request carries tools, which a model server without tool calling
    rejects. Agents to hand off to are tools already, and a parent reads back
    what its sub-agents' tools returned."""
    agent = _agent([])
    start = BaseEvent(
        event_type=ExecutionEventType.TOOL_START,
        author="agent",
        action=ToolAction(call_id="call_r", name="search", parameters={}),
    )
    child = AgentDescription(
        name="Child",
        description="Does the work.",
        invocation_json_schema=None,
        invocation_parameters=None,
    )
    context = InteractionContext(
        conversation_id="c",
        user_id="u",
        agent_input=BaseEvent(
            event_type=ExecutionEventType.USER_INPUT, author="user", content=[Part(text="go")]
        ),
        conversation_history=[start, _result("x\n" * 10_000)],
        handoff_targets_description=[child] if hands_off else [],
    )
    manager = ToolUseContextManager(max_tool_result_chars=2_000)

    names = [d["function"]["name"] for d in manager.get_tool_definitions(agent, context)]
    assert sorted(names) == (["Child", "read_tool_result"] if hands_off else [])
    result = next(m for m in await manager.get_context(context, agent) if m.role == "tool")
    pruned, _ = ToolResultPruner().prune([result], over_chars=200)
    assert ("read_tool_result(" in str(result.content) + str(pruned[0].content)) is hands_off
