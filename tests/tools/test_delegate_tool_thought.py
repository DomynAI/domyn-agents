"""The model's thought reaches a relayed DelegateTool's relay through `tool_action`.

The runner passes the call being executed as the argument `tool_action`;
the adapter that relays the call sends its own TOOL_START and has nothing else to
take the thought from.
"""

from typing import Any

import pytest

from domyn.agents.core import ToolAction
from domyn.agents.tools import DelegateTool


def _tool() -> DelegateTool:
    return DelegateTool(
        tool_name="search",
        tool_parameters={"query": {"type": "string"}},
        required_params=["query"],
    )


@pytest.mark.asyncio
async def test_relay_receives_the_thought() -> None:
    calls: list[tuple[str, dict[str, Any], str | None]] = []

    async def relay(name: str, kwargs: dict[str, Any], thought: str | None = None) -> str:
        calls.append((name, kwargs, thought))
        return "ok"

    tool = _tool()
    tool.relay = relay
    action = ToolAction(name="search", parameters={"query": "q"}, thought="look it up")
    result = await tool.execute(query="q", state={}, tool_action=action)

    assert result.success
    assert calls == [("search", {"query": "q"}, "look it up")]


@pytest.mark.asyncio
async def test_relay_gets_none_without_a_thought() -> None:
    calls: list[tuple[str, dict[str, Any], str | None]] = []

    async def relay(name: str, kwargs: dict[str, Any], thought: str | None) -> str:
        calls.append((name, kwargs, thought))
        return "ok"

    tool = _tool()
    tool.relay = relay
    for action in (None, ToolAction(name="search", parameters={"query": "q"})):
        result = await tool.execute(query="q", tool_action=action)
        assert result.success
    assert calls == [("search", {"query": "q"}, None)] * 2
