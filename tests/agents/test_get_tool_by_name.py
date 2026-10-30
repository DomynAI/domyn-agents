"""Tests for Agent.get_tool_by_name, in particular resolving a display name that
disambiguates MCP tools sharing a name across servers (e.g. "cal.search")."""

from unittest.mock import MagicMock

from domyn.agents import Agent
from domyn.agents.llm import BaseLLMProvider
from domyn.agents.tools import MCPTool


def _make_agent_with_tool(tool: MCPTool) -> Agent:
    return Agent(
        name="TestAgent",
        description="A test agent.",
        instruction="You are a test agent.",
        llm_provider=MagicMock(BaseLLMProvider),
        tools=[tool],
    )


def test_get_tool_by_name_resolves_renamed_tool_repeatedly():
    """A tool renamed for disambiguation must resolve by its new (dotted) name on
    every lookup, not just the first — get_tool_by_name must not mutate the tool
    as a side effect of resolving it."""
    tool = MCPTool(tool_name="search", server_url="https://api.example.com/mcp")
    tool.name = "cal.search"
    agent = _make_agent_with_tool(tool)

    first = agent.get_tool_by_name("cal.search")
    second = agent.get_tool_by_name("cal.search")

    assert first is tool
    assert second is tool
    assert tool.name == "cal.search"
    assert tool.tool_name == "search"


def test_get_tool_by_name_returns_none_for_unknown_name():
    tool = MCPTool(tool_name="search", server_url="https://api.example.com/mcp")
    agent = _make_agent_with_tool(tool)

    assert agent.get_tool_by_name("does_not_exist") is None
