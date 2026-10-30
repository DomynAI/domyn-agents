"""Tests for the mcp_init parameter in load_application_from_config_new
and the initialize_mcp_tools tool-list mutation fix in Agent."""

import copy
from unittest.mock import AsyncMock, Mock, patch

import pytest

from domyn.agents.agents import Agent
from domyn.agents.llm import BaseLLMProvider, LLMMessage
from domyn.agents.loader import load_application_from_config_new
from domyn.agents.tools import MCPTool
from domyn.agents.tools.mcp import MCPServerStreamableHTTPConfig

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _MockLLMProvider(BaseLLMProvider):
    type: str = "mock"
    name: str = "mock-llm"
    model_name: str = "mock-model"
    supported_structured_outputs: None = None

    async def generate(self, messages: LLMMessage, **kwargs):
        pass

    async def generate_streaming(self, messages: LLMMessage, **kwargs):
        yield  # type: ignore[misc]


def _base_config() -> dict:
    """Return a fresh deep copy of a minimal valid config each time."""
    return {
        "version": "1.0",
        "entrypoint": "TestAgent",
        "llm_providers": [
            {
                "name": "mock-llm",
                "type": "vllm",
                "model_name": "test-model",
                "url": "http://test.com",
            }
        ],
        "planner_strategies": [{"name": "react", "type": "react"}],
        "agents": [
            {
                "name": "TestAgent",
                "description": "Test agent",
                "instruction": "Test",
                "llm_provider": "mock-llm",
                "planner_strategy": "react",
            }
        ],
    }


def _make_server(
    name: str, url: str, tool_autodiscovery: bool = True
) -> MCPServerStreamableHTTPConfig:
    return MCPServerStreamableHTTPConfig(name=name, url=url, tool_autodiscovery=tool_autodiscovery)


def _make_mcp_tool(name: str, server_url: str, group: str) -> MCPTool:
    return MCPTool(tool_name=name, server_url=server_url, group=group)


def _agent_with_servers(
    *servers: MCPServerStreamableHTTPConfig, tools: list | None = None
) -> Agent:
    return Agent(
        name="TestAgent",
        description="Test",
        instruction="Test",
        llm_provider=_MockLLMProvider(),
        tools=list(tools or []),
        mcp_servers=list(servers),
    )


# ---------------------------------------------------------------------------
# mcp_init parameter tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_mcp_init_true_calls_initialize_mcp_tools():
    """When mcp_init=True, initialize_mcp_tools is called for each agent."""
    with patch.object(Agent, "initialize_mcp_tools", new_callable=AsyncMock) as mock_init:
        await load_application_from_config_new(_base_config(), mcp_init=True)

    mock_init.assert_called_once()


@pytest.mark.asyncio
async def test_mcp_init_default_calls_initialize_mcp_tools():
    """Default value of mcp_init is True — initialize_mcp_tools is called."""
    with patch.object(Agent, "initialize_mcp_tools", new_callable=AsyncMock) as mock_init:
        await load_application_from_config_new(_base_config())

    mock_init.assert_called_once()


@pytest.mark.asyncio
async def test_mcp_init_false_skips_initialize_mcp_tools():
    """When mcp_init=False, initialize_mcp_tools must NOT be called."""
    with patch.object(Agent, "initialize_mcp_tools", new_callable=AsyncMock) as mock_init:
        await load_application_from_config_new(_base_config(), mcp_init=False)

    mock_init.assert_not_called()


@pytest.mark.asyncio
async def test_mcp_init_false_returns_valid_agent():
    """Agent returned when mcp_init=False is a fully constructed Agent instance."""
    agent = await load_application_from_config_new(_base_config(), mcp_init=False)

    assert isinstance(agent, Agent)
    assert agent.name == "TestAgent"


@pytest.mark.asyncio
async def test_mcp_init_false_skips_for_all_subagents():
    """mcp_init=False must also skip initialize_mcp_tools for sub-agents."""
    config = _base_config()
    config["agents"] = [
        {
            "name": "SubAgent",
            "description": "Sub-agent",
            "instruction": "Sub",
            "llm_provider": "mock-llm",
            "planner_strategy": "react",
        },
        {
            "name": "TestAgent",
            "description": "Root agent",
            "instruction": "Root",
            "llm_provider": "mock-llm",
            "planner_strategy": "react",
            "sub_agents": ["SubAgent"],
        },
    ]
    config["entrypoint"] = "TestAgent"

    with patch.object(Agent, "initialize_mcp_tools", new_callable=AsyncMock) as mock_init:
        await load_application_from_config_new(config, mcp_init=False)

    mock_init.assert_not_called()


# ---------------------------------------------------------------------------
# initialize_mcp_tools mutation-fix tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_initialize_mcp_tools_discovery_failure_preserves_original_tools():
    """On discover_tools failure, self.tools must remain unchanged."""
    server = _make_server("failing-server", "https://failing.example.com")
    original_tool = _make_mcp_tool("existing_tool", "https://other.example.com", "other-server")

    agent = _agent_with_servers(server, tools=[original_tool])

    with (
        patch.object(MCPTool, "initialize", new_callable=AsyncMock),
        patch.object(
            MCPServerStreamableHTTPConfig,
            "discover_tools",
            new_callable=AsyncMock,
            side_effect=RuntimeError("network error"),
        ),
    ):
        await agent.initialize_mcp_tools()

    assert original_tool in agent.tools
    assert len(agent.tools) == 1


@pytest.mark.asyncio
async def test_initialize_mcp_tools_discovery_success_replaces_server_tools():
    """On successful discovery, stale tools for the server are replaced with new ones."""
    server = _make_server("my-server", "https://mcp.example.com")

    stale_tool = _make_mcp_tool("old_tool", "https://mcp.example.com", "my-server")
    unrelated_tool = _make_mcp_tool("unrelated_tool", "https://other.example.com", "other-server")

    agent = _agent_with_servers(server, tools=[stale_tool, unrelated_tool])

    discovered_schema = Mock()
    discovered_schema.name = "new_tool"

    with (
        patch.object(
            MCPServerStreamableHTTPConfig,
            "discover_tools",
            new_callable=AsyncMock,
            return_value=[discovered_schema],
        ),
        patch.object(MCPTool, "initialize", new_callable=AsyncMock),
    ):
        await agent.initialize_mcp_tools()

    tool_names = [t.name for t in agent.tools]
    assert "old_tool" not in tool_names, "Stale tool from the same server must be removed"
    assert "new_tool" in tool_names, "Newly discovered tool must be added"
    assert "unrelated_tool" in tool_names, "Tool from another server must be preserved"


@pytest.mark.asyncio
async def test_initialize_mcp_tools_no_autodiscovery_server_keeps_tools():
    """Servers with tool_autodiscovery=False must not touch self.tools."""
    server = _make_server("static-server", "https://static.example.com", tool_autodiscovery=False)
    existing_tool = _make_mcp_tool("existing_tool", "https://static.example.com", "static-server")

    agent = _agent_with_servers(server, tools=[existing_tool])

    with (
        patch.object(MCPTool, "initialize", new_callable=AsyncMock),
        patch.object(
            MCPServerStreamableHTTPConfig, "discover_tools", new_callable=AsyncMock
        ) as mock_discover,
    ):
        await agent.initialize_mcp_tools()

    mock_discover.assert_not_called()
    assert existing_tool in agent.tools
