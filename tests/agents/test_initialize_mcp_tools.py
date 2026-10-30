"""Tests for Agent.initialize_mcp_tools - MCP server tool autodiscovery."""

from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from mcp.client.auth import OAuthClientProvider
from mcp.types import Tool

from domyn.agents import Agent
from domyn.agents.llm import BaseLLMProvider
from domyn.agents.tools import MCPTool
from domyn.agents.tools.mcp import MCPServerStreamableHTTPConfig


def _make_server_tool(name: str, title: str | None, description: str | None) -> Mock:
    """Create a mock MCP tool schema as returned by discover_tools()."""
    t = Mock(spec=Tool)
    t.name = name
    t.title = title
    t.description = description
    t.input_schema = {"properties": {}}
    return t


def _make_agent_with_server(server: MCPServerStreamableHTTPConfig) -> Agent:
    return Agent(
        name="McpAgentTest",
        description="A test agent with an MCP server.",
        instruction="You are a test agent.",
        llm_provider=MagicMock(BaseLLMProvider),
        mcp_servers=[server],
    )


@pytest.mark.asyncio
async def test_autodiscovered_tools_get_title_and_description_metadata():
    server = MCPServerStreamableHTTPConfig(
        name="docs", url="https://mcp.example.com/mcp", tool_autodiscovery=True
    )
    discovered = [
        _make_server_tool(
            "chat_with_documents_md",
            "Chat with documents (markdown)",
            "Chat with documents.",
        ),
        _make_server_tool("list_documents", None, None),
    ]
    agent = _make_agent_with_server(server)

    with (
        patch.object(
            MCPServerStreamableHTTPConfig, "discover_tools", AsyncMock(return_value=discovered)
        ),
        patch.object(MCPTool, "initialize", AsyncMock()),
    ):
        await agent.initialize_mcp_tools()

    tools_by_name = {t.tool_name: t for t in agent.tools if isinstance(t, MCPTool)}
    assert tools_by_name["chat_with_documents_md"].metadata == {
        "title": "Chat with documents (markdown)",
        "description": "Chat with documents.",
    }
    # No title/description from the server → keys omitted, consumers apply
    # their own fallback.
    assert tools_by_name["list_documents"].metadata == {}


@pytest.mark.asyncio
async def test_autodiscovered_tools_share_the_configured_oauth_provider():
    fake_provider = Mock(spec=OAuthClientProvider)
    server = MCPServerStreamableHTTPConfig(
        name="docs",
        url="https://mcp.example.com/mcp",
        tool_autodiscovery=True,
        use_oauth=True,
        oauth_provider=fake_provider,
    )
    discovered = [
        _make_server_tool("tool_a", None, None),
        _make_server_tool("tool_b", None, None),
    ]
    agent = _make_agent_with_server(server)

    with (
        patch.object(
            MCPServerStreamableHTTPConfig, "discover_tools", AsyncMock(return_value=discovered)
        ),
        patch.object(MCPTool, "initialize", AsyncMock()),
    ):
        await agent.initialize_mcp_tools()

    mcp_tools = [t for t in agent.tools if isinstance(t, MCPTool)]
    assert len(mcp_tools) == 2
    for t in mcp_tools:
        assert t.oauth_provider is fake_provider
    assert mcp_tools[0].oauth_provider is mcp_tools[1].oauth_provider
