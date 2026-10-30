import pytest

from domyn.agents.tools.mcp import MCPCredentialsCheckResult, MCPTool


def _make_mcp_tool(name: str, use_oauth: bool = True) -> MCPTool:
    return MCPTool(
        tool_name=name,
        server_url="https://example.com/mcp",
        use_oauth=use_oauth,
    )


# ---------------------------------------------------------------------------
# MCPTool.check_auth_credentials()
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_check_auth_credentials_always_valid():
    """No server ping — check_auth_credentials() always reports valid,
    regardless of the tool's actual auth state."""
    tool = _make_mcp_tool("my_tool", use_oauth=True)

    result = await tool.check_auth_credentials()

    assert result == MCPCredentialsCheckResult(
        credentials_valid=True,
        tool_uid=tool.uid,
        tool_name="my_tool",
        server_url="https://example.com/mcp",
    )
