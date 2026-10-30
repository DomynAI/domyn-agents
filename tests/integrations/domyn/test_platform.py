"""Tests for domyn.agents.integrations.domyn.domyn_platform — schema fetching."""

import pytest

from domyn.agents.agents import Agent
from domyn.agents.integrations.domyn import (
    fetch_delegate_tools,
    populate_delegate_tool_schemas,
)
from domyn.agents.integrations.domyn_platform_sdk import PlatformSettings
from domyn.agents.tools import DelegateTool

_BASE_URL = "https://platform.example.com"

_TOOL_DEFS = [
    {
        "name": "web_search",
        "description": "Search the web for up-to-date information.",
        "parameters": [
            {
                "name": "query",
                "type": "str",
                "description": "The search query.",
                "is_required": True,
            },
        ],
    },
    {
        "name": "search_papers_arxiv",
        "description": "Search academic papers on arXiv.",
        "parameters": [
            {"name": "query", "type": "str", "description": "Search terms.", "is_required": True},
            {"name": "max_results", "type": "int", "description": "Max results.", "default": 5},
        ],
    },
]


# ---------------------------------------------------------------------------
# fetch_delegate_tools
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fetch_delegate_tools_returns_populated_tools(platform):
    platform.serve(_TOOL_DEFS)

    tools = await fetch_delegate_tools(
        api_key="key", space_id="space", channel_id="chan", base_url=_BASE_URL
    )

    assert len(tools) == 2
    assert all(isinstance(t, DelegateTool) for t in tools)

    ws = {t.tool_name: t for t in tools}

    assert ws["web_search"].tool_description == "Search the web for up-to-date information."
    assert ws["web_search"].tool_parameters == {
        "query": {"type": "string", "description": "The search query."}
    }
    assert ws["web_search"].required_params == ["query"]

    assert ws["search_papers_arxiv"].tool_parameters == {
        "query": {"type": "string", "description": "Search terms."},
        "max_results": {"type": "integer", "description": "Max results.", "default": 5},
    }
    assert ws["search_papers_arxiv"].required_params == ["query"]


@pytest.mark.asyncio
async def test_fetch_delegate_tools_hands_the_settings_to_the_catalog(platform):
    platform.serve(_TOOL_DEFS)

    await fetch_delegate_tools(
        api_key="key",
        space_id="space",
        channel_id="chan",
        base_url=_BASE_URL,
        configuration_id="cfg-99",
    )

    assert platform.calls[-1]["settings"] == PlatformSettings(
        api_key="key",
        space_id="space",
        channel_id="chan",
        base_url=_BASE_URL,
        configuration_id="cfg-99",
    )


@pytest.mark.asyncio
async def test_fetch_delegate_tools_reads_the_environment(platform, monkeypatch):
    platform.serve(_TOOL_DEFS)
    monkeypatch.setenv("DOMYN_API_KEY", "env-key")
    monkeypatch.setenv("DOMYN_SPACE_ID", "env-space")
    monkeypatch.setenv("DOMYN_CHANNEL_ID", "env-chan")
    monkeypatch.setenv("DOMYN_BASE_URL", "platform.example.com")
    monkeypatch.delenv("DOMYN_CONFIGURATION_ID", raising=False)

    assert len(await fetch_delegate_tools()) == 2
    assert platform.calls[-1]["settings"].space_id == "env-space"


@pytest.mark.asyncio
async def test_fetch_delegate_tools_uses_an_injected_catalog():
    class Catalog:
        async def list_tools(self, *, agent_name=None):
            return _TOOL_DEFS[:1]

    tools = await fetch_delegate_tools(
        api_key="key", space_id="space", channel_id="chan", base_url=_BASE_URL, catalog=Catalog()
    )

    assert [t.tool_name for t in tools] == ["web_search"]


@pytest.mark.asyncio
async def test_fetch_delegate_tools_raises_when_the_platform_does(platform):
    platform.fail(PermissionError("403"))

    with pytest.raises(PermissionError):
        await fetch_delegate_tools(
            api_key="bad-key", space_id="space", channel_id="chan", base_url=_BASE_URL
        )


@pytest.mark.asyncio
async def test_fetch_delegate_tools_raises_without_credentials(platform):
    with pytest.raises(ValueError, match="api_key"):
        await fetch_delegate_tools(api_key="", space_id="", channel_id="", base_url="")


# ---------------------------------------------------------------------------
# populate_delegate_tool_schemas
# ---------------------------------------------------------------------------


def _make_agent_with_delegate_tools(*tools: DelegateTool) -> Agent:
    from unittest.mock import MagicMock

    llm = MagicMock()
    return Agent(
        name="TestAgent",
        description="test",
        instruction="test",
        llm_provider=llm,
        tools=list(tools),
    )


@pytest.mark.asyncio
async def test_populate_fills_empty_tool_parameters(platform):
    platform.serve(_TOOL_DEFS)

    web_search = DelegateTool(tool_name="web_search", tool_description="")
    agent = _make_agent_with_delegate_tools(web_search)

    await populate_delegate_tool_schemas(
        root=agent,
        api_key="key",
        space_id="space",
        channel_id="chan",
        base_url=_BASE_URL,
    )

    assert web_search.tool_parameters == {
        "query": {"type": "string", "description": "The search query."}
    }
    assert web_search.required_params == ["query"]
    assert web_search.tool_description == "Search the web for up-to-date information."


@pytest.mark.asyncio
async def test_populate_overwrites_local_schema_with_platform_schema(platform):
    platform.serve(_TOOL_DEFS)

    web_search = DelegateTool(
        tool_name="web_search",
        tool_description="My local description.",
        tool_parameters={"query": {"type": "string", "description": "Local description."}},
        required_params=["query"],
    )
    agent = _make_agent_with_delegate_tools(web_search)

    await populate_delegate_tool_schemas(
        root=agent,
        api_key="key",
        space_id="space",
        channel_id="chan",
        base_url=_BASE_URL,
    )

    # Tool is found on platform so it must be kept.
    assert web_search in agent.tools
    # Platform schema must overwrite the local one.
    assert web_search.tool_parameters == {
        "query": {"type": "string", "description": "The search query."}
    }
    assert web_search.tool_description == "Search the web for up-to-date information."


@pytest.mark.asyncio
async def test_populate_drops_tool_not_found_on_platform(platform):
    platform.serve(_TOOL_DEFS)

    unknown = DelegateTool(tool_name="unknown_tool", tool_description="")
    agent = _make_agent_with_delegate_tools(unknown)

    await populate_delegate_tool_schemas(
        root=agent,
        api_key="key",
        space_id="space",
        channel_id="chan",
        base_url=_BASE_URL,
    )

    # Tool not found on platform must be removed from the agent.
    assert unknown not in agent.tools


@pytest.mark.asyncio
async def test_populate_drops_tools_on_fetch_failure(platform):
    platform.fail()

    tool = DelegateTool(tool_name="web_search", tool_description="")
    agent = _make_agent_with_delegate_tools(tool)

    # Should not raise — fetch failure is logged as a warning.
    await populate_delegate_tool_schemas(
        root=agent,
        api_key="key",
        space_id="space",
        channel_id="chan",
        base_url=_BASE_URL,
    )

    # All delegate tools must be dropped when the remote fetch fails entirely.
    assert tool not in agent.tools


@pytest.mark.asyncio
async def test_populate_keeps_tools_on_fetch_failure_when_drop_disabled(platform):
    """A periodic refresh (drop_on_fetch_failure=False) must not wipe out the
    existing tool set on a transient fetch failure — only the one-time startup
    call should fail closed."""
    platform.fail()

    tool = DelegateTool(
        tool_name="web_search",
        tool_description="desc",
        tool_parameters={"query": {"type": "string"}},
        required_params=["query"],
    )
    agent = _make_agent_with_delegate_tools(tool)

    await populate_delegate_tool_schemas(
        root=agent,
        api_key="key",
        space_id="space",
        channel_id="chan",
        base_url=_BASE_URL,
        drop_on_fetch_failure=False,
    )

    # The existing tool and its schema must be untouched.
    assert tool in agent.tools
    assert tool.tool_parameters == {"query": {"type": "string"}}
    assert tool.required_params == ["query"]


@pytest.mark.asyncio
async def test_populate_drops_tool_with_explicit_schema_not_found_on_platform(platform):
    """A tool with a local schema that is absent on the platform must still be dropped."""
    platform.serve(_TOOL_DEFS)

    tool = DelegateTool(
        tool_name="unknown_tool",
        tool_description="Local description.",
        tool_parameters={"param": {"type": "string"}},
        required_params=["param"],
    )
    agent = _make_agent_with_delegate_tools(tool)

    await populate_delegate_tool_schemas(
        root=agent,
        api_key="key",
        space_id="space",
        channel_id="chan",
        base_url=_BASE_URL,
    )

    assert tool not in agent.tools


@pytest.mark.asyncio
async def test_populate_fills_tools_in_sub_agents(platform):
    platform.serve(_TOOL_DEFS)

    from unittest.mock import MagicMock

    llm = MagicMock()
    web_search = DelegateTool(tool_name="web_search", tool_description="")
    sub_agent = Agent(
        name="SubAgent", description="sub", instruction="sub", llm_provider=llm, tools=[web_search]
    )
    root = Agent(
        name="RootAgent",
        description="root",
        instruction="root",
        llm_provider=llm,
        sub_agents=[sub_agent],
    )

    await populate_delegate_tool_schemas(
        root=root,
        api_key="key",
        space_id="space",
        channel_id="chan",
        base_url=_BASE_URL,
    )

    assert web_search.tool_parameters["query"]["type"] == "string"


# ---------------------------------------------------------------------------
# populate_delegate_tool_schemas — attach_new_tools
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_populate_attaches_every_platform_tool_when_none_declared(platform):
    platform.serve(_TOOL_DEFS)

    agent = _make_agent_with_delegate_tools()

    await populate_delegate_tool_schemas(
        root=agent,
        api_key="key",
        space_id="space",
        channel_id="chan",
        base_url=_BASE_URL,
    )

    names = {t.tool_name for t in agent.tools}
    assert names == {"web_search", "search_papers_arxiv"}


@pytest.mark.asyncio
async def test_populate_skips_attaching_tool_already_present_on_sub_agent(platform):
    platform.serve(_TOOL_DEFS)

    from unittest.mock import MagicMock

    llm = MagicMock()
    web_search = DelegateTool(tool_name="web_search", tool_description="")
    sub_agent = Agent(
        name="SubAgent", description="sub", instruction="sub", llm_provider=llm, tools=[web_search]
    )
    root = Agent(
        name="RootAgent",
        description="root",
        instruction="root",
        llm_provider=llm,
        sub_agents=[sub_agent],
    )

    await populate_delegate_tool_schemas(
        root=root,
        api_key="key",
        space_id="space",
        channel_id="chan",
        base_url=_BASE_URL,
    )

    # web_search already lives on the sub-agent — must not be duplicated onto root.
    root_names = {t.tool_name for t in root.tools}
    assert root_names == {"search_papers_arxiv"}
    assert web_search in sub_agent.tools


@pytest.mark.asyncio
async def test_populate_does_not_attach_when_disabled(platform):
    platform.serve(_TOOL_DEFS)

    agent = _make_agent_with_delegate_tools()

    await populate_delegate_tool_schemas(
        root=agent,
        api_key="key",
        space_id="space",
        channel_id="chan",
        base_url=_BASE_URL,
        attach_new_tools=False,
    )

    assert not platform.called
    assert agent.tools == []
