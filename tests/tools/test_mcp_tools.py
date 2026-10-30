import asyncio
from typing import Any
from unittest.mock import ANY, AsyncMock, Mock, patch

import pytest
from fastmcp import Client
from fastmcp.client import StreamableHttpTransport
from fastmcp.exceptions import ClientError, ToolError, ValidationError
from mcp.types import TextContent, Tool

from domyn.agents.core import ApprovalType, VisibilityScope
from domyn.agents.hooks import HookResult
from domyn.agents.tools import MCPTool, StreamableToolResult
from domyn.agents.tools.base import (
    FunctionDeclaration,
    ToolResult,
    UpdateToolResult,
)


async def _drain(agen) -> ToolResult:
    """Consume an MCPTool async generator and return the final ToolResult."""
    result = None
    async for item in agen:
        result = item
    return result


async def _drain_from_new_tasks(agen, step_timeout: float = 2.0) -> list[Any]:
    """Consume an MCPTool async generator the way the runner does, one new task per step."""
    items: list[Any] = []
    while True:
        step = asyncio.ensure_future(agen.__anext__())
        done, _ = await asyncio.wait({step}, timeout=step_timeout)
        if not done:
            step.cancel()
            await asyncio.gather(step, return_exceptions=True)
            await agen.aclose()
            pytest.fail(f"execute() stopped yielding after {items!r}")
        try:
            items.append(step.result())
        except StopAsyncIteration:
            return items


# ---------------------------------------------------------------------------
# Initialization tests (sync, no fixtures needed)
# ---------------------------------------------------------------------------


def test_mcp_tool_initialization():
    """Test basic MCPTool initialization."""
    tool = MCPTool(
        tool_name="test_tool",
        server_url="https://api.example.com/mcp",
    )
    assert tool.name == "test_tool"
    assert tool.type == "mcp"
    assert tool.use_oauth is False
    assert tool.headers is None


def test_mcp_tool_initialization_with_awaitable_headers():
    """Test MCPTool accepts coroutines as header values."""

    async def get_token():
        return "dynamic-token"

    headers = {"Authorization": get_token}
    tool = MCPTool(
        tool_name="awaitable_tool",
        server_url="https://api.example.com/mcp",
        headers=headers,
    )
    assert "Authorization" in tool.headers


def test_mcp_tool_declaration_without_schema():
    """Test declaration property before schema is loaded."""
    tool = MCPTool(
        tool_name="test_tool",
        server_url="https://api.example.com/mcp",
    )
    declaration = tool.declaration
    assert isinstance(declaration, FunctionDeclaration)
    assert declaration.name == "test_tool"
    assert declaration.description == "test_tool"
    assert declaration.parameters == []


def test_mcp_tool_declaration_with_schema_description():
    """Test that declaration uses description from schema when present."""
    tool = MCPTool(tool_name="described_tool", server_url="https://api.example.com/mcp")
    tool._tool_schema = {
        "properties": {},
        "description": "Does something useful",
    }
    declaration = tool.declaration
    assert declaration.name == "described_tool"
    assert declaration.description == "Does something useful"
    assert declaration.parameters == []


def test_mcp_tool_name_setter_does_not_change_tool_name():
    """Renaming the display name (e.g. to disambiguate tools with the same name across
    MCP servers) must never change tool_name, the identifier execute() sends to the
    remote server."""
    tool = MCPTool(tool_name="search", server_url="https://api.example.com/mcp")
    tool._tool_schema = {"properties": {}, "description": "Searches things"}

    tool.name = "cal.search"

    assert tool.tool_name == "search"
    assert tool.name == "cal.search"
    assert tool.declaration.name == "cal.search"


def test_mcp_tool_declaration_falls_back_to_title():
    """If description is absent, declaration should fall back to the schema title."""
    tool = MCPTool(tool_name="titled_tool", server_url="https://api.example.com/mcp")
    tool._tool_schema = {
        "properties": {},
        "title": "My Tool Title",
    }
    declaration = tool.declaration
    assert declaration.description == "My Tool Title"


def test_mcp_tool_declaration_falls_back_to_tool_name():
    """If both description and title are absent, fall back to tool_name."""
    tool = MCPTool(tool_name="fallback_tool", server_url="https://api.example.com/mcp")
    tool._tool_schema = {"properties": {}}
    declaration = tool.declaration
    assert declaration.description == "fallback_tool"


@pytest.mark.parametrize(
    "json_type, expected_py_type",
    [
        ("string", "str"),
        ("integer", "int"),
        ("number", "float"),
        ("boolean", "bool"),
        ("array", "list"),
        ("object", "dict"),
    ],
)
def test_mcp_tool_declaration_type_mapping(json_type, expected_py_type):
    """Each JSON Schema primitive type should be mapped to the correct Python type."""
    tool = MCPTool(tool_name="typed_tool", server_url="https://api.example.com/mcp")
    tool._tool_schema = {
        "description": "Typed tool",
        "properties": {
            "param": {"type": json_type, "description": "A parameter"},
        },
    }
    declaration = tool.declaration
    assert len(declaration.parameters) == 1
    param = declaration.parameters[0]
    assert param.name == "param"
    assert param.type == expected_py_type


def test_mcp_tool_declaration_unknown_type_falls_back_to_any():
    """An unrecognised JSON Schema type (e.g. 'null') should map to 'any'."""
    tool = MCPTool(tool_name="unknown_type_tool", server_url="https://api.example.com/mcp")
    tool._tool_schema = {
        "description": "Tool with unknown type",
        "properties": {
            "weird": {"type": "null", "description": "A weird param"},
        },
    }
    declaration = tool.declaration
    assert declaration.parameters[0].type == "any"


def test_mcp_tool_declaration_missing_type_falls_back_to_any():
    """A parameter with no 'type' key should default to 'any'."""
    tool = MCPTool(tool_name="no_type_tool", server_url="https://api.example.com/mcp")
    tool._tool_schema = {
        "description": "Tool with untyped param",
        "properties": {
            "untyped": {"description": "No type specified"},
        },
    }
    declaration = tool.declaration
    assert declaration.parameters[0].type == "any"


@pytest.mark.parametrize(
    "json_type, expected_py_type",
    [
        (["string", "number"], "str | float"),
        (["string", "null"], "str | None"),
        (["null"], "any"),
        (["integer"], "int"),
        (["string", "string"], "str"),
        (["string", "weird"], "any"),
        (["string", {"type": "number"}], "any"),
        ([], "any"),
        ({"type": "string"}, "any"),
    ],
)
def test_mcp_tool_declaration_type_union(json_type, expected_py_type):
    """A list of types is valid JSON Schema: it maps to a union, or to 'any' when it can't."""
    tool = MCPTool(tool_name="union_tool", server_url="https://api.example.com/mcp")
    tool._tool_schema = {"properties": {"param": {"type": json_type}}}
    assert tool.declaration.parameters[0].type == expected_py_type


def test_mcp_tool_with_type_union_param_stays_usable():
    """Lokalise's `key_id`: reading the tool raised `unhashable type: 'list'`."""
    tool = MCPTool(tool_name="update_key", server_url="https://api.example.com/mcp")
    tool._tool_schema = {
        "description": "Update a key",
        "type": "object",
        "properties": {"key_id": {"type": ["string", "number"], "description": "Key id"}},
        "required": ["key_id"],
    }

    assert tool.description == "Update a key"
    assert tool.required_param_names() == {"key_id"}
    parameters = tool.tool_definition["function"]["parameters"]
    assert parameters["properties"]["key_id"]["type"] == ["string", "number"]


def test_mcp_tool_declaration_default_param_description():
    """A parameter without a 'description' key should get a default description."""
    tool = MCPTool(tool_name="nodesc_tool", server_url="https://api.example.com/mcp")
    tool._tool_schema = {
        "description": "Tool",
        "properties": {
            "my_param": {"type": "string"},
        },
    }
    declaration = tool.declaration
    assert declaration.parameters[0].description == "Parameter my_param"


def test_mcp_tool_declaration_multiple_params_order_preserved():
    """All parameters should appear in the declaration and preserve insertion order."""
    tool = MCPTool(tool_name="multi_param_tool", server_url="https://api.example.com/mcp")
    tool._tool_schema = {
        "description": "Multi-param tool",
        "properties": {
            "query": {"type": "string", "description": "Search query"},
            "limit": {"type": "integer", "description": "Max results"},
            "include_metadata": {"type": "boolean", "description": "Include extra info"},
            "filters": {"type": "object", "description": "Filter object"},
            "tags": {"type": "array", "description": "Tag list"},
            "score": {"type": "number", "description": "Relevance score"},
        },
    }
    declaration = tool.declaration

    assert len(declaration.parameters) == 6
    names = [p.name for p in declaration.parameters]
    assert names == ["query", "limit", "include_metadata", "filters", "tags", "score"]

    types = [p.type for p in declaration.parameters]
    assert types == ["str", "int", "bool", "dict", "list", "float"]


# ---------------------------------------------------------------------------
# Client initialization tests (need patch + discovery mock)
# ---------------------------------------------------------------------------


def _make_discovery_client(tool_name: str, input_schema: dict | None = None):
    """Helper: create a mock client for initialize() / tool discovery tests."""
    mock_tool = Mock(spec=Tool)
    mock_tool.name = tool_name
    mock_tool.title = None
    mock_tool.description = None
    mock_tool.input_schema = input_schema or {"properties": {}}

    mock_client = AsyncMock(spec=Client)
    mock_client.list_tools = AsyncMock(return_value=[mock_tool])
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    return mock_client, mock_tool


@pytest.mark.asyncio
async def test_mcp_tool_initialize_success():
    """Test successful MCP client initialization and tool discovery."""
    tool = MCPTool(
        tool_name="test_tool",
        server_url="https://api.example.com/mcp",
    )

    mock_client, mock_tool = _make_discovery_client(
        "test_tool",
        {"properties": {"param1": {"type": "string"}}, "description": "Test description"},
    )

    with patch("domyn.agents.tools.mcp.mcp_tool.Client", return_value=mock_client):
        await tool.initialize()

        # Enriched with the parameter descriptions, on a copy: the server's schema is
        # cached and shared by every MCPTool on that server.
        assert tool._tool_schema["properties"] == {
            "param1": {"type": "string", "description": "Parameter param1"}
        }
        assert mock_tool.input_schema == {
            "properties": {"param1": {"type": "string"}},
            "description": "Test description",
        }
        mock_client.list_tools.assert_called_once()


@pytest.mark.asyncio
async def test_mcp_tool_initialize_with_oauth():
    """Test MCP client initialization with OAuth."""
    tool = MCPTool(
        tool_name="secure_tool",
        server_url="https://api.example.com/mcp",
        use_oauth=True,
    )

    mock_client, _ = _make_discovery_client("secure_tool")

    with patch("domyn.agents.tools.mcp.mcp_tool.Client", return_value=mock_client):
        await tool.initialize()
        assert tool.use_oauth is True


@pytest.mark.asyncio
async def test_mcp_tool_initialize_resolves_awaitable_headers():
    """Coroutine header values are awaited and resolved to strings during initialization."""

    async def get_token() -> str:
        return "dynamic-token-xyz"

    tool = MCPTool(
        tool_name="awaitable_auth_tool",
        server_url="https://api.example.com/mcp",
        headers={"Authorization": get_token, "X-Static": "static-value"},
    )

    mock_client, _ = _make_discovery_client("awaitable_auth_tool")

    with patch("domyn.agents.tools.mcp.mcp_tool.Client", return_value=mock_client):
        await tool.initialize()

    assert tool.headers == {
        "Authorization": "dynamic-token-xyz",
        "X-Static": "static-value",
    }


@pytest.mark.asyncio
async def test_mcp_tool_initialize_resolves_all_awaitable_headers():
    """Multiple coroutine header values are all awaited during initialization."""

    async def get_token() -> str:
        return "bearer-abc"

    async def get_api_key() -> str:
        return "key-123"

    tool = MCPTool(
        tool_name="multi_awaitable_tool",
        server_url="https://api.example.com/mcp",
        headers={"Authorization": get_token, "X-Api-Key": get_api_key},
    )

    mock_client, _ = _make_discovery_client("multi_awaitable_tool")

    with patch("domyn.agents.tools.mcp.mcp_tool.Client", return_value=mock_client):
        await tool.initialize()

    assert tool.headers == {"Authorization": "bearer-abc", "X-Api-Key": "key-123"}


@pytest.mark.asyncio
async def test_mcp_tool_initialize_uses_streamable_http_transport():
    """initialize() creates client via StreamableHttpTransport with correct args."""
    tool = MCPTool(
        tool_name="transport_tool",
        server_url="https://api.example.com/mcp",
        headers={"Authorization": "Bearer token"},
    )

    mock_client, _ = _make_discovery_client("transport_tool")
    mock_transport = Mock(spec=StreamableHttpTransport)

    with (
        patch(
            "domyn.agents.tools.mcp.mcp_tool.StreamableHttpTransport",
            return_value=mock_transport,
        ) as mock_transport_cls,
        patch("domyn.agents.tools.mcp.mcp_tool.Client", return_value=mock_client),
    ):
        await tool.initialize()

    mock_transport_cls.assert_called_once_with(
        url="https://api.example.com/mcp",
        headers={"Authorization": "Bearer token"},
        auth=None,
    )


@pytest.mark.asyncio
async def test_mcp_tool_initialize_streamable_http_transport_no_headers():
    """initialize() passes headers=None to StreamableHttpTransport when no headers set."""
    tool = MCPTool(
        tool_name="no_headers_tool",
        server_url="https://api.example.com/mcp",
    )

    mock_client, _ = _make_discovery_client("no_headers_tool")

    with (
        patch(
            "domyn.agents.tools.mcp.mcp_tool.StreamableHttpTransport",
        ) as mock_transport_cls,
        patch("domyn.agents.tools.mcp.mcp_tool.Client", return_value=mock_client),
    ):
        await tool.initialize()

    mock_transport_cls.assert_called_once_with(
        url="https://api.example.com/mcp",
        headers=None,
        auth=None,
    )


@pytest.mark.asyncio
async def test_mcp_tool_initialize_tool_not_found():
    """Test initialization when tool is not found on server."""
    tool = MCPTool(
        tool_name="nonexistent_tool",
        server_url="https://api.example.com/mcp",
    )

    mock_other_tool = Mock()
    mock_other_tool.name = "other_tool"

    mock_client = AsyncMock(spec=Client)
    mock_client.list_tools = AsyncMock(return_value=[mock_other_tool])
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)

    with (
        patch("domyn.agents.tools.mcp.mcp_tool.Client", return_value=mock_client),
        pytest.raises(RuntimeError, match="Tool 'nonexistent_tool' not found"),
    ):
        await tool.initialize()


@pytest.mark.asyncio
async def test_mcp_tool_initialize_caching():
    """Test that client is cached and not recreated."""
    tool = MCPTool(
        tool_name="cached_tool",
        server_url="https://api.example.com/mcp",
    )

    mock_client, _ = _make_discovery_client("cached_tool")

    with patch("domyn.agents.tools.mcp.mcp_tool.Client", return_value=mock_client):
        client1 = await tool.initialize()
        client2 = await tool.initialize()

        assert client1 is client2
        assert mock_client.list_tools.call_count == 1


# ---------------------------------------------------------------------------
# Execution tests (use make_mock_response / make_mock_client fixtures)
# ---------------------------------------------------------------------------

QUERY_SCHEMA: dict[str, Any] = {"properties": {"query": {"type": "string"}}}


@pytest.mark.asyncio
async def test_mcp_tool_execute_success(make_mock_response, make_mock_client):
    """Test successful tool execution with TextContent response."""
    tool = MCPTool(tool_name="test_tool", server_url="https://api.example.com/mcp", timeout=60)
    tool._tool_schema = QUERY_SCHEMA

    mock_response = make_mock_response("Result data")
    mock_client = make_mock_client(mock_response)
    tool._client = mock_client
    tool._build_client = AsyncMock(return_value=mock_client)

    result = await _drain(tool.execute(query="test query"))

    assert isinstance(result, ToolResult)
    assert result.success is True
    assert result.data == "Result data"
    assert result.metadata["mcp_server"] == "https://api.example.com/mcp"
    assert result.metadata["tool_name"] == "test_tool"
    mock_client.call_tool.assert_called_once_with(
        name="test_tool",
        arguments={"query": "test query"},
        meta={"state": {}},
        progress_handler=ANY,
    )


@pytest.mark.asyncio
async def test_mcp_tool_execute_with_multiple_content(make_mock_response, make_mock_client):
    """Test execution with multiple content items in response."""
    tool = MCPTool(tool_name="multi_tool", server_url="https://api.example.com/mcp")
    tool._tool_schema = {"properties": {}}

    mock_response = make_mock_response(
        content=[
            TextContent(type="text", text="First result"),
            TextContent(type="text", text="Second result"),
        ]
    )
    mock_client = make_mock_client(mock_response)
    tool._client = mock_client
    tool._build_client = AsyncMock(return_value=mock_client)

    result = await _drain(tool.execute())

    assert result.success is True
    assert result.data == "First result\nSecond result"


@pytest.mark.asyncio
async def test_mcp_tool_execute_unwraps_fastmcp_wrapped_structured_content(
    make_mock_response, make_mock_client
):
    """FastMCP wrap_result is a transport envelope, not part of the logical tool result."""
    tool = MCPTool(tool_name="wrapped_tool", server_url="https://api.example.com/mcp")
    tool._tool_schema = {"properties": {}}
    payload = {"summary": "ready-to-send", "details": {"risk": "medium"}}

    mock_response = make_mock_response(
        structured_content={"result": payload},
        meta={"fastmcp": {"wrap_result": True}},
    )
    mock_client = make_mock_client(mock_response)
    tool._client = mock_client
    tool._build_client = AsyncMock(return_value=mock_client)

    result = await _drain(tool.execute())

    assert result.success is True
    assert result.data == payload
    assert result.metadata["fastmcp"] == {"wrap_result": True}


@pytest.mark.asyncio
async def test_mcp_tool_execute_unwraps_result_key_via_unified_fallback(
    make_mock_response, make_mock_client
):
    """A structured result with a "result" key is lifted into `data` via the unified
    legacy fallback (consistent with FunctionTool/StreamableFunctionTool)."""
    tool = MCPTool(tool_name="plain_structured_tool", server_url="https://api.example.com/mcp")
    tool._tool_schema = {"properties": {}}
    structured_content = {"result": {"value": 42}}

    mock_response = make_mock_response(structured_content=structured_content)
    mock_client = make_mock_client(mock_response)
    tool._client = mock_client
    tool._build_client = AsyncMock(return_value=mock_client)

    result = await _drain(tool.execute())

    assert result.success is True
    assert result.data == {"value": 42}
    assert result.metadata["mcp_server"] == "https://api.example.com/mcp"


@pytest.mark.asyncio
async def test_mcp_tool_execute_multi_key_result_envelope_lifts_result(
    make_mock_response, make_mock_client
):
    """A multi-key dict carrying a "result" key is treated as a legacy envelope:
    `result` is lifted into `data` (sibling keys are not preserved), matching the
    FunctionTool convention."""
    tool = MCPTool(tool_name="multi_key_tool", server_url="https://api.example.com/mcp")
    tool._tool_schema = {"properties": {}}
    structured_content = {"result": {"value": 42}, "status": "ok"}

    mock_response = make_mock_response(
        structured_content=structured_content,
        meta={"fastmcp": {"wrap_result": True}},
    )
    mock_client = make_mock_client(mock_response)
    tool._client = mock_client
    tool._build_client = AsyncMock(return_value=mock_client)

    result = await _drain(tool.execute())

    assert result.success is True
    assert result.data == {"value": 42}


@pytest.mark.asyncio
async def test_mcp_tool_execute_casts_tool_result_shaped_structured_content(
    make_mock_response, make_mock_client
):
    """A ToolResult-shaped structured content is cast directly, merging the MCP envelope."""
    tool = MCPTool(tool_name="shaped_tool", server_url="https://api.example.com/mcp")
    tool._tool_schema = {"properties": {}}
    structured_content = {
        "success": True,
        "data": {"value": 42},
        "metadata": {"server_field": "v"},
    }

    mock_response = make_mock_response(structured_content=structured_content)
    mock_client = make_mock_client(mock_response)
    tool._client = mock_client
    tool._build_client = AsyncMock(return_value=mock_client)

    result = await _drain(tool.execute())

    assert result.success is True
    # Cast directly: data is the inner payload, not the whole envelope.
    assert result.data == {"value": 42}
    # MCP envelope metadata is merged in alongside the server-provided metadata.
    assert result.metadata["mcp_server"] == "https://api.example.com/mcp"
    assert result.metadata["tool_name"] == "shaped_tool"
    assert result.metadata["server_field"] == "v"


@pytest.mark.asyncio
async def test_mcp_tool_execute_filters_unexpected_params(make_mock_response, make_mock_client):
    """Test that unexpected parameters are filtered out."""
    tool = MCPTool(tool_name="filter_tool", server_url="https://api.example.com/mcp")
    tool._tool_schema = {"properties": {"valid_param": {"type": "string"}}}

    mock_client = make_mock_client(make_mock_response("Success"))
    tool._client = mock_client
    tool._build_client = AsyncMock(return_value=mock_client)

    result = await _drain(tool.execute(valid_param="ok", invalid_param="should be filtered"))

    mock_client.call_tool.assert_called_once_with(
        name="filter_tool",
        arguments={"valid_param": "ok"},
        meta={"state": {}},
        progress_handler=ANY,
    )
    assert result.success is True


@pytest.mark.asyncio
async def test_mcp_tool_execute_returns_a_long_response_whole(make_mock_response, make_mock_client):
    """A cut here, before the result became an event, lost the rest for good: not
    in the history, not in storage. Fitting it to the model is the context
    manager's job, which keeps the start and the end."""
    tool = MCPTool(tool_name="long_tool", server_url="https://api.example.com/mcp")
    tool._tool_schema = {"properties": {}}

    mock_client = make_mock_client(make_mock_response("x" * 15000))
    tool._client = mock_client
    tool._build_client = AsyncMock(return_value=mock_client)

    result = await _drain(tool.execute())

    assert result.success is True
    assert result.data == "x" * 15000


@pytest.mark.asyncio
async def test_mcp_tool_execute_error_handling(make_mock_client):
    """Test error handling during execution (generic Exception)."""
    tool = MCPTool(tool_name="error_tool", server_url="https://api.example.com/mcp")
    tool._tool_schema = {"properties": {}}

    mock_client = make_mock_client(side_effect=Exception("Connection error"))
    tool._client = mock_client
    tool._build_client = AsyncMock(return_value=mock_client)

    result = await _drain(tool.execute())

    assert isinstance(result, ToolResult)
    assert result.success is False
    assert "Connection error" in (result.error or "")


@pytest.mark.asyncio
async def test_mcp_tool_execute_tool_error(make_mock_client):
    """ToolError is caught and appends 'Is the MCP server running?' message."""
    tool = MCPTool(tool_name="tool_error_tool", server_url="https://api.example.com/mcp")
    tool._tool_schema = {"properties": {}}

    mock_client = make_mock_client(side_effect=ToolError("tool failed"))
    tool._client = mock_client
    tool._build_client = AsyncMock(return_value=mock_client)

    result = await _drain(tool.execute())

    assert isinstance(result, ToolResult)
    assert result.success is False
    assert "tool failed" in (result.error or "")
    assert "Is the MCP server running?" in (result.error or "")


@pytest.mark.asyncio
async def test_mcp_tool_execute_validation_error(make_mock_client):
    """ValidationError is caught and reported without server availability hint."""
    tool = MCPTool(tool_name="validation_error_tool", server_url="https://api.example.com/mcp")
    tool._tool_schema = {"properties": {}}

    mock_client = make_mock_client(side_effect=ValidationError("bad input"))
    tool._client = mock_client
    tool._build_client = AsyncMock(return_value=mock_client)

    result = await _drain(tool.execute())

    assert isinstance(result, ToolResult)
    assert result.success is False
    assert "bad input" in (result.error or "")
    assert "Is the MCP server running?" not in (result.error or "")


@pytest.mark.asyncio
async def test_mcp_tool_execute_client_error(make_mock_client):
    """ClientError is caught and appends 'Is the MCP server running?' message."""
    tool = MCPTool(tool_name="client_error_tool", server_url="https://api.example.com/mcp")
    tool._tool_schema = {"properties": {}}

    mock_client = make_mock_client(side_effect=ClientError("transport failure"))
    tool._client = mock_client
    tool._build_client = AsyncMock(return_value=mock_client)

    result = await _drain(tool.execute())

    assert isinstance(result, ToolResult)
    assert result.success is False
    assert "transport failure" in (result.error or "")
    assert "Is the MCP server running?" in (result.error or "")


@pytest.mark.asyncio
async def test_mcp_tool_execute_list_response(make_mock_client):
    """Test execution with list response format."""
    tool = MCPTool(tool_name="list_tool", server_url="https://api.example.com/mcp")
    tool._tool_schema = {"properties": {}}

    mock_response = [
        TextContent(type="text", text="Item 1"),
        TextContent(type="text", text="Item 2"),
    ]
    mock_client = make_mock_client(mock_response)
    tool._client = mock_client
    tool._build_client = AsyncMock(return_value=mock_client)

    result = await _drain(tool.execute())

    assert result.success is True
    assert result.data == "Item 1\nItem 2"


@pytest.mark.asyncio
async def test_mcp_tool_execute_non_text_content(make_mock_response, make_mock_client):
    """Test execution with non-TextContent response."""
    tool = MCPTool(tool_name="generic_tool", server_url="https://api.example.com/mcp")
    tool._tool_schema = {"properties": {}}

    mock_item = Mock()
    mock_item.__str__ = Mock(return_value="Generic content")
    mock_response = make_mock_response(content=[mock_item])
    mock_client = make_mock_client(mock_response)
    tool._client = mock_client
    tool._build_client = AsyncMock(return_value=mock_client)

    result = await _drain(tool.execute())

    assert result.success is True
    assert "Generic content" in result.data


# ---------------------------------------------------------------------------
# from_config tests (sync)
# ---------------------------------------------------------------------------


def test_mcp_tool_from_config_basic():
    """Test creating MCPTool from basic config dict."""
    config = {
        "server_url": "https://api.example.com/mcp",
        "tool_name": "config_tool",
    }
    tool = MCPTool.from_config(config)

    assert isinstance(tool, MCPTool)
    assert tool.server_url == "https://api.example.com/mcp"
    assert tool.tool_name == "config_tool"
    assert tool.name == "config_tool"


def test_mcp_tool_from_config_with_name_override():
    """Test from_config with name override."""
    config = {
        "server_url": "https://api.example.com/mcp",
        "tool_name": "original_name",
        "name": "override_name",
    }
    tool = MCPTool.from_config(config)

    assert tool.tool_name == "override_name"
    assert tool.name == "override_name"


def test_mcp_tool_from_config_with_oauth():
    """Test from_config with OAuth settings."""
    config = {
        "server_url": "https://api.example.com/mcp",
        "tool_name": "secure_tool",
        "use_oauth": True,
    }
    tool = MCPTool.from_config(config)

    assert tool.use_oauth is True


def test_mcp_tool_from_config_with_headers():
    """Test from_config with headers."""
    config = {
        "server_url": "https://api.example.com/mcp",
        "tool_name": "auth_tool",
        "headers": {"Authorization": "Bearer token"},
    }
    tool = MCPTool.from_config(config)

    assert tool.headers == {"Authorization": "Bearer token"}


def test_mcp_tool_from_config_with_visibility():
    """Test from_config with visibility levels."""
    for vis_value, vis_enum in [
        ("global", VisibilityScope.GLOBAL),
        ("trusted", VisibilityScope.TRUSTED),
        ("scoped", VisibilityScope.SCOPED),
        ("isolated", VisibilityScope.ISOLATED),
    ]:
        config = {
            "server_url": "https://api.example.com/mcp",
            "tool_name": "auth_tool",
            "visibility": vis_value,
        }
        tool = MCPTool.from_config(config)
        assert tool.visibility == vis_enum


def test_mcp_tool_from_config_with_hooks():
    """Test from_config with hooks configuration."""
    config = {
        "server_url": "https://api.example.com/mcp",
        "tool_name": "hooked_tool",
        "pre_hooks": [],
        "post_hooks": [],
    }
    tool = MCPTool.from_config(config)

    assert tool.pre_hooks == []
    assert tool.post_hooks == []


def test_mcp_tool_from_config_with_approval():
    """Test from_config with user confirmation."""
    config = {
        "server_url": "https://api.example.com/mcp",
        "tool_name": "confirm_tool",
        "approval_type": "user_confirm",
    }
    tool = MCPTool.from_config(config)

    assert tool.approval_type == ApprovalType.USER_CONFIRM


def test_mcp_tool_from_config_registry_registration():
    """Test from_config with registry registration."""
    from domyn.agents.registry import get_tool_registry

    registry = get_tool_registry()
    initial_count = len(registry._tools)

    config = {
        "server_url": "https://api.example.com/mcp",
        "tool_name": "registry_tool",
    }
    MCPTool.from_config(config, save_in_registry=True)

    assert len(registry._tools) == initial_count + 1
    assert "registry_tool" in registry._tools


# ---------------------------------------------------------------------------
# Visibility / state scoping tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_mcp_tool_execute(make_mock_response, make_mock_client):
    """Test execution extracting from state and updating from response."""
    tool = MCPTool(tool_name="meta_tool", server_url="https://api.example.com/mcp")
    tool._tool_schema = QUERY_SCHEMA

    mock_client = make_mock_client(
        make_mock_response(meta={"state": {"user_id": "user-456", "session_id": "session-updated"}})
    )
    tool._client = mock_client
    tool._build_client = AsyncMock(return_value=mock_client)

    result = await _drain(
        tool.execute(
            query="test",
            state={
                "meta_tool": {"user_id": "user-123", "session_id": "session-abc"},
                "other_field": "ignored",
            },
        )
    )

    mock_client.call_tool.assert_called_once_with(
        name="meta_tool",
        arguments={"query": "test"},
        meta={"state": {"user_id": "user-123", "session_id": "session-abc"}},
        progress_handler=ANY,
    )

    assert result.success is True
    assert result.state is not None
    assert result.state["meta_tool"]["user_id"] == "user-456"
    assert result.state["meta_tool"]["session_id"] == "session-updated"
    assert result.state["other_field"] == "ignored"


@pytest.mark.asyncio
async def test_mcp_tool_global_visibility(make_mock_response, make_mock_client):
    """Test tools with global visibility correctly read and update the state."""
    tool = MCPTool(
        tool_name="global_tool",
        server_url="https://api.example.com/mcp",
        visibility=VisibilityScope.GLOBAL,
    )
    tool._tool_schema = QUERY_SCHEMA

    mock_client = make_mock_client(
        make_mock_response(meta={"state": {"global_field": "global_value_updated"}})
    )
    tool._client = mock_client
    tool._build_client = AsyncMock(return_value=mock_client)

    result = await _drain(
        tool.execute(
            query="test",
            state={"global_field": "global_value_old", "ignored_field": "ignored_value"},
        )
    )

    mock_client.call_tool.assert_called_once_with(
        name="global_tool",
        arguments={"query": "test"},
        meta={"state": {"global_field": "global_value_old", "ignored_field": "ignored_value"}},
        progress_handler=ANY,
    )

    assert result.success is True
    assert result.state is not None
    assert result.state["global_field"] == "global_value_updated"
    assert result.state["ignored_field"] == "ignored_value"


@pytest.mark.asyncio
async def test_mcp_tool_scoped_visibility(make_mock_response, make_mock_client):
    """Test SCOPED visibility: state namespaced by group, no run_config."""
    tool = MCPTool(
        tool_name="search_tool",
        server_url="https://api.partner.com/mcp",
        visibility=VisibilityScope.SCOPED,
        group="partner-server",
    )
    tool._tool_schema = QUERY_SCHEMA

    mock_client = make_mock_client(make_mock_response(meta={"state": {"cursor": "page2"}}))
    tool._client = mock_client
    tool._build_client = AsyncMock(return_value=mock_client)

    result = await _drain(
        tool.execute(
            query="test",
            state={
                "partner-server": {"cursor": "page1"},
                "other-server": {"ignored": True},
            },
            run_config={"user_id": "secret"},
        )
    )

    mock_client.call_tool.assert_called_once_with(
        name="search_tool",
        arguments={"query": "test"},
        meta={"state": {"cursor": "page1"}},
        progress_handler=ANY,
    )

    assert result.success is True
    assert result.state["partner-server"]["cursor"] == "page2"
    assert result.state["other-server"]["ignored"] is True


@pytest.mark.asyncio
async def test_mcp_tool_scopes_tool_returned_state(make_mock_response, make_mock_client):
    """A `state` returned inside the tool's result is scoped by visibility, exactly
    like the state returned via `_meta`."""
    tool = MCPTool(
        tool_name="search_tool",
        server_url="https://api.partner.com/mcp",
        visibility=VisibilityScope.SCOPED,
        group="partner-server",
    )
    tool._tool_schema = QUERY_SCHEMA

    # Tool returns a ToolResult-shaped dict carrying its own state (unscoped).
    structured_content = {
        "success": True,
        "data": "done",
        "state": {"cursor": "from_result"},
    }
    mock_client = make_mock_client(make_mock_response(structured_content=structured_content))
    tool._client = mock_client
    tool._build_client = AsyncMock(return_value=mock_client)

    result = await _drain(tool.execute(query="test", state={"partner-server": {"cursor": "page1"}}))

    assert result.success is True
    assert result.data == "done"
    # The returned state is nested under the group key (SCOPED), not at top level.
    assert result.state["partner-server"]["cursor"] == "from_result"
    assert "cursor" not in result.state


@pytest.mark.asyncio
async def test_mcp_tool_isolated_merges_meta_and_result_state(make_mock_response, make_mock_client):
    """ISOLATED: a `_meta` state and a tool-returned state are merged before scoping,
    so neither clobbers the other under the shared tool_name key."""
    tool = MCPTool(tool_name="my_tool", server_url="https://api.example.com/mcp")
    tool._tool_schema = QUERY_SCHEMA

    structured_content = {
        "success": True,
        "data": "done",
        "state": {"from_result": 2},
    }
    mock_response = make_mock_response(
        structured_content=structured_content,
        meta={"state": {"from_meta": 1}},
    )
    mock_client = make_mock_client(mock_response)
    tool._client = mock_client
    tool._build_client = AsyncMock(return_value=mock_client)

    result = await _drain(tool.execute(query="test"))

    assert result.success is True
    # Both states survive, merged under the tool_name key.
    assert result.state["my_tool"] == {"from_meta": 1, "from_result": 2}


@pytest.mark.asyncio
async def test_mcp_tool_isolated_ignores_group(make_mock_response, make_mock_client):
    """Test ISOLATED visibility: state per tool_name, NOT per group."""
    tool = MCPTool(
        tool_name="my_tool",
        server_url="https://api.partner.com/mcp",
        group="partner-server",  # group is set but ISOLATED ignores it
    )
    tool._tool_schema = QUERY_SCHEMA

    mock_client = make_mock_client(make_mock_response(meta={"state": {"key": "updated"}}))
    tool._client = mock_client
    tool._build_client = AsyncMock(return_value=mock_client)

    result = await _drain(
        tool.execute(
            query="test",
            state={
                "my_tool": {"key": "original"},
                "partner-server": {"should_not_read": True},
            },
        )
    )

    mock_client.call_tool.assert_called_once_with(
        name="my_tool",
        arguments={"query": "test"},
        meta={"state": {"key": "original"}},
        progress_handler=ANY,
    )

    assert result.state["my_tool"]["key"] == "updated"
    assert result.state["partner-server"]["should_not_read"] is True


@pytest.mark.asyncio
async def test_mcp_tool_trusted_visibility(make_mock_response, make_mock_client):
    """Test TRUSTED visibility: state per group + run_config (PII)."""
    tool = MCPTool(
        tool_name="ingest_doc",
        server_url="https://acme.partner.com/mcp",
        visibility=VisibilityScope.TRUSTED,
        group="acme-server",
    )
    tool._tool_schema = {"properties": {"doc_url": {"type": "string"}}}

    mock_client = make_mock_client(
        make_mock_response("Ingested", meta={"state": {"doc_id": "doc-999"}})
    )
    tool._client = mock_client
    tool._build_client = AsyncMock(return_value=mock_client)

    result = await _drain(
        tool.execute(
            doc_url="https://example.com/file.pdf",
            state={
                "acme-server": {"session": "abc"},
                "other-server": {"private": True},
            },
            run_config={"user_id": "user-42", "space_id": "space-7"},
        )
    )

    mock_client.call_tool.assert_called_once_with(
        name="ingest_doc",
        arguments={"doc_url": "https://example.com/file.pdf"},
        meta={"user_id": "user-42", "space_id": "space-7", "state": {"session": "abc"}},
        progress_handler=ANY,
    )

    assert result.success is True
    assert result.state["acme-server"]["doc_id"] == "doc-999"
    assert result.state["other-server"]["private"] is True


@pytest.mark.asyncio
async def test_mcp_tool_execute_from_run_config(make_mock_response, make_mock_client):
    """Test that GLOBAL tools receive run_config in meta."""
    tool = MCPTool(
        tool_name="meta_tool",
        server_url="https://api.example.com/mcp",
        visibility=VisibilityScope.GLOBAL,
    )
    tool._tool_schema = QUERY_SCHEMA

    mock_client = make_mock_client(make_mock_response(meta={"user_id": "user-updated"}))
    tool._client = mock_client
    tool._build_client = AsyncMock(return_value=mock_client)

    result = await _drain(
        tool.execute(
            query="test",
            state={},
            run_config={"user_id": "user-from-config", "other_config": "value"},
        )
    )

    mock_client.call_tool.assert_called_once_with(
        name="meta_tool",
        arguments={"query": "test"},
        meta={"user_id": "user-from-config", "other_config": "value", "state": {}},
        progress_handler=ANY,
    )

    assert result.success is True
    assert result.metadata is not None
    assert result.metadata.get("user_id") == "user-updated"


@pytest.mark.asyncio
async def test_mcp_tool_isolated_excludes_run_config(make_mock_response, make_mock_client):
    """Test that ISOLATED tools do NOT receive run_config in meta (PII protection)."""
    tool = MCPTool(tool_name="external_tool", server_url="https://api.partner.com/mcp")
    tool._tool_schema = QUERY_SCHEMA

    mock_client = make_mock_client(make_mock_response())
    tool._client = mock_client
    tool._build_client = AsyncMock(return_value=mock_client)

    result = await _drain(
        tool.execute(
            query="test",
            state={"external_tool": {"session": "abc"}},
            run_config={"user_id": "secret-user", "space_id": "secret-space"},
        )
    )

    mock_client.call_tool.assert_called_once_with(
        name="external_tool",
        arguments={"query": "test"},
        meta={"state": {"session": "abc"}},
        progress_handler=ANY,
    )
    assert result.success is True
    actual_meta = mock_client.call_tool.call_args.kwargs["meta"]
    assert "user_id" not in actual_meta
    assert "space_id" not in actual_meta


@pytest.mark.asyncio
async def test_mcp_tool_default_visibility_is_isolated():
    """Test that default visibility is ISOLATED (fail-safe: no run_config)."""
    tool = MCPTool(
        tool_name="default_tool",
        server_url="https://api.example.com/mcp",
    )
    assert tool.visibility == VisibilityScope.ISOLATED


@pytest.mark.asyncio
async def test_mcp_tool_execute_no_response_meta(make_mock_response, make_mock_client):
    """Test execution when response has no meta."""
    tool = MCPTool(tool_name="meta_tool", server_url="https://api.example.com/mcp")
    tool._tool_schema = QUERY_SCHEMA

    mock_client = make_mock_client(make_mock_response())
    tool._client = mock_client
    tool._build_client = AsyncMock(return_value=mock_client)

    result = await _drain(tool.execute(query="test", state={"user_id": "user-123"}))

    mock_client.call_tool.assert_called_once_with(
        name="meta_tool",
        arguments={"query": "test"},
        meta={"state": {}},
        progress_handler=ANY,
    )

    assert result.success is True
    assert result.state is not None
    assert result.state["user_id"] == "user-123"


@pytest.mark.asyncio
async def test_mcp_tool_disconnect():
    """Test disconnect method."""
    tool = MCPTool(
        tool_name="disconnect_tool",
        server_url="https://api.example.com/mcp",
    )
    tool._client = AsyncMock(spec=Client)

    await tool.disconnect()

    assert tool._client is None


# ---------------------------------------------------------------------------
# Healthcheck / timeout tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_mcp_tool_healthcheck_ping_fails_raises_runtime_error(
    make_mock_response, make_mock_client
):
    """If the healthcheck ping fails while the tool call is in-flight, execute()
    should return a failed ToolResult with a meaningful error message."""
    tool = MCPTool(tool_name="slow_tool", server_url="https://api.example.com/mcp")
    tool._tool_schema = {"properties": {}}
    # Reduce timeout so the test doesn't sleep 5 s for real.
    tool._timeout_ping = 0.05

    # call_tool hangs until cancelled (simulates a slow remote call).
    async def _slow_call_tool(*args, **kwargs):
        await asyncio.sleep(10)  # will be cancelled by TaskGroup

    main_client = make_mock_client()
    main_client.call_tool = AsyncMock(side_effect=_slow_call_tool)
    tool._client = main_client

    # probe_client whose ping() always raises → triggers the healthcheck RuntimeError.
    probe_client = AsyncMock(spec=Client)
    probe_client.__aenter__ = AsyncMock(return_value=probe_client)
    probe_client.__aexit__ = AsyncMock(return_value=None)
    probe_client.ping = AsyncMock(side_effect=ConnectionError("server gone"))

    tool._build_client = AsyncMock(return_value=probe_client)

    result = await _drain(tool.execute())

    assert isinstance(result, ToolResult)
    assert result.success is False
    assert "stopped responding" in (result.error or "")


@pytest.mark.asyncio
async def test_mcp_tool_healthcheck_exits_cleanly_when_call_finishes_fast(
    make_mock_response, make_mock_client
):
    """If the tool call completes before the first healthcheck ping, the healthcheck
    loop should exit without ever calling ping()."""
    tool = MCPTool(tool_name="fast_tool", server_url="https://api.example.com/mcp")
    tool._tool_schema = {"properties": {}}
    # Long ping interval → healthcheck will never reach ping() before call_done fires.
    tool._timeout_ping = 60.0

    mock_response = make_mock_response("Done fast")
    main_client = make_mock_client(mock_response)
    tool._client = main_client

    probe_client = AsyncMock(spec=Client)
    probe_client.__aenter__ = AsyncMock(return_value=probe_client)
    probe_client.__aexit__ = AsyncMock(return_value=None)
    probe_client.ping = AsyncMock(return_value=None)

    tool._build_client = AsyncMock(return_value=probe_client)

    result = await _drain(tool.execute())

    assert result.success is True
    assert result.data == "Done fast"
    # ping should not have been called because the tool finished before the first sleep expired.
    probe_client.ping.assert_not_called()


@pytest.mark.asyncio
async def test_mcp_tool_a_finished_call_does_not_wait_for_the_ping_interval(
    make_mock_response, make_mock_client
):
    """The call returns when it is done, not when the health check next wakes up,
    and no probe connection is opened for a call that ends before the first ping.

    The loop slept the whole interval, and the TaskGroup waited for it: every call
    took at least five seconds, whatever the tool did."""
    tool = MCPTool(tool_name="fast_tool", server_url="https://api.example.com/mcp")
    tool._tool_schema = {"properties": {}}
    tool._timeout_ping = 30.0
    tool._client = make_mock_client(make_mock_response("Done"))
    tool._build_client = AsyncMock()

    started = asyncio.get_running_loop().time()
    result = await _drain(tool.execute())

    assert result.success is True
    assert asyncio.get_running_loop().time() - started < 5.0
    tool._build_client.assert_not_called()


@pytest.mark.asyncio
async def test_mcp_tool_healthcheck_calls_ping_after_each_interval(
    make_mock_response, make_mock_client
):
    """Healthcheck should call ping() once for each interval that elapses while
    the tool call is in-flight."""
    tool = MCPTool(tool_name="multi_ping_tool", server_url="https://api.example.com/mcp")
    tool._tool_schema = {"properties": {}}
    tool._timeout_ping = 0.05  # Short interval so we can get 2 pings in ~0.1 s.

    ping_count = 0
    call_started = asyncio.Event()

    async def _slow_call_tool(*args, **kwargs):
        call_started.set()
        # Stay alive long enough for 2 ping intervals, then return.
        await asyncio.sleep(0.15)
        return make_mock_response("Slow but OK")

    main_client = make_mock_client()
    main_client.call_tool = AsyncMock(side_effect=_slow_call_tool)
    tool._client = main_client

    probe_client = AsyncMock(spec=Client)
    probe_client.__aenter__ = AsyncMock(return_value=probe_client)
    probe_client.__aexit__ = AsyncMock(return_value=None)

    async def _count_pings():
        nonlocal ping_count
        ping_count += 1

    probe_client.ping = AsyncMock(side_effect=_count_pings)
    tool._build_client = AsyncMock(return_value=probe_client)

    result = await _drain(tool.execute())

    assert result.success is True
    assert ping_count >= 2


@pytest.mark.asyncio
async def test_mcp_tool_call_tool_raises_propagates_as_failed_result(make_mock_client):
    """If call_tool raises (e.g. network error), execute() should catch it and
    return a failed ToolResult — not propagate the raw exception."""
    tool = MCPTool(tool_name="broken_tool", server_url="https://api.example.com/mcp")
    tool._tool_schema = {"properties": {}}

    probe_client = AsyncMock(spec=Client)
    probe_client.__aenter__ = AsyncMock(return_value=probe_client)
    probe_client.__aexit__ = AsyncMock(return_value=None)
    probe_client.ping = AsyncMock(return_value=None)

    mock_client = make_mock_client(side_effect=RuntimeError("network timeout"))
    tool._client = mock_client
    tool._build_client = AsyncMock(return_value=probe_client)

    result = await _drain(tool.execute())

    assert isinstance(result, ToolResult)
    assert result.success is False
    assert "network timeout" in (result.error or "")


@pytest.mark.asyncio
async def test_mcp_tool_progress_notifications_yielded_before_result(
    make_mock_response, make_mock_client
):
    """Progress notifications sent during the tool call should be yielded as
    StreamableToolResult items before the final ToolResult."""
    tool = MCPTool(tool_name="progress_tool", server_url="https://api.example.com/mcp")
    tool._tool_schema = {"properties": {}}
    tool._timeout_ping = 60.0  # Don't let healthcheck interfere.

    async def _call_with_progress(*args, **kwargs):
        handler = kwargs.get("progress_handler")
        if handler:
            await handler(0.5, 1.0, "halfway")
            await handler(1.0, 1.0, "done")
        return make_mock_response("Final result")

    main_client = make_mock_client()
    main_client.call_tool = AsyncMock(side_effect=_call_with_progress)
    tool._client = main_client

    probe_client = AsyncMock(spec=Client)
    probe_client.__aenter__ = AsyncMock(return_value=probe_client)
    probe_client.__aexit__ = AsyncMock(return_value=None)
    probe_client.ping = AsyncMock(return_value=None)
    tool._build_client = AsyncMock(return_value=probe_client)

    items = []
    async for item in tool.execute():
        items.append(item)

    updateable = [i for i in items if isinstance(i, UpdateToolResult)]
    final = [i for i in items if isinstance(i, ToolResult)]

    assert len(final) == 1
    assert final[0].success is True
    assert final[0].data == "Final result"
    assert len(updateable) >= 2
    assert updateable[0].data["progress"] == 0.5
    assert updateable[1].data["progress"] == 1.0


def _probe_client(ping: AsyncMock) -> AsyncMock:
    probe = AsyncMock(spec=Client)
    probe.__aenter__ = AsyncMock(return_value=probe)
    probe.__aexit__ = AsyncMock(return_value=None)
    probe.ping = ping
    return probe


@pytest.mark.asyncio
async def test_mcp_tool_tool_error_after_progress_returns_failure_when_resumed_from_new_tasks(
    make_mock_client,
):
    """A ToolError raised after a progress update yields a failed ToolResult when
    each step of the generator runs in a new task, as the runner drives it."""
    tool = MCPTool(tool_name="progress_then_error_tool", server_url="https://api.example.com/mcp")
    tool._tool_schema = {"properties": {}}
    tool._timeout_ping = 0.05

    async def _progress_then_error(*args, **kwargs):
        await kwargs["progress_handler"](0.5, 1.0, "halfway")
        await asyncio.sleep(0.2)
        raise ToolError("tool failed")

    main_client = make_mock_client()
    main_client.call_tool = AsyncMock(side_effect=_progress_then_error)
    tool._client = main_client
    tool._build_client = AsyncMock(return_value=_probe_client(AsyncMock(return_value=None)))

    items = await _drain_from_new_tasks(tool.execute())

    assert len(items) == 2
    assert isinstance(items[0], UpdateToolResult)
    assert isinstance(items[1], ToolResult)
    assert items[1].success is False
    assert "tool failed" in (items[1].error or "")


@pytest.mark.asyncio
async def test_mcp_tool_ping_failure_after_progress_returns_failure_when_resumed_from_new_tasks(
    make_mock_client,
):
    """A health check that fails after a progress update, while the call stalls, yields a
    failed ToolResult when each step of the generator runs in a new task."""
    tool = MCPTool(tool_name="progress_then_stall_tool", server_url="https://api.example.com/mcp")
    tool._tool_schema = {"properties": {}}
    tool._timeout_ping = 0.05

    async def _progress_then_stall(*args, **kwargs):
        await kwargs["progress_handler"](0.5, 1.0, "halfway")
        await asyncio.sleep(30)

    main_client = make_mock_client()
    main_client.call_tool = AsyncMock(side_effect=_progress_then_stall)
    tool._client = main_client
    tool._build_client = AsyncMock(
        return_value=_probe_client(AsyncMock(side_effect=ConnectionError("server gone")))
    )

    items = await _drain_from_new_tasks(tool.execute())

    assert len(items) == 2
    assert isinstance(items[0], UpdateToolResult)
    assert isinstance(items[1], ToolResult)
    assert items[1].success is False
    assert "stopped responding" in (items[1].error or "")


@pytest.mark.asyncio
@pytest.mark.parametrize("stop", ["cancel_step", "close_generator"])
async def test_mcp_tool_stopped_consumer_cancels_the_remote_call_and_health_check(
    make_mock_client, stop
):
    """A consumer that cancels the step or closes the generator mid-call cancels the
    remote call and closes the health check connection."""
    tool = MCPTool(tool_name="progress_then_stall_tool", server_url="https://api.example.com/mcp")
    tool._tool_schema = {"properties": {}}
    tool._timeout_ping = 0.05
    call_cancelled = asyncio.Event()

    async def _progress_then_stall(*args, **kwargs):
        await kwargs["progress_handler"](0.5, 1.0, "halfway")
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            call_cancelled.set()
            raise

    main_client = make_mock_client()
    main_client.call_tool = AsyncMock(side_effect=_progress_then_stall)
    tool._client = main_client
    probe = _probe_client(AsyncMock(return_value=None))
    tool._build_client = AsyncMock(return_value=probe)

    agen = tool.execute()
    await asyncio.ensure_future(agen.__anext__())
    if stop == "cancel_step":
        step = asyncio.ensure_future(agen.__anext__())
        await asyncio.sleep(0.2)
        step.cancel()
        outcome = await asyncio.gather(step, return_exceptions=True)
        assert isinstance(outcome[0], asyncio.CancelledError)
    else:
        await asyncio.sleep(0.2)
        await agen.aclose()

    await asyncio.wait_for(call_cancelled.wait(), timeout=1)
    probe.__aexit__.assert_awaited_once()


@pytest.mark.asyncio
async def test_mcp_tool_health_check_stops_when_its_cancellation_is_lost(make_mock_client):
    """The health check stops once the call is over, even when its own cancellation
    never reaches it.

    Python 3.11's asyncio.wait_for drops a cancellation that arrives as the awaited
    ping completes; the loop then pinged forever and the cancelled step never ended.
    Here the ping swallows the cancellation itself, which loses it on every version.
    """
    tool = MCPTool(tool_name="progress_then_stall_tool", server_url="https://api.example.com/mcp")
    tool._tool_schema = {"properties": {}}
    tool._timeout_ping = 0
    pinging = asyncio.Event()
    lost: list[bool] = []

    async def _progress_then_stall(*args, **kwargs):
        await kwargs["progress_handler"](0.5, 1.0, "halfway")
        await asyncio.sleep(30)

    async def _ping_that_loses_the_cancellation():
        if lost:
            # A loop that pings again ends here rather than hanging the test; the
            # assertion on the number of pings is what catches it.
            raise ConnectionError("pinged again")
        pinging.set()
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            lost.append(True)
            return None

    main_client = make_mock_client()
    main_client.call_tool = AsyncMock(side_effect=_progress_then_stall)
    tool._client = main_client
    ping = AsyncMock(side_effect=_ping_that_loses_the_cancellation)
    tool._build_client = AsyncMock(return_value=_probe_client(ping))

    agen = tool.execute()
    await asyncio.ensure_future(agen.__anext__())
    step = asyncio.ensure_future(agen.__anext__())
    await asyncio.wait_for(pinging.wait(), timeout=1)
    step.cancel()

    outcome = await asyncio.wait_for(asyncio.gather(step, return_exceptions=True), timeout=2)
    assert lost, "the ping should have swallowed the cancellation"
    assert ping.await_count == 1, "the health check went on pinging after the call ended"
    assert isinstance(outcome[0], asyncio.CancelledError)


class TestTheConfiguredTimeoutIsApplied:
    """It was declared and never passed on, so a tool configured with a timeout
    behaved like one without — and the same server answered differently
    depending on whether it was declared under `mcp_servers` or under `tools`.
    """

    @pytest.mark.asyncio
    async def test_it_reaches_the_client(self):
        tool = MCPTool(tool_name="t", server_url="https://api.example.com/mcp", timeout=45)

        client = await tool._build_client()

        assert client._session_kwargs["read_timeout_seconds"] == 45

    @pytest.mark.asyncio
    async def test_zero_means_no_limit_rather_than_no_time(self):
        """A zero passed through would be a tool that times out instantly."""
        tool = MCPTool(tool_name="t", server_url="https://api.example.com/mcp")

        client = await tool._build_client()

        assert client._session_kwargs["read_timeout_seconds"] is None


class TestProtocolMode:
    """Legacy handshake by default: in the modern era, servers that enforce
    `Mcp-Param-*` headers (GitHub's) reject every call that has an annotated
    parameter, since this client does not mirror them."""

    @pytest.mark.asyncio
    async def test_a_tool_connects_with_the_legacy_handshake_by_default(self):
        tool = MCPTool(tool_name="t", server_url="https://api.example.com/mcp")
        with patch("domyn.agents.tools.mcp.mcp_tool.Client") as client_cls:
            await tool._build_client()
        assert client_cls.call_args.kwargs["mode"] == "legacy"

    @pytest.mark.asyncio
    async def test_a_tool_can_ask_for_auto(self):
        tool = MCPTool(
            tool_name="t", server_url="https://api.example.com/mcp", protocol_mode="auto"
        )
        with patch("domyn.agents.tools.mcp.mcp_tool.Client") as client_cls:
            await tool._build_client()
        assert client_cls.call_args.kwargs["mode"] == "auto"

    @pytest.mark.parametrize("headers", [None, {"Authorization": "x"}])
    @pytest.mark.asyncio
    async def test_a_server_config_passes_it_on_with_and_without_headers(self, headers):
        from domyn.agents.tools.mcp import MCPServerStreamableHTTPConfig

        config = MCPServerStreamableHTTPConfig(
            name="s", url="https://x.example.com/mcp", headers=headers
        )
        assert (await config.create_client()).mode == "legacy"
        auto = config.model_copy(update={"protocol_mode": "auto"})
        assert (await auto.create_client()).mode == "auto"

    def test_from_config_reads_it(self):
        tool = MCPTool.from_config(
            {"server_url": "https://x.example.com/mcp", "name": "t", "protocol_mode": "auto"},
            save_in_registry=False,
        )
        assert tool.protocol_mode == "auto"
