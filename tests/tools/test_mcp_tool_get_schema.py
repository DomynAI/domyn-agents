"""Tests for MCPTool.get_tool_schema - napoleon docstring parsing & schema enrichment."""

from typing import Any
from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastmcp import Client
from mcp.types import Tool

from domyn.agents.tools import MCPTool

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mcp_tool_obj(
    tool_name: str,
    description: str | None,
    input_schema: dict[str, Any] | None,
    title: str | None = None,
) -> Mock:
    """Create a mock MCP tool as returned by client.list_tools()."""
    t = Mock(spec=Tool)
    t.name = tool_name
    t.description = description
    t.input_schema = input_schema
    t.title = title
    return t


def _mock_client_with_tools(*tools: Mock) -> AsyncMock:
    """Return an AsyncMock Client whose list_tools() returns the given tools."""
    client = AsyncMock(spec=Client)
    client.list_tools = AsyncMock(return_value=list(tools))
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=None)
    return client


async def _get_schema(tool_name: str, remote_tools: list[Mock]) -> dict[str, Any] | None:
    """Convenience: create an MCPTool, inject a mock client, call get_tool_schema."""
    mcp_tool = MCPTool(
        tool_name=tool_name,
        server_url="https://mcp.example.com",
    )
    mock_client = _mock_client_with_tools(*remote_tools)
    mcp_tool._client = mock_client
    await mcp_tool.get_tool_schema()
    return mcp_tool._tool_schema


# ---------------------------------------------------------------------------
# Basic behaviour
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_schema_loaded_from_matching_tool():
    """get_tool_schema populates _tool_schema from the matching remote tool."""
    remote = _make_mcp_tool_obj(
        "search",
        "Search the database.",
        {
            "properties": {"query": {"type": "string", "description": "Search query"}},
            "required": ["query"],
        },
    )

    schema = await _get_schema("search", [remote])

    assert schema is not None
    assert schema["description"] == "Search the database."
    assert schema["properties"]["query"]["type"] == "string"
    assert schema["required"] == ["query"]


@pytest.mark.asyncio
async def test_schema_is_none_when_tool_not_found():
    """_tool_schema stays None if the remote server has no matching tool."""
    other_tool = _make_mcp_tool_obj("other", "Not the droid.", {"properties": {}})

    schema = await _get_schema("missing_tool", [other_tool])

    assert schema is None


@pytest.mark.asyncio
async def test_schema_caches_after_first_call():
    """Calling get_tool_schema twice does NOT issue a second list_tools RPC."""
    remote = _make_mcp_tool_obj("cached", "A tool.", {"properties": {}})
    mock_client = _mock_client_with_tools(remote)

    mcp_tool = MCPTool(tool_name="cached", server_url="https://example.com")
    mcp_tool._client = mock_client

    await mcp_tool.get_tool_schema()
    await mcp_tool.get_tool_schema()

    mock_client.list_tools.assert_called_once()


# ---------------------------------------------------------------------------
# MCP spec `title` → metadata propagation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_server_title_populates_metadata():
    """A server-provided title lands in metadata["title"]."""
    remote = _make_mcp_tool_obj(
        "chat_with_documents_md",
        "Chat with documents.",
        {"properties": {}},
        title="Chat with documents (markdown)",
    )
    mcp_tool = MCPTool(tool_name="chat_with_documents_md", server_url="https://mcp.example.com")
    mcp_tool._client = _mock_client_with_tools(remote)

    await mcp_tool.get_tool_schema()

    assert mcp_tool.metadata["title"] == "Chat with documents (markdown)"


@pytest.mark.asyncio
async def test_missing_server_title_leaves_metadata_untouched():
    """No title from the server → no "title" key added to metadata."""
    remote = _make_mcp_tool_obj("list_documents", "List documents.", {"properties": {}})
    mcp_tool = MCPTool(tool_name="list_documents", server_url="https://mcp.example.com")
    mcp_tool._client = _mock_client_with_tools(remote)

    await mcp_tool.get_tool_schema()

    assert "title" not in mcp_tool.metadata


@pytest.mark.asyncio
async def test_config_title_takes_precedence_over_server_title():
    """An explicit title from config is not overwritten by the server one."""
    remote = _make_mcp_tool_obj("search", "Search.", {"properties": {}}, title="Server Title")
    mcp_tool = MCPTool(
        tool_name="search",
        server_url="https://mcp.example.com",
        metadata={"title": "Config Title"},
    )
    mcp_tool._client = _mock_client_with_tools(remote)

    await mcp_tool.get_tool_schema()

    assert mcp_tool.metadata["title"] == "Config Title"


# ---------------------------------------------------------------------------
# Napoleon docstring description extraction
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_napoleon_description_overrides_raw_description():
    """When the description is a napoleon docstring, the parsed summary is used."""
    napoleon_doc = (
        "Search a knowledge base for relevant documents.\n"
        "\n"
        "Args:\n"
        "    query (str): The search query string.\n"
        "    top_k (int): Number of results to return.\n"
    )
    remote = _make_mcp_tool_obj(
        "search",
        napoleon_doc,
        {
            "properties": {
                "query": {"type": "string"},
                "top_k": {"type": "integer"},
            },
        },
    )

    schema = await _get_schema("search", [remote])

    assert schema is not None
    # The napoleon parser should extract "Search a knowledge base for relevant documents."
    assert schema["description"] == "Search a knowledge base for relevant documents."
    assert schema["properties"]["query"]["type"] == "string"
    assert schema["properties"]["top_k"]["type"] == "integer"
    assert schema["properties"]["query"]["description"] == "The search query string."
    assert schema["properties"]["top_k"]["description"] == "Number of results to return."


@pytest.mark.asyncio
async def test_raw_description_used_when_not_napoleon():
    """A plain (non-napoleon) description is used as-is."""
    remote = _make_mcp_tool_obj(
        "ping",
        "Ping the server to check if it is alive.",
        {"properties": {}},
    )

    schema = await _get_schema("ping", [remote])

    assert schema is not None
    assert schema["description"] == "Ping the server to check if it is alive."
    assert schema["properties"] == {}


@pytest.mark.asyncio
async def test_none_description_used_when_description_is_none():
    """When description is None (not a string), napoleon is skipped."""
    remote = _make_mcp_tool_obj(
        "no_desc",
        None,
        {"properties": {"x": {"type": "integer"}}},
    )

    schema = await _get_schema("no_desc", [remote])

    assert schema is not None
    # Falls back to tool.description which is None
    assert schema["description"] is None
    assert schema["properties"]["x"]["description"] == "Parameter x"
    assert schema["properties"]["x"]["type"] == "integer"


# ---------------------------------------------------------------------------
# Napoleon arg descriptions → input_schema enrichment
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_napoleon_args_fill_missing_param_descriptions():
    """Parameters WITHOUT a description get it from the napoleon Args section."""
    napoleon_doc = (
        "Fetch user data.\n"
        "\n"
        "Args:\n"
        "    user_id (str): The unique user identifier.\n"
        "    include_deleted (bool): Whether to include soft-deleted records.\n"
    )
    remote = _make_mcp_tool_obj(
        "get_user",
        napoleon_doc,
        {
            "properties": {
                "user_id": {"type": "string"},  # no description
                "include_deleted": {"type": "boolean"},  # no description
            },
        },
    )

    schema = await _get_schema("get_user", [remote])

    assert schema["properties"]["user_id"]["description"] == "The unique user identifier."
    assert (
        schema["properties"]["include_deleted"]["description"]
        == "Whether to include soft-deleted records."
    )
    assert schema["properties"]["user_id"]["type"] == "string"
    assert schema["properties"]["include_deleted"]["type"] == "boolean"


@pytest.mark.asyncio
async def test_existing_param_description_not_overwritten():
    """Parameters that ALREADY have a description are left untouched."""
    napoleon_doc = "Do something.\n\nArgs:\n    name (str): Name from docstring.\n"
    remote = _make_mcp_tool_obj(
        "greet",
        napoleon_doc,
        {
            "properties": {
                "name": {"type": "string", "description": "Original schema description"},
            },
        },
    )

    schema = await _get_schema("greet", [remote])

    assert schema["properties"]["name"]["description"] == "Original schema description"


@pytest.mark.asyncio
async def test_param_without_napoleon_match_gets_fallback_description():
    """A param with no schema description AND no napoleon entry gets a generic fallback."""
    napoleon_doc = "List items.\n\nArgs:\n    category (str): Item category.\n"
    remote = _make_mcp_tool_obj(
        "list_items",
        napoleon_doc,
        {
            "properties": {
                "category": {"type": "string"},  # has napoleon match
                "page": {"type": "integer"},  # no napoleon match, no description
            },
        },
    )

    schema = await _get_schema("list_items", [remote])

    assert schema["properties"]["category"]["description"] == "Item category."
    assert schema["properties"]["page"]["description"] == "Parameter page"


@pytest.mark.asyncio
async def test_mixed_params_some_with_descriptions():
    """Only missing descriptions are filled; existing ones are preserved."""
    napoleon_doc = (
        "Send a message.\n"
        "\n"
        "Args:\n"
        "    recipient (str): The message recipient.\n"
        "    body (str): Message body text.\n"
        "    priority (int): Message priority level.\n"
    )
    remote = _make_mcp_tool_obj(
        "send_message",
        napoleon_doc,
        {
            "properties": {
                "recipient": {"type": "string"},  # filled from napoleon
                "body": {"type": "string", "description": "The full message body"},  # kept as-is
                "priority": {"type": "integer"},  # filled from napoleon
            },
        },
    )

    schema = await _get_schema("send_message", [remote])

    assert schema["properties"]["recipient"]["description"] == "The message recipient."
    assert schema["properties"]["body"]["description"] == "The full message body"
    assert schema["properties"]["priority"]["description"] == "Message priority level."


# ---------------------------------------------------------------------------
# Realistic multi-tool server scenarios
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_selects_correct_tool_from_multiple():
    """When the server lists several tools, only the matching one is used."""
    search_tool = _make_mcp_tool_obj(
        "search",
        "Search documents.\n\nArgs:\n    query (str): Search text.\n",
        {"properties": {"query": {"type": "string"}}},
    )
    delete_tool = _make_mcp_tool_obj(
        "delete",
        "Delete a document.\n\nArgs:\n    doc_id (str): Document ID.\n",
        {"properties": {"doc_id": {"type": "string"}}},
    )

    schema = await _get_schema("delete", [search_tool, delete_tool])

    assert schema is not None
    assert schema["description"] == "Delete a document."
    assert "doc_id" in schema["properties"]
    assert "query" not in schema["properties"]


@pytest.mark.asyncio
async def test_realistic_rag_tool_docstring():
    """End-to-end test with a typical RAG tool napoleon docstring."""
    docstring = (
        "Retrieve relevant documents from the knowledge base using semantic search.\n"
        "\n"
        "Args:\n"
        "    query (str): The natural language search query.\n"
        "    top_k (int): Maximum number of documents to return.\n"
        "    filter_metadata (dict): Optional metadata filters to narrow results.\n"
        "    include_scores (bool): Whether to include similarity scores.\n"
        "\n"
        "Returns:\n"
        "    list: A list of matching documents with metadata.\n"
    )
    remote = _make_mcp_tool_obj(
        "semantic_search",
        docstring,
        {
            "properties": {
                "query": {"type": "string"},
                "top_k": {"type": "integer"},
                "filter_metadata": {"type": "object"},
                "include_scores": {"type": "boolean"},
            },
            "required": ["query"],
        },
    )

    schema = await _get_schema("semantic_search", [remote])

    assert schema is not None
    assert (
        schema["description"]
        == "Retrieve relevant documents from the knowledge base using semantic search."
    )
    assert schema["properties"]["query"]["description"] == "The natural language search query."
    assert schema["properties"]["top_k"]["description"] == "Maximum number of documents to return."
    assert (
        schema["properties"]["filter_metadata"]["description"]
        == "Optional metadata filters to narrow results."
    )
    assert (
        schema["properties"]["include_scores"]["description"]
        == "Whether to include similarity scores."
    )
    assert schema["required"] == ["query"]


@pytest.mark.asyncio
async def test_realistic_api_tool_no_docstring_args():
    """Tool with a napoleon description but no Args section - no param enrichment."""
    docstring = (
        "Get the current server health status.\n\nReturns:\n    dict: Health check response.\n"
    )
    remote = _make_mcp_tool_obj(
        "health_check",
        docstring,
        {
            "properties": {
                "verbose": {"type": "boolean"},
            },
        },
    )

    schema = await _get_schema("health_check", [remote])

    assert schema is not None
    assert schema["description"] == "Get the current server health status."
    # No napoleon args → param keeps no description (no fallback added when napoleon has no args)
    assert schema["properties"]["verbose"]["description"] == "Parameter verbose"
    assert schema["properties"]["verbose"]["type"] == "boolean"


@pytest.mark.asyncio
async def test_input_schema_is_none_uses_empty_dict():
    """When input_schema is None, an empty dict is used as fallback."""
    remote = _make_mcp_tool_obj("bare_tool", "A bare tool.", None)

    schema = await _get_schema("bare_tool", [remote])

    assert schema is not None
    assert schema.get("properties", {}) == {}
    assert schema["description"] == "A bare tool."


@pytest.mark.asyncio
async def test_parameters_style_docstring():
    """Napoleon Parameters: section (alias for Args:) is also supported."""
    docstring = (
        "Create a new project.\n"
        "\n"
        "Parameters:\n"
        "    name (str): The project name.\n"
        "    template (str): Template to use.\n"
    )
    remote = _make_mcp_tool_obj(
        "create_project",
        docstring,
        {
            "properties": {
                "name": {"type": "string"},
                "template": {"type": "string"},
            },
        },
    )

    schema = await _get_schema("create_project", [remote])

    assert schema is not None
    assert schema["description"] == "Create a new project."
    assert schema["properties"]["name"]["description"] == "The project name."
    assert schema["properties"]["template"]["description"] == "Template to use."


@pytest.mark.asyncio
async def test_empty_description_string():
    """An empty string description is not napoleon - used as-is."""
    remote = _make_mcp_tool_obj("empty", "", {"properties": {}})

    schema = await _get_schema("empty", [remote])

    assert schema is not None
    # _parse_napoleon_docstring returns None for empty → description falls back to tool.description
    assert schema["description"] == ""


@pytest.mark.asyncio
async def test_multiline_param_description():
    """Napoleon args with multi-line descriptions are collapsed to single line."""
    docstring = (
        "Analyze text sentiment.\n"
        "\n"
        "Args:\n"
        "    text (str): The input text to analyze. Can be a paragraph\n"
        "        or a full document.\n"
        "    language (str): ISO 639-1 language code.\n"
    )
    remote = _make_mcp_tool_obj(
        "analyze_sentiment",
        docstring,
        {
            "properties": {
                "text": {"type": "string"},
                "language": {"type": "string"},
            },
        },
    )

    schema = await _get_schema("analyze_sentiment", [remote])

    assert schema is not None
    # The napoleon parser collapses multi-line descs
    assert "input text to analyze" in schema["properties"]["text"]["description"]
    assert schema["properties"]["language"]["description"] == "ISO 639-1 language code."


# ---------------------------------------------------------------------------
# Testinvocation_json_schema
# ---------------------------------------------------------------------------


async def _mcp_tool_with_schema(
    tool_name: str,
    description: str | None,
    input_schema: dict[str, Any] | None,
) -> MCPTool:
    """Create an MCPTool whose _tool_schema is populated via get_tool_schema()."""
    remote = _make_mcp_tool_obj(tool_name, description, input_schema)
    mock_client = _mock_client_with_tools(remote)
    tool = MCPTool(tool_name=tool_name, server_url="https://mcp.example.com")
    tool._client = mock_client
    await tool.get_tool_schema()
    return tool


def test_fallback_schema_when_not_loaded():
    """Accessing invocation_json_schema before initialize() returns a fallback schema."""
    tool = MCPTool(tool_name="unloaded", server_url="https://mcp.example.com")
    schema = tool.invocation_json_schema
    assert schema["properties"]["name"] == {"const": "unloaded", "type": "string"}
    assert schema["properties"]["arguments"] == {}
    assert schema["required"] == ["arguments"]
    assert schema["description"] == "unloaded"


@pytest.mark.asyncio
async def test_top_level_keys():
    """Schema has 'properties', 'required', and 'description' at the top level."""
    tool = await _mcp_tool_with_schema(
        "my_tool",
        "My tool description",
        {
            "properties": {"x": {"type": "string", "description": "A param"}},
            "required": ["x"],
        },
    )
    schema = tool.invocation_json_schema

    assert set(schema.keys()) == {"properties", "required", "description"}


@pytest.mark.asyncio
async def test_name_property_matches_function_tool_format():
    """The 'name' property uses {"const": <tool_name>, "type": "string"}."""
    tool = await _mcp_tool_with_schema("search", "Search.", {"properties": {}})
    schema = tool.invocation_json_schema

    assert schema["properties"]["name"] == {"const": "search", "type": "string"}
    assert schema["description"] == "Search."


@pytest.mark.asyncio
async def test_arguments_property_contains_properties_and_required():
    """The 'arguments' property has nested 'properties' and 'required' keys."""
    tool = await _mcp_tool_with_schema(
        "calc",
        "Calculator",
        {
            "properties": {
                "a": {"type": "integer", "description": "First operand"},
                "b": {"type": "integer", "description": "Second operand"},
            },
            "required": ["a", "b"],
        },
    )
    schema = tool.invocation_json_schema

    args = schema["properties"]["arguments"]
    assert "properties" in args
    assert "required" in args
    assert "description" not in args
    assert args["required"] == ["a", "b"]
    assert set(args["properties"].keys()) == {"a", "b"}


@pytest.mark.asyncio
async def test_description_from_schema():
    """Top-level 'description' reflects the tool description."""
    tool = await _mcp_tool_with_schema("t", "My desc", {"properties": {}})
    assert tool.invocation_json_schema["description"] == "My desc"


@pytest.mark.asyncio
async def test_description_defaults_to_empty_string():
    """When description is None, it falls through as None from get_tool_schema."""
    tool = await _mcp_tool_with_schema("t", None, {"properties": {}})
    # None description → napoleon is None → napoleon_desc is None → falls back to tool.description (None)
    assert tool.invocation_json_schema["description"] is None


# ---------------------------------------------------------------------------
# MCPTool.invocation_json_schema structural parity with FunctionTool.invocation_json_schema
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_simple_string_param():
    """Single string param produces the same shape as FunctionTool."""
    from domyn.agents.tools import FunctionTool

    def greet(name: str) -> str:
        """Greet a user.

        Args:
            name (str): The user name.

        Returns:
            str: Greeting.
        """
        return f"Hello, {name}"

    ft = FunctionTool(func=greet)
    ft_schema = ft.invocation_json_schema

    mcp_tool = await _mcp_tool_with_schema(
        "greet",
        "Greet a user.\n\nArgs:\n    name (str): The user name.\n\nReturns:\n    str: Greeting.",
        {
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        },
    )
    mcp_schema = mcp_tool.invocation_json_schema

    # Top-level structure
    assert set(ft_schema.keys()) == set(mcp_schema.keys())
    assert ft_schema["required"] == mcp_schema["required"]
    assert ft_schema["description"] == mcp_schema["description"]

    # Name constant
    assert ft_schema["properties"]["name"] == mcp_schema["properties"]["name"]

    # Arguments envelope
    ft_args = ft_schema["properties"]["arguments"]
    mcp_args = mcp_schema["properties"]["arguments"]
    assert set(ft_args.keys()) == set(mcp_args.keys())
    assert set(ft_args.keys()).issuperset({"properties", "required"})
    assert ft_args["required"] == mcp_args["required"]

    # Param schemas
    assert ft_args["properties"]["name"] == mcp_args["properties"]["name"]


@pytest.mark.asyncio
async def test_multiple_typed_params():
    """Multiple params with int/bool/string types match FunctionTool output."""
    from domyn.agents.tools import FunctionTool

    def search(query: str, limit: int, verbose: bool) -> str:
        """Search items.

        Args:
            query (str): Search query.
            limit (int): Max results.
            verbose (bool): Verbose output.

        Returns:
            str: Results.
        """
        return ""

    ft = FunctionTool(func=search)
    ft_schema = ft.invocation_json_schema

    docstring = (
        "Search items.\n\n"
        "Args:\n"
        "    query (str): Search query.\n"
        "    limit (int): Max results.\n"
        "    verbose (bool): Verbose output.\n\n"
        "Returns:\n"
        "    str: Results."
    )
    mcp_tool = await _mcp_tool_with_schema(
        "search",
        docstring,
        {
            "properties": {
                "query": {"type": "string"},
                "limit": {"type": "integer"},
                "verbose": {"type": "boolean"},
            },
            "required": ["query", "limit", "verbose"],
        },
    )
    mcp_schema = mcp_tool.invocation_json_schema

    ft_args = ft_schema["properties"]["arguments"]
    mcp_args = mcp_schema["properties"]["arguments"]

    # Same param names
    assert set(ft_args["properties"].keys()) == set(mcp_args["properties"].keys())

    # Each param schema matches
    for param_name in ["query", "limit", "verbose"]:
        assert ft_args["properties"][param_name] == mcp_args["properties"][param_name], (
            f"Mismatch on param '{param_name}': "
            f"FunctionTool={ft_args['properties'][param_name]} vs "
            f"MCPTool={mcp_args['properties'][param_name]}"
        )
    assert ft_args["required"] == mcp_args["required"] == ["query", "limit", "verbose"]
    assert ft_schema["description"] == mcp_schema["description"] == "Search items."


@pytest.mark.asyncio
async def test_no_params_tool():
    """A tool with zero params produces matching schemas."""
    from domyn.agents.tools import FunctionTool

    def ping() -> str:
        """Ping the server."""
        return "pong"

    ft = FunctionTool(func=ping)
    ft_schema = ft.invocation_json_schema

    mcp_tool = await _mcp_tool_with_schema("ping", "Ping the server.", {"properties": {}})
    mcp_schema = mcp_tool.invocation_json_schema

    assert ft_schema["properties"]["name"] == mcp_schema["properties"]["name"]
    assert ft_schema["required"] == mcp_schema["required"]
    assert ft_schema["properties"]["arguments"]["properties"] == {}
    assert mcp_schema["properties"]["arguments"]["properties"] == {}


@pytest.mark.asyncio
async def test_array_param():
    """Array-typed param in MCP uses the same JSON Schema as FunctionTool."""
    from domyn.agents.tools import FunctionTool

    def tag(items: list[str]) -> str:
        """Tag items.

        Args:
            items (List[str]): List of items.

        Returns:
            str: Result.
        """
        return ""

    ft = FunctionTool(func=tag)
    ft_args = ft.invocation_json_schema["properties"]["arguments"]

    mcp_tool = await _mcp_tool_with_schema(
        "tag",
        "Tag items.\n\nArgs:\n    items (List[str]): List of items.\n\nReturns:\n    str: Result.",
        {
            "properties": {
                "items": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
            "required": ["items"],
        },
    )
    mcp_args = mcp_tool.invocation_json_schema["properties"]["arguments"]

    assert ft_args["properties"] == mcp_args["properties"]


@pytest.mark.asyncio
async def test_object_param():
    """Dict-typed param in MCP uses the same JSON Schema as FunctionTool."""
    from domyn.agents.tools import FunctionTool

    def configure(options: dict) -> str:
        """Configure settings.

        Args:
            options (dict): Settings dict.

        Returns:
            str: Status.
        """
        return ""

    ft = FunctionTool(func=configure)
    ft_args = ft.invocation_json_schema["properties"]["arguments"]

    mcp_tool = await _mcp_tool_with_schema(
        "configure",
        "Configure settings.\n\nArgs:\n    options (dict): Settings dict.\n\nReturns:\n    str: Status.",
        {
            "properties": {
                "options": {
                    "type": "object",
                    "additionalProperties": {},
                },
            },
            "required": ["options"],
        },
    )
    mcp_args = mcp_tool.invocation_json_schema["properties"]["arguments"]

    assert ft_args == mcp_args


@pytest.mark.asyncio
async def test_napoleon_enriched_schema_matches_function_tool():
    """Full round-trip: get_tool_schema with napoleon → invocation_json_schema matches FunctionTool."""
    from domyn.agents.tools import FunctionTool

    def calculate(a: int, b: int) -> int:
        """Add two numbers.

        Args:
            a (int): First number.
            b (int): Second number.

        Returns:
            int: The sum.
        """
        return a + b

    ft = FunctionTool(func=calculate)
    ft_schema = ft.invocation_json_schema

    # Use the same napoleon docstring so get_tool_schema enriches the schema
    docstring = (
        "Add two numbers.\n\n"
        "Args:\n"
        "    a (int): First number.\n"
        "    b (int): Second number.\n\n"
        "Returns:\n"
        "    int: The sum."
    )
    mcp_tool = await _mcp_tool_with_schema(
        "calculate",
        docstring,
        {
            "properties": {
                "a": {"type": "integer"},
                "b": {"type": "integer"},
            },
            "required": ["a", "b"],
        },
    )
    mcp_schema = mcp_tool.invocation_json_schema

    assert ft_schema == mcp_schema


# ---------------------------------------------------------------------------
# Nested schemas reach the model intact
# ---------------------------------------------------------------------------

# Gmail's create_draft: `attachments` is an array of $ref objects, once shown to the
# model as `{"type": "array", "items": {}}`.
GMAIL_CREATE_DRAFT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "to": {"type": "array", "items": {"type": "string"}},
        "subject": {"type": "string"},
        "attachments": {
            "type": "array",
            "items": {"$ref": "#/$defs/Attachment"},
            "description": "Files to attach to the draft",
        },
    },
    "required": ["to"],
    "$defs": {
        "Attachment": {
            "type": "object",
            "properties": {
                "content": {"type": "string", "format": "byte"},
                "filename": {"type": "string"},
                "id": {"type": "string"},
                "inline": {"type": "boolean"},
                "mimeType": {"type": "string"},
            },
            "required": ["content"],
        }
    },
}


def _walk(node: Any):
    """Yield every dict in a schema, at any depth."""
    if isinstance(node, dict):
        yield node
        for value in node.values():
            yield from _walk(value)
    elif isinstance(node, list):
        for item in node:
            yield from _walk(item)


def _keys_anywhere(schema: Any) -> set[str]:
    """Every key at any depth (a substring check would match `#/$defs/`)."""
    return {key for node in _walk(schema) for key in node}


async def _gmail_tool() -> MCPTool:
    return await _mcp_tool_with_schema("create_draft", "Create a draft.", GMAIL_CREATE_DRAFT_SCHEMA)


@pytest.mark.asyncio
async def test_definitions_are_inlined_into_the_stored_schema():
    tool = await _gmail_tool()

    assert "$ref" not in _keys_anywhere(tool._tool_schema)
    assert "$defs" not in _keys_anywhere(tool._tool_schema)


@pytest.mark.asyncio
async def test_the_servers_schema_is_not_modified():
    """`list_tools` results are cached and shared across MCPTool instances."""
    await _gmail_tool()

    assert "$defs" in GMAIL_CREATE_DRAFT_SCHEMA
    assert GMAIL_CREATE_DRAFT_SCHEMA["properties"]["subject"] == {"type": "string"}


@pytest.mark.asyncio
async def test_tool_definition_keeps_the_item_structure():
    """The regression: the model must see what an attachment is made of."""
    tool = await _gmail_tool()

    params = tool.tool_definition["function"]["parameters"]
    items = params["properties"]["attachments"]["items"]

    assert items["properties"]["content"] == {"type": "string", "format": "byte"}
    assert "content" in items["required"]
    assert set(items["properties"]) == {"content", "filename", "id", "inline", "mimeType"}


@pytest.mark.asyncio
async def test_tool_definition_keeps_a_plain_arrays_item_type():
    """Not only $ref: `list[str]` used to flatten to `items: {}` as well."""
    tool = await _gmail_tool()

    params = tool.tool_definition["function"]["parameters"]

    assert params["properties"]["to"]["items"] == {"type": "string"}


@pytest.mark.asyncio
async def test_tool_definition_has_the_shape_the_planner_expects():
    """`get_tool_definitions` injects `reasoning` into these three keys."""
    tool = await _gmail_tool()

    params = tool.tool_definition["function"]["parameters"]

    assert params["type"] == "object"
    assert params["required"] == ["to"]
    assert isinstance(params["properties"], dict)


@pytest.mark.asyncio
async def test_tool_definition_defaults_the_missing_keys():
    """Some servers omit the top-level type; Anthropic requires it."""
    tool = await _mcp_tool_with_schema("bare", "A tool.", {})

    params = tool.tool_definition["function"]["parameters"]

    assert params == {"type": "object", "properties": {}, "required": []}


@pytest.mark.asyncio
async def test_the_tool_description_does_not_travel_inside_the_parameters():
    """It describes the tool, so it belongs in the function's own description."""
    tool = await _gmail_tool()

    definition = tool.tool_definition

    assert definition["function"]["description"] == "Create a draft."
    assert "description" not in definition["function"]["parameters"]


@pytest.mark.asyncio
async def test_the_dialect_declaration_is_stripped():
    """`$schema` means nothing once re-rooted under function.parameters."""
    tool = await _mcp_tool_with_schema(
        "t", "A tool.", {"$schema": "https://json-schema.org/draft/2020-12/schema"}
    )

    assert "$schema" not in tool.tool_definition["function"]["parameters"]


@pytest.mark.asyncio
async def test_tool_definition_does_not_alias_the_stored_schema():
    tool = await _gmail_tool()

    tool.tool_definition["function"]["parameters"]["properties"]["subject"]["description"] = "x"

    assert tool._tool_schema["properties"]["subject"]["description"] == "Parameter subject"


def test_tool_definition_before_discovery_falls_back_to_the_base():
    """Unchanged behaviour for a tool whose schema was never fetched."""
    tool = MCPTool(tool_name="unloaded", server_url="https://mcp.example.com")

    definition = tool.tool_definition

    assert definition["function"]["name"] == "unloaded"
    assert definition["function"]["parameters"] == {
        "type": "object",
        "properties": {},
        "required": [],
    }


@pytest.mark.asyncio
async def test_the_react_path_carries_the_nested_structure_too():
    """Re-rooted under `arguments`, a #/$defs pointer could never resolve there."""
    tool = await _gmail_tool()

    arguments = tool.invocation_json_schema["properties"]["arguments"]
    items = arguments["properties"]["attachments"]["items"]

    assert "$ref" not in _keys_anywhere(arguments)
    assert items["required"] == ["content"]


@pytest.mark.asyncio
async def test_an_unresolvable_reference_is_dropped_and_logged():
    logger = Mock()
    with patch("domyn.agents.tools.mcp.mcp_tool.get_logger", return_value=logger):
        tool = await _mcp_tool_with_schema(
            "broken",
            "A tool.",
            {"properties": {"a": {"$ref": "#/$defs/Gone", "description": "kept"}}},
        )

    assert "$ref" not in _keys_anywhere(tool._tool_schema)
    assert tool._tool_schema["properties"]["a"] == {"description": "kept"}
    warning = logger.warning.call_args[0][0]
    assert "#/$defs/Gone" in warning
    assert "broken" in warning
    assert "https://mcp.example.com" in warning


# A FastMCP tool taking a pydantic model: a bare $ref parameter, described by `Args:`
# and, as a type, by the model's own docstring.
RUN_JOB_DOC = "Run a job.\n\nArgs:\n    config: The settings this job runs with.\n"
RUN_JOB_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"config": {"$ref": "#/$defs/Config"}},
    "$defs": {
        "Config": {
            "type": "object",
            "description": "Generic configuration model.",
            "properties": {"retries": {"type": "integer"}},
        }
    },
}


@pytest.mark.asyncio
async def test_the_docstring_describes_a_ref_parameter_not_its_definition():
    tool = await _mcp_tool_with_schema("run_job", RUN_JOB_DOC, RUN_JOB_SCHEMA)

    config = tool._tool_schema["properties"]["config"]

    assert config["description"] == "The settings this job runs with."
    assert config["properties"] == {"retries": {"type": "integer"}}


@pytest.mark.asyncio
async def test_without_a_docstring_a_ref_parameter_keeps_the_definitions_description():
    """Better than the generic "Parameter config" it would otherwise get."""
    tool = await _mcp_tool_with_schema("run_job", "Run a job.", RUN_JOB_SCHEMA)

    assert tool._tool_schema["properties"]["config"]["description"] == (
        "Generic configuration model."
    )


@pytest.mark.asyncio
async def test_a_ref_parameters_own_description_beats_the_docstring():
    schema = {
        **RUN_JOB_SCHEMA,
        "properties": {"config": {"$ref": "#/$defs/Config", "description": "From the server."}},
    }

    tool = await _mcp_tool_with_schema("run_job", RUN_JOB_DOC, schema)

    assert tool._tool_schema["properties"]["config"]["description"] == "From the server."


@pytest.mark.asyncio
async def test_tool_definition_gives_an_array_its_items():
    """Chat Completions validators reject an array without them."""
    tool = await _mcp_tool_with_schema(
        "tag", "Tag things.", {"type": "object", "properties": {"tags": {"type": "array"}}}
    )

    params = tool.tool_definition["function"]["parameters"]

    assert params["properties"]["tags"]["items"] == {}


@pytest.mark.asyncio
async def test_tool_definition_drops_a_top_level_combinator():
    """Anthropic rejects allOf/anyOf/oneOf at the top of input_schema."""
    tool = await _mcp_tool_with_schema(
        "lookup",
        "Look someone up.",
        {
            "type": "object",
            "properties": {"email": {"type": "string"}, "phone": {"type": "string"}},
            "anyOf": [{"required": ["email"]}, {"required": ["phone"]}],
        },
    )

    params = tool.tool_definition["function"]["parameters"]

    assert "anyOf" not in params
    assert set(params["properties"]) == {"email", "phone"}


@pytest.mark.asyncio
async def test_a_schema_too_large_to_inline_is_capped_and_logged():
    names = [f"D{i}" for i in range(8)]
    schema = {
        "type": "object",
        "properties": {"root": {"$ref": "#/$defs/D0"}},
        "$defs": {
            name: {
                "type": "object",
                "properties": {f"to_{other}": {"$ref": f"#/$defs/{other}"} for other in names},
            }
            for name in names
        },
    }
    logger = Mock()
    with patch("domyn.agents.tools.mcp.mcp_tool.get_logger", return_value=logger):
        tool = await _mcp_tool_with_schema("graph", "A tool.", schema)

    assert "$ref" not in _keys_anywhere(tool.tool_definition)
    warning = logger.warning.call_args[0][0]
    assert "too large" in warning
    assert "graph" in warning
