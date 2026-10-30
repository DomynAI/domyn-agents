import json

import httpx
import pytest
import respx
from jsonschema import Draft202012Validator

from domyn.agents.tools import RestFunctionTool
from domyn.agents.tools.base import FunctionDeclaration, Param, ToolResult


def test_rest_function_tool_initialization():
    tool = RestFunctionTool(
        name="Test REST Tool",
        description="A tool to test REST API calls",
        url="https://api.example.com/data",
        method="GET",
    )
    assert callable(tool.callable_function)
    assert isinstance(tool.declaration, FunctionDeclaration)


@respx.mock
@pytest.mark.asyncio
async def test_rest_get_function_tool_call():
    expected_response = {"userId": 1, "id": 1, "title": "title", "completed": False}

    route = respx.get("https://jsonplaceholder.typicode.com/todos/1").mock(
        return_value=httpx.Response(200, json=expected_response)
    )

    tool = RestFunctionTool(
        name="Test REST Tool",
        description="A tool to test REST API calls",
        url="https://jsonplaceholder.typicode.com/todos/1",
        method="GET",
        headers={"accept": "application/json"},
    )

    result = await tool.execute()
    assert isinstance(result, ToolResult)

    assert result.data == expected_response
    assert route.called
    assert route.call_count == 1
    assert route.calls[0].request.headers["accept"] == "application/json"


@respx.mock
@pytest.mark.asyncio
async def test_rest_get_function_tool_call_with_params():
    expected_response = {"userId": 1, "id": 1, "title": "title", "completed": False}

    route = respx.get("https://jsonplaceholder.typicode.com/todos").mock(
        return_value=httpx.Response(200, json=expected_response)
    )

    tool = RestFunctionTool(
        name="Test REST Tool",
        description="A tool to test REST API calls",
        url="https://jsonplaceholder.typicode.com/todos",
        method="GET",
        headers={"accept": "application/json"},
        body=[Param(name="id", type="int", description="The ID of the todo item")],
    )

    result = await tool.execute(id=1)
    assert isinstance(result, ToolResult)

    assert result.data == expected_response
    assert route.called
    assert route.call_count == 1
    assert route.calls[0].request.headers["accept"] == "application/json"
    assert str(route.calls[0].request.url) == "https://jsonplaceholder.typicode.com/todos?id=1"

    # case where params are not removed

    tool = RestFunctionTool(
        name="Test REST Tool",
        description="A tool to test REST API calls",
        url="https://jsonplaceholder.typicode.com/todos?already=present",
        method="GET",
        headers={"accept": "plain/text"},
    )

    result = await tool.execute()

    assert result.data == expected_response
    assert route.called
    assert route.calls[1].request.headers["accept"] == "plain/text"
    assert (
        str(route.calls[1].request.url)
        == "https://jsonplaceholder.typicode.com/todos?already=present"
    )

    # mixed with already present and new params

    tool = RestFunctionTool(
        name="Test REST Tool",
        description="A tool to test REST API calls",
        url="https://jsonplaceholder.typicode.com/todos?already=present",
        method="GET",
        headers={"accept": "text/xml"},
        body=[Param(name="id", type="int", description="The ID of the todo item")],
    )

    result = await tool.execute(id=1)

    assert result.data == expected_response
    assert route.called
    assert route.calls[2].request.headers["accept"] == "text/xml"
    assert (
        str(route.calls[2].request.url)
        == "https://jsonplaceholder.typicode.com/todos?already=present&id=1"
    )


@respx.mock
@pytest.mark.asyncio
async def test_rest_post_function_tool_call():
    expected_response = {"field": "test"}

    route = respx.post("https://jsonplaceholder.typicode.com/posts").mock(
        return_value=httpx.Response(200, json=expected_response)
    )

    tool = RestFunctionTool(
        name="Test REST Tool",
        description="A tool to test POST REST API calls",
        url="https://jsonplaceholder.typicode.com/posts",
        method="POST",
        headers={"accept": "application/json"},
        config_params=[{"name": "api-key", "value": "my-secret-key", "type": "str"}],
        body=[Param(name="field", type="str", description="A test field")],
    )

    result = await tool.execute(field="test", wrong_param="should be ignored")
    assert isinstance(result, ToolResult)

    assert result.data == expected_response
    assert route.called
    assert route.call_count == 1
    assert route.calls[0].request.headers["accept"] == "application/json"
    assert json.loads(route.calls[0].request.content) == {
        "field": "test",
        "api-key": "my-secret-key",
    }


@respx.mock
@pytest.mark.asyncio
async def test_rest_post_function_tool_call_without_execute():
    expected_response = {"field": "test"}

    route = respx.post("https://jsonplaceholder.typicode.com/posts").mock(
        return_value=httpx.Response(200, json=expected_response)
    )

    tool = RestFunctionTool(
        name="Test REST Tool",
        description="A tool to test POST REST API calls",
        url="https://jsonplaceholder.typicode.com/posts",
        method="POST",
        headers={"accept": "application/json"},
        config_params=[{"name": "api-key", "value": "my-secret-key", "type": "str"}],
        body=[Param(name="field", type="str", description="A test field")],
    )

    result = await tool(field="test", wrong_param="should be ignored")
    assert not isinstance(result, ToolResult) and isinstance(result, dict)

    assert result == expected_response
    assert route.called
    assert route.call_count == 1
    assert route.calls[0].request.headers["accept"] == "application/json"
    assert json.loads(route.calls[0].request.content) == {
        "field": "test",
        "api-key": "my-secret-key",
    }


@respx.mock
@pytest.mark.asyncio
async def test_rest_post_exception():
    expected_response = "error"

    route = respx.post("https://jsonplaceholder.typicode.com/posts").mock(
        return_value=httpx.Response(500, json=expected_response)
    )

    tool = RestFunctionTool(
        name="Test REST Tool",
        description="A tool to test POST REST API calls",
        url="https://jsonplaceholder.typicode.com/posts",
        method="POST",
        headers={"accept": "application/json"},
        config_params=[{"name": "api-key", "value": "my-secret-key", "type": "str"}],
        body=[Param(name="field", type="str", description="A test field")],
    )

    result = await tool.execute(field="test", wrong_param="should be ignored")

    assert isinstance(result, ToolResult)
    assert result.success is False
    assert route.called
    assert route.call_count == 1
    assert route.calls[0].request.headers["accept"] == "application/json"
    assert json.loads(route.calls[0].request.content) == {
        "field": "test",
        "api-key": "my-secret-key",
    }


@respx.mock
@pytest.mark.asyncio
async def test_rest_post_function_tool_call_from_config():
    expected_response = {"field": "test"}

    route = respx.post("https://jsonplaceholder.typicode.com/posts").mock(
        return_value=httpx.Response(200, json=expected_response)
    )

    tool = RestFunctionTool.from_config(
        {
            "name": "Test REST Tool",
            "description": "A tool to test POST REST API calls",
            "url": "https://jsonplaceholder.typicode.com/posts",
            "method": "POST",
            "headers": {"accept": "application/json"},
            "body": [{"name": "field", "type": "str", "description": "A test field"}],
        }
    )

    result = await tool.execute(field="test", wrong_param="should be ignored")
    assert isinstance(result, ToolResult)
    assert result.data == expected_response
    assert route.called
    assert route.call_count == 1
    assert route.calls[0].request.headers["accept"] == "application/json"
    assert json.loads(route.calls[0].request.content) == {"field": "test"}


invocation_schema_test_cases = [
    (
        [{"name": "param1", "type": "str", "description": "A string parameter"}],
        {
            "properties": {"param1": {"description": "A string parameter", "type": "string"}},
            "required": ["param1"],
        },
    ),
    (
        [{"name": "param1", "type": "dict[str, int]", "description": "description param1"}],
        {
            "properties": {
                "param1": {
                    "description": "description param1",
                    "type": "object",
                    "additionalProperties": {"type": "integer"},
                }
            },
            "required": ["param1"],
        },
    ),
    (
        [{"name": "param1", "type": "dict[str, int|str]", "description": "description param1"}],
        {
            "properties": {
                "param1": {
                    "description": "description param1",
                    "type": "object",
                    "additionalProperties": {"anyOf": [{"type": "integer"}, {"type": "string"}]},
                }
            },
            "required": ["param1"],
        },
    ),
    (
        [{"name": "param1", "type": "Dict[str, int|str]", "description": "description param1"}],
        {
            "properties": {
                "param1": {
                    "description": "description param1",
                    "type": "object",
                    "additionalProperties": {"anyOf": [{"type": "integer"}, {"type": "string"}]},
                }
            },
            "required": ["param1"],
        },
    ),
    (
        [{"name": "param1", "type": "list[int|str]", "description": "description param1"}],
        {
            "properties": {
                "param1": {
                    "description": "description param1",
                    "type": "array",
                    "items": {"anyOf": [{"type": "integer"}, {"type": "string"}]},
                }
            },
            "required": ["param1"],
        },
    ),
    (
        [{"name": "param1", "type": "List[int|str]", "description": "description param1"}],
        {
            "properties": {
                "param1": {
                    "description": "description param1",
                    "type": "array",
                    "items": {"anyOf": [{"type": "integer"}, {"type": "string"}]},
                }
            },
            "required": ["param1"],
        },
    ),
    (
        [
            {
                "name": "param1",
                "type": "Dict[str, List[int|str]]",
                "description": "description param1",
            }
        ],
        {
            "properties": {
                "param1": {
                    "description": "description param1",
                    "type": "object",
                    "additionalProperties": {
                        "type": "array",
                        "items": {"anyOf": [{"type": "integer"}, {"type": "string"}]},
                    },
                }
            },
            "required": ["param1"],
        },
    ),
    (
        [
            {
                "name": "param1",
                "type": "Dict[str, dict[str, int|str]]",
                "description": "description param1",
            }
        ],
        {
            "properties": {
                "param1": {
                    "description": "description param1",
                    "type": "object",
                    "additionalProperties": {
                        "type": "object",
                        "additionalProperties": {
                            "anyOf": [{"type": "integer"}, {"type": "string"}]
                        },
                    },
                }
            },
            "required": ["param1"],
        },
    ),
    (
        [
            {
                "name": "param1",
                "type": "List[list[dict[str, float|None]]]",
                "description": "description param1",
            }
        ],
        {
            "properties": {
                "param1": {
                    "description": "description param1",
                    "type": "array",
                    "items": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": {
                                "anyOf": [{"type": "number"}, {"type": "null"}]
                            },
                        },
                    },
                }
            },
            "required": ["param1"],
        },
    ),
    (
        [
            {"name": "param1", "type": "int", "description": "An integer parameter"},
            {"name": "param2", "type": "float", "description": "A float parameter"},
        ],
        {
            "properties": {
                "param1": {"description": "An integer parameter", "type": "integer"},
                "param2": {"description": "A float parameter", "type": "number"},
            },
            "required": ["param1", "param2"],
        },
    ),
]


@pytest.mark.parametrize("body, expected_schema", invocation_schema_test_cases)
@pytest.mark.asyncio
async def test_rest_function_tool_invocation_schema(body: dict, expected_schema: dict):
    tool = RestFunctionTool.from_config(
        {
            "name": "Test REST Tool",
            "description": "A tool to test REST API calls",
            "url": "https://api.example.com/data",
            "method": "GET",
            "body": body,
        }
    )

    schema = tool.invocation_json_schema
    Draft202012Validator.check_schema(schema)
    assert schema["properties"]["arguments"] == expected_schema
