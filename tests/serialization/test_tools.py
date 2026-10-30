import json
from collections.abc import AsyncGenerator

from domyn.agents.tools import FunctionTool, RestFunctionTool, StreamableFunctionTool
from domyn.agents.tools.base import ConfigParam, FunctionDeclaration, Param


def function_tool(x: int) -> int:
    """A test function that returns the input value."""
    return x


async def stream_function_tool(x: int) -> AsyncGenerator[str, None]:
    """A test function that returns the input value."""
    for char in "streamed response":
        yield char


def test_function_tool_serialization():
    tool = FunctionTool(
        func=function_tool,
        config_params=[ConfigParam(name="param1", value="value1")],
    )
    serialized_dict = json.loads(tool.model_dump_json())

    tool_from_dict = FunctionTool.from_config(serialized_dict)

    assert tool_from_dict.callable_function.__name__ == tool.callable_function.__name__
    assert tool_from_dict.type == tool.type
    assert tool_from_dict.declaration == tool.declaration
    assert tool_from_dict.pre_hooks == tool.pre_hooks
    assert tool_from_dict.post_hooks == tool.post_hooks


def test_streamable_function_tool_serialization():
    tool = StreamableFunctionTool(
        func=stream_function_tool,
    )
    serialized_dict = dict(tool)

    tool_from_dict = StreamableFunctionTool.from_config(serialized_dict)

    assert tool_from_dict.callable_function.__name__ == tool.callable_function.__name__
    assert tool_from_dict.type == tool.type
    assert tool_from_dict.declaration == tool.declaration
    assert tool_from_dict.pre_hooks == tool.pre_hooks
    assert tool_from_dict.post_hooks == tool.post_hooks


def test_rest_function_tool_serialization():
    tool = RestFunctionTool(
        name="Test REST Tool",
        description="A test REST tool",
        url="https://api.example.com/data",
        method="POST",
        headers={"Authorization": "Bearer token"},
        body=[Param(type="string", name="param1", description="A query parameter")],
    )
    serialized_dict = dict(tool)

    tool_from_dict = RestFunctionTool.from_config(serialized_dict)

    assert tool_from_dict.name == tool.name
    assert tool_from_dict.description == tool.description
    assert tool_from_dict.url == tool.url
    assert tool_from_dict.method == tool.method
    assert tool_from_dict.headers == tool.headers
    assert tool_from_dict.body == tool.body
    assert tool_from_dict.type == tool.type
    assert tool_from_dict.declaration == tool.declaration
    assert tool_from_dict.pre_hooks == tool.pre_hooks
    assert tool_from_dict.post_hooks == tool.post_hooks

    assert tool_from_dict.callable_function.__name__ == tool.callable_function.__name__
