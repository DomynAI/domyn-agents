import json
from dataclasses import dataclass
from random import randint
from typing import Annotated

import pytest
from jsonschema import Draft202012Validator

from domyn.agents import Part
from domyn.agents.core.decorators import tool
from domyn.agents.tools import ConfigParam, FunctionTool, StreamableFunctionTool
from domyn.agents.tools.base import (
    Param,
    ToolResult,
    _parse_napoleon_docstring,
)


@pytest.mark.parametrize(
    "docstring, expected",
    [
        (
            """
            Doc of function

            Args:
                param1 (int): The first parameter.
                param2 (str): The second parameter.

            Returns:
                bool: The return value.
            """,
            {
                "description": "Doc of function",
                "args": [
                    {"name": "param1", "type": "int", "description": "The first parameter."},
                    {"name": "param2", "type": "str", "description": "The second parameter."},
                ],
                "returns": {"type": "bool", "description": "The return value."},
                "raises": [],
            },
        ),
        (
            """
            Doc of function

            Args:
                param1: The first parameter.
                param2 (str): The second parameter.
            """,
            {
                "description": "Doc of function",
                "args": [
                    {"name": "param1", "type": "any", "description": "The first parameter."},
                    {"name": "param2", "type": "str", "description": "The second parameter."},
                ],
                "returns": None,
                "raises": [],
            },
        ),
        (
            """
            Doc of function

            Args:
                param1: The first parameter.
                param2 (str | None): The second parameter.
            """,
            {
                "description": "Doc of function",
                "args": [
                    {"name": "param1", "type": "any", "description": "The first parameter."},
                    {
                        "name": "param2",
                        "type": "str | None",
                        "description": "The second parameter.",
                    },
                ],
                "returns": None,
                "raises": [],
            },
        ),
        (
            """
            Doc of function

            Args:
                param1 (List[str]): The first parameter.
                param2 (str | None): The second parameter.
            """,
            {
                "description": "Doc of function",
                "args": [
                    {"name": "param1", "type": "List[str]", "description": "The first parameter."},
                    {
                        "name": "param2",
                        "type": "str | None",
                        "description": "The second parameter.",
                    },
                ],
                "returns": None,
                "raises": [],
            },
        ),
        (
            """Doc of function

            Args:
                param1 (str): The first parameter.
                param2 (str, optional): The second parameter.

            Returns:
                str: response""",
            {
                "description": "Doc of function",
                "args": [
                    {"name": "param1", "type": "str", "description": "The first parameter."},
                    {
                        "name": "param2",
                        "type": "str | None",
                        "description": "The second parameter.",
                    },
                ],
                "returns": {"description": "response", "type": "str"},
                "raises": [],
            },
        ),
        (
            """Doc of function

            Args:
                param1 (list[dict[str, str]]): The first parameter.

            Returns:
                str: response""",
            {
                "description": "Doc of function",
                "args": [
                    {
                        "name": "param1",
                        "type": "list[dict[str, str]]",
                        "description": "The first parameter.",
                    },
                ],
                "returns": {"description": "response", "type": "str"},
                "raises": [],
            },
        ),
        (
            """Doc of function

            Args:
                param1 (list[dict[str, str]], optional): The first parameter.

            Returns:
                str: response""",
            {
                "description": "Doc of function",
                "args": [
                    {
                        "name": "param1",
                        "type": "list[dict[str, str]] | None",
                        "description": "The first parameter.",
                    },
                ],
                "returns": {"description": "response", "type": "str"},
                "raises": [],
            },
        ),
        (
            """Doc of function

            Args:
                param1 (list[dict[str,    str]],     optional): The first parameter.

            Returns:
                str: response""",
            {
                "description": "Doc of function",
                "args": [
                    {
                        "name": "param1",
                        "type": "list[dict[str, str]] | None",
                        "description": "The first parameter.",
                    },
                ],
                "returns": {"description": "response", "type": "str"},
                "raises": [],
            },
        ),
        ("Doc of function", None),
        (None, None),
    ],
)
def test_parse_napoleon_docstring(docstring, expected):
    result = _parse_napoleon_docstring(docstring)
    if expected:
        assert result is not None
        assert result["description"] == expected["description"]
        assert len(result["args"]) == len(expected["args"])
        assert result["args"] == expected["args"]
        assert result["returns"] == expected["returns"]
    else:
        assert result is None


def test_function_tool_parsing():
    def sample_function(param1: int, param2: str) -> bool:
        """
        Sample function.

        Args:
            param1 (int): The first parameter.
            param2 (str): The second parameter.

        Returns:
            bool: The return value.
        """
        return True

    tool = FunctionTool(sample_function)
    assert tool.declaration.name == "sample_function"
    assert tool.declaration.description == "Sample function."
    assert tool.declaration.parameters is not None
    assert tool.declaration.parameters == [
        Param(type="int", name="param1", description="The first parameter."),
        Param(type="str", name="param2", description="The second parameter."),
    ]

    assert tool.declaration.returns == Param(
        type="bool", name=None, description="The return value."
    )
    Draft202012Validator.check_schema(tool.invocation_json_schema)

    def sample_function(param, param2: int):
        "Sample Function"

    tool = FunctionTool(sample_function)
    assert tool.declaration.name == "sample_function"
    assert tool.declaration.description == "Sample Function"
    assert tool.declaration.parameters is not None
    assert tool.declaration.parameters == [
        Param(type="any", name="param", description="Parameter param"),
        Param(type="int", name="param2", description="Parameter param2"),
    ]
    Draft202012Validator.check_schema(tool.invocation_json_schema)

    def sample_function(param, param2: int, param3: str | None):
        "Sample Function"

    tool = FunctionTool(sample_function)
    assert tool.declaration.name == "sample_function"
    assert tool.declaration.description == "Sample Function"
    assert tool.declaration.parameters is not None
    assert tool.declaration.parameters == [
        Param(type="any", name="param", description="Parameter param"),
        Param(type="int", name="param2", description="Parameter param2"),
        Param(type="str | None", name="param3", description="Parameter param3"),
    ]
    Draft202012Validator.check_schema(tool.invocation_json_schema)

    def sample_function(
        param: Annotated[int, "The first annotated parameters"], param2: str
    ) -> bool:
        """Sample function."""

    tool = FunctionTool(sample_function)
    assert tool.declaration.name == "sample_function"
    assert tool.declaration.description == "Sample function."
    assert tool.declaration.parameters is not None
    assert tool.declaration.parameters == [
        Param(type="int", name="param", description="The first annotated parameters"),
        Param(type="str", name="param2", description="Parameter param2"),
    ]
    Draft202012Validator.check_schema(tool.invocation_json_schema)

    def sample_function(
        param: Annotated[int, "The first annotated parameters"], param2: str | None = None
    ) -> bool:
        """Sample function."""

    tool = FunctionTool(sample_function)
    assert tool.declaration.name == "sample_function"
    assert tool.declaration.description == "Sample function."
    assert tool.declaration.parameters is not None
    assert tool.declaration.parameters == [
        Param(type="int", name="param", description="The first annotated parameters"),
        Param(
            type="str | None",
            name="param2",
            description="Parameter param2",
            is_required=False,
            default=None,
        ),
    ]
    Draft202012Validator.check_schema(tool.invocation_json_schema)


def test_parsing_in_class():
    class ExampleClass:
        def sample_function_in_class(
            self, param: Annotated[int, "The first annotated parameters"], param2: str
        ) -> bool:
            """Sample function in class."""

    tool = FunctionTool(ExampleClass().sample_function_in_class)
    assert tool.declaration.name == "sample_function_in_class"
    assert tool.declaration.description == "Sample function in class."
    assert tool.declaration.parameters is not None
    assert tool.declaration.parameters == [
        Param(type="int", name="param", description="The first annotated parameters"),
        Param(type="str", name="param2", description="Parameter param2"),
    ]
    Draft202012Validator.check_schema(tool.invocation_json_schema)


def test_streamable_function_tool_parsing():
    async def sample_async_generator(param1: int, param2: str):
        """
        Sample async generator function.

        Args:
            param1 (int): The first parameter.
            param2 (str): The second parameter.
        """
        yield "part1"
        yield "part2"
        yield {"result": True}

    tool = StreamableFunctionTool(sample_async_generator)
    assert tool.declaration.name == "sample_async_generator"
    assert tool.declaration.description == "Sample async generator function."
    assert tool.declaration.parameters is not None
    assert tool.declaration.parameters == [
        Param(type="int", name="param1", description="The first parameter."),
        Param(type="str", name="param2", description="The second parameter."),
    ]
    assert tool.declaration.returns is None
    assert tool._is_async_generator
    Draft202012Validator.check_schema(tool.invocation_json_schema)


def test_function_with_no_docs_and_config_param():
    async def sample_async_func_mismatch_b(param1: int, param2: str, param3: float):
        return "hello"

    tool = FunctionTool(
        func=sample_async_func_mismatch_b,
        config_params=[ConfigParam(type="float", name="param3", value=1.0)],
    )
    assert tool.declaration.parameters == [
        Param(type="int", name="param1", description="Parameter param1"),
        Param(type="str", name="param2", description="Parameter param2"),
    ]
    Draft202012Validator.check_schema(tool.invocation_json_schema)


def test_mismatch_arg_and_docs():
    async def sample_async_func_mismatch(param1: int, param2: str, param3: float):
        """
        Sample async function with mismatch.

        Args:
            param1 (int): The first parameter from doc.
            param2 (str): The second parameter from doc.
        """
        return "hello"

    with pytest.raises(ValueError):
        tool = FunctionTool(sample_async_func_mismatch)
        assert tool.declaration

    async def sample_async_func_mismatch_b(param1: int, param2: str, param3: float):
        """
        Sample async function with mismatch.

        Args:
            param1 (int): The first parameter from doc.
            param2 (str): The second parameter from doc.
        """
        return "hello"

    tool = FunctionTool(
        func=sample_async_func_mismatch_b,
        config_params=[ConfigParam(type="float", name="param3", value=1.0)],
    )
    assert tool.declaration.parameters == [
        Param(type="int", name="param1", description="The first parameter from doc."),
        Param(type="str", name="param2", description="The second parameter from doc."),
    ]
    Draft202012Validator.check_schema(tool.invocation_json_schema)

    async def sample_async_func_mismatch_c(param1: int, param2: str, param3: float):
        """
        Sample async function with mismatch.

        Args:
            param1 (int | float): The first parameter from doc.
            param2 (str | None): The second parameter from doc.
        """
        return "hello"

    tool = FunctionTool(
        func=sample_async_func_mismatch_c,
        config_params=[ConfigParam(type="float", name="param3", value=1.0)],
    )
    assert tool.declaration.parameters == [
        Param(type="int | float", name="param1", description="The first parameter from doc."),
        Param(type="str | None", name="param2", description="The second parameter from doc."),
    ]
    Draft202012Validator.check_schema(tool.invocation_json_schema)


def test_invocation_schema_simple_param():
    def sample_function(param1: int, param2: str) -> bool:
        """
        Sample function.

        Args:
            param1 (int): The first parameter.
            param2 (str | None): The second parameter.

        Returns:
            bool: The return value.
        """
        return True

    tool = FunctionTool(
        func=sample_function,
    )
    assert tool.declaration.parameters == [
        Param(type="int", name="param1", description="The first parameter."),
        Param(type="str | None", name="param2", description="The second parameter."),
    ]
    assert tool.invocation_json_schema == json.loads(
        '{"properties": {"name": {"const": "sample_function", "type": "string"}, "arguments": {"properties": {"param1": {"type": "integer", "description": "The first parameter."}, "param2": {"anyOf": [{"type": "string"}, {"type": "null"}], "description": "The second parameter."}}, "required": ["param1", "param2"]}}, "required": ["arguments"], "description": "Sample function."}'
    )
    Draft202012Validator.check_schema(tool.invocation_json_schema)


def test_invocation_schema_simple_union_param():
    def sample_function(param1: int, param2: str | float) -> bool:
        """
        Sample function.

        Args:
            param1 (int): The first parameter.
            param2 (str | float): The second parameter.

        Returns:
            bool: The return value.
        """
        return True

    tool = FunctionTool(
        func=sample_function,
    )
    assert tool.declaration.parameters == [
        Param(type="int", name="param1", description="The first parameter."),
        Param(type="str | float", name="param2", description="The second parameter."),
    ]
    assert tool.invocation_json_schema == json.loads(
        '{"properties": {"name": {"const": "sample_function", "type": "string"}, "arguments": {"properties": {"param1": {"type": "integer", "description": "The first parameter."}, "param2": {"anyOf": [{"type": "string"}, {"type": "number"}], "description": "The second parameter."}}, "required": ["param1", "param2"]}}, "required": ["arguments"], "description": "Sample function."}'
    )
    Draft202012Validator.check_schema(tool.invocation_json_schema)


def test_invocation_schema_simple_with_optional():
    def sample_function(param1: int, param2: str) -> bool:
        """Doc of function

        Args:
            param1 (str): The first parameter.
            param2 (str, optional): The second parameter.

        Returns:
            str: response"""
        return True

    tool = FunctionTool(
        func=sample_function,
    )
    assert tool.declaration.parameters == [
        Param(type="str", name="param1", description="The first parameter."),
        Param(type="str | None", name="param2", description="The second parameter."),
    ]
    assert tool.invocation_json_schema == json.loads(
        '{"properties": {"name": {"const": "sample_function", "type": "string"}, "arguments": {"properties": {"param1": {"type": "string", "description": "The first parameter."}, "param2": {"anyOf": [{"type": "string"}, {"type": "null"}], "description": "The second parameter."}}, "required": ["param1", "param2"]}}, "required": ["arguments"], "description": "Doc of function"}'
    )
    Draft202012Validator.check_schema(tool.invocation_json_schema)


def test_invocation_schema_simple_with_list():
    def sample_function(param1: int, param2: list[str]) -> bool:
        """Doc of function

        Args:
            param1 (str): The first parameter.
            param2 (List[str]): The second parameter.

        Returns:
            str: response"""
        return True

    tool = FunctionTool(
        func=sample_function,
    )
    assert tool.declaration.parameters == [
        Param(type="str", name="param1", description="The first parameter."),
        Param(type="List[str]", name="param2", description="The second parameter."),
    ]
    assert tool.invocation_json_schema == json.loads(
        '{"properties": {"name": {"const": "sample_function", "type": "string"}, "arguments": {"properties": {"param1": {"description": "The first parameter.", "type": "string"}, "param2": {"description": "The second parameter.", "type": "array", "items": {"type": "string"}}}, "required": ["param1", "param2"]}}, "required": ["arguments"], "description": "Doc of function"}'
    )
    Draft202012Validator.check_schema(tool.invocation_json_schema)


def test_declaration_function_in_dataclass():
    @dataclass
    class ToolClass:
        val: str

        @tool(config_params=[ConfigParam(type="str", name="key", value="test")])
        def function_to_test(self, param1: str, param2: list[str], param3: str, key: str = ""):
            """Send an email using SendGrid.

            Args:
                param1 (str): the param1 from doc.
                param2 (List[str]): the param2 from doc.
                param3 (str): the param3 from doc.
                key (str): The plain text content of the email. Optional.


            Returns:
                bool: True if the email was sent successfully, False otherwise.
            """

    t = ToolClass(val="test")

    assert t.function_to_test.declaration.description == "Send an email using SendGrid."
    assert t.function_to_test.declaration.parameters == [
        Param(type="str", name="param1", description="the param1 from doc."),
        Param(type="List[str]", name="param2", description="the param2 from doc."),
        Param(type="str", name="param3", description="the param3 from doc."),
    ]
    Draft202012Validator.check_schema(t.function_to_test.invocation_json_schema)


def test_invocation_schema_with_any():
    class Something:
        pass

    def sample_function(param1: int, param2: str, param3: Something) -> bool:
        """Doc of function

        Args:
            param1 (str): The first parameter.
            param2 (str, optional): The second parameter.
            param3 (something): The third parameter.

        Returns:
            str: response"""
        return True

    tool = FunctionTool(
        func=sample_function,
    )

    assert tool.declaration.parameters == [
        Param(type="str", name="param1", description="The first parameter."),
        Param(type="str | None", name="param2", description="The second parameter."),
        Param(type="something", name="param3", description="The third parameter."),
    ]

    # force different params
    tool.declaration.parameters = [
        Param(type="something", name="param1", description="The first parameter."),
        Param(type="str", name="param2", description="The second parameter."),
        Param(type="any", name="param3", description="The third parameter."),
    ]

    assert tool.invocation_json_schema == json.loads(
        '{"properties": {"name": {"const": "sample_function", "type": "string"}, "arguments": {"properties": {"param1": {"description": "The first parameter."}, "param2": {"description": "The second parameter.", "type": "string"}, "param3": {"description": "The third parameter."}}, "required": ["param1", "param2", "param3"]}}, "required": ["arguments"], "description": "Doc of function"}'
    )
    Draft202012Validator.check_schema(tool.invocation_json_schema)


def test_tool_result_emitted_content():
    # empty / falsy values
    assert ToolResult(success=True, emitted_content=None).emitted_content == []
    assert ToolResult(success=True, emitted_content=[]).emitted_content == []

    # single Part instance
    part = Part(text="hello")
    result = ToolResult(success=True, emitted_content=[part])
    assert result.emitted_content == [part]

    # Part(html=...) survives a JSON round-trip (this was the bug: list-of-dicts was
    # stringified into a single Part(text=...) instead of being converted back to Parts)
    html_part = Part(html="<b>hello</b>")
    original = ToolResult(
        success=True, data="HTML displayed correctly", emitted_content=[html_part]
    )
    reparsed = ToolResult.model_validate(json.loads(original.model_dump_json()))
    assert len(reparsed.emitted_content) == 1
    assert reparsed.emitted_content[0].html == "<b>hello</b>"
    assert reparsed.emitted_content[0].text is None

    # multiple parts survive round-trip
    multi = ToolResult(
        success=True,
        emitted_content=[Part(text="intro"), Part(html="<p>body</p>"), Part(thought="thinking")],
    )
    reparsed_multi = ToolResult.model_validate(json.loads(multi.model_dump_json()))
    assert len(reparsed_multi.emitted_content) == 3
    assert reparsed_multi.emitted_content[0].text == "intro"
    assert reparsed_multi.emitted_content[1].html == "<p>body</p>"
    assert reparsed_multi.emitted_content[2].thought == "thinking"

    # single dict input is wrapped into one Part
    result_from_dict = ToolResult(success=True, emitted_content={"text": "from dict"})
    assert len(result_from_dict.emitted_content) == 1
    assert result_from_dict.emitted_content[0].text == "from dict"

    # non-Part, non-dict scalar falls back to Part(text=str(...))
    result_from_str = ToolResult(success=True, emitted_content="raw string")
    assert len(result_from_str.emitted_content) == 1
    assert result_from_str.emitted_content[0].text == "raw string"


@pytest.mark.asyncio
async def test_pass_state():
    async def sample_function_a(param1: int, state: dict) -> dict:
        state["param1"] = param1
        return {"result": True, "state": state}

    def sample_function_b(param2: int, state: dict) -> dict:
        state["param2"] = param2
        return {"result": True, "state": state}

    initial_state = {"param": True}

    tool_a = FunctionTool(
        func=sample_function_a,
    )

    tool_b = FunctionTool(
        func=sample_function_b,
    )

    a = randint(1, 100)
    b = randint(1, 100)

    result_a = await tool_a.execute(param1=a, state=initial_state)
    result_b = await tool_b.execute(param2=b, state=initial_state)

    assert result_a.state == {"param": True, "param1": a}
    assert result_b.state == {"param": True, "param2": b}
    assert initial_state == {"param": True}


def test_tool_result_from_result_dict_shaped():
    """from_result_dict casts ToolResult-shaped dicts and returns None otherwise."""
    shaped = ToolResult.from_result_dict({"success": True, "data": {"v": 1}, "metadata": {"m": 2}})
    assert isinstance(shaped, ToolResult)
    assert shaped.success is True
    assert shaped.data == {"v": 1}
    assert shaped.metadata == {"m": 2}

    # Missing required `success`.
    assert ToolResult.from_result_dict({"data": {"v": 1}}) is None
    # Extra key (`result`) forbidden by the model.
    assert ToolResult.from_result_dict({"success": True, "result": 1}) is None
    # Legacy "result" envelope: not cast unless the fallback is opted in.
    assert ToolResult.from_result_dict({"result": {"v": 1}}) is None
    # Non-dict inputs.
    assert ToolResult.from_result_dict("a string") is None
    assert ToolResult.from_result_dict(["a", "list"]) is None


def test_tool_result_from_result_dict_result_key_fallback():
    """With result_key_fallback, a legacy "result" envelope is lifted into data."""
    coerced = ToolResult.from_result_dict(
        {"result": {"v": 1}, "metadata": {"m": 2}}, result_key_fallback=True
    )
    assert isinstance(coerced, ToolResult)
    assert coerced.success is True
    assert coerced.data == {"v": 1}
    assert coerced.metadata == {"m": 2}
    assert coerced.state is None

    # default_state is used when the legacy envelope omits "state".
    with_default = ToolResult.from_result_dict(
        {"result": 1}, result_key_fallback=True, default_state={"s": 9}
    )
    assert with_default is not None
    assert with_default.state == {"s": 9}

    # Explicit "state" in the dict wins over default_state.
    with_state = ToolResult.from_result_dict(
        {"result": 1, "state": {"s": 1}}, result_key_fallback=True, default_state={"s": 9}
    )
    assert with_state is not None
    assert with_state.state == {"s": 1}

    # A ToolResult-shaped dict still casts directly even with the fallback on.
    shaped = ToolResult.from_result_dict({"success": True, "data": "x"}, result_key_fallback=True)
    assert shaped is not None and shaped.data == "x"

    # No "result" key and not ToolResult-shaped -> still None.
    assert ToolResult.from_result_dict({"foo": "bar"}, result_key_fallback=True) is None


@pytest.mark.asyncio
async def test_function_tool_casts_tool_result_shaped_dict():
    """A dict shaped like a ToolResult is cast directly, not buried in `data`."""

    def sample_function(param1: int) -> dict:
        return {
            "success": True,
            "data": {"value": param1},
            "metadata": {"source": "unit-test"},
            "state": {"counter": param1},
        }

    tool = FunctionTool(func=sample_function)

    result = await tool.execute(param1=7)

    assert isinstance(result, ToolResult)
    assert result.success is True
    assert result.data == {"value": 7}
    assert result.metadata == {"source": "unit-test"}
    assert result.state == {"counter": 7}


@pytest.mark.asyncio
async def test_function_tool_legacy_result_key_still_works():
    """A dict with a `result` key but not ToolResult-shaped uses the legacy backoff."""

    def sample_function(param1: int) -> dict:
        return {"result": {"value": param1}, "metadata": {"source": "legacy"}}

    tool = FunctionTool(func=sample_function)

    result = await tool.execute(param1=3)

    assert result.success is True
    assert result.data == {"value": 3}
    assert result.metadata == {"source": "legacy"}


@pytest.mark.asyncio
async def test_function_tool_tool_result_shape_with_extra_result_key_uses_legacy():
    """A dict with both `success` and a `result` key is not ToolResult-shaped
    (extra `result` key), so it falls through to the legacy backoff."""

    def sample_function() -> dict:
        return {"success": True, "result": {"value": 1}}

    tool = FunctionTool(func=sample_function)

    result = await tool.execute()

    assert result.success is True
    # Legacy branch lifts `result` into `data`.
    assert result.data == {"value": 1}


@pytest.mark.asyncio
async def test_function_tool_plain_dict_goes_into_data():
    """A plain dict without `success`/`result` is wrapped into the `data` attribute."""

    def sample_function() -> dict:
        return {"foo": "bar"}

    tool = FunctionTool(func=sample_function)

    result = await tool.execute()

    assert result.success is True
    assert result.data == {"foo": "bar"}


@pytest.mark.asyncio
async def test_streamable_function_tool_casts_tool_result_shaped_dict():
    """StreamableFunctionTool also casts a ToolResult-shaped dict directly."""

    async def sample_generator(param1: int):
        yield {
            "success": True,
            "data": {"value": param1},
            "metadata": {"source": "stream"},
        }

    tool = StreamableFunctionTool(func=sample_generator)

    results = [item async for item in tool.execute(param1=5)]
    final = results[-1]

    assert isinstance(final, ToolResult)
    assert final.success is True
    assert final.data == {"value": 5}
    assert final.metadata == {"source": "stream"}
