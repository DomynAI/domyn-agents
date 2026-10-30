from collections.abc import Callable
from typing import Any

import httpx
import pytest

from domyn.agents import Part
from domyn.agents.hooks import (
    FunctionHook,
    HookError,
    HookPhase,
    HookRestAPIData,
    HookResult,
    HookType,
)
from domyn.agents.tools import FunctionTool, ToolResult

CONFIG_PARAMS = {"param1": "value1", "param2": 2, "param3": True}


@pytest.fixture
def tool_args() -> dict:
    return {"arg1": "value1", "arg2": 2, "arg3": True}


def pre_hook_no_result(tool: FunctionTool, tool_args: dict, **config_params: Any) -> None:
    pass


def pre_hook_with_hook_result(
    tool: FunctionTool, tool_args: dict, **config_params: Any
) -> HookResult:
    return HookResult(
        emitted_content=None,
        name=pre_hook_with_hook_result.__name__,
        success=True,
        error=None,
        new_tool_args=None,
    )


def pre_hook_with_dict_result(
    tool: FunctionTool, tool_args: dict, **config_params: Any
) -> dict[Any, Any]:
    return {}


def pre_hook_with_list_return_type(
    tool: FunctionTool, tool_args: dict, **config_params: Any
) -> list[Any]:
    return []


def pre_hook_modifies_tool_args(
    tool: FunctionTool, tool_args: dict, **config_params: Any
) -> dict[str, Any]:
    return {"new_tool_args": {"param1": "value1", "param2": 2.3}}


def pre_hook_modifies_tool_args_2(
    tool: FunctionTool, tool_args: dict, **config_params: Any
) -> HookResult:
    return HookResult(
        emitted_content=None,
        name=pre_hook_modifies_tool_args_2.__name__,
        success=True,
        error=None,
        new_tool_args={"param1": "value1", "param2": 2.3},
    )


def pre_hook_modifies_tool_args_and_event(
    tool: FunctionTool, tool_args: dict, **config_params: Any
) -> HookResult:
    return HookResult(
        emitted_content={"metadata_key": "metadata_value"},
        name=pre_hook_modifies_tool_args_and_event.__name__,
        success=True,
        error=None,
        new_tool_args={"param1": "value1", "param2": 2.3},
    )


def pre_hook_with_emitted_content(
    tool: FunctionTool, tool_args: dict, **config_params: Any
) -> HookResult:
    return HookResult(
        emitted_content={"key": "value"},
        name=pre_hook_with_emitted_content.__name__,
        success=True,
        error=None,
        new_tool_args=None,
    )


def pre_hook_with_emitted_content_2(
    tool: FunctionTool, tool_args: dict, **config_params: Any
) -> dict[str, Any]:
    return {"emitted_content": {"key": "value"}}


def pre_hook_with_emitted_content_3(
    tool: FunctionTool, tool_args: dict, **config_params: Any
) -> dict[str, Any]:
    return {"key": "value"}


def pre_hook_with_event_and_result_dict(
    tool: FunctionTool, tool_args: dict, **config_params: Any
) -> dict[str, Any]:
    return {
        "emitted_content": {"key": "value"},
        "new_tool_args": {"param1": "value1", "param2": 2.3},
    }


def pre_hook_with_config_params_and_tool_args(
    tool: FunctionTool, tool_args: dict, **config_params: Any
) -> dict[str, Any]:
    return {
        "emitted_content": config_params,
        "new_tool_args": {"param1": "value1", "param2": 2.3},
    }


def pre_hook_only_config_params(
    tool: FunctionTool, tool_args: dict, **config_params: Any
) -> dict[str, Any]:
    return {
        "emitted_content": config_params,
    }


############ Post-hook functions ############


def post_hook_no_result(tool: FunctionTool, result: ToolResult, **config_params: Any) -> None:
    pass


def post_hook_with_hook_result(
    tool: FunctionTool, result: ToolResult, **config_params: Any
) -> HookResult:
    return HookResult(
        emitted_content=None,
        name=post_hook_with_hook_result.__name__,
        success=True,
        error=None,
        new_result=None,
    )


def post_hook_with_hook_result_2(
    tool: FunctionTool, result: ToolResult, **config_params: Any
) -> HookResult:
    return HookResult(
        emitted_content=None,
        name=post_hook_with_hook_result_2.__name__,
        success=True,
        error=None,
        new_result=ToolResult(success=True),
    )


def post_hook_with_dict_result(
    tool: FunctionTool, result: ToolResult, **config_params: Any
) -> dict[Any, Any]:
    return {}


def post_hook_with_list_return_type(
    tool: FunctionTool, result: ToolResult, **config_params: Any
) -> list[Any]:
    return []


def post_hook_modifies_result(
    tool: FunctionTool, result: ToolResult, **config_params: Any
) -> dict[str, Any]:
    return {"new_result": {"param1": "value1", "param2": 2.3}}


def post_hook_modifies_result_2(
    tool: FunctionTool, result: ToolResult, **config_params: Any
) -> dict[str, Any]:
    return {
        "new_result": {"data": {"param1": "value1", "param2": 2.3}, "success": True, "metadata": {}}
    }


def post_hook_modifies_result_3(
    tool: FunctionTool, result: ToolResult, **config_params: Any
) -> HookResult:
    return HookResult(
        emitted_content=None,
        name=post_hook_modifies_result_3.__name__,
        success=True,
        error=None,
        new_result=ToolResult(success=True, data={"param1": "value1", "param2": 2.3}, metadata={}),
    )


def post_hook_modifies_result_4(
    tool: FunctionTool, result: ToolResult, **config_params: Any
) -> ToolResult:
    return ToolResult(success=True, data={"param1": "value1", "param2": 2.3}, metadata={})


def post_hook_modifies_tool_args_and_event(
    tool: FunctionTool, result: ToolResult, **config_params: Any
) -> HookResult:
    return HookResult(
        emitted_content={"metadata_key": "metadata_value"},
        name=post_hook_modifies_tool_args_and_event.__name__,
        success=True,
        error=None,
        new_result=ToolResult(success=True, data={"param1": "value1", "param2": 2.3}, metadata={}),
    )


def post_hook_with_event_content(
    tool: FunctionTool, result: ToolResult, **config_params: Any
) -> HookResult:
    return HookResult(
        emitted_content={"key": "value"},
        name=post_hook_with_event_content.__name__,
        success=True,
        error=None,
        new_result=None,
    )


def post_hook_with_event_content_2(
    tool: FunctionTool, result: ToolResult, **config_params: Any
) -> dict[str, Any]:
    return {"emitted_content": {"key": "value"}}


def post_hook_with_event_content_3(
    tool: FunctionTool, result: ToolResult, **config_params: Any
) -> dict[str, Any]:
    return {"key": "value"}


def post_hook_with_event_and_result_dict(
    tool: FunctionTool, result: ToolResult, **config_params: Any
) -> dict[str, Any]:
    return {
        "emitted_content": {"key": "value"},
        "new_result": {"param1": "value1", "param2": 2.3},
    }


def post_hook_with_event_and_result_dict_2(
    tool: FunctionTool, result: ToolResult, **config_params: Any
) -> dict[str, Any]:
    return {
        "emitted_content": {"key": "value"},
        "new_result": {
            "success": True,
            "metadata": {},
            "data": {"param1": "value1", "param2": 2.3},
        },
    }


def post_hook_config_params_and_result(
    tool: FunctionTool, result: ToolResult, **config_params: Any
) -> dict[str, Any]:
    return {
        "emitted_content": CONFIG_PARAMS,
        "new_result": {
            "success": True,
            "metadata": {},
            "data": {"param1": "value1", "param2": 2.3},
        },
    }


def post_hook_only_config_params(
    tool: FunctionTool, result: ToolResult, **config_params: Any
) -> dict[str, Any]:
    return {
        "emitted_content": CONFIG_PARAMS,
    }


def tool_function():
    pass


@pytest.fixture
def tool() -> FunctionTool:
    return FunctionTool(func=tool_function)


@pytest.fixture
def tool_result() -> ToolResult:
    return ToolResult(success=True)


listener_pre_hooks_data = [
    (
        pre_hook_no_result,
        {"emitted_content": [], "success": True, "error": None, "new_tool_args": None},
    ),
    (
        pre_hook_with_hook_result,
        {"emitted_content": [], "success": True, "error": None, "new_tool_args": None},
    ),
    (
        pre_hook_with_dict_result,
        {"emitted_content": [], "success": True, "error": None, "new_tool_args": None},
    ),
    (
        pre_hook_with_list_return_type,
        {
            "emitted_content": [Part(text="[]")],
            "success": True,
            "error": None,
            "new_tool_args": None,
        },
    ),
    (
        pre_hook_modifies_tool_args,
        {
            "emitted_content": [],
            "success": False,
            "error": HookError(
                error_code="ValueError",
                error_message="Listener pre-hook cannot modify or return the tool arguments.",
            ),
            "new_tool_args": None,
        },
    ),
    (
        pre_hook_modifies_tool_args_2,
        {
            "emitted_content": [],
            "success": False,
            "error": HookError(
                error_code="ValueError",
                error_message="Listener pre-hook cannot modify or return the tool arguments.",
            ),
            "new_tool_args": None,
        },
    ),
    (
        pre_hook_with_emitted_content,
        {
            "emitted_content": [Part(metadata={"key": "value"})],
            "success": True,
            "error": None,
            "new_tool_args": None,
        },
    ),
    (
        pre_hook_with_emitted_content_2,
        {
            "emitted_content": [Part(metadata={"key": "value"})],
            "success": True,
            "error": None,
            "new_tool_args": None,
        },
    ),
    (
        pre_hook_with_emitted_content_3,
        {
            "emitted_content": [Part(metadata={"key": "value"})],
            "success": True,
            "error": None,
            "new_tool_args": None,
        },
    ),
    (
        pre_hook_with_event_and_result_dict,
        {
            "emitted_content": [],
            "success": False,
            "error": HookError(
                error_code="ValueError",
                error_message="Listener pre-hook cannot modify or return the tool arguments.",
            ),
            "new_tool_args": None,
        },
    ),
    (
        pre_hook_modifies_tool_args_and_event,
        {
            "emitted_content": [],
            "success": False,
            "error": HookError(
                error_code="ValueError",
                error_message="Listener pre-hook cannot modify or return the tool arguments.",
            ),
            "new_tool_args": None,
        },
    ),
    (
        pre_hook_only_config_params,
        {
            "emitted_content": [Part(metadata=CONFIG_PARAMS)],
            "success": True,
            "error": None,
            "new_tool_args": None,
        },
    ),
]

updater_pre_hooks_data = [
    (
        pre_hook_no_result,
        {
            "emitted_content": [],
            "success": False,
            "error": HookError(
                error_code="ValueError",
                error_message="Updater pre-hook must return the tool arguments.",
            ),
            "new_tool_args": None,
        },
    ),
    (
        pre_hook_with_hook_result,
        {
            "emitted_content": [],
            "success": False,
            "error": HookError(
                error_code="ValueError",
                error_message="Updater pre-hook must return the tool arguments.",
            ),
            "new_tool_args": None,
        },
    ),
    (
        pre_hook_with_dict_result,
        {
            "emitted_content": [],
            "success": False,
            "error": HookError(
                error_code="ValueError",
                error_message="Updater pre-hook must return the tool arguments.",
            ),
            "new_tool_args": None,
        },
    ),
    (
        pre_hook_with_list_return_type,
        {
            "emitted_content": [],
            "success": False,
            "error": HookError(
                error_code="ValueError",
                error_message="Updater pre-hook must return the tool arguments.",
            ),
            "new_tool_args": None,
        },
    ),
    (
        pre_hook_modifies_tool_args,
        {
            "emitted_content": [],
            "success": True,
            "error": None,
            "new_tool_args": {"param1": "value1", "param2": 2.3},
        },
    ),
    (
        pre_hook_modifies_tool_args_2,
        {
            "emitted_content": [],
            "success": True,
            "error": None,
            "new_tool_args": {"param1": "value1", "param2": 2.3},
        },
    ),
    (
        pre_hook_with_emitted_content,
        {
            "emitted_content": [],
            "success": False,
            "error": HookError(
                error_code="ValueError",
                error_message="Updater pre-hook must return the tool arguments.",
            ),
            "new_tool_args": None,
        },
    ),
    (
        pre_hook_with_emitted_content_2,
        {
            "emitted_content": [],
            "success": False,
            "error": HookError(
                error_code="ValueError",
                error_message="Updater pre-hook must return the tool arguments.",
            ),
            "new_tool_args": None,
        },
    ),
    (
        pre_hook_with_emitted_content_3,
        {
            "emitted_content": [],
            "success": False,
            "error": HookError(
                error_code="ValueError",
                error_message="Updater pre-hook must return the tool arguments.",
            ),
            "new_tool_args": None,
        },
    ),
    (
        pre_hook_with_emitted_content_2,
        {
            "emitted_content": [],
            "success": False,
            "error": HookError(
                error_code="ValueError",
                error_message="Updater pre-hook must return the tool arguments.",
            ),
            "new_tool_args": None,
        },
    ),
    (
        pre_hook_with_emitted_content_3,
        {
            "emitted_content": [],
            "success": False,
            "error": HookError(
                error_code="ValueError",
                error_message="Updater pre-hook must return the tool arguments.",
            ),
            "new_tool_args": None,
        },
    ),
    (
        pre_hook_with_event_and_result_dict,
        {
            "emitted_content": [Part(metadata={"key": "value"})],
            "success": True,
            "error": None,
            "new_tool_args": {"param1": "value1", "param2": 2.3},
        },
    ),
    (
        pre_hook_modifies_tool_args_and_event,
        {
            "emitted_content": [Part(metadata={"metadata_key": "metadata_value"})],
            "success": True,
            "error": None,
            "new_tool_args": {"param1": "value1", "param2": 2.3},
        },
    ),
    (
        pre_hook_with_config_params_and_tool_args,
        {
            "emitted_content": [Part(metadata=CONFIG_PARAMS)],
            "success": True,
            "error": None,
            "new_tool_args": {"param1": "value1", "param2": 2.3},
        },
    ),
]

listener_post_hooks_data = [
    (
        post_hook_no_result,
        {"emitted_content": [], "success": True, "error": None, "new_result": None},
    ),
    (
        post_hook_with_hook_result,
        {"emitted_content": [], "success": True, "error": None, "new_result": None},
    ),
    (
        post_hook_with_hook_result_2,
        {
            "emitted_content": [],
            "success": False,
            "error": HookError(
                error_code="ValueError",
                error_message="Listener post-hook cannot modify or return the tool result.",
            ),
            "new_result": None,
        },
    ),
    (
        post_hook_with_dict_result,
        {"emitted_content": [], "success": True, "error": None, "new_result": None},
    ),
    (
        post_hook_with_list_return_type,
        {
            "emitted_content": [Part(text="[]")],
            "success": True,
            "error": None,
            "new_result": None,
        },
    ),
    (
        post_hook_modifies_result,
        {
            "emitted_content": [],
            "success": False,
            "error": HookError(error_code="ValidationError", error_message=""),
            "new_result": None,
        },
    ),
    (
        post_hook_modifies_result_2,
        {
            "emitted_content": [],
            "success": False,
            "error": HookError(error_code="ValueError", error_message=""),
            "new_result": None,
        },
    ),
    (
        post_hook_modifies_result_3,
        {
            "emitted_content": [],
            "success": False,
            "error": HookError(error_code="ValueError", error_message=""),
            "new_result": None,
        },
    ),
    (
        post_hook_modifies_result_4,
        {
            "emitted_content": [],
            "success": False,
            "error": HookError(error_code="ValueError", error_message=""),
            "new_result": None,
        },
    ),
    (
        post_hook_modifies_tool_args_and_event,
        {
            "emitted_content": [],
            "success": False,
            "error": HookError(
                error_code="ValueError",
                error_message="",
            ),
            "new_result": None,
        },
    ),
    (
        post_hook_with_event_content,
        {
            "emitted_content": [Part(metadata={"key": "value"})],
            "success": True,
            "error": None,
            "new_result": None,
        },
    ),
    (
        post_hook_with_event_content_2,
        {
            "emitted_content": [Part(metadata={"key": "value"})],
            "success": True,
            "error": None,
            "new_result": None,
        },
    ),
    (
        post_hook_with_event_content_3,
        {
            "emitted_content": [Part(metadata={"key": "value"})],
            "success": True,
            "error": None,
            "new_result": None,
        },
    ),
    (
        post_hook_with_event_and_result_dict,
        {
            "emitted_content": [],
            "success": False,
            "error": HookError(
                error_code="ValidationError",
                error_message="",
            ),
            "new_result": None,
        },
    ),
    (
        post_hook_with_event_and_result_dict_2,
        {
            "emitted_content": [],
            "success": False,
            "error": HookError(
                error_code="ValueError",
                error_message="",
            ),
            "new_result": None,
        },
    ),
    (
        post_hook_only_config_params,
        {
            "emitted_content": [Part(metadata=CONFIG_PARAMS)],
            "success": True,
            "new_result": None,
            "error": None,
        },
    ),
]

updater_post_hooks_data = [
    (
        post_hook_no_result,
        {
            "emitted_content": [],
            "success": False,
            "error": HookError(error_code="ValueError", error_message=""),
            "new_result": None,
        },
    ),
    (
        post_hook_with_hook_result,
        {
            "emitted_content": [],
            "success": False,
            "error": HookError(error_code="ValueError", error_message=""),
            "new_result": None,
        },
    ),
    (
        post_hook_with_hook_result_2,
        {
            "emitted_content": [],
            "success": True,
            "error": None,
            "new_result": ToolResult(success=True),
        },
    ),
    (
        post_hook_with_dict_result,
        {
            "emitted_content": [],
            "success": False,
            "error": HookError(error_code="ValueError", error_message=""),
            "new_result": None,
        },
    ),
    (
        post_hook_with_list_return_type,
        {
            "emitted_content": [],
            "success": False,
            "error": HookError(error_code="ValueError", error_message=""),
            "new_result": None,
        },
    ),
    (
        post_hook_modifies_result,
        {
            "emitted_content": [],
            "success": False,
            "error": HookError(error_code="ValidationError", error_message=""),
            "new_result": None,
        },
    ),
    (
        post_hook_modifies_result_2,
        {
            "emitted_content": [],
            "success": True,
            "error": None,
            "new_result": ToolResult(
                success=True, data={"param1": "value1", "param2": 2.3}, metadata={}
            ),
        },
    ),
    (
        post_hook_modifies_result_3,
        {
            "emitted_content": [],
            "success": True,
            "error": None,
            "new_result": ToolResult(
                success=True, data={"param1": "value1", "param2": 2.3}, metadata={}
            ),
        },
    ),
    (
        post_hook_modifies_result_4,
        {
            "emitted_content": [],
            "success": True,
            "error": None,
            "new_result": ToolResult(
                success=True, data={"param1": "value1", "param2": 2.3}, metadata={}
            ),
        },
    ),
    (
        post_hook_modifies_tool_args_and_event,
        {
            "emitted_content": [Part(metadata={"metadata_key": "metadata_value"})],
            "success": True,
            "error": None,
            "new_result": ToolResult(
                success=True, data={"param1": "value1", "param2": 2.3}, metadata={}
            ),
        },
    ),
    (
        post_hook_with_event_content,
        {
            "emitted_content": [],
            "success": False,
            "error": HookError(error_code="ValueError", error_message=""),
            "new_result": None,
        },
    ),
    (
        post_hook_with_event_content_2,
        {
            "emitted_content": [],
            "success": False,
            "error": HookError(error_code="ValueError", error_message=""),
            "new_result": None,
        },
    ),
    (
        post_hook_with_event_content_3,
        {
            "emitted_content": [],
            "success": False,
            "error": HookError(error_code="ValueError", error_message=""),
            "new_result": None,
        },
    ),
    (
        post_hook_with_event_and_result_dict,
        {
            "emitted_content": [],
            "success": False,
            "error": HookError(error_code="ValidationError", error_message=""),
            "new_result": None,
        },
    ),
    (
        post_hook_with_event_and_result_dict_2,
        {
            "emitted_content": [Part(metadata={"key": "value"})],
            "success": True,
            "error": None,
            "new_result": ToolResult(
                success=True,
                data={"param1": "value1", "param2": 2.3},
                metadata={},
            ),
        },
    ),
    (
        post_hook_config_params_and_result,
        {
            "emitted_content": [Part(metadata=CONFIG_PARAMS)],
            "success": True,
            "error": None,
            "new_result": ToolResult(
                success=True,
                data={"param1": "value1", "param2": 2.3},
                metadata={},
            ),
        },
    ),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("hook_function, expected", updater_post_hooks_data)
async def test_updater_function_tool_post_hook(
    hook_function: Callable, expected: dict[str, Any], tool: FunctionTool, tool_result: ToolResult
) -> None:
    updater_post_hook = FunctionHook(
        trigger=HookPhase.POST_TOOL_CALL,
        fun=hook_function,
        config_params=CONFIG_PARAMS,
        type=HookType.UPDATER,
        break_on_error=False,
    )
    assert updater_post_hook.name == hook_function.__name__
    assert updater_post_hook.type == "updater"
    result = None
    async for item in updater_post_hook.run(tool=tool, result=tool_result):
        if not item.is_update:
            result = item
    assert isinstance(result, HookResult)
    assert result.name == updater_post_hook.name
    assert result.emitted_content == expected["emitted_content"]
    assert result.success == expected["success"]
    if expected["error"] is None:
        assert result.error is None
    else:
        assert isinstance(result.error, HookError)
        assert result.error.error_code == expected["error"].error_code
    assert result.new_result == expected["new_result"]


@pytest.mark.asyncio
@pytest.mark.parametrize("hook_function, expected", listener_post_hooks_data)
async def test_listener_function_tool_post_hook(
    hook_function: Callable, expected: dict[str, Any], tool: FunctionTool, tool_result: ToolResult
) -> None:
    listener_post_hook = FunctionHook(
        trigger=HookPhase.POST_TOOL_CALL,
        fun=hook_function,
        config_params=CONFIG_PARAMS,
        type=HookType.LISTENER,
        break_on_error=False,
    )
    assert listener_post_hook.name == hook_function.__name__
    assert listener_post_hook.type == "listener"
    result = None
    async for item in listener_post_hook.run(tool=tool, result=tool_result):
        if not item.is_update:
            result = item
    assert isinstance(result, HookResult)
    assert result.name == listener_post_hook.name
    assert result.emitted_content == expected["emitted_content"]
    assert result.success == expected["success"]
    if expected["error"] is None:
        assert result.error is None
    else:
        assert isinstance(result.error, HookError)
        assert result.error.error_code == expected["error"].error_code
    assert result.new_result == expected["new_result"]


@pytest.mark.asyncio
@pytest.mark.parametrize("hook_function, expected", updater_pre_hooks_data)
async def test_updater_function_tool_pre_hook(
    hook_function: Callable, expected: dict[str, Any], tool: FunctionTool, tool_args: dict
) -> None:
    updater_pre_hook = FunctionHook(
        trigger=HookPhase.PRE_TOOL_CALL,
        fun=hook_function,
        config_params=CONFIG_PARAMS,
        type=HookType.UPDATER,
        break_on_error=False,
    )
    assert updater_pre_hook.name == hook_function.__name__
    assert updater_pre_hook.type == "updater"
    result = None
    async for item in updater_pre_hook.run(tool=tool, tool_args=tool_args):
        if not item.is_update:
            result = item
    assert isinstance(result, HookResult)
    assert result.name == updater_pre_hook.name
    assert result.emitted_content == expected["emitted_content"]
    assert result.success == expected["success"]
    if expected["error"] is None:
        assert result.error is None
    else:
        assert isinstance(result.error, HookError)
        assert result.error.error_code == expected["error"].error_code
    assert result.error == expected["error"]
    assert result.new_tool_args == expected["new_tool_args"]


@pytest.mark.asyncio
@pytest.mark.parametrize("hook_function, expected", listener_pre_hooks_data)
async def test_listener_function_tool_pre_hook(
    hook_function: Callable, expected: dict[str, Any], tool: FunctionTool, tool_args: dict
) -> None:
    listener_pre_hook = FunctionHook(
        trigger=HookPhase.PRE_TOOL_CALL,
        fun=hook_function,
        config_params=CONFIG_PARAMS,
        type=HookType.LISTENER,
        break_on_error=False,
    )
    assert listener_pre_hook.name == hook_function.__name__
    assert listener_pre_hook.type == "listener"
    result = None
    async for item in listener_pre_hook.run(tool=tool, tool_args=tool_args):
        if not item.is_update:
            result = item
    assert isinstance(result, HookResult)
    assert result.name == listener_pre_hook.name
    assert result.emitted_content == expected["emitted_content"]
    assert result.success == expected["success"]
    if expected["error"] is None:
        assert result.error is None
    else:
        assert isinstance(result.error, HookError)
        assert result.error.error_code == expected["error"].error_code
    assert result.error == expected["error"]
    assert result.new_tool_args == expected["new_tool_args"]


def mock_json_response(status_code: int, json_response: dict) -> Callable:
    def mock_send(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code, json=json_response, request=request)

    return mock_send


listener_rest_pre_hooks_testdata = [
    (
        {"emitted_content": None},
        {"emitted_content": [], "success": True, "error": None},
    ),
    (
        {"emitted_content": None, "new_tool_args": None},
        {"emitted_content": [], "success": True, "error": None},
    ),
    (
        {"emitted_content": {"info": "test"}},
        {"emitted_content": [Part(metadata={"info": "test"})], "success": True, "error": None},
    ),
    (
        {"key": "value"},
        {"emitted_content": [Part(metadata={"key": "value"})], "success": True, "error": None},
    ),
    (
        {"key": 1234},
        {"emitted_content": [Part(metadata={"key": 1234})], "success": True, "error": None},
    ),
    (
        {"new_tool_args": {"param": "value"}},
        {
            "emitted_content": [],
            "success": False,
            "error": HookError(error_code="ValueError", error_message=""),
        },
    ),
    (
        {"emitted_content": {"info": "test"}, "new_tool_args": {"param": "value"}},
        {
            "emitted_content": [],
            "success": False,
            "error": HookError(error_code="ValueError", error_message=""),
        },
    ),
    (
        {"emitted_content": {"info": "test"}, "new_tool_args": "value"},
        {
            "emitted_content": [],
            "success": False,
            "error": HookError(error_code="ValidationError", error_message=""),
        },
    ),
    (
        {"error_code": 400, "error": "Bad Request"},
        {
            "emitted_content": [],
            "success": False,
            "error": HookError(error_code=400, error_message='{"error":"Bad Request"}'),
        },
    ),
]

listener_rest_post_hooks_testdata = [
    (
        {"emitted_content": None},
        {"emitted_content": [], "success": True, "error": None},
    ),
    (
        {"emitted_content": None, "new_tool_args": None},
        {"emitted_content": [], "success": True, "error": None},
    ),
    (
        {"emitted_content": {"info": "test"}},
        {"emitted_content": [Part(metadata={"info": "test"})], "success": True, "error": None},
    ),
    (
        {"key": "value"},
        {"emitted_content": [Part(metadata={"key": "value"})], "success": True, "error": None},
    ),
    (
        {"key": 1234},
        {"emitted_content": [Part(metadata={"key": 1234})], "success": True, "error": None},
    ),
    (
        {"new_result": {"param": "value"}},
        {
            "emitted_content": [],
            "success": False,
            "error": HookError(error_code="ValidationError", error_message=""),
        },
    ),
    (
        {"emitted_content": {"info": "test"}, "new_result": {"param": "value"}},
        {
            "emitted_content": [],
            "success": False,
            "error": HookError(error_code="ValidationError", error_message=""),
        },
    ),
    (
        {
            "emitted_content": {"info": "test"},
            "new_result": ToolResult(success=True).model_dump(mode="json"),
        },
        {
            "emitted_content": [],
            "success": False,
            "error": HookError(error_code="ValueError", error_message=""),
        },
    ),
    (
        {
            "emitted_content": None,
            "new_result": ToolResult(success=True, data={"param": "value"}).model_dump(mode="json"),
        },
        {
            "emitted_content": [],
            "success": False,
            "error": HookError(error_code="ValueError", error_message=""),
        },
    ),
    (
        {"error_code": 400, "error": "Bad Request"},
        {
            "emitted_content": [],
            "success": False,
            "error": HookError(error_code=400, error_message='{"error":"Bad Request"}'),
        },
    ),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("response, expected", listener_rest_pre_hooks_testdata)
async def test_listener_rest_function_tool_pre_hook(
    response: Any, expected: dict, tool: FunctionTool, tool_args: dict
) -> None:
    status_code = response.get("error_code", 200)
    response.pop("error_code", None)
    mock_send = mock_json_response(json_response=response, status_code=status_code)
    transport = httpx.MockTransport(mock_send)
    hook_name = "test_rest_listener_tool_hook"
    client = httpx.AsyncClient(transport=transport)
    restapi_data = HookRestAPIData(
        url="http://mocked.url",
        method="POST",
        headers={"Authorization": "Bearer"},
    )
    hook = FunctionHook(
        trigger=HookPhase.PRE_TOOL_CALL,
        name=hook_name,
        rest_api=restapi_data,
        fun=None,
        config_params=CONFIG_PARAMS,
        httpx_client=client,
        type=HookType.LISTENER,
        break_on_error=False,
    )
    assert hook.name == hook_name
    assert hook.type == "listener"

    executed = None
    async for item in hook.run(tool=tool, tool_args=tool_args):
        if not item.is_update:
            executed = item
    assert isinstance(executed, HookResult)
    assert executed.new_tool_args is None
    assert executed.emitted_content == expected.get("emitted_content")
    assert executed.success == expected.get("success")
    error = expected.get("error")
    if error is not None:
        assert isinstance(executed.error, HookError)
        assert executed.error.error_code == error.error_code
        if isinstance(executed.error.error_code, int):
            assert executed.error.error_message == error.error_message
    else:
        assert executed.error == error


updater_rest_pre_hooks_testdata = [
    (
        {"emitted_content": None},
        {
            "emitted_content": [],
            "success": False,
            "error": HookError(error_code="ValueError", error_message=""),
        },
    ),
    (
        {"emitted_content": None, "new_tool_args": None},
        {
            "emitted_content": [],
            "success": False,
            "error": HookError(error_code="ValueError", error_message=""),
        },
    ),
    (
        {"emitted_content": {"info": "test"}},
        {
            "emitted_content": [],
            "success": False,
            "error": HookError(error_code="ValueError", error_message=""),
        },
    ),
    (
        {"key": "value"},
        {
            "emitted_content": [],
            "success": False,
            "error": HookError(error_code="ValueError", error_message=""),
        },
    ),
    (
        {"key": 1234},
        {
            "emitted_content": [],
            "success": False,
            "error": HookError(error_code="ValueError", error_message=""),
        },
    ),
    (
        {"new_tool_args": {"param": "value"}},
        {
            "emitted_content": [],
            "success": True,
            "new_tool_args": {"param": "value"},
        },
    ),
    (
        {"emitted_content": {"info": "test"}, "new_tool_args": {"param": "value"}},
        {
            "emitted_content": [Part(metadata={"info": "test"})],
            "success": True,
            "new_tool_args": {"param": "value"},
        },
    ),
    (
        {"emitted_content": {"info": "test"}, "new_tool_args": "value"},
        {
            "emitted_content": [],
            "success": False,
            "error": HookError(error_code="ValidationError", error_message=""),
        },
    ),
    (
        {"error_code": 400, "error": "Bad Request"},
        {
            "emitted_content": [],
            "success": False,
            "error": HookError(error_code=400, error_message='{"error":"Bad Request"}'),
        },
    ),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("response, expected", updater_rest_pre_hooks_testdata)
async def test_updater_rest_function_tool_pre_hook(
    response: Any, expected: dict, tool: FunctionTool, tool_args: dict
) -> None:
    status_code = response.get("error_code", 200)
    response.pop("error_code", None)
    mock_send = mock_json_response(json_response=response, status_code=status_code)
    transport = httpx.MockTransport(mock_send)
    hook_name = "test_rest_listener_tool_hook"
    client = httpx.AsyncClient(transport=transport)
    restapi_data = HookRestAPIData(
        url="http://mocked.url",
        method="POST",
        headers={"Authorization": "Bearer"},
    )
    hook = FunctionHook(
        trigger=HookPhase.PRE_TOOL_CALL,
        name=hook_name,
        rest_api=restapi_data,
        fun=None,
        config_params=CONFIG_PARAMS,
        httpx_client=client,
        type=HookType.UPDATER,
        break_on_error=False,
    )
    assert hook.name == hook_name
    assert hook.type == "updater"

    executed = None
    async for item in hook.run(tool=tool, tool_args=tool_args):
        if not item.is_update:
            executed = item
    assert isinstance(executed, HookResult)
    assert executed.new_tool_args == expected.get("new_tool_args")
    assert executed.emitted_content == expected.get("emitted_content")
    assert executed.success == expected.get("success")
    error = expected.get("error")
    if error is not None:
        assert isinstance(executed.error, HookError)
        assert executed.error.error_code == error.error_code
        if isinstance(executed.error.error_code, int):
            assert executed.error.error_message == error.error_message
    else:
        assert executed.error == error


@pytest.mark.asyncio
@pytest.mark.parametrize("response, expected", listener_rest_post_hooks_testdata)
async def test_listener_rest_function_tool_post_hook(
    response: Any, expected: dict, tool: FunctionTool
) -> None:
    result = ToolResult(success=True)
    status_code = response.get("error_code", 200)
    response.pop("error_code", None)
    mock_send = mock_json_response(json_response=response, status_code=status_code)
    transport = httpx.MockTransport(mock_send)
    hook_name = "test_rest_listener_tool_hook"
    client = httpx.AsyncClient(transport=transport)
    restapi_data = HookRestAPIData(
        url="http://mocked.url",
        method="POST",
        headers={"Authorization": "Bearer"},
    )
    hook = FunctionHook(
        trigger=HookPhase.POST_TOOL_CALL,
        name=hook_name,
        rest_api=restapi_data,
        fun=None,
        config_params=CONFIG_PARAMS,
        httpx_client=client,
        type=HookType.LISTENER,
        break_on_error=False,
    )
    assert hook.name == hook_name
    assert hook.type == "listener"

    executed = None
    async for item in hook.run(tool=tool, result=result):
        if not item.is_update:
            executed = item
    assert isinstance(executed, HookResult)
    assert executed.new_result is None
    assert executed.emitted_content == expected.get("emitted_content")
    assert executed.success == expected.get("success")
    error = expected.get("error")
    if error is not None:
        assert isinstance(executed.error, HookError)
        assert executed.error.error_code == error.error_code
        if isinstance(executed.error.error_code, int):
            assert executed.error.error_message == error.error_message
    else:
        assert executed.error == error


updater_rest_post_hooks_testdata = [
    (
        {"emitted_content": None},
        {
            "emitted_content": [],
            "success": False,
            "error": HookError(error_code="ValueError", error_message=""),
        },
    ),
    (
        {"emitted_content": None, "new_tool_args": None},
        {
            "emitted_content": [],
            "success": False,
            "error": HookError(error_code="ValueError", error_message=""),
        },
    ),
    (
        {"emitted_content": {"info": "test"}},
        {
            "emitted_content": [],
            "success": False,
            "error": HookError(error_code="ValueError", error_message=""),
        },
    ),
    (
        {"key": "value"},
        {
            "emitted_content": [],
            "success": False,
            "error": HookError(error_code="ValueError", error_message=""),
        },
    ),
    (
        {"key": 1234},
        {
            "emitted_content": [],
            "success": False,
            "error": HookError(error_code="ValueError", error_message=""),
        },
    ),
    (
        {"new_result": {"param": "value"}},
        {
            "emitted_content": [],
            "success": False,
            "error": HookError(error_code="ValidationError", error_message=""),
        },
    ),
    (
        {"emitted_content": {"info": "test"}, "new_result": {"param": "value"}},
        {
            "emitted_content": [],
            "success": False,
            "error": HookError(error_code="ValidationError", error_message=""),
        },
    ),
    (
        {
            "emitted_content": {"info": "test"},
            "new_result": ToolResult(success=True, data={"param": "value"}).model_dump(mode="json"),
        },
        {
            "emitted_content": [Part(metadata={"info": "test"})],
            "success": True,
            "new_result": ToolResult(success=True, data={"param": "value"}),
        },
    ),
    (
        {
            "emitted_content": None,
            "new_result": ToolResult(success=True, data={"param": "value"}).model_dump(mode="json"),
        },
        {
            "emitted_content": [],
            "success": True,
            "new_result": ToolResult(success=True, data={"param": "value"}),
        },
    ),
    (
        {"error_code": 400, "error": "Bad Request"},
        {
            "emitted_content": [],
            "success": False,
            "error": HookError(error_code=400, error_message='{"error":"Bad Request"}'),
        },
    ),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("response, expected", updater_rest_post_hooks_testdata)
async def test_updater_rest_function_tool_post_hook(
    response: Any, expected: dict, tool: FunctionTool, tool_result: ToolResult
) -> None:
    status_code = response.get("error_code", 200)
    response.pop("error_code", None)
    mock_send = mock_json_response(json_response=response, status_code=status_code)
    transport = httpx.MockTransport(mock_send)
    hook_name = "test_rest_listener_tool_hook"
    client = httpx.AsyncClient(transport=transport)
    restapi_data = HookRestAPIData(
        url="http://mocked.url",
        method="POST",
        headers={"Authorization": "Bearer"},
    )
    hook = FunctionHook(
        trigger=HookPhase.POST_TOOL_CALL,
        name=hook_name,
        rest_api=restapi_data,
        fun=None,
        config_params=CONFIG_PARAMS,
        httpx_client=client,
        type=HookType.UPDATER,
        break_on_error=False,
    )
    assert hook.name == hook_name
    assert hook.type == "updater"

    executed = None
    async for item in hook.run(tool=tool, result=tool_result):
        if not item.is_update:
            executed = item
    assert isinstance(executed, HookResult)
    assert executed.new_result == expected.get("new_result")
    assert executed.emitted_content == expected.get("emitted_content")
    assert executed.success == expected.get("success")
    error = expected.get("error")
    if error is not None:
        assert isinstance(executed.error, HookError)
        assert executed.error.error_code == error.error_code
        if isinstance(executed.error.error_code, int):
            assert executed.error.error_message == error.error_message
    else:
        assert executed.error == error
