import json

from domyn.agents.hooks import FunctionHook, HookPhase, HookType
from domyn.agents.tools import ToolResult


def listener_pre_tool_hook(tool, tool_args) -> None:
    pass


def updater_pre_tool_hook(tool, tool_args) -> dict:
    return {"new_tool_args": tool_args}


def listener_post_tool_hook(tool, result: ToolResult) -> None:
    pass


def updater_post_tool_hook(tool, result: ToolResult) -> ToolResult:
    return result


CONFIG_PARAMS = {"param1": "value1", "param2": 2.0, "param3": True}


def test_listener_function_tool_pre_hook_serialization():
    original_hook = FunctionHook(
        fun=listener_pre_tool_hook,
        config_params=CONFIG_PARAMS,
        trigger=HookPhase.PRE_TOOL_CALL,
        type=HookType.LISTENER,
    )
    serialized = original_hook.model_dump_json()
    loaded_hook = FunctionHook(**json.loads(serialized))
    assert loaded_hook.model_dump() == original_hook.model_dump()


def test_updater_function_tool_pre_hook_serialization():
    original_hook = FunctionHook(
        fun=updater_pre_tool_hook,
        config_params=CONFIG_PARAMS,
        trigger=HookPhase.PRE_TOOL_CALL,
        type=HookType.UPDATER,
    )
    serialized = original_hook.model_dump_json()
    loaded_hook = FunctionHook(**json.loads(serialized))
    assert loaded_hook.model_dump() == original_hook.model_dump()


def test_listener_function_tool_post_hook_serialization():
    original_hook = FunctionHook(
        fun=listener_post_tool_hook,
        config_params=CONFIG_PARAMS,
        trigger=HookPhase.POST_TOOL_CALL,
        type=HookType.LISTENER,
    )
    serialized = original_hook.model_dump_json()
    loaded_hook = FunctionHook(**json.loads(serialized))
    assert loaded_hook.model_dump() == original_hook.model_dump()


def test_updater_function_tool_post_hook_serialization():
    original_hook = FunctionHook(
        fun=updater_post_tool_hook,
        config_params=CONFIG_PARAMS,
        trigger=HookPhase.POST_TOOL_CALL,
        type=HookType.UPDATER,
    )
    serialized = original_hook.model_dump_json()
    loaded_hook = FunctionHook(**json.loads(serialized))
    assert loaded_hook.model_dump() == original_hook.model_dump()
