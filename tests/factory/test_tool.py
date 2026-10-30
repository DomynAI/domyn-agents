import pytest

from domyn.agents.hooks import HookType
from domyn.agents.tools import Tool, ToolResult


def mock_tool(*args, **kwargs):
    pass


def mock_tool_with_config(key: str, *args, **kwargs):
    return key


def updater_pre_tool_hook(tool, tool_args) -> dict:
    return {"new_tool_args": tool_args}


def listener_post_tool_hook(tool, result: ToolResult) -> None:
    pass


hook_configs = [
    {
        "name": "updater_pre_tool_hook",
        "path": "tests/factory/test_tool.py",
        "type": "updater",
        "trigger": "pre_tool_call",
        "config_params": {"key": "value"},
    },
    {
        "name": "listener_post_tool_hook",
        "path": "tests/factory/test_tool.py",
        "type": "listener",
        "trigger": "post_tool_call",
        "config_params": {"key": "value"},
    },
]

function_tool_config = {
    "type": "function",
    "module": "tests.factory.test_tool",
    "name": "test_function",
    "path": "tests/factory/test_tool.py",
}

function_tool_config_with_hooks = {
    "type": "function",
    "path": "tests/factory/test_tool.py",
    "name": "test_function",
    "hooks": hook_configs,
}

tool_testdata = [
    {
        "name": "mock_tool",
        "module": "tests.factory.test_tool",
        "type": "function",
        "path": "tests/factory/test_tool.py",
        "hooks": hook_configs,
    },
    {
        "name": "mock_tool_with_config",
        "type": "function",
        "path": "tests/factory/test_tool.py",
        "config_params": [
            {
                "name": "key",
                "type": "str",
                "description": "API key for authentication",
                "value": "value",
            }
        ],
        "hooks": hook_configs,
    },
]


@pytest.mark.parametrize("config", tool_testdata)
def test_tool_factory(config):
    tool = Tool.from_config(config)
    assert tool.name == config["name"]
    assert tool.type == config["type"]
    assert tool.func["path"] == config["path"]
    config_params = config.get("config_params", [])
    for i, param in enumerate(config_params):
        assert param["name"] == tool.config_params[i].name
        assert param["type"] == tool.config_params[i].type
        assert param["description"] == tool.config_params[i].description
        assert param["value"] == tool.config_params[i].value
        # Check if the config value is correctly set in the tool's config dictionary
    pre_hook_cfgs = [h for h in config["hooks"] if h["trigger"] == "pre_tool_call"]
    post_hook_cfgs = [h for h in config["hooks"] if h["trigger"] == "post_tool_call"]
    assert len(tool.pre_hooks) == len(pre_hook_cfgs)
    assert len(tool.post_hooks) == len(post_hook_cfgs)
    for i, pre_hook in enumerate(pre_hook_cfgs):
        assert tool.pre_hooks[i].name == pre_hook["name"]
        assert tool.pre_hooks[i].type == HookType(pre_hook["type"])
        assert tool.pre_hooks[i].fun["path"] == pre_hook["path"]
        assert tool.pre_hooks[i].config_params == pre_hook.get("config_params")
    for i, post_hook in enumerate(post_hook_cfgs):
        assert tool.post_hooks[i].name == post_hook["name"]
        assert tool.post_hooks[i].type == HookType(post_hook["type"])
        assert tool.post_hooks[i].fun["path"] == post_hook["path"]
        assert tool.post_hooks[i].config_params == post_hook.get("config_params")
