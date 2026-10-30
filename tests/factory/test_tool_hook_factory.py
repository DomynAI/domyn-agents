import pytest

from domyn.agents.hooks import HookFactory, HookPhase
from domyn.agents.registry import get_hook_registry
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

pre_hooks_testdata = [
    {
        "name": "listener_pre_tool_hook",
        "path": "tests/factory/test_tool_hook_factory.py",
        "type": "listener",
        "config_params": CONFIG_PARAMS,
    },
    {
        "name": "updater_pre_tool_hook",
        "path": "tests/factory/test_tool_hook_factory.py",
        "type": "listener",
        "config_params": CONFIG_PARAMS,
    },
    {
        "name": "mock_rest_listener",
        "url": "https://api.mockservice.com/agent_hook",
        "type": "listener",
        "method": "POST",
        "headers": {"Authorization": "Bearer mocktoken"},
        "config_params": CONFIG_PARAMS,
    },
    {
        "name": "mock_rest_listener",
        "url": "https://api.mockservice.com/agent_hook",
        "type": "updater",
        "method": "POST",
        "headers": {"Authorization": "Bearer mocktoken"},
        "config_params": CONFIG_PARAMS,
    },
]

post_hooks_testdata = [
    {
        "name": "listener_post_tool_hook",
        "path": "tests/factory/test_tool_hook_factory.py",
        "type": "updater",
        "config_params": CONFIG_PARAMS,
    },
    {
        "name": "updater_post_tool_hook",
        "path": "tests/factory/test_tool_hook_factory.py",
        "type": "updater",
        "config_params": CONFIG_PARAMS,
    },
    {
        "name": "mock_rest_listener",
        "url": "https://api.mockservice.com/agent_hook",
        "type": "listener",
        "method": "POST",
        "headers": {"Authorization": "Bearer mocktoken"},
        "config_params": CONFIG_PARAMS,
    },
    {
        "name": "mock_rest_listener",
        "url": "https://api.mockservice.com/agent_hook",
        "type": "updater",
        "method": "POST",
        "headers": {"Authorization": "Bearer mocktoken"},
        "config_params": CONFIG_PARAMS,
    },
]


@pytest.mark.parametrize("config", pre_hooks_testdata)
def test_tool_pre_hook_factory(config: dict):
    result = HookFactory.from_config(config, trigger=HookPhase.PRE_TOOL_CALL)
    assert result.name == config["name"]
    assert result.type == config["type"]
    if "path" in config:
        assert result.fun["path"] == config["path"]
    elif "url" in config:
        assert result.rest_api.url == config["url"]
        assert result.rest_api.method == config["method"]
        assert result.rest_api.headers == config["headers"]
    assert result.config_params == config["config_params"]


@pytest.mark.parametrize("config", post_hooks_testdata)
def test_tool_post_hook_factory(config: dict):
    result = HookFactory.from_config(config, trigger=HookPhase.POST_TOOL_CALL)
    assert result.name == config["name"]
    assert result.type == config["type"]
    if "path" in config:
        assert result.fun["path"] == config["path"]
    elif "url" in config:
        assert result.rest_api.url == config["url"]
        assert result.rest_api.method == config["method"]
        assert result.rest_api.headers == config["headers"]
    assert result.config_params == config["config_params"]


# A hook's phase reaches the factory either in its config (as YAML writes it) or as
# an argument (as an agent's tool list passes it); naming must not depend on which.
_PHASE_IN_CONFIG = ({"trigger": "pre_agent_call"}, {})
_PHASE_AS_ARGUMENT = ({}, {"trigger": HookPhase.PRE_TOOL_CALL})


@pytest.mark.parametrize(
    ("phase_config", "phase_kwargs"),
    [_PHASE_IN_CONFIG, _PHASE_AS_ARGUMENT],
    ids=["phase-in-config", "phase-as-argument"],
)
class TestHookFactoryNameResolution:
    """Hook name resolution: the config's name, the registry key, references and aliases."""

    def setup_method(self):
        get_hook_registry().clear()

    def teardown_method(self):
        get_hook_registry().clear()

    def _rest_hook(self, phase_config, phase_kwargs, **config):
        return HookFactory.from_config(
            {"url": "https://example.com/hook", "type": "listener", **config, **phase_config},
            **phase_kwargs,
        )

    def test_a_rest_hook_takes_the_name_in_its_config(self, phase_config, phase_kwargs):
        hook = self._rest_hook(phase_config, phase_kwargs, name="my_rest_hook")
        assert hook.name == "my_rest_hook"

    def test_the_registry_keys_it_by_that_name_only(self, phase_config, phase_kwargs):
        hook = HookFactory.from_config(
            {
                "name": "registered_hook",
                "url": "https://example.com/hook",
                "type": "updater",
                **phase_config,
            },
            save_in_registry=True,
            **phase_kwargs,
        )
        registry = get_hook_registry()
        assert registry.list_hook_names() == ["registered_hook"]
        assert registry.get_hook("registered_hook") is hook

    def test_a_reference_with_another_name_is_an_aliased_copy(self, phase_config, phase_kwargs):
        HookFactory.from_config(
            {
                "name": "original",
                "url": "https://example.com/hook",
                "type": "updater",
                **phase_config,
            },
            save_in_registry=True,
            **phase_kwargs,
        )
        aliased = HookFactory.from_config({"reference": "original", "name": "my_alias"})

        assert aliased.name == "my_alias"
        assert get_hook_registry().get_hook("original").name == "original"

    def test_a_reference_without_a_name_keeps_the_original_one(self, phase_config, phase_kwargs):
        HookFactory.from_config(
            {
                "name": "original",
                "url": "https://example.com/hook",
                "type": "listener",
                **phase_config,
            },
            save_in_registry=True,
            **phase_kwargs,
        )
        hook = HookFactory.from_config({"reference": "original"})
        assert hook.name == "original"
