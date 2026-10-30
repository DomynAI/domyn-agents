import pytest

from domyn.agents.hooks import HookFactory as AgentHookFactory


def print_start_agent_name(context, agent):
    pass


def print_end_agent_name(context, agent):
    pass


CONFIG_PARAMS = {"key": "value"}

config_testdata = [
    {
        "name": "print_start_agent_name",
        "path": "tests/factory/test_agent_hook_factory.py",
        "type": "listener",
        "trigger": "pre_agent_call",
        "config_params": CONFIG_PARAMS,
    },
    {
        "name": "print_end_agent_name",
        "path": "tests/factory/test_agent_hook_factory.py",
        "type": "updater",
        "trigger": "post_agent_call",
        "config_params": CONFIG_PARAMS,
    },
    {
        "name": "mock_rest_listener",
        "url": "https://api.mockservice.com/agent_hook",
        "type": "listener",
        "trigger": "pre_agent_call",
        "method": "POST",
        "headers": {"Authorization": "Bearer mocktoken"},
        "config_params": CONFIG_PARAMS,
    },
    {
        "name": "mock_rest_listener",
        "url": "https://api.mockservice.com/agent_hook",
        "type": "updater",
        "trigger": "post_agent_call",
        "method": "POST",
        "headers": {"Authorization": "Bearer mocktoken"},
        "config_params": CONFIG_PARAMS,
    },
]


@pytest.mark.parametrize("config", config_testdata)
def test_agent_hook_factory(config: dict):
    result = AgentHookFactory.from_config(config)
    assert result.name == config["name"]
    assert result.type == config["type"]
    assert result.config_params == CONFIG_PARAMS
