import pytest

from domyn.agents.agents import AgentFactory, ContextVisibility, HandoffScope
from domyn.agents.tools import ToolResult


def pre_agent_hook(context, agent):
    pass


def post_agent_hook(context, agent):
    pass


def listener_pre_tool_hook(tool, tool_args) -> None:
    pass


def updater_pre_tool_hook(tool, tool_args) -> dict:
    return {"new_tool_args": tool_args}


def listener_post_tool_hook(tool, result: ToolResult) -> None:
    pass


def updater_post_tool_hook(tool, result: ToolResult) -> ToolResult:
    return result


tool_hook_configs = [
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

agent_hooks = [
    {
        "name": "pre_agent_hook",
        "path": "tests/factory/test_agent_factory.py",
        "type": "listener",
        "trigger": "pre_agent_call",
        "config_params": {"key": "value"},
    },
    {
        "name": "post_agent_hook",
        "path": "tests/factory/test_agent_factory.py",
        "type": "updater",
        "trigger": "post_agent_call",
        "config_params": {"key": "value"},
    },
]

tools = [
    {
        "name": "mock_tool",
        "module": "tests.factory.test_tool",
        "type": "function",
        "path": "tests/factory/test_tool.py",
        "hooks": tool_hook_configs,
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
        "hooks": tool_hook_configs,
    },
]

llm_provider = {
    "name": "qwen3",
    "type": "vllm",
    "model_name": "Qwen/Qwen3-Next-80B-A3B-Instruct-FP8",
    "url": "http://mock_url/chat/completions",
}

context_manager = {
    "name": "summary",
    "type": "agents_summary",
    "min_events_for_summary": 5,
    "prompting": {"type": "structural_tags"},
    "llm_provider": llm_provider,
}

planner_strategy = {
    "name": "basic",
    "type": "react",
    "max_iterations": 10,
    "verbose": True,
    "use_stop": False,
    "context_manager": context_manager,
}

agent_testdata = [
    {
        "name": "MockAgent",
        "description": "A mock description",
        "instruction": "A mock instruction",
        "context_visibility": "standard",
        "llm_provider": llm_provider,
        "tools": tools,
        "planner_strategy": planner_strategy,
        "hooks": agent_hooks,
    }
]


@pytest.mark.parametrize("config", agent_testdata)
def test_planner_strategy_factory(config):
    agent = AgentFactory.from_config(config, save_in_registry=False)
    assert agent.name == config["name"]
    assert agent.description == config["description"]
    assert agent.instruction == config["instruction"]
    assert agent.context_visibility == ContextVisibility(config["context_visibility"])
    assert agent.llm_provider.name == config["llm_provider"]["name"]
    assert len(agent.tools) == len(config["tools"])
    assert agent.planner.name == config["planner_strategy"]["name"]

    pre_hook_count = sum(1 for h in config["hooks"] if h["trigger"] == "pre_agent_call")
    post_hook_count = sum(1 for h in config["hooks"] if h["trigger"] == "post_agent_call")
    assert len(agent.pre_hooks) == pre_hook_count
    assert len(agent.post_hooks) == post_hook_count
    for tool, tool_config in zip(agent.tools, config["tools"], strict=False):
        assert tool.name == tool_config["name"]
        tool_pre_count = sum(1 for h in tool_config["hooks"] if h["trigger"] == "pre_tool_call")
        tool_post_count = sum(1 for h in tool_config["hooks"] if h["trigger"] == "post_tool_call")
        assert len(tool.pre_hooks) == tool_pre_count
        assert len(tool.post_hooks) == tool_post_count
        if "config_params" in tool_config:
            for i, param in enumerate(tool_config["config_params"]):
                assert tool.config_params[i].name == param["name"]
                assert tool.config_params[i].value == param["value"]
                assert tool.config_params[i].type == param["type"]
                assert tool.config_params[i].description == param["description"]


@pytest.mark.parametrize(
    "legacy, expected_cv, expected_hs",
    [
        ("global", ContextVisibility.GLOBAL, HandoffScope.GLOBAL),
        ("peers", ContextVisibility.PEERS, HandoffScope.DIRECT_PEERS),
        ("standard", ContextVisibility.STANDARD, HandoffScope.EXPLICIT),
    ],
)
def test_legacy_handoff_visibility_is_translated(legacy, expected_cv, expected_hs):
    config = {
        "name": "LegacyAgent",
        "description": "d",
        "instruction": "i",
        "llm_provider": llm_provider,
        "planner_strategy": planner_strategy,
        "handoff_visibility": legacy,
    }
    agent = AgentFactory.from_config(config, save_in_registry=False)
    assert agent.context_visibility == expected_cv
    assert agent.handoff_scope == expected_hs


def test_explicit_new_keys_win_over_legacy():
    config = {
        "name": "LegacyAgent",
        "description": "d",
        "instruction": "i",
        "llm_provider": llm_provider,
        "planner_strategy": planner_strategy,
        "handoff_visibility": "global",
        "context_visibility": "standard",
        "handoff_scope": "explicit",
    }
    agent = AgentFactory.from_config(config, save_in_registry=False)
    assert agent.context_visibility == ContextVisibility.STANDARD
    assert agent.handoff_scope == HandoffScope.EXPLICIT
