from domyn.agents.loader import ConfigurationV1
from domyn.agents.planner_strategy import ToolUseWithThoughtPlannerStrategy


def _config_dict(planner_strategy: dict) -> dict:
    return {
        "version": "1.0",
        "entrypoint": "TestAgent",
        "llm_providers": [
            {
                "name": "test-provider",
                "type": "vllm",
                "model_name": "test-model",
                "url": "http://test.com",
            }
        ],
        "planner_strategies": [planner_strategy],
        "agents": [
            {
                "name": "TestAgent",
                "description": "Test agent",
                "instruction": "Test",
                "llm_provider": "test-provider",
                "planner_strategy": planner_strategy["name"],
            }
        ],
    }


def test_an_inline_compactor_reaches_the_agents_planner():
    """`compactor` resolves through normal `ConfigurationV1` loading, the same
    way `context_manager` does — no post-load override needed."""
    config_dict = _config_dict(
        {
            "name": "tool_use_thought",
            "type": "tool_use_thought",
            "compactor": {"strategy": "prune_only"},
        }
    )

    config = ConfigurationV1.from_dict(config_dict)
    agent = config.agents[0]

    assert isinstance(agent.planner, ToolUseWithThoughtPlannerStrategy)
    assert agent.planner.compactor.strategy_name == "prune_only"


def test_an_omitted_compactor_keeps_the_framework_default():
    config_dict = _config_dict({"name": "tool_use_thought", "type": "tool_use_thought"})

    config = ConfigurationV1.from_dict(config_dict)
    agent = config.agents[0]

    assert agent.planner.compactor.strategy_name == "two_phase"
