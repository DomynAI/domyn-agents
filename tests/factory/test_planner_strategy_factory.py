import pytest

from domyn.agents.planner_strategy import PlannerStrategyFactory

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

planner_strategies_testdata = [
    {
        "name": "basic",
        "type": "react",
        "max_iterations": 10,
        "verbose": True,
        "use_stop": False,
        "context_manager": context_manager,
    }
]


@pytest.mark.parametrize("config", planner_strategies_testdata)
def test_planner_strategy_factory(config):
    strategy = PlannerStrategyFactory.from_config(config)
    assert strategy.type == config["type"]
    assert strategy.name == config["name"]
    assert strategy.max_iterations == config["max_iterations"]
    assert strategy.verbose == config["verbose"]
    assert strategy.use_stop == config["use_stop"]


class TestTheCompactorField:
    """`compactor` is resolved the same way `context_manager` is: through the
    factory, as part of normal config loading, not by mutating the planner
    after the fact."""

    def test_an_inline_dict_resolves_to_the_named_strategy(self):
        strategy = PlannerStrategyFactory.from_config(
            {
                "name": "tool_use_thought",
                "type": "tool_use_thought",
                "compactor": {"strategy": "prune_only"},
            }
        )
        assert strategy.compactor.strategy_name == "prune_only"

    def test_omitted_keeps_the_framework_default(self):
        strategy = PlannerStrategyFactory.from_config({"name": "tool_use", "type": "tool_use"})
        assert strategy.compactor.strategy_name == "two_phase"

    def test_tool_use_also_resolves_it(self):
        strategy = PlannerStrategyFactory.from_config(
            {
                "name": "tool_use",
                "type": "tool_use",
                "compactor": {"strategy": "noop"},
            }
        )
        assert strategy.compactor.active is False

    def test_react_rejects_a_compactor(self):
        """React's execute() loop never reads a compactor — accepting the key
        there would silently do nothing, which is worse than refusing it."""
        config = dict(planner_strategies_testdata[0])
        config["compactor"] = {"strategy": "prune_only"}
        with pytest.raises(ValueError, match="compactor"):
            PlannerStrategyFactory.from_config(config)
