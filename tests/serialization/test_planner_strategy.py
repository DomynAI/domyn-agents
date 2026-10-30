import json

import pytest

from domyn.agents import VLLMProvider
from domyn.agents.planner_strategy import (
    ContextManager,
    PlannerStrategyFactory,
    ReactPlannerStrategy,
)
from domyn.agents.planner_strategy.context_management.full_messages import FullMessageContext
from domyn.agents.planner_strategy.context_management.summary import PreviousAgentsSummaryContext
from domyn.agents.planner_strategy.prompting.json_prompting import JSONMessagePrompting
from domyn.agents.planner_strategy.prompting.structural_tags_prompting import (
    StructuralTagMessagePrompting,
)

llm_provider = VLLMProvider(
    model_name="test-model",
    generation_params={"max_length": 1000},
    url="https://llm.example.com/v1/chat/completions",
    type="vllm",
)

react_testdata = [
    (
        "full_stop_structural",
        FullMessageContext(
            narrow_non_visible_agents=True,
            prompting=StructuralTagMessagePrompting(),
            name="full_message_context",
        ),
        True,
    ),
    (
        "full_nostop_structural",
        FullMessageContext(
            narrow_non_visible_agents=True,
            prompting=StructuralTagMessagePrompting(),
            name="full_message_context",
        ),
        False,
    ),
    (
        "summary_stop_structural",
        PreviousAgentsSummaryContext(
            llm_provider=llm_provider,
            prompting=StructuralTagMessagePrompting(),
            name="full_message_context",
        ),
        True,
    ),
    (
        "full_stop_json",
        FullMessageContext(
            narrow_non_visible_agents=True,
            prompting=JSONMessagePrompting(),
            name="full_message_context",
        ),
        True,
    ),
    (
        "full_nostop_json",
        FullMessageContext(
            narrow_non_visible_agents=True,
            prompting=JSONMessagePrompting(),
            name="full_message_context",
        ),
        False,
    ),
    (
        "summary_stop_json",
        PreviousAgentsSummaryContext(
            llm_provider=llm_provider,
            prompting=JSONMessagePrompting(),
            name="full_message_context",
        ),
        True,
    ),
]


@pytest.mark.parametrize("name,prompting,stop", react_testdata)
def test_planner_strategy_serialization(name: str, prompting: ContextManager, stop: bool):
    original = ReactPlannerStrategy(name=name, context_manager=prompting, use_stop=stop)
    serialized = original.model_dump_json()
    loaded = PlannerStrategyFactory.from_config(config=json.loads(serialized))
    assert loaded.model_dump() == original.model_dump()
