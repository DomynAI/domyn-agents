"""Agent entries under `agents`: a missing name, sub-agents named from the registry."""

import pytest

from domyn.agents.agents import WrappedAgent
from domyn.agents.core.decorators import agent
from domyn.agents.loader import ConfigurationV1
from domyn.agents.registry import get_agent_registry

_LLM = {
    "name": "test-provider",
    "type": "vllm",
    "model_name": "test-model",
    "url": "http://localhost:8000/v1/chat/completions",
}


def _agent(**fields) -> dict:
    return {
        "description": "Test agent",
        "instruction": "You are a test agent",
        "llm_provider": "test-provider",
        **fields,
    }


@pytest.mark.parametrize(
    "agents",
    [
        [_agent()],
        # The sub-agent ordering reads every name before an agent is validated.
        [_agent(sub_agents=["Helper"]), _agent(name="Helper")],
    ],
    ids=["alone", "with_sub_agents"],
)
def test_an_agent_without_a_name_is_refused(agents):
    with pytest.raises(ValueError, match=r"Agent #1 under 'agents' has no 'name'"):
        ConfigurationV1.from_dict({"version": "1.0", "llm_providers": [_LLM], "agents": agents})


def test_a_sub_agent_named_from_the_registry_can_be_a_wrapped_agent():
    """A function registered with ``@agent`` is a WrappedAgent; naming it used to fail validation."""

    @agent(name="WrappedHelper")
    async def helper(context): ...

    try:
        config = ConfigurationV1.from_dict(
            {
                "version": "1.0",
                "llm_providers": [_LLM],
                "agents": [_agent(name="Root", sub_agents=["WrappedHelper"])],
            }
        )
        (sub,) = config.agents[0].sub_agents
        assert isinstance(sub, WrappedAgent)
        assert sub is get_agent_registry().get_agent("WrappedHelper")
    finally:
        get_agent_registry()._agents.pop("WrappedHelper", None)
