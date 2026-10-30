"""``get_system_description_from_root``: the snapshot of an agent tree hosts store with a
configuration (a host keeps one per configuration it runs)."""

import pytest
from pydantic import ValidationError

from domyn.agents import Agent, FunctionTool
from domyn.agents.core.utils import (
    SystemAgentDescription,
    SystemDescription,
    get_system_description_from_root,
)
from domyn.agents.llm import VLLMProvider

_PROVIDER = VLLMProvider(name="p", model_name="m", url="http://mock_url/v1/chat/completions")


def search(query: str) -> str:
    """Search the catalogue."""
    return query


def refund(order_id: str, amount: float) -> str:
    """Refund an order."""
    return order_id


def _agent(name: str, tools=(), sub_agents=()) -> Agent:
    return Agent(
        name=name,
        description=f"{name} agent",
        instruction="i",
        llm_provider=_PROVIDER,
        tools=[FunctionTool(func=t) for t in tools],
        sub_agents=list(sub_agents),
    )


class TestDescribingATree:
    def test_every_agent_and_tool_of_the_tree_is_described(self):
        billing = _agent("Billing", tools=[refund])
        root = _agent("Root", tools=[search], sub_agents=[billing])

        description = get_system_description_from_root(root)

        assert description.agents["Root"] == SystemAgentDescription(
            name="Root", description="Root agent", tools=["search"], subagents=["Billing"]
        )
        assert description.agents["Billing"].tools == ["refund"]
        assert set(description.tools) == {"search", "refund"}
        assert description.tools["refund"].description == "Refund an order."
        assert set(description.tools["refund"].parameters) == {"order_id", "amount"}

    def test_an_agent_reached_twice_is_described_once(self):
        shared = _agent("Shared")
        root = _agent("Root", sub_agents=[_agent("A", sub_agents=[shared]), shared])

        description = get_system_description_from_root(root)

        assert sorted(description.agents) == ["A", "Root", "Shared"]

    def test_the_stored_format_is_plain_data(self):
        root = _agent("Root", tools=[search])
        description = get_system_description_from_root(root)

        dumped = description.model_dump()

        assert set(dumped) == {"agents", "tools"}
        assert set(dumped["agents"]["Root"]) == {"name", "description", "tools", "subagents"}
        assert set(dumped["tools"]["search"]) == {"name", "description", "parameters"}
        assert SystemDescription.model_validate(dumped) == description


class TestTheSnapshotIsConsistent:
    def test_a_subagent_that_is_not_described_is_refused(self):
        with pytest.raises(ValidationError, match="Subagent 'Ghost'"):
            SystemDescription(
                agents={
                    "Root": SystemAgentDescription(
                        name="Root", description="d", tools=[], subagents=["Ghost"]
                    )
                },
                tools={},
            )

    def test_a_tool_that_is_not_described_is_refused(self):
        with pytest.raises(ValidationError, match="Tool 'missing'"):
            SystemDescription(
                agents={
                    "Root": SystemAgentDescription(
                        name="Root", description="d", tools=["missing"], subagents=[]
                    )
                },
                tools={},
            )
