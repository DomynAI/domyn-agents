"""Agents package for domyn agents framework."""

from typing import Any

from domyn.agents.agents.agent import Agent
from domyn.agents.agents.base import ContextVisibility, HandoffScope
from domyn.agents.agents.delegate_agent import DelegateAgent
from domyn.agents.agents.wrapped import WrappedAgent
from domyn.agents.core.interfaces import BaseAgent


class AgentFactory:
    @staticmethod
    def from_config(config: dict[str, Any], save_in_registry: bool) -> BaseAgent:
        """Create an agent from configuration.

        Agent classes self-register by their ``type`` via ``BaseAgent.__init_subclass__``,
        so any imported ``BaseAgent`` subclass (e.g. ``DelegateAgent``) is resolvable here.
        """
        agent_type = config.get("type")
        if agent_type in (None, "agent"):
            # No type / legacy lowercase alias -> the default rich Agent.
            agent_cls: type[BaseAgent] = Agent
        else:
            agent_cls = BaseAgent._registry.get(agent_type)
            if agent_cls is None:
                raise ValueError(
                    f"Unknown agent type: {agent_type!r}. Define a BaseAgent subclass with "
                    "this `type` and import it so it self-registers (see DelegateAgent)."
                )
        obj = agent_cls.from_config(config, save_in_registry=save_in_registry)

        if save_in_registry:
            from domyn.agents.registry import get_agent_registry

            get_agent_registry().register_agent(obj.name, obj)

        return obj


__all__ = [
    "Agent",
    "AgentFactory",
    "ContextVisibility",
    "DelegateAgent",
    "HandoffScope",
    "WrappedAgent",
]
