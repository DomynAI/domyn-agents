from typing import Any

from domyn.agents.core import BaseAgent


class AgentRegistry:
    """Registry for agent instances."""

    def __init__(self):
        self._agents: dict[str, BaseAgent] = {}

    def register_agent(self, name: str, agent: BaseAgent | Any) -> None:

        self._agents[name] = agent

    def get_agent(self, name: str) -> BaseAgent:

        from domyn.agents.agents import Agent, WrappedAgent

        _c = self._agents.get(name)
        if not _c:
            raise KeyError(f"Agent '{name}' not found in registry.")
        if isinstance(_c, Agent | WrappedAgent):
            return _c
        else:
            return _c()

    def list_agent_names(self) -> list[str]:
        return list(self._agents.keys())

    def list_agents(self) -> list[BaseAgent]:
        return list(self._agents.values())

    def clear(self) -> None:
        self._agents.clear()

    def __setitem__(self, name: str, agent: BaseAgent) -> None:
        self.register_agent(name, agent)

    def __getitem__(self, name: str) -> BaseAgent:
        return self.get_agent(name)


_agents_registry = AgentRegistry()


def get_agent_registry() -> AgentRegistry:
    return _agents_registry
