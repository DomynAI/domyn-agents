"""Tests for AgentRegistry."""

from unittest.mock import Mock

import pytest

from domyn.agents.core import BaseAgent
from domyn.agents.registry import AgentRegistry, get_agent_registry


class MockAgent(BaseAgent):
    """Mock agent for testing."""

    def __init__(self, name: str = "test_agent", description: str = "Test agent"):
        super().__init__(name=name, description=description)


class TestAgentRegistry:
    """Test cases for AgentRegistry."""

    def setup_method(self):
        """Setup for each test."""
        self.registry = AgentRegistry()
        self.mock_agent = MockAgent()

    def test_init(self):
        """Test registry initialization."""
        assert self.registry._agents == {}

    def test_register_agent(self):
        """Test registering an agent."""
        self.registry.register_agent("test_agent", self.mock_agent)
        assert "test_agent" in self.registry._agents
        assert self.registry._agents["test_agent"] == self.mock_agent

    def test_get_agent_direct_instance(self):
        """Test getting an agent that's directly stored as instance."""
        from domyn.agents.agents import Agent

        # Create a mock Agent instance
        mock_agent_instance = Mock(spec=Agent)
        self.registry.register_agent("test_agent", mock_agent_instance)

        result = self.registry.get_agent("test_agent")
        assert result == mock_agent_instance

    def test_get_agent_callable(self):
        """Test getting an agent that's stored as callable."""
        mock_callable = Mock(return_value=self.mock_agent)
        self.registry.register_agent("test_agent", mock_callable)

        result = self.registry.get_agent("test_agent")
        assert result == self.mock_agent
        mock_callable.assert_called_once()

    def test_get_agent_nonexistent(self):
        """Test getting non-existent agent raises KeyError."""
        # Based on implementation, this should raise KeyError when trying to call None()
        with pytest.raises(KeyError, match="Agent 'nonexistent' not found in registry"):
            self.registry.get_agent("nonexistent")

    def test_list_agent_names(self):
        """Test listing agent names."""
        self.registry.register_agent("agent1", MockAgent(name="agent1"))
        self.registry.register_agent("agent2", MockAgent(name="agent2"))

        names = self.registry.list_agent_names()
        assert set(names) == {"agent1", "agent2"}

    def test_list_agents(self):
        """Test listing agents."""
        agent1 = MockAgent(name="agent1")
        agent2 = MockAgent(name="agent2")

        self.registry.register_agent("agent1", agent1)
        self.registry.register_agent("agent2", agent2)

        agents = self.registry.list_agents()
        assert len(agents) == 2
        assert agent1 in agents
        assert agent2 in agents

    def test_clear(self):
        """Test clearing all agents."""
        self.registry.register_agent("test_agent", self.mock_agent)
        assert len(self.registry._agents) == 1

        self.registry.clear()
        assert len(self.registry._agents) == 0

    def test_setitem(self):
        """Test __setitem__ method."""
        self.registry["test_agent"] = self.mock_agent
        assert "test_agent" in self.registry._agents
        assert self.registry._agents["test_agent"] == self.mock_agent

    def test_getitem_existing(self):
        """Test __getitem__ method with existing agent."""
        from domyn.agents.agents import Agent

        mock_agent_instance = Mock(spec=Agent)
        self.registry.register_agent("test_agent", mock_agent_instance)

        result = self.registry["test_agent"]
        assert result == mock_agent_instance

    def test_getitem_nonexistent(self):
        """Test __getitem__ method with non-existent agent raises KeyError."""
        # Based on the implementation, this should raise KeyError when trying to call None()
        with pytest.raises(KeyError, match="Agent 'nonexistent' not found in registry"):
            _ = self.registry["nonexistent"]


class TestGetAgentRegistry:
    """Test cases for get_agent_registry function."""

    def test_get_agent_registry_singleton(self):
        """Test that get_agent_registry returns the same instance."""
        registry1 = get_agent_registry()
        registry2 = get_agent_registry()

        assert registry1 is registry2
        assert isinstance(registry1, AgentRegistry)

    def test_get_agent_registry_functionality(self):
        """Test that the global registry works correctly."""
        registry = get_agent_registry()
        mock_agent = MockAgent()

        # Clear any existing state
        registry.clear()

        registry.register_agent("global_test", mock_agent)
        assert "global_test" in registry.list_agent_names()

        # Clean up
        registry.clear()
