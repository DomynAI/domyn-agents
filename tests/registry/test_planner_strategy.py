"""Tests for PlannerStrategyRegistry."""

from unittest.mock import Mock

import pytest

from domyn.agents.planner_strategy import BasePlannerStrategy
from domyn.agents.planner_strategy.context_management.base import MessageContextManager
from domyn.agents.planner_strategy.prompting.messages_building import MessageBuilder
from domyn.agents.planner_strategy.prompting.structural_tags_prompting import (
    StructuralTagMessagePrompting,
)
from domyn.agents.registry import (
    PlannerStrategyRegistry,
    get_planner_strategy_registry,
)


class MockMessagePrompting(MessageContextManager):
    """Mock message prompting for testing."""

    type: str = "mock"
    prompting: MessageBuilder = StructuralTagMessagePrompting()

    def __init__(self):
        super().__init__(type="mock", name="mock_prompting")

    async def get_context(self, context, agent, use_stop, **kwargs):
        """Mock build_messages method."""
        return []


class MockPlannerStrategy(BasePlannerStrategy):
    """Mock planner strategy for testing."""

    name: str = "mock_strategy"

    def __init__(self, name: str = "test_strategy"):
        # Create a mock prompt builder
        mock_prompt_builder = MockMessagePrompting()
        super().__init__(name=name, context_manager=mock_prompt_builder)

    async def execute(self, agent, context):
        """Mock execute method."""

        # Return an async generator
        async def _mock_generator():
            yield Mock()

        return _mock_generator()

    async def parse_llm_response(self, response: str):
        """Mock parse_llm_response method."""
        return {"parsed": response}


class TestPlannerStrategyRegistry:
    """Test cases for PlannerStrategyRegistry."""

    def setup_method(self):
        """Setup for each test."""
        self.registry = PlannerStrategyRegistry()
        self.mock_strategy = MockPlannerStrategy()

    def test_init(self):
        """Test registry initialization."""
        assert self.registry._strategies == {}

    def test_register_planner_strategy(self):
        """Test registering a planner strategy."""
        self.registry.register_planner_strategy("test_strategy", self.mock_strategy)
        assert "test_strategy" in self.registry._strategies
        assert self.registry._strategies["test_strategy"] == self.mock_strategy

    def test_get_planner_strategy_direct_instance(self):
        """Test getting a planner strategy that's directly stored as instance."""
        self.registry.register_planner_strategy("test_strategy", self.mock_strategy)

        result = self.registry.get_planner_strategy("test_strategy")
        assert result == self.mock_strategy

    def test_get_planner_strategy_callable(self):
        """Test getting a planner strategy that's stored as callable."""
        mock_callable = Mock(return_value=self.mock_strategy)
        self.registry.register_planner_strategy("test_strategy", mock_callable)

        result = self.registry.get_planner_strategy("test_strategy")
        assert result == self.mock_strategy
        mock_callable.assert_called_once()

    def test_get_planner_strategy_nonexistent(self):
        """Test getting non-existent planner strategy raises TypeError."""
        # Based on implementation, this should raise TypeError when trying to call None()
        with pytest.raises(KeyError, match="Planner strategy 'nonexistent' not found in registry"):
            self.registry.get_planner_strategy("nonexistent")

    def test_list_planner_strategy_names(self):
        """Test listing planner strategy names."""
        strategy1 = MockPlannerStrategy(name="strategy1")
        strategy2 = MockPlannerStrategy(name="strategy2")

        self.registry.register_planner_strategy("strategy1", strategy1)
        self.registry.register_planner_strategy("strategy2", strategy2)

        names = self.registry.list_planner_strategy_names()
        assert set(names) == {"strategy1", "strategy2"}

    def test_list_planner_strategies(self):
        """Test listing planner strategies."""
        strategy1 = MockPlannerStrategy(name="strategy1")
        strategy2 = MockPlannerStrategy(name="strategy2")

        self.registry.register_planner_strategy("strategy1", strategy1)
        self.registry.register_planner_strategy("strategy2", strategy2)

        strategies = self.registry.list_planner_strategies()
        assert len(strategies) == 2
        assert strategy1 in strategies
        assert strategy2 in strategies

    def test_clear(self):
        """Test clearing all planner strategies."""
        self.registry.register_planner_strategy("test_strategy", self.mock_strategy)
        assert len(self.registry._strategies) == 1

        self.registry.clear()
        assert len(self.registry._strategies) == 0

    def test_setitem(self):
        """Test __setitem__ method."""
        self.registry["test_strategy"] = self.mock_strategy
        assert "test_strategy" in self.registry._strategies
        assert self.registry._strategies["test_strategy"] == self.mock_strategy

    def test_getitem_existing(self):
        """Test __getitem__ method with existing strategy."""
        self.registry.register_planner_strategy("test_strategy", self.mock_strategy)

        result = self.registry["test_strategy"]
        assert result == self.mock_strategy

    def test_getitem_nonexistent(self):
        """Test __getitem__ method with non-existent strategy raises KeyError."""
        # Based on the implementation, this should raise KeyError when trying to call None()
        with pytest.raises(KeyError, match="Planner strategy 'nonexistent' not found in registry"):
            _ = self.registry["nonexistent"]


class TestGetPlannerStrategyRegistry:
    """Test cases for get_planner_strategy_registry function."""

    def test_get_planner_strategy_registry_singleton(self):
        """Test that get_planner_strategy_registry returns the same instance."""
        registry1 = get_planner_strategy_registry()
        registry2 = get_planner_strategy_registry()

        assert registry1 is registry2
        assert isinstance(registry1, PlannerStrategyRegistry)

    def test_get_planner_strategy_registry_functionality(self):
        """Test that the global registry works correctly."""
        registry = get_planner_strategy_registry()
        mock_strategy = MockPlannerStrategy()

        # Clear any existing state
        registry.clear()

        registry.register_planner_strategy("global_test", mock_strategy)
        assert "global_test" in registry.list_planner_strategy_names()

        # Clean up
        registry.clear()
