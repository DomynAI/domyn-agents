from typing import Any

from domyn.agents.planner_strategy import BasePlannerStrategy


class PlannerStrategyRegistry:
    """Registry for planner strategy instances."""

    def __init__(self):
        self._strategies: dict[str, BasePlannerStrategy] = {}

    def register_planner_strategy(self, name: str, strategy: BasePlannerStrategy | Any) -> None:

        self._strategies[name] = strategy

    def get_planner_strategy(self, name: str) -> BasePlannerStrategy:

        _c = self._strategies.get(name)
        if _c is None:
            raise KeyError(f"Planner strategy '{name}' not found in registry.")
        if isinstance(_c, BasePlannerStrategy):
            return _c
        else:
            return _c()

    def list_planner_strategy_names(self) -> list[str]:
        return list(self._strategies.keys())

    def list_planner_strategies(self) -> list[BasePlannerStrategy]:
        return list(self._strategies.values())

    def clear(self) -> None:
        """Clear all registered planner strategies."""
        self._strategies.clear()

    def __setitem__(self, name: str, strategy: BasePlannerStrategy) -> None:
        self.register_planner_strategy(name, strategy)

    def __getitem__(self, name: str) -> BasePlannerStrategy:
        return self.get_planner_strategy(name)


_planner_strategy_registry = PlannerStrategyRegistry()


def get_planner_strategy_registry() -> PlannerStrategyRegistry:
    return _planner_strategy_registry
