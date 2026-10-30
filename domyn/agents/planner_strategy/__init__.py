"""Planner strategies, context managers and their factories."""

import copy
from enum import StrEnum
from importlib import import_module
from typing import TYPE_CHECKING, Any

from domyn.agents.llm import BaseLLMProvider, LLMProviderFactory
from domyn.agents.planner_strategy.base import BasePlannerStrategy
from domyn.agents.planner_strategy.context_management.base import ContextManager


class ContextManagementType(StrEnum):
    FULL_MESSAGES = "full_messages"
    AGENT_SUMMARY = "agents_summary"
    OLD_SUMMARY = "old_summary"
    INTERACTIONS_SUMMARY = "interactions_summary"
    TOOL_USE = "tool_use"
    THOUGHT_TOOL_USE = "thought_tool_use"


class PromptingType(StrEnum):
    STRUCTURAL_TAGS = "structural_tags"
    JSON_SCHEMA = "json_schema"


class ContextManagerFactory:
    """Factory for creating context manager instances."""

    @staticmethod
    def from_config(config: dict) -> ContextManager:
        config = copy.deepcopy(config)
        llm_provider_cfg = config.pop("llm_provider", None)
        if llm_provider_cfg is not None:
            if isinstance(llm_provider_cfg, dict):
                config["llm_provider"] = LLMProviderFactory.from_config(llm_provider_cfg)
            elif isinstance(llm_provider_cfg, BaseLLMProvider):
                config["llm_provider"] = llm_provider_cfg
            else:
                raise ValueError(
                    "Invalid llm_provider configuration. Must be a dict or BaseLLMProvider instance."
                )

        # Native tool-use context managers don't use a prompting layer — they
        # format native tool-call messages directly. Short-circuit before
        # prompting-factory dispatch so callers can omit the `prompting` key
        # (or carry a stale auto-injected one from upstream defaults).
        context_type_early = config.get("type")
        if context_type_early in (
            ContextManagementType.TOOL_USE.value,
            ContextManagementType.THOUGHT_TOOL_USE.value,
        ):
            config.pop("prompting", None)
            if context_type_early == ContextManagementType.TOOL_USE.value:
                from domyn.agents.planner_strategy.context_management.tool_use import (
                    ToolUseContextManager,
                )

                return ToolUseContextManager(**config)
            from domyn.agents.planner_strategy.context_management.tool_use import (
                ToolUseWithThoughtContextManager,
            )

            return ToolUseWithThoughtContextManager(**config)

        prompt_type = config.get("prompting", {}).get("type")
        if not prompt_type:
            raise ValueError("Missing 'type' field in prompting for context_manager configuration")

        match prompt_type:
            case PromptingType.STRUCTURAL_TAGS.value:
                from domyn.agents.planner_strategy.prompting.structural_tags_prompting import (
                    StructuralTagMessagePrompting,
                )

                config["prompting"] = StructuralTagMessagePrompting(**config.get("prompting", {}))

            case PromptingType.JSON_SCHEMA.value:
                from domyn.agents.planner_strategy.prompting.json_prompting import (
                    JSONMessagePrompting,
                )

                config["prompting"] = JSONMessagePrompting(**config.get("prompting", {}))

            case _:
                raise ValueError(f"Unknown prompting type: {prompt_type}")

        context_type = config.get("type")
        if not context_type:
            raise ValueError("Missing 'type' field in prompt builder configuration")

        match context_type:
            case ContextManagementType.FULL_MESSAGES.value:
                from domyn.agents.planner_strategy.context_management.full_messages import (
                    FullMessageContext,
                )

                return FullMessageContext(**config)
            case ContextManagementType.AGENT_SUMMARY.value:
                from domyn.agents.planner_strategy.context_management.summary import (
                    PreviousAgentsSummaryContext,
                )

                return PreviousAgentsSummaryContext(
                    **config,
                )
            case ContextManagementType.OLD_SUMMARY.value:
                from domyn.agents.planner_strategy.context_management.summary import (
                    OldEventsSummaryContext,
                )

                return OldEventsSummaryContext(
                    **config,
                )
            case ContextManagementType.INTERACTIONS_SUMMARY.value:
                from domyn.agents.planner_strategy.context_management.summary import (
                    PreviousInteractionsSummaryContext,
                )

                return PreviousInteractionsSummaryContext(
                    **config,
                )
            case _:
                raise ValueError(f"Unknown prompt builder type: {context_type}")


class PlannerStrategyType(StrEnum):
    REACT = "react"
    TOOL_USE = "tool_use"
    TOOL_USE_THOUGHT = "tool_use_thought"


def _resolve_compactor_config(compactor_cfg: Any) -> Any:
    """Turn a raw ``compactor`` config value into a ``Compactor`` instance.

    Mirrors how `context_manager` is resolved just above each branch: a dict
    is the `CompactionConfig`-shaped YAML form, an already-built `Compactor`
    passes through unchanged (the in-Python construction path), and nothing
    at all falls back to the framework default (`two_phase`).
    """
    from domyn.agents.planner_strategy.context_management.compaction import (
        CompactionConfig,
        Compactor,
        build_compactor,
    )

    if compactor_cfg is None:
        return Compactor()
    if isinstance(compactor_cfg, Compactor):
        return compactor_cfg
    if isinstance(compactor_cfg, dict):
        return build_compactor(CompactionConfig(**compactor_cfg))
    raise ValueError("Invalid compactor configuration. Must be a dict or Compactor instance.")


class PlannerStrategyFactory:
    """Factory for creating planner strategy instances."""

    @staticmethod
    def from_config(
        config: dict[str, Any], save_in_registry: bool = False
    ) -> "BasePlannerStrategy":
        config = copy.deepcopy(config)
        strategy_type = config.get("type")
        if not strategy_type:
            raise ValueError("Missing 'type' field in planner strategy configuration")

        if strategy_type == PlannerStrategyType.REACT.value:
            from domyn.agents.planner_strategy.react import ReactPlannerStrategy

            config = copy.deepcopy(config)
            context_manager_cfg = config.get("context_manager")
            if context_manager_cfg is None:
                # Nothing named: the planner's own default (ReactPlannerStrategy).
                config.pop("context_manager", None)
            elif isinstance(context_manager_cfg, dict):
                config["context_manager"] = ContextManagerFactory.from_config(context_manager_cfg)
            elif not isinstance(context_manager_cfg, ContextManager):
                raise ValueError(
                    "Invalid context_manager configuration. Must be a dict or BaseMessagePrompting instance."
                )

            if "compactor" in config:
                raise ValueError(
                    "react planner strategies do not support a compactor: "
                    "compaction only runs on the tool_use / tool_use_thought execute loop"
                )

            obj = ReactPlannerStrategy(**config)

        elif strategy_type == PlannerStrategyType.TOOL_USE.value:
            from domyn.agents.planner_strategy.context_management.tool_use import (
                ToolUseContextManager,
            )
            from domyn.agents.planner_strategy.tool_use import ToolUsePlannerStrategy

            context_manager_cfg = config.pop("context_manager", {})
            if isinstance(context_manager_cfg, ToolUseContextManager):
                context_manager = context_manager_cfg
            elif isinstance(context_manager_cfg, dict):
                context_manager = ToolUseContextManager(**context_manager_cfg)
            else:
                context_manager = ToolUseContextManager()

            config["context_manager"] = context_manager
            config["compactor"] = _resolve_compactor_config(config.pop("compactor", None))
            obj = ToolUsePlannerStrategy(**config)

        elif strategy_type == PlannerStrategyType.TOOL_USE_THOUGHT.value:
            from domyn.agents.planner_strategy.context_management.tool_use import (
                ToolUseContextManager,
                ToolUseWithThoughtContextManager,
            )
            from domyn.agents.planner_strategy.tool_use_thought import (
                ToolUseWithThoughtPlannerStrategy,
            )

            context_manager_cfg = config.pop("context_manager", {})
            if isinstance(
                context_manager_cfg, (ToolUseWithThoughtContextManager, ToolUseContextManager)
            ):
                context_manager = context_manager_cfg
            elif isinstance(context_manager_cfg, dict):
                context_manager = ToolUseWithThoughtContextManager(**context_manager_cfg)
            else:
                context_manager = ToolUseWithThoughtContextManager()

            config["context_manager"] = context_manager
            config["compactor"] = _resolve_compactor_config(config.pop("compactor", None))
            obj = ToolUseWithThoughtPlannerStrategy(**config)

        else:
            raise ValueError(f"Unknown provider type: {strategy_type}")

        if save_in_registry:
            from domyn.agents.registry import get_planner_strategy_registry

            get_planner_strategy_registry().register_planner_strategy(obj.name, obj)

        return obj


__all__ = [
    "CompactionPolicy",
    "Compactor",
    "ContextManagerFactory",
    "HeuristicEstimator",
    "JSONPrompting",
    "PlannerStrategyFactory",
    "PruneOnlyCompaction",
    "ReactPlannerStrategy",
    "ToolUseContextManager",
    "ToolUsePlannerStrategy",
    "ToolUseWithThoughtContextManager",
    "ToolUseWithThoughtPlannerStrategy",
]


# Re-exported on first use: the defining modules import this package, so naming
# them here at import time would be a cycle.
_LAZY_EXPORTS = {
    "ReactPlannerStrategy": "domyn.agents.planner_strategy.react",
    "ToolUsePlannerStrategy": "domyn.agents.planner_strategy.tool_use",
    "ToolUseWithThoughtPlannerStrategy": "domyn.agents.planner_strategy.tool_use_thought",
    "Compactor": "domyn.agents.planner_strategy.context_management.compaction",
    "CompactionPolicy": "domyn.agents.planner_strategy.context_management.compaction",
    "PruneOnlyCompaction": "domyn.agents.planner_strategy.context_management.compaction",
    "HeuristicEstimator": "domyn.agents.planner_strategy.context_management.budget",
    "ToolUseContextManager": "domyn.agents.planner_strategy.context_management.tool_use",
    "ToolUseWithThoughtContextManager": "domyn.agents.planner_strategy.context_management.tool_use",
    "JSONPrompting": "domyn.agents.planner_strategy.prompting.json_prompting",
}

if TYPE_CHECKING:
    from domyn.agents.planner_strategy.context_management.budget import (
        HeuristicEstimator as HeuristicEstimator,
    )
    from domyn.agents.planner_strategy.context_management.compaction import (
        CompactionPolicy as CompactionPolicy,
    )
    from domyn.agents.planner_strategy.context_management.compaction import Compactor as Compactor
    from domyn.agents.planner_strategy.context_management.compaction import (
        PruneOnlyCompaction as PruneOnlyCompaction,
    )
    from domyn.agents.planner_strategy.context_management.tool_use import (
        ToolUseContextManager as ToolUseContextManager,
    )
    from domyn.agents.planner_strategy.context_management.tool_use import (
        ToolUseWithThoughtContextManager as ToolUseWithThoughtContextManager,
    )
    from domyn.agents.planner_strategy.prompting.json_prompting import (
        JSONPrompting as JSONPrompting,
    )
    from domyn.agents.planner_strategy.react import ReactPlannerStrategy as ReactPlannerStrategy
    from domyn.agents.planner_strategy.tool_use import (
        ToolUsePlannerStrategy as ToolUsePlannerStrategy,
    )
    from domyn.agents.planner_strategy.tool_use_thought import (
        ToolUseWithThoughtPlannerStrategy as ToolUseWithThoughtPlannerStrategy,
    )


def __getattr__(name: str) -> Any:
    if name in _LAZY_EXPORTS:
        value = getattr(import_module(_LAZY_EXPORTS[name]), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
