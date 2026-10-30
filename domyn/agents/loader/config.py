import copy
from collections.abc import Iterable
from queue import Queue
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator

from domyn.agents.agents import Agent, AgentFactory, DelegateAgent
from domyn.agents.hooks import FunctionHook, HookFactory, HookPhase
from domyn.agents.llm import BaseLLMProvider
from domyn.agents.loader import load_from_file
from domyn.agents.logger import get_logger
from domyn.agents.planner_strategy import ReactPlannerStrategy, ToolUsePlannerStrategy
from domyn.agents.planner_strategy.context_management.full_messages import FullMessageContext
from domyn.agents.planner_strategy.context_management.summary import (
    OldEventsSummaryContext,
    PreviousAgentsSummaryContext,
    PreviousInteractionsSummaryContext,
)
from domyn.agents.registry import (
    get_agent_registry,
    get_hook_registry,
    get_llm_registry,
    get_planner_strategy_registry,
    get_tool_registry,
)
from domyn.agents.tools import DelegateTool, FunctionTool, StreamableFunctionTool
from domyn.agents.tools.builtin import BuiltinBundleConfig, BuiltinTools
from domyn.agents.tools.mcp import MCPServerStreamableHTTPConfig
from domyn.agents.tools.mcp.oauth_providers import (
    DEFAULT_OAUTH_PROVIDER_CONFIG,
    OAuthProviderConfig,
    OAuthProviderInstanceCache,
)

logger = get_logger()


class ConfigurationV1(BaseModel):
    agents: list[Agent | DelegateAgent]
    tools: list[FunctionTool | StreamableFunctionTool | DelegateTool] = []
    hooks: list[FunctionHook] = Field(
        default_factory=list,
        description="All hooks (agent and tool, pre and post identified by triggers).",
    )
    llm_providers: list[BaseLLMProvider] = []
    planner_strategies: list[ReactPlannerStrategy | ToolUsePlannerStrategy] = []
    mcp_servers: list[Any] = Field(
        default_factory=list,
        description="MCP server configurations for auto-fetching tools",
    )
    oauth_providers: list[OAuthProviderConfig] = Field(
        default_factory=list,
        description=(
            "Named OAuth provider templates, referenced by name from mcp_servers "
            "and tools entries via their own oauth_provider key."
        ),
    )
    tool_bundles: list[BuiltinBundleConfig] = Field(
        default_factory=list,
        description=(
            "Named built-in tool bundles (files/shell/browser), referenced by name per agent "
            "via 'tool_bundles'. Separate from 'tools' because one bundle expands into "
            "many tools at once, which the 1:1 Tool.from_config path cannot express."
        ),
    )
    context_manager: list[
        FullMessageContext
        | OldEventsSummaryContext
        | PreviousAgentsSummaryContext
        | PreviousInteractionsSummaryContext
    ] = []
    entrypoint: str = Field(description="Name of the entrypoint agent")
    final: bool = Field(
        default=False,
        description="Whether this configuration is finalized and cannot be modified",
    )
    metadata: dict[str, Any] | None = Field(
        default=None,
        description="Additional metadata that can contain any custom key-value pairs",
    )
    version: str = Field(default="1.0")

    @field_validator("version")
    @classmethod
    def validate_version(cls, v: str) -> str:
        """Ensure version is always 1.0"""
        if v != "1.0":
            raise ValueError("Version must be 1.0 and cannot be changed")
        return v

    @classmethod
    def model_validate(  # type: ignore[override]
        cls,
        obj: Any,
        *,
        strict: bool | None = None,
        from_attributes: bool | None = None,
        context: dict[str, Any] | None = None,
    ) -> "ConfigurationV1":
        """
        Override model_validate to automatically handle raw configuration dictionaries
        and support multiple validation scenarios.

        Handles:
        1. ConfigurationV1 objects (returns as-is for efficiency)
        2. Raw configuration dicts (processes through from_dict)
        3. Processed configuration dicts (standard Pydantic validation)

        This prevents issues with multiple validations and from_dict calls.
        """
        # Case 1: Already a ConfigurationV1 object - return as-is
        # This prevents issues when re-validating an already validated object
        if isinstance(obj, cls):
            return obj

        # Case 2: Dictionary input
        if isinstance(obj, dict):
            # Case 2a: Raw configuration (string references) - use from_dict
            if cls._is_raw_config(obj):
                return cls.from_dict(obj)

            # Case 2b: Processed configuration (object references) - standard validation
            # This handles dicts from model_dump() of already processed configs
            return super().model_validate(
                obj, strict=strict, from_attributes=from_attributes, context=context
            )

        return super().model_validate(
            obj, strict=strict, from_attributes=from_attributes, context=context
        )

    @classmethod
    def _is_raw_config(cls, obj: dict[str, Any]) -> bool:
        """
        Determine if a configuration dictionary is in raw format and needs processing.

        A raw config typically has string references for tools, llm_providers, etc.
        """

        # Check for raw agent definitions (with string tool and hook references)
        agents = obj.get("agents", [])
        if agents and isinstance(agents[0], dict):
            agent_tools = agents[0].get("tools", [])
            if agent_tools and isinstance(agent_tools[0], str):
                return True
            planner_strategy = agents[0].get("planner_strategy")
            if planner_strategy and isinstance(planner_strategy, str):
                return True
            llm_provider = agents[0].get("llm_provider")
            if llm_provider and isinstance(llm_provider, str):
                return True
            agent_hooks = agents[0].get("hooks", [])
            if agent_hooks and isinstance(agent_hooks[0], str):
                return True

        # Check for raw hook definitions in the unified 'hooks' key
        hooks = obj.get("hooks", [])
        if hooks and isinstance(hooks[0], dict):
            hook_trigger = hooks[0].get("trigger", None)
            try:
                hook_trigger = HookPhase(hook_trigger)
            except ValueError:
                hook_trigger = None
            reference = hooks[0].get("reference", None)
            if hook_trigger or (reference and isinstance(reference, str)):
                return True

        # Check for raw tool definitions (with string hook references)
        tools = obj.get("tools", [])
        if tools and isinstance(tools[0], dict):
            tool_hooks = tools[0].get("hooks", [])
            if tool_hooks and isinstance(tool_hooks[0], str):
                return True
        return False

    @classmethod
    def _customize_agent_schema_properties(
        cls, original_properties: dict[str, Any]
    ) -> dict[str, Any]:
        """Customize and reorder agent schema properties."""
        original_properties = original_properties.copy()

        # YAML uses planner_strategy; planner is the Python field.
        # TODO: one name for both (renaming Agent.planner breaks the API).
        if "planner" in original_properties:
            del original_properties["planner"]

        fields_to_remove = ["type", "default_execute_agent_function"]
        for field in fields_to_remove:
            if field in original_properties:
                del original_properties[field]

        ordered_properties = {}

        # Basic agent info
        basic_fields = ["name", "description", "instruction", "max_iterations"]
        for field in basic_fields:
            if field in original_properties:
                ordered_properties[field] = original_properties[field]

        # Configuration references
        if "llm_provider" in original_properties:
            ordered_properties["llm_provider"] = {
                "type": "string",
                "description": "Name of LLM provider instance defined in configuration or registry.",
                "title": "LLM Provider Name",
            }
        ordered_properties["planner_strategy"] = {
            "description": "Name of the planner strategy to use. Must be defined in this configuration or present in registry. Omitted: a tool_use_thought planner with default settings.",
            "type": "string",
            "title": "Planner Strategy name",
        }
        if "tools" in original_properties:
            ordered_properties["tools"] = {
                "description": "List of tool names that refer to tools either defined in this configuration or present in registry.",
                "items": {"type": "string", "description": "Name of a tool"},
                "title": "Tools",
                "type": "array",
                "default": [],
            }
        # Not an Agent field (expanded away during from_dict, like planner_strategy
        # above), but a legitimate raw-config key — documented unconditionally.
        ordered_properties["tool_bundles"] = {
            "description": "List of tool bundle names (defined in the top-level tool_bundles list) whose built-in tools should be mounted on this agent.",
            "items": {"type": "string", "description": "Name of a tool bundle"},
            "title": "Tool Bundles",
            "type": "array",
            "default": [],
        }
        if "sub_agents" in original_properties:
            ordered_properties["sub_agents"] = {
                "description": "List of sub-agent names that refer to other agents defined in this configuration.",
                "items": {
                    "type": "string",
                    "description": "Name of an agent defined in the agents list",
                },
                "title": "Sub Agents",
                "type": "array",
                "default": [],
            }
        if "hooks" in original_properties:
            ordered_properties["hooks"] = {
                "description": "List of hook names (defined in the top-level hooks list) to attach to this agent.",
                "items": {"type": "string", "description": "Name of a hook"},
                "title": "Hooks",
                "type": "array",
                "default": [],
            }

        if "context_visibility" in original_properties:
            ordered_properties["context_visibility"] = original_properties["context_visibility"]

        if "handoff_scope" in original_properties:
            ordered_properties["handoff_scope"] = original_properties["handoff_scope"]

        already_added = {
            *basic_fields,
            "llm_provider",
            "planner_strategy",
            "tools",
            "tool_bundles",
            "sub_agents",
            "hooks",
            "context_visibility",
            "handoff_scope",
        }
        for key, value in original_properties.items():
            if key not in already_added:
                ordered_properties[key] = value

        return ordered_properties

    @classmethod
    def _customize_tool_schema_properties(cls, tool_schema: dict[str, Any]) -> None:
        """Customize tool schema properties to specify that name is required and handle reference vs normal tool configuration."""
        # Handle tools with anyOf structure (multiple tool types)
        if "anyOf" in tool_schema:
            cls._customize_tool_schema_for_reference_logic(tool_schema)
            for tool_type in tool_schema["anyOf"]:
                if "properties" in tool_type:
                    cls._customize_tool_hooks_properties(tool_type["properties"])

        # Handle tools with direct properties structure
        elif "properties" in tool_schema:
            cls._customize_tool_hooks_properties(tool_schema["properties"])

    @classmethod
    def _customize_schema_for_reference_logic(
        cls,
        schema: dict[str, Any],
        item_type: str = "tool",
        optional_fields: list[str] | None = None,
    ) -> None:
        """
        Customize schema to handle reference vs normal configuration for tools and hooks.

        Args:
            schema: The schema to customize (should have anyOf structure)
            item_type: Type of item being processed ("tool", "hook", etc.)
            optional_fields: List of optional fields to include in reference variant
        """
        if "anyOf" not in schema:
            return

        if optional_fields is None:
            optional_fields = ["config_params", "hooks"]

        # Create a single reference variant with appropriate optional fields
        reference_properties = {
            "name": {"type": "string", "title": "Name", "description": f"Name of the {item_type}"},
            "reference": {
                "type": "string",
                "title": "Reference",
                "description": f"Reference to an existing {item_type} in the registry",
            },
        }

        # Add optional fields for reference variants
        for field in optional_fields:
            if field == "config_params":
                reference_properties[field] = {
                    "type": "array",
                    "items": {"type": "object"},
                    "default": [],
                    "title": "Config Params",
                    "description": f"Optional configuration parameters for the referenced {item_type}",
                }
            elif field == "hooks":
                reference_properties[field] = {
                    "type": "array",
                    "items": {"type": "string"},
                    "default": [],
                    "title": "Hooks",
                    "description": f"Optional hooks for the referenced {item_type}",
                }

        reference_variant = {
            "type": "object",
            "properties": reference_properties,
            "required": ["name", "reference"],
            "title": f"{item_type.title()} Reference",
            "description": f"Reference to an existing {item_type} in the registry with optional configuration overrides",
        }

        # Keep all original definition variants
        definition_variants = []
        for item_variant in schema["anyOf"]:
            if "properties" not in item_variant:
                definition_variants.append(item_variant)
                continue

            variant_properties = {
                k: v for k, v in item_variant["properties"].items() if k != "reference"
            }

            definition_variant = {**item_variant, "properties": variant_properties}
            definition_variants.append(definition_variant)

        schema["anyOf"] = [reference_variant, *definition_variants]

    @classmethod
    def _customize_tool_schema_for_reference_logic(cls, tool_schema: dict[str, Any]) -> None:
        """Customize tool schema to handle reference vs normal tool configuration."""
        cls._customize_schema_for_reference_logic(
            tool_schema,
            item_type="tool",
            optional_fields=["config_params", "hooks"],
        )

    @classmethod
    def _customize_tool_hooks_properties(cls, properties: dict[str, Any]) -> None:
        """Customize tool hooks property to only accept string references to named hooks."""
        if "hooks" in properties:
            properties["hooks"] = {
                "description": "List of hook names that refer to hooks defined in the top-level hooks list.",
                "items": {
                    "type": "string",
                    "description": "Name of a hook defined in the top-level hooks list",
                },
                "title": "Hooks",
                "type": "array",
                "default": [],
            }

    @classmethod
    def _customize_planner_strategy_schema_properties(
        cls, planner_strategy_schema: dict[str, Any]
    ) -> None:
        """Customize planner strategy schema properties to specify prompting should only be strings."""
        # Handle planner strategies with anyOf structure (multiple planner strategy types)
        if "anyOf" in planner_strategy_schema:
            for strategy_type in planner_strategy_schema["anyOf"]:
                if "properties" in strategy_type:
                    cls._customize_planner_strategy_properties(strategy_type["properties"])

        # Handle planner strategies with direct properties structure
        elif "properties" in planner_strategy_schema:
            cls._customize_planner_strategy_properties(planner_strategy_schema["properties"])

    @classmethod
    def _customize_planner_strategy_properties(cls, properties: dict[str, Any]) -> None:
        """Customize planner strategy properties to only accept string references for prompting."""
        # Remove prompt_builder since config uses prompting string references
        properties.pop("context_manager", None)

        properties["context_manager"] = {
            "description": "Name of a context manager strategy defined in the context_managers list",
            "type": "string",
            "title": "Context manager Strategy",
        }

    @classmethod
    def _customize_hook_schema_properties(cls, hook_schema: dict[str, Any], hook_type: str) -> None:
        """Customize hook schema properties to handle reference vs normal hook configuration."""
        # Handle hooks with anyOf structure (multiple hook types)
        if "anyOf" in hook_schema:
            cls._customize_schema_for_reference_logic(
                hook_schema,
                item_type=hook_type,
                optional_fields=[
                    "config_params"
                ],  # hooks typically only have config_params as optional
            )

        # Handle hooks with direct properties structure
        elif "properties" in hook_schema:
            # Plain hook schemas need no customization
            pass

    @classmethod
    def model_json_schema(  # type: ignore[override]
        cls, by_alias: bool = True, ref_template: str = "#/$defs/{model}", **kwargs: Any
    ) -> dict[str, Any]:
        """Custom JSON schema for API documentation."""
        # Get the base schema
        schema = super().model_json_schema(by_alias=by_alias, ref_template=ref_template, **kwargs)

        # Function to remove additionalProperties from the schema
        def remove_additional_properties(obj: Any) -> Any:
            if isinstance(obj, dict):
                # Remove additionalProperties if present
                if "additionalProperties" in obj:
                    del obj["additionalProperties"]

                # Recursively process all nested objects
                for key, value in obj.items():
                    obj[key] = remove_additional_properties(value)

            elif isinstance(obj, list):
                return [remove_additional_properties(item) for item in obj]

            return obj

        # Function to resolve $ref and inline definitions with recursion protection
        def resolve_refs(
            obj: Any, defs: dict[str, Any], visited_refs: set[str] | None = None
        ) -> Any:
            if visited_refs is None:
                visited_refs = set()

            if isinstance(obj, dict):
                if "$ref" in obj:
                    ref_path = obj["$ref"]
                    if ref_path.startswith("#/$defs/"):
                        def_name = ref_path.split("/")[-1]

                        # Check for circular reference
                        if ref_path in visited_refs:
                            # Return a simplified reference to avoid infinite recursion
                            return {
                                "type": "object",
                                "description": def_name,  # Circular reference avoided
                            }

                        if def_name in defs:
                            visited_refs.add(ref_path)
                            resolved = resolve_refs(
                                defs[def_name].copy(), defs, visited_refs.copy()
                            )
                            visited_refs.discard(ref_path)
                            return resolved
                    return obj
                else:
                    return {k: resolve_refs(v, defs, visited_refs) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [resolve_refs(item, defs, visited_refs) for item in obj]
            else:
                return obj

        # Get the definitions and resolve all references
        defs = schema.get("$defs", {})
        resolved_schema = resolve_refs(schema, defs)

        # Remove the $defs section since we've inlined everything
        if "$defs" in resolved_schema:
            del resolved_schema["$defs"]

        clean_schema = remove_additional_properties(resolved_schema)

        clean_schema["title"] = "Domyn Agent Configuration"
        clean_schema["description"] = "Configuration schema for Domyn AI Agents system"

        if "properties" in clean_schema:
            if "version" in clean_schema["properties"]:
                del clean_schema["properties"]["version"]
            if "required" in clean_schema and "version" in clean_schema["required"]:
                clean_schema["required"].remove("version")

            # Customize agents properties to specify sub_agents should only be strings
            if (
                "agents" in clean_schema["properties"]
                and "items" in clean_schema["properties"]["agents"]
            ):
                agent_schema = clean_schema["properties"]["agents"]["items"]
                if "properties" in agent_schema:
                    agent_schema["properties"] = cls._customize_agent_schema_properties(
                        agent_schema["properties"]
                    )

            # Customize tools properties to specify hooks should only be strings
            if (
                "tools" in clean_schema["properties"]
                and "items" in clean_schema["properties"]["tools"]
            ):
                tool_schema = clean_schema["properties"]["tools"]["items"]
                cls._customize_tool_schema_properties(tool_schema)

            # Customize planner strategies properties to specify prompting should be strings
            if (
                "planner_strategies" in clean_schema["properties"]
                and "items" in clean_schema["properties"]["planner_strategies"]
            ):
                planner_strategy_schema = clean_schema["properties"]["planner_strategies"]["items"]
                cls._customize_planner_strategy_schema_properties(planner_strategy_schema)

            # Customize top-level hooks properties to handle reference vs normal configuration
            if (
                "hooks" in clean_schema["properties"]
                and "items" in clean_schema["properties"]["hooks"]
            ):
                hook_schema = clean_schema["properties"]["hooks"]["items"]
                cls._customize_hook_schema_properties(hook_schema, "hook")

        return clean_schema

    @classmethod
    def from_dict(
        cls,
        config_dict: dict[str, Any],
        save_in_registry: bool = False,
        *,
        extra_workspace_roots: Iterable[str] = (),
    ) -> "ConfigurationV1":
        """Create Configuration from a dictionary, processing agents, tools, and hooks.

        ``extra_workspace_roots`` widens every built-in tool bundle's
        ``Workspace`` (see ``BuiltinBundleConfig.build``) — a parameter rather
        than a config key because which directories those are is resolved at
        run time by the caller (e.g. the CLI's configured skill directories),
        not something a persisted file should freeze.
        """

        # Helper function for deep merging dictionaries
        def deep_merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
            """Deep merge two dictionaries, with override values taking precedence.

            Args:
                base: Base dictionary
                override: Dictionary with values to override/add

            Returns:
                Merged dictionary
            """
            result = copy.deepcopy(base)
            for key, value in override.items():
                if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                    # Both are dicts, merge recursively
                    result[key] = deep_merge_dicts(result[key], value)
                else:
                    # Override or add the value
                    result[key] = value
            return result

        # LLM PROVIDER
        providers_registry = get_llm_registry()
        llm_providers = {
            name: providers_registry.get_llm_provider(name)
            for name in providers_registry.list_llm_provider_names()
        }
        for llm_provider_config in config_dict.get("llm_providers", []):
            from domyn.agents.llm import PROTECTED_FIELDS, LLMProviderFactory

            # Check if this is a reference to a registry provider
            if "reference" in llm_provider_config:
                reference_name = llm_provider_config["reference"]
                provider_name = llm_provider_config.get("name")

                if not provider_name:
                    raise ValueError("LLM provider with 'reference' must have 'name' field")

                # Extract config_params for overrides (optional - can be empty for simple rename)
                config_params = llm_provider_config.get("config_params", {})

                # Check for protected field overrides
                for field in config_params:
                    if field in PROTECTED_FIELDS:
                        raise ValueError(
                            f"Cannot override protected field '{field}' via config_params. "
                            f"Protected fields are: {', '.join(sorted(PROTECTED_FIELDS))}"
                        )

                # Lookup base provider ONLY in registry (not in config-level providers)
                if reference_name not in get_llm_registry().list_llm_provider_names():
                    raise ValueError(
                        f"Referenced LLM provider '{reference_name}' not found in registry. "
                        "The 'reference' field can only reference providers registered in the LLM registry, "
                        "not providers defined in this configuration file."
                    )

                base_provider = get_llm_registry().get_llm_provider(reference_name)

                # Create deep copy to avoid shared mutable state
                llm_provider = base_provider.model_copy(deep=True)

                # Update name to the new instance name
                llm_provider.name = provider_name

                # Apply all config_params overrides (generalized for any field)
                for field, value in config_params.items():
                    # Deep merge for dict fields, direct set for others
                    if isinstance(value, dict) and hasattr(llm_provider, field):
                        current_value = getattr(llm_provider, field)
                        if isinstance(current_value, dict):
                            # Deep merge: recursively merge nested dicts
                            setattr(llm_provider, field, deep_merge_dicts(current_value, value))
                        else:
                            # Current value is not a dict, just replace
                            setattr(llm_provider, field, value)
                    else:
                        # Not a dict or field doesn't exist, set directly
                        setattr(llm_provider, field, value)

                # Store the new instance (allows overwriting if same name appears twice)
                llm_providers[provider_name] = llm_provider
            else:
                llm_provider = LLMProviderFactory.from_config(
                    llm_provider_config, save_in_registry=save_in_registry
                )
                # Allow overwriting if same name appears twice
                llm_providers[llm_provider.name] = llm_provider

        context_managers = {}
        for context_manager_config in config_dict.get("context_managers", []):
            from domyn.agents.planner_strategy import ContextManagerFactory

            if "llm_provider" in context_manager_config:
                new_provider = llm_providers.get(context_manager_config["llm_provider"])
                if new_provider is None:
                    raise ValueError(
                        f"LLM Provider {context_manager_config['llm_provider']} not found in configuration"
                    )
                context_manager_config["llm_provider"] = new_provider

            context_manager = ContextManagerFactory.from_config(
                context_manager_config,
            )
            context_managers[context_manager.name] = context_manager

        # PLANNER STRATEGY
        planner_strategies = {}
        for planner_strategy_config in config_dict.get("planner_strategies", []):
            from domyn.agents.planner_strategy import PlannerStrategyFactory

            if "context_manager" in planner_strategy_config:
                new_context_manager = context_managers.get(
                    planner_strategy_config["context_manager"]
                )
                if new_context_manager is None:
                    raise ValueError(
                        f"Context manager {planner_strategy_config['context_manager']} not found in configuration"
                    )
                planner_strategy_config["context_manager"] = new_context_manager

            planner_strategy = PlannerStrategyFactory.from_config(
                planner_strategy_config, save_in_registry=save_in_registry
            )
            planner_strategies[planner_strategy.name] = planner_strategy

        hooks = {phase: {} for phase in HookPhase}

        registered_hooks = get_hook_registry().list_hook_names()

        for hook_config in config_dict.get("hooks", []):
            reference = hook_config.get("reference")
            if reference in registered_hooks:
                loaded_hook = get_hook_registry().get_hook(reference)
                if not loaded_hook:
                    raise ValueError(f"Hook '{reference}' could not be loaded from hook registry")
                # Apply alias/overrides from config
                alias = hook_config.get("name")
                updates: dict = {}
                if alias and alias != loaded_hook.name:
                    updates["name"] = alias
                    updates["config_params"] = hook_config.get("config_params", {})
                raw_triggers = hook_config.get("on_event_triggers")
                if raw_triggers is not None:
                    from domyn.agents.hooks import _parse_on_event_triggers

                    updates["on_event_triggers"] = _parse_on_event_triggers(raw_triggers)
                if "silent" in hook_config:
                    updates["silent"] = hook_config["silent"]
                if updates:
                    loaded_hook = loaded_hook.model_copy(update=updates)
                hooks[loaded_hook.trigger][loaded_hook.name] = loaded_hook
            else:
                trigger_value = hook_config.get("trigger")
                if trigger_value is None:
                    raise ValueError(
                        f"Hook '{hook_config.get('name', '?')}' must have a 'trigger' field. "
                        "Valid values: pre_agent_call, post_agent_call, pre_tool_approval, "
                        "pre_tool_call, post_tool_call, on_agent_event"
                    )
                # Normalize trigger to HookPhase enum value
                try:
                    trigger = HookPhase(trigger_value)
                except ValueError as err:
                    raise ValueError(
                        f"Invalid trigger '{trigger_value}' for hook '{hook_config.get('name', '?')}'. "
                        f"Valid values: {[phase.value for phase in HookPhase]}."
                    ) from err
                hook_config = dict(hook_config)
                hook_config["trigger"] = trigger
                loaded_hook = HookFactory.from_config(
                    hook_config, save_in_registry=save_in_registry
                )
                hooks[trigger][loaded_hook.name] = loaded_hook

        # OAUTH PROVIDERS
        oauth_provider_configs = {}
        for oauth_provider_config in config_dict.get("oauth_providers", []):
            cfg = OAuthProviderConfig.from_dict(oauth_provider_config)
            oauth_provider_configs[cfg.name] = cfg
        oauth_provider_cache = OAuthProviderInstanceCache()

        def _resolve_oauth_provider(entry_config: dict[str, Any], server_url: str | None) -> None:
            """Mutate ``entry_config`` in place: inject a built OAuth instance
            under ``oauth_provider`` whenever ``use_oauth`` is set — the
            named one if referenced by string, the shared default otherwise.
            Left untouched if it's already a live provider object (built
            directly in Python, bypassing this loader).
            """
            if not entry_config.get("use_oauth") or not server_url:
                return
            name = entry_config.get("oauth_provider")
            if name is None:
                template = DEFAULT_OAUTH_PROVIDER_CONFIG
            elif isinstance(name, str):
                if name not in oauth_provider_configs:
                    raise ValueError(f"OAuth provider '{name}' not found in configuration")
                template = oauth_provider_configs[name]
            else:
                return
            entry_config["oauth_provider"] = oauth_provider_cache.get_or_build(template, server_url)

        mcp_servers = {}
        for mcp_server_config in config_dict.get("mcp_servers", []):
            match mcp_server_config:
                case MCPServerStreamableHTTPConfig():
                    server = mcp_server_config
                case dict():
                    _resolve_oauth_provider(mcp_server_config, mcp_server_config.get("url"))
                    server = MCPServerStreamableHTTPConfig.from_dict(mcp_server_config)
                case _:
                    raise ValueError(f"Invalid MCP server configuration: {mcp_server_config}")

            mcp_servers[server.name] = server

        # TOOL BUNDLES (built-in files/shell/browser — one entry expands into many tools,
        # so they live outside the 1:1 `tools:`/Tool.from_config path, same as
        # `mcp_servers` above)
        tool_bundles: dict[str, BuiltinBundleConfig] = {}
        for tool_bundle_config in config_dict.get("tool_bundles", []):
            match tool_bundle_config:
                case BuiltinBundleConfig():
                    tool_bundle = tool_bundle_config
                case dict():
                    tool_bundle = BuiltinBundleConfig(**tool_bundle_config)
                case _:
                    raise ValueError(f"Invalid tool bundle configuration: {tool_bundle_config}")

            tool_bundles[tool_bundle.name] = tool_bundle

        # DECISION PROVIDERS (Jev/Clef, or an LLM standing in). Named here and
        # referenced per agent, like tool bundles: one endpoint, one client,
        # shared by every agent that points at it.
        from domyn.agents.decisions.config import DecisionProviderConfig, DecisionsConfig

        decision_providers: dict[str, Any] = {}
        for decision_config in config_dict.get("decision_providers", []):
            entry = (
                decision_config
                if isinstance(decision_config, DecisionProviderConfig)
                else DecisionProviderConfig(**decision_config)
            )
            decision_providers[entry.name] = entry.build()

        # TOOLS
        processed_tools = {}
        for tool_config in config_dict.get("tools", []):
            from domyn.agents.tools import Tool

            # Resolve hook references
            new_hooks = []

            for hook_name in tool_config.get("hooks", []):
                if hook_name in hooks[HookPhase.PRE_TOOL_CALL]:
                    new_hooks.append(hooks[HookPhase.PRE_TOOL_CALL][hook_name])
                elif hook_name in hooks[HookPhase.POST_TOOL_CALL]:
                    new_hooks.append(hooks[HookPhase.POST_TOOL_CALL][hook_name])
                elif hook_name in hooks[HookPhase.PRE_TOOL_APPROVAL]:
                    new_hooks.append(hooks[HookPhase.PRE_TOOL_APPROVAL][hook_name])
                elif hook_name in get_hook_registry().list_hook_names():
                    new_hooks.append(get_hook_registry()[hook_name])
                else:
                    raise ValueError(f"Tool hook '{hook_name}' not found in configuration")
            tool_config["hooks"] = new_hooks

            if tool_config.get("type") == "mcp":
                _resolve_oauth_provider(tool_config, tool_config.get("server_url"))

            try:
                tool = Tool.from_config(tool_config, save_in_registry=save_in_registry)
                processed_tools[(tool.uid, tool.name)] = tool
            except (ValueError, TypeError) as e:
                # Skipped rather than raised, so one bad entry does not stop the
                # whole configuration loading; the warning is what makes it visible.
                logger.warning(
                    f"Skipping tool '{tool_config.get('name')}' (type "
                    f"'{tool_config.get('type')}'): {type(e).__name__}"
                )
                logger.debug(f"Tool '{tool_config.get('name')}' could not be built, detail: {e}")
                for agent_config in config_dict.get("agents", []):
                    if tool_config.get("name") in agent_config.get("tools", []):
                        agent_config["tools"].remove(tool_config.get("name"))
                continue

        # AGENTS
        processed_agents: dict[str, Agent] = {}

        # reorder agents
        agents_lists = []
        registered_agent_names = get_agent_registry().list_agent_names()
        agents_queue = Queue()
        for index, agent_config in enumerate(config_dict.get("agents", [])):
            # Checked here because the ordering below reads every agent's name,
            # before Agent.from_config gets to validate it.
            if not isinstance(agent_config, dict) or not agent_config.get("name"):
                raise ValueError(f"Agent #{index + 1} under 'agents' has no 'name'")
            agents_queue.put(agent_config)

        queue_size = agents_queue.qsize()

        queue_check = []

        while agents_queue.empty() is False:
            agent_config = agents_queue.get()

            sub_agents_names = agent_config.get("sub_agents", [])

            if not sub_agents_names:
                agents_lists.append(agent_config)
            else:
                if all(
                    sub_a in [a["name"] for a in agents_lists] or sub_a in registered_agent_names
                    for sub_a in sub_agents_names
                ):
                    agents_lists.append(agent_config)
                else:
                    agents_queue.put(agent_config)

            if agents_queue.qsize() == queue_size:
                if [a["name"] for a in agents_queue.queue] in queue_check:
                    raise ValueError("Circular dependency detected in agent sub-agents")

                queue_check.append([a["name"] for a in agents_queue.queue])

            queue_size = agents_queue.qsize()

        for agent_config in agents_lists:
            # Resolve LLM provider reference
            llm_provider_ref = agent_config.get("llm_provider")
            if llm_provider_ref:
                # Only string format supported - provider instances defined at level 0
                if not isinstance(llm_provider_ref, str):
                    raise ValueError(
                        f"LLM provider must be a string (provider instance name), got {type(llm_provider_ref)}. "
                        "Use 'reference' field in level 0 llm_providers to create provider instances with overrides."
                    )

                # Lookup provider: first in config-level providers, then registry
                if llm_providers and llm_provider_ref in llm_providers:
                    agent_config["llm_provider"] = llm_providers[llm_provider_ref]
                elif llm_provider_ref in get_llm_registry().list_llm_provider_names():
                    agent_config["llm_provider"] = get_llm_registry().get_llm_provider(
                        llm_provider_ref
                    )
                else:
                    raise ValueError(
                        f"LLM provider '{llm_provider_ref}' not found in configuration or registry"
                    )

            # Resolve planner strategy reference
            planner_strategy_name = agent_config.get("planner_strategy")
            if planner_strategy_name:
                if planner_strategies and planner_strategy_name in planner_strategies:
                    agent_config["planner_strategy"] = planner_strategies[planner_strategy_name]
                elif (
                    planner_strategy_name
                    in get_planner_strategy_registry().list_planner_strategy_names()
                ):
                    agent_config["planner_strategy"] = (
                        get_planner_strategy_registry().get_planner_strategy(planner_strategy_name)
                    )
                else:
                    raise ValueError(
                        f"Planner strategy '{planner_strategy_name}' not found in configuration"
                    )

            # Resolve tool references
            new_tools = []
            for tool_name in agent_config.get("tools", []):
                tool_references = [
                    key for key in processed_tools if key[1] == tool_name or key[0] == tool_name
                ]
                if len(tool_references) > 1:
                    raise ValueError(
                        f"Ambiguous tool reference '{tool_name}' found in configuration. Multiple tools with the same name or UID exist."
                    )
                if processed_tools and len(tool_references) == 1:
                    new_tools.append(processed_tools[tool_references[0]])
                elif tool_name in get_tool_registry().list_tool_names():
                    new_tools.append(get_tool_registry().get_tool(tool_name))
                else:
                    raise ValueError(f"Tool '{tool_name}' not found in configuration")

            # expand built-in tool bundles referenced by name into real tool
            # instances — built here rather than in the tools loop above because
            # one bundle produces several tools at once (see `tool_bundles` block)
            agent_builtin_tools: list[BuiltinTools] = []
            for bundle_name in agent_config.get("tool_bundles", []):
                if bundle_name not in tool_bundles:
                    raise ValueError(f"Tool bundle '{bundle_name}' not found in configuration")
                built = tool_bundles[bundle_name].build(extra_roots=extra_workspace_roots)
                agent_builtin_tools.append(built)
                new_tools.extend(built.tools)

            agent_config["tools"] = new_tools
            agent_config["builtin_tools"] = agent_builtin_tools
            agent_config.pop("tool_bundles", None)

            # a decision model is optional, per agent, and absent by default:
            # without this key every caller behaves exactly as it does today
            declared_decisions = agent_config.pop("decisions", None)
            if declared_decisions:
                agent_config["decisions"] = DecisionsConfig.from_dict(declared_decisions).build(
                    decision_providers
                )

            # resolve agent hook references (names → hook objects)
            new_hooks = []
            all_agent_hooks = {
                **hooks[HookPhase.PRE_AGENT_CALL],
                **hooks[HookPhase.POST_AGENT_CALL],
                **hooks[HookPhase.ON_AGENT_EVENT],
            }
            for hook_name in agent_config.get("hooks", []):
                if hook_name in all_agent_hooks:
                    new_hooks.append(all_agent_hooks[hook_name])
                elif hook_name in get_hook_registry().list_hook_names():
                    new_hooks.append(get_hook_registry()[hook_name])
                else:
                    raise ValueError(f"Agent hook '{hook_name}' not found in configuration")
            agent_config["hooks"] = new_hooks

            # sub agents
            new_sub_agents = []
            for sub_agent_name in agent_config.get("sub_agents", []):
                if sub_agent_name in processed_agents:
                    new_sub_agents.append(processed_agents[sub_agent_name])
                elif sub_agent_name in get_agent_registry().list_agent_names():
                    new_sub_agents.append(get_agent_registry().get_agent(sub_agent_name))
                else:
                    raise ValueError(f"Sub-agent '{sub_agent_name}' not found in configuration")

            agent_config["sub_agents"] = new_sub_agents

            # MCP servers
            agent_mcp_servers = []
            for mcp_server in agent_config.get("mcp_servers", []):
                if isinstance(mcp_server, str) and mcp_server in mcp_servers:
                    agent_mcp_servers.append(mcp_servers[mcp_server])
                elif isinstance(mcp_server, MCPServerStreamableHTTPConfig):
                    agent_mcp_servers.append(mcp_server)
                elif isinstance(mcp_server, dict):
                    agent_mcp_servers.append(MCPServerStreamableHTTPConfig.from_dict(mcp_server))
                else:
                    raise ValueError(f"MCP server '{mcp_server}' not found in configuration")

            agent_config["mcp_servers"] = agent_mcp_servers
            _agent = AgentFactory.from_config(agent_config, save_in_registry=save_in_registry)

            processed_agents[_agent.name] = _agent

        # Create the processed config
        all_hooks = (
            list(hooks[HookPhase.PRE_AGENT_CALL].values())
            + list(hooks[HookPhase.POST_AGENT_CALL].values())
            + list(hooks[HookPhase.PRE_TOOL_CALL].values())
            + list(hooks[HookPhase.POST_TOOL_CALL].values())
        )
        processed_config = {
            "agents": list(processed_agents.values()),
            "tools": list(processed_tools.values()),
            "hooks": all_hooks,
            "llm_providers": list(llm_providers.values()),
            "planner_strategies": list(planner_strategies.values()),
            "mcp_servers": list(mcp_servers.values()),
            "tool_bundles": list(tool_bundles.values()),
            "entrypoint": config_dict.get("entrypoint", ""),
            "final": config_dict.get("final", False),
            "metadata": config_dict.get("metadata"),
            "version": config_dict.get("version", "1.0"),
        }

        return cls.model_validate(processed_config)

    def to_dict(self) -> dict[str, Any]:
        """Serialize Configuration to a dictionary suitable for YAML export."""
        # Serialize agents back to config dictionaries
        serialized_agents = []
        for agent in self.agents:
            # Use model_dump to get a serializable dict
            agent_dict = agent.model_dump(exclude_none=True)
            serialized_agents.append(agent_dict)

        # Serialize tools back to config dictionaries
        serialized_tools = []
        for tool in self.tools:
            tool_dict = tool.model_dump(exclude_none=True)
            serialized_tools.append(tool_dict)

        # Serialize all hooks into a single list; trigger field already carried on each hook
        serialized_hooks = []
        for hook in self.hooks:
            hook_dict = hook.model_dump(exclude_none=True)
            serialized_hooks.append(hook_dict)

        return {
            "agents": serialized_agents,
            "tools": serialized_tools,
            "hooks": serialized_hooks,
            "entrypoint": self.entrypoint,
            "final": self.final,
            "metadata": self.metadata,
            "version": self.version,
        }

    @classmethod
    def from_yaml(cls, file_path: str) -> "ConfigurationV1":
        """Load Configuration from a YAML file."""
        config_dict = load_from_file(file_path)
        return cls.from_dict(config_dict)

    def to_yaml(self, file_path: str) -> None:
        """Save Configuration to a YAML file."""
        config_dict = self.to_dict()
        with open(file_path, "w", encoding="utf-8") as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
