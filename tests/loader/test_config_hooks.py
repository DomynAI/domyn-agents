"""Tests for hook loading in ConfigurationV1.from_dict.

Covers:
- Non-registered (path-based) agent hooks: pre_agent_call, post_agent_call
- Non-registered (path-based) tool hooks: pre_tool_call, post_tool_call
- Registered hooks (via @hook decorator):
    * No trigger required in the top-level hooks dict entry
    * No top-level hooks entry needed at all - resolved directly by agent/tool
- Listener vs updater hook types
- Hooks attached to agents vs hooks attached to tools
- Error cases: missing trigger for unregistered hooks, unknown hook names,
  invalid trigger values, missing hook reference in registry
"""

import copy

import pytest

from domyn.agents import ExecutionEventType
from domyn.agents.core.decorators import hook
from domyn.agents.hooks import FunctionHook
from domyn.agents.hooks.base import HookPhase, HookType, InputFields
from domyn.agents.loader import ConfigurationV1
from domyn.agents.registry import get_hook_registry

FunctionAgentHook = FunctionHook
FunctionToolPreHook = FunctionHook
FunctionToolPostHook = FunctionHook

# ---------------------------------------------------------------------------
# Minimal config helpers
# ---------------------------------------------------------------------------

_MINIMAL_LLM = {
    "name": "test-provider",
    "type": "vllm",
    "model_name": "test-model",
    "url": "http://localhost:8000/v1/chat/completions",
}

_MINIMAL_PLANNER = {
    "name": "react",
    "type": "react",
}


def _agent_cfg(**extra) -> dict:
    base = {
        "name": "TestAgent",
        "description": "Test agent",
        "instruction": "You are a test agent",
        "llm_provider": "test-provider",
        "planner_strategy": "react",
    }
    base.update(extra)
    return base


def _base_config(**extra) -> dict:
    cfg: dict = {
        "version": "1.0",
        "entrypoint": "TestAgent",
        "llm_providers": [_MINIMAL_LLM],
        "planner_strategies": [_MINIMAL_PLANNER],
        "agents": [_agent_cfg()],
    }
    cfg.update(extra)
    return copy.deepcopy(cfg)


# ---------------------------------------------------------------------------
# Hook functions - defined at module level so they can be imported by path.
# The test file itself is used as the "path" for path-based hook loading.
# ---------------------------------------------------------------------------


def _pre_agent_listener_fn(context, agent, **kwargs):
    """Listener pre-agent hook."""


def _post_agent_listener_fn(context, agent, **kwargs):
    """Listener post-agent hook."""


def _pre_agent_updater_fn(context, agent, **kwargs):
    """Updater pre-agent hook - must return new_context."""
    return {"new_context": context}


def _post_agent_updater_fn(context, agent, **kwargs):
    """Updater post-agent hook - must return new_context."""
    return {"new_context": context}


def _pre_tool_listener_fn(tool, tool_args, **kwargs):
    """Listener pre-tool hook."""


def _post_tool_listener_fn(tool, result, **kwargs):
    """Listener post-tool hook."""


def _pre_tool_updater_fn(tool, tool_args, **kwargs):
    """Updater pre-tool hook."""
    return None


def _post_tool_updater_fn(tool, result, **kwargs):
    """Updater post-tool hook."""
    return None


def _pre_tool_approval_fn(tool, tool_args, **kwargs):
    """Approval hook: leaves the decision to the human."""
    return None


def _on_agent_event_listener_fn(event, context, **kwargs):
    """Listener on_agent_event hook."""


def _on_agent_event_updater_fn(event, context, **kwargs):
    """Updater on_agent_event hook - pass through."""
    return event


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def clean_hook_registries():
    """Clear hook registries before and after every test."""
    get_hook_registry().clear()
    yield
    get_hook_registry().clear()


# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

# This file is the source module for path-based hook loading in tests.
_THIS_FILE = "tests/loader/test_config_hooks.py"
_TOOLS_PATH = "tests/loader/customer_service_tools.py"


def _tool_cfg_with_hook(hook_name: str) -> dict:
    return {
        "name": "get_order_info",
        "type": "function",
        "path": _TOOLS_PATH,
        "hooks": [hook_name],
    }


# ===========================================================================
# 1. NON-REGISTERED (PATH-BASED) HOOKS
#    Not in any registry → MUST carry a "trigger" field in the dict.
# ===========================================================================


class TestNonRegisteredAgentHooks:
    """Hooks loaded from a Python file path - not pre-registered."""

    def test_listener_pre_agent_hook_loaded(self):
        cfg = _base_config(
            hooks=[
                {
                    "name": "_pre_agent_listener_fn",
                    "type": "listener",
                    "trigger": "pre_agent_call",
                    "path": _THIS_FILE,
                }
            ],
            agents=[_agent_cfg(hooks=["_pre_agent_listener_fn"])],
        )
        result = ConfigurationV1.from_dict(cfg)

        agent = result.agents[0]
        assert len(agent.hooks) == 1
        h = agent.hooks[0]
        assert isinstance(h, FunctionAgentHook)
        assert h.trigger == HookPhase.PRE_AGENT_CALL
        assert h.type == HookType.LISTENER
        assert h.name == "_pre_agent_listener_fn"

    def test_listener_post_agent_hook_loaded(self):
        cfg = _base_config(
            hooks=[
                {
                    "name": "_post_agent_listener_fn",
                    "type": "listener",
                    "trigger": "post_agent_call",
                    "path": _THIS_FILE,
                }
            ],
            agents=[_agent_cfg(hooks=["_post_agent_listener_fn"])],
        )
        result = ConfigurationV1.from_dict(cfg)

        h = result.agents[0].hooks[0]
        assert isinstance(h, FunctionAgentHook)
        assert h.trigger == HookPhase.POST_AGENT_CALL

    def test_updater_pre_agent_hook_loaded(self):
        cfg = _base_config(
            hooks=[
                {
                    "name": "_pre_agent_updater_fn",
                    "type": "updater",
                    "trigger": "pre_agent_call",
                    "path": _THIS_FILE,
                }
            ],
            agents=[_agent_cfg(hooks=["_pre_agent_updater_fn"])],
        )
        result = ConfigurationV1.from_dict(cfg)

        h = result.agents[0].hooks[0]
        assert isinstance(h, FunctionAgentHook)
        assert h.trigger == HookPhase.PRE_AGENT_CALL
        assert h.type == HookType.UPDATER

    def test_updater_post_agent_hook_loaded(self):
        cfg = _base_config(
            hooks=[
                {
                    "name": "_post_agent_updater_fn",
                    "type": "updater",
                    "trigger": "post_agent_call",
                    "path": _THIS_FILE,
                }
            ],
            agents=[_agent_cfg(hooks=["_post_agent_updater_fn"])],
        )
        result = ConfigurationV1.from_dict(cfg)

        h = result.agents[0].hooks[0]
        assert isinstance(h, FunctionAgentHook)
        assert h.trigger == HookPhase.POST_AGENT_CALL

    def test_pre_and_post_hooks_both_attached(self):
        """Both pre- and post-agent hooks can coexist on the same agent."""
        cfg = _base_config(
            hooks=[
                {
                    "name": "_pre_agent_listener_fn",
                    "type": "listener",
                    "trigger": "pre_agent_call",
                    "path": _THIS_FILE,
                },
                {
                    "name": "_post_agent_listener_fn",
                    "type": "listener",
                    "trigger": "post_agent_call",
                    "path": _THIS_FILE,
                },
            ],
            agents=[_agent_cfg(hooks=["_pre_agent_listener_fn", "_post_agent_listener_fn"])],
        )
        result = ConfigurationV1.from_dict(cfg)

        hooks = result.agents[0].hooks
        assert len(hooks) == 2
        phases = {h.trigger for h in hooks}
        assert HookPhase.PRE_AGENT_CALL in phases
        assert HookPhase.POST_AGENT_CALL in phases

    def test_hooks_appear_in_config_hooks_list(self):
        """Loaded hooks are present in ConfigurationV1.hooks."""
        cfg = _base_config(
            hooks=[
                {
                    "name": "_pre_agent_listener_fn",
                    "type": "listener",
                    "trigger": "pre_agent_call",
                    "path": _THIS_FILE,
                }
            ],
        )
        result = ConfigurationV1.from_dict(cfg)

        assert any(h.name == "_pre_agent_listener_fn" for h in result.hooks)

    def test_missing_trigger_raises_for_unregistered_hook(self):
        """An unregistered hook without a trigger must raise ValueError."""
        cfg = _base_config(
            hooks=[
                {
                    "name": "_pre_agent_listener_fn",
                    "type": "listener",
                    # deliberately omitted "trigger"
                    "path": _THIS_FILE,
                }
            ],
        )
        with pytest.raises(ValueError, match="trigger"):
            ConfigurationV1.from_dict(cfg)

    def test_invalid_trigger_value_raises(self):
        cfg = _base_config(
            hooks=[
                {
                    "name": "_pre_agent_listener_fn",
                    "type": "listener",
                    "trigger": "not_a_valid_phase",
                    "path": _THIS_FILE,
                }
            ],
        )
        with pytest.raises(ValueError, match="Invalid trigger"):
            ConfigurationV1.from_dict(cfg)

    def test_unknown_hook_name_on_agent_raises(self):
        """Referencing an unknown hook name in an agent's hooks list raises."""
        cfg = _base_config(
            agents=[_agent_cfg(hooks=["nonexistent_hook"])],
        )
        with pytest.raises(ValueError, match="nonexistent_hook"):
            ConfigurationV1.from_dict(cfg)


class TestNonRegisteredToolHooks:
    """Tool hooks loaded from a file path - not pre-registered."""

    def test_listener_pre_tool_hook_loaded(self):
        cfg = _base_config(
            hooks=[
                {
                    "name": "_pre_tool_listener_fn",
                    "type": "listener",
                    "trigger": "pre_tool_call",
                    "path": _THIS_FILE,
                }
            ],
            tools=[_tool_cfg_with_hook("_pre_tool_listener_fn")],
            agents=[_agent_cfg(tools=["get_order_info"])],
        )
        result = ConfigurationV1.from_dict(cfg)

        h = result.tools[0].hooks[0]
        assert isinstance(h, FunctionToolPreHook)
        assert h.trigger == HookPhase.PRE_TOOL_CALL
        assert h.type == HookType.LISTENER

    def test_updater_pre_tool_hook_loaded(self):
        cfg = _base_config(
            hooks=[
                {
                    "name": "_pre_tool_updater_fn",
                    "type": "updater",
                    "trigger": "pre_tool_call",
                    "path": _THIS_FILE,
                }
            ],
            tools=[_tool_cfg_with_hook("_pre_tool_updater_fn")],
            agents=[_agent_cfg(tools=["get_order_info"])],
        )
        result = ConfigurationV1.from_dict(cfg)

        h = result.tools[0].hooks[0]
        assert isinstance(h, FunctionToolPreHook)
        assert h.trigger == HookPhase.PRE_TOOL_CALL
        assert h.type == HookType.UPDATER

    def test_listener_post_tool_hook_loaded(self):
        cfg = _base_config(
            hooks=[
                {
                    "name": "_post_tool_listener_fn",
                    "type": "listener",
                    "trigger": "post_tool_call",
                    "path": _THIS_FILE,
                }
            ],
            tools=[_tool_cfg_with_hook("_post_tool_listener_fn")],
            agents=[_agent_cfg(tools=["get_order_info"])],
        )
        result = ConfigurationV1.from_dict(cfg)

        h = result.tools[0].hooks[0]
        assert isinstance(h, FunctionToolPostHook)
        assert h.trigger == HookPhase.POST_TOOL_CALL
        assert h.type == HookType.LISTENER

    def test_updater_post_tool_hook_loaded(self):
        cfg = _base_config(
            hooks=[
                {
                    "name": "_post_tool_updater_fn",
                    "type": "updater",
                    "trigger": "post_tool_call",
                    "path": _THIS_FILE,
                }
            ],
            tools=[_tool_cfg_with_hook("_post_tool_updater_fn")],
            agents=[_agent_cfg(tools=["get_order_info"])],
        )
        result = ConfigurationV1.from_dict(cfg)

        h = result.tools[0].hooks[0]
        assert isinstance(h, FunctionToolPostHook)
        assert h.trigger == HookPhase.POST_TOOL_CALL
        assert h.type == HookType.UPDATER

    def test_pre_tool_approval_hook_is_attached_to_the_tool(self):
        cfg = _base_config(
            hooks=[
                {
                    "name": "_pre_tool_approval_fn",
                    "type": "updater",
                    "trigger": "pre_tool_approval",
                    "path": _THIS_FILE,
                }
            ],
            tools=[_tool_cfg_with_hook("_pre_tool_approval_fn")],
            agents=[_agent_cfg(tools=["get_order_info"])],
        )
        result = ConfigurationV1.from_dict(cfg)

        tool = result.tools[0]
        assert [h.name for h in tool.approval_hooks] == ["_pre_tool_approval_fn"]
        assert tool.pre_hooks == []

    def test_missing_trigger_raises_for_unregistered_tool_hook(self):
        cfg = _base_config(
            hooks=[
                {
                    "name": "_pre_tool_listener_fn",
                    "type": "listener",
                    # deliberately omitted "trigger"
                    "path": _THIS_FILE,
                }
            ],
        )
        with pytest.raises(ValueError, match="trigger"):
            ConfigurationV1.from_dict(cfg)

    def test_unknown_tool_hook_name_on_tool_raises(self):
        cfg = _base_config(
            tools=[_tool_cfg_with_hook("nonexistent_tool_hook")],
            agents=[_agent_cfg(tools=["get_order_info"])],
        )
        with pytest.raises(ValueError, match="nonexistent_tool_hook"):
            ConfigurationV1.from_dict(cfg)


# ===========================================================================
# 2. REGISTERED HOOKS (via @hook decorator)
#
#    Key behaviour in config.py (post-fix):
#      • In the top-level "hooks" list, the registry check uses
#        hook_config["reference"] - so a "reference" key is required to
#        bypass the trigger requirement.
#      • When referenced directly from an agent's / tool's hooks list (by
#        name only), the loader falls back to the registry automatically -
#        no "reference" key needed there.
#      • In both paths the trigger is taken from the registry object itself.
# ===========================================================================


class TestRegisteredAgentHooks:
    """Hooks pre-registered via @hook decorator."""

    # ------------------------------------------------------------------
    # Helpers that register and return the hook name
    # ------------------------------------------------------------------

    def _reg_pre_listener(self, name: str = "reg_pre_agent_listener") -> str:
        return hook(trigger=HookPhase.PRE_AGENT_CALL, name=name, is_listener=True)(
            _pre_agent_listener_fn
        ).name

    def _reg_post_listener(self, name: str = "reg_post_agent_listener") -> str:
        return hook(trigger=HookPhase.POST_AGENT_CALL, name=name, is_listener=True)(
            _post_agent_listener_fn
        ).name

    def _reg_pre_updater(self, name: str = "reg_pre_agent_updater") -> str:
        return hook(trigger=HookPhase.PRE_AGENT_CALL, name=name, is_listener=False)(
            _pre_agent_updater_fn
        ).name

    # ------------------------------------------------------------------
    # Tests
    # ------------------------------------------------------------------

    def test_registered_hook_in_top_level_list_without_trigger(self):
        """A top-level hook entry with a 'reference' pointing to a registered
        hook does NOT require a 'trigger' field - trigger is taken from the
        registry object."""
        hook_name = self._reg_pre_listener()

        cfg = _base_config(
            hooks=[
                {
                    "name": hook_name,
                    "reference": hook_name,
                    # NO "trigger" key - taken from the registry object
                }
            ],
            agents=[_agent_cfg(hooks=[hook_name])],
        )
        result = ConfigurationV1.from_dict(cfg)

        agent = result.agents[0]
        assert len(agent.hooks) == 1
        h = agent.hooks[0]
        assert isinstance(h, FunctionAgentHook)
        assert h.trigger == HookPhase.PRE_AGENT_CALL
        assert h.name == hook_name

    def test_registered_hook_resolved_directly_without_top_level_entry(self):
        """A registered hook can be referenced by name in an agent's hooks list
        without any top-level 'hooks' section entry - the loader falls back to
        the agent hook registry."""
        hook_name = self._reg_pre_listener("direct_pre_agent")

        cfg = _base_config(
            # No top-level "hooks" key at all
            agents=[_agent_cfg(hooks=[hook_name])],
        )
        result = ConfigurationV1.from_dict(cfg)

        agent = result.agents[0]
        assert len(agent.hooks) == 1
        assert agent.hooks[0].name == hook_name
        assert isinstance(agent.hooks[0], FunctionAgentHook)

    def test_registered_updater_hook_trigger_taken_from_registry(self):
        """Trigger on the loaded hook reflects the registry object."""
        hook_name = self._reg_pre_updater()

        cfg = _base_config(
            hooks=[{"name": hook_name, "reference": hook_name}],  # no trigger in dict
            agents=[_agent_cfg(hooks=[hook_name])],
        )
        result = ConfigurationV1.from_dict(cfg)

        h = result.agents[0].hooks[0]
        assert isinstance(h, FunctionAgentHook)
        assert h.trigger == HookPhase.PRE_AGENT_CALL

    def test_registered_post_hook_trigger_taken_from_registry(self):
        hook_name = self._reg_post_listener()

        cfg = _base_config(
            hooks=[{"name": hook_name, "reference": hook_name}],  # no trigger in dict
            agents=[_agent_cfg(hooks=[hook_name])],
        )
        result = ConfigurationV1.from_dict(cfg)

        h = result.agents[0].hooks[0]
        assert isinstance(h, FunctionAgentHook)
        assert h.trigger == HookPhase.POST_AGENT_CALL

    def test_mix_registered_and_path_based_agent_hooks(self):
        """A registered (reference, no trigger) hook and a path-loaded
        (with trigger) hook can coexist on the same agent."""
        registered_name = self._reg_post_listener("mix_post_reg")

        cfg = _base_config(
            hooks=[
                {
                    "name": "_pre_agent_listener_fn",
                    "type": "listener",
                    "trigger": "pre_agent_call",
                    "path": _THIS_FILE,
                },
                {"name": registered_name, "reference": registered_name},
            ],
            agents=[_agent_cfg(hooks=["_pre_agent_listener_fn", registered_name])],
        )
        result = ConfigurationV1.from_dict(cfg)

        hooks = result.agents[0].hooks
        assert len(hooks) == 2
        phases = {h.trigger for h in hooks}
        assert HookPhase.PRE_AGENT_CALL in phases
        assert HookPhase.POST_AGENT_CALL in phases

    def test_hook_name_only_without_reference_requires_trigger(self):
        """A top-level hooks entry with only 'name' (no 'reference') is treated
        as a non-registered hook and MUST carry a 'trigger' field."""
        self._reg_pre_listener("reg_name_only")  # registered, but no "reference" in dict

        cfg = _base_config(
            hooks=[
                {
                    "name": "reg_name_only",
                    # no "reference" key → loader treats it as non-registered
                    # no "trigger" either → must raise
                }
            ],
        )
        with pytest.raises(ValueError, match="trigger"):
            ConfigurationV1.from_dict(cfg)

    def test_unregistered_hook_without_trigger_raises(self):
        """An unregistered hook without a trigger must raise."""
        cfg = _base_config(
            hooks=[
                {
                    "name": "ghost_hook",
                    # no "reference", no "trigger", not in registry
                }
            ],
        )
        with pytest.raises(ValueError, match="trigger"):
            ConfigurationV1.from_dict(cfg)


class TestRegisteredToolHooks:
    """Registered tool hooks - trigger taken from registry via 'reference' key."""

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _reg_pre_listener(self, name: str = "reg_pre_tool_listener") -> str:
        return hook(trigger=HookPhase.PRE_TOOL_CALL, name=name, is_listener=True)(
            _pre_tool_listener_fn
        ).name

    def _reg_post_listener(self, name: str = "reg_post_tool_listener") -> str:
        return hook(trigger=HookPhase.POST_TOOL_CALL, name=name, is_listener=True)(
            _post_tool_listener_fn
        ).name

    def _reg_pre_updater(self, name: str = "reg_pre_tool_updater") -> str:
        return hook(trigger=HookPhase.PRE_TOOL_CALL, name=name, is_listener=False)(
            _pre_tool_updater_fn
        ).name

    def _reg_post_updater(self, name: str = "reg_post_tool_updater") -> str:
        return hook(trigger=HookPhase.POST_TOOL_CALL, name=name, is_listener=False)(
            _post_tool_updater_fn
        ).name

    # ------------------------------------------------------------------
    # Tests
    # ------------------------------------------------------------------

    def test_registered_pre_tool_hook_via_reference_without_trigger(self):
        """Top-level reference to a registered pre-tool hook - no trigger needed."""
        hook_name = self._reg_pre_listener()

        cfg = _base_config(
            hooks=[{"name": hook_name, "reference": hook_name}],  # no trigger
            tools=[_tool_cfg_with_hook(hook_name)],
            agents=[_agent_cfg(tools=["get_order_info"])],
        )
        result = ConfigurationV1.from_dict(cfg)

        h = result.tools[0].hooks[0]
        assert isinstance(h, FunctionToolPreHook)
        assert h.trigger == HookPhase.PRE_TOOL_CALL
        assert h.name == hook_name

    def test_registered_post_tool_hook_via_reference_without_trigger(self):
        hook_name = self._reg_post_listener()

        cfg = _base_config(
            hooks=[{"name": hook_name, "reference": hook_name}],
            tools=[_tool_cfg_with_hook(hook_name)],
            agents=[_agent_cfg(tools=["get_order_info"])],
        )
        result = ConfigurationV1.from_dict(cfg)

        h = result.tools[0].hooks[0]
        assert isinstance(h, FunctionToolPostHook)
        assert h.trigger == HookPhase.POST_TOOL_CALL

    def test_registered_updater_pre_tool_hook(self):
        hook_name = self._reg_pre_updater()

        cfg = _base_config(
            hooks=[{"name": hook_name, "reference": hook_name}],
            tools=[_tool_cfg_with_hook(hook_name)],
            agents=[_agent_cfg(tools=["get_order_info"])],
        )
        result = ConfigurationV1.from_dict(cfg)

        h = result.tools[0].hooks[0]
        assert isinstance(h, FunctionToolPreHook)
        assert h.type == HookType.UPDATER

    def test_registered_updater_post_tool_hook(self):
        hook_name = self._reg_post_updater()

        cfg = _base_config(
            hooks=[{"name": hook_name, "reference": hook_name}],
            tools=[_tool_cfg_with_hook(hook_name)],
            agents=[_agent_cfg(tools=["get_order_info"])],
        )
        result = ConfigurationV1.from_dict(cfg)

        h = result.tools[0].hooks[0]
        assert isinstance(h, FunctionToolPostHook)
        assert h.type == HookType.UPDATER

    def test_registered_tool_hook_resolved_directly_without_top_level_entry(self):
        """A registered tool hook referenced by name in a tool's hooks list is
        resolved from the tool hook registry without needing a top-level entry."""
        hook_name = self._reg_pre_listener("direct_tool_hook")

        cfg = _base_config(
            # No top-level "hooks" key at all
            tools=[_tool_cfg_with_hook(hook_name)],
            agents=[_agent_cfg(tools=["get_order_info"])],
        )
        result = ConfigurationV1.from_dict(cfg)

        assert result.tools[0].hooks[0].name == hook_name
        assert isinstance(result.tools[0].hooks[0], FunctionToolPreHook)

    def test_mix_registered_and_path_based_tool_hooks(self):
        """A tool can have both a registered (reference) hook and a path-loaded hook."""
        registered_name = self._reg_pre_listener("mix_pre_reg_tool")

        cfg = _base_config(
            hooks=[
                {
                    "name": "_post_tool_listener_fn",
                    "type": "listener",
                    "trigger": "post_tool_call",
                    "path": _THIS_FILE,
                },
                {"name": registered_name, "reference": registered_name},
            ],
            tools=[
                {
                    "name": "get_order_info",
                    "type": "function",
                    "path": _TOOLS_PATH,
                    "hooks": [registered_name, "_post_tool_listener_fn"],
                }
            ],
            agents=[_agent_cfg(tools=["get_order_info"])],
        )
        result = ConfigurationV1.from_dict(cfg)

        tool_hooks = result.tools[0].hooks
        assert len(tool_hooks) == 2
        hook_types = {type(h) for h in tool_hooks}
        assert FunctionToolPreHook in hook_types
        assert FunctionToolPostHook in hook_types


# ===========================================================================
# 3. REFERENCE-STYLE HOOKS (explicit "reference" field in the dict entry)
#    config.py checks hook_config["reference"] against registered_hooks.
#    The registry object carries its own trigger - no trigger in dict needed.
#    When "name" differs from "reference" the hook is aliased.
# ===========================================================================


class TestReferenceStyleHooks:
    def test_agent_hook_reference_same_as_name(self):
        """Simple reference where name == reference - no trigger required."""
        hook(trigger=HookPhase.PRE_AGENT_CALL, name="ref_pre_agent", is_listener=True)(
            _pre_agent_listener_fn
        )

        cfg = _base_config(
            hooks=[
                {
                    "name": "ref_pre_agent",
                    "reference": "ref_pre_agent",
                    # no trigger - trigger comes from the registry object
                }
            ],
            agents=[_agent_cfg(hooks=["ref_pre_agent"])],
        )
        result = ConfigurationV1.from_dict(cfg)

        h = result.agents[0].hooks[0]
        assert isinstance(h, FunctionAgentHook)
        assert h.name == "ref_pre_agent"
        assert h.trigger == HookPhase.PRE_AGENT_CALL

    def test_agent_hook_reference_with_alias(self):
        """When 'name' differs from 'reference' the loader looks up by reference
        in the registry and aliases the hook - no trigger required in the dict."""
        hook(trigger=HookPhase.POST_AGENT_CALL, name="original_post_hook", is_listener=True)(
            _post_agent_listener_fn
        )

        cfg = _base_config(
            hooks=[
                {
                    "name": "aliased_post_hook",
                    "reference": "original_post_hook",
                    # "reference" is in the registry → trigger NOT required
                }
            ],
            agents=[_agent_cfg(hooks=["aliased_post_hook"])],
        )
        result = ConfigurationV1.from_dict(cfg)

        h = result.agents[0].hooks[0]
        assert h.name == "aliased_post_hook"
        assert isinstance(h, FunctionAgentHook)
        assert h.trigger == HookPhase.POST_AGENT_CALL

    def test_tool_hook_reference_resolved_from_registry(self):
        hook(trigger=HookPhase.PRE_TOOL_CALL, name="ref_pre_tool", is_listener=True)(
            _pre_tool_listener_fn
        )

        cfg = _base_config(
            hooks=[
                {
                    "name": "ref_pre_tool",
                    "reference": "ref_pre_tool",
                }
            ],
            tools=[_tool_cfg_with_hook("ref_pre_tool")],
            agents=[_agent_cfg(tools=["get_order_info"])],
        )
        result = ConfigurationV1.from_dict(cfg)

        h = result.tools[0].hooks[0]
        assert isinstance(h, FunctionToolPreHook)
        assert h.name == "ref_pre_tool"

    def test_nonexistent_reference_raises(self):
        """A reference to a name absent from all registries must raise."""
        cfg = _base_config(
            hooks=[
                {
                    "name": "ghost",
                    "reference": "ghost",
                }
            ],
            agents=[_agent_cfg(hooks=["ghost"])],
        )
        with pytest.raises(ValueError, match="ghost"):
            ConfigurationV1.from_dict(cfg)


# ===========================================================================
# 4. REGRESSION: mixed reference + trigger hooks via model_validate
#
#    Previously _is_raw_config only checked hooks[0] for a "trigger" field.
#    If hooks[0] was a reference-style hook (no "trigger") and hooks[1] was a
#    path-based hook (with "trigger"), _is_raw_config returned False and the
#    raw dict bypassed from_dict entirely, causing Pydantic validation errors.
#
#    After the fix, _is_raw_config also accepts a "reference" key on hooks[0]
#    as a signal that the config is raw and must go through from_dict.
# ===========================================================================


class TestMixedHookStylesViaModelValidate:
    """Regression tests for configs with both reference-style and trigger-style
    hooks when validated through ConfigurationV1.model_validate (which calls
    _is_raw_config to decide whether to delegate to from_dict)."""

    def test_reference_hook_first_trigger_hook_second_via_model_validate(self):
        """Reference hook at hooks[0], path-based hook at hooks[1].

        Before the fix, _is_raw_config checked only hooks[0] for a "trigger"
        field.  Since hooks[0] had no "trigger" (only "reference"), the method
        returned False and the raw dict was fed straight into Pydantic - which
        broke because hook dicts aren't valid FunctionHook objects.

        After the fix the "reference" key is also accepted as evidence of a
        raw config, so from_dict is called and both hooks are resolved."""
        hook(trigger=HookPhase.POST_AGENT_CALL, name="mv_ref_post_hook", is_listener=True)(
            _post_agent_listener_fn
        )

        cfg = _base_config(
            hooks=[
                # hooks[0] — reference style, NO trigger
                {"name": "mv_ref_post_hook", "reference": "mv_ref_post_hook"},
                # hooks[1] — path-based, HAS trigger
                {
                    "name": "_pre_agent_listener_fn",
                    "type": "listener",
                    "trigger": "pre_agent_call",
                    "path": _THIS_FILE,
                },
            ],
            agents=[_agent_cfg(hooks=["mv_ref_post_hook", "_pre_agent_listener_fn"])],
        )

        # Must not raise — previously would crash with a Pydantic validation error
        result = ConfigurationV1.model_validate(cfg)

        agent = result.agents[0]
        assert len(agent.hooks) == 2
        phases = {h.trigger for h in agent.hooks}
        assert HookPhase.PRE_AGENT_CALL in phases
        assert HookPhase.POST_AGENT_CALL in phases

    def test_trigger_hook_first_reference_hook_second_via_model_validate(self):
        """Path-based hook at hooks[0], reference hook at hooks[1].

        With a trigger-bearing hooks[0], _is_raw_config correctly returned True
        even before the fix.  This test ensures the ordering doesn't regress."""
        hook(trigger=HookPhase.POST_AGENT_CALL, name="mv_ref_post_hook2", is_listener=True)(
            _post_agent_listener_fn
        )

        cfg = _base_config(
            hooks=[
                # hooks[0] — path-based, HAS trigger (was always detected correctly)
                {
                    "name": "_pre_agent_listener_fn",
                    "type": "listener",
                    "trigger": "pre_agent_call",
                    "path": _THIS_FILE,
                },
                # hooks[1] — reference style, NO trigger
                {"name": "mv_ref_post_hook2", "reference": "mv_ref_post_hook2"},
            ],
            agents=[_agent_cfg(hooks=["_pre_agent_listener_fn", "mv_ref_post_hook2"])],
        )

        result = ConfigurationV1.model_validate(cfg)

        agent = result.agents[0]
        assert len(agent.hooks) == 2
        phases = {h.trigger for h in agent.hooks}
        assert HookPhase.PRE_AGENT_CALL in phases
        assert HookPhase.POST_AGENT_CALL in phases

    def test_only_reference_hooks_via_model_validate(self):
        """All hooks are reference-style (no trigger in any dict entry).

        _is_raw_config must detect the reference key and route to from_dict."""
        hook(trigger=HookPhase.PRE_AGENT_CALL, name="mv_only_ref_pre", is_listener=True)(
            _pre_agent_listener_fn
        )
        hook(trigger=HookPhase.POST_AGENT_CALL, name="mv_only_ref_post", is_listener=True)(
            _post_agent_listener_fn
        )

        cfg = _base_config(
            hooks=[
                {"name": "mv_only_ref_pre", "reference": "mv_only_ref_pre"},
                {"name": "mv_only_ref_post", "reference": "mv_only_ref_post"},
            ],
            agents=[_agent_cfg(hooks=["mv_only_ref_pre", "mv_only_ref_post"])],
        )

        result = ConfigurationV1.model_validate(cfg)

        agent = result.agents[0]
        assert len(agent.hooks) == 2
        phases = {h.trigger for h in agent.hooks}
        assert HookPhase.PRE_AGENT_CALL in phases
        assert HookPhase.POST_AGENT_CALL in phases


# ===========================================================================
# 5. ON_EVENT_TRIGGERS SPECIFICATION
#    on_agent_event hooks can declare which ExecutionEventType values fire them.
#    When omitted the default (all standard event types) is used.
#    Supported for path-based hooks and for registered hooks loaded by reference.
# ===========================================================================


class TestOnEventTriggersFromConfig:
    """on_event_triggers can be specified in YAML/dict hook configs."""

    def test_default_on_event_triggers_used_when_not_specified(self):
        """When on_event_triggers is absent the field keeps its model default."""
        cfg = _base_config(
            hooks=[
                {
                    "name": "_on_agent_event_listener_fn",
                    "type": "listener",
                    "trigger": "on_agent_event",
                    "path": _THIS_FILE,
                }
            ],
            agents=[_agent_cfg(hooks=["_on_agent_event_listener_fn"])],
        )
        result = ConfigurationV1.from_dict(cfg)

        h = result.agents[0].hooks[0]
        assert h.trigger == HookPhase.ON_AGENT_EVENT
        assert ExecutionEventType.USER_INPUT in h.on_event_triggers
        assert ExecutionEventType.RESPONSE in h.on_event_triggers

    def test_custom_on_event_triggers_subset_path_based(self):
        """Specifying on_event_triggers restricts the hook to those event types."""
        cfg = _base_config(
            hooks=[
                {
                    "name": "_on_agent_event_listener_fn",
                    "type": "listener",
                    "trigger": "on_agent_event",
                    "path": _THIS_FILE,
                    "on_event_triggers": ["user_input", "response"],
                }
            ],
            agents=[_agent_cfg(hooks=["_on_agent_event_listener_fn"])],
        )
        result = ConfigurationV1.from_dict(cfg)

        h = result.agents[0].hooks[0]
        assert h.trigger == HookPhase.ON_AGENT_EVENT
        assert h.on_event_triggers == [
            ExecutionEventType.USER_INPUT,
            ExecutionEventType.RESPONSE,
        ]

    def test_single_event_trigger(self):
        """A single event type in on_event_triggers is accepted."""
        cfg = _base_config(
            hooks=[
                {
                    "name": "_on_agent_event_updater_fn",
                    "type": "updater",
                    "trigger": "on_agent_event",
                    "path": _THIS_FILE,
                    "on_event_triggers": ["tool_start"],
                }
            ],
            agents=[_agent_cfg(hooks=["_on_agent_event_updater_fn"])],
        )
        result = ConfigurationV1.from_dict(cfg)

        h = result.agents[0].hooks[0]
        assert h.on_event_triggers == [ExecutionEventType.TOOL_START]

    def test_invalid_event_trigger_value_raises(self):
        """An unrecognised string in on_event_triggers raises ValueError."""
        cfg = _base_config(
            hooks=[
                {
                    "name": "_on_agent_event_listener_fn",
                    "type": "listener",
                    "trigger": "on_agent_event",
                    "path": _THIS_FILE,
                    "on_event_triggers": ["not_a_real_event"],
                }
            ],
            agents=[_agent_cfg(hooks=["_on_agent_event_listener_fn"])],
        )
        with pytest.raises(ValueError, match="not_a_real_event"):
            ConfigurationV1.from_dict(cfg)

    def test_on_event_triggers_override_for_registered_hook_by_reference(self):
        """on_event_triggers in a reference-style dict overrides the registry default."""
        hook(trigger=HookPhase.ON_AGENT_EVENT, name="reg_event_hook", is_listener=True)(
            _on_agent_event_listener_fn
        )

        cfg = _base_config(
            hooks=[
                {
                    "name": "reg_event_hook",
                    "reference": "reg_event_hook",
                    "on_event_triggers": ["agent_start", "agent_end"],
                }
            ],
            agents=[_agent_cfg(hooks=["reg_event_hook"])],
        )
        result = ConfigurationV1.from_dict(cfg)

        h = result.agents[0].hooks[0]
        assert h.trigger == HookPhase.ON_AGENT_EVENT
        assert h.on_event_triggers == [
            ExecutionEventType.AGENT_START,
            ExecutionEventType.AGENT_END,
        ]

    def test_on_event_triggers_with_alias_and_reference(self):
        """on_event_triggers and name alias can be combined for a registered hook."""
        hook(trigger=HookPhase.ON_AGENT_EVENT, name="base_event_hook", is_listener=True)(
            _on_agent_event_listener_fn
        )

        cfg = _base_config(
            hooks=[
                {
                    "name": "aliased_event_hook",
                    "reference": "base_event_hook",
                    "on_event_triggers": ["tool_end"],
                }
            ],
            agents=[_agent_cfg(hooks=["aliased_event_hook"])],
        )
        result = ConfigurationV1.from_dict(cfg)

        h = result.agents[0].hooks[0]
        assert h.name == "aliased_event_hook"
        assert h.on_event_triggers == [ExecutionEventType.TOOL_END]

    def test_on_event_triggers_not_applied_to_non_event_hook(self):
        """on_event_triggers in config for a non-on_agent_event hook is parsed but
        stored on the hook without causing errors (it is simply unused at runtime)."""
        cfg = _base_config(
            hooks=[
                {
                    "name": "_pre_agent_listener_fn",
                    "type": "listener",
                    "trigger": "pre_agent_call",
                    "path": _THIS_FILE,
                    "on_event_triggers": ["user_input"],
                }
            ],
            agents=[_agent_cfg(hooks=["_pre_agent_listener_fn"])],
        )
        result = ConfigurationV1.from_dict(cfg)

        h = result.agents[0].hooks[0]
        assert h.trigger == HookPhase.PRE_AGENT_CALL
        assert h.on_event_triggers == [ExecutionEventType.USER_INPUT]


# ===========================================================================
# 6. SILENT FLAG
#    A hook can be marked "silent" so its own HOOK_START/HOOK_UPDATE/HOOK_END/
#    HOOK_ERROR lifecycle events are not persisted/forwarded (Runner concern),
#    while the flag itself must round-trip correctly through config loading.
# ===========================================================================


class TestSilentFlagFromConfig:
    def test_silent_defaults_to_false_when_absent(self):
        cfg = _base_config(
            hooks=[
                {
                    "name": "_pre_agent_listener_fn",
                    "type": "listener",
                    "trigger": "pre_agent_call",
                    "path": _THIS_FILE,
                }
            ],
            agents=[_agent_cfg(hooks=["_pre_agent_listener_fn"])],
        )
        result = ConfigurationV1.from_dict(cfg)

        assert result.agents[0].hooks[0].silent is False

    def test_silent_true_parsed_for_path_based_hook(self):
        cfg = _base_config(
            hooks=[
                {
                    "name": "_pre_agent_listener_fn",
                    "type": "listener",
                    "trigger": "pre_agent_call",
                    "path": _THIS_FILE,
                    "silent": True,
                }
            ],
            agents=[_agent_cfg(hooks=["_pre_agent_listener_fn"])],
        )
        result = ConfigurationV1.from_dict(cfg)

        assert result.agents[0].hooks[0].silent is True

    def test_silent_true_parsed_for_tool_hook(self):
        cfg = _base_config(
            hooks=[
                {
                    "name": "_pre_tool_listener_fn",
                    "type": "listener",
                    "trigger": "pre_tool_call",
                    "path": _THIS_FILE,
                    "silent": True,
                }
            ],
            tools=[_tool_cfg_with_hook("_pre_tool_listener_fn")],
            agents=[_agent_cfg(tools=["get_order_info"])],
        )
        result = ConfigurationV1.from_dict(cfg)

        assert result.tools[0].hooks[0].silent is True

    def test_silent_override_for_registered_hook_by_reference(self):
        """A hook registered as non-silent can be made silent per use-site via
        a 'silent' key in a reference-style dict entry."""
        hook(trigger=HookPhase.PRE_AGENT_CALL, name="ref_silent_hook", is_listener=True)(
            _pre_agent_listener_fn
        )

        cfg = _base_config(
            hooks=[
                {
                    "name": "ref_silent_hook",
                    "reference": "ref_silent_hook",
                    "silent": True,
                }
            ],
            agents=[_agent_cfg(hooks=["ref_silent_hook"])],
        )
        result = ConfigurationV1.from_dict(cfg)

        h = result.agents[0].hooks[0]
        assert h.silent is True
        # The registry copy itself must remain unaffected by the override.
        assert get_hook_registry().get_hook("ref_silent_hook").silent is False

    def test_registered_hook_silent_true_by_default_carries_through_reference(self):
        """A hook registered with silent=True keeps that value when referenced
        without an explicit 'silent' override."""
        hook(
            trigger=HookPhase.POST_AGENT_CALL,
            name="reg_already_silent",
            is_listener=True,
            silent=True,
        )(_post_agent_listener_fn)

        cfg = _base_config(
            hooks=[{"name": "reg_already_silent", "reference": "reg_already_silent"}],
            agents=[_agent_cfg(hooks=["reg_already_silent"])],
        )
        result = ConfigurationV1.from_dict(cfg)

        assert result.agents[0].hooks[0].silent is True

    def test_hook_decorator_silent_param(self):
        """The @hook-style decorator call accepts and stores silent."""
        h = hook(
            trigger=HookPhase.PRE_TOOL_CALL,
            name="decorator_silent_hook",
            is_listener=True,
            silent=True,
        )(_pre_tool_listener_fn)

        assert h.silent is True
