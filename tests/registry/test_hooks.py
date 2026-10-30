"""Tests for HookRegistry."""

from domyn.agents.hooks import FunctionHook, HookPhase, HookType
from domyn.agents.registry import HookRegistry, get_hook_registry


def dummy_agent_updater(context, agent):
    return context


def dummy_agent_listener(context, agent):
    pass


def dummy_pre_hook_function(tool, tool_args):
    return "pre_hook_result"


def dummy_post_hook_function(tool, result):
    return "post_hook_result"


class TestHookRegistry:
    """Test cases for HookRegistry."""

    def setup_method(self):
        self.registry = HookRegistry()
        self.agent_updater_hook = FunctionHook(
            fun=dummy_agent_updater, type=HookType.UPDATER, trigger=HookPhase.PRE_AGENT_CALL
        )
        self.agent_listener_hook = FunctionHook(
            fun=dummy_agent_listener, type=HookType.LISTENER, trigger=HookPhase.PRE_AGENT_CALL
        )
        self.pre_updater_hook = FunctionHook(
            fun=dummy_pre_hook_function, type=HookType.UPDATER, trigger=HookPhase.PRE_TOOL_CALL
        )
        self.post_updater_hook = FunctionHook(
            fun=dummy_post_hook_function, type=HookType.UPDATER, trigger=HookPhase.POST_TOOL_CALL
        )
        self.pre_listener_hook = FunctionHook(
            fun=dummy_pre_hook_function, type=HookType.LISTENER, trigger=HookPhase.PRE_TOOL_CALL
        )
        self.post_listener_hook = FunctionHook(
            fun=dummy_post_hook_function, type=HookType.LISTENER, trigger=HookPhase.POST_TOOL_CALL
        )

    def test_init(self):
        assert self.registry._hooks == {}

    def test_register_agent_updater_hook(self):
        self.registry.register("agent_updater", self.agent_updater_hook)
        assert "agent_updater" in self.registry._hooks
        assert self.registry._hooks["agent_updater"] == self.agent_updater_hook

    def test_register_agent_listener_hook(self):
        self.registry.register("agent_listener", self.agent_listener_hook)
        assert "agent_listener" in self.registry._hooks
        assert self.registry._hooks["agent_listener"] == self.agent_listener_hook

    def test_register_pre_updater_hook(self):
        self.registry.register("pre_updater", self.pre_updater_hook)
        assert "pre_updater" in self.registry._hooks
        assert self.registry._hooks["pre_updater"] == self.pre_updater_hook

    def test_register_post_updater_hook(self):
        self.registry.register("post_updater", self.post_updater_hook)
        assert "post_updater" in self.registry._hooks
        assert self.registry._hooks["post_updater"] == self.post_updater_hook

    def test_register_pre_listener_hook(self):
        self.registry.register("pre_listener", self.pre_listener_hook)
        assert "pre_listener" in self.registry._hooks
        assert self.registry._hooks["pre_listener"] == self.pre_listener_hook

    def test_register_post_listener_hook(self):
        self.registry.register("post_listener", self.post_listener_hook)
        assert "post_listener" in self.registry._hooks
        assert self.registry._hooks["post_listener"] == self.post_listener_hook

    def test_get_hook_existing(self):
        self.registry.register("test_hook", self.pre_updater_hook)
        result = self.registry.get_hook("test_hook")
        assert result == self.pre_updater_hook

    def test_get_hook_nonexistent(self):
        result = self.registry.get_hook("nonexistent")
        assert result is None

    def test_list_hook_names(self):
        self.registry.register("hook1", self.agent_updater_hook)
        self.registry.register("hook2", self.pre_listener_hook)

        names = self.registry.list_hook_names()
        assert set(names) == {"hook1", "hook2"}

    def test_list_hooks(self):
        self.registry.register("hook1", self.agent_updater_hook)
        self.registry.register("hook2", self.post_listener_hook)

        hooks = self.registry.list_hooks()
        assert len(hooks) == 2
        assert self.agent_updater_hook in hooks
        assert self.post_listener_hook in hooks

    def test_clear(self):
        self.registry.register("test_hook", self.pre_updater_hook)
        assert len(self.registry._hooks) == 1

        self.registry.clear()
        assert len(self.registry._hooks) == 0

    def test_setitem(self):
        self.registry["test_hook"] = self.pre_updater_hook
        assert "test_hook" in self.registry._hooks
        assert self.registry._hooks["test_hook"] == self.pre_updater_hook

    def test_getitem_existing(self):
        self.registry.register("test_hook", self.pre_updater_hook)
        result = self.registry["test_hook"]
        assert result == self.pre_updater_hook

    def test_getitem_nonexistent(self):
        result = self.registry["nonexistent"]
        assert result is None

    def test_all_phases_in_single_registry(self):
        """All hook phases coexist in a single registry."""
        self.registry.register("agent_pre", self.agent_updater_hook)
        self.registry.register("agent_post", self.agent_listener_hook)
        self.registry.register("tool_pre", self.pre_updater_hook)
        self.registry.register("tool_post", self.post_listener_hook)

        assert set(self.registry.list_hook_names()) == {
            "agent_pre",
            "agent_post",
            "tool_pre",
            "tool_post",
        }


class TestGetHookRegistry:
    """Test cases for get_hook_registry function."""

    def test_get_hook_registry_singleton(self):
        registry1 = get_hook_registry()
        registry2 = get_hook_registry()

        assert registry1 is registry2
        assert isinstance(registry1, HookRegistry)

    def test_get_hook_registry_functionality(self):
        registry = get_hook_registry()
        mock_hook = FunctionHook(
            fun=dummy_pre_hook_function, type=HookType.UPDATER, trigger=HookPhase.PRE_TOOL_CALL
        )

        registry.clear()
        registry.register("global_test", mock_hook)
        assert "global_test" in registry.list_hook_names()

        registry.clear()
