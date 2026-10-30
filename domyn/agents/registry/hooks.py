from domyn.agents.hooks import FunctionHook


class HookRegistry:
    """Registry for hook functions (agent and tool hooks)."""

    def __init__(self):
        self._hooks: dict[str, FunctionHook] = {}

    def register(self, name: str, hook: FunctionHook) -> None:
        self._hooks[name] = hook

    def get_hook(self, name: str) -> FunctionHook | None:
        return self._hooks.get(name, None)

    def list_hook_names(self) -> list[str]:
        return list(self._hooks.keys())

    def list_hooks(self) -> list[FunctionHook]:
        return list(self._hooks.values())

    def clear(self) -> None:
        self._hooks.clear()

    def __setitem__(self, name: str, hook: FunctionHook) -> None:
        self.register(name, hook)

    def __getitem__(self, name: str) -> FunctionHook | None:
        return self.get_hook(name)


_hook_registry = HookRegistry()


def get_hook_registry() -> HookRegistry:
    return _hook_registry
