from typing import Any

from domyn.agents.llm import BaseLLMProvider


class LLMProviderRegistry:
    """Registry for LLM provider instances."""

    def __init__(self) -> None:
        # Values are instances or zero-argument factories: the decorator
        # registers the class, callers that build one register the instance.
        self._providers: dict[str, BaseLLMProvider | Any] = {}

    def register_llm_provider(self, name: str, provider: BaseLLMProvider | Any) -> None:

        self._providers[name] = provider

    def get_llm_provider(self, name: str) -> BaseLLMProvider | None:
        """Get LLM provider by name, or ``None`` when it is not registered.

        The optional return type is not a new behaviour — this always returned
        ``None`` for an unknown name. It used to be annotated as if it could not,
        which turned every caller's check into apparently dead code.
        """

        _c = self._providers.get(name)
        if isinstance(_c, BaseLLMProvider) or _c is None:
            return _c
        instantiated: BaseLLMProvider = _c()
        return instantiated

    def list_llm_provider_names(self) -> list[str]:
        return list(self._providers.keys())

    def list_llm_providers(self) -> list[BaseLLMProvider]:
        found = (self.get_llm_provider(name) for name in self._providers)
        return [provider for provider in found if provider is not None]

    def clear(self) -> None:
        """Clear all registered LLM providers."""
        self._providers.clear()

    def __setitem__(self, name: str, provider: BaseLLMProvider) -> None:
        self.register_llm_provider(name, provider)

    def __getitem__(self, name: str) -> BaseLLMProvider | None:
        return self.get_llm_provider(name)


_llm_registry = LLMProviderRegistry()


def get_llm_registry() -> LLMProviderRegistry:
    """Get the global LLM provider registry instance."""
    return _llm_registry
