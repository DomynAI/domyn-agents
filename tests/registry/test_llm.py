"""Tests for LLMProviderRegistry."""

from collections.abc import AsyncGenerator
from unittest.mock import Mock

import pytest
from pydantic import Field

from domyn.agents.llm import BaseLLMProvider, LLMMessage, LLMResponse, StructuredOutputType
from domyn.agents.registry import LLMProviderRegistry, get_llm_registry


class MockLLMProvider(BaseLLMProvider):
    """Mock LLM provider for testing."""

    supported_structured_outputs: list[StructuredOutputType] | None = Field(
        default=[
            StructuredOutputType.JSON_SCHEMA,
            StructuredOutputType.LLGUIDANCE,
            StructuredOutputType.GBNF,
            StructuredOutputType.LARK,
        ]
    )

    def __init__(self, model_name: str = "test_model", name: str = "test_provider"):
        super().__init__(model_name=model_name, name=name)

    async def generate(self, messages: list[LLMMessage], **kwargs) -> LLMResponse:
        """Mock generate method."""
        return LLMResponse(content="test response", generated_tokens=10, execution_time=0.1)

    async def generate_streaming(
        self, messages: list[LLMMessage], **kwargs
    ) -> AsyncGenerator[LLMResponse, None]:
        """Mock generate method."""
        for _ in range(3):
            yield LLMResponse(content="test gen response", generated_tokens=10, execution_time=0.1)


class TestLLMProviderRegistry:
    """Test cases for LLMProviderRegistry."""

    def setup_method(self):
        """Setup for each test."""
        self.registry = LLMProviderRegistry()
        self.mock_provider = MockLLMProvider()

    def test_init(self):
        """Test registry initialization."""
        assert self.registry._providers == {}

    def test_register_llm_provider(self):
        """Test registering an LLM provider."""
        self.registry.register_llm_provider("test_provider", self.mock_provider)
        assert "test_provider" in self.registry._providers
        assert self.registry._providers["test_provider"] == self.mock_provider

    def test_get_llm_provider_nonexistent(self):
        """Test getting non-existent LLM provider returns None."""
        result = self.registry.get_llm_provider("nonexistent")
        assert result is None

    def test_list_llm_provider_names(self):
        """Test listing LLM provider names."""
        provider1 = MockLLMProvider(name="provider1")
        provider2 = MockLLMProvider(name="provider2")

        self.registry.register_llm_provider("provider1", provider1)
        self.registry.register_llm_provider("provider2", provider2)

        names = self.registry.list_llm_provider_names()
        assert set(names) == {"provider1", "provider2"}

    def test_list_llm_providers(self):
        """Test listing LLM providers."""
        provider1 = MockLLMProvider(name="provider1")
        provider2 = MockLLMProvider(name="provider2")

        self.registry.register_llm_provider("provider1", provider1)
        self.registry.register_llm_provider("provider2", provider2)

        providers = self.registry.list_llm_providers()
        assert len(providers) == 2
        assert provider1 in providers
        assert provider2 in providers

    def test_clear(self):
        """Test clearing all LLM providers."""
        self.registry.register_llm_provider("test_provider", self.mock_provider)
        assert len(self.registry._providers) == 1

        self.registry.clear()
        assert len(self.registry._providers) == 0

    def test_setitem(self):
        """Test __setitem__ method."""
        self.registry["test_provider"] = self.mock_provider
        assert "test_provider" in self.registry._providers
        assert self.registry._providers["test_provider"] == self.mock_provider

    def test_getitem(self):
        """Test __getitem__ method."""
        self.registry.register_llm_provider("test_provider", self.mock_provider)

        result = self.registry["test_provider"]
        assert result == self.mock_provider
        assert isinstance(result, MockLLMProvider)


class TestGetLLMRegistry:
    """Test cases for get_llm_registry function."""

    def test_get_llm_registry_singleton(self):
        """Test that get_llm_registry returns the same instance."""
        registry1 = get_llm_registry()
        registry2 = get_llm_registry()

        assert registry1 is registry2
        assert isinstance(registry1, LLMProviderRegistry)

    def test_get_llm_registry_functionality(self):
        """Test that the global registry works correctly."""
        registry = get_llm_registry()
        mock_provider = MockLLMProvider()

        # Clear any existing state
        registry.clear()

        registry.register_llm_provider("global_test", mock_provider)
        assert "global_test" in registry.list_llm_provider_names()

        # Clean up
        registry.clear()
