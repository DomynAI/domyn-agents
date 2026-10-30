from enum import Enum
from typing import Any

from domyn.agents.llm.anthropic import AnthropicProvider
from domyn.agents.llm.base import (
    BaseLLMProvider,
    LLMMessage,
    LLMResponse,
    LLMToolCall,
    StructuredOutput,
    StructuredOutputType,
    ToolCallFunction,
)
from domyn.agents.llm.cerebras import CerebrasProvider
from domyn.agents.llm.custom import CustomLLMProvider
from domyn.agents.llm.delegate import DelegateLLMProvider
from domyn.agents.llm.domyn import DomynProvider
from domyn.agents.llm.open_router import OpenRouterProvider
from domyn.agents.llm.openai import OpenAIProvider
from domyn.agents.llm.vllm import VLLMProvider


class ProviderType(Enum):
    DOMYN = "domyn"
    VLLM = "vllm"
    OPEN_ROUTER = "open_router"
    CEREBRAS = "cerebras"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    CUSTOM = "custom"


# Protected fields that cannot be overridden via config_params
PROTECTED_FIELDS = ["type", "model_name", "name", "url", "reference"]


class LLMProviderFactory:
    @staticmethod
    def from_config(config: dict[str, Any], save_in_registry: bool = False) -> "BaseLLMProvider":
        """Create an LLM provider from a configuration dictionary."""
        provider_type = config.get("type")
        if not provider_type:
            raise ValueError("Missing 'type' field in LLM provider configuration")

        config_params = config.get("config_params", {})
        if config_params:
            # Check for protected field overrides in config_params
            for field in config_params:
                if field in PROTECTED_FIELDS:
                    raise ValueError(
                        f"Cannot set protected field '{field}' via config_params. "
                        f"Protected fields are: {', '.join(sorted(PROTECTED_FIELDS))}"
                    )
            # Build factory config by flattening all config_params fields
            factory_config = {k: v for k, v in config.items() if k != "config_params"}

            # Flatten all fields from config_params (not just generation_params and api_key)
            for field, value in config_params.items():
                factory_config[field] = value

            config = factory_config

        provider_config = {k: v for k, v in config.items() if k != "type"}

        if provider_type == ProviderType.DOMYN.value:
            obj = DomynProvider(**provider_config)
        elif provider_type == ProviderType.VLLM.value:
            obj = VLLMProvider(**provider_config)
        elif provider_type == ProviderType.OPEN_ROUTER.value:
            obj = OpenRouterProvider(**provider_config)
        elif provider_type == ProviderType.CEREBRAS.value:
            obj = CerebrasProvider(**provider_config)
        elif provider_type == ProviderType.OPENAI.value:
            obj = OpenAIProvider(**provider_config)
        elif provider_type == ProviderType.ANTHROPIC.value:
            obj = AnthropicProvider(**provider_config)
        elif provider_type == "delegate":
            obj = DelegateLLMProvider(**provider_config)
        elif provider_type == ProviderType.CUSTOM.value:
            obj = CustomLLMProvider(**provider_config)
        else:
            raise ValueError(f"Unknown provider type: {provider_type}")

        if save_in_registry:
            from domyn.agents.registry import get_llm_registry

            get_llm_registry().register_llm_provider(obj.model_name, obj)

        return obj

    @staticmethod
    def get_llm_provider(
        config: dict[str, Any], save_in_registry: bool = False
    ) -> "BaseLLMProvider":
        """Backward-compatible alias for legacy callers/tests."""
        return LLMProviderFactory.from_config(config, save_in_registry=save_in_registry)


__all__ = [
    "AnthropicProvider",
    "BaseLLMProvider",
    "CerebrasProvider",
    "CustomLLMProvider",
    "DelegateLLMProvider",
    "DomynProvider",
    "LLMMessage",
    "LLMProviderFactory",
    "LLMResponse",
    "LLMToolCall",
    "OpenAIProvider",
    "OpenRouterProvider",
    "StructuredOutput",
    "StructuredOutputType",
    "ToolCallFunction",
    "VLLMProvider",
]
