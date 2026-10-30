import pytest

from domyn.agents.llm import VLLMProvider
from domyn.agents.loader import ConfigurationV1
from domyn.agents.registry import get_llm_registry


def test_llm_provider_string_format():
    """Test that string format works for providers defined at level 0."""

    config_dict = {
        "version": "1.0",
        "entrypoint": "TestAgent",
        "llm_providers": [
            {
                "name": "test-provider",
                "type": "vllm",
                "model_name": "test-model",
                "url": "http://test.com",
                "config_params": {"generation_params": {"temperature": 0.7}},
            }
        ],
        "context_managers": [
            {
                "name": "summary",
                "prompting": {"type": "structural_tags"},
                "type": "agents_summary",
                "min_events_for_summary": 5,
                "llm_provider": "test-provider",
            }
        ],
        "planner_strategies": [{"name": "react", "type": "react", "context_manager": "summary"}],
        "agents": [
            {
                "name": "TestAgent",
                "description": "Test agent",
                "instruction": "Test",
                "llm_provider": "test-provider",
                "planner_strategy": "react",
            }
        ],
    }

    config = ConfigurationV1.from_dict(config_dict)
    agent = config.agents[0]

    assert isinstance(agent.llm_provider, VLLMProvider)
    assert agent.llm_provider.generation_params["temperature"] == 0.7


def test_llm_provider_reference_from_registry_with_generation_params_override():
    """Test reference format with generation_params override from registry."""
    # Register a provider
    registry_provider = VLLMProvider(
        name="registry-provider",
        model_name="test-model",
        url="http://test.com",
        generation_params={"temperature": 0.5, "max_tokens": 1000},
    )
    get_llm_registry().register_llm_provider("registry-provider", registry_provider)

    try:
        config_dict = {
            "version": "1.0",
            "entrypoint": "TestAgent",
            "llm_providers": [
                {
                    "name": "custom-provider",
                    "reference": "registry-provider",
                    "config_params": {"generation_params": {"temperature": 0.9}},
                }
            ],
            "context_managers": [
                {
                    "name": "summary",
                    "prompting": {"type": "structural_tags"},
                    "type": "agents_summary",
                    "min_events_for_summary": 5,
                    "llm_provider": "custom-provider",
                }
            ],
            "planner_strategies": [
                {"name": "react", "type": "react", "context_manager": "summary"}
            ],
            "agents": [
                {
                    "name": "TestAgent",
                    "description": "Test agent",
                    "instruction": "Test",
                    "llm_provider": "custom-provider",
                    "planner_strategy": "react",
                }
            ],
        }

        config = ConfigurationV1.from_dict(config_dict)
        agent = config.agents[0]

        # Should have overridden temperature but kept max_tokens
        assert agent.llm_provider.generation_params["temperature"] == 0.9
        assert agent.llm_provider.generation_params["max_tokens"] == 1000
        assert agent.llm_provider.name == "custom-provider"

    finally:
        get_llm_registry().clear()


def test_llm_provider_reference_with_api_key_override():
    """Test reference format with api_key override from registry."""
    # Register a provider
    registry_provider = VLLMProvider(
        name="registry-provider",
        model_name="test-model",
        url="http://test.com",
        api_key="base-key",
    )
    get_llm_registry().register_llm_provider("registry-provider", registry_provider)

    try:
        config_dict = {
            "version": "1.0",
            "entrypoint": "TestAgent",
            "llm_providers": [
                {
                    "name": "custom-provider",
                    "reference": "registry-provider",
                    "config_params": {"api_key": "override-key"},
                }
            ],
            "context_managers": [
                {
                    "name": "summary",
                    "prompting": {"type": "structural_tags"},
                    "type": "agents_summary",
                    "min_events_for_summary": 5,
                    "llm_provider": "custom-provider",
                }
            ],
            "planner_strategies": [
                {"name": "react", "type": "react", "context_manager": "summary"}
            ],
            "agents": [
                {
                    "name": "TestAgent",
                    "description": "Test agent",
                    "instruction": "Test",
                    "llm_provider": "custom-provider",
                    "planner_strategy": "react",
                }
            ],
        }

        config = ConfigurationV1.from_dict(config_dict)
        agent = config.agents[0]

        assert agent.llm_provider.api_key == "override-key"
        assert agent.llm_provider.name == "custom-provider"

    finally:
        get_llm_registry().clear()


def test_llm_provider_reference_with_multiple_overrides():
    """Test reference format with multiple field overrides from registry."""
    # Register a provider
    registry_provider = VLLMProvider(
        name="registry-provider",
        model_name="test-model",
        url="http://test.com",
        api_key="base-key",
        generation_params={"temperature": 0.7, "max_tokens": 2000},
    )
    get_llm_registry().register_llm_provider("registry-provider", registry_provider)

    try:
        config_dict = {
            "version": "1.0",
            "entrypoint": "TestAgent",
            "llm_providers": [
                {
                    "name": "custom-provider",
                    "reference": "registry-provider",
                    "config_params": {
                        "api_key": "override-key",
                        "generation_params": {"temperature": 0.9, "top_p": 0.95},
                    },
                }
            ],
            "context_managers": [
                {
                    "name": "summary",
                    "prompting": {"type": "structural_tags"},
                    "type": "agents_summary",
                    "min_events_for_summary": 5,
                    "llm_provider": "custom-provider",
                }
            ],
            "planner_strategies": [
                {"name": "react", "type": "react", "context_manager": "summary"}
            ],
            "agents": [
                {
                    "name": "TestAgent",
                    "description": "Test agent",
                    "instruction": "Test",
                    "llm_provider": "custom-provider",
                    "planner_strategy": "react",
                }
            ],
        }

        config = ConfigurationV1.from_dict(config_dict)
        agent = config.agents[0]

        # Check api_key override
        assert agent.llm_provider.api_key == "override-key"

        # Check generation_params merge (overridden + kept + new)
        assert agent.llm_provider.generation_params["temperature"] == 0.9  # overridden
        assert agent.llm_provider.generation_params["max_tokens"] == 2000  # kept from base
        assert agent.llm_provider.generation_params["top_p"] == 0.95  # new param added

    finally:
        get_llm_registry().clear()


def test_multiple_agents_with_different_provider_instances():
    """Test that multiple agents can use different provider instances."""
    config_dict = {
        "version": "1.0",
        "entrypoint": "Agent1",
        "llm_providers": [
            {
                "name": "base-provider",
                "type": "vllm",
                "model_name": "test-model",
                "url": "http://test.com",
                "config_params": {"generation_params": {"temperature": 0.7}},
            },
            {
                "name": "analytical-provider",
                "type": "vllm",
                "model_name": "test-model",
                "url": "http://test.com",
                "config_params": {"generation_params": {"temperature": 0.3}},
            },
            {
                "name": "creative-provider",
                "type": "vllm",
                "model_name": "test-model",
                "url": "http://test.com",
                "config_params": {"generation_params": {"temperature": 0.9}},
            },
        ],
        "context_managers": [
            {
                "name": "summary",
                "prompting": {"type": "structural_tags"},
                "type": "agents_summary",
                "min_events_for_summary": 5,
                "llm_provider": "base-provider",
            }
        ],
        "planner_strategies": [{"name": "react", "type": "react", "context_manager": "summary"}],
        "agents": [
            {
                "name": "Agent1",
                "description": "Agent 1",
                "instruction": "Test",
                "llm_provider": "analytical-provider",
                "planner_strategy": "react",
            },
            {
                "name": "Agent2",
                "description": "Agent 2",
                "instruction": "Test",
                "llm_provider": "creative-provider",
                "planner_strategy": "react",
            },
            {
                "name": "Agent3",
                "description": "Agent 3",
                "instruction": "Test",
                "llm_provider": "base-provider",
                "planner_strategy": "react",
            },
        ],
    }

    config = ConfigurationV1.from_dict(config_dict)

    # Each agent should have independent provider with different temperature
    assert config.agents[0].llm_provider.generation_params["temperature"] == 0.3
    assert config.agents[1].llm_provider.generation_params["temperature"] == 0.9
    assert config.agents[2].llm_provider.generation_params["temperature"] == 0.7

    # Verify they are different instances
    assert config.agents[0].llm_provider is not config.agents[1].llm_provider
    assert config.agents[0].llm_provider is not config.agents[2].llm_provider
    assert config.agents[1].llm_provider is not config.agents[2].llm_provider


def test_llm_provider_reference_missing_name_raises_error():
    """Test that reference without 'name' raises error."""
    # Register a provider
    registry_provider = VLLMProvider(
        name="registry-provider",
        model_name="test-model",
        url="http://test.com",
    )
    get_llm_registry().register_llm_provider("registry-provider", registry_provider)

    try:
        config_dict = {
            "version": "1.0",
            "entrypoint": "TestAgent",
            "llm_providers": [
                {
                    "reference": "registry-provider",
                    "config_params": {"generation_params": {"temperature": 0.9}},
                },  # Missing name
            ],
            "context_managers": [
                {
                    "name": "summary",
                    "prompting": {"type": "structural_tags"},
                    "type": "agents_summary",
                    "min_events_for_summary": 5,
                    "llm_provider": "registry-provider",
                }
            ],
            "planner_strategies": [
                {"name": "react", "type": "react", "context_manager": "summary"}
            ],
            "agents": [
                {
                    "name": "TestAgent",
                    "description": "Test agent",
                    "instruction": "Test",
                    "llm_provider": "registry-provider",
                    "planner_strategy": "react",
                }
            ],
        }

        with pytest.raises(ValueError, match="must have 'name' field"):
            ConfigurationV1.from_dict(config_dict)

    finally:
        get_llm_registry().clear()


def test_llm_provider_reference_not_found_raises_error():
    """Test that referencing non-existent registry provider raises error."""
    config_dict = {
        "version": "1.0",
        "entrypoint": "TestAgent",
        "llm_providers": [
            {
                "name": "custom-provider",
                "reference": "non-existent-provider",
                "config_params": {"generation_params": {"temperature": 0.9}},
            }
        ],
        "context_managers": [
            {
                "name": "summary",
                "prompting": {"type": "structural_tags"},
                "type": "agents_summary",
                "min_events_for_summary": 5,
                "llm_provider": "custom-provider",
            }
        ],
        "planner_strategies": [{"name": "react", "type": "react", "context_manager": "summary"}],
        "agents": [
            {
                "name": "TestAgent",
                "description": "Test agent",
                "instruction": "Test",
                "llm_provider": "custom-provider",
                "planner_strategy": "react",
            }
        ],
    }

    with pytest.raises(ValueError, match="not found in registry"):
        ConfigurationV1.from_dict(config_dict)


def test_llm_provider_agent_reference_not_found_raises_error():
    """Test that agent referencing non-existent provider raises error."""
    config_dict = {
        "version": "1.0",
        "entrypoint": "TestAgent",
        "llm_providers": [],
        "context_managers": [
            {
                "name": "full_messages",
                "type": "full_messages",
                "narrow_non_visible_agents": False,
                "prompting": {"type": "structural_tags"},
            }
        ],
        "planner_strategies": [
            {"name": "react", "type": "react", "context_manager": "full_messages"}
        ],
        "agents": [
            {
                "name": "TestAgent",
                "description": "Test agent",
                "instruction": "Test",
                "llm_provider": "non-existent-provider",
                "planner_strategy": "react",
            }
        ],
    }

    with pytest.raises(ValueError, match="not found in configuration or registry"):
        ConfigurationV1.from_dict(config_dict)


def test_llm_provider_dict_at_agent_level_raises_error():
    """Test that dict format at agent level raises error with helpful message."""
    config_dict = {
        "version": "1.0",
        "entrypoint": "TestAgent",
        "llm_providers": [
            {
                "name": "base-provider",
                "type": "vllm",
                "model_name": "test-model",
                "url": "http://test.com",
                "config_params": {"generation_params": {"temperature": 0.7}},
            }
        ],
        "context_managers": [
            {
                "name": "summary",
                "prompting": {"type": "structural_tags"},
                "type": "agents_summary",
                "min_events_for_summary": 5,
                "llm_provider": "base-provider",
            }
        ],
        "planner_strategies": [{"name": "react", "type": "react", "context_manager": "summary"}],
        "agents": [
            {
                "name": "TestAgent",
                "description": "Test agent",
                "instruction": "Test",
                "llm_provider": {
                    "name": "base-provider",
                    "generation_params": {"temperature": 0.9},
                },
                "planner_strategy": "react",
            }
        ],
    }

    with pytest.raises(ValueError, match="must be a string"):
        ConfigurationV1.from_dict(config_dict)


def test_llm_provider_invalid_type_at_agent_level_raises_error():
    """Test that invalid llm_provider type at agent level raises error."""
    config_dict = {
        "version": "1.0",
        "entrypoint": "TestAgent",
        "llm_providers": [
            {
                "name": "base-provider",
                "type": "vllm",
                "model_name": "test-model",
                "url": "http://test.com",
                "config_params": {"generation_params": {"temperature": 0.7}},
            }
        ],
        "context_managers": [
            {
                "name": "summary",
                "prompting": {"type": "structural_tags"},
                "type": "agents_summary",
                "min_events_for_summary": 5,
                "llm_provider": "base-provider",
            }
        ],
        "planner_strategies": [{"name": "react", "type": "react", "context_manager": "summary"}],
        "agents": [
            {
                "name": "TestAgent",
                "description": "Test agent",
                "instruction": "Test",
                "llm_provider": 123,  # Invalid type (int)
                "planner_strategy": "react",
            }
        ],
    }

    with pytest.raises(ValueError, match="must be a string"):
        ConfigurationV1.from_dict(config_dict)


def test_llm_provider_agent_references_registry_directly():
    """Test that agents can reference registry providers directly."""
    # Register a provider
    registry_provider = VLLMProvider(
        name="registry-provider",
        model_name="test-model",
        url="http://test.com",
        generation_params={"temperature": 0.5},
    )
    get_llm_registry().register_llm_provider("registry-provider", registry_provider)

    try:
        config_dict = {
            "version": "1.0",
            "entrypoint": "TestAgent",
            "llm_providers": [],  # No providers defined in config
            "context_managers": [
                {
                    "name": "summary",
                    "prompting": {"type": "structural_tags"},
                    "type": "agents_summary",
                    "min_events_for_summary": 5,
                    "llm_provider": "registry-provider",
                }
            ],
            "planner_strategies": [
                {"name": "react", "type": "react", "context_manager": "summary"}
            ],
            "agents": [
                {
                    "name": "TestAgent",
                    "description": "Test agent",
                    "instruction": "Test",
                    "llm_provider": "registry-provider",  # Direct reference to registry
                    "planner_strategy": "react",
                }
            ],
        }

        config = ConfigurationV1.from_dict(config_dict)
        agent = config.agents[0]

        # Should use provider from registry
        assert agent.llm_provider.generation_params["temperature"] == 0.5
        assert agent.llm_provider.name == "registry-provider"

    finally:
        # Clean up registry
        get_llm_registry().clear()


def test_generation_params_deep_merge():
    """Test that generation_params are deeply merged, not replaced."""
    # Register a provider
    registry_provider = VLLMProvider(
        name="registry-provider",
        model_name="test-model",
        url="http://test.com",
        generation_params={
            "temperature": 0.7,
            "max_tokens": 2000,
            "top_p": 0.9,
            "frequency_penalty": 0.0,
        },
    )
    get_llm_registry().register_llm_provider("registry-provider", registry_provider)

    try:
        config_dict = {
            "version": "1.0",
            "entrypoint": "TestAgent",
            "llm_providers": [
                {
                    "name": "custom-provider",
                    "reference": "registry-provider",
                    "config_params": {
                        "generation_params": {
                            "temperature": 0.95,  # Override
                            "top_k": 50,  # Add new
                        }
                    },
                }
            ],
            "context_managers": [
                {
                    "name": "summary",
                    "prompting": {"type": "structural_tags"},
                    "type": "agents_summary",
                    "min_events_for_summary": 5,
                    "llm_provider": "custom-provider",
                }
            ],
            "planner_strategies": [
                {"name": "react", "type": "react", "context_manager": "summary"}
            ],
            "agents": [
                {
                    "name": "TestAgent",
                    "description": "Test agent",
                    "instruction": "Test",
                    "llm_provider": "custom-provider",
                    "planner_strategy": "react",
                }
            ],
        }

        config = ConfigurationV1.from_dict(config_dict)
        agent = config.agents[0]

        params = agent.llm_provider.generation_params

        # Overridden value
        assert params["temperature"] == 0.95

        # Kept from base
        assert params["max_tokens"] == 2000
        assert params["top_p"] == 0.9
        assert params["frequency_penalty"] == 0.0

        # New value added
        assert params["top_k"] == 50

    finally:
        get_llm_registry().clear()


def test_llm_provider_reference_without_config_params_simple_rename():
    """Test that config_params is optional for reference providers (simple rename/alias)."""
    # Register a provider
    registry_provider = VLLMProvider(
        name="registry-provider",
        model_name="test-model",
        url="http://test.com",
        generation_params={"temperature": 0.7, "max_tokens": 2000},
        api_key="base-key",
    )
    get_llm_registry().register_llm_provider("registry-provider", registry_provider)

    try:
        config_dict = {
            "version": "1.0",
            "entrypoint": "TestAgent",
            "llm_providers": [
                {
                    "name": "my-alias",
                    "reference": "registry-provider",
                    # No config_params - just creating an alias
                }
            ],
            "context_managers": [
                {
                    "name": "summary",
                    "prompting": {"type": "structural_tags"},
                    "type": "agents_summary",
                    "min_events_for_summary": 5,
                    "llm_provider": "my-alias",
                }
            ],
            "planner_strategies": [
                {"name": "react", "type": "react", "context_manager": "summary"}
            ],
            "agents": [
                {
                    "name": "TestAgent",
                    "description": "Test agent",
                    "instruction": "Test",
                    "llm_provider": "my-alias",
                    "planner_strategy": "react",
                }
            ],
        }

        config = ConfigurationV1.from_dict(config_dict)
        agent = config.agents[0]

        # Should have exact same values as registry provider
        assert agent.llm_provider.name == "my-alias"  # Name changed
        assert agent.llm_provider.model_name == "test-model"
        assert agent.llm_provider.generation_params["temperature"] == 0.7
        assert agent.llm_provider.generation_params["max_tokens"] == 2000
        assert agent.llm_provider.api_key == "base-key"

    finally:
        get_llm_registry().clear()


def test_llm_provider_reference_with_empty_config_params():
    """Test that empty config_params works (same as no config_params)."""
    # Register a provider
    registry_provider = VLLMProvider(
        name="registry-provider",
        model_name="test-model",
        url="http://test.com",
        generation_params={"temperature": 0.7},
    )
    get_llm_registry().register_llm_provider("registry-provider", registry_provider)

    try:
        config_dict = {
            "version": "1.0",
            "entrypoint": "TestAgent",
            "llm_providers": [
                {
                    "name": "same-name-provider",
                    "reference": "registry-provider",
                    "config_params": {},  # Empty dict - no overrides
                }
            ],
            "context_managers": [
                {
                    "name": "summary",
                    "prompting": {"type": "structural_tags"},
                    "type": "agents_summary",
                    "min_events_for_summary": 5,
                    "llm_provider": "same-name-provider",
                }
            ],
            "planner_strategies": [
                {"name": "react", "type": "react", "context_manager": "summary"}
            ],
            "agents": [
                {
                    "name": "TestAgent",
                    "description": "Test agent",
                    "instruction": "Test",
                    "llm_provider": "same-name-provider",
                    "planner_strategy": "react",
                }
            ],
        }

        config = ConfigurationV1.from_dict(config_dict)
        agent = config.agents[0]

        # Should have exact same values as registry provider
        assert agent.llm_provider.name == "same-name-provider"
        assert agent.llm_provider.generation_params["temperature"] == 0.7

    finally:
        get_llm_registry().clear()


def test_llm_provider_protected_field_override_raises_error():
    """Test that overriding protected fields via config_params raises error."""
    # Register a provider
    registry_provider = VLLMProvider(
        name="registry-provider",
        model_name="test-model",
        url="http://test.com",
    )
    get_llm_registry().register_llm_provider("registry-provider", registry_provider)

    try:
        config_dict = {
            "version": "1.0",
            "entrypoint": "TestAgent",
            "llm_providers": [
                {
                    "name": "custom-provider",
                    "reference": "registry-provider",
                    "config_params": {
                        "model_name": "different-model",  # Protected field
                        "generation_params": {"temperature": 0.9},
                    },
                }
            ],
            "context_managers": [
                {
                    "name": "summary",
                    "prompting": {"type": "structural_tags"},
                    "type": "agents_summary",
                    "min_events_for_summary": 5,
                    "llm_provider": "custom-provider",
                }
            ],
            "planner_strategies": [
                {"name": "react", "type": "react", "context_manager": "summary"}
            ],
            "agents": [
                {
                    "name": "TestAgent",
                    "description": "Test agent",
                    "instruction": "Test",
                    "llm_provider": "custom-provider",
                    "planner_strategy": "react",
                }
            ],
        }

        with pytest.raises(ValueError, match="Cannot override protected field 'model_name'"):
            ConfigurationV1.from_dict(config_dict)

    finally:
        get_llm_registry().clear()


def test_llm_provider_arbitrary_field_override():
    """Test that arbitrary (non-protected) fields can be overridden via config_params."""

    # Register a provider
    class CustomVLLMProvider(VLLMProvider):
        arbitrary_field: str = "default-value"

    registry_provider = CustomVLLMProvider(
        name="registry-provider",
        model_name="test-model",
        url="http://test.com",
        generation_params={"temperature": 0.7},
    )
    get_llm_registry().register_llm_provider("registry-provider", registry_provider)

    try:
        config_dict = {
            "version": "1.0",
            "entrypoint": "TestAgent",
            "llm_providers": [
                {
                    "name": "custom-provider",
                    "reference": "registry-provider",
                    "config_params": {
                        "api_key": "custom-key",
                        "generation_params": {"temperature": 0.9, "max_tokens": 3000},
                        "arbitrary_field": "custom-value",
                        # Any other field that exists on the provider model
                    },
                }
            ],
            "context_managers": [
                {
                    "name": "summary",
                    "prompting": {"type": "structural_tags"},
                    "type": "agents_summary",
                    "min_events_for_summary": 5,
                    "llm_provider": "custom-provider",
                }
            ],
            "planner_strategies": [
                {"name": "react", "type": "react", "context_manager": "summary"}
            ],
            "agents": [
                {
                    "name": "TestAgent",
                    "description": "Test agent",
                    "instruction": "Test",
                    "llm_provider": "custom-provider",
                    "planner_strategy": "react",
                }
            ],
        }

        config = ConfigurationV1.from_dict(config_dict)
        agent = config.agents[0]

        # Verify overrides were applied
        assert agent.llm_provider.api_key == "custom-key"
        assert agent.llm_provider.generation_params["temperature"] == 0.9
        assert agent.llm_provider.generation_params["max_tokens"] == 3000
        assert agent.llm_provider.arbitrary_field == "custom-value"

    finally:
        get_llm_registry().clear()


def test_llm_provider_duplicate_name_overwrites():
    """Test that duplicate provider names result in overwriting (last one wins)."""
    config_dict = {
        "version": "1.0",
        "entrypoint": "TestAgent",
        "llm_providers": [
            {
                "name": "my-provider",
                "type": "vllm",
                "model_name": "test-model",
                "url": "http://test.com",
                "config_params": {"generation_params": {"temperature": 0.5}},
            },
            {
                "name": "my-provider",  # Same name - should overwrite
                "type": "vllm",
                "model_name": "test-model",
                "url": "http://test.com",
                "config_params": {"generation_params": {"temperature": 0.9}},
            },
        ],
        "context_managers": [
            {
                "name": "summary",
                "prompting": {"type": "structural_tags"},
                "type": "agents_summary",
                "min_events_for_summary": 5,
                "llm_provider": "my-provider",
            }
        ],
        "planner_strategies": [{"name": "react", "type": "react", "context_manager": "summary"}],
        "agents": [
            {
                "name": "TestAgent",
                "description": "Test agent",
                "instruction": "Test",
                "llm_provider": "my-provider",
                "planner_strategy": "react",
            }
        ],
    }

    config = ConfigurationV1.from_dict(config_dict)
    agent = config.agents[0]

    # Should use the last definition (0.9, not 0.5)
    assert agent.llm_provider.generation_params["temperature"] == 0.9


def test_deep_merge_nested_dict_fields():
    """Test that nested dictionary fields are deeply merged, not replaced."""

    # Register a provider with nested dict structure
    class CustomVLLMProvider(VLLMProvider):
        arbitrary_field: dict

    registry_provider = CustomVLLMProvider(
        name="registry-provider",
        model_name="test-model",
        url="http://test.com",
        generation_params={
            "temperature": 0.7,
            "max_tokens": 2000,
            "sampling": {
                "top_p": 0.9,
                "top_k": 50,
                "repetition_penalty": 1.0,
            },
        },
        arbitrary_field={"nested_key": "nested_value", "another_key": 42},
    )
    get_llm_registry().register_llm_provider("registry-provider", registry_provider)

    try:
        config_dict = {
            "version": "1.0",
            "entrypoint": "TestAgent",
            "llm_providers": [
                {
                    "name": "custom-provider",
                    "reference": "registry-provider",
                    "config_params": {
                        "generation_params": {
                            "temperature": 0.95,  # Override top-level param
                            "sampling": {
                                "top_p": 0.95,  # Override nested param
                                "min_tokens": 10,  # Add new nested param
                            },
                        },
                        "arbitrary_field": {
                            "additional_key": True,
                            "another_key": 35,  # Keep existing key
                        },
                    },
                }
            ],
            "context_managers": [
                {
                    "name": "summary",
                    "prompting": {"type": "structural_tags"},
                    "type": "agents_summary",
                    "min_events_for_summary": 5,
                    "llm_provider": "custom-provider",
                }
            ],
            "planner_strategies": [
                {"name": "react", "type": "react", "context_manager": "summary"}
            ],
            "agents": [
                {
                    "name": "TestAgent",
                    "description": "Test agent",
                    "instruction": "Test",
                    "llm_provider": "custom-provider",
                    "planner_strategy": "react",
                }
            ],
        }

        config = ConfigurationV1.from_dict(config_dict)
        agent = config.agents[0]

        params = agent.llm_provider.generation_params

        # Top-level: overridden value
        assert params["temperature"] == 0.95

        # Top-level: kept from base
        assert params["max_tokens"] == 2000

        # Nested: overridden value
        assert params["sampling"]["top_p"] == 0.95

        # Nested: kept from base (deep merge preserves unspecified nested fields)
        assert params["sampling"]["top_k"] == 50
        assert params["sampling"]["repetition_penalty"] == 1.0

        # Nested: new value added
        assert params["sampling"]["min_tokens"] == 10

        additional = agent.llm_provider.arbitrary_field
        assert additional["nested_key"] == "nested_value"
        assert additional["another_key"] == 35
        assert additional["additional_key"] is True

    finally:
        get_llm_registry().clear()
