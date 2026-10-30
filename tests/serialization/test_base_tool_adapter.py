"""Tests for BaseTool and BaseToolAdapter serialization and validation."""

import json
from typing import Any

import pytest

from domyn.agents.core import ApprovalType
from domyn.agents.tools import BaseTool, ConfigParam, FunctionTool, Param, Tool


def sample_function(x: int, y: str = "default") -> str:
    """A sample function for testing.

    Args:
        x (int): An integer parameter
        y (str): A string parameter with default value

    Returns:
        str: A formatted string
    """
    return f"x={x}, y={y}"


def test_validate_python_with_existing_tool_instance():
    """Test that existing BaseTool instances are returned as-is."""
    original_tool = FunctionTool(sample_function)
    result = Tool.validate_python(original_tool)

    assert result is original_tool
    assert result.type == "function"
    assert result.declaration.name == "sample_function"


def test_validate_python_with_function_tool_config():
    """Test creating FunctionTool from dictionary configuration."""
    config = {
        "type": "function",
        "func": sample_function,
        "config_params": [{"name": "y", "type": "str", "value": "test_value"}],
    }

    result = Tool.validate_python(config)

    assert isinstance(result, FunctionTool)
    assert result.type == "function"
    assert result.declaration.name == "sample_function"
    assert len(result.config_params) == 1
    assert result.config_params[0].name == "y"
    assert result.config_params[0].value == "test_value"


def test_validate_python_with_function_tool_direct():
    """Test creating FunctionTool with direct function parameter."""
    config = {"type": "function", "func": sample_function}

    result = Tool.validate_python(config)

    assert isinstance(result, FunctionTool)
    assert result.type == "function"
    assert result.declaration.name == "sample_function"
    assert result.config_params == []


def test_validate_python_with_save_in_registry():
    """Test saving tool in registry during validation."""
    config = {
        "type": "function",
        "func": sample_function,
        "save_in_registry": True,
        "registry_name": "test_registry_tool",
    }

    result = Tool.validate_python(config)

    assert isinstance(result, FunctionTool)
    assert result.type == "function"


def test_validate_python_missing_type_and_reference():
    """Test error when neither type nor reference is provided."""
    config = {"name": "invalid_tool"}

    with pytest.raises(ValueError, match="Tool type or reference must be specified"):
        Tool.validate_python(config)


def test_validate_python_unknown_tool_type():
    """Test error when unknown tool type is provided."""
    config = {"type": "unknown_type", "name": "unknown_tool"}

    with pytest.raises(ValueError, match="Unknown tool type: unknown_type"):
        Tool.validate_python(config)


def test_validate_python_reference_not_found():
    """Test error when referenced tool is not found in registry."""
    config = {"reference": "nonexistent_tool"}

    # The registry may raise KeyError or ValueError depending on implementation
    with pytest.raises((ValueError, KeyError), match=r"Tool.*nonexistent_tool.*not found"):
        Tool.validate_python(config)


def test_from_config_method():
    """Test the from_config class method."""
    config = {"type": "function", "func": sample_function}

    result = Tool.from_config(config)

    assert isinstance(result, FunctionTool)
    assert result.type == "function"
    assert result.declaration.name == "sample_function"


def test_model_json_schema_method():
    """Test the model_json_schema method."""
    schema = Tool.model_json_schema()

    assert isinstance(schema, dict)
    assert "discriminator" in schema
    assert "oneOf" in schema or "anyOf" in schema

    discriminator = schema["discriminator"]
    assert discriminator["propertyName"] == "type"
    assert "mapping" in discriminator


def test_tool_registration_on_subclass_creation():
    """Test that tools are automatically registered when subclassing BaseTool."""
    initial_registry_size = len(BaseTool._registry)

    class TestCustomTool(BaseTool):
        type: str = "test_custom"

        def call(self, **kwargs) -> str:
            return "test result"

    assert len(BaseTool._registry) == initial_registry_size + 1
    assert "test_custom" in BaseTool._registry
    assert BaseTool._registry["test_custom"] == TestCustomTool


def test_multiple_tool_types_in_registry():
    """Test that multiple tool types can coexist in registry."""
    assert "function" in BaseTool._registry
    assert BaseTool._registry["function"] == FunctionTool

    class AnotherTestTool(BaseTool):
        type: str = "another_test"

        def call(self, **kwargs) -> str:
            return "another result"

    assert "another_test" in BaseTool._registry
    assert BaseTool._registry["another_test"] == AnotherTestTool

    assert "function" in BaseTool._registry
    assert "another_test" in BaseTool._registry


def test_tool_type_field_consistency():
    """Test that tool type field matches registry key."""
    for tool_type, tool_class in BaseTool._registry.items():
        if tool_type == "function":
            instance = tool_class(sample_function)
        else:
            try:
                instance = tool_class()
            except Exception:
                continue

        assert instance.type == tool_type


def test_function_tool_serialization_roundtrip():
    """Test complete serialization and deserialization roundtrip."""
    original_tool = FunctionTool(func=sample_function)

    # Serialize to dict
    serialized_dict = original_tool.model_dump()

    if "reference" in serialized_dict:
        del serialized_dict["reference"]

    serialized_dict["type"] = "function"
    serialized_dict["func"] = sample_function

    # Deserialize back
    restored_tool = Tool.validate_python(serialized_dict)

    assert isinstance(restored_tool, FunctionTool)
    assert restored_tool.type == original_tool.type
    assert restored_tool.declaration.name == original_tool.declaration.name
    assert len(restored_tool.config_params) == len(original_tool.config_params)


def test_json_serialization_structure():
    """Test the structure of JSON serialization."""
    tool = FunctionTool(
        func=sample_function, config_params=[ConfigParam(name="json_param", value="json_value")]
    )

    json_str = tool.model_dump_json()
    json_dict = json.loads(json_str)

    assert json_dict["type"] == "function"
    assert "func" in json_dict
    assert len(json_dict["config_params"]) == 1
    assert json_dict["config_params"][0]["name"] == "json_param"
    assert json_dict["config_params"][0]["value"] == "json_value"


def test_function_tool_with_empty_config():
    """Test FunctionTool with minimal configuration."""
    tool = FunctionTool(sample_function)

    serialized = tool.model_dump()
    assert serialized["type"] == "function"
    assert serialized["config_params"] == []
    assert serialized["approval_type"] is None


def test_tool_adapter_discriminator_functionality():
    """Test that the discriminator correctly identifies tool types."""
    configs = [{"type": "function", "func": sample_function}]

    for config in configs:
        result = Tool.validate_python(config)
        assert result.type == config["type"]

        if config["type"] == "function":
            assert isinstance(result, FunctionTool)


def test_tool_validation_with_extra_fields():
    """Test that extra fields in config are handled appropriately."""
    config = {
        "type": "function",
        "func": sample_function,
        "extra_field": "should_be_ignored_or_handled",
    }

    try:
        result = Tool.validate_python(config)
        assert isinstance(result, FunctionTool)
        assert result.type == "function"
    except Exception as e:
        assert "extra_field" in str(e) or "forbidden" in str(e).lower()
