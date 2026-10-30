"""Tests for ToolRegistry."""

from unittest.mock import Mock

import pytest

from domyn.agents.registry import ToolRegistry, get_tool_registry
from domyn.agents.tools import FunctionTool


def dummy_function():
    """Dummy function for testing."""
    return "test_result"


def dummy_function2():
    """Second dummy function for testing."""
    return "test_result2"


class TestToolRegistry:
    """Test cases for ToolRegistry."""

    def setup_method(self):
        """Setup for each test."""
        self.registry = ToolRegistry()
        self.mock_tool = FunctionTool(func=dummy_function)

    def test_init(self):
        """Test registry initialization."""
        assert self.registry._tools == {}

    def test_register_tool(self):
        """Test registering a tool."""
        self.registry.register_tool(self.mock_tool)
        assert self.mock_tool.name in self.registry._tools
        assert self.registry._tools[self.mock_tool.name] == self.mock_tool

    def test_get_tool_existing(self):
        """Test getting an existing tool."""
        self.registry.register_tool(self.mock_tool)
        result = self.registry.get_tool(self.mock_tool.name)
        assert result == self.mock_tool

    def test_get_tool_nonexistent(self):
        """Test getting non-existent tool raises KeyError."""
        with pytest.raises(KeyError, match="Tool 'nonexistent' not found in registry"):
            _ = self.registry.get_tool("nonexistent")

    def test_list_tool_names(self):
        """Test listing tool names."""
        tool1 = FunctionTool(func=dummy_function)
        tool2 = FunctionTool(func=dummy_function2)

        self.registry.register_tool(tool1)
        self.registry.register_tool(tool2)

        names = self.registry.list_tool_names()
        assert tool1.name in names
        assert tool2.name in names

    def test_list_tools(self):
        """Test listing tools."""
        tool1 = FunctionTool(func=dummy_function)
        tool2 = FunctionTool(func=dummy_function2)

        self.registry.register_tool(tool1)
        self.registry.register_tool(tool2)

        tools = self.registry.list_tools()
        assert len(tools) == 2
        assert tool1 in tools
        assert tool2 in tools

    def test_clear(self):
        """Test clearing all tools."""
        self.registry.register_tool(self.mock_tool)
        assert len(self.registry._tools) == 1

        self.registry.clear()
        assert len(self.registry._tools) == 0

    def test_getitem_existing(self):
        """Test __getitem__ method with existing tool."""
        self.registry.register_tool(self.mock_tool)
        result = self.registry[self.mock_tool.name]
        assert result == self.mock_tool

    def test_getitem_nonexistent(self):
        """Test __getitem__ method with non-existent tool raises KeyError."""
        with pytest.raises(KeyError, match="Tool 'nonexistent' not found in registry"):
            _ = self.registry["nonexistent"]

    def test_setitem(self):
        """Test __setitem__ method."""
        tool_name = "test_tool"
        self.registry[tool_name] = self.mock_tool
        assert tool_name in self.registry._tools
        assert self.registry._tools[tool_name] == self.mock_tool


class TestGetToolRegistry:
    """Test cases for get_tool_registry function."""

    def test_get_tool_registry_singleton(self):
        """Test that get_tool_registry returns the same instance."""
        registry1 = get_tool_registry()
        registry2 = get_tool_registry()

        assert registry1 is registry2
        assert isinstance(registry1, ToolRegistry)

    def test_get_tool_registry_functionality(self):
        """Test that the global registry works correctly."""
        registry = get_tool_registry()
        mock_tool = FunctionTool(func=dummy_function)

        # Clear any existing state
        registry.clear()

        registry.register_tool(mock_tool)
        assert mock_tool.name in registry.list_tool_names()

        # Clean up
        registry.clear()
