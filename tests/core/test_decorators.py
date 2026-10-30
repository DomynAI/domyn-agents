"""Comprehensive tests for all decorator types in domyn.agents."""

import logging
from collections.abc import AsyncGenerator

import pytest

from domyn.agents import BaseEvent, ExecutionEventType
from domyn.agents.agents import Agent, ContextVisibility, WrappedAgent

# Import only what we need for testing decorators without triggering the complex dependency chain
from domyn.agents.core import BaseAgent, InteractionContext
from domyn.agents.core.decorators import (
    agent,
    hook,
    llm_provider,
    tool,
)
from domyn.agents.hooks import FunctionHook, HookPhase, HookResult
from domyn.agents.llm import (
    BaseLLMProvider,
    LLMMessage,
    LLMResponse,
    StructuredOutput,
    StructuredOutputType,
)
from domyn.agents.registry import (
    AgentRegistry,
    HookRegistry,
    LLMProviderRegistry,
    ToolRegistry,
    get_agent_registry,
    get_hook_registry,
    get_llm_registry,
    get_tool_registry,
)
from domyn.agents.tools import FunctionTool, StreamableFunctionTool, ToolResult, ToolType


class TestToolDecorator:
    """Test cases for @tool decorator."""

    def setup_method(self):
        """Clear registry before each test."""
        registry: ToolRegistry = get_tool_registry()
        registry.clear()

    @pytest.mark.asyncio
    async def test_basic_tool_registration(self):
        """Test basic tool registration with default parameters."""

        @tool()
        def add_numbers(a: int, b: int) -> int:
            """Add two numbers together."""
            return a + b

        registry: ToolRegistry = get_tool_registry()

        assert add_numbers(4, 5) == 9
        # Check that tool is registered
        assert "add_numbers" in registry.list_tool_names()

        # Check that the registered tool is a FunctionTool
        registered_tool = registry._tools["add_numbers"]
        assert isinstance(registered_tool, FunctionTool)
        assert registered_tool.name == "add_numbers"
        assert registered_tool.description == "Add two numbers together."

        @tool(name="new_name", description="New description")
        def add_numbers(a: int, b: int) -> int:
            """Add two numbers together."""
            return a + b

        assert "new_name" in registry.list_tool_names()
        assert registry["new_name"] == registry.get_tool("new_name")
        assert isinstance(registry["new_name"], FunctionTool)

        assert (await registry.get_tool("new_name").execute(a=2, b=3)).data == registry.get_tool(
            "new_name"
        ).callable_function(2, 3)

        _p = add_numbers
        assert isinstance(_p, FunctionTool)

        @tool(name="new_name_stream", type=ToolType.STREAMABLE_FUNCTION)
        async def stream_data() -> AsyncGenerator[str, None]:
            """Stream data generator."""
            yield "chunk1"
            yield "chunk2"

        assert isinstance(registry.get_tool("new_name_stream"), StreamableFunctionTool)
        assert isinstance(stream_data, StreamableFunctionTool)

        with pytest.raises(TypeError):

            @tool(name="new_name_stream", type=ToolType.STREAMABLE_FUNCTION)
            async def stream_data() -> str:
                """Not a Stream data generator."""
                return "hello"

        with pytest.raises(KeyError):
            registry["non_existent_tool"]

    def test_tool_overwrite_registration(self):
        """Test that re-registering a tool overwrites the previous one."""

        @tool()
        def test_func() -> str:
            """First version."""
            return "v1"

        # Register second version with same name
        @tool()
        def test_func() -> str:  # noqa: F811 — the overwrite under test
            """Second version."""
            return "v2"

        registry = get_tool_registry()

        # Should have only one registration
        assert len([k for k in registry._tools if k == "test_func"]) == 1

        # Should return the latest version
        result = registry["test_func"].callable_function()
        assert result == "v2"


class TestAgentDecorator:
    """Test cases for @agent decorator."""

    def setup_method(self):
        """Clear registry before each test."""
        registry = get_agent_registry()
        registry.clear()

    def test_agent_class_registration(self):
        """Test registration of Agent subclass."""

        @agent()
        class TestAgentCustomClass:
            """Test agent class."""

            async def execute(
                self, context: InteractionContext
            ) -> AsyncGenerator[BaseEvent, None]: ...

        registry: AgentRegistry = get_agent_registry()

        # Check that agent is registered
        assert "TestAgentCustomClass" in registry.list_agent_names()

        # Check that we get a factory method
        agent_obj = registry["TestAgentCustomClass"]
        assert isinstance(agent_obj, WrappedAgent)
        assert not isinstance(
            agent_obj.wrapped_object, type
        )  # Ensure it's an instance, not a class

        # Check that factory creates an instance
        assert agent_obj.wrapped_object.__class__.__name__ == "TestAgentCustomClass"

        # check if the objects are the same
        a = TestAgentCustomClass()
        assert isinstance(a, WrappedAgent)
        assert a is registry["TestAgentCustomClass"]

    def test_agent_with_custom_name(self):
        """Test agent registration with custom name."""

        @agent(name="CustomAgentName")
        class SomeAgent:
            """Some agent class."""

        registry = get_agent_registry()

        # Check registration with custom name
        assert "CustomAgentName" in registry.list_agent_names()
        assert "SomeAgent" not in registry.list_agent_names()

        assert SomeAgent() is registry.get_agent("CustomAgentName")

    def test_wrapped_function_agent(self):
        """Test wrapping a function as an agent."""

        @agent()
        def agent_function(data: str) -> str:
            """Function to be wrapped as agent."""
            return f"Function processed: {data}"

        registry = get_agent_registry()

        # Check that function is registered
        assert "agent_function" in registry.list_agent_names()

        # Check that factory creates a WrappedAgent
        agent_instance = registry.get_agent("agent_function")
        assert isinstance(agent_instance, WrappedAgent)
        assert agent_instance.name == "agent_function"

        assert agent_instance.wrapped_object("input") == "Function processed: input"

        assert agent_function() is registry.get_agent("agent_function")

    def test_wrapped_agents_get_the_settings_they_have(self, caplog):
        """Both kinds keep the WrappedAgent fields given; the class kind used to drop them."""

        @agent(context_visibility=ContextVisibility.GLOBAL)
        class AClass:
            async def execute(self): ...

        @agent(context_visibility=ContextVisibility.GLOBAL)
        async def a_function(context): ...

        assert AClass().context_visibility == ContextVisibility.GLOBAL
        assert a_function().context_visibility == ContextVisibility.GLOBAL
        assert caplog.records == []

    @pytest.mark.parametrize("kind", ["class", "function"])
    def test_wrapped_agents_warn_about_settings_they_ignore(self, caplog, kind):
        with caplog.at_level(logging.WARNING, logger="domyn.agents"):
            if kind == "class":

                @agent(name="Ignoring", llm_provider="p", tools=["t"], instruction="i")
                class Ignoring:
                    async def execute(self): ...

            else:

                @agent(name="Ignoring", llm_provider="p", tools=["t"], instruction="i")
                async def ignoring(context): ...

        assert [r.getMessage() for r in caplog.records] == [
            "@agent 'Ignoring' is not an Agent subclass: ignoring instruction, llm_provider, tools"
        ]
        assert get_agent_registry().get_agent("Ignoring").tools == []


class TestLLMProviderDecorator:
    """Test cases for @llm_provider decorator."""

    def setup_method(self):
        """Clear registry before each test."""
        registry: LLMProviderRegistry = get_llm_registry()
        registry.clear()

    def test_llm_provider_registration(self):
        """Test basic LLM provider registration."""

        @llm_provider()
        class TestLLMProvider(BaseLLMProvider):
            """Test LLM provider."""

            supported_structured_outputs: None = None

            async def generate(
                self,
                messages: list[LLMMessage],
                structured_outputs: StructuredOutput | None,
                **kwargs,
            ) -> LLMResponse:
                return LLMResponse(
                    content="Test response", usage={}, generated_tokens=10, execution_time=0.1
                )

            async def generate_streaming(
                self, messages: list[LLMMessage], **kwargs
            ) -> AsyncGenerator[LLMResponse, None]:
                for _ in range(3):
                    yield LLMResponse(
                        content="Test stream response",
                        usage={},
                        generated_tokens=10,
                        execution_time=0.1,
                    )

        registry = get_llm_registry()

        # Check that provider is registered
        assert "TestLLMProvider" in registry.list_llm_provider_names()

        # Check that factory creates an instance
        assert registry["TestLLMProvider"].__class__.__name__ == "TestLLMProvider"
        assert isinstance(registry["TestLLMProvider"], BaseLLMProvider)

    def test_llm_provider_with_custom_name(self):
        """Test LLM provider registration with custom name."""

        @llm_provider(name="CustomLLM")
        class SomeLLMProvider(BaseLLMProvider):
            """Some LLM provider."""

            async def generate(
                self,
                messages: list[LLMMessage],
                structured_outputs: StructuredOutput | None,
                **kwargs,
            ) -> LLMResponse:
                return LLMResponse(
                    content="Response", usage={}, generated_tokens=5, execution_time=0.05
                )

            async def generate_streaming(
                self, messages: list[LLMMessage], **kwargs
            ) -> AsyncGenerator[LLMResponse, None]:
                for _ in range(3):
                    yield LLMResponse(
                        content="Test stream response",
                        usage={},
                        generated_tokens=10,
                        execution_time=0.1,
                    )

        registry = get_llm_registry()

        # Check registration with custom name
        assert "CustomLLM" in registry.list_llm_provider_names()
        assert "SomeLLMProvider" not in registry.list_llm_provider_names()

    def test_llm_provider_invalid_inheritance(self):
        """Test that decorator raises error for non-BaseLLMProvider classes."""
        with pytest.raises(TypeError, match="must inherit from BaseLLMProvider"):

            @llm_provider()
            class InvalidProvider:
                """Invalid provider not inheriting from BaseLLMProvider."""

    def test_llm_provider_overwrite_registration(self):
        """Test that re-registering an LLM provider overwrites the previous one."""

        @llm_provider()
        class TestLLMProvider(BaseLLMProvider):
            """First version."""

            async def generate(
                self,
                messages: list[LLMMessage],
                structured_outputs: StructuredOutput | None,
                **kwargs,
            ) -> LLMResponse:
                return LLMResponse(content="v1", usage={}, generated_tokens=1, execution_time=0.01)

            async def generate_streaming(
                self, messages: list[LLMMessage], **kwargs
            ) -> AsyncGenerator[LLMResponse, None]:
                for _ in range(3):
                    yield LLMResponse(
                        content="Test stream response",
                        usage={},
                        generated_tokens=10,
                        execution_time=0.1,
                    )

        @llm_provider()
        class TestLLMProvider(BaseLLMProvider):  # noqa: F811 — the overwrite under test
            """Second version."""

            supported_structured_outputs: None = None

            async def generate(
                self,
                messages: list[LLMMessage],
                structured_outputs: StructuredOutput | None,
                **kwargs,
            ) -> LLMResponse:
                return LLMResponse(content="v2", usage={}, generated_tokens=2, execution_time=0.02)

            async def generate_streaming(
                self, messages: list[LLMMessage], **kwargs
            ) -> AsyncGenerator[LLMResponse, None]:
                for _ in range(3):
                    yield LLMResponse(
                        content="Test stream response",
                        usage={},
                        generated_tokens=10,
                        execution_time=0.1,
                    )

        registry = get_llm_registry()

        # Should have only one registration
        assert len([k for k in registry._providers if k == "TestLLMProvider"]) == 1

        # Should create instance of the latest version
        factory = registry._providers["TestLLMProvider"]
        instance = factory()
        # The registered version is identified by its docstring.
        assert not isinstance(instance, TestLLMProvider.__class__)
        assert isinstance(instance, BaseLLMProvider)
        assert instance.__class__.__doc__ == "Second version."

    def test_llm_provider_singleton_behavior(self):
        """Test that LLM provider factory returns the same instance when called multiple times."""

        @llm_provider(model_name="test-model")
        class SingletonLLMProvider(BaseLLMProvider):
            """LLM provider to test singleton behavior."""

            supported_structured_outputs: None = None

            async def generate(
                self,
                messages: list[LLMMessage],
                structured_outputs: StructuredOutput | None,
                **kwargs,
            ) -> LLMResponse:
                return LLMResponse(
                    content="Singleton response", usage={}, generated_tokens=1, execution_time=0.01
                )

            async def generate_streaming(
                self, messages: list[LLMMessage], **kwargs
            ) -> AsyncGenerator[LLMResponse, None]:
                for _ in range(3):
                    yield LLMResponse(
                        content="Test stream response",
                        usage={},
                        generated_tokens=10,
                        execution_time=0.1,
                    )

        registry: LLMProviderRegistry = get_llm_registry()
        # Calling the class returns the registered instance
        assert SingletonLLMProvider() is registry["SingletonLLMProvider"]
        assert isinstance(registry["SingletonLLMProvider"], BaseLLMProvider)
        assert registry["SingletonLLMProvider"].model_name == "test-model"


class TestRegistryConsistency:
    """Test registry behavior across different decorator types."""

    def setup_method(self):
        """Clear all registries before each test."""
        get_tool_registry().clear()
        get_agent_registry().clear()
        get_llm_registry().clear()
        get_hook_registry().clear()

    def test_multiple_decorator_types_coexist(self):
        """Test that different decorator types work together."""

        @tool()
        def test_tool() -> str:
            """Test tool."""
            return "tool_result"

        @agent()
        class TestAgent:
            """Test agent."""

            async def execute(self, context: InteractionContext) -> AsyncGenerator[BaseEvent, None]:
                yield None

        @llm_provider()
        class TestLLM(BaseLLMProvider):
            """Test LLM."""

            supported_structured_outputs: None = None

            async def generate(
                self,
                messages: list[LLMMessage],
                structured_outputs: StructuredOutput | None,
                **kwargs,
            ) -> LLMResponse:
                return LLMResponse(
                    content="test", usage={}, generated_tokens=1, execution_time=0.01
                )

            async def generate_streaming(
                self, messages: list[LLMMessage], **kwargs
            ) -> AsyncGenerator[LLMResponse, None]:
                for _ in range(3):
                    yield LLMResponse(
                        content="Test stream response",
                        usage={},
                        generated_tokens=10,
                        execution_time=0.1,
                    )

        @hook(trigger=HookPhase.PRE_AGENT_CALL)
        def test_hook(context: InteractionContext, agent: Agent) -> InteractionContext:
            """Test hook."""
            return context

        @hook(trigger=HookPhase.POST_TOOL_CALL)
        def test_hook_post(
            c: InteractionContext, tool: FunctionTool, result: ToolResult
        ) -> InteractionContext:
            """Test hook post tool"""
            return c

        @hook(trigger=HookPhase.PRE_TOOL_CALL)
        def test_hook_pre(tool: FunctionTool, tool_args: dict):
            """Test hook pre tool"""
            return {"new_tool_args": tool_args}

        # Check all registries have their respective items
        assert len(get_tool_registry().list_tools()) == 1
        assert len(get_agent_registry().list_agents()) == 1
        assert len(get_llm_registry().list_llm_providers()) == 1
        assert len(get_hook_registry().list_hooks()) == 3

        assert len(get_tool_registry().list_tool_names()) == 1
        assert len(get_agent_registry().list_agent_names()) == 1
        assert len(get_llm_registry().list_llm_provider_names()) == 1
        assert len(get_hook_registry().list_hook_names()) == 3

    def test_registry_isolation(self):
        """Test that registries are isolated from each other."""

        @tool(name="shared_name")
        def tool_func() -> str:
            """Tool with shared name."""
            return "tool"

        @agent(name="shared_name")
        class AgentClass(Agent):
            """Agent with shared name."""

            async def execute(self, context: InteractionContext) -> InteractionContext:
                return context

        # Same name should be allowed in different registries
        assert "shared_name" in get_tool_registry()._tools
        assert "shared_name" in get_agent_registry()._agents

        # Should be different objects
        tool_obj = get_tool_registry()._tools["shared_name"]
        agent_factory = get_agent_registry()._agents["shared_name"]
        assert tool_obj != agent_factory


class TestHookDecorators:
    """Additional tests for hook decorators: agent_hook, tool_pre_hook, tool_post_hook."""

    def setup_method(self):
        get_hook_registry().clear()

    @pytest.mark.asyncio
    async def test_agent_hook_updater_and_listener_registration(self):
        """Ensure agent_hook registers proper class types for updater vs listener."""

        @hook(trigger=HookPhase.PRE_AGENT_CALL, is_listener=False)
        def update_context(
            context: InteractionContext, agent: Agent
        ) -> dict[str, InteractionContext | dict]:  # type: ignore[override]
            context.agent_input.author = "updated_author"
            return {"new_context": context, "event_metadata": {"event_content": "updated"}}

        @hook(trigger=HookPhase.PRE_AGENT_CALL, is_listener=True)
        def listen_context(context: InteractionContext, agent: Agent) -> dict:  # type: ignore[override]
            return {"event_metadata": "listened"}

        registry: HookRegistry = get_hook_registry()
        assert set(registry.list_hook_names()) == {"update_context", "listen_context"}

        # Type assertions
        assert isinstance(registry["update_context"], FunctionHook)
        assert isinstance(registry["listen_context"], FunctionHook)

        # Run the hooks manually using their run methods
        ctx = InteractionContext(
            conversation_id="test_conversation",
            user_id="test_user",
            agent_input=BaseEvent(author="test", event_type=ExecutionEventType.STARTED),
        )  # type: ignore[call-arg]
        agent = BaseAgent(
            name="TestAgent", description=""
        )  # minimal stand-in; hooks treat it opaquely
        # Updater returns a (possibly) modified context
        updated_ctx = None
        async for result in registry["update_context"].run(ctx, agent):
            if isinstance(result, HookResult):
                updated_ctx = result.new_context
        assert getattr(updated_ctx.agent_input, "author", None) == "updated_author"

    def test_tool_pre_and_post_hooks_registration_and_types(self):
        """Validate pre/post tool hooks registration for listener vs updater."""

        @tool()
        def sample_tool() -> str:
            return "ok"

        tool_instance: FunctionTool = get_tool_registry()["sample_tool"]  # type: ignore[assignment]

        @hook(trigger=HookPhase.PRE_TOOL_CALL, is_listener=False)
        def pre_update(tool: FunctionTool, tool_args: dict):  # type: ignore[override]
            return {"new_tool_args": {"pre": True}}

        @hook(trigger=HookPhase.PRE_TOOL_CALL, is_listener=True)
        def pre_listen(tool: FunctionTool, tool_args: dict):  # type: ignore[override]
            return {}

        @hook(trigger=HookPhase.POST_TOOL_CALL, is_listener=False)
        def post_update(tool: FunctionTool, result: ToolResult):  # type: ignore[override]
            # Return a modified ToolResult-like object; simplest is to pass through
            return {"new_result": result}

        @hook(trigger=HookPhase.POST_TOOL_CALL, is_listener=True)
        def post_listen(tool: FunctionTool, result: ToolResult):  # type: ignore[override]
            return {}

        hook_registry: HookRegistry = get_hook_registry()
        # Expect 4 hook names
        assert set(hook_registry.list_hook_names()) == {
            "pre_update",
            "pre_listen",
            "post_update",
            "post_listen",
        }

        assert isinstance(hook_registry["pre_update"], FunctionHook)
        assert isinstance(hook_registry["pre_listen"], FunctionHook)
        assert isinstance(hook_registry["post_update"], FunctionHook)
        assert isinstance(hook_registry["post_listen"], FunctionHook)

        # Execute hooks to ensure run methods work and enforce parameters
        import asyncio

        async def run_hook(hook, *args, **kwargs):
            result = None
            async for item in hook.run(*args, **kwargs):
                if not item.is_update:
                    result = item
            return result

        # Pre updater returns dict
        pre_update_result = asyncio.run(
            run_hook(hook_registry["pre_update"], tool_instance, tool_args={})
        )  # type: ignore[arg-type]
        assert pre_update_result.new_tool_args == {"pre": True}

        # Pre listener returns None
        pre_listen_result = asyncio.run(
            run_hook(hook_registry["pre_listen"], tool_instance, tool_args={})
        )  # type: ignore[arg-type]
        assert pre_listen_result.new_tool_args is None

        # Simulate a tool result; build a minimal ToolResult
        tool_exec_result = asyncio.run(tool_instance.execute())
        post_update_result = asyncio.run(
            run_hook(hook_registry["post_update"], tool_instance, tool_exec_result)
        )  # type: ignore[arg-type]
        assert post_update_result.new_result is tool_exec_result
        post_listen_result = asyncio.run(
            run_hook(hook_registry["post_listen"], tool_instance, tool_exec_result)
        )  # type: ignore[arg-type]
        assert post_listen_result.new_result is None

    def test_hook_overwrite(self):
        """Re-registering a hook with the same name overwrites previous one."""

        @hook(trigger=HookPhase.PRE_TOOL_CALL)
        def duplicate(tool, tool_args):  # type: ignore[override]
            return {"new_tool_args": {"v": 1}}

        @hook(trigger=HookPhase.PRE_TOOL_CALL)
        def duplicate(tool, tool_args):  # type: ignore[override]  # noqa: F811
            return {"new_tool_args": {"v": 2}}

        registry = get_hook_registry()
        assert len([n for n in registry.list_hook_names() if n == "duplicate"]) == 1
        import asyncio

        async def run_hook(hook, *args, **kwargs):
            result = None
            async for item in hook.run(*args, **kwargs):
                if not item.is_update:
                    result = item
            return result

        res = asyncio.run(
            run_hook(registry["duplicate"], object(), tool_args={}),
        )  # type: ignore[arg-type]
        assert res.new_tool_args == {"v": 2}

    def test_on_event_hook_decorator_with_on_event_triggers(self):
        """on_event_triggers passed to @hook limits which event types fire the hook."""
        import asyncio

        selected = [ExecutionEventType.USER_INPUT, ExecutionEventType.RESPONSE]

        @hook(
            trigger=HookPhase.ON_AGENT_EVENT,
            on_event_triggers=selected,
        )
        def filtered_hook(event: BaseEvent, context: InteractionContext) -> None:
            pass

        registry = get_hook_registry()
        hook_obj = registry["filtered_hook"]
        assert isinstance(hook_obj, FunctionHook)
        assert hook_obj.on_event_triggers == selected

        # None means all event types (default)
        @hook(trigger=HookPhase.ON_AGENT_EVENT)
        def unfiltered_hook(event: BaseEvent, context: InteractionContext) -> None:
            pass

        unfiltered = registry["unfiltered_hook"]
        assert unfiltered.on_event_triggers is not None
        assert len(unfiltered.on_event_triggers) > len(selected)
