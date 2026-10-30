"""Tests for DelegateTool and DelegateLLMProvider runner handling."""

import asyncio
from collections.abc import AsyncGenerator
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from domyn.agents import Runner
from domyn.agents.agents import Agent
from domyn.agents.core import BaseEvent, ExecutionEventType, LLMCallAction, Part, ToolAction
from domyn.agents.llm import (
    BaseLLMProvider,
    DelegateLLMProvider,
    LLMMessage,
    LLMProviderFactory,
    LLMResponse,
    StructuredOutputType,
)
from domyn.agents.tools import DelegateTool, ToolResult

# ── Utilities ────────────────────────────────────────────────────────────────


def async_gen_from_list(values: list):
    """Return an async generator factory that yields items from *values*."""
    _iter = iter(values)
    finished = False

    async def _gen(*args, **kwargs) -> AsyncGenerator:
        nonlocal finished
        if finished:
            raise StopAsyncIteration
        for v in _iter:
            yield v
        finished = True

    return _gen


def make_llm_provider(response: LLMResponse) -> BaseLLMProvider:
    class _MockLLMProvider(BaseLLMProvider):
        type: str = "mock"
        name: str = "mock"
        model_name: str = "mock"
        supported_structured_outputs: None = None

        async def generate(self, messages: Any, **kwargs: Any) -> LLMResponse:
            return response

        async def generate_streaming(
            self, messages: Any, **kwargs: Any
        ) -> AsyncGenerator[LLMResponse, None]:
            yield response

    return _MockLLMProvider()


# ── DelegateTool unit tests ───────────────────────────────────────────────────


class TestDelegateToolFromConfig:
    def test_basic_fields(self):
        tool = DelegateTool.from_config(
            {
                "name": "web_search",
                "description": "Search the web",
                "parameters": {
                    "query": {"type": "string", "description": "Search query"},
                },
                "required_params": ["query"],
            }
        )
        assert tool.tool_name == "web_search"
        assert tool.name == "web_search"
        assert tool.description == "Search the web"
        assert tool.required_params == ["query"]

    def test_defaults(self):
        tool = DelegateTool.from_config({"name": "noop"})
        assert tool.tool_description == ""
        assert tool.tool_parameters == {}
        assert tool.required_params == []
        assert tool.wrapped is None


class TestDelegateToolDefinition:
    def test_openai_schema(self):
        tool = DelegateTool(
            tool_name="translate",
            tool_description="Translate text",
            tool_parameters={
                "text": {"type": "string"},
                "target_lang": {"type": "string"},
            },
            required_params=["text"],
        )
        defn = tool.tool_definition
        assert defn["type"] == "function"
        fn = defn["function"]
        assert fn["name"] == "translate"
        assert fn["description"] == "Translate text"
        params = fn["parameters"]
        assert params["type"] == "object"
        assert "text" in params["properties"]
        assert params["required"] == ["text"]

    def test_required_defaults_to_all_params(self):
        tool = DelegateTool(
            tool_name="noop",
            tool_parameters={"a": {"type": "string"}, "b": {"type": "integer"}},
        )
        required = tool.tool_definition["function"]["parameters"]["required"]
        assert set(required) == {"a", "b"}


class TestDelegateToolExecute:
    @pytest.mark.asyncio
    async def test_execute_without_wrapped_raises(self):
        tool = DelegateTool(tool_name="remote")
        with pytest.raises(RuntimeError, match="no active relay"):
            await tool.execute(param="value")

    @pytest.mark.asyncio
    async def test_execute_with_sync_wrapped(self):
        def my_func(query: str) -> str:
            return f"result:{query}"

        tool = DelegateTool(tool_name="search", wrapped=my_func)
        result = await tool.execute(query="hello")
        assert result.success is True
        assert result.data == "result:hello"

    @pytest.mark.asyncio
    async def test_execute_with_async_wrapped(self):
        async def my_async_func(query: str) -> dict:
            return {"answer": query}

        tool = DelegateTool(tool_name="ask", wrapped=my_async_func)
        result = await tool.execute(query="test")
        assert result.success is True
        assert result.data == {"answer": "test"}

    @pytest.mark.asyncio
    async def test_execute_filters_extra_kwargs(self):
        """Extra kwargs that the wrapped callable doesn't accept are ignored."""

        def narrow(x: int) -> int:
            return x * 2

        tool = DelegateTool(tool_name="double", wrapped=narrow)
        result = await tool.execute(x=5, run_config={}, state={})
        assert result.success is True
        assert result.data == 10

    @pytest.mark.asyncio
    async def test_execute_wraps_exception_in_tool_result(self):
        def boom(x: int) -> int:
            raise ValueError("boom")

        tool = DelegateTool(tool_name="crash", wrapped=boom)
        result = await tool.execute(x=1)
        assert result.success is False
        assert "boom" in result.error

    @pytest.mark.asyncio
    async def test_execute_returns_tool_result_as_is(self):
        async def returns_result(**kwargs: Any) -> ToolResult:
            return ToolResult(success=True, data="custom", metadata={"k": "v"})

        tool = DelegateTool(tool_name="direct", wrapped=returns_result)
        result = await tool.execute()
        assert result.data == "custom"
        assert result.metadata == {"k": "v"}


# ── DelegateLLMProvider unit tests ───────────────────────────────────────────


class TestDelegateLLMProviderFactory:
    def test_factory_creates_delegate_provider(self):
        provider = LLMProviderFactory.get_llm_provider({"type": "delegate", "name": "my-delegate"})
        assert isinstance(provider, DelegateLLMProvider)
        assert provider.wrapped is None

    def test_factory_with_no_extra_fields(self):
        provider = LLMProviderFactory.get_llm_provider({"type": "delegate"})
        assert isinstance(provider, DelegateLLMProvider)


class TestDelegateLLMProviderGenerate:
    @pytest.mark.asyncio
    async def test_generate_raises(self):
        provider = DelegateLLMProvider()
        with pytest.raises(RuntimeError, match="must not be called directly"):
            await provider.generate(messages=[])

    @pytest.mark.asyncio
    async def test_generate_streaming_raises(self):
        provider = DelegateLLMProvider()
        with pytest.raises(RuntimeError, match="must not be called directly"):
            async for _ in provider.generate_streaming(messages=[]):
                pass


# ── Runner integration: DelegateTool with no wrapped ─────────────────────────


@pytest.mark.asyncio
async def test_runner_delegate_tool_no_wrapped_yields_tool_error(runner: Runner):
    """TOOL_START on a DelegateTool(wrapped=None) with no relay yields TOOL_ERROR."""
    delegate_tool = DelegateTool(tool_name="remote_search", tool_description="Remote search")

    agent = Agent(
        name="TestAgent",
        description="test",
        instruction="test",
        llm_provider=make_llm_provider(
            LLMResponse(content="done", tool_calls=[], generated_tokens=0, execution_time=0.0)
        ),
        tools=[delegate_tool],
    )

    # Planner emits TOOL_START for the delegate tool then RESPONSE
    tool_start = BaseEvent(
        event_type=ExecutionEventType.TOOL_START,
        author="TestAgent",
        action=ToolAction(name="remote_search", parameters={"query": "hello"}),
    )
    response = BaseEvent(
        event_type=ExecutionEventType.RESPONSE,
        author="TestAgent",
        content=[Part(text="done")],
        is_partial=False,
    )

    with patch.object(
        agent, "_execute", async_gen_from_list([(tool_start, None), (response, None)])
    ):
        events = [
            e
            async for e in runner.run(
                application_name="app",
                user_id="u1",
                root=agent,
                user_input=BaseEvent(
                    event_type=ExecutionEventType.USER_INPUT,
                    author="user",
                    content=[Part(text="search for me")],
                ),
                conversation_id="conv1",
            )
        ]

    event_types = [e.event_type for e in events]
    assert ExecutionEventType.TOOL_START in event_types
    tool_error = next((e for e in events if e.event_type == ExecutionEventType.TOOL_ERROR), None)
    assert tool_error is not None
    assert tool_error.error_message is not None
    assert "no active relay" in tool_error.error_message


@pytest.mark.asyncio
async def test_runner_gates_a_delegate_tool_configured_for_approval(runner: Runner):
    """``from_config`` keeps ``approval_type``."""
    called: list[dict] = []
    tool = DelegateTool.from_config({"name": "delete_space", "approval_type": "user_confirm"})
    tool.wrapped = lambda **kwargs: called.append(kwargs) or "deleted"
    agent = Agent(
        name="TestAgent",
        description="test",
        instruction="test",
        llm_provider=make_llm_provider(
            LLMResponse(content="done", tool_calls=[], generated_tokens=0, execution_time=0.0)
        ),
        tools=[tool],
    )
    tool_start = BaseEvent(
        event_type=ExecutionEventType.TOOL_START,
        author="TestAgent",
        action=ToolAction(name="delete_space", parameters={"space": "prod"}),
    )

    with patch.object(agent, "_execute", async_gen_from_list([(tool_start, None)])):
        events = [
            e
            async for e in runner.run(
                application_name="app",
                user_id="u1",
                root=agent,
                user_input=BaseEvent(
                    event_type=ExecutionEventType.USER_INPUT,
                    author="user",
                    content=[Part(text="delete prod")],
                ),
                conversation_id="conv1",
            )
        ]

    assert any(e.event_type is ExecutionEventType.TOOL_NEEDS_FEEDBACK for e in events)
    assert called == []


# ── Runner integration: DelegateLLMProvider with wrapped ─────────────────────


@pytest.mark.asyncio
async def test_runner_delegate_llm_with_wrapped_sync(runner: Runner):
    """LLM_START with DelegateLLMProvider(wrapped=real) executes via wrapped and feeds response back."""
    real_response = LLMResponse(
        content="42",
        tool_calls=[],
        generated_tokens=5,
        execution_time=0.01,
    )
    real_provider = make_llm_provider(real_response)
    delegate_provider = DelegateLLMProvider(wrapped=real_provider)

    agent = Agent(
        name="DelegateAgent",
        description="test",
        instruction="test",
        llm_provider=delegate_provider,
        stream=False,
    )

    # Planner emits LLM_START (as in bidirectional protocol) then RESPONSE after receiving
    # the LLMResponse back via asend(). We simulate the planner as an async generator that
    # yields LLM_START, receives the LLMResponse, then yields RESPONSE.
    llm_start = BaseEvent(
        event_type=ExecutionEventType.LLM_START,
        author="DelegateAgent",
        action=LLMCallAction(messages=[], stream=False),
    )
    response_event = BaseEvent(
        event_type=ExecutionEventType.RESPONSE,
        author="DelegateAgent",
        content=[Part(text="42")],
        is_partial=False,
    )

    received_llm_response: list[LLMResponse] = []

    async def mock_execute(context):
        # Yield LLM_START — runner should intercept and send back LLMResponse via asend()
        llm_resp = yield llm_start, context
        received_llm_response.append(llm_resp)
        yield response_event, context

    with patch.object(agent, "_execute", mock_execute):
        events = [
            e
            async for e in runner.run(
                application_name="app",
                user_id="u1",
                root=agent,
                user_input=BaseEvent(
                    event_type=ExecutionEventType.USER_INPUT,
                    author="user",
                    content=[Part(text="what is the answer?")],
                ),
                conversation_id="conv2",
            )
        ]

    # Runner should have intercepted LLM_START and fed back the wrapped provider's response
    assert len(received_llm_response) == 1
    assert received_llm_response[0].content == "42"

    event_types = [e.event_type for e in events]
    # LLM_START is yielded for observability
    assert ExecutionEventType.LLM_START in event_types


@pytest.mark.asyncio
async def test_runner_delegate_llm_with_wrapped_streaming(runner: Runner):
    """LLM_START stream=True with DelegateLLMProvider(wrapped=real) accumulates chunks correctly."""
    chunk1 = LLMResponse(content="hel", tool_calls=[], generated_tokens=1, execution_time=0.0)
    chunk2 = LLMResponse(content="lo", tool_calls=[], generated_tokens=1, execution_time=0.0)

    class StreamingProvider(BaseLLMProvider):
        type: str = "streaming_mock"
        name: str = "streaming_mock"
        model_name: str = "mock"
        supported_structured_outputs: None = None

        async def generate(self, messages: Any, **kwargs: Any) -> LLMResponse:
            return LLMResponse(
                content="hello", tool_calls=[], generated_tokens=2, execution_time=0.0
            )

        async def generate_streaming(
            self, messages: Any, **kwargs: Any
        ) -> AsyncGenerator[LLMResponse, None]:
            yield chunk1
            yield chunk2

    real_provider = StreamingProvider()
    delegate_provider = DelegateLLMProvider(wrapped=real_provider)

    agent = Agent(
        name="StreamAgent",
        description="test",
        instruction="test",
        llm_provider=delegate_provider,
        stream=True,
    )

    llm_start = BaseEvent(
        event_type=ExecutionEventType.LLM_START,
        author="StreamAgent",
        action=LLMCallAction(messages=[], stream=True),
    )
    response_event = BaseEvent(
        event_type=ExecutionEventType.RESPONSE,
        author="StreamAgent",
        content=[],
        is_partial=False,
    )

    accumulated: list[LLMResponse] = []

    async def mock_execute(context):
        llm_resp = yield llm_start, context
        accumulated.append(llm_resp)
        yield response_event, context

    with patch.object(agent, "_execute", mock_execute):
        async for _ in runner.run(
            application_name="app",
            user_id="u1",
            root=agent,
            user_input=BaseEvent(
                event_type=ExecutionEventType.USER_INPUT,
                author="user",
                content=[Part(text="say hello")],
            ),
            conversation_id="conv3",
        ):
            pass

    assert len(accumulated) == 1
    assert accumulated[0].content == "hello"  # chunks joined: "hel" + "lo"
