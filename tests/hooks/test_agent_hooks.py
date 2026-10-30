"""Unified tests for agent hooks

Covers:
- Ordinary (sync/async) functions → single final event, no intermediate updates
- Async-generator functions → zero or more is_update=True chunks followed by a final event
- Return-type coverage: HookResult, dict (with/without is_update key), str, list, None,
  float, InteractionContext, …
- Updater-specific rules: final result must carry new_context
- Listener-specific rules: final result must NOT carry new_context
- FunctionAgentHook.drain() - synthetic final appended when no explicit final yielded
- REST-backed hooks (mocked via httpx.MockTransport)
"""

import asyncio
from collections.abc import AsyncGenerator, Callable
from datetime import datetime
from typing import Any

import httpx
import pytest

from domyn.agents import BaseEvent, ExecutionEventType, Part
from domyn.agents.core import BaseAgent, InteractionContext
from domyn.agents.hooks import FunctionHook as FunctionAgentHook
from domyn.agents.hooks import HookError, HookPhase, HookRestAPIData, HookResult, HookType

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CONFIG_PARAMS = {"param1": "value1", "param2": 2, "param3": True}


@pytest.fixture
def context() -> InteractionContext:
    return InteractionContext(
        conversation_id="123",
        user_id="user_1",
        interaction_id="456",
        created_at=datetime(2000, 1, 1, 0, 0, 0, 0),
        updated_at=datetime(2000, 1, 1, 0, 0, 0, 0),
        agent_input=BaseEvent(
            event_type=ExecutionEventType.USER_INPUT,
            author="user",
            timestamp=datetime(2000, 1, 1, 0, 0, 0, 0),
            content=[Part(text="test input")],
            event_id="789",
        ),
        interaction_history=[
            BaseEvent(
                event_type=ExecutionEventType.LLM_START,
                author="agent",
                timestamp=datetime(2000, 1, 1, 0, 0, 0, 0),
                content=[Part(text="test previous input")],
                event_id="789",
            )
        ],
    )


@pytest.fixture
def agent() -> BaseAgent:
    return BaseAgent(name="test_agent", description="mock agent for testing")


# ---------------------------------------------------------------------------
# Helper: build an updated InteractionContext
# ---------------------------------------------------------------------------


def new_context() -> InteractionContext:
    return InteractionContext(
        conversation_id="4747",
        user_id="user_23",
        interaction_id="576",
        created_at=datetime(2010, 1, 1, 0, 0, 0, 0),
        updated_at=datetime(2010, 1, 1, 0, 0, 0, 0),
        agent_input=BaseEvent(
            event_type=ExecutionEventType.USER_INPUT,
            author="user2341",
            timestamp=datetime(2002, 1, 1, 0, 0, 0, 0),
            content=[Part(text="test test test test"), Part(text=" - updated by hook")],
            event_id="5857",
        ),
        interaction_history=[
            BaseEvent(
                event_type=ExecutionEventType.LLM_END,
                author="agent_test",
                timestamp=datetime(2025, 1, 1, 0, 0, 0, 0),
                content=[Part(text="test previous input...")],
                event_id="887",
            )
        ],
    )


def new_context_dict() -> dict[str, Any]:
    return new_context().model_dump()


# ---------------------------------------------------------------------------
# Helper: drain async-generator into a list
# ---------------------------------------------------------------------------


async def collect(gen: AsyncGenerator) -> list[HookResult]:
    return [item async for item in gen]


# ===========================================================================
# ORDINARY-FUNCTION hook definitions
# (sync; wrap_as_async_generator wraps them into a 1-yield generator)
# ===========================================================================


def hook_no_result(context: InteractionContext, agent: BaseAgent, **config_params) -> None:
    pass


def hook_string_event_content(
    context: InteractionContext, agent: BaseAgent, **config_params
) -> str:
    return "This is a string event"


def hook_dict_event_content(context: InteractionContext, agent: BaseAgent, **config_params) -> dict:
    return {"event": "This is a dict event"}


def hook_list_event_content(context: InteractionContext, agent: BaseAgent, **config_params) -> list:
    return ["this", "is", "a", "list", "event"]


def hook_float_event_content(
    context: InteractionContext, agent: BaseAgent, **config_params
) -> float:
    return 2.0


def hook_returning_context(
    context: InteractionContext, agent: BaseAgent, **config_params
) -> InteractionContext:
    return new_context()


def hook_returning_context_in_dict(
    context: InteractionContext, agent: BaseAgent, **config_params
) -> dict[str, InteractionContext]:
    return {"new_context": new_context()}


def hook_with_new_context_and_event_content_in_dict(
    context: InteractionContext, agent: BaseAgent, **config_params
) -> dict[str, Any]:
    return {"new_context": new_context(), "emitted_content": {"info": "context updated"}}


def hook_with_new_context_and_event_content_in_dict_empty_event(
    context: InteractionContext, agent: BaseAgent, **config_params
) -> dict[str, Any]:
    return {"new_context": new_context(), "emitted_content": {}}


def hook_with_new_context_and_event_content_in_dict_list_event(
    context: InteractionContext, agent: BaseAgent, **config_params
) -> dict[str, Any]:
    return {"new_context": new_context(), "emitted_content": []}


def hook_with_new_context_and_str_event_content(
    context: InteractionContext, agent: BaseAgent, **config_params
) -> dict[str, Any]:
    return {"new_context": new_context(), "emitted_content": "metadata123"}


def hook_with_new_context_and_int_event_content(
    context: InteractionContext, agent: BaseAgent, **config_params
) -> dict[str, Any]:
    return {"new_context": new_context(), "emitted_content": 123}


# --- HookResult-returning ordinary functions ---


def hook_returning_result_with_context_and_str_metadata(
    context: InteractionContext, agent: BaseAgent, **config_params
) -> HookResult:
    return HookResult(
        emitted_content="event metadata",
        new_context=new_context(),
        success=True,
        error=None,
        name=hook_returning_result_with_context_and_str_metadata.__name__,
    )


def hook_returning_result_with_context_and_dict_metadata(
    context: InteractionContext, agent: BaseAgent, **config_params
) -> HookResult:
    return HookResult(
        emitted_content={"info": "context updated"},
        new_context=new_context(),
        success=True,
        error=None,
        name=hook_returning_result_with_context_and_dict_metadata.__name__,
    )


def hook_returning_result_with_context_invalid_content(
    context: InteractionContext, agent: BaseAgent, **config_params
) -> HookResult:
    return HookResult(
        emitted_content=1234,  # type: ignore[arg-type]  # invalid - raises ValidationError
        new_context=new_context(),
        success=True,
        error=None,
        name=hook_returning_result_with_context_invalid_content.__name__,
    )


def hook_returning_result_with_context_no_content(
    context: InteractionContext, agent: BaseAgent, **config_params
) -> HookResult:
    return HookResult(
        emitted_content=None,
        new_context=new_context(),
        success=True,
        error=None,
        name=hook_returning_result_with_context_no_content.__name__,
    )


def hook_returning_result_no_context(
    context: InteractionContext, agent: BaseAgent, **config_params
) -> HookResult:
    return HookResult(
        new_context=None,
        emitted_content="Annotated return",
        success=True,
        error=None,
        name=hook_returning_result_no_context.__name__,
    )


# --- dict returning with Parts list as event_content ---


def hook_returning_context_and_parts_list(
    context: InteractionContext, agent: BaseAgent, **config_params
) -> dict[str, Any]:
    return {"new_context": new_context(), "emitted_content": [Part(text="event metadata")]}


# --- dict returning with a dict new_context (valid vs invalid) ---


def hook_returning_valid_context_dict(
    context: InteractionContext, agent: BaseAgent, **config_params
) -> dict[str, Any]:
    return {"new_context": new_context_dict(), "emitted_content": "event metadata"}


def hook_returning_invalid_context_dict(
    context: InteractionContext, agent: BaseAgent, **config_params
) -> dict[str, Any]:
    return {"new_context": {"key1": "value1"}, "emitted_content": "event metadata"}


def hook_returning_invalid_context_int(
    context: InteractionContext, agent: BaseAgent, **config_params
) -> dict[str, Any]:
    return {"new_context": 123, "emitted_content": "event metadata"}


# --- config_params passthrough ---


def hook_returning_context_and_config_params(
    context: InteractionContext, agent: BaseAgent, **config_params
) -> dict[str, Any]:
    return {"new_context": new_context(), "emitted_content": config_params}


def hook_returning_only_config_params(
    context: InteractionContext, agent: BaseAgent, **config_params
) -> dict[str, Any]:
    return {"emitted_content": config_params}


# ===========================================================================
# ASYNC-GENERATOR hook definitions
# (can yield multiple is_update=True chunks before the final item)
# ===========================================================================


async def async_gen_updater_str_chunks_then_final(
    context: InteractionContext, agent: BaseAgent, **config_params: Any
) -> AsyncGenerator:
    """Two is_update str chunks then a final HookResult with new_context."""
    yield HookResult(
        new_context=None, emitted_content="chunk 1", success=True, error=None, is_update=True
    )
    yield HookResult(
        new_context=None, emitted_content="chunk 2", success=True, error=None, is_update=True
    )
    yield HookResult(
        new_context=new_context(), emitted_content="final update", success=True, error=None
    )


async def async_gen_updater_dict_chunk_then_final(
    context: InteractionContext, agent: BaseAgent, **config_params: Any
) -> AsyncGenerator:
    """One dict chunk (is_update key) then a final HookResult."""
    yield {"is_update": True, "emitted_content": {"key": "value"}}
    yield HookResult(new_context=new_context(), emitted_content=None, success=True, error=None)


async def async_gen_updater_result_chunk_then_dict_final(
    context: InteractionContext, agent: BaseAgent, **config_params: Any
) -> AsyncGenerator:
    """is_update HookResult chunk then a dict final (new_context key)."""
    yield HookResult(
        new_context=None,
        emitted_content=[Part(text="stream chunk")],
        success=True,
        error=None,
        is_update=True,
    )
    yield {"new_context": new_context(), "emitted_content": {"info": "done"}}


async def async_gen_updater_missing_new_context(
    context: InteractionContext, agent: BaseAgent, **config_params: Any
) -> AsyncGenerator:
    """Updater that forgets new_context in its final yield → error result."""
    yield HookResult(
        new_context=None, emitted_content="intermediate", success=True, error=None, is_update=True
    )
    yield HookResult(new_context=None, emitted_content="bad final", success=True, error=None)


async def async_gen_raises_exception(
    context: InteractionContext, agent: BaseAgent, **config_params: Any
) -> AsyncGenerator:
    """Raises an exception mid-stream."""
    yield HookResult(
        new_context=None, emitted_content="before error", success=True, error=None, is_update=True
    )
    raise RuntimeError("unexpected error in hook")
    yield


async def async_gen_listener_str_chunks_then_final(
    context: InteractionContext, agent: BaseAgent, **config_params: Any
) -> AsyncGenerator:
    """Listener: two is_update chunks then a final HookResult (no new_context)."""
    yield HookResult(
        new_context=None, emitted_content="chunk A", success=True, error=None, is_update=True
    )
    yield HookResult(
        new_context=None, emitted_content="chunk B", success=True, error=None, is_update=True
    )
    yield HookResult(new_context=None, emitted_content="listener done", success=True, error=None)


async def async_gen_listener_only_update_chunks_no_final(
    context: InteractionContext, agent: BaseAgent, **config_params: Any
) -> AsyncGenerator:
    """Listener: only streaming chunks, no explicit final → drain() adds synthetic final."""
    yield HookResult(
        new_context=None, emitted_content="only chunk", success=True, error=None, is_update=True
    )


async def async_gen_listener_tries_to_set_context(
    context: InteractionContext, agent: BaseAgent, **config_params: Any
) -> AsyncGenerator:
    """Listener that incorrectly sets new_context in the final result → error."""
    yield HookResult(
        new_context=None, emitted_content="intermediate", success=True, error=None, is_update=True
    )
    yield HookResult(
        new_context=new_context(), emitted_content="bad final", success=True, error=None
    )


# ===========================================================================
# LISTENER HOOK - ordinary-function parametrized test data
# (one yield per call; wraps via wrap_as_async_generator → exactly 1 result)
# ===========================================================================

listener_ordinary_function_testdata: list[tuple[Callable, dict[str, Any]]] = [
    # None return → empty event_content, success
    (
        hook_no_result,
        {"new_context": None, "emitted_content": [], "success": True, "error": None},
    ),
    # str return → Part(text=…)
    (
        hook_string_event_content,
        {
            "new_context": None,
            "emitted_content": [Part(text="This is a string event")],
            "success": True,
            "error": None,
        },
    ),
    # dict WITHOUT new_context key → Part(metadata=…)
    (
        hook_dict_event_content,
        {
            "new_context": None,
            "emitted_content": [Part(metadata={"event": "This is a dict event"})],
            "success": True,
            "error": None,
        },
    ),
    # list → stringified
    (
        hook_list_event_content,
        {
            "new_context": None,
            "emitted_content": [Part(text="['this', 'is', 'a', 'list', 'event']")],
            "success": True,
            "error": None,
        },
    ),
    # float → stringified
    (
        hook_float_event_content,
        {
            "new_context": None,
            "emitted_content": [Part(text="2.0")],
            "success": True,
            "error": None,
        },
    ),
    # InteractionContext → listener must NOT receive new_context → ValueError
    (
        hook_returning_context,
        {
            "new_context": None,
            "emitted_content": [],
            "success": False,
            "error": HookError(error_code="ValueError", error_message=""),
        },
    ),
    # dict with new_context → listener must NOT receive new_context → ValueError
    (
        hook_returning_context_in_dict,
        {
            "new_context": None,
            "emitted_content": [],
            "success": False,
            "error": HookError(error_code="ValueError", error_message=""),
        },
    ),
    # dict with new_context + event_content dict → ValueError
    (
        hook_with_new_context_and_event_content_in_dict,
        {
            "new_context": None,
            "emitted_content": [],
            "success": False,
            "error": HookError(error_code="ValueError", error_message=""),
        },
    ),
    # HookResult with new_context + str metadata → ValueError
    (
        hook_returning_result_with_context_and_str_metadata,
        {
            "new_context": None,
            "emitted_content": [],
            "success": False,
            "error": HookError(error_code="ValueError", error_message=""),
        },
    ),
    # HookResult with new_context + dict metadata → ValueError
    (
        hook_returning_result_with_context_and_dict_metadata,
        {
            "new_context": None,
            "emitted_content": [],
            "success": False,
            "error": HookError(error_code="ValueError", error_message=""),
        },
    ),
    # HookResult with new_context + int content → int coerced to Part, listener rejects new_context
    (
        hook_returning_result_with_context_invalid_content,
        {
            "new_context": None,
            "emitted_content": [],
            "success": False,
            "error": HookError(error_code="ValueError", error_message=""),
        },
    ),
    # HookResult with new_context only → ValueError
    (
        hook_returning_result_with_context_no_content,
        {
            "new_context": None,
            "emitted_content": [],
            "success": False,
            "error": HookError(error_code="ValueError", error_message=""),
        },
    ),
    # HookResult with no new_context → success, event_content forwarded
    (
        hook_returning_result_no_context,
        {
            "new_context": None,
            "emitted_content": [Part(text="Annotated return")],
            "success": True,
            "error": None,
        },
    ),
    # dict with new_context + Parts list → ValueError
    (
        hook_returning_context_and_parts_list,
        {
            "new_context": None,
            "emitted_content": [],
            "success": False,
            "error": HookError(error_code="ValueError", error_message=""),
        },
    ),
    # valid context dict → ValueError (listener cannot set context)
    (
        hook_returning_valid_context_dict,
        {
            "new_context": None,
            "emitted_content": [],
            "success": False,
            "error": HookError(error_code="ValueError", error_message=""),
        },
    ),
    # invalid context dict (bad keys) → ValidationError
    (
        hook_returning_invalid_context_dict,
        {
            "new_context": None,
            "emitted_content": [],
            "success": False,
            "error": HookError(error_code="ValidationError", error_message=""),
        },
    ),
    # invalid context (int) → ValidationError
    (
        hook_returning_invalid_context_int,
        {
            "new_context": None,
            "emitted_content": [],
            "success": False,
            "error": HookError(error_code="ValidationError", error_message=""),
        },
    ),
    # only config_params in dict event_content → Part(metadata=config_params)
    (
        hook_returning_only_config_params,
        {
            "new_context": None,
            "emitted_content": [Part(metadata=CONFIG_PARAMS)],
            "success": True,
            "error": None,
        },
    ),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("hook_function, expected", listener_ordinary_function_testdata)
async def test_listener_ordinary_function_hook_yields_single_final_event(
    hook_function: Callable,
    expected: dict[str, Any],
    context: InteractionContext,
    agent: BaseAgent,
) -> None:
    """Ordinary functions (sync) produce exactly one final event (no is_update chunks)."""
    hook = FunctionAgentHook(
        fun=hook_function,
        config_params=CONFIG_PARAMS,
        trigger=HookPhase.PRE_AGENT_CALL,
        type=HookType.LISTENER,
        break_on_error=False,
    )
    assert hook.name == hook_function.__name__
    assert hook.type == "listener"

    results = await collect(hook.run(context=context, agent=agent))

    # Ordinary function → exactly 1 result (the final event, no update chunks)
    assert len(results) == 1
    executed = results[0]

    assert isinstance(executed, HookResult)
    assert executed.is_update is False
    assert executed.new_context == expected["new_context"]
    assert executed.success == expected["success"]
    if expected["error"] is None:
        assert executed.error is None
    else:
        assert isinstance(executed.error, HookError)
        assert executed.error.error_code == expected["error"].error_code
    assert executed.name == hook.name
    assert executed.emitted_content == expected["emitted_content"]


# ===========================================================================
# UPDATER HOOK - ordinary-function parametrized test data
# ===========================================================================

updater_ordinary_function_testdata: list[tuple[Callable, dict[str, Any]]] = [
    # None return → no new_context → ValueError
    (
        hook_no_result,
        {
            "new_context": None,
            "emitted_content": [],
            "success": False,
            "error": HookError(error_code="ValueError", error_message=""),
        },
    ),
    # str return → no new_context → ValueError
    (
        hook_string_event_content,
        {
            "new_context": None,
            "emitted_content": [],
            "success": False,
            "error": HookError(
                error_code="ValueError",
                error_message="The updater agent hook function must return the new context.",
            ),
        },
    ),
    # dict WITHOUT new_context key → ValueError
    (
        hook_dict_event_content,
        {
            "new_context": None,
            "emitted_content": [],
            "success": False,
            "error": HookError(error_code="ValueError", error_message=""),
        },
    ),
    # list return → no new_context → ValueError
    (
        hook_list_event_content,
        {
            "new_context": None,
            "emitted_content": [],
            "success": False,
            "error": HookError(
                error_code="ValueError",
                error_message="The updater agent hook function must return the new context.",
            ),
        },
    ),
    # float return → no new_context → ValueError
    (
        hook_float_event_content,
        {
            "new_context": None,
            "emitted_content": [],
            "success": False,
            "error": HookError(
                error_code="ValueError",
                error_message="The updater agent hook function must return the new context.",
            ),
        },
    ),
    # InteractionContext returned directly → new_context set, no event_content
    (
        hook_returning_context,
        {"new_context": new_context(), "emitted_content": [], "success": True, "error": None},
    ),
    # dict {"new_context": ctx} → new_context set, empty event_content
    (
        hook_returning_context_in_dict,
        {"new_context": new_context(), "emitted_content": [], "success": True, "error": None},
    ),
    # dict with new_context + dict event_content → Part(metadata=…)
    (
        hook_with_new_context_and_event_content_in_dict,
        {
            "new_context": new_context(),
            "emitted_content": [Part(metadata={"info": "context updated"})],
            "success": True,
            "error": None,
        },
    ),
    # dict with new_context + empty dict event_content → []
    (
        hook_with_new_context_and_event_content_in_dict_empty_event,
        {"new_context": new_context(), "emitted_content": [], "success": True, "error": None},
    ),
    # dict with new_context + empty list event_content → empty
    (
        hook_with_new_context_and_event_content_in_dict_list_event,
        {
            "new_context": new_context(),
            "emitted_content": [],
            "success": True,
            "error": None,
        },
    ),
    # dict with new_context + str event_content → Part(text=…)
    (
        hook_with_new_context_and_str_event_content,
        {
            "new_context": new_context(),
            "emitted_content": [Part(text="metadata123")],
            "success": True,
            "error": None,
        },
    ),
    # dict with new_context + int event_content → Part(text="123")
    (
        hook_with_new_context_and_int_event_content,
        {
            "new_context": new_context(),
            "emitted_content": [Part(text="123")],
            "success": True,
            "error": None,
        },
    ),
    # HookResult with new_context + str metadata → success
    (
        hook_returning_result_with_context_and_str_metadata,
        {
            "new_context": new_context(),
            "emitted_content": [Part(text="event metadata")],
            "success": True,
            "error": None,
        },
    ),
    # HookResult with new_context + dict metadata → Part(metadata=…)
    (
        hook_returning_result_with_context_and_dict_metadata,
        {
            "new_context": new_context(),
            "emitted_content": [Part(metadata={"info": "context updated"})],
            "success": True,
            "error": None,
        },
    ),
    # HookResult with new_context + int content → int coerced to Part, updater succeeds
    (
        hook_returning_result_with_context_invalid_content,
        {
            "new_context": new_context(),
            "emitted_content": [Part(text="1234")],
            "success": True,
            "error": None,
        },
    ),
    # HookResult with new_context + no content → success, empty event_content
    (
        hook_returning_result_with_context_no_content,
        {"new_context": new_context(), "emitted_content": [], "success": True, "error": None},
    ),
    # HookResult without new_context → ValueError
    (
        hook_returning_result_no_context,
        {
            "new_context": None,
            "emitted_content": [],
            "success": False,
            "error": HookError(error_code="ValueError", error_message=""),
        },
    ),
    # dict with new_context + Parts list → list of Parts passed through
    (
        hook_returning_context_and_parts_list,
        {
            "new_context": new_context(),
            "emitted_content": [Part(text="event metadata")],
            "success": True,
            "error": None,
        },
    ),
    # dict with valid (full) new_context dict + str event_content → success
    (
        hook_returning_valid_context_dict,
        {
            "new_context": new_context(),
            "emitted_content": [Part(text="event metadata")],
            "success": True,
            "error": None,
        },
    ),
    # dict with invalid context dict (bad keys) → ValidationError
    (
        hook_returning_invalid_context_dict,
        {
            "new_context": None,
            "emitted_content": [],
            "success": False,
            "error": HookError(error_code="ValidationError", error_message=""),
        },
    ),
    # dict with int context → ValidationError
    (
        hook_returning_invalid_context_int,
        {
            "new_context": None,
            "emitted_content": [],
            "success": False,
            "error": HookError(error_code="ValidationError", error_message=""),
        },
    ),
    # dict with new_context + config_params dict → Part(metadata=config_params)
    (
        hook_returning_context_and_config_params,
        {
            "new_context": new_context(),
            "emitted_content": [Part(metadata=CONFIG_PARAMS)],
            "success": True,
            "error": None,
        },
    ),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("hook_function, expected", updater_ordinary_function_testdata)
async def test_updater_ordinary_function_hook_yields_single_final_event(
    hook_function: Callable,
    expected: dict[str, Any],
    context: InteractionContext,
    agent: BaseAgent,
) -> None:
    """Ordinary functions (sync) produce exactly one final event (no is_update chunks)."""
    hook = FunctionAgentHook(
        fun=hook_function,
        config_params=CONFIG_PARAMS,
        trigger=HookPhase.PRE_AGENT_CALL,
        type=HookType.UPDATER,
        break_on_error=False,
    )
    assert hook.name == hook_function.__name__
    assert hook.type == "updater"

    results = await collect(hook.run(context=context, agent=agent))

    # Ordinary function → exactly 1 result (the final event, no update chunks)
    assert len(results) == 1
    executed = results[0]

    assert isinstance(executed, HookResult)
    assert executed.is_update is False
    assert executed.new_context == expected["new_context"]
    assert executed.success == expected["success"]
    if expected["error"] is None:
        assert executed.error is None
    else:
        assert isinstance(executed.error, HookError)
        assert executed.error.error_code == expected["error"].error_code
    assert executed.name == hook.name
    assert executed.emitted_content == expected["emitted_content"]


# ===========================================================================
# UPDATER HOOK - async-generator tests
# (multiple is_update=True chunks before the final item)
# ===========================================================================


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "hook_function, expected_chunks, expected_final",
    [
        (
            async_gen_updater_str_chunks_then_final,
            [
                HookResult(
                    new_context=None,
                    emitted_content=[Part(text="chunk 1")],
                    success=True,
                    error=None,
                    is_update=True,
                ),
                HookResult(
                    new_context=None,
                    emitted_content=[Part(text="chunk 2")],
                    success=True,
                    error=None,
                    is_update=True,
                ),
            ],
            {
                "success": True,
                "error": None,
                "new_context": new_context(),
                "emitted_content": [Part(text="final update")],
                "is_update": False,
            },
        ),
        (
            async_gen_updater_dict_chunk_then_final,
            [
                HookResult(
                    new_context=None,
                    emitted_content=[Part(metadata={"key": "value"})],
                    success=True,
                    error=None,
                    is_update=True,
                ),
            ],
            {
                "success": True,
                "error": None,
                "new_context": new_context(),
                "emitted_content": [],
                "is_update": False,
            },
        ),
        (
            async_gen_updater_result_chunk_then_dict_final,
            [
                HookResult(
                    new_context=None,
                    emitted_content=[Part(text="stream chunk")],
                    success=True,
                    error=None,
                    is_update=True,
                ),
            ],
            {
                "success": True,
                "error": None,
                "new_context": new_context(),
                "emitted_content": [Part(metadata={"info": "done"})],
                "is_update": False,
            },
        ),
    ],
    ids=[
        "agentresult_str_chunks_then_final",
        "dict_is_update_chunk_then_agentresult_final",
        "agentresult_chunk_then_dict_final",
    ],
)
async def test_updater_async_gen_hook_yields_chunks_then_final(
    hook_function: Any,
    expected_chunks: list[HookResult],
    expected_final: dict,
    context: InteractionContext,
    agent: BaseAgent,
) -> None:
    """Async-generator hooks emit is_update chunks followed by a single final result."""
    hook = FunctionAgentHook(
        fun=hook_function, trigger=HookPhase.PRE_AGENT_CALL, type=HookType.UPDATER
    )
    results = await collect(hook.run(context=context, agent=agent))

    assert len(results) == len(expected_chunks) + 1

    for i, expected_chunk in enumerate(expected_chunks):
        assert results[i].is_update is True
        assert results[i].emitted_content == expected_chunk.emitted_content
        assert results[i].name == hook.name

    final = results[-1]
    assert isinstance(final, HookResult)
    assert final.is_update is expected_final["is_update"]
    assert final.success is expected_final["success"]
    assert final.error == expected_final["error"]
    assert final.new_context == expected_final["new_context"]
    assert final.emitted_content == expected_final["emitted_content"]
    assert final.name == hook.name


@pytest.mark.asyncio
async def test_updater_async_gen_hook_missing_new_context_yields_error(
    context: InteractionContext, agent: BaseAgent
) -> None:
    """A final item without new_context produces a failed HookResult (ValueError)."""
    hook = FunctionAgentHook(
        fun=async_gen_updater_missing_new_context,
        trigger=HookPhase.PRE_AGENT_CALL,
        type=HookType.UPDATER,
    )
    results = await collect(hook.run(context=context, agent=agent))

    final = results[-1]
    assert isinstance(final, HookResult)
    assert final.success is False
    assert isinstance(final.error, HookError)
    assert final.error.error_code == "ValueError"


@pytest.mark.asyncio
async def test_updater_async_gen_hook_exception_yields_error(
    context: InteractionContext, agent: BaseAgent
) -> None:
    """An exception raised inside the generator is caught and a failed result is yielded."""
    hook = FunctionAgentHook(
        fun=async_gen_raises_exception, trigger=HookPhase.PRE_AGENT_CALL, type=HookType.UPDATER
    )
    results = await collect(hook.run(context=context, agent=agent))

    final = results[-1]
    assert isinstance(final, HookResult)
    assert final.success is False
    assert isinstance(final.error, HookError)
    assert final.error.error_code == "RuntimeError"


# ===========================================================================
# LISTENER HOOK - async-generator tests
# ===========================================================================


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "hook_function, expected_chunks, expected_final",
    [
        (
            async_gen_listener_str_chunks_then_final,
            [
                HookResult(
                    new_context=None,
                    emitted_content=[Part(text="chunk A")],
                    success=True,
                    error=None,
                    is_update=True,
                ),
                HookResult(
                    new_context=None,
                    emitted_content=[Part(text="chunk B")],
                    success=True,
                    error=None,
                    is_update=True,
                ),
            ],
            {
                "success": True,
                "error": None,
                "new_context": None,
                "emitted_content": [Part(text="listener done")],
                "is_update": False,
            },
        ),
    ],
    ids=["agentresult_str_chunks_then_final"],
)
async def test_listener_async_gen_hook_yields_chunks_then_final(
    hook_function: Any,
    expected_chunks: list[HookResult],
    expected_final: dict,
    context: InteractionContext,
    agent: BaseAgent,
) -> None:
    """Listener async-generator emits is_update chunks followed by a single final result."""
    hook = FunctionAgentHook(
        fun=hook_function, trigger=HookPhase.PRE_AGENT_CALL, type=HookType.LISTENER
    )
    results = await collect(hook.run(context=context, agent=agent))

    assert len(results) == len(expected_chunks) + 1

    for i, expected_chunk in enumerate(expected_chunks):
        assert results[i].is_update is True
        assert results[i].emitted_content == expected_chunk.emitted_content
        assert results[i].name == hook.name

    final = results[-1]
    assert isinstance(final, HookResult)
    assert final.is_update is expected_final["is_update"]
    assert final.success is expected_final["success"]
    assert final.error == expected_final["error"]
    assert final.new_context == expected_final["new_context"]
    assert final.emitted_content == expected_final["emitted_content"]
    assert final.name == hook.name


@pytest.mark.asyncio
async def test_listener_async_gen_hook_sets_context_yields_error(
    context: InteractionContext, agent: BaseAgent
) -> None:
    """Listener that sets new_context in final item produces a failed result (ValueError)."""
    hook = FunctionAgentHook(
        fun=async_gen_listener_tries_to_set_context,
        trigger=HookPhase.PRE_AGENT_CALL,
        type=HookType.LISTENER,
    )
    results = await collect(hook.run(context=context, agent=agent))

    final = results[-1]
    assert isinstance(final, HookResult)
    assert final.success is False
    assert isinstance(final.error, HookError)
    assert final.error.error_code == "ValueError"


@pytest.mark.asyncio
async def test_listener_async_gen_hook_exception_yields_error(
    context: InteractionContext, agent: BaseAgent
) -> None:
    """An exception raised inside a listener generator is caught and a failed result is yielded."""
    hook = FunctionAgentHook(
        fun=async_gen_raises_exception, trigger=HookPhase.PRE_AGENT_CALL, type=HookType.LISTENER
    )
    results = await collect(hook.run(context=context, agent=agent))

    final = results[-1]
    assert isinstance(final, HookResult)
    assert final.success is False
    assert isinstance(final.error, HookError)
    assert final.error.error_code == "RuntimeError"


# ===========================================================================
# queue-based parallel execution
# ===========================================================================


@pytest.mark.asyncio
async def test_listener_drain_no_explicit_final_gets_synthetic_success(
    context: InteractionContext, agent: BaseAgent
) -> None:
    """When a listener hook never yields a final (non-update) result, drain() appends a synthetic one."""
    hook = FunctionAgentHook(
        fun=async_gen_listener_only_update_chunks_no_final,
        trigger=HookPhase.PRE_AGENT_CALL,
        type=HookType.LISTENER,
    )
    queue: asyncio.Queue = asyncio.Queue()
    await hook.drain_to_queue(context=context, agent=agent, queue=queue)

    items = []
    while not queue.empty():
        item = await queue.get()
        items.append(item)

    assert len(items) == 2  # one is_update chunk + one synthetic final

    chunk = items[0]
    assert isinstance(chunk, HookResult)
    assert chunk.is_update is True
    assert chunk.emitted_content == [Part(text="only chunk")]

    synthetic = items[1]
    assert isinstance(synthetic, HookResult)
    assert synthetic.is_update is False
    assert synthetic.success is True
    assert synthetic.error is None
    assert synthetic.new_context is None


@pytest.mark.asyncio
async def test_listener_drain_ordinary_function_includes_final_only(
    context: InteractionContext, agent: BaseAgent
) -> None:
    """Ordinary function hooks produce exactly one final event via drain(), no updates."""
    hook = FunctionAgentHook(
        fun=hook_string_event_content,
        trigger=HookPhase.PRE_AGENT_CALL,
        type=HookType.LISTENER,
    )
    queue: asyncio.Queue = asyncio.Queue()
    await hook.drain_to_queue(context=context, agent=agent, queue=queue)

    items = []
    while not queue.empty():
        item = await queue.get()
        items.append(item)

    assert len(items) == 1
    final = items[0]
    assert isinstance(final, HookResult)
    assert final.is_update is False
    assert final.success is True
    assert final.emitted_content == [Part(text="This is a string event")]


@pytest.mark.asyncio
async def test_listener_drain_none_returning_function(
    context: InteractionContext, agent: BaseAgent
) -> None:
    """A listener hook that returns None produces exactly one final result with empty event_content."""
    hook = FunctionAgentHook(
        fun=hook_no_result,
        trigger=HookPhase.PRE_AGENT_CALL,
        type=HookType.LISTENER,
    )
    queue: asyncio.Queue = asyncio.Queue()
    await hook.drain_to_queue(context=context, agent=agent, queue=queue)

    items = []
    while not queue.empty():
        item = await queue.get()
        items.append(item)

    assert len(items) == 1
    final = items[0]
    assert isinstance(final, HookResult)
    assert final.is_update is False
    assert final.success is True
    assert final.error is None
    assert final.emitted_content == []
    assert final.new_context is None


# ===========================================================================
# Construction validation
# ===========================================================================


# ===========================================================================
# REST-backed hooks (mocked via httpx.MockTransport)
# ===========================================================================


def _mock_json_response(status_code: int, json_response: dict) -> Callable:
    def mock_send(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code, json=json_response, request=request)

    return mock_send


listener_rest_hook_testdata: list[tuple[dict, dict[str, Any]]] = [
    # empty list event_content → empty
    (
        {"emitted_content": []},
        {"emitted_content": [], "success": True, "error": None},
    ),
    # dict event_content → Part(metadata=…)
    (
        {"emitted_content": {"info": "test"}},
        {"emitted_content": [Part(metadata={"info": "test"})], "success": True, "error": None},
    ),
    # list of strings → stringified
    (
        {"emitted_content": ["this", "is", "a", "list"]},
        {
            "emitted_content": [Part(text="['this', 'is', 'a', 'list']")],
            "success": True,
            "error": None,
        },
    ),
    # HTTP 500 → error result
    (
        {"error": "Something went wrong", "error_code": 500},
        {
            "emitted_content": [],
            "success": False,
            "error": HookError(error_code=500, error_message='{"error":"Something went wrong"}'),
        },
    ),
    # HTTP string error_code
    (
        {"error": "Something went wrong", "error_code": "Error"},
        {
            "emitted_content": [],
            "success": False,
            "error": HookError(
                error_code="Error", error_message='{"error":"Something went wrong"}'
            ),
        },
    ),
    # int event_content → stringified
    (
        {"emitted_content": 12345},
        {"emitted_content": [Part(text="12345")], "success": True, "error": None},
    ),
    # arbitrary dict (no event_content / new_context keys) → Part(metadata=…)
    (
        {"key1": "value1", "key2": "value2"},
        {
            "emitted_content": [Part(metadata={"key1": "value1", "key2": "value2"})],
            "success": True,
            "error": None,
        },
    ),
    # explicit null new_context with event_content → listener ok
    (
        {"emitted_content": {"key1": "value1", "key2": "value2"}, "new_context": None},
        {
            "emitted_content": [Part(metadata={"key1": "value1", "key2": "value2"})],
            "success": True,
            "error": None,
        },
    ),
    # new_context present → listener must NOT receive new_context → ValueError
    (
        {"emitted_content": [], "new_context": new_context().model_dump(mode="json")},
        {
            "emitted_content": [],
            "success": False,
            "error": HookError(error_code="ValueError", error_message=""),
        },
    ),
    # invalid context dict → ValidationError
    (
        {"emitted_content": {"info": "test"}, "new_context": {"key": "value"}},
        {
            "emitted_content": [],
            "new_context": None,
            "success": False,
            "error": HookError(error_code="ValidationError", error_message=""),
        },
    ),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("response, expected", listener_rest_hook_testdata)
async def test_listener_rest_agent_hook(
    response: dict,
    expected: dict,
    context: InteractionContext,
    agent: BaseAgent,
) -> None:
    status_code = response.pop("error_code", 200)
    mock_send = _mock_json_response(json_response=response, status_code=status_code)
    transport = httpx.MockTransport(mock_send)
    hook_name = "test_rest_listener_agent_hook"
    client = httpx.AsyncClient(transport=transport)
    restapi_data = HookRestAPIData(
        url="http://mocked.url", method="POST", headers={"Authorization": "Bearer"}
    )
    hook = FunctionAgentHook(
        name=hook_name,
        rest_api=restapi_data,
        fun=None,
        config_params=CONFIG_PARAMS,
        httpx_client=client,
        trigger=HookPhase.PRE_AGENT_CALL,
        type=HookType.LISTENER,
        break_on_error=False,
    )
    assert hook.name == hook_name
    assert hook.type == "listener"

    executed = None
    async for item in hook.run(context=context, agent=agent):
        executed = item

    assert isinstance(executed, HookResult)
    assert executed.name == hook_name
    assert executed.new_context is None
    assert executed.emitted_content == expected.get("emitted_content")
    assert executed.success == expected.get("success")
    error = expected.get("error")
    if error is not None:
        assert isinstance(executed.error, HookError)
        assert executed.error.error_code == error.error_code
        if isinstance(executed.error.error_code, int):
            assert executed.error.error_message == error.error_message
    else:
        assert executed.error is None


# ===========================================================================
# ON_AGENT_EVENT hook - dict return treated as modified_event
# ===========================================================================


def on_event_hook_returns_base_event(event: BaseEvent, context, **kwargs) -> BaseEvent:
    return BaseEvent(
        event_type=ExecutionEventType.RESPONSE,
        author="modified_author",
        content=[Part(text="modified content")],
    )


def on_event_hook_returns_full_dict(event: BaseEvent, context, **kwargs) -> dict:
    return {
        "event_type": ExecutionEventType.RESPONSE,
        "author": "dict_author",
        "content": [Part(text="from dict")],
    }


def on_event_hook_returns_partial_dict_no_author(event: BaseEvent, context, **kwargs) -> dict:
    return {"event_type": ExecutionEventType.RESPONSE}


def on_event_hook_returns_none(event: BaseEvent, context, **kwargs) -> None:
    return None


def on_event_listener_cannot_modify(event: BaseEvent, context, **kwargs) -> BaseEvent:
    return BaseEvent(event_type=ExecutionEventType.RESPONSE, author="should_fail")


@pytest.fixture
def sample_event() -> BaseEvent:
    return BaseEvent(
        event_type=ExecutionEventType.USER_INPUT,
        author="original_author",
        interaction_id="orig_interaction",
        turn_id="orig_turn",
        content=[Part(text="original content")],
    )


@pytest.mark.asyncio
async def test_on_event_updater_returning_base_event(
    sample_event: BaseEvent, context: InteractionContext
) -> None:
    """Updater returning a BaseEvent sets modified_event on the HookResult."""
    hook = FunctionAgentHook(
        fun=on_event_hook_returns_base_event,
        trigger=HookPhase.ON_AGENT_EVENT,
        type=HookType.UPDATER,
    )
    results = await collect(hook.run(event=sample_event, context=context))

    assert len(results) == 1
    result = results[0]
    assert result.success is True
    assert isinstance(result.modified_event, BaseEvent)
    assert result.modified_event.author == "modified_author"
    assert result.modified_event.event_type == ExecutionEventType.RESPONSE


@pytest.mark.asyncio
async def test_on_event_updater_returning_full_dict_becomes_base_event(
    sample_event: BaseEvent,
    context: InteractionContext,
) -> None:
    """A dict with all fields explicit is converted to BaseEvent by process_result using those values."""
    hook = FunctionAgentHook(
        fun=on_event_hook_returns_full_dict,
        trigger=HookPhase.ON_AGENT_EVENT,
        type=HookType.UPDATER,
    )
    results = await collect(hook.run(event=sample_event, context=context))

    assert len(results) == 1
    result = results[0]
    assert result.success is True
    assert isinstance(result.modified_event, BaseEvent)
    assert result.modified_event.author == "dict_author"
    assert result.modified_event.event_type == ExecutionEventType.RESPONSE


@pytest.mark.asyncio
async def test_on_event_updater_returning_partial_dict_fills_defaults_from_original(
    sample_event: BaseEvent,
    context: InteractionContext,
) -> None:
    """A partial dict missing author/turn_id/interaction_id inherits them from the original event."""
    hook = FunctionAgentHook(
        fun=on_event_hook_returns_partial_dict_no_author,
        trigger=HookPhase.ON_AGENT_EVENT,
        type=HookType.UPDATER,
    )
    results = await collect(hook.run(event=sample_event, context=context))

    assert len(results) == 1
    result = results[0]
    assert result.success is True
    assert isinstance(result.modified_event, BaseEvent)
    assert result.modified_event.event_type == ExecutionEventType.RESPONSE
    # author/interaction_id/turn_id fall back to the original event's values
    assert result.modified_event.author == sample_event.author
    assert result.modified_event.interaction_id == sample_event.interaction_id
    assert result.modified_event.turn_id == sample_event.turn_id


@pytest.mark.asyncio
async def test_on_event_updater_returning_none_passes_through(
    sample_event: BaseEvent, context: InteractionContext
) -> None:
    """Updater returning None leaves modified_event as None (pass-through)."""
    hook = FunctionAgentHook(
        fun=on_event_hook_returns_none,
        trigger=HookPhase.ON_AGENT_EVENT,
        type=HookType.UPDATER,
    )
    results = await collect(hook.run(event=sample_event, context=context))

    assert len(results) == 1
    result = results[0]
    assert result.success is True
    assert result.modified_event is None


@pytest.mark.asyncio
async def test_on_event_listener_returning_modified_event_raises_error(
    sample_event: BaseEvent,
    context: InteractionContext,
) -> None:
    """Listener hook that returns a modified event is rejected with a validation error."""
    hook = FunctionAgentHook(
        fun=on_event_listener_cannot_modify,
        trigger=HookPhase.ON_AGENT_EVENT,
        type=HookType.LISTENER,
    )
    results = await collect(hook.run(event=sample_event, context=context))

    assert len(results) == 1
    result = results[0]
    assert result.success is False
    assert isinstance(result.error, HookError)
    assert result.error.error_code == "ValueError"


updater_rest_hook_testdata: list[tuple[dict, dict[str, Any]]] = [
    # explicit null new_context → ValueError
    (
        {"new_context": None, "emitted_content": []},
        {
            "new_context": None,
            "emitted_content": [],
            "success": False,
            "error": HookError(
                error_code="ValueError",
                error_message="The updater agent hook function must return the new context.",
            ),
        },
    ),
    # list event_content, no new_context → ValueError
    (
        {"emitted_content": ["this", "is", "a", "list"]},
        {
            "emitted_content": [],
            "success": False,
            "error": HookError(error_code="ValueError", error_message=""),
        },
    ),
    # HTTP 500 → error
    (
        {"error": "Something went wrong", "error_code": 500},
        {
            "emitted_content": [],
            "success": False,
            "error": HookError(error_code=500, error_message='{"error":"Something went wrong"}'),
        },
    ),
    # HTTP string error code
    (
        {"error": "Something went wrong", "error_code": "Error"},
        {
            "emitted_content": [],
            "success": False,
            "error": HookError(
                error_code="Error", error_message='{"error":"Something went wrong"}'
            ),
        },
    ),
    # int event_content, no new_context → ValueError
    (
        {"emitted_content": 12345},
        {
            "emitted_content": [],
            "success": False,
            "error": HookError(error_code="ValueError", error_message=""),
        },
    ),
    # arbitrary dict (no new_context) → ValueError
    (
        {"key1": "value1", "key2": "value2"},
        {
            "emitted_content": [],
            "success": False,
            "error": HookError(error_code="ValueError", error_message=""),
        },
    ),
    # explicit null new_context with event_content dict → ValueError
    (
        {"emitted_content": {"key1": "value1", "key2": "value2"}, "new_context": None},
        {
            "emitted_content": [],
            "success": False,
            "error": HookError(error_code="ValueError", error_message=""),
        },
    ),
    # valid new_context (full dict) + dict event_content → success
    (
        {
            "new_context": new_context().model_dump(mode="json"),
            "emitted_content": {"info": "test"},
        },
        {
            "new_context": new_context(),
            "emitted_content": [Part(metadata={"info": "test"})],
            "success": True,
            "error": None,
        },
    ),
    # invalid context dict → ValidationError
    (
        {"emitted_content": {"info": "test"}, "new_context": {"key": "value"}},
        {
            "emitted_content": [],
            "new_context": None,
            "success": False,
            "error": HookError(error_code="ValidationError", error_message=""),
        },
    ),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("response, expected", updater_rest_hook_testdata)
async def test_updater_rest_agent_hook(
    response: dict,
    expected: dict,
    context: InteractionContext,
    agent: BaseAgent,
) -> None:
    status_code = response.pop("error_code", 200)
    mock_send = _mock_json_response(json_response=response, status_code=status_code)
    transport = httpx.MockTransport(mock_send)
    hook_name = "test_rest_updater_agent_hook"
    client = httpx.AsyncClient(transport=transport)
    restapi_data = HookRestAPIData(
        url="http://mocked.url", method="POST", headers={"Authorization": "Bearer"}
    )
    hook = FunctionAgentHook(
        name=hook_name,
        rest_api=restapi_data,
        fun=None,
        config_params=CONFIG_PARAMS,
        httpx_client=client,
        trigger=HookPhase.PRE_AGENT_CALL,
        type=HookType.UPDATER,
        break_on_error=False,
    )
    assert hook.name == hook_name
    assert hook.type == "updater"

    executed = None
    async for item in hook.run(context=context, agent=agent):
        executed = item

    assert isinstance(executed, HookResult)
    assert executed.name == hook_name
    assert executed.new_context == expected.get("new_context")
    assert executed.emitted_content == expected.get("emitted_content")
    assert executed.success == expected.get("success")
    error = expected.get("error")
    if error is not None:
        assert isinstance(executed.error, HookError)
        assert executed.error.error_code == error.error_code
        if isinstance(executed.error.error_code, int):
            assert executed.error.error_message == error.error_message
    else:
        assert executed.error is None
