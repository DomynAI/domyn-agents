"""Tests for `input_fields` driving non-REST function-hook kwargs.

When `input_fields` is set on a non-REST FunctionHook, the underlying callable
receives only the selected fields (by name), not the canonical phase_params
(`context`, `agent`, `tool`, ...).
"""

from datetime import datetime
from typing import Any

import httpx
import pytest

from domyn.agents import BaseEvent, ExecutionEventType, Part
from domyn.agents.core import BaseAgent, InteractionContext
from domyn.agents.hooks import FunctionHook
from domyn.agents.hooks.base import (
    HookPhase,
    HookRestAPIData,
    HookResult,
    HookType,
    InputFields,
)
from domyn.agents.tools import FunctionTool, ToolResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _drain(hook: FunctionHook, **kwargs: Any) -> HookResult:
    """Run the hook and return the final non-update HookResult."""
    final: HookResult | None = None
    async for item in hook.run(**kwargs):
        if not item.is_update:
            final = item
    assert final is not None, "hook did not yield a final HookResult"
    return final


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def agent() -> BaseAgent:
    return BaseAgent(name="my_agent", description="mock")


@pytest.fixture
def last_event() -> BaseEvent:
    return BaseEvent(
        event_type=ExecutionEventType.LLM_END,
        author="agent",
        timestamp=datetime(2024, 1, 1),
        content=[Part(text="hi")],
        event_id="evt-1",
    )


@pytest.fixture
def context(last_event: BaseEvent) -> InteractionContext:
    return InteractionContext(
        conversation_id="c1",
        user_id="u1",
        interaction_id="i1",
        created_at=datetime(2024, 1, 1),
        updated_at=datetime(2024, 1, 1),
        agent_input=BaseEvent(
            event_type=ExecutionEventType.USER_INPUT,
            author="user",
            timestamp=datetime(2024, 1, 1),
            content=[Part(text="hello")],
            event_id="evt-0",
        ),
        last_event=last_event,
        interaction_history=[last_event],
    )


def search() -> None:
    pass


@pytest.fixture
def tool() -> FunctionTool:
    return FunctionTool(func=search)


# ---------------------------------------------------------------------------
# Non-REST: kwargs come from input_fields (mutually exclusive with phase_params)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pre_tool_call_payload_fields_kwargs(tool: FunctionTool) -> None:
    captured: dict[str, Any] = {}

    def hook_fn(tool_name: str, tool_args: dict) -> None:
        captured["tool_name"] = tool_name
        captured["tool_args"] = tool_args

    hook = FunctionHook(
        trigger=HookPhase.PRE_TOOL_CALL,
        type=HookType.LISTENER,
        fun=hook_fn,
        input_fields=[InputFields.TOOL_NAME, InputFields.TOOL_ARGS],
    )

    args = {"q": "anything"}
    final: HookResult | None = None
    async for item in hook.run(tool=tool, tool_args=args):
        if not item.is_update:
            final = item

    assert final is not None and final.success is True
    assert captured == {"tool_name": "search", "tool_args": args}


@pytest.mark.asyncio
async def test_pre_agent_call_payload_fields_native_objects(
    agent: BaseAgent, context: InteractionContext, last_event: BaseEvent
) -> None:
    """Non-REST hooks must receive native objects (BaseEvent), not JSON-dumped dicts."""
    captured: dict[str, Any] = {}

    def hook_fn(agent_name: str, last_event: BaseEvent) -> None:
        captured["agent_name"] = agent_name
        captured["last_event"] = last_event

    hook = FunctionHook(
        trigger=HookPhase.PRE_AGENT_CALL,
        type=HookType.LISTENER,
        fun=hook_fn,
        input_fields=[InputFields.AGENT_NAME, InputFields.LAST_EVENT],
    )

    final: HookResult | None = None
    async for item in hook.run(context=context, agent=agent):
        if not item.is_update:
            final = item

    assert final is not None and final.success is True
    assert captured["agent_name"] == "my_agent"
    assert isinstance(captured["last_event"], BaseEvent)
    assert captured["last_event"].event_id == last_event.event_id


@pytest.mark.asyncio
async def test_payload_fields_signature_mutually_exclusive_with_phase_params(
    tool: FunctionTool,
) -> None:
    """Function declaring only payload fields (no `tool` kwarg) must work."""

    def hook_fn(tool_args: dict) -> None:
        # Signature deliberately omits `tool` — would TypeError if canonical kwargs leaked through.
        assert "tool" not in tool_args

    hook = FunctionHook(
        trigger=HookPhase.PRE_TOOL_CALL,
        type=HookType.LISTENER,
        fun=hook_fn,
        input_fields=[InputFields.TOOL_ARGS],
    )

    final: HookResult | None = None
    async for item in hook.run(tool=tool, tool_args={"a": 1}):
        if not item.is_update:
            final = item
    assert final is not None and final.success is True


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_payload_field_outside_trigger_whitelist_rejected() -> None:
    def hook_fn(tool_name: str) -> None:
        pass

    with pytest.raises(ValueError, match="Invalid input fields"):
        FunctionHook(
            trigger=HookPhase.PRE_AGENT_CALL,  # AGENT_NAME ok, TOOL_NAME not
            type=HookType.LISTENER,
            fun=hook_fn,
            input_fields=[InputFields.TOOL_NAME],
        )


def test_signature_missing_payload_field_rejected() -> None:
    def hook_fn(agent_name: str) -> None:  # missing `last_event`
        pass

    with pytest.raises(ValueError, match="missing expected parameters"):
        FunctionHook(
            trigger=HookPhase.PRE_AGENT_CALL,
            type=HookType.LISTENER,
            fun=hook_fn,
            input_fields=[InputFields.AGENT_NAME, InputFields.LAST_EVENT],
        )


# ---------------------------------------------------------------------------
# REST regression: input_fields still produces a JSON-serialized body
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rest_payload_fields_body_is_json_serialized(
    agent: BaseAgent, context: InteractionContext, last_event: BaseEvent
) -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    hook = FunctionHook(
        trigger=HookPhase.PRE_AGENT_CALL,
        type=HookType.LISTENER,
        fun=None,
        rest_api=HookRestAPIData(url="http://x", method="POST"),
        httpx_client=client,
        input_fields=[InputFields.AGENT_NAME, InputFields.LAST_EVENT],
    )

    async for _ in hook.run(context=context, agent=agent):
        pass

    body = captured["body"]
    assert body["agent_name"] == "my_agent"
    assert isinstance(body["last_event"], dict)
    assert body["last_event"]["event_id"] == last_event.event_id


@pytest.mark.asyncio
async def test_rest_payload_fields_interaction_history_serialized(
    agent: BaseAgent, context: InteractionContext, last_event: BaseEvent
) -> None:
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    hook = FunctionHook(
        trigger=HookPhase.PRE_AGENT_CALL,
        type=HookType.LISTENER,
        fun=None,
        rest_api=HookRestAPIData(url="http://x", method="POST"),
        httpx_client=client,
        input_fields=[InputFields.INTERACTION_HISTORY],
    )

    async for _ in hook.run(context=context, agent=agent):
        pass

    history = captured["body"]["interaction_history"]
    assert isinstance(history, list)
    assert history[0]["event_id"] == last_event.event_id


@pytest.mark.asyncio
async def test_post_tool_call_payload_fields_native_tool_result(tool: FunctionTool) -> None:
    captured: dict[str, Any] = {}

    def hook_fn(tool_result: ToolResult) -> None:
        captured["tool_result"] = tool_result

    hook = FunctionHook(
        trigger=HookPhase.POST_TOOL_CALL,
        type=HookType.LISTENER,
        fun=hook_fn,
        input_fields=[InputFields.TOOL_RESULT],
    )

    result = ToolResult(success=True, data={"k": "v"})
    final: HookResult | None = None
    async for item in hook.run(tool=tool, result=result):
        if not item.is_update:
            final = item

    assert final is not None and final.success is True
    assert isinstance(captured["tool_result"], ToolResult)
    assert captured["tool_result"].data == {"k": "v"}


# ---------------------------------------------------------------------------
# A. _filter_args projection — every field arm
# ---------------------------------------------------------------------------


@pytest.fixture
def empty_history_context() -> InteractionContext:
    """Context whose user_input is not USER_INPUT — agent_input + last_event resolve to None."""
    return InteractionContext(
        conversation_id="c1",
        user_id="u1",
        interaction_id="i1",
        agent_input=BaseEvent(
            event_type=ExecutionEventType.LLM_END,
            author="agent",
            timestamp=datetime(2024, 1, 1),
            content=[Part(text="x")],
            event_id="evt-x",
        ),
    )


@pytest.mark.asyncio
async def test_agent_input_projected_from_user_input_text(
    agent: BaseAgent, context: InteractionContext
) -> None:
    captured: dict[str, Any] = {}

    def hook_fn(agent_input: str) -> None:
        captured["agent_input"] = agent_input

    hook = FunctionHook(
        trigger=HookPhase.PRE_AGENT_CALL,
        type=HookType.LISTENER,
        fun=hook_fn,
        input_fields=[InputFields.AGENT_INPUT],
    )
    await _drain(hook, context=context, agent=agent)
    assert captured["agent_input"] == "hello"


@pytest.mark.asyncio
async def test_agent_input_empty_string_when_context_has_no_input(
    agent: BaseAgent, empty_history_context: InteractionContext
) -> None:
    captured: dict[str, Any] = {}

    def hook_fn(agent_input: str) -> None:
        captured["agent_input"] = agent_input

    hook = FunctionHook(
        trigger=HookPhase.PRE_AGENT_CALL,
        type=HookType.LISTENER,
        fun=hook_fn,
        input_fields=[InputFields.AGENT_INPUT],
    )
    await _drain(hook, context=empty_history_context, agent=agent)
    assert captured["agent_input"] == ""


@pytest.mark.asyncio
async def test_interaction_history_projected_as_list(
    agent: BaseAgent, context: InteractionContext, last_event: BaseEvent
) -> None:
    captured: dict[str, Any] = {}

    def hook_fn(interaction_history: list) -> None:
        captured["interaction_history"] = interaction_history

    hook = FunctionHook(
        trigger=HookPhase.PRE_AGENT_CALL,
        type=HookType.LISTENER,
        fun=hook_fn,
        input_fields=[InputFields.INTERACTION_HISTORY],
    )
    await _drain(hook, context=context, agent=agent)
    history = captured["interaction_history"]
    assert isinstance(history, list)
    assert len(history) == 1
    assert isinstance(history[0], BaseEvent)
    assert history[0].event_id == last_event.event_id


@pytest.mark.asyncio
async def test_interaction_history_empty_when_context_none(agent: BaseAgent) -> None:
    captured: dict[str, Any] = {}

    def hook_fn(interaction_history: list) -> None:
        captured["interaction_history"] = interaction_history

    hook = FunctionHook(
        trigger=HookPhase.PRE_AGENT_CALL,
        type=HookType.LISTENER,
        fun=hook_fn,
        input_fields=[InputFields.INTERACTION_HISTORY],
    )
    await _drain(hook, agent=agent)
    assert captured["interaction_history"] == []


@pytest.mark.asyncio
async def test_last_event_none_when_history_empty(
    agent: BaseAgent, empty_history_context: InteractionContext
) -> None:
    captured: dict[str, Any] = {}

    def hook_fn(last_event: BaseEvent | None) -> None:
        captured["last_event"] = last_event

    hook = FunctionHook(
        trigger=HookPhase.PRE_AGENT_CALL,
        type=HookType.LISTENER,
        fun=hook_fn,
        input_fields=[InputFields.LAST_EVENT],
    )
    await _drain(hook, context=empty_history_context, agent=agent)
    assert captured["last_event"] is None


@pytest.mark.asyncio
async def test_agent_name_empty_string_when_agent_missing(
    context: InteractionContext,
) -> None:
    captured: dict[str, Any] = {}

    def hook_fn(agent_name: str) -> None:
        captured["agent_name"] = agent_name

    hook = FunctionHook(
        trigger=HookPhase.PRE_AGENT_CALL,
        type=HookType.LISTENER,
        fun=hook_fn,
        input_fields=[InputFields.AGENT_NAME],
    )
    await _drain(hook, context=context)
    assert captured["agent_name"] == ""


@pytest.mark.asyncio
async def test_current_event_passed_through_for_on_agent_event(
    last_event: BaseEvent,
) -> None:
    captured: dict[str, Any] = {}

    def hook_fn(current_event: BaseEvent) -> None:
        captured["current_event"] = current_event

    hook = FunctionHook(
        trigger=HookPhase.ON_AGENT_EVENT,
        type=HookType.LISTENER,
        fun=hook_fn,
        input_fields=[InputFields.CURRENT_EVENT],
    )
    await _drain(hook, event=last_event)
    assert isinstance(captured["current_event"], BaseEvent)
    assert captured["current_event"].event_id == last_event.event_id


@pytest.mark.asyncio
async def test_tool_args_empty_dict_when_none(tool: FunctionTool) -> None:
    captured: dict[str, Any] = {}

    def hook_fn(tool_args: dict) -> None:
        captured["tool_args"] = tool_args

    hook = FunctionHook(
        trigger=HookPhase.PRE_TOOL_CALL,
        type=HookType.LISTENER,
        fun=hook_fn,
        input_fields=[InputFields.TOOL_ARGS],
    )
    await _drain(hook, tool=tool, tool_args=None)
    assert captured["tool_args"] == {}


# ---------------------------------------------------------------------------
# B. Trigger x field whitelist (validate_input_fields)
# ---------------------------------------------------------------------------


def test_post_agent_call_accepts_agent_fields() -> None:
    def hook_fn(
        agent_input: str,
        agent_name: str,
        last_event: BaseEvent | None,
        interaction_history: list,
    ) -> None:
        pass

    FunctionHook(
        trigger=HookPhase.POST_AGENT_CALL,
        type=HookType.LISTENER,
        fun=hook_fn,
        input_fields=[
            InputFields.AGENT_INPUT,
            InputFields.AGENT_NAME,
            InputFields.LAST_EVENT,
            InputFields.INTERACTION_HISTORY,
        ],
    )


def test_post_agent_call_rejects_tool_result() -> None:
    def hook_fn(tool_result: Any) -> None:
        pass

    with pytest.raises(ValueError, match="Invalid input fields"):
        FunctionHook(
            trigger=HookPhase.POST_AGENT_CALL,
            type=HookType.LISTENER,
            fun=hook_fn,
            input_fields=[InputFields.TOOL_RESULT],
        )


def test_pre_tool_call_rejects_tool_result() -> None:
    def hook_fn(tool_result: Any) -> None:
        pass

    with pytest.raises(ValueError, match="Invalid input fields"):
        FunctionHook(
            trigger=HookPhase.PRE_TOOL_CALL,
            type=HookType.LISTENER,
            fun=hook_fn,
            input_fields=[InputFields.TOOL_RESULT],
        )


def test_post_tool_call_rejects_tool_args() -> None:
    def hook_fn(tool_args: dict) -> None:
        pass

    with pytest.raises(ValueError, match="Invalid input fields"):
        FunctionHook(
            trigger=HookPhase.POST_TOOL_CALL,
            type=HookType.LISTENER,
            fun=hook_fn,
            input_fields=[InputFields.TOOL_ARGS],
        )


def test_on_agent_event_rejects_agent_name() -> None:
    def hook_fn(agent_name: str) -> None:
        pass

    with pytest.raises(ValueError, match="Invalid input fields"):
        FunctionHook(
            trigger=HookPhase.ON_AGENT_EVENT,
            type=HookType.LISTENER,
            fun=hook_fn,
            input_fields=[InputFields.AGENT_NAME],
        )


@pytest.mark.asyncio
async def test_empty_input_fields_list_accepted(tool: FunctionTool) -> None:
    called = {"hit": False}

    def hook_fn() -> None:
        called["hit"] = True

    hook = FunctionHook(
        trigger=HookPhase.PRE_TOOL_CALL,
        type=HookType.LISTENER,
        fun=hook_fn,
        input_fields=[],
    )
    final = await _drain(hook, tool=tool, tool_args={"a": 1})
    assert final.success is True
    assert called["hit"] is True


# ---------------------------------------------------------------------------
# C. Legacy regression: input_fields=None preserves canonical phase params
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_input_fields_none_passes_canonical_phase_params(
    agent: BaseAgent, context: InteractionContext
) -> None:
    captured: dict[str, Any] = {}

    def hook_fn(context: InteractionContext, agent: BaseAgent) -> None:
        captured["context"] = context
        captured["agent"] = agent

    hook = FunctionHook(
        trigger=HookPhase.PRE_AGENT_CALL,
        type=HookType.LISTENER,
        fun=hook_fn,
    )
    assert hook.input_fields is None
    await _drain(hook, context=context, agent=agent)
    assert captured["context"] is context
    assert captured["agent"] is agent


@pytest.mark.asyncio
async def test_input_fields_none_post_tool_call_passes_tool_and_result(
    tool: FunctionTool,
) -> None:
    captured: dict[str, Any] = {}

    def hook_fn(tool: FunctionTool, result: ToolResult) -> None:
        captured["tool"] = tool
        captured["result"] = result

    hook = FunctionHook(
        trigger=HookPhase.POST_TOOL_CALL,
        type=HookType.LISTENER,
        fun=hook_fn,
    )
    result = ToolResult(success=True, data={"x": 1})
    await _drain(hook, tool=tool, result=result)
    assert captured["tool"] is tool
    assert captured["result"] is result


# ---------------------------------------------------------------------------
# D. _create_body (REST) — extras vs. no extras
# ---------------------------------------------------------------------------


def _capture_transport(captured: dict[str, Any]) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        import json

        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={})

    return httpx.MockTransport(handler)


@pytest.mark.asyncio
async def test_rest_input_fields_body_omits_config_params_extras(
    agent: BaseAgent, context: InteractionContext
) -> None:
    captured: dict[str, Any] = {}
    client = httpx.AsyncClient(transport=_capture_transport(captured))
    hook = FunctionHook(
        trigger=HookPhase.PRE_AGENT_CALL,
        type=HookType.LISTENER,
        fun=None,
        rest_api=HookRestAPIData(url="http://x", method="POST"),
        httpx_client=client,
        input_fields=[InputFields.AGENT_NAME],
    )
    async for _ in hook.run(context=context, agent=agent):
        pass

    body = captured["body"]
    assert set(body.keys()) == {"agent_name"}


@pytest.mark.asyncio
async def test_rest_input_fields_none_body_includes_config_params_envelope(
    agent: BaseAgent, context: InteractionContext
) -> None:
    captured: dict[str, Any] = {}
    client = httpx.AsyncClient(transport=_capture_transport(captured))
    hook = FunctionHook(
        trigger=HookPhase.PRE_AGENT_CALL,
        type=HookType.LISTENER,
        fun=None,
        rest_api=HookRestAPIData(url="http://x", method="POST"),
        httpx_client=client,
        config_params={"extra_key": "extra_value"},
    )
    async for _ in hook.run(context=context, agent=agent):
        pass

    body = captured["body"]
    assert "context" in body
    assert "agent" in body
    assert body["config_params"] == {"extra_key": "extra_value"}


@pytest.mark.asyncio
async def test_rest_post_tool_call_tool_result_serialized(tool: FunctionTool) -> None:
    captured: dict[str, Any] = {}
    client = httpx.AsyncClient(transport=_capture_transport(captured))
    hook = FunctionHook(
        trigger=HookPhase.POST_TOOL_CALL,
        type=HookType.LISTENER,
        fun=None,
        rest_api=HookRestAPIData(url="http://x", method="POST"),
        httpx_client=client,
        input_fields=[InputFields.TOOL_NAME, InputFields.TOOL_RESULT],
    )
    result = ToolResult(success=True, data={"k": "v"})
    async for _ in hook.run(tool=tool, result=result):
        pass

    body = captured["body"]
    assert body["tool_name"] == "search"
    assert isinstance(body["tool_result"], dict)
    assert body["tool_result"]["data"] == {"k": "v"}


@pytest.mark.asyncio
async def test_rest_on_agent_event_current_event_serialized(last_event: BaseEvent) -> None:
    captured: dict[str, Any] = {}
    client = httpx.AsyncClient(transport=_capture_transport(captured))
    hook = FunctionHook(
        trigger=HookPhase.ON_AGENT_EVENT,
        type=HookType.LISTENER,
        fun=None,
        rest_api=HookRestAPIData(url="http://x", method="POST"),
        httpx_client=client,
        input_fields=[InputFields.CURRENT_EVENT],
    )
    async for _ in hook.run(event=last_event):
        pass

    body = captured["body"]
    assert isinstance(body["current_event"], dict)
    assert body["current_event"]["event_id"] == last_event.event_id


@pytest.mark.asyncio
async def test_rest_input_fields_none_value_serialized_as_empty_dict(
    agent: BaseAgent, empty_history_context: InteractionContext
) -> None:
    captured: dict[str, Any] = {}
    client = httpx.AsyncClient(transport=_capture_transport(captured))
    hook = FunctionHook(
        trigger=HookPhase.PRE_AGENT_CALL,
        type=HookType.LISTENER,
        fun=None,
        rest_api=HookRestAPIData(url="http://x", method="POST"),
        httpx_client=client,
        input_fields=[InputFields.LAST_EVENT],
    )
    async for _ in hook.run(context=empty_history_context, agent=agent):
        pass

    assert captured["body"] == {"last_event": {}}


# ---------------------------------------------------------------------------
# E. Updater hooks under input_fields
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_updater_pre_tool_call_can_return_new_tool_args_via_input_fields(
    tool: FunctionTool,
) -> None:
    def hook_fn(tool_name: str, tool_args: dict) -> dict:
        return {"new_tool_args": {**tool_args, "added": True}}

    hook = FunctionHook(
        trigger=HookPhase.PRE_TOOL_CALL,
        type=HookType.UPDATER,
        fun=hook_fn,
        input_fields=[InputFields.TOOL_NAME, InputFields.TOOL_ARGS],
    )
    final = await _drain(hook, tool=tool, tool_args={"q": "hi"})
    assert final.success is True
    assert final.new_tool_args == {"q": "hi", "added": True}


@pytest.mark.asyncio
async def test_updater_on_agent_event_returns_modified_baseevent(
    last_event: BaseEvent,
) -> None:
    def hook_fn(current_event: BaseEvent) -> BaseEvent:
        return BaseEvent(
            event_type=current_event.event_type,
            author=current_event.author,
            timestamp=current_event.timestamp,
            content=[Part(text="rewritten")],
            event_id="evt-modified",
            turn_id=current_event.turn_id,
            interaction_id=current_event.interaction_id,
        )

    hook = FunctionHook(
        trigger=HookPhase.ON_AGENT_EVENT,
        type=HookType.UPDATER,
        fun=hook_fn,
        input_fields=[InputFields.CURRENT_EVENT],
    )
    final = await _drain(hook, event=last_event)
    assert final.success is True
    assert isinstance(final.modified_event, BaseEvent)
    assert final.modified_event.event_id == "evt-modified"


@pytest.mark.asyncio
async def test_listener_on_agent_event_returning_event_is_rejected(
    last_event: BaseEvent,
) -> None:
    def hook_fn(current_event: BaseEvent) -> BaseEvent:
        return last_event

    hook = FunctionHook(
        trigger=HookPhase.ON_AGENT_EVENT,
        type=HookType.LISTENER,
        fun=hook_fn,
        input_fields=[InputFields.CURRENT_EVENT],
    )
    final = await _drain(hook, event=last_event)
    assert final.success is False
    assert final.error is not None
    assert "Listener on_agent_event hook cannot modify the event" in (
        final.error.error_message or ""
    )


# ---------------------------------------------------------------------------
# F. Decorator integration
# ---------------------------------------------------------------------------


@pytest.fixture
def _clean_registry():
    from domyn.agents.registry import get_hook_registry

    get_hook_registry().clear()
    yield
    get_hook_registry().clear()


@pytest.mark.asyncio
async def test_decorator_input_fields_flows_to_hook(
    _clean_registry: None, agent: BaseAgent, context: InteractionContext
) -> None:
    from domyn.agents.core.decorators import hook as hook_deco

    captured: dict[str, Any] = {}

    @hook_deco(
        trigger=HookPhase.PRE_AGENT_CALL,
        is_listener=True,
        input_fields=[InputFields.AGENT_NAME],
        name="decorator_input_fields_hook",
    )
    def my_hook(agent_name: str) -> None:
        captured["agent_name"] = agent_name

    assert isinstance(my_hook, FunctionHook)
    assert my_hook.input_fields == [InputFields.AGENT_NAME]

    await _drain(my_hook, context=context, agent=agent)
    assert captured["agent_name"] == "my_agent"
