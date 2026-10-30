"""Tests for the marker on visibility-only tool events.

hermes reports the tools it ran itself through TOOL_START/TOOL_END/TOOL_ERROR that are built
exactly like a real tool request. The platform runner must not execute those — it holds tools of
its own that may share a name — so it needs to tell them apart from the event alone. That is what
``METADATA_VISIBILITY_ONLY`` is for; these tests pin the producing half of that contract, the
consuming half lives in ``tests/runner/test_delegate_visibility_tool_start.py``.
"""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

from domyn.agents.core import (
    METADATA_VISIBILITY_ONLY,
    AgentAction,
    BaseEvent,
    ExecutionEventType,
)
from domyn.agents.integrations.hermes_platform_gateway.adapter import (
    DomynPlatformAdapter,
    _build_session_key,
)

CHANNEL_ID = "chan-1"
CHAT_ID = "conv-1"
_PKG = "domyn.agents.integrations.hermes_platform_gateway"


def _make_adapter() -> tuple[DomynPlatformAdapter, MagicMock]:
    mock_client = MagicMock()
    mock_client.send_event = AsyncMock()

    adapter = DomynPlatformAdapter(
        config=MagicMock(),
        channel_id=CHANNEL_ID,
        relay_client_factory=lambda _cb: mock_client,
        tool_rename_mapping={},
    )
    adapter._turn_by_session[_build_session_key(CHANNEL_ID, CHAT_ID)] = BaseEvent(
        event_type=ExecutionEventType.AGENT_START,
        author="u1",
        conversation_id=CHAT_ID,
        interaction_id="int-1",
        turn_id="turn-1",
        action=AgentAction(name="invoke", parameters={"input": "hi"}),
    )
    adapter._client = mock_client
    return adapter, mock_client


def _sent_event(mock_client: MagicMock) -> BaseEvent:
    mock_client.send_event.assert_awaited_once()
    return mock_client.send_event.await_args.args[0]


def test_visibility_tool_start_is_marked():
    adapter, mock_client = _make_adapter()

    asyncio.run(
        adapter.emit_visibility_tool_start(
            chat_id=CHAT_ID,
            tool_name="web_search",
            args={"query": "q"},
            thought=None,
            call_id="visibility-1",
        )
    )

    event = _sent_event(mock_client)
    assert event.event_type == ExecutionEventType.TOOL_START
    assert event.metadata.get(METADATA_VISIBILITY_ONLY) is True


def test_visibility_tool_end_is_marked():
    """Marked too, though the runner matches END by call_id: the flag describes the event."""
    adapter, mock_client = _make_adapter()

    asyncio.run(
        adapter.emit_visibility_tool_end(
            chat_id=CHAT_ID,
            tool_name="web_search",
            call_id="visibility-1",
            observation="done",
        )
    )

    event = _sent_event(mock_client)
    assert event.event_type == ExecutionEventType.TOOL_END
    assert event.metadata.get(METADATA_VISIBILITY_ONLY) is True


def test_visibility_tool_end_carries_the_tool_metadata():
    """A hermes built-in's own ToolResult.metadata reaches the platform, the same
    way a local tool's TOOL_END carries it."""
    adapter, mock_client = _make_adapter()

    asyncio.run(
        adapter.emit_visibility_tool_end(
            chat_id=CHAT_ID,
            tool_name="web_search",
            call_id="visibility-1",
            observation="done",
            metadata={"token_usage": {"total": 12}},
        )
    )

    event = _sent_event(mock_client)
    assert event.metadata.get("token_usage") == {"total": 12}
    assert event.metadata.get(METADATA_VISIBILITY_ONLY) is True


def test_tool_metadata_cannot_clear_the_visibility_flag():
    """The flag says what the event *is*; a tool overwriting it would make the
    runner execute a call nobody made."""
    adapter, mock_client = _make_adapter()

    asyncio.run(
        adapter.emit_visibility_tool_end(
            chat_id=CHAT_ID,
            tool_name="web_search",
            call_id="visibility-1",
            observation="done",
            metadata={METADATA_VISIBILITY_ONLY: False},
        )
    )

    assert _sent_event(mock_client).metadata.get(METADATA_VISIBILITY_ONLY) is True


def test_visibility_tool_error_is_marked():
    adapter, mock_client = _make_adapter()

    asyncio.run(
        adapter.emit_visibility_tool_error(
            chat_id=CHAT_ID,
            tool_name="web_search",
            call_id="visibility-1",
            error_message="boom",
        )
    )

    event = _sent_event(mock_client)
    assert event.event_type == ExecutionEventType.TOOL_ERROR
    assert event.metadata.get(METADATA_VISIBILITY_ONLY) is True


def _post_tool_hook(monkeypatch):
    """Run register() with a stubbed adapter and return (hook, mock_adapter), with
    a pending visibility call so the hook reaches the TOOL_END branch."""
    monkeypatch.setenv("DOMYN_API_KEY", "key-1")
    monkeypatch.setenv("DOMYN_BASE_URL", "api.example.com")
    monkeypatch.setenv("DOMYN_SPACE_ID", "space-1")
    monkeypatch.setenv("DOMYN_CHANNEL_ID", CHANNEL_ID)

    mock_adapter = MagicMock()
    mock_adapter._chat_id_by_task = {"task-1": CHAT_ID}
    mock_adapter.pop_visibility_call.return_value = "visibility-1"

    with (
        patch(f"{_PKG}.client.fetch_tools", return_value=[]),
        patch(f"{_PKG}._start_refresh_loop"),
        patch(f"{_PKG}.adapter.DomynPlatformAdapter", return_value=mock_adapter),
        patch(f"{_PKG}.relay_client.DomynRelayClient"),
    ):
        ctx = MagicMock()
        from domyn.agents.integrations.hermes_platform_gateway import register

        register(ctx)
        factory = ctx.register_platform.call_args.kwargs["adapter_factory"]
        factory(MagicMock())

    hook = next(
        c.args[1] for c in ctx.register_hook.call_args_list if c.args[0] == "post_tool_call"
    )
    return hook, mock_adapter


def test_post_tool_call_forwards_the_tool_result_metadata(monkeypatch):
    """The producing end of the contract: a built-in returning a ToolResult with
    metadata hands it to the TOOL_END event instead of dropping it."""
    hook, mock_adapter = _post_tool_hook(monkeypatch)
    result = json.dumps(
        {
            "success": True,
            "data": "done",
            "metadata": {"token_usage": {"total": 12}},
        }
    )

    hook(tool_name="web_search", task_id="task-1", result=result, error=None)

    kwargs = mock_adapter.emit_visibility_tool_end.call_args.kwargs
    assert kwargs["observation"] == "done"
    assert kwargs["metadata"] == {"token_usage": {"total": 12}}


def test_post_tool_call_forwards_metadata_on_the_error_path(monkeypatch):
    """A ToolResult carrying `error` routes to TOOL_ERROR — the metadata has to
    ride that event too, or the contract only half lands."""
    hook, mock_adapter = _post_tool_hook(monkeypatch)
    result = json.dumps(
        {
            "success": False,
            "data": None,
            "error": "upstream timed out",
            "metadata": {"attempts": 3},
        }
    )

    hook(tool_name="web_search", task_id="task-1", result=result, error=None)

    mock_adapter.emit_visibility_tool_end.assert_not_called()
    kwargs = mock_adapter.emit_visibility_tool_error.call_args.kwargs
    assert kwargs["error_message"] == "upstream timed out"
    assert kwargs["metadata"] == {"attempts": 3}


def test_visibility_tool_error_keeps_the_flag_over_tool_metadata():
    adapter, mock_client = _make_adapter()

    asyncio.run(
        adapter.emit_visibility_tool_error(
            chat_id=CHAT_ID,
            tool_name="web_search",
            call_id="visibility-1",
            error_message="boom",
            metadata={"attempts": 3, METADATA_VISIBILITY_ONLY: False},
        )
    )

    event = _sent_event(mock_client)
    assert event.metadata.get("attempts") == 3
    assert event.metadata.get(METADATA_VISIBILITY_ONLY) is True


def test_post_tool_call_without_metadata_sends_none(monkeypatch):
    """A tool that carries no metadata must not manufacture an empty dict — the
    event's own default already covers that."""
    hook, mock_adapter = _post_tool_hook(monkeypatch)

    hook(tool_name="web_search", task_id="task-1", result="plain output", error=None)

    assert mock_adapter.emit_visibility_tool_end.call_args.kwargs["metadata"] is None


def test_real_tool_call_is_not_marked():
    """The round-trip TOOL_START must stay unmarked, or the platform would never run anything."""
    adapter, mock_client = _make_adapter()

    asyncio.run(
        adapter.call_tool(
            session_key=_build_session_key(CHANNEL_ID, CHAT_ID),
            tool_name="web_search",
            args={"query": "q"},
            timeout=0.01,
        )
    )

    event = _sent_event(mock_client)
    assert event.event_type == ExecutionEventType.TOOL_START
    assert not event.metadata.get(METADATA_VISIBILITY_ONLY)
