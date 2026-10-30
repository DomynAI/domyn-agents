"""Tests for the user-cancel path (``_handle_cancel``) and its interaction
with background delegations.

hermes' ``/stop`` only interrupts the session's RUNNING agent; subagents
dispatched with ``delegate_task(background=true)`` are detached from the
parent's child list and live in the module-level ``tools.async_delegation``
registry. On cancel the adapter must:

1. interrupt them explicitly via ``interrupt_for_session`` with the GATEWAY
   session key (the registry knows nothing about the adapter-internal key);
2. clear the turn's ``_bg_delegation_by_session`` in-flight counter (the
   dropped completions would never drain it) so the NEXT turn's final
   answer isn't downgraded to a non-terminal RESPONSE;
3. swallow the late ``[ASYNC DELEGATION ...]`` completion re-injection in
   ``handle_message`` instead of letting it forge a new agent turn — keyed
   by the delegation ids read from the registry at cancel time (so a NEW
   dispatch after the cancel is never swallowed), with a conversation-level
   fallback when the registry is unreadable.
"""

import asyncio
import sys
import types
from unittest.mock import AsyncMock, MagicMock

import pytest

from domyn.agents.core import AgentAction, BaseEvent, ExecutionEventType
from domyn.agents.integrations.hermes_platform_gateway.adapter import (
    DomynPlatformAdapter,
    _build_session_key,
)


def _make_adapter(channel_id: str = "chan-1", conv_id: str = "conv-1"):
    mock_client = MagicMock()
    mock_client.send_event = AsyncMock()

    adapter = DomynPlatformAdapter(
        config=MagicMock(),
        channel_id=channel_id,
        relay_client_factory=lambda _cb: mock_client,
        tool_rename_mapping={},
    )
    turn = BaseEvent(
        event_type=ExecutionEventType.AGENT_START,
        author="u1",
        conversation_id=conv_id,
        interaction_id="int-1",
        turn_id="turn-1",
        action=AgentAction(name="invoke", parameters={"input": "hello"}),
    )
    adapter._turn_by_session[_build_session_key(channel_id, conv_id)] = turn
    adapter._client = mock_client
    return adapter, mock_client


def _cancel_event(conv_id: str = "conv-1") -> BaseEvent:
    return BaseEvent(
        event_type=ExecutionEventType.AGENT_END,
        author="u1",
        conversation_id=conv_id,
        error_code="user_cancelled",
    )


def _install_async_delegation(monkeypatch, mod: types.ModuleType) -> None:
    tools_pkg = types.ModuleType("tools")
    tools_pkg.async_delegation = mod
    monkeypatch.setitem(sys.modules, "tools", tools_pkg)
    monkeypatch.setitem(sys.modules, "tools.async_delegation", mod)


def _session_records() -> dict:
    """Registry contents shared by both fixtures: two conv-1 delegations
    (one running, one already completed — its completion may still be queued)
    and one belonging to another conversation."""
    return {
        "deleg_match": {
            "delegation_id": "deleg_match",
            "status": "running",
            "session_key": "agent:main:domyn:dm:conv-1",
            "interrupt_fn": MagicMock(),
        },
        "deleg_other_session": {
            "delegation_id": "deleg_other_session",
            "status": "running",
            "session_key": "agent:main:domyn:dm:conv-OTHER",
            "interrupt_fn": MagicMock(),
        },
        "deleg_done": {
            "delegation_id": "deleg_done",
            "status": "completed",
            "session_key": "agent:main:domyn:dm:conv-1",
            "interrupt_fn": MagicMock(),
        },
    }


@pytest.fixture
def async_delegation_stub(monkeypatch):
    """Stub a RECENT hermes: ``interrupt_for_session`` plus the records
    registry (present in every version that has background delegation).
    Returns the module."""
    import threading

    mod = types.ModuleType("tools.async_delegation")
    mod.interrupt_for_session = MagicMock(return_value=1)
    mod._records_lock = threading.Lock()
    mod._records = _session_records()
    _install_async_delegation(monkeypatch, mod)
    return mod


@pytest.fixture
def legacy_async_delegation_stub(monkeypatch):
    """Stub an OLDER hermes: no ``interrupt_for_session``, records registry only.

    Returns the records dict so tests can assert which interrupt_fns fired.
    """
    import threading

    mod = types.ModuleType("tools.async_delegation")
    mod._records_lock = threading.Lock()
    mod._records = _session_records()
    _install_async_delegation(monkeypatch, mod)
    return mod._records


def _internal_message(chat_id: str, text: str):
    from gateway.platforms.base import MessageEvent
    from gateway.session import SessionSource

    return MessageEvent(
        text=text,
        source=SessionSource(platform=None, chat_id=chat_id, chat_type="dm", chat_name=chat_id),
        internal=True,
        message_id=None,
    )


# ---------------------------------------------------------------------------
# _handle_cancel — background delegation interrupt
# ---------------------------------------------------------------------------


def test_cancel_interrupts_background_delegations_with_gateway_key(async_delegation_stub):
    adapter, _ = _make_adapter()

    asyncio.run(adapter._handle_cancel(_cancel_event()))

    async_delegation_stub.interrupt_for_session.assert_called_once()
    kwargs = async_delegation_stub.interrupt_for_session.call_args.kwargs
    # Gateway key shape (conftest stub mirrors the real build_session_key DM
    # rule): agent:main:<platform>:dm:<conv> — NOT the adapter-internal key.
    assert kwargs["session_key"] == "agent:main:domyn:dm:conv-1"
    assert kwargs["reason"] == "user_cancelled"


def test_cancel_interrupts_via_records_on_legacy_hermes(legacy_async_delegation_stub):
    """Deployed hermes without interrupt_for_session: fall back to filtering
    the records registry by gateway session key — running + matching records
    only, never a global interrupt_all sweep."""
    adapter, _ = _make_adapter()

    asyncio.run(adapter._handle_cancel(_cancel_event()))

    records = legacy_async_delegation_stub
    records["deleg_match"]["interrupt_fn"].assert_called_once()
    records["deleg_other_session"]["interrupt_fn"].assert_not_called()
    records["deleg_done"]["interrupt_fn"].assert_not_called()


def test_cancel_survives_bare_async_delegation_module(monkeypatch):
    """Module importable but with neither interrupt_for_session nor a records
    registry: no crash, cancel still acks, and the drop degrades to the
    conversation-level fallback."""
    mod = types.ModuleType("tools.async_delegation")
    _install_async_delegation(monkeypatch, mod)
    adapter, mock_client = _make_adapter()

    asyncio.run(adapter._handle_cancel(_cancel_event()))

    end_event: BaseEvent = mock_client.send_event.await_args.args[0]
    assert end_event.event_type == ExecutionEventType.AGENT_END
    assert "conv-1" in adapter._drop_async_completion_conversations
    assert adapter._drop_async_completion_ids == set()


def test_cancel_arms_completion_drop_by_delegation_id(async_delegation_stub):
    """ALL of the session's registered ids are armed — including the already
    completed one, whose completion may still be queued for re-injection —
    but never another conversation's. No conversation-level fallback needed."""
    adapter, _ = _make_adapter()

    asyncio.run(adapter._handle_cancel(_cancel_event()))

    assert adapter._drop_async_completion_ids == {"deleg_match", "deleg_done"}
    assert adapter._drop_async_completion_conversations == set()


def test_cancel_arms_completion_drop_by_id_on_legacy_hermes(legacy_async_delegation_stub):
    adapter, _ = _make_adapter()

    asyncio.run(adapter._handle_cancel(_cancel_event()))

    assert adapter._drop_async_completion_ids == {"deleg_match", "deleg_done"}
    assert adapter._drop_async_completion_conversations == set()


def test_cancel_clears_stale_background_delegation_counter(async_delegation_stub):
    """Cancelled delegations' completion re-entries are dropped BEFORE the
    decrement in handle_message, so nothing would ever drain the in-flight
    counter — the cancel must clear it wholesale or the NEXT turn's final
    answer goes out as a non-terminal RESPONSE with its turn kept alive
    forever."""
    adapter, _ = _make_adapter()
    adapter.mark_background_delegation("conv-1")
    adapter.mark_background_delegation("conv-1")  # counter, not flag: 2 in flight

    asyncio.run(adapter._handle_cancel(_cancel_event()))

    assert adapter._bg_delegation_by_session == {}


def test_cancel_survives_missing_async_delegation_module(monkeypatch):
    """Older hermes without tools.async_delegation: cancel must still ack,
    with the drop degraded to the conversation-level fallback."""
    monkeypatch.setitem(sys.modules, "tools", None)
    monkeypatch.setitem(sys.modules, "tools.async_delegation", None)
    adapter, mock_client = _make_adapter()

    asyncio.run(adapter._handle_cancel(_cancel_event()))

    # The canned AGENT_END ack still went out.
    end_event: BaseEvent = mock_client.send_event.await_args.args[0]
    assert end_event.event_type == ExecutionEventType.AGENT_END
    assert "conv-1" in adapter._drop_async_completion_conversations


def test_cancel_still_sends_ack_and_stop(async_delegation_stub):
    adapter, mock_client = _make_adapter()
    dispatched = []

    async def _capture(msg):
        dispatched.append(msg)

    adapter.handle_message = _capture  # bypass the override; capture dispatch

    asyncio.run(adapter._handle_cancel(_cancel_event()))

    end_event: BaseEvent = mock_client.send_event.await_args.args[0]
    assert end_event.event_type == ExecutionEventType.AGENT_END
    assert dispatched and dispatched[0].text == "/stop"


def test_cancel_ack_does_not_reuse_the_request_event_id(async_delegation_stub):
    """The cancel AGENT_END must carry its own event_id.

    The delegating transport records the ids it sends and drops anything replayed
    back with one of them, so an AGENT_END reusing the inbound AGENT_START's id is
    swallowed as an echo and the cancel never acks. The correlation back to the
    request rides in metadata instead.
    """
    adapter, mock_client = _make_adapter()
    turn = adapter._turn_by_session[_build_session_key("chan-1", "conv-1")]

    async def _noop(_msg):
        return None

    adapter.handle_message = _noop

    asyncio.run(adapter._handle_cancel(_cancel_event()))

    end_event: BaseEvent = mock_client.send_event.await_args.args[0]
    assert end_event.event_id
    assert end_event.event_id != turn.event_id
    assert end_event.metadata["request_id"] == turn.event_id
    # Everything else still anchors the ack to the cancelled turn.
    assert end_event.turn_id == turn.turn_id
    assert end_event.interaction_id == turn.interaction_id
    assert end_event.conversation_id == turn.conversation_id


# ---------------------------------------------------------------------------
# handle_message — dropping late completion re-injections
# ---------------------------------------------------------------------------


def _spy_base_handle_message(monkeypatch):
    """Replace the stub base's handle_message with a recorder (auto-restored)."""
    from gateway.platforms.base import BasePlatformAdapter

    forwarded = []

    async def _spy(self, msg):
        forwarded.append(msg)

    monkeypatch.setattr(BasePlatformAdapter, "handle_message", _spy)
    return forwarded


def test_handle_message_drops_armed_id_and_prunes_it(monkeypatch):
    adapter, _ = _make_adapter()
    adapter._drop_async_completion_ids.add("deleg_1")
    forwarded = _spy_base_handle_message(monkeypatch)

    msg = _internal_message("conv-1", "[ASYNC DELEGATION BATCH COMPLETE — deleg_1]\n...")
    asyncio.run(adapter.handle_message(msg))

    assert forwarded == []
    # The watcher re-injects each id exactly once — the mark self-cleans.
    assert adapter._drop_async_completion_ids == set()


def test_handle_message_passes_completion_for_unarmed_id(monkeypatch):
    """A NEW dispatch after the cancel gets a fresh id, so its completion
    passes through even in the same conversation — no re-arm needed."""
    adapter, _ = _make_adapter()
    adapter._drop_async_completion_ids.add("deleg_cancelled")
    forwarded = _spy_base_handle_message(monkeypatch)

    msg = _internal_message("conv-1", "[ASYNC DELEGATION COMPLETE — deleg_new]\n...")
    asyncio.run(adapter.handle_message(msg))

    assert len(forwarded) == 1
    assert adapter._drop_async_completion_ids == {"deleg_cancelled"}


def test_handle_message_drops_via_conversation_fallback(monkeypatch):
    """Registry unreadable at cancel time → the coarse conversation-level
    drop still swallows the completion, whatever its id."""
    adapter, _ = _make_adapter()
    adapter._drop_async_completion_conversations.add("conv-1")
    forwarded = _spy_base_handle_message(monkeypatch)

    msg = _internal_message("conv-1", "[ASYNC DELEGATION BATCH COMPLETE — deleg_1]\n...")
    asyncio.run(adapter.handle_message(msg))

    assert forwarded == []


def test_handle_message_passes_completion_for_other_conversation(monkeypatch):
    adapter, _ = _make_adapter()
    adapter._drop_async_completion_conversations.add("conv-OTHER")
    forwarded = _spy_base_handle_message(monkeypatch)

    msg = _internal_message("conv-1", "[ASYNC DELEGATION COMPLETE — deleg_1]\n...")
    asyncio.run(adapter.handle_message(msg))

    assert len(forwarded) == 1


def test_handle_message_passes_other_internal_messages(monkeypatch):
    """Only async-delegation completions are dropped — other internal
    notifications (e.g. background process watches) pass through."""
    adapter, _ = _make_adapter()
    adapter._drop_async_completion_ids.add("proc-1")
    adapter._drop_async_completion_conversations.add("conv-1")
    forwarded = _spy_base_handle_message(monkeypatch)

    msg = _internal_message("conv-1", "[IMPORTANT: Background process proc-1 matched ...]")
    asyncio.run(adapter.handle_message(msg))

    assert len(forwarded) == 1


def test_handle_message_passes_non_internal_messages(monkeypatch):
    adapter, _ = _make_adapter()
    adapter._drop_async_completion_ids.add("deleg_1")
    adapter._drop_async_completion_conversations.add("conv-1")
    forwarded = _spy_base_handle_message(monkeypatch)

    from gateway.platforms.base import MessageEvent
    from gateway.session import SessionSource

    msg = MessageEvent(
        text="[ASYNC DELEGATION COMPLETE — deleg_1] pasted by the user",
        source=SessionSource(platform=None, chat_id="conv-1", chat_type="dm", chat_name="conv-1"),
        message_id="m-1",
    )
    asyncio.run(adapter.handle_message(msg))

    assert len(forwarded) == 1


def test_new_background_dispatch_rearms_conversation_fallback():
    adapter, _ = _make_adapter()
    adapter._drop_async_completion_conversations.add("conv-1")

    adapter.mark_background_delegation("conv-1")

    assert "conv-1" not in adapter._drop_async_completion_conversations


def test_new_background_dispatch_keeps_id_drops_armed():
    """The stale-completion race: a new dispatch must NOT disarm the ids of
    the cancelled delegations — their late completions still get dropped."""
    adapter, _ = _make_adapter()
    adapter._drop_async_completion_ids.add("deleg_cancelled")

    adapter.mark_background_delegation("conv-1")

    assert adapter._drop_async_completion_ids == {"deleg_cancelled"}
