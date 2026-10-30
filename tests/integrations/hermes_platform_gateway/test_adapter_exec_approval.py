"""Tests for send_exec_approval choice rendering and gateway compatibility.

The gateway calls send_exec_approval with keyword-only ``allow_permanent``
and ``smart_denied`` args. The adapter must accept them (otherwise the gateway's
``getattr(type(adapter), "send_exec_approval")`` call raises TypeError and it
falls back to plain text: "Button-based approval failed, falling back to text")
and translate them into the same set of choices as the gateway's plain-text
fallback (``_format_exec_approval_fallback``).
"""

import asyncio
import inspect
import sys
import types
from unittest.mock import AsyncMock, MagicMock

import pytest

from domyn.agents.core import (
    AgentAction,
    ApprovalType,
    BaseEvent,
    ExecutionEventType,
    FeedbackAction,
    FeedbackParameters,
)
from domyn.agents.integrations.hermes_platform_gateway.adapter import (
    DomynPlatformAdapter,
    _build_session_key,
    _derive_approval_choice,
)


def _make_adapter(channel_id: str = "chan-1", chat_id: str = "conv-1"):
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
        conversation_id=chat_id,
        interaction_id=f"int-{chat_id}",
        turn_id=f"turn-{chat_id}",
        action=AgentAction(name="invoke", parameters={"input": "hi"}),
    )
    adapter._turn_by_session[_build_session_key(channel_id, chat_id)] = turn
    adapter._client = mock_client
    return adapter, mock_client


# ---------------------------------------------------------------------------
# Signature compatibility with the gateway call site
# ---------------------------------------------------------------------------


def test_signature_accepts_gateway_kwargs():
    """The exact kwargs the gateway passes must be accepted without TypeError."""
    sig = inspect.signature(DomynPlatformAdapter.send_exec_approval)
    sig.bind(
        MagicMock(),  # self
        chat_id="conv-1",
        command="rm -rf /",
        session_key="sess-1",
        description="dangerous command",
        metadata={},
        allow_permanent=True,
        smart_denied=False,
    )


# ---------------------------------------------------------------------------
# Choice rendering mirrors _format_exec_approval_fallback
# ---------------------------------------------------------------------------


def test_default_offers_all_choices():
    adapter, _ = _make_adapter()

    result = asyncio.run(
        adapter.send_exec_approval(chat_id="conv-1", command="ls", session_key="s")
    )

    assert result.success is True


def test_no_active_turn_returns_failure():
    adapter, mock_client = _make_adapter()

    result = asyncio.run(
        adapter.send_exec_approval(chat_id="unknown", command="ls", session_key="s")
    )

    assert result.success is False
    mock_client.send_event.assert_not_awaited()


# ---------------------------------------------------------------------------
# _derive_approval_choice — remember_scope takes priority, legacy param falls back
# ---------------------------------------------------------------------------


def test_derive_choice_returns_denied_regardless_of_scope():
    action = FeedbackAction(
        name="exec_approval",
        approval_type=ApprovalType.USER_CONFIRM,
        approve=False,
        remember_scope="conversation",
    )
    assert (
        _derive_approval_choice(
            action, denied="deny", scope_choices={"conversation": "session"}, valid_legacy=("once",)
        )
        == "deny"
    )


def test_derive_choice_prefers_remember_scope_over_legacy_param():
    action = FeedbackAction(
        name="exec_approval",
        approval_type=ApprovalType.USER_CONFIRM,
        approve=True,
        remember_scope="conversation",
        parameters=[FeedbackParameters(name="choice", value="always", type="str")],
    )
    assert (
        _derive_approval_choice(
            action,
            denied="deny",
            scope_choices={"conversation": "session", "user": "always"},
            valid_legacy=("once", "session", "always"),
        )
        == "session"
    )


def test_derive_choice_once_wins_over_a_conflicting_legacy_param():
    """An explicit ``remember_scope="once"`` must be authoritative, the same
    way every other recognised scope is — a stale legacy ``choice`` param
    must not be able to widen it behind the user's back."""
    action = FeedbackAction(
        name="exec_approval",
        approval_type=ApprovalType.USER_CONFIRM,
        approve=True,
        remember_scope="once",
        parameters=[FeedbackParameters(name="choice", value="always", type="str")],
    )
    assert (
        _derive_approval_choice(
            action,
            denied="deny",
            scope_choices={"once": "once", "conversation": "session"},
            valid_legacy=("once", "session", "always"),
        )
        == "once"
    )


def test_derive_choice_falls_back_to_legacy_param_when_scope_unrecognised():
    action = FeedbackAction(
        name="exec_approval",
        approval_type=ApprovalType.USER_CONFIRM,
        approve=True,
        remember_scope=None,
        parameters=[FeedbackParameters(name="choice", value="always", type="str")],
    )
    assert (
        _derive_approval_choice(
            action,
            denied="deny",
            scope_choices={"conversation": "session"},
            valid_legacy=("once", "always"),
        )
        == "always"
    )


def test_derive_choice_defaults_to_once():
    action = FeedbackAction(
        name="exec_approval", approval_type=ApprovalType.USER_CONFIRM, approve=True
    )
    assert (
        _derive_approval_choice(
            action,
            denied="deny",
            scope_choices={"conversation": "session"},
            valid_legacy=("always",),
        )
        == "once"
    )


# ---------------------------------------------------------------------------
# Inbound feedback handlers — remember_scope reaches the external resolver
# ---------------------------------------------------------------------------


def _fake_tools_approval_module(monkeypatch: pytest.MonkeyPatch, resolve_mock: MagicMock) -> None:
    tools_pkg = sys.modules.get("tools") or types.ModuleType("tools")
    approval_mod = types.ModuleType("tools.approval")
    approval_mod.resolve_gateway_approval = resolve_mock
    tools_pkg.approval = approval_mod
    monkeypatch.setitem(sys.modules, "tools", tools_pkg)
    monkeypatch.setitem(sys.modules, "tools.approval", approval_mod)


def _fake_tools_slash_confirm_module(
    monkeypatch: pytest.MonkeyPatch, resolve_mock: AsyncMock
) -> None:
    tools_pkg = sys.modules.get("tools") or types.ModuleType("tools")
    slash_mod = types.ModuleType("tools.slash_confirm")
    slash_mod.resolve = resolve_mock
    tools_pkg.slash_confirm = slash_mod
    monkeypatch.setitem(sys.modules, "tools", tools_pkg)
    monkeypatch.setitem(sys.modules, "tools.slash_confirm", slash_mod)


def _feedback_event(*, turn_id: str, action: FeedbackAction) -> BaseEvent:
    return BaseEvent(
        event_type=ExecutionEventType.TOOL_FEEDBACK,
        author="u1",
        conversation_id="conv-1",
        interaction_id="int-conv-1",
        turn_id=turn_id,
        action=action,
    )


def test_exec_approval_feedback_maps_conversation_scope_to_session(monkeypatch: pytest.MonkeyPatch):
    adapter, _ = _make_adapter()
    turn_id = "turn-conv-1"
    adapter._pending_exec_approvals[turn_id] = ("conv-1", "session-key-1")
    resolve_mock = MagicMock(return_value=1)
    _fake_tools_approval_module(monkeypatch, resolve_mock)

    action = FeedbackAction(
        name="exec_approval",
        approval_type=ApprovalType.USER_CONFIRM,
        approve=True,
        remember_scope="conversation",
    )
    matched = asyncio.run(
        adapter._handle_exec_approval_feedback(_feedback_event(turn_id=turn_id, action=action))
    )

    assert matched is True
    resolve_mock.assert_called_once_with("session-key-1", "session")


@pytest.mark.parametrize("scope", ["user", "project", "global"])
def test_exec_approval_feedback_maps_wider_scopes_to_always(
    monkeypatch: pytest.MonkeyPatch, scope: str
):
    adapter, _ = _make_adapter()
    turn_id = "turn-conv-1"
    adapter._pending_exec_approvals[turn_id] = ("conv-1", "session-key-1")
    resolve_mock = MagicMock(return_value=1)
    _fake_tools_approval_module(monkeypatch, resolve_mock)

    action = FeedbackAction(
        name="exec_approval",
        approval_type=ApprovalType.USER_CONFIRM,
        approve=True,
        remember_scope=scope,
    )
    asyncio.run(
        adapter._handle_exec_approval_feedback(_feedback_event(turn_id=turn_id, action=action))
    )

    resolve_mock.assert_called_once_with("session-key-1", "always")


def test_exec_approval_feedback_once_wins_over_a_conflicting_legacy_param(
    monkeypatch: pytest.MonkeyPatch,
):
    adapter, _ = _make_adapter()
    turn_id = "turn-conv-1"
    adapter._pending_exec_approvals[turn_id] = ("conv-1", "session-key-1")
    resolve_mock = MagicMock(return_value=1)
    _fake_tools_approval_module(monkeypatch, resolve_mock)

    action = FeedbackAction(
        name="exec_approval",
        approval_type=ApprovalType.USER_CONFIRM,
        approve=True,
        remember_scope="once",
        parameters=[FeedbackParameters(name="choice", value="always", type="str")],
    )
    asyncio.run(
        adapter._handle_exec_approval_feedback(_feedback_event(turn_id=turn_id, action=action))
    )

    resolve_mock.assert_called_once_with("session-key-1", "once")


def test_exec_approval_feedback_falls_back_to_legacy_choice_param(monkeypatch: pytest.MonkeyPatch):
    adapter, _ = _make_adapter()
    turn_id = "turn-conv-1"
    adapter._pending_exec_approvals[turn_id] = ("conv-1", "session-key-1")
    resolve_mock = MagicMock(return_value=1)
    _fake_tools_approval_module(monkeypatch, resolve_mock)

    action = FeedbackAction(
        name="exec_approval",
        approval_type=ApprovalType.USER_CONFIRM,
        approve=True,
        parameters=[FeedbackParameters(name="choice", value="session", type="str")],
    )
    asyncio.run(
        adapter._handle_exec_approval_feedback(_feedback_event(turn_id=turn_id, action=action))
    )

    resolve_mock.assert_called_once_with("session-key-1", "session")


def test_slash_confirm_feedback_maps_conversation_scope_to_always(monkeypatch: pytest.MonkeyPatch):
    adapter, _ = _make_adapter()
    turn_id = "turn-conv-1"
    adapter._pending_slash_confirms[turn_id] = ("conv-1", "session-key-1")
    resolve_mock = AsyncMock(return_value="")
    _fake_tools_slash_confirm_module(monkeypatch, resolve_mock)

    action = FeedbackAction(
        name="slash_confirm",
        approval_type=ApprovalType.USER_CONFIRM,
        approve=True,
        remember_scope="conversation",
    )
    asyncio.run(
        adapter._handle_slash_confirm_feedback(_feedback_event(turn_id=turn_id, action=action))
    )

    resolve_mock.assert_called_once_with("session-key-1", turn_id, "always")


def test_slash_confirm_feedback_denied_maps_to_cancel(monkeypatch: pytest.MonkeyPatch):
    adapter, _ = _make_adapter()
    turn_id = "turn-conv-1"
    adapter._pending_slash_confirms[turn_id] = ("conv-1", "session-key-1")
    resolve_mock = AsyncMock(return_value="")
    _fake_tools_slash_confirm_module(monkeypatch, resolve_mock)

    action = FeedbackAction(
        name="slash_confirm", approval_type=ApprovalType.USER_CONFIRM, approve=False
    )
    asyncio.run(
        adapter._handle_slash_confirm_feedback(_feedback_event(turn_id=turn_id, action=action))
    )

    resolve_mock.assert_called_once_with("session-key-1", turn_id, "cancel")
