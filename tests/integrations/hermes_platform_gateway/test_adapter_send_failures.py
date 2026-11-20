"""A send that fails is named by its error's type above debug: the error's text, which can quote
what was being sent, stays at debug."""

import asyncio
import logging
from unittest.mock import AsyncMock

import pytest

from domyn.agents.core import BaseEvent, ExecutionEventType
from tests.integrations.hermes_platform_gateway.test_adapter_visibility_events import (
    CHAT_ID,
    _make_adapter,
)
from tests.logs import split_at_debug

_DETAIL = "text the error quotes"

_SENDS = {
    "slash_confirm": lambda adapter: adapter.send_slash_confirm(
        chat_id=CHAT_ID, title="t", message="m", session_key="s", confirm_id="c"
    ),
    "exec_approval": lambda adapter: adapter.send_exec_approval(
        chat_id=CHAT_ID, command="ls", session_key="s"
    ),
    "visibility_tool_start": lambda adapter: adapter.emit_visibility_tool_start(
        chat_id=CHAT_ID, tool_name="t", args={}, thought=None, call_id="c"
    ),
    "visibility_tool_end": lambda adapter: adapter.emit_visibility_tool_end(
        chat_id=CHAT_ID, tool_name="t", call_id="c", observation="o"
    ),
    "visibility_tool_error": lambda adapter: adapter.emit_visibility_tool_error(
        chat_id=CHAT_ID, tool_name="t", call_id="c", error_message="e"
    ),
    "llm_start": lambda adapter: adapter.emit_llm_start(chat_id=CHAT_ID, model="m", call_id="c"),
    "llm_end": lambda adapter: adapter.emit_llm_end(
        chat_id=CHAT_ID, model="m", call_id="c", assistant_response="a"
    ),
}


@pytest.mark.parametrize("send", list(_SENDS.values()), ids=list(_SENDS))
def test_a_failed_send_keeps_the_errors_text_at_debug(caplog, send) -> None:
    adapter, client = _make_adapter()
    client.send_event.side_effect = RuntimeError(_DETAIL)
    caplog.set_level(logging.DEBUG)

    asyncio.run(send(adapter))

    above, debug = split_at_debug(caplog)
    assert "RuntimeError" in above and _DETAIL not in above
    assert _DETAIL in debug


def test_a_failed_cancel_keeps_the_errors_text_at_debug(caplog) -> None:
    """Both halves of a cancel can fail: the ack to the platform and the /stop to hermes."""
    adapter, client = _make_adapter()
    client.send_event.side_effect = RuntimeError(_DETAIL)
    adapter.handle_message = AsyncMock(side_effect=RuntimeError(_DETAIL))
    caplog.set_level(logging.DEBUG)
    cancel = BaseEvent(
        event_type=ExecutionEventType.AGENT_END,
        author="u1",
        conversation_id=CHAT_ID,
        error_code="user_cancelled",
    )

    asyncio.run(adapter._handle_cancel(cancel))

    above, debug = split_at_debug(caplog)
    assert "cancel AGENT_END - RuntimeError" in above and "/stop dispatch failed" in above
    assert _DETAIL not in above and _DETAIL in debug
