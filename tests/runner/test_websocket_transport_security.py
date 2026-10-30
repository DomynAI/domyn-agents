"""Tests for WebSocketTransport security - cross-conversation isolation.

These tests demonstrate that the fix prevents cross-conversation data leakage
where concurrent conversations sharing the same WebSocketTransport instance
could leak events across conversation boundaries.
"""

from collections.abc import AsyncGenerator
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from domyn.agents.core import AgentAction, BaseEvent, ExecutionEventType, InteractionContext, Part
from domyn.agents.runner.websocket_transport import WebSocketTransport


def create_test_context(
    conversation_id: str = "test-conversation",
    user_text: str = "test task",
) -> InteractionContext:
    """Create a test InteractionContext."""
    from uuid import uuid4

    user_input = BaseEvent(
        event_type=ExecutionEventType.USER_INPUT,
        author="user",
        content=[Part(text=user_text)],
        interaction_id=conversation_id,
        turn_id="turn-1",
    )

    return InteractionContext(
        conversation_id=conversation_id,
        thread_id=f"thread-{uuid4()}",
        user_id="test-user",
        user_input=user_input,
        agent_input=user_input,
        interaction_id=conversation_id,
        turn_id="turn-1",
    )


def create_tool_end_event(
    conversation_id: str,
    result: str = "test result",
) -> BaseEvent:
    """Create a TOOL_END event."""
    return BaseEvent(
        event_type=ExecutionEventType.TOOL_END,
        author="test_agent",
        content=[Part(text=result)],
        action=AgentAction(name="test_tool", parameters={"result": result}),
        interaction_id=conversation_id,
        turn_id="turn-1",
    )


class TestCrossConversationIsolation:
    """Tests for cross-conversation event isolation security."""

    @pytest.mark.asyncio
    async def test_concurrent_conversations_no_cross_leakage(self) -> None:
        """Test that A's TOOL_END is not sent on B's connection.

        This is the core security test demonstrating the fix for the
        cross-conversation leakage vulnerability:

        1. Conversation A starts and receives TOOL_START (pauses)
        2. A's TOOL_END is queued via send_result()
        3. Conversation B starts before A resumes
        4. B's connection must send AGENT_START for B, NOT A's TOOL_END
        """
        transport = WebSocketTransport(ws_url="ws://test")

        ctx_a = create_test_context(conversation_id="conversation-A", user_text="task A")
        ctx_b = create_test_context(conversation_id="conversation-B", user_text="task B")

        # Track what each connection sends independently
        conn_sends: dict[int, list[BaseEvent]] = {1: [], 2: []}

        async def mock_relay_a(ws: Any, agent_name: str) -> AsyncGenerator[BaseEvent, None]:
            """Connection 1: yields TOOL_START to pause A mid-session."""
            yield BaseEvent(event_type=ExecutionEventType.TOOL_START, author="remote")

        async def mock_relay_b(ws: Any, agent_name: str) -> AsyncGenerator[BaseEvent, None]:
            """Connection 2: yields AGENT_END to complete B."""
            yield BaseEvent(
                event_type=ExecutionEventType.AGENT_END,
                author="remote",
                content=[Part(text="done")],
            )

        relay_fns = [mock_relay_a, mock_relay_b]
        relay_idx = {"i": 0}

        def relay_dispatcher(ws: Any, agent_name: str) -> AsyncGenerator[BaseEvent, None]:
            fn = relay_fns[relay_idx["i"]]
            relay_idx["i"] += 1
            return fn(ws, agent_name)

        def make_ws(conn_id: int) -> AsyncMock:
            ws = AsyncMock()
            ws.send = AsyncMock(
                side_effect=lambda msg: conn_sends[conn_id].append(
                    BaseEvent.model_validate_json(msg)
                )
            )
            ws.__aenter__ = AsyncMock(return_value=ws)
            ws.__aexit__ = AsyncMock(return_value=False)
            return ws

        ws1, ws2 = make_ws(1), make_ws(2)

        with (
            patch.object(transport, "_relay", relay_dispatcher),
            patch("websockets.asyncio.client.connect", side_effect=[ws1, ws2]),
        ):
            # A starts and pauses at TOOL_START
            gen_a = transport.execute("delegate", ctx_a)
            event_a = await anext(gen_a)
            assert event_a.event_type == ExecutionEventType.TOOL_START

            # Queue TOOL_END for A with sensitive data
            await transport.send_result(
                create_tool_end_event("conversation-A", "SECRET_FROM_A"),
                "conversation-A",
            )

            # B starts before A resumes - this is where the old code was vulnerable
            gen_b = transport.execute("delegate", ctx_b)
            event_b = await anext(gen_b)
            assert event_b.event_type == ExecutionEventType.AGENT_END

            await gen_a.aclose()
            await gen_b.aclose()

        # SECURITY ASSERTION: conn 1 (A) must have sent AGENT_START for A
        assert len(conn_sends[1]) == 1
        assert conn_sends[1][0].event_type == ExecutionEventType.AGENT_START
        assert conn_sends[1][0].interaction_id == "conversation-A"

        # SECURITY ASSERTION: conn 2 (B) must have sent AGENT_START for B,
        # NOT A's TOOL_END with SECRET_FROM_A
        assert len(conn_sends[2]) == 1
        first_on_conn2 = conn_sends[2][0]
        assert first_on_conn2.event_type == ExecutionEventType.AGENT_START, (
            f"SECURITY VULNERABILITY: B's connection sent {first_on_conn2.event_type} "
            f"instead of AGENT_START. This means A's TOOL_END leaked into B!"
        )
        assert first_on_conn2.interaction_id == "conversation-B", (
            f"SECURITY VULNERABILITY: B's connection sent event for "
            f"'{first_on_conn2.interaction_id}' instead of 'conversation-B'"
        )

    @pytest.mark.asyncio
    async def test_pending_queue_isolation(self) -> None:
        """Test that pending queues are isolated per conversation.

        This directly tests the _pending_queues structure to ensure
        events are properly isolated by conversation_id.
        """
        transport = WebSocketTransport(ws_url="ws://test")

        # Queue events for both conversations
        tool_end_a = create_tool_end_event(
            conversation_id="conversation-A",
            result="SECRET_A",
        )
        tool_end_b = create_tool_end_event(
            conversation_id="conversation-B",
            result="SECRET_B",
        )

        await transport.send_result(tool_end_a, "conversation-A")
        await transport.send_result(tool_end_b, "conversation-B")

        # Verify queues are isolated
        assert "conversation-A" in transport._pending_queues
        assert "conversation-B" in transport._pending_queues

        # Get events from queues
        queue_a = transport._pending_queues["conversation-A"]
        queue_b = transport._pending_queues["conversation-B"]

        event_from_a_queue = await queue_a.get()
        event_from_b_queue = await queue_b.get()

        # Verify isolation
        assert event_from_a_queue.interaction_id == "conversation-A"
        assert event_from_b_queue.interaction_id == "conversation-B"

        # Verify content isolation - CRITICAL SECURITY CHECK
        assert "SECRET_A" in str(event_from_a_queue.content)
        assert "SECRET_B" in str(event_from_b_queue.content)
        assert "SECRET_A" not in str(event_from_b_queue.content), (
            "SECURITY VULNERABILITY: B's queue contains A's secret!"
        )
        assert "SECRET_B" not in str(event_from_a_queue.content), (
            "SECURITY VULNERABILITY: A's queue contains B's secret!"
        )

    @pytest.mark.asyncio
    async def test_sequential_conversations_isolation(self) -> None:
        """Test that sequential conversations are properly isolated."""
        transport = WebSocketTransport(ws_url="ws://test")
        sent_events: list[BaseEvent] = []

        ctx_a = create_test_context(
            conversation_id="conversation-A",
            user_text="task A",
        )
        ctx_b = create_test_context(
            conversation_id="conversation-B",
            user_text="task B",
        )

        async def mock_relay_immediate_end(
            ws: Any, agent_name: str
        ) -> AsyncGenerator[BaseEvent, None]:
            yield BaseEvent(
                event_type=ExecutionEventType.AGENT_END,
                author="remote",
                content=[Part(text="done")],
            )

        with (
            patch.object(transport, "_relay", mock_relay_immediate_end),
            patch("websockets.asyncio.client.connect") as mock_connect,
        ):
            mock_ws = AsyncMock()
            mock_ws.send = AsyncMock(
                side_effect=lambda msg: sent_events.append(BaseEvent.model_validate_json(msg))
            )
            mock_ws.__aenter__ = AsyncMock(return_value=mock_ws)
            mock_ws.__aexit__ = AsyncMock(return_value=False)
            mock_connect.return_value = mock_ws

            # Complete conversation A
            gen_a = transport.execute("delegate", ctx_a)
            async for _ in gen_a:
                pass
            await gen_a.aclose()

            # Clear sent events to check B separately
            sent_events.clear()

            # Then start conversation B
            gen_b = transport.execute("delegate", ctx_b)
            async for _ in gen_b:
                pass
            await gen_b.aclose()

            # B should only send B's events
            for event in sent_events:
                assert event.interaction_id == "conversation-B"


class TestWebSocketDedup:
    """The WS transport dedups replayed events by event_id (shared base logic)."""

    @pytest.mark.asyncio
    async def test_dedups_duplicate_event_ids(self) -> None:
        transport = WebSocketTransport(ws_url="ws://test")
        ctx = create_test_context(conversation_id="conv-dedup")

        async def mock_relay_with_dup(ws: Any, agent_name: str) -> AsyncGenerator[BaseEvent, None]:
            yield BaseEvent(
                event_type=ExecutionEventType.RESPONSE,
                author="remote",
                event_id="e1",
                is_partial=True,
                content=[Part(text="a")],
            )
            # Duplicate replay of e1 — must be dropped.
            yield BaseEvent(
                event_type=ExecutionEventType.RESPONSE,
                author="remote",
                event_id="e1",
                is_partial=True,
                content=[Part(text="a")],
            )
            yield BaseEvent(
                event_type=ExecutionEventType.RESPONSE,
                author="remote",
                event_id="e2",
                is_partial=False,
                content=[Part(text="done")],
            )

        with (
            patch.object(transport, "_relay", mock_relay_with_dup),
            patch("websockets.asyncio.client.connect") as mock_connect,
        ):
            mock_ws = AsyncMock()
            mock_ws.send = AsyncMock()
            mock_ws.__aenter__ = AsyncMock(return_value=mock_ws)
            mock_ws.__aexit__ = AsyncMock(return_value=False)
            mock_connect.return_value = mock_ws

            got = [ev async for ev in transport.execute("delegate", ctx)]

        assert [e.event_id for e in got] == ["e1", "e2"]


class TestWebSocketReplayParity:
    """WS parity with SSE: send conversation-id + last-event-id on the handshake so
    the relay replays forward events on reconnect, and dedup those replays."""

    @staticmethod
    def _mk_conn(call_headers: list[dict]) -> Any:
        def make_conn(_url: str, **kwargs: Any) -> Any:
            call_headers.append(kwargs.get("additional_headers", {}))
            ws = AsyncMock()
            ws.send = AsyncMock()
            ws.__aenter__ = AsyncMock(return_value=ws)
            ws.__aexit__ = AsyncMock(return_value=False)
            return ws

        return make_conn

    @pytest.mark.asyncio
    async def test_sends_conversation_and_cursor_headers(self) -> None:
        transport = WebSocketTransport(ws_url="ws://test", headers={"api-key": "k"})
        ctx = create_test_context(conversation_id="conv-hdr")

        tool_start = BaseEvent(
            event_type=ExecutionEventType.TOOL_START,
            author="remote",
            event_id="evt-ts",
            action=AgentAction(name="t", parameters={}),
        )
        final = BaseEvent(
            event_type=ExecutionEventType.RESPONSE,
            author="remote",
            event_id="evt-final",
            is_partial=False,
            content=[Part(text="done")],
        )
        batches = [[tool_start], [final]]
        idx = {"i": 0}

        async def mock_relay(ws: Any, agent_name: str) -> AsyncGenerator[BaseEvent, None]:
            batch = batches[idx["i"]]
            idx["i"] += 1
            for ev in batch:
                yield ev

        call_headers: list[dict] = []
        with (
            patch.object(transport, "_relay", mock_relay),
            patch("websockets.asyncio.client.connect", side_effect=self._mk_conn(call_headers)),
        ):
            _ = [ev async for ev in transport.execute("delegate", ctx)]

        # First connect: conversation-id present, no cursor yet.
        assert call_headers[0].get("conversation-id") == "conv-hdr"
        assert "last-event-id" not in call_headers[0]
        # Reconnect after the TOOL_START boundary: cursor = last seen event id.
        assert call_headers[1]["conversation-id"] == "conv-hdr"
        assert call_headers[1]["last-event-id"] == "evt-ts"

    @pytest.mark.asyncio
    async def test_dedups_replayed_events_across_reconnect(self) -> None:
        transport = WebSocketTransport(ws_url="ws://test")
        ctx = create_test_context(conversation_id="conv-replay")

        tool_start = BaseEvent(
            event_type=ExecutionEventType.TOOL_START,
            author="remote",
            event_id="evt-ts",
            action=AgentAction(name="t", parameters={}),
        )
        # On reconnect the relay replays evt-ts (already seen) plus a fresh final.
        replay = BaseEvent(
            event_type=ExecutionEventType.TOOL_START,
            author="remote",
            event_id="evt-ts",
            action=AgentAction(name="t", parameters={}),
        )
        final = BaseEvent(
            event_type=ExecutionEventType.RESPONSE,
            author="remote",
            event_id="evt-final",
            is_partial=False,
            content=[Part(text="done")],
        )
        batches = [[tool_start], [replay, final]]
        idx = {"i": 0}

        async def mock_relay(ws: Any, agent_name: str) -> AsyncGenerator[BaseEvent, None]:
            batch = batches[idx["i"]]
            idx["i"] += 1
            for ev in batch:
                yield ev

        with (
            patch.object(transport, "_relay", mock_relay),
            patch("websockets.asyncio.client.connect", side_effect=self._mk_conn([])),
        ):
            got = [ev async for ev in transport.execute("delegate", ctx)]

        assert [e.event_id for e in got] == ["evt-ts", "evt-final"]


class TestWebSocketTurnFilter:
    """Events belonging to another turn are dropped (cross-turn replay).

    The relay's retained log is keyed by channel and replayed in full on a connect
    without a resume cursor. Since the transport is rebuilt per turn, its first
    connect never carries one, so a new turn's stream can open with the previous
    turn's events; they are identified by the turn_id every producer stamps.
    """

    @staticmethod
    def _mk_conn() -> Any:
        def make_conn(_url: str, **_kwargs: Any) -> Any:
            ws = AsyncMock()
            ws.send = AsyncMock()
            ws.__aenter__ = AsyncMock(return_value=ws)
            ws.__aexit__ = AsyncMock(return_value=False)
            return ws

        return make_conn

    async def _run(self, events: list[BaseEvent], ctx: InteractionContext) -> list[str]:
        transport = WebSocketTransport(ws_url="ws://test")

        async def mock_relay(ws: Any, agent_name: str) -> AsyncGenerator[BaseEvent, None]:
            for ev in events:
                yield ev

        with (
            patch.object(transport, "_relay", mock_relay),
            patch("websockets.asyncio.client.connect", side_effect=self._mk_conn()),
        ):
            got = [ev async for ev in transport.execute("delegate", ctx)]
        return [e.event_id for e in got]

    @staticmethod
    def _evt(
        event_id: str,
        *,
        turn_id: str | None,
        event_type: ExecutionEventType = ExecutionEventType.RESPONSE,
        is_partial: bool = True,
    ) -> BaseEvent:
        return BaseEvent(
            event_type=event_type,
            author="remote",
            event_id=event_id,
            turn_id=turn_id,
            is_partial=is_partial,
            content=[Part(text="x")],
        )

    @pytest.mark.asyncio
    async def test_drops_events_from_another_turn(self) -> None:
        ctx = create_test_context(conversation_id="conv-turn")  # turn_id == "turn-1"
        got = await self._run(
            [
                self._evt("old-1", turn_id="turn-0"),
                self._evt("new-1", turn_id="turn-1"),
                self._evt("new-2", turn_id="turn-1", is_partial=False),
            ],
            ctx,
        )
        assert got == ["new-1", "new-2"]

    @pytest.mark.asyncio
    async def test_drops_terminal_event_from_another_turn(self) -> None:
        """A stale AGENT_END must not end the current turn's session early."""
        ctx = create_test_context(conversation_id="conv-turn-end")
        got = await self._run(
            [
                self._evt("old-end", turn_id="turn-0", event_type=ExecutionEventType.AGENT_END),
                self._evt("new-1", turn_id="turn-1"),
                self._evt("new-2", turn_id="turn-1", is_partial=False),
            ],
            ctx,
        )
        assert got == ["new-1", "new-2"]

    @pytest.mark.asyncio
    async def test_drops_foreign_agent_call_before_nesting_is_counted(self) -> None:
        """A replayed AGENT_CALL from another turn must not skew nesting depth.

        If counted, the current turn's final RESPONSE would be treated as a nested
        sub-agent result and suppressed instead of ending the session.
        """
        ctx = create_test_context(conversation_id="conv-turn-nest")
        got = await self._run(
            [
                self._evt("old-call", turn_id="turn-0", event_type=ExecutionEventType.AGENT_CALL),
                self._evt("new-final", turn_id="turn-1", is_partial=False),
            ],
            ctx,
        )
        assert got == ["new-final"]

    @pytest.mark.asyncio
    async def test_delivers_events_without_turn_id(self) -> None:
        """Fail-open: producers that don't stamp turn_id are unaffected."""
        ctx = create_test_context(conversation_id="conv-turn-anon")
        got = await self._run(
            [
                self._evt("anon-1", turn_id=None),
                self._evt("anon-2", turn_id=None, is_partial=False),
            ],
            ctx,
        )
        assert got == ["anon-1", "anon-2"]

    @pytest.mark.asyncio
    async def test_still_dedups_within_the_current_turn(self) -> None:
        """Turn filter and event_id seen-set are complementary, not redundant."""
        ctx = create_test_context(conversation_id="conv-turn-dedup")
        got = await self._run(
            [
                self._evt("e1", turn_id="turn-1"),
                self._evt("e1", turn_id="turn-1"),
                self._evt("e2", turn_id="turn-1", is_partial=False),
            ],
            ctx,
        )
        assert got == ["e1", "e2"]
