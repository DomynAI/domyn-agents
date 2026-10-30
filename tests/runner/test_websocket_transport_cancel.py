"""WebSocketTransport.cancel() tells the remote agent to stop, against a real server."""

import asyncio

import pytest
from websockets.asyncio.server import serve

from domyn.agents import BaseEvent, ExecutionEventType, Part
from domyn.agents.core import InteractionContext
from domyn.agents.runner.websocket_transport import WebSocketTransport


def _context(conversation_id: str = "conv-1") -> InteractionContext:
    user_input = BaseEvent(
        event_type=ExecutionEventType.USER_INPUT,
        author="user",
        content=[Part(text="count to a million")],
        turn_id="turn-1",
    )
    return InteractionContext(
        conversation_id=conversation_id,
        user_id="user",
        agent_input=user_input,
        interaction_id="int-1",
        turn_id="turn-1",
    )


class _Relay:
    """Records what every connection sends; answers the first with `reply`."""

    def __init__(self, reply: BaseEvent):
        self.reply = reply
        self.received: list[tuple[str | None, BaseEvent]] = []
        self.connections = 0

    async def handler(self, ws):
        self.connections += 1
        first = self.connections == 1
        async for raw in ws:
            self.received.append(
                (ws.request.headers.get("conversation-id"), BaseEvent.model_validate_json(raw))
            )
            if first:
                await ws.send(self.reply.model_dump_json())


def _partial(text: str) -> BaseEvent:
    return BaseEvent(
        event_type=ExecutionEventType.RESPONSE,
        author="remote",
        content=[Part(text=text)],
        is_partial=True,
        turn_id="turn-1",
    )


async def _run_until_first_event(transport: WebSocketTransport, context: InteractionContext):
    events = transport.execute("Counter", context)
    first = await anext(events)
    # What DelegateAgent's task being cancelled does to the stream.
    await events.aclose()
    return first


@pytest.mark.asyncio
async def test_cancel_sends_user_cancelled_for_a_run_in_progress():
    relay = _Relay(reply=_partial("1, 2, 3"))
    async with serve(relay.handler, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        transport = WebSocketTransport(ws_url=f"ws://127.0.0.1:{port}", timeout=5)

        await _run_until_first_event(transport, _context())
        await transport.cancel("conv-1")
        await asyncio.sleep(0.1)

    assert relay.connections == 2
    conversation, signal = relay.received[-1]
    assert conversation == "conv-1"
    assert signal.event_type == ExecutionEventType.AGENT_END
    assert signal.error_code == "user_cancelled"
    assert signal.author == "Counter"
    assert signal.conversation_id == "conv-1"


@pytest.mark.asyncio
async def test_cancel_sends_nothing_for_a_conversation_it_never_started():
    relay = _Relay(reply=_partial("1"))
    async with serve(relay.handler, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        transport = WebSocketTransport(ws_url=f"ws://127.0.0.1:{port}", timeout=5)

        await _run_until_first_event(transport, _context("conv-1"))
        await transport.cancel("conv-2")
        await asyncio.sleep(0.1)

    assert relay.connections == 1


@pytest.mark.asyncio
async def test_cancel_sends_nothing_once_the_run_has_ended():
    final = BaseEvent(
        event_type=ExecutionEventType.RESPONSE,
        author="remote",
        content=[Part(text="done")],
        turn_id="turn-1",
    )
    relay = _Relay(reply=final)
    async with serve(relay.handler, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        transport = WebSocketTransport(ws_url=f"ws://127.0.0.1:{port}", timeout=5)

        events = [e async for e in transport.execute("Counter", _context())]
        await transport.cancel("conv-1")
        await asyncio.sleep(0.1)

    assert [e.event_type for e in events] == [ExecutionEventType.RESPONSE]
    assert relay.connections == 1


@pytest.mark.asyncio
async def test_a_relay_that_cannot_be_reached_is_only_logged():
    transport = WebSocketTransport(ws_url="ws://127.0.0.1:9", timeout=1)
    transport._running["conv-1"] = "Counter"

    await transport.cancel("conv-1")  # must not raise
