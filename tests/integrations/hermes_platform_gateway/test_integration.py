"""Integration tests against a real in-process HTTP + WebSocket stub.

Each test spins up a StubServer on a free port, exercises the actual
network stack of ``DomynRelayClient`` (async WS for ``send_event`` /
``on_event``), and tears the server down.
"""

import asyncio
import contextlib
import socket

import pytest
import pytest_asyncio
from aiohttp import WSMsgType, web

from domyn.agents.core import (
    AgentAction,
    BaseEvent,
    ExecutionEventType,
    RelayMessage,
    ToolAction,
)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class StubServer:
    """Minimal platform relay stub: a WebSocket handler."""

    def __init__(self, port: int) -> None:
        self.port = port
        self._ws_clients: list[web.WebSocketResponse] = []
        self._runner: web.AppRunner | None = None
        self.received_events: list[BaseEvent] = []
        self.last_ws_headers: dict[str, str] = {}

    # ------------------------------------------------------------------
    # aiohttp handlers
    # ------------------------------------------------------------------

    async def _ws_handler(self, request: web.Request) -> web.WebSocketResponse:
        self.last_ws_headers = dict(request.headers)
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        self._ws_clients.append(ws)
        try:
            async for msg in ws:
                if msg.type != WSMsgType.TEXT:
                    continue
                try:
                    event = RelayMessage.model_validate_json(msg.data).payload
                except Exception:
                    continue
                self.received_events.append(event)
                if event.event_type != ExecutionEventType.TOOL_START:
                    continue
                action = event.action
                params = (action.parameters or {}) if action else {}
                call_id = action.call_id if action else ""
                observation = f"echo: {params.get('message', '')}"
                await ws.send_str(
                    RelayMessage(
                        payload=BaseEvent(
                            event_type=ExecutionEventType.TOOL_END,
                            author="stub",
                            action=ToolAction(
                                name=(action.name if action else "echo"),
                                call_id=call_id,
                                parameters={},
                                observation=observation,
                            ),
                        )
                    ).model_dump_json()
                )
        finally:
            if ws in self._ws_clients:
                self._ws_clients.remove(ws)
        return ws

    # ------------------------------------------------------------------
    # Helpers called from tests
    # ------------------------------------------------------------------

    async def push_agent_start(self, input_text: str) -> None:
        msg = RelayMessage(
            payload=BaseEvent(
                event_type=ExecutionEventType.AGENT_START,
                author="platform",
                action=AgentAction(name="invoke", parameters={"input": input_text}),
            )
        ).model_dump_json()
        for ws in list(self._ws_clients):
            await ws.send_str(msg)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        app = web.Application()
        app.router.add_get("/relay/v1/ws", self._ws_handler)
        self._runner = web.AppRunner(app)
        await self._runner.setup()
        await web.TCPSite(self._runner, "127.0.0.1", self.port).start()

    async def stop(self) -> None:
        for ws in list(self._ws_clients):
            with contextlib.suppress(Exception):
                await ws.close()
        self._ws_clients.clear()
        if self._runner:
            await self._runner.cleanup()


@pytest_asyncio.fixture
async def stub():
    server = StubServer(_free_port())
    await server.start()
    yield server
    await server.stop()


async def _wait_connected(client, timeout: float = 2.0) -> None:
    """Poll the relay client until the underlying WS is established."""
    deadline = asyncio.get_running_loop().time() + timeout
    while client._ws is None:
        if asyncio.get_running_loop().time() > deadline:
            raise TimeoutError("relay client did not connect")
        await asyncio.sleep(0.02)


async def _wait_for_event(
    received: list,
    predicate,
    timeout: float = 2.0,
) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while not any(predicate(e) for e in received):
        if asyncio.get_running_loop().time() > deadline:
            raise TimeoutError("expected event not received")
        await asyncio.sleep(0.02)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tool_round_trip(stub):
    """TOOL_START sent over the relay elicits a matching TOOL_END."""
    from domyn.agents.integrations.hermes_platform_gateway.relay_client import (
        DomynRelayClient,
    )

    received: list[BaseEvent] = []

    async def on_event(event, meta=None, config_id=None):
        received.append(event)

    client = DomynRelayClient(
        ws_url=f"ws://127.0.0.1:{stub.port}/relay/v1/ws",
        headers={},
        on_event=on_event,
    )
    await client.connect()
    try:
        await _wait_connected(client)

        call_id = "call-1"
        await client.send_event(
            BaseEvent(
                event_type=ExecutionEventType.TOOL_START,
                author="test",
                action=ToolAction(
                    name="echo",
                    parameters={"message": "hello"},
                    call_id=call_id,
                ),
            )
        )
        await _wait_for_event(received, lambda e: e.event_type == ExecutionEventType.TOOL_END)
    finally:
        await client.disconnect()

    tool_ends = [e for e in received if e.event_type == ExecutionEventType.TOOL_END]
    assert len(tool_ends) == 1
    assert tool_ends[0].action.call_id == call_id
    assert tool_ends[0].action.observation == "echo: hello"


@pytest.mark.asyncio
async def test_tool_round_trip_empty_message(stub):
    """Empty-string parameter still produces a matching TOOL_END."""
    from domyn.agents.integrations.hermes_platform_gateway.relay_client import (
        DomynRelayClient,
    )

    received: list[BaseEvent] = []

    async def on_event(event, meta=None, config_id=None):
        received.append(event)

    client = DomynRelayClient(
        ws_url=f"ws://127.0.0.1:{stub.port}/relay/v1/ws",
        headers={},
        on_event=on_event,
    )
    await client.connect()
    try:
        await _wait_connected(client)
        await client.send_event(
            BaseEvent(
                event_type=ExecutionEventType.TOOL_START,
                author="test",
                action=ToolAction(name="echo", parameters={"message": ""}, call_id="c"),
            )
        )
        await _wait_for_event(received, lambda e: e.event_type == ExecutionEventType.TOOL_END)
    finally:
        await client.disconnect()

    tool_ends = [e for e in received if e.event_type == ExecutionEventType.TOOL_END]
    assert len(tool_ends) == 1
    assert tool_ends[0].action.observation == "echo: "


@pytest.mark.asyncio
async def test_agent_start_dispatched_to_on_event(stub):
    """An AGENT_START pushed by the relay reaches the on_event callback."""
    from domyn.agents.integrations.hermes_platform_gateway.relay_client import (
        DomynRelayClient,
    )

    received: list[BaseEvent] = []

    async def on_event(event, meta=None, config_id=None):
        received.append(event)

    client = DomynRelayClient(
        ws_url=f"ws://127.0.0.1:{stub.port}/relay/v1/ws",
        headers={},
        on_event=on_event,
    )
    await client.connect()
    try:
        await _wait_connected(client)
        await stub.push_agent_start("what time is it?")
        await _wait_for_event(received, lambda e: e.event_type == ExecutionEventType.AGENT_START)
    finally:
        await client.disconnect()

    starts = [e for e in received if e.event_type == ExecutionEventType.AGENT_START]
    assert len(starts) == 1
    assert starts[0].action.parameters["input"] == "what time is it?"


@pytest.mark.asyncio
async def test_send_event_reaches_stub(stub):
    """send_event delivers a relay message that the stub receives."""
    from domyn.agents.integrations.hermes_platform_gateway.relay_client import (
        DomynRelayClient,
    )

    client = DomynRelayClient(
        ws_url=f"ws://127.0.0.1:{stub.port}/relay/v1/ws",
        headers={},
        on_event=None,
    )
    await client.connect()
    try:
        await _wait_connected(client)
        await client.send_event(
            BaseEvent(
                event_type=ExecutionEventType.RESPONSE,
                author="hermes",
                is_partial=True,
            )
        )
        await _wait_for_event(
            stub.received_events,
            lambda e: e.event_type == ExecutionEventType.RESPONSE,
        )
    finally:
        await client.disconnect()

    response_events = [
        e for e in stub.received_events if e.event_type == ExecutionEventType.RESPONSE
    ]
    assert len(response_events) == 1
    assert response_events[0].author == "hermes"
    assert response_events[0].is_partial is True


@pytest.mark.asyncio
async def test_auth_headers_forwarded(stub):
    """WebSocket handshake forwards auth headers to the server."""
    from domyn.agents.integrations.hermes_platform_gateway.relay_client import (
        DomynRelayClient,
    )

    client = DomynRelayClient(
        ws_url=f"ws://127.0.0.1:{stub.port}/relay/v1/ws",
        headers={"api-key": "k1", "space-id": "s1", "channel-id": "c1"},
        on_event=None,
    )
    await client.connect()
    try:
        await _wait_connected(client)
    finally:
        await client.disconnect()

    assert stub.last_ws_headers.get("api-key") == "k1"
    assert stub.last_ws_headers.get("space-id") == "s1"
    assert stub.last_ws_headers.get("channel-id") == "c1"
