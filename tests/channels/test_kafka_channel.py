"""KafkaChannel, against an in-process stand-in for aiokafka (no broker needed)."""

import asyncio
import sys
import types
import uuid
from collections.abc import AsyncGenerator
from typing import Any, NamedTuple

import pytest

from domyn.agents.adapters import DomynAgentAdapter
from domyn.agents.agents import Agent
from domyn.agents.core import ApprovalType, BaseEvent, ExecutionEventType, Part
from domyn.agents.llm import BaseLLMProvider, LLMResponse
from domyn.agents.planner_strategy import ToolUsePlannerStrategy
from domyn.agents.server import AgentServer
from domyn.agents.tools import DelegateTool

# ---------------------------------------------------------------------------
# A broker in memory: one partition per topic, committed offsets per group.
# ---------------------------------------------------------------------------


class TopicPartition(NamedTuple):
    topic: str
    partition: int


class _Broker:
    def __init__(self) -> None:
        self.logs: dict[str, list[types.SimpleNamespace]] = {}
        self.committed: dict[tuple[str, TopicPartition], int] = {}
        self.changed = asyncio.Event()

    def publish(self, topic: str, value: bytes | None, key: bytes | None = None) -> None:
        log = self.logs.setdefault(topic, [])
        log.append(
            types.SimpleNamespace(topic=topic, partition=0, offset=len(log), key=key, value=value)
        )
        self.changed.set()

    def events(self, topic: str) -> list[BaseEvent]:
        return [BaseEvent.model_validate_json(r.value) for r in self.logs.get(topic, [])]


class _Consumer:
    def __init__(self, *topics: str, group_id: str, **_: Any) -> None:
        self.topics, self.group_id = topics, group_id
        self.position: dict[str, int] = {}
        self.stopped = False

    async def start(self) -> None:
        for topic in self.topics:
            tp = TopicPartition(topic, 0)
            self.position[topic] = BROKER.committed.get((self.group_id, tp), 0)

    async def getone(self) -> types.SimpleNamespace:
        while True:
            if self.stopped:
                raise RuntimeError("consumer stopped")
            for topic in self.topics:
                log = BROKER.logs.get(topic, [])
                if self.position[topic] < len(log):
                    self.position[topic] += 1
                    return log[self.position[topic] - 1]
            BROKER.changed.clear()
            await BROKER.changed.wait()

    async def commit(self, offsets: dict[TopicPartition, int]) -> None:
        for tp, offset in offsets.items():
            BROKER.committed[(self.group_id, tp)] = offset

    async def stop(self) -> None:
        self.stopped = True
        BROKER.changed.set()


class _Producer:
    def __init__(self, **_: Any) -> None:
        pass

    async def start(self) -> None:
        pass

    async def send_and_wait(self, topic: str, value: bytes, key: bytes | None = None) -> None:
        BROKER.publish(topic, value, key)

    async def stop(self) -> None:
        pass


BROKER: _Broker


@pytest.fixture(autouse=True)
def fake_aiokafka(monkeypatch: pytest.MonkeyPatch) -> _Broker:
    global BROKER
    BROKER = _Broker()
    module = types.ModuleType("aiokafka")
    module.AIOKafkaConsumer = _Consumer  # type: ignore[attr-defined]
    module.AIOKafkaProducer = _Producer  # type: ignore[attr-defined]
    structs = types.ModuleType("aiokafka.structs")
    structs.TopicPartition = TopicPartition  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "aiokafka", module)
    monkeypatch.setitem(sys.modules, "aiokafka.structs", structs)
    return BROKER


def _channel(**kwargs: Any) -> Any:
    from domyn.agents.channels.kafka import KafkaChannel

    settings: dict[str, Any] = {"topics": ["in"], "output_topic": "out", **kwargs}
    return KafkaChannel(bootstrap_servers="broker:9092", **settings)


async def _next(stream: AsyncGenerator[Any, None], timeout: float = 1.0) -> Any:
    return await asyncio.wait_for(stream.__anext__(), timeout)


def _event(kind: ExecutionEventType, **kwargs: Any) -> BaseEvent:
    return BaseEvent(event_type=kind, author="Analyst", **kwargs)


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_any_payload_is_a_turn_whose_conversation_is_the_key() -> None:
    channel = _channel()
    stream = channel.receive()
    BROKER.publish("in", b'{"alert_id": "A-1", "amount": 9500}', b"A-1")

    conversation_id, event = await _next(stream)

    assert conversation_id == "A-1"
    assert event.event_type == ExecutionEventType.AGENT_START
    assert event.conversation_id == "A-1"
    assert event.content[0].text == '{"alert_id": "A-1", "amount": 9500}'
    await channel.close()


@pytest.mark.asyncio
async def test_an_event_as_json_is_read_as_that_event() -> None:
    channel = _channel()
    stream = channel.receive()
    sent = BaseEvent(
        event_type=ExecutionEventType.AGENT_START,
        author="engine",
        content=[Part(text="look at A-2")],
    )
    BROKER.publish("in", sent.model_dump_json().encode(), b"A-2")

    conversation_id, event = await _next(stream)

    assert conversation_id == "A-2"
    assert event.author == "engine"
    assert event.content[0].text == "look at A-2"
    await channel.close()


@pytest.mark.asyncio
async def test_a_message_that_cannot_be_read_is_skipped_and_committed() -> None:
    channel = _channel()
    stream = channel.receive()
    BROKER.publish("in", b"\xff\xfe", b"bad")
    BROKER.publish("in", b'{"event_type": "agent_start"}', b"invalid")
    BROKER.publish("in", b"hello", b"good")

    conversation_id, _ = await _next(stream)

    assert conversation_id == "good"
    assert BROKER.committed[("domyn-serve", TopicPartition("in", 0))] == 2
    await channel.close()


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_only_the_chosen_complete_events_are_written() -> None:
    channel = _channel()

    await channel.send_event("A-1", _event(ExecutionEventType.LLM_START))
    await channel.send_event("A-1", _event(ExecutionEventType.RESPONSE, is_partial=True))
    await channel.send_event_priority("A-1", _event(ExecutionEventType.RESPONSE))

    written = BROKER.logs["out"]
    assert [r.key for r in written] == [b"A-1"]
    assert BaseEvent.model_validate_json(written[0].value).event_type == (
        ExecutionEventType.RESPONSE
    )
    await channel.close()


@pytest.mark.asyncio
async def test_the_written_events_can_be_chosen() -> None:
    channel = _channel(events=["tool_start"])

    await channel.send_event("A-1", _event(ExecutionEventType.TOOL_START))
    await channel.send_event("A-1", _event(ExecutionEventType.RESPONSE))

    assert [e.event_type for e in BROKER.events("out")] == [ExecutionEventType.TOOL_START]
    await channel.close()


# ---------------------------------------------------------------------------
# Turns at once, and commits
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reading_waits_while_max_in_flight_turns_run() -> None:
    channel = _channel(max_in_flight=1)
    stream = channel.receive()
    BROKER.publish("in", b"first", b"A-1")
    BROKER.publish("in", b"second", b"A-2")

    assert (await _next(stream))[0] == "A-1"
    pending = asyncio.ensure_future(stream.__anext__())
    await asyncio.sleep(0.2)
    assert not pending.done()

    await channel.send_event("A-1", _event(ExecutionEventType.TOOL_NEEDS_FEEDBACK))

    assert (await asyncio.wait_for(pending, 1.0))[0] == "A-2"
    await channel.close()


@pytest.mark.asyncio
async def test_an_offset_is_committed_once_every_earlier_turn_has_ended() -> None:
    channel = _channel()
    stream = channel.receive()
    BROKER.publish("in", b"first", b"A-1")
    BROKER.publish("in", b"second", b"A-2")
    await _next(stream)
    await _next(stream)
    tp = ("domyn-serve", TopicPartition("in", 0))

    await channel.send_event("A-2", _event(ExecutionEventType.RESPONSE))
    assert BROKER.committed[tp] == 0  # A-1 still running: it is read again after a crash

    await channel.send_event("A-1", _event(ExecutionEventType.AGENT_END))
    assert BROKER.committed[tp] == 2
    await channel.close()


@pytest.mark.asyncio
async def test_the_adapter_refusing_a_turn_ends_it() -> None:
    channel = _channel(max_in_flight=1)
    stream = channel.receive()
    BROKER.publish("in", b"first", b"A-1")
    await _next(stream)

    refused = _event(
        ExecutionEventType.ERROR_OCCURRED,
        error_code="WorkerBusy",
        metadata={"request_id": "r-1"},
    )
    await channel.send_event_priority("A-1", refused)

    assert BROKER.committed[("domyn-serve", TopicPartition("in", 0))] == 1
    await channel.close()


@pytest.mark.asyncio
async def test_close_ends_reading() -> None:
    channel = _channel()
    stream = channel.receive()
    pending = asyncio.ensure_future(stream.__anext__())
    await asyncio.sleep(0.05)

    await channel.close()

    with pytest.raises(StopAsyncIteration):
        await asyncio.wait_for(pending, 1.0)


def test_a_missing_aiokafka_says_how_to_install_it(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "aiokafka", None)

    with pytest.raises(ImportError, match=r"domyn\[kafka\]"):
        _channel()


# ---------------------------------------------------------------------------
# With AgentServer: an approval answered through the topics
# ---------------------------------------------------------------------------


class _ActThenAnswer(BaseLLMProvider):
    """Asks to block the card, then answers once the tool has a result."""

    type: str = "mock"
    name: str = "mock"
    model_name: str = "mock"
    supported_structured_outputs: None = None

    async def generate(self, messages: Any, **kwargs: Any) -> LLMResponse:
        if any(getattr(m, "role", None) == "tool" for m in messages):
            return LLMResponse(content="Card blocked.", generated_tokens=2, execution_time=0.0)
        return LLMResponse(
            content="",
            generated_tokens=2,
            execution_time=0.0,
            tool_calls=[
                {
                    "id": f"tc-{uuid.uuid4().hex[:6]}",
                    "type": "function",
                    "function": {"name": "block_card", "arguments": '{"card_id": "K-1"}'},
                }
            ],
        )

    async def generate_streaming(
        self, messages: Any, **kwargs: Any
    ) -> AsyncGenerator[LLMResponse, None]:
        yield await self.generate(messages, **kwargs)


def _server(blocked: list[str]) -> tuple[AgentServer, Any]:
    tool = DelegateTool(
        tool_name="block_card",
        tool_description="Block a card.",
        tool_parameters={"card_id": {"type": "string"}},
        required_params=["card_id"],
        approval_type=ApprovalType.USER_CONFIRM,
        wrapped=lambda card_id: blocked.append(card_id) or "blocked",
    )
    agent = Agent(
        name="Analyst",
        description="test",
        instruction="Block cards.",
        llm_provider=_ActThenAnswer(model_name="mock"),
        planner=ToolUsePlannerStrategy(),
        tools=[tool],
    )
    adapter = DomynAgentAdapter(root=agent)
    channel = _channel(topics=["alerts", "decisions"])
    return AgentServer(channel=channel, adapter=adapter), channel


async def _wait_for(kind: ExecutionEventType, count: int = 1) -> list[BaseEvent]:
    for _ in range(100):
        found = [e for e in BROKER.events("out") if e.event_type == kind]
        if len(found) >= count:
            return found
        await asyncio.sleep(0.02)
    raise AssertionError(f"no {kind.value} on the output topic")


@pytest.mark.asyncio
async def test_a_parked_turn_is_answered_through_the_topics() -> None:
    blocked: list[str] = []
    server, channel = _server(blocked)
    task = asyncio.create_task(server.serve())
    BROKER.publish("alerts", b'{"alert_id": "A-1"}', b"A-1")

    [request] = await _wait_for(ExecutionEventType.TOOL_NEEDS_FEEDBACK)
    assert blocked == []
    # Committed once parked: a restart would not read the alert again.
    assert BROKER.committed[("domyn-serve", TopicPartition("alerts", 0))] == 1

    answer = BaseEvent(
        event_type=ExecutionEventType.TOOL_FEEDBACK,
        author="analyst",
        action=request.action.model_copy(update={"approve": True}),
    )
    BROKER.publish("decisions", answer.model_dump_json().encode(), b"A-1")
    await _wait_for(ExecutionEventType.RESPONSE)
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    await channel.close()

    assert blocked == ["K-1"]
    assert BROKER.committed[("domyn-serve", TopicPartition("decisions", 0))] == 1
