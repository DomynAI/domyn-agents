"""KafkaChannel — InputChannel over Kafka topics, for agents that work on events.

Setup
-----
Install the dependency::

    uv add "domyn[kafka]"

Then list it under ``serve.channels`` and run ``domyn serve agent.yaml --channel multi``::

    serve:
      channels:
        - type: kafka
          bootstrap_servers: localhost:9092
          topics: [fraud.alerts, fraud.decisions]
          output_topic: fraud.events

or build it in code and give it to ``AgentServer``.

How it works
------------
* ``receive()`` reads the input ``topics``. The message's key is the conversation's id,
  so every message about one case lands in one conversation (and, Kafka's ordering being
  per key, in order). The value is either a ``BaseEvent`` as JSON (an ``AGENT_START``, or
  a ``TOOL_FEEDBACK`` answering an approval), or anything else, which reaches the agent
  as the text of an ``AGENT_START``: a JSON alert from another system needs no wrapping.
* ``send_event()`` / ``send_event_priority()`` write the events named in ``events`` to
  ``output_topic`` as JSON, keyed by the conversation's id. By default: the approval
  requests (``TOOL_NEEDS_FEEDBACK``), the final answer (``RESPONSE``), errors and the end
  of a turn (``AGENT_END``). A partial event is never written.
* **At most ``max_in_flight`` turns at a time.** While that many are running, the channel
  stops reading, and the messages wait in Kafka rather than being refused by the adapter.
  A turn counts as running from its message until the first of: its approval request,
  its final answer, an ``AGENT_END`` or the adapter's refusal. In a tree of agents a
  sub-agent's ``AGENT_END`` ends that count early; the adapter's own limit still applies.
* **An offset is committed once its turn has ended** (or parked on an approval), never
  when the message is read, and only up to the oldest message still running in its
  partition. A message whose turn was cut short by a crash is read again on restart.

Limitations
-----------
* Anyone who can write to an input topic can start turns and answer approvals: restrict
  it with Kafka's ACLs. ``AgentServer`` takes an authenticator for a check of your own.
* A message read again after a crash starts its turn again on whatever the conversation
  holds; with an in-memory conversation manager, that is nothing.
* ``RESPONSE_TOKEN`` is never written: an agent with ``stream`` on has its final answer
  in the tokens only, so its ``RESPONSE`` does not reach the topic.
"""

import asyncio
import json
import uuid
from collections import deque
from collections.abc import AsyncGenerator, Iterable, Mapping
from typing import TYPE_CHECKING, Any

from pydantic import ValidationError

from domyn.agents.core import BaseEvent, ExecutionEventType, Part
from domyn.agents.logger import get_logger

logger = get_logger()

if TYPE_CHECKING:
    from aiokafka import AIOKafkaConsumer, AIOKafkaProducer
    from aiokafka.structs import ConsumerRecord, TopicPartition

DEFAULT_EVENTS = (
    ExecutionEventType.TOOL_NEEDS_FEEDBACK,
    ExecutionEventType.RESPONSE,
    ExecutionEventType.ERROR_OCCURRED,
    ExecutionEventType.AGENT_END,
)

# The events after which a turn is no longer running: the runner stopped on a
# question, answered, or ended; ERROR_OCCURRED only when it is the adapter refusing
# the turn (it carries the request's id), not an error the turn carries on from.
_TURN_ENDING = (
    ExecutionEventType.TOOL_NEEDS_FEEDBACK,
    ExecutionEventType.RESPONSE,
    ExecutionEventType.AGENT_END,
)

# A turn that waits for a free place does not poll Kafka meanwhile, and the group
# would drop a consumer that did not poll within the default five minutes.
_MAX_POLL_INTERVAL_MS = 3_600_000


class KafkaChannel:
    """InputChannel that reads turns from Kafka topics and writes events to one.

    Parameters
    ----------
    bootstrap_servers:
        The brokers, ``host:port`` or a list of them.
    topics:
        The topics to read turns and approval answers from.
    output_topic:
        The topic the events are written to.
    group_id:
        The consumer group. Run several servers with one group to share the work.
    events:
        The event types written to ``output_topic``.
    max_in_flight:
        The number of turns running at once before the channel stops reading.
    options:
        Passed to both the consumer and the producer: ``security_protocol``,
        ``sasl_mechanism``, ``ssl_context``…
    """

    def __init__(
        self,
        bootstrap_servers: str | list[str],
        topics: Iterable[str],
        output_topic: str,
        group_id: str = "domyn-serve",
        events: Iterable[ExecutionEventType | str] = DEFAULT_EVENTS,
        max_in_flight: int = 8,
        options: Mapping[str, Any] | None = None,
    ) -> None:
        try:
            import aiokafka  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                'aiokafka is required for KafkaChannel. Run: uv add "domyn[kafka]"'
            ) from exc
        self.topics = list(topics)
        if not self.topics:
            raise ValueError("KafkaChannel needs at least one topic to read")
        if max_in_flight < 1:
            raise ValueError("max_in_flight must be at least 1")
        self.bootstrap_servers = bootstrap_servers
        self.output_topic = output_topic
        self.group_id = group_id
        self.events = frozenset(ExecutionEventType(e) for e in events)
        self.max_in_flight = max_in_flight
        self._options = dict(options or {})

        self._consumer: AIOKafkaConsumer | None = None
        self._producer: AIOKafkaProducer | None = None
        self._producer_lock = asyncio.Lock()
        self._slots = asyncio.Semaphore(max_in_flight)
        self._stopping = False
        # The messages of the turns still running, oldest first, by conversation.
        self._running: dict[str, deque[tuple[TopicPartition, int]]] = {}
        # Per partition: the offsets read and not yet done, and the next one to read.
        self._open: dict[TopicPartition, set[int]] = {}
        self._next: dict[TopicPartition, int] = {}

    # ------------------------------------------------------------------
    # InputChannel contract
    # ------------------------------------------------------------------

    async def send_event(self, conversation_id: str, event: BaseEvent) -> None:
        """Write the event to the output topic if it is one of ``events``."""
        await self._send(conversation_id, event)

    async def send_event_priority(self, conversation_id: str, event: BaseEvent) -> None:
        """The same as ``send_event``: every write waits for the broker."""
        await self._send(conversation_id, event)

    async def receive(self) -> AsyncGenerator[tuple[str, BaseEvent], None]:
        """Yield one event per message, holding back while ``max_in_flight`` turns run."""
        from aiokafka import AIOKafkaConsumer

        self._consumer = AIOKafkaConsumer(
            *self.topics,
            bootstrap_servers=self.bootstrap_servers,
            group_id=self.group_id,
            enable_auto_commit=False,
            auto_offset_reset="earliest",
            max_poll_interval_ms=_MAX_POLL_INTERVAL_MS,
            **self._options,
        )
        await self._consumer.start()
        logger.info(f"KafkaChannel: reading {', '.join(self.topics)}")
        try:
            while not self._stopping:
                await self._slots.acquire()
                try:
                    message = await self._consumer.getone()
                except BaseException:
                    self._slots.release()
                    raise
                tp = _partition_of(message)
                self._open.setdefault(tp, set()).add(message.offset)
                self._next[tp] = message.offset + 1

                read = self._event_of(message)
                if read is None or read[1].event_type not in (
                    ExecutionEventType.AGENT_START,
                    ExecutionEventType.TOOL_FEEDBACK,
                ):
                    # Nothing will run for it: done as soon as it is read.
                    await self._done(tp, message.offset)
                    self._slots.release()
                    if read is not None:
                        yield read
                    continue

                conversation_id, event = read
                self._running.setdefault(conversation_id, deque()).append((tp, message.offset))
                logger.debug(
                    f"KafkaChannel: {event.event_type.value} conv={conversation_id} "
                    f"{tp.topic}[{tp.partition}]@{message.offset}"
                )
                yield conversation_id, event
        except Exception as exc:
            if not self._stopping:
                raise
            logger.debug(f"KafkaChannel: reading stopped: {type(exc).__name__}")
        finally:
            await self._stop_consumer()

    async def close(self) -> None:
        """Stop reading and writing."""
        self._stopping = True
        await self._stop_consumer()
        if self._producer is not None:
            producer, self._producer = self._producer, None
            await producer.stop()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _event_of(self, message: "ConsumerRecord[Any, Any]") -> tuple[str, BaseEvent] | None:
        """The message as ``(conversation_id, event)``; ``None`` for one that cannot be read."""
        key = message.key.decode("utf-8", errors="replace") if message.key else None
        try:
            text = message.value.decode("utf-8") if message.value is not None else ""
        except UnicodeDecodeError:
            logger.warning(f"KafkaChannel: skipped a message that is not UTF-8 (key={key})")
            return None
        if not text.strip():
            logger.warning(f"KafkaChannel: skipped an empty message (key={key})")
            return None

        event: BaseEvent | None = None
        try:
            body = json.loads(text)
        except json.JSONDecodeError:
            body = None
        if isinstance(body, dict) and "event_type" in body:
            try:
                event = BaseEvent.model_validate(body)
            except ValidationError as exc:
                logger.warning(
                    f"KafkaChannel: skipped an event that does not validate (key={key}): "
                    f"{type(exc).__name__}"
                )
                logger.debug(f"KafkaChannel: validation failure detail: {exc}")
                return None
        if event is None:
            event = BaseEvent(
                event_type=ExecutionEventType.AGENT_START,
                author=message.topic,
                content=[Part(text=text)],
            )

        conversation_id = key or event.conversation_id or str(uuid.uuid4())
        return conversation_id, event.model_copy(update={"conversation_id": conversation_id})

    async def _send(self, conversation_id: str, event: BaseEvent) -> None:
        if event.event_type in self.events and not event.is_partial:
            try:
                producer = await self._get_producer()
                await producer.send_and_wait(
                    self.output_topic,
                    event.model_dump_json().encode(),
                    key=conversation_id.encode(),
                )
            except Exception as exc:
                logger.warning(
                    f"KafkaChannel: failed to write {event.event_type.value} "
                    f"conv={conversation_id}: {type(exc).__name__}"
                )
                logger.debug(f"KafkaChannel: write failure detail: {exc}", exc_info=True)
        if _ends_turn(event):
            await self._turn_ended(conversation_id)

    async def _get_producer(self) -> "AIOKafkaProducer":
        async with self._producer_lock:
            if self._producer is None:
                from aiokafka import AIOKafkaProducer

                producer = AIOKafkaProducer(
                    bootstrap_servers=self.bootstrap_servers, **self._options
                )
                await producer.start()
                self._producer = producer
            return self._producer

    async def _turn_ended(self, conversation_id: str) -> None:
        running = self._running.get(conversation_id)
        if not running:
            return  # a later event of a turn already counted as ended
        tp, offset = running.popleft()
        if not running:
            del self._running[conversation_id]
        self._slots.release()
        await self._done(tp, offset)

    async def _done(self, tp: "TopicPartition", offset: int) -> None:
        """Mark one offset done and commit up to the oldest one still open."""
        open_offsets = self._open.get(tp)
        if open_offsets is None:
            return
        open_offsets.discard(offset)
        upto = min(open_offsets) if open_offsets else self._next[tp]
        if self._consumer is None:
            return
        try:
            await self._consumer.commit({tp: upto})
        except Exception as exc:
            # The partition may have moved to another member of the group: its
            # messages from here on are read again there.
            logger.warning(
                f"KafkaChannel: commit failed for {tp.topic}[{tp.partition}]: {type(exc).__name__}"
            )
            logger.debug(f"KafkaChannel: commit failure detail: {exc}")

    async def _stop_consumer(self) -> None:
        if self._consumer is not None:
            consumer, self._consumer = self._consumer, None
            await consumer.stop()


def _partition_of(message: "ConsumerRecord[Any, Any]") -> "TopicPartition":
    from aiokafka.structs import TopicPartition

    return TopicPartition(message.topic, message.partition)


def _ends_turn(event: BaseEvent) -> bool:
    if event.is_partial:
        return False
    if event.event_type in _TURN_ENDING:
        return True
    return event.event_type == ExecutionEventType.ERROR_OCCURRED and "request_id" in (
        event.metadata or {}
    )
