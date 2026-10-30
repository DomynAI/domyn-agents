"""AgentServer — composes an InputChannel with a FrameworkAdapter.

``AgentServer`` is the thin orchestration layer that:

  1. Calls ``auth.authenticate(event)`` on every inbound event (when auth is configured).
  2. Optionally normalises the event with an ``InboundNormalizer``.
  3. Dispatches ``AGENT_START`` and ``TOOL_FEEDBACK`` events to ``adapter.invoke()``.
  4. Routes ``TOOL_END`` / ``TOOL_ERROR`` to ``adapter.on_tool_response()``.
  5. Routes ``AGENT_END`` cancel signals to ``adapter.on_cancel()``.

``TOOL_FEEDBACK`` starts a turn rather than being handed to a live one: the runner
returns as soon as it suspends on ``TOOL_NEEDS_FEEDBACK``, so a human approval
arriving later has to re-enter through ``invoke()``.

Concurrency is managed inside the adapter (semaphore + busy buffer).
``AgentServer`` itself is single-threaded: it drives the channel's ``receive()``
generator and creates an ``asyncio.Task`` per ``AGENT_START``.

Example
-------
::

    from domyn.agents.channels import DomynRelayChannel
    from domyn.agents.adapters import DomynAgentAdapter
    from domyn.agents.server import AgentServer

    channel = DomynRelayChannel.from_env(
        api_key=os.environ["DOMYN_API_KEY"],
        channel_id=os.environ["DOMYN_CHANNEL_ID"],
        space_id=os.environ["DOMYN_SPACE_ID"],
        base_url=os.environ["DOMYN_BASE_URL"],
    )
    adapter = DomynAgentAdapter(root=my_agent)
    server = AgentServer(channel=channel, adapter=adapter)
    await server.serve()
"""

import asyncio

from domyn.agents.core import BaseEvent, ExecutionEventType
from domyn.agents.interfaces import (
    ChannelAuthenticator,
    EmitFn,
    EmitPriorityFn,
    FrameworkAdapter,
    InboundNormalizer,
    InputChannel,
)
from domyn.agents.logger import get_logger

logger = get_logger()


class AgentServer:
    """Composes an ``InputChannel`` and a ``FrameworkAdapter`` into a running server.

    Parameters
    ----------
    channel:
        Transport that delivers inbound events and accepts outbound ones.
    adapter:
        Runtime that executes agent logic for each invocation.
    auth:
        Optional authenticator called before the event reaches the adapter.
        When omitted, all requests are allowed.
    normalizer:
        Optional normalizer that maps raw channel messages to ``BaseEvent``.
        Not needed when the channel already delivers ``BaseEvent`` objects
        (e.g. ``DomynRelayChannel``).
    """

    def __init__(
        self,
        *,
        channel: InputChannel,
        adapter: FrameworkAdapter,
        auth: ChannelAuthenticator | None = None,
        normalizer: InboundNormalizer | None = None,
    ) -> None:
        self._channel = channel
        self._adapter = adapter
        self._auth = auth
        self._normalizer = normalizer

    async def serve(self) -> None:
        """Run until cancelled or the channel's ``close()`` is called.

        Drives the channel's ``receive()`` generator and dispatches events:

        * ``AGENT_START`` / ``TOOL_FEEDBACK`` → ``adapter.invoke()`` (fire-and-forget task)
        * ``TOOL_END`` / ``TOOL_ERROR``       → ``adapter.on_tool_response()`` (sync)
        * ``AGENT_END``                       → ``adapter.on_cancel()`` (sync)
        * anything else                       → ignored
        """
        try:
            async for conv_id, event in self._channel.receive():
                await self._dispatch(conv_id, event)
        except asyncio.CancelledError:
            logger.info("AgentServer: serve loop cancelled")
        finally:
            await self._adapter.close()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    # Event types that start a runner turn. ``TOOL_FEEDBACK`` belongs here because
    # resuming after a human approval is done by invoking the runner again with the
    # feedback as its input — the runner returns when it suspends on
    # ``TOOL_NEEDS_FEEDBACK``, so there is no live turn left to hand it to.
    # Dropping it (as this used to) made the whole approval gate unusable on every
    # channel: the agent asked, the human answered, and the answer went nowhere.
    _TURN_STARTING = (ExecutionEventType.AGENT_START, ExecutionEventType.TOOL_FEEDBACK)

    async def _dispatch(self, conv_id: str, event: BaseEvent) -> None:
        et = event.event_type

        if et in self._TURN_STARTING:
            # Authenticated like any other turn input. An approval decides whether a
            # gated tool runs, so accepting an unauthenticated one would let a
            # third party approve actions on the user's behalf.
            if not await self._authenticate(conv_id, event):
                return

            normalised = self._normalize(conv_id, event)
            if normalised is None:
                return

            emit: EmitFn = self._make_emit(conv_id)
            emit_priority: EmitPriorityFn = self._make_emit_priority(conv_id)

            # Fire and forget — concurrency is managed inside the adapter.
            # Store the reference in a local var to satisfy RUF006; the task
            # is intentionally detached (the adapter tracks it internally).
            _task = asyncio.create_task(
                self._adapter.invoke(normalised, emit, emit_priority),
                name=f"agent-server:invoke:{normalised.event_id or conv_id}",
            )
            del _task  # fire-and-forget by design

        elif et in (ExecutionEventType.TOOL_END, ExecutionEventType.TOOL_ERROR):
            self._adapter.on_tool_response(event)

        elif et == ExecutionEventType.AGENT_END:
            self._adapter.on_cancel(event)

        else:
            logger.debug("AgentServer: ignoring event type %s", et.value)

    async def _authenticate(self, conv_id: str, event: BaseEvent) -> bool:
        """Whether the event may proceed. Failure to authenticate is a rejection."""
        if self._auth is None:
            return True
        try:
            allowed = await self._auth.authenticate(event)
        except Exception as exc:
            logger.warning("AgentServer: auth.authenticate raised: %s — rejecting event", exc)
            return False
        if not allowed:
            logger.info(
                "AgentServer: rejected %s (auth failed) conv=%s", event.event_type.value, conv_id
            )
            return False
        return True

    def _normalize(self, conv_id: str, event: BaseEvent) -> BaseEvent | None:
        """Apply the optional normalizer. ``None`` means "drop this event"."""
        if self._normalizer is None:
            return event
        try:
            return self._normalizer.normalize(event)
        except Exception as exc:
            logger.warning(
                "AgentServer: normalizer raised for conv=%s: %s — skipping", conv_id, exc
            )
            return None

    def _make_emit(self, conv_id: str) -> EmitFn:
        channel = self._channel

        async def emit(event: BaseEvent) -> None:
            await channel.send_event(conv_id, event)

        return emit

    def _make_emit_priority(self, conv_id: str) -> EmitPriorityFn:
        channel = self._channel

        async def emit_priority(event: BaseEvent) -> None:
            await channel.send_event_priority(conv_id, event)

        return emit_priority
