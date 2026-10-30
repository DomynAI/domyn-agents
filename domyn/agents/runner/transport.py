"""Transport abstraction for runner-to-runner communication.

A RunnerTransport is the channel between two runners. The local runner
delegates agent execution to a remote runner via a transport. Events flow
back from the remote runner; results (TOOL_END, LLM_END) can be sent back
via send_result() for back-delegated steps.

Custom transport types self-register by subclassing ``RunnerTransport`` and
declaring a ``type`` class variable::

    from domyn.agents.runner.transport import RunnerTransport

    class MyTransport(RunnerTransport):
        type = "my_transport"

        @classmethod
        def from_config(cls, config: dict) -> "MyTransport":
            return cls(url=config["url"])

        async def execute(self, agent_name, context):
            ...

        async def send_result(self, event, conversation_id=None):
            ...

As soon as the class is defined (i.e. the module containing it is imported)
``my_transport`` becomes available as a ``transport.type`` value in YAML
configs.  No manual registration call is needed.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from typing import Any, ClassVar

from domyn.agents.core import (
    AgentAction,
    BaseEvent,
    ExecutionEventType,
    InteractionContext,
    Part,
    RelayMessage,
)


class RunnerTransport(ABC):
    """Bidirectional channel between a local runner and a remote runner.

    Subclasses declare a ``type`` class variable (a plain string) and
    implement ``from_config`` to be discoverable by ``build_transport``.
    """

    # Subclasses set this to the string that appears as ``transport.type``
    # in YAML / dict configs.  The base class has no type so it is never
    # instantiated directly from config.
    type: ClassVar[str | None] = None

    _registry: ClassVar[dict[str, type[RunnerTransport]]] = {}

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        tag = getattr(cls, "type", None)
        if tag is not None:
            RunnerTransport._registry[tag] = cls

    @classmethod
    def from_config(cls, config: dict) -> RunnerTransport:
        """Build an instance from a raw config dict.

        Must be overridden by concrete subclasses.
        """
        raise NotImplementedError(f"from_config not implemented for {cls.__name__}")

    @abstractmethod
    def execute(
        self,
        agent_name: str,
        context: InteractionContext,
        run_config: dict[str, Any] | None = None,
    ) -> AsyncGenerator[BaseEvent, None]:
        """Stream events from the remote runner for the given agent.

        ``run_config`` is the visibility-filtered caller config (PII) to forward to
        the remote agent. It travels in the ``RelayMessage`` envelope's ``run_config``
        field — NOT inside the ``BaseEvent`` payload. ``None``/empty forwards nothing.
        """
        ...

    @abstractmethod
    async def send_result(self, event: BaseEvent, conversation_id: str | None = None) -> None:
        """Send a result event (TOOL_END, LLM_END) back to the remote runner."""
        ...

    async def cancel(self, conversation_id: str) -> None:
        """Signal the remote runner to stop processing.

        Default is a no-op. Transports that can propagate a cancellation
        signal to the remote side should override this.
        """
        return None

    async def pause_receiving(self) -> None:
        """Pause consuming events and release this transport's receive channel.

        Call when the caller is about to stop reading events for a while (e.g.
        the runner blocks executing a tool locally). A transport that holds an
        open receive connection should release it now — otherwise events the
        remote publishes meanwhile are lost on a connected-but-unread channel;
        the next :meth:`send_result` re-establishes it. Default no-op, which is
        correct for transports where not reading has no side effect.
        """
        return None

    # ------------------------------------------------------------------
    # Replay filtering (at-least-once relay -> exactly-once to runner)
    # ------------------------------------------------------------------
    #
    # The relay is at-least-once and its retained log is keyed by *channel*, which is
    # deterministic per (conversation, agent) and therefore shared by every turn of a
    # conversation. On any (re)connect without a resume cursor it replays the whole
    # retained window. The transport is the authoritative place that guarantees the
    # runner sees each event once, and it does so on two complementary axes:
    #
    #   * ``_is_current_turn`` — CROSS-turn replays. A transport instance does not
    #     survive a turn (callers may rebuild it per turn), so
    #     nothing can carry a cursor or a seen-set into the next turn; the filter is
    #     stateless instead, keyed on the ``turn_id`` every producer stamps on its
    #     outbound events.
    #   * ``_mark_seen`` — INTRA-turn replays (mid-turn reconnect, ``pause_receiving``
    #     then reopen). These carry the current ``turn_id`` and so pass the turn filter.
    #     The seen-set is a plain local created inside each transport's ``execute()``
    #     (seeded from ``context.interaction_history`` so it survives a fresh-transport
    #     resume), so there is no per-conversation keying and no cleanup.
    #
    # Both helpers are stateless; the state they operate on is owned by ``execute()``.

    @staticmethod
    def _is_current_turn(event: BaseEvent, turn_id: str | None) -> bool:
        """Return False only when *event* provably belongs to a different turn.

        The relay replays a conversation's whole retained window on a cursor-less
        connect, so a new turn's stream can carry the previous turn's events. Every
        producer copies the caller's ``turn_id`` onto its outbound events (the hermes
        gateway and the domyn and langgraph adapters all do), which makes the
        foreign ones identifiable with no state at all.

        Fail-open: when either id is missing there is nothing to compare, so the event
        is delivered rather than dropped.
        """
        if not turn_id or not event.turn_id:
            return True
        return event.turn_id == turn_id

    @staticmethod
    def _mark_seen(seen: set[str], event: BaseEvent) -> bool:
        """Return True if *event* is new (recording its id); False if already seen.

        Events without an ``event_id`` are always treated as new (never dropped).
        The caller keeps ``seen`` as a local for the lifetime of one ``execute()``.
        """
        eid = event.event_id
        if not eid:
            return True
        if eid in seen:
            return False
        seen.add(eid)
        return True

    @staticmethod
    def _seed_seen(context: InteractionContext) -> set[str]:
        """Build the seen-set from prior interaction history for this conversation.

        Gives restart/resume survival: after a crash+``TOOL_FEEDBACK`` resume the
        runner rebuilds ``interaction_history`` from the durable store, so a fresh
        ``execute()`` re-seeds and dedups any pre-restart events the relay replays.
        """
        return {event.event_id for event in (context.interaction_history or []) if event.event_id}

    def _wrap_outbound(self, event: BaseEvent, run_config: dict[str, Any] | None = None) -> str:
        """Serialize an outbound event for the wire, carrying run_config out-of-band.

        When ``run_config`` is present it is wrapped in a ``RelayMessage`` envelope
        whose ``run_config`` field carries it as a separate argument, keeping the
        ``BaseEvent`` payload free of PII. When absent (the ISOLATED/SCOPED default)
        the event is serialized raw, preserving the pre-existing wire format
        byte-for-byte so existing delegates and relays are unaffected. Receivers
        parse ``RelayMessage``-first with a raw ``BaseEvent`` fallback, so both are
        accepted.
        """
        if not run_config:
            return event.model_dump_json()
        return RelayMessage(payload=event, run_config=run_config).model_dump_json(exclude_none=True)

    def _build_outbound_event(self, agent_name: str, context: InteractionContext) -> BaseEvent:
        """Build the event that (re)starts *agent_name* on the remote runner.

        Shared by all transports so the wire payload is constructed identically:

        * ``TOOL_FEEDBACK`` resume — forward the feedback event verbatim so the
          remote sees the real ``event_type`` and resumes rather than restarting.
        * AGENT_START path — ``context.agent_input`` is an ``AGENT_START``: the one
          the runner built via ``_build_agent_start_event`` for the entrypoint, or
          the hook-processed one from ``_handle_agent_transition`` for a sub-agent.
          Copy it verbatim so ALL of its information (full content, action,
          metadata/run_config, flags) reaches the remote; only normalise the author
          and id fields.
        * Fallback — build a minimal ``AGENT_START`` from the user's text. Only
          reached when ``agent_input`` is not an AGENT_START (e.g. driving a
          transport directly, outside a Runner).
        """
        agent_input = context.agent_input
        if agent_input is not None and agent_input.event_type == ExecutionEventType.TOOL_FEEDBACK:
            return agent_input
        if agent_input is not None and agent_input.event_type == ExecutionEventType.AGENT_START:
            return agent_input.model_copy(
                update={
                    "author": agent_name,
                    "conversation_id": context.conversation_id,
                    "interaction_id": context.interaction_id,
                    "turn_id": context.turn_id,
                }
            )

        user_text = self._extract_user_text(context)
        return BaseEvent(
            event_type=ExecutionEventType.AGENT_START,
            author=agent_name,
            content=[Part(text=user_text)],
            action=AgentAction(
                name=agent_name,
                parameters={"task": user_text},
            ),
            conversation_id=context.conversation_id,
            interaction_id=context.interaction_id,
            turn_id=context.turn_id,
        )

    @staticmethod
    def _extract_user_text(context: InteractionContext) -> str:
        """Pull plain text from the input event in *context*.

        For root-agent calls the input is a ``USER_INPUT`` event whose
        ``content`` carries the user message directly.

        For sub-agent calls the input is the ``AGENT_START`` event emitted by
        the parent orchestrator's planner. The text lives in
        ``action.parameters`` (keys ``task`` and/or ``user_request``) because the
        planner encodes the handoff as an agent tool-call — there is no
        ``content`` on the event.
        """
        user_input = context.agent_input

        parts = getattr(user_input, "content", []) or []
        text_from_content = " ".join(p.text for p in parts if p.text)
        if text_from_content:
            return text_from_content

        action = getattr(user_input, "action", None)
        if action is not None:
            params: dict[str, Any] = getattr(action, "parameters", None) or {}
            task: str = params.get("task", "")
            user_request: str = params.get("user_request", "")
            if task and user_request and task != user_request:
                return f"{user_request}\n\nTask: {task}"
            return task or user_request

        return ""


def register_transport(type_name: str, factory: type[RunnerTransport]) -> None:
    """Register a transport class under *type_name*.

    Prefer declaring ``type = "my_type"`` on the subclass — that self-registers
    automatically.  Use this only when you cannot modify the class (e.g. a
    third-party transport).
    """
    RunnerTransport._registry[type_name] = factory


def build_transport(config: dict) -> RunnerTransport:
    """Build a ``RunnerTransport`` from a raw config dict.

    Resolves ``config["type"]`` (default ``"websocket"``) against the
    auto-populated ``RunnerTransport._registry``.  Any subclass that declares
    a ``type`` class variable is available here as soon as its module has been
    imported.

    Raises ``ValueError`` for unknown types.
    """
    # Ensure built-in transports are imported so they self-register.
    import domyn.agents.runner.sse_transport
    import domyn.agents.runner.websocket_transport  # noqa: F401

    transport_type: str = config.get("type", "websocket")
    cls = RunnerTransport._registry.get(transport_type)
    if cls is None:
        raise ValueError(
            f"Unknown delegate transport type: {transport_type!r}. "
            "Define a RunnerTransport subclass with type='{transport_type}' "
            "and ensure its module is imported before loading the config."
        )
    return cls.from_config(config)
