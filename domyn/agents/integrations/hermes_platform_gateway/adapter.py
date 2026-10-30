"""DomynPlatformAdapter — gateway-mode bridge to the Domyn relay.

Per turn:
    AGENT_START (in) → handle_message dispatch
                    → (tool calls round-trip via the relay)
                    → RESPONSE (out) for anything that is not the turn's last word: a
                      clarify question, or any send made while the session still has
                      background delegations or tool calls pending (is_partial=True)
                    → AGENT_END (out) terminates the turn, so send() emits it — and drops
                      the turn — only once nothing is pending. Any outbound message can
                      become that terminator, hermes' busy/steer ack included.

Per platform tool call:
    pre_tool_call hook stashes (chat_id, thought) by task_id
    tool handler calls adapter.call_tool(session_key, tool_name, args, thought)
    adapter emits TOOL_START, awaits TOOL_END/TOOL_ERROR from the relay
    _resolve_tool_call wakes the awaiting handler via call_soon_threadsafe
"""

import asyncio
import contextlib
import json
import uuid
from collections.abc import Callable
from datetime import datetime
from typing import Any

from gateway.config import Platform, PlatformConfig
from gateway.platform_registry import PlatformEntry, platform_registry
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
)
from gateway.session import SessionSource, build_session_key

from domyn.agents.core import (
    METADATA_VISIBILITY_ONLY,
    ApprovalType,
    BaseEvent,
    ExecutionEventType,
    FeedbackAction,
    FeedbackParameters,
    Part,
    ToolAction,
)
from domyn.agents.core.utils import extract_parts
from domyn.agents.integrations.hermes_platform_gateway._context import request_log_scope
from domyn.agents.logger import get_logger

logger = get_logger()


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _build_session_key(channel_id: str, conversation_id: str) -> str:
    return f"domyn:{channel_id}:{conversation_id}"


def _extract_user_text(event: Any) -> str:
    """Pull plain user text out of an AGENT_START relay event."""
    action = getattr(event, "action", None)
    params = getattr(action, "parameters", None) if action else None
    if params:
        for key in ("input", "text", "message", "content"):
            val = params.get(key)
            if isinstance(val, str) and val:
                return val
    for part in getattr(event, "content", None) or []:
        text = getattr(part, "text", None)
        if text:
            return text
    if params:
        return json.dumps(params)
    return ""


def _serialise_observation(observation: Any) -> str:
    """Return *observation* as a JSON string (passthrough if already valid JSON)."""
    if isinstance(observation, str):
        try:
            json.loads(observation)
            return observation
        except json.JSONDecodeError:
            pass
    return json.dumps(observation)


def _derive_approval_choice(
    action: Any,
    *,
    denied: str,
    scope_choices: dict[str, str],
    valid_legacy: tuple[str, ...],
) -> str:
    """Map a FeedbackAction's decision to one of tools.approval's/slash_confirm's buckets.

    ``remember_scope`` (set by the frontend's scope picker) takes priority;
    ``parameters[name="choice"]`` is a fallback for frontend versions that
    predate remember_scope.
    """
    if not getattr(action, "approve", False):
        return denied
    remember_scope = getattr(action, "remember_scope", None)
    if remember_scope in scope_choices:
        return scope_choices[remember_scope]
    for param in getattr(action, "parameters", None) or []:
        name = getattr(param, "name", None)
        value = getattr(param, "value", None)
        if name is None and isinstance(param, dict):
            name = param.get("name")
            value = param.get("value")
        if name == "choice" and value in valid_legacy:
            return value
    return "once"


# Register "domyn" as a dynamic Platform value so ``Platform("domyn")``
# resolves via the enum's _missing_() hook at adapter construction time.
# Module-level side-effect because we need it before any DomynPlatformAdapter
# instance is built (including the standalone test ones that don't go
# through ctx.register_platform).
def _ensure_domyn_registered() -> None:
    if not platform_registry.is_registered("domyn"):
        platform_registry.register(
            PlatformEntry(
                name="domyn",
                label="Domyn",
                adapter_factory=lambda cfg: None,
                check_fn=lambda: True,
            )
        )


_ensure_domyn_registered()


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


class DomynPlatformAdapter(BasePlatformAdapter):
    """Bridges the Domyn relay WebSocket to hermes' gateway runner."""

    # =====================================================================
    # Lifecycle (BasePlatformAdapter contract)
    # =====================================================================

    def __init__(
        self,
        config: PlatformConfig,
        *,
        channel_id: str,
        relay_client_factory: Callable[
            [Callable[[Any, dict[str, Any] | None, str | None], Any]], Any
        ],
        config_syncer: Any = None,
        tool_rename_mapping: dict[str, str],
    ) -> None:
        super().__init__(config=config, platform=Platform("domyn"))
        self._channel_id = channel_id
        # Reconciles pod settings (currently the command-approval mode) against
        # the platform configuration. Driven per turn from _handle_agent_start so
        # a configuration activated after this pod spawned still takes effect.
        # None disables the per-turn sync entirely (tests, missing config URL).
        self._config_syncer = config_syncer
        # Per-conversation state — keyed by adapter-internal session_key.
        self._turn_by_session: dict[str, Any] = {}
        # Inbound relay envelope ``meta`` (W3C ``traceparent`` + platform tags),
        # kept per conversation so outbound frames — emitted cross-loop, long
        # after the inbound coroutine's context has unwound — can re-join the
        # same distributed trace. Lifecycle mirrors ``_turn_by_session``.
        self._meta_by_session: dict[str, Any] = {}
        # Per-tool-call state — keyed by hermes' registry task_id.
        self._chat_id_by_task: dict[str, str] = {}
        self._thought_by_task: dict[str, str] = {}
        # Per-call_id state — for visibility (built-in tools) vs real
        # platform tool round-trips. Disjoint keyspaces.
        self._visibility_call_id_by_task: dict[str, str] = {}
        self._pending_calls: dict[str, Any] = {}  # call_id -> (Future, loop)
        self._session_by_call: dict[str, str] = {}  # call_id -> session_key

        # Pending gateway slash-confirm prompts that were surfaced to the
        # Domyn frontend as TOOL_NEEDS_FEEDBACK. Keyed by confirm_id (which
        # we also use as the FeedbackAction.call_id) → (chat_id, session_key).
        self._pending_slash_confirms: dict[str, tuple[str, str]] = {}
        # Pending dangerous-command exec approvals (tools.approval, separate
        # subsystem from tools.slash_confirm). call_id → (chat_id, session_key).
        self._pending_exec_approvals: dict[str, tuple[str, str]] = {}
        # Conversations whose last turn was cancelled by the user. One-shot:
        # consumed by the next AGENT_START to prepend _CANCEL_NOTE so the LLM
        # doesn't try to also answer the cancelled question it sees in the
        # session history.
        self._cancelled_conversations: set[str] = set()
        # Per-conversation keep-alive tasks — keyed by session_key. Lifetime
        # mirrors the _turn_by_session entry (see _start_heartbeat).
        self._heartbeat_by_session: dict[str, asyncio.Task[None]] = {}
        # Delegation ids whose "[ASYNC DELEGATION ... COMPLETE — <id>]"
        # re-injection must be swallowed by handle_message instead of forging
        # a new agent turn. Armed on cancel with every id the session has in
        # the async_delegation registry regardless of status — a delegation
        # that finished concurrently with the cancel already queued its
        # completion. Pruned when the completion is dropped (the watcher
        # re-injects each id exactly once). Keying by id means a NEW
        # legitimate dispatch after the cancel needs no re-arm: its fresh id
        # is never in this set.
        self._drop_async_completion_ids: set[str] = set()
        # Conversation-level fallback for the above, armed only when the
        # registry is unreadable at cancel time (module missing or bare).
        # Coarser: it would also swallow completions of dispatches NEWER
        # than the cancel, so mark_background_delegation re-arms delivery by
        # discarding the conversation.
        self._drop_async_completion_conversations: set[str] = set()
        # Per-hermes-session state — keyed by hermes session_id, populated by
        # on_session_start before the first pre_llm_call fires.
        self._chat_id_by_hermes_session: dict[str, str] = {}  # session_id → chat_id
        # Per-LLM-turn state — keyed by hermes session_id.
        self._llm_by_session: dict[str, tuple[str, str]] = {}  # session_id → (chat_id, call_id)
        # Count of in-flight background delegate_task dispatches per session.
        # While > 0, outbound sends are emitted as non-terminal RESPONSEs
        # (is_partial=True) instead of a final AGENT_END — see send(). Each
        # completion re-entry (synthetic internal message) decrements; the
        # send that follows the last completion is the one that closes the
        # turn with AGENT_END.
        self._bg_delegation_by_session: dict[str, int] = {}
        # Sessions currently inside _send_with_retry — i.e. delivering a
        # real/final response, not a hermes interim narrative. Lets send() emit
        # to the wire even when reply_to is None (e.g. a background-delegation
        # completion re-enters as a synthetic message with no message_id).
        self._authoritative_send_sessions: set[str] = set()
        # Idempotency token per logical send, keyed by session_key and owned by
        # _send_with_retry for the duration of its retry loop. The base adapter
        # retries a failed delivery by calling send() again; without a stable id
        # each attempt would go out as a distinct event, so a write that actually
        # reached the relay before failing would surface as a duplicate that no
        # dedup layer can catch (same turn, different event_id). See send().
        self._send_event_id_by_session: dict[str, str] = {}
        self._client = relay_client_factory(self._on_event)
        self._tool_rename_mapping = tool_rename_mapping

    # ---------------------------------------------------------------------
    # Logging helper — every method hermes calls on us logs one line at the
    # top so the gateway → adapter interface is fully visible in the log.
    # ---------------------------------------------------------------------

    @staticmethod
    def _log_gateway_call(name: str, **kwargs: Any) -> None:
        """Log a single ``gateway → <method>(...)`` line with kwarg previews."""
        parts: list[str] = []
        for key, val in kwargs.items():
            preview = repr(val)
            if len(preview) > 120:
                preview = preview[:120] + "..."
            parts.append(f"{key}={preview}")
        logger.debug("gateway → %s(%s)", name, ", ".join(parts))

    def _carrier_for_conversation(self, conversation_id: str | None) -> Any | None:
        """Return the stored inbound relay ``meta`` for *conversation_id*, if any."""
        if not conversation_id:
            return None
        return self._meta_by_session.get(_build_session_key(self._channel_id, conversation_id))

    async def _send_traced(self, event: Any) -> None:
        """Send *event* to the relay inside the turn's log/trace scope.

        Single choke point for every outbound frame: binds the event's
        correlation IDs to the log context and re-parents a span on the turn's
        inbound trace context, so the relay client's outbound-frame log (and any
        nested logging) carries the same ``trace`` id as the platform plus the
        per-request IDs — matching the platform's log shape.
        """
        with request_log_scope(
            carrier=self._carrier_for_conversation(getattr(event, "conversation_id", None)),
            conversation_id=getattr(event, "conversation_id", None),
            interaction_id=getattr(event, "interaction_id", None),
            turn_id=getattr(event, "turn_id", None),
            event_id=getattr(event, "event_id", None),
            author=getattr(event, "author", None),
        ):
            await self._client.send_event(event)

    async def connect(self, **kwargs) -> bool:
        self._log_gateway_call("connect")
        await self._client.connect()
        self._mark_connected()
        return True

    async def disconnect(self) -> None:
        self._log_gateway_call("disconnect")
        await self._client.disconnect()
        self._fail_pending("disconnect")
        self._mark_disconnected()

    async def send_typing(self, chat_id: str, metadata: Any = None) -> None:
        self._log_gateway_call("send_typing", chat_id=chat_id, metadata=metadata)
        return None

    # Seconds between relay keep-alive pings. Must stay well below the platform's
    # SSE read timeout: the heartbeat is the liveness signal that keeps a silent
    # turn (long LLM generation, long tool run, detached delegate_task subagents)
    # from being killed as delegate_timeout.
    _HEARTBEAT_INTERVAL: float = 15.0

    # ---------------------------------------------------------------------
    # Turn-lifetime heartbeat.
    #
    # Deliberately NOT tied to hermes' _keep_typing: that task only lives
    # while a _process_message_background task is running, and a top-level
    # delegate_task always dispatches its subagents detached (hermes forces
    # background=True at depth 0). The parent turn wraps up, its
    # processing task exits — cancelling _keep_typing — and the subagents
    # then run for minutes with no gateway task alive. The FE keeps its SSE
    # stream open across that window because send() kept the turn alive
    # (bg_pending), so the liveness signal must span the *turn*, not the
    # processing task.
    #
    # Lifetime therefore mirrors the _turn_by_session entry: started when
    # AGENT_START stores the turn, stopped wherever the turn is popped
    # (final send, clarify going out, user cancel, disconnect). A clarify
    # wait correctly has NO heartbeat — the agent is waiting on the user —
    # and the answer's AGENT_START restores turn + heartbeat together.
    # ---------------------------------------------------------------------

    def _start_heartbeat(self, session_key: str) -> None:
        """Spawn the keep-alive loop for *session_key* (idempotent)."""
        self._log_gateway_call(
            "_start_heartbeat",
            session_key=session_key,
            heartbeat_count=len(self._heartbeat_by_session),
        )
        existing = self._heartbeat_by_session.get(session_key)
        if existing is not None and not existing.done():
            return
        self._heartbeat_by_session[session_key] = asyncio.create_task(
            self._heartbeat_loop(session_key)
        )

    def _stop_heartbeat(self, session_key: str) -> None:
        """Cancel the keep-alive loop for *session_key*, if any."""
        self._log_gateway_call(
            "_stop_heartbeat",
            session_key=session_key,
            heartbeat_count=len(self._heartbeat_by_session),
        )
        task = self._heartbeat_by_session.pop(session_key, None)
        if task is not None and not task.done():
            task.cancel()

    async def _heartbeat_loop(self, session_key: str) -> None:
        """Emit periodic TOOL_UPDATE heartbeats while the turn is open."""
        tick = 0
        try:
            while True:
                await asyncio.sleep(self._HEARTBEAT_INTERVAL)
                turn = self._turn_by_session.get(session_key)
                if turn is None:
                    # Every pop site calls _stop_heartbeat, so a missing turn
                    # means we lost that race — self-terminate rather than
                    # ping for a turn the FE no longer waits on.
                    return
                tick += 1
                action_name = getattr(turn.action, "name", None) or "task"
                heartbeat = BaseEvent(
                    event_type=ExecutionEventType.TOOL_UPDATE,
                    author=turn.author,
                    interaction_id=turn.interaction_id,
                    turn_id=turn.turn_id,
                    conversation_id=turn.conversation_id,
                    action=turn.action,
                    content=[
                        Part(
                            metadata={
                                "call_id": str(uuid.uuid4()),
                                "progress": tick,
                                "total": None,
                                "message": f"Still running '{action_name}' after {tick * self._HEARTBEAT_INTERVAL} seconds...",
                            }
                        )
                    ],
                    is_partial=True,
                )
                try:
                    await self._send_traced(heartbeat)
                except asyncio.CancelledError:
                    raise
                except Exception:
                    pass
        finally:
            if self._heartbeat_by_session.get(session_key) is asyncio.current_task():
                self._heartbeat_by_session.pop(session_key, None)

    async def get_chat_info(self, chat_id: str) -> dict[str, Any]:
        self._log_gateway_call("get_chat_info", chat_id=chat_id)
        return {"name": chat_id, "type": "dm", "chat_id": chat_id}

    async def interrupt_session_activity(self, session_key: str, chat_id: str) -> None:
        self._log_gateway_call(
            "interrupt_session_activity",
            session_key=session_key,
            chat_id=chat_id,
        )
        await super().interrupt_session_activity(session_key, chat_id)

    async def on_processing_start(self, event: MessageEvent) -> None:
        self._log_gateway_call("on_processing_start", event=event)
        await super().on_processing_start(event)

    async def on_processing_complete(self, event: MessageEvent, outcome: Any) -> None:
        self._log_gateway_call(
            "on_processing_complete",
            chat_id=getattr(getattr(event, "source", None), "chat_id", None),
            outcome=getattr(outcome, "value", outcome),
        )
        await super().on_processing_complete(event, outcome)

    async def send_clarify(
        self,
        chat_id: str,
        question: str,
        choices: list | None,
        clarify_id: str,
        session_key: str,
        metadata: dict[str, Any] | None = None,
    ) -> SendResult:
        """Send a clarify prompt to the user as a plain text message.

        Multiple choice (``choices`` non-empty) is rendered as a numbered list,
        and ``mark_awaiting_text(clarify_id)`` is called so the gateway's
        text-intercept (``GatewayRunner._maybe_intercept_clarify_text``) captures
        the reply: a number, the option text, or a free-text answer. Open-ended
        (``choices`` empty) sends the question alone; the next user message in
        the session resolves the clarify through the same intercept.
        """
        if choices:
            lines = [f"❓ {question}", ""]
            for i, choice in enumerate(choices, start=1):
                lines.append(f"  {i}. {choice}")
            lines.append("")
            lines.append("Reply with the number, the option text, or your own answer.")
            text = "\n".join(lines)
            # Text fallback: enable text-capture so the gateway intercept
            # picks up the user's typed reply (e.g. "2" or choice text).
            from tools.clarify_gateway import mark_awaiting_text

            mark_awaiting_text(clarify_id)
        else:
            text = f"❓ {question}"
        return await self.send(
            chat_id=chat_id,
            content=text,
            metadata={"action": "clarify", **(metadata or {})},
        )

    async def send_slash_confirm(
        self,
        chat_id: str,
        title: str,
        message: str,
        session_key: str,
        confirm_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> SendResult:
        """Surface the gateway's slash-confirm prompt to the Domyn frontend.

        Emits a ``TOOL_NEEDS_FEEDBACK`` carrying a ``FeedbackAction`` with
        ``approval_type=USER_CONFIRM``. The frontend renders Approve Once /
        Always Approve / Cancel and replies with ``TOOL_FEEDBACK``. The
        inbound handler maps the reply to one of the three slash-confirm
        choices and calls ``tools.slash_confirm.resolve``.

        Returning ``success=True`` suppresses the gateway's text fallback —
        the resolver's reply, when non-empty, is forwarded as a normal
        ``AGENT_END`` from ``_handle_slash_confirm_feedback``.
        """
        self._log_gateway_call(
            "send_slash_confirm",
            chat_id=chat_id,
            title=title,
            message=message,
            session_key=session_key,
            confirm_id=confirm_id,
            metadata=metadata,
        )
        turn = self._turn_by_session.get(_build_session_key(self._channel_id, chat_id))
        if turn is None:
            return SendResult(success=False, error="no active turn for slash-confirm")

        action = FeedbackAction(
            name=title.lstrip("/") or "slash_confirm",
            parameters=[],
            approval_type=ApprovalType.USER_CONFIRM,
            approve=True,
            call_id=confirm_id,
            thought=message,
        )
        event = BaseEvent(
            event_type=ExecutionEventType.TOOL_NEEDS_FEEDBACK,
            author=turn.author,
            interaction_id=turn.interaction_id,
            turn_id=turn.turn_id,
            conversation_id=turn.conversation_id,
            content=[Part(text=message)],
            action=action,
            need_feedback=True,
            metadata={
                "slash_confirm": {
                    "title": title,
                    "choices": ["once", "always", "cancel"],
                },
                **(metadata or {}),
            },
        )

        # Stash before send so a fast frontend reply cannot race.
        self._pending_slash_confirms[turn.turn_id] = (chat_id, session_key)
        logger.debug("domyn-adapter: pending exec-approval=%s", self._pending_exec_approvals)
        try:
            await self._send_traced(event)
        except Exception as exc:
            self._pending_slash_confirms.pop(turn.turn_id, None)
            logger.warning("domyn-adapter: send_slash_confirm failed - %s", exc)
            return SendResult(success=False, error=str(exc), retryable=True)
        return SendResult(success=True, message_id=event.event_id)

    async def send_exec_approval(
        self,
        chat_id: str,
        command: str,
        session_key: str,
        description: str = "dangerous command",
        metadata: dict[str, Any] | None = None,
        *,
        allow_permanent: bool = True,
        allow_session: bool = True,
        smart_denied: bool = False,
    ) -> SendResult:
        """Surface a dangerous-command approval to the Domyn frontend.

        Mirror of :meth:`send_slash_confirm` for the ``tools.approval``
        subsystem. The frontend renders the offered choices and replies with
        ``TOOL_FEEDBACK``; the inbound handler calls
        ``tools.approval.resolve_gateway_approval`` with the derived choice.

        The gateway passes ``allow_permanent`` / ``allow_session`` / ``smart_denied``
        (its button-based approval call) so we render the same choices as its
        plain-text fallback (``_format_exec_approval_fallback``):
          - ``smart_denied`` → only ``once`` / ``deny`` (owner override of a
            Smart DENY, one operation only)
          - ``not allow_session`` → hide ``session`` (e.g. no session scope to
            allowlist against)
          - ``not allow_permanent`` → hide ``always`` (e.g. a risk scan flagged
            the command — permanent allowlisting is inappropriate)
          - otherwise → ``once`` / ``session`` / ``always`` / ``deny``

        Accepting these keyword-only args keeps us signature-compatible with
        the gateway; missing one raises ``TypeError`` and forces the gateway
        onto its plain-text fallback ("Button-based approval failed...") —
        this happened for ``allow_session`` when it was added upstream and
        this adapter hadn't caught up yet.

        Returning ``success=True`` suppresses the gateway's plain-text
        fallback, so the prompt is delivered exactly once,
        as a structured HITL event the frontend can render.
        """
        self._log_gateway_call(
            "send_exec_approval",
            chat_id=chat_id,
            command=command,
            session_key=session_key,
            description=description,
            allow_permanent=allow_permanent,
            allow_session=allow_session,
            smart_denied=smart_denied,
            metadata=metadata,
        )
        turn = self._turn_by_session.get(_build_session_key(self._channel_id, chat_id))
        if turn is None:
            return SendResult(success=False, error="no active turn for exec-approval")

        call_id = str(uuid.uuid4())
        cmd_preview = command if len(command) <= 3800 else command[:3800] + "..."
        body = f"⚠️ **Command Approval Required**\n```\n{cmd_preview}\n```\nReason: {description}"
        action = FeedbackAction(
            name="exec_approval",
            parameters=[
                FeedbackParameters(name="command", value=command, type="str"),
                FeedbackParameters(name="description", value=description, type="str"),
            ],
            approval_type=ApprovalType.USER_CONFIRM,
            approve=True,
            call_id=call_id,
            thought=body,
        )
        event = BaseEvent(
            event_type=ExecutionEventType.TOOL_NEEDS_FEEDBACK,
            author=turn.author,
            interaction_id=turn.interaction_id,
            turn_id=turn.turn_id,
            conversation_id=turn.conversation_id,
            content=[Part(text=body)],
            action=action,
            need_feedback=True,
            metadata={
                "exec_approval": {
                    "command": command,
                    "description": description,
                },
                **(metadata or {}),
            },
        )

        self._pending_exec_approvals[turn.turn_id] = (chat_id, session_key)
        logger.debug("domyn-adapter: pending exec-approval=%s", self._pending_exec_approvals)
        try:
            await self._send_traced(event)
        except Exception as exc:
            self._pending_exec_approvals.pop(turn.turn_id, None)
            logger.warning("domyn-adapter: send_exec_approval failed - %s", exc)
            return SendResult(success=False, error=str(exc), retryable=True)
        return SendResult(success=True, message_id=event.event_id)

    async def _handle_exec_approval_feedback(self, event: Any) -> bool:
        """Route a TOOL_FEEDBACK reply to a pending exec-approval.

        Returns True iff this event matched a pending exec-approval call_id
        (in which case the dispatcher should NOT also try the slash-confirm
        handler).

        Choice derivation (see :func:`_derive_approval_choice`):
          - ``approve=False`` → ``"deny"``
          - ``approve=True`` + ``remember_scope="once"`` → ``"once"``
          - ``approve=True`` + ``remember_scope="conversation"`` → ``"session"``
          - ``approve=True`` + ``remember_scope`` in
            ``"user"``/``"project"``/``"global"`` → ``"always"``
          - ``approve=True`` + no recognised ``remember_scope`` → falls
            back to legacy ``parameters[name="choice"]`` (``"once"`` /
            ``"session"`` / ``"always"``), else ``"once"``
        """
        logger.debug(
            "domyn-adapter: checking exec-approval feedback for call_id=%s",
            getattr(getattr(event, "action", None), "call_id", None),
        )
        action = getattr(event, "action", None)
        logger.debug("domyn-adapter: feedback action=%s", action)
        if action is None:
            logger.debug(
                "domyn-adapter: TOOL_FEEDBACK with no action, cannot be exec-approval feedback"
            )
            return False
        turn_id = getattr(event, "turn_id", None)
        if not turn_id:
            logger.debug(
                "domyn-adapter: TOOL_FEEDBACK with no turn_id in action, cannot be exec-approval feedback"
            )
            return False
        logger.debug("domyn-adapter: looking up pending exec-approval for turn_id=%s", turn_id)
        logger.debug("domyn-adapter: pending exec-approvals: %s", self._pending_exec_approvals)
        pending = self._pending_exec_approvals.pop(turn_id, None)
        if pending is None:
            logger.debug("domyn-adapter: no pending exec-approval for turn_id=%s", turn_id)
            return False
        _chat_id, session_key = pending

        choice = _derive_approval_choice(
            action,
            denied="deny",
            scope_choices={
                "once": "once",
                "conversation": "session",
                "user": "always",
                "project": "always",
                "global": "always",
            },
            valid_legacy=("once", "session", "always"),
        )

        try:
            from tools.approval import resolve_gateway_approval
        except Exception as exc:
            logger.warning("domyn-adapter: tools.approval import failed - %s", exc)
            return True

        try:
            resolved = resolve_gateway_approval(session_key, choice)
            logger.info(
                "domyn-adapter: exec-approval resolved session=%s choice=%s n=%d",
                session_key,
                choice,
                resolved,
            )
        except Exception as exc:
            logger.warning("domyn-adapter: resolve_gateway_approval failed - %s", exc)
        return True

    async def _handle_slash_confirm_feedback(self, event: Any) -> None:
        """Route a TOOL_FEEDBACK whose turn_id matches a pending slash-confirm.

        Three-way choice is derived from the FeedbackAction (see
        :func:`_derive_approval_choice`):
          - ``approve=False`` → ``"cancel"``
          - ``approve=True`` + ``remember_scope="once"`` → ``"once"``
          - ``approve=True`` + any other recognised ``remember_scope``
            (``conversation``/``user``/``project``/``global``) → ``"always"``
            — this handler's protocol has no middle "session" bucket
          - ``approve=True`` + no recognised ``remember_scope`` → falls
            back to legacy ``parameters[name="choice", value="always"]``,
            else ``"once"``

        Events whose turn_id isn't pending are ignored — that keeps any
        future agent-level HITL traffic flowing through its own path.
        """
        action = getattr(event, "action", None)
        if action is None:
            logger.debug(
                "domyn-adapter: TOOL_FEEDBACK with no action, cannot be slash-confirm feedback"
            )
            return
        turn_id = getattr(event, "turn_id", None)
        if not turn_id:
            logger.debug(
                "domyn-adapter: TOOL_FEEDBACK with no turn_id, cannot be slash-confirm feedback"
            )
            return
        pending = self._pending_slash_confirms.pop(turn_id, None)
        if pending is None:
            logger.debug(
                "domyn-adapter: no pending slash-confirm for turn_id=%s, ignoring feedback", turn_id
            )
            return
        chat_id, session_key = pending

        choice = _derive_approval_choice(
            action,
            denied="cancel",
            scope_choices={
                "once": "once",
                "conversation": "always",
                "user": "always",
                "project": "always",
                "global": "always",
            },
            valid_legacy=("once", "always"),
        )

        try:
            from tools import slash_confirm as _slash_confirm_mod
        except Exception as exc:
            logger.warning("domyn-adapter: tools.slash_confirm import failed - %s", exc)
            return

        try:
            logger.info(
                "domyn-adapter: resolving slash-confirm session=%s choice=%s",
                session_key,
                choice,
            )
            result_text = await _slash_confirm_mod.resolve(session_key, turn_id, choice)
        except Exception as exc:
            logger.warning("domyn-adapter: slash_confirm.resolve failed - %s", exc)
            return

        if result_text:
            await self.send(chat_id=chat_id, content=result_text)

    # =====================================================================
    # Inbound events from the relay
    # =====================================================================

    async def _on_event(
        self,
        event: Any,
        meta: dict[str, Any] | None = None,
        config_id: str | None = None,
    ) -> None:
        et = getattr(event, "event_type", None)
        # Scope the whole dispatch on the inbound frame's trace context (relay
        # envelope ``meta.traceparent``) + correlation IDs, so every log line
        # emitted while handling this frame is trace-joined and request-tagged.
        with request_log_scope(
            carrier=meta,
            conversation_id=getattr(event, "conversation_id", None),
            interaction_id=getattr(event, "interaction_id", None),
            turn_id=getattr(event, "turn_id", None),
            event_id=getattr(event, "event_id", None),
            author=getattr(event, "author", None),
        ):
            logger.debug(
                "domyn-adapter: inbound event type=%s conversation_id=%s call_id=%s",
                getattr(et, "value", et),
                getattr(event, "conversation_id", None),
                getattr(getattr(event, "action", None), "call_id", None),
            )
            if et == ExecutionEventType.AGENT_START:
                await self._handle_agent_start(event, meta, config_id)
                return
            if et == ExecutionEventType.TOOL_FEEDBACK:
                # Dispatch to whichever approval subsystem owns the call_id.
                # Exec-approval check first because it's the more common path
                # (dangerous-command guard fires on every risky shell call).
                if not await self._handle_exec_approval_feedback(event):
                    await self._handle_slash_confirm_feedback(event)
                return
            if et in (ExecutionEventType.TOOL_END, ExecutionEventType.TOOL_ERROR):
                self._resolve_tool_call(event)
                return
            if (
                et == ExecutionEventType.AGENT_END
                and getattr(event, "error_code", None) == "user_cancelled"
            ):
                await self._handle_cancel(event)

    _STOP_MESSAGE = "Response stopped as per user request."

    # Prepended to the first user message after a cancel. Same bracketed
    # system-note pattern hermes' gateway uses for its own restart/tool-tail
    # notes: the cancelled question stays in the session history (so
    # "continue" still works), but the model is told not to service it.
    _CANCEL_NOTE = (
        "[System note: The user cancelled the previous request. Do not "
        "resume, answer, or mention it unless the user explicitly asks. "
        "Address only the message below.]"
    )

    async def _handle_cancel(self, event: Any) -> None:
        """Handle an AGENT_END(user_cancelled) signal from the platform.

        The gateway keys its ``_active_sessions`` / ``_session_tasks`` by
        ``gateway.session.build_session_key`` (``agent:main:domyn:dm:<conv>``),
        NOT by the adapter-internal key — so calling
        ``cancel_session_processing`` / ``interrupt_session_activity`` with
        our key silently no-ops and hermes keeps running. Instead, route the
        cancellation through hermes' own ``/stop`` command: ``handle_message``
        recomputes the gateway key itself and dispatches
        ``_interrupt_and_clear_session`` — soft agent interrupt (the
        interruption stays in session history), run-generation invalidation
        (late responses from the dying turn are suppressed), session-lock
        release even if the executor thread is hung, and pending-message
        discard.

        The canned AGENT_END below is the single terminal event the platform
        sees. The ``/stop`` confirmation text hermes sends afterwards arrives
        in :meth:`send` with ``reply_to=None`` (the synthetic event carries no
        message_id) and is swallowed by the interim-noop branch.

        ``/stop`` only reaches the session's RUNNING agent (and, through
        ``AIAgent.interrupt``, its synchronous ``_active_children``).
        Background subagents — ``delegate_task(background=true)`` — are
        explicitly detached from the parent's child list and live in the
        module-level ``tools.async_delegation`` registry, so ``/stop`` never
        touches them (upstream, only the TUI gateway calls
        ``interrupt_for_session``). Without the explicit interrupt below they
        keep running after the cancel and later re-inject their completion as
        a fresh internal turn; :meth:`handle_message` drops that re-injection
        for conversations marked here.
        """
        conv_id = getattr(event, "conversation_id", None)
        if not conv_id:
            logger.warning("domyn-adapter: user_cancelled AGENT_END missing conversation_id")
            return

        session_key = _build_session_key(self._channel_id, conv_id)
        # Pop the turn up front: the FE ack must reference it, and any send
        # from the dying turn (or the /stop confirmation) must find no turn
        # and be dropped instead of reaching the relay.
        turn = self._turn_by_session.pop(session_key, None)
        self._meta_by_session.pop(session_key, None)
        self._stop_heartbeat(session_key)

        logger.info(
            "domyn-adapter: cancelling session %s (turn=%s)",
            session_key,
            getattr(turn, "event_id", None),
        )

        # Immediately unblock any platform tool calls waiting on TOOL_END.
        self._fail_pending_for_session(session_key, "user_cancelled")

        # Mark the conversation so the next question carries _CANCEL_NOTE.
        self._cancelled_conversations.add(conv_id)

        # Clear the turn's in-flight background-delegation counter wholesale:
        # every delegation is interrupted below and its completion re-entry
        # dropped by handle_message BEFORE the decrement, so nothing would
        # ever drain the counter — and a stale count would make the NEXT
        # turn's final answer go out as a non-terminal RESPONSE with its turn
        # kept alive forever.
        self._bg_delegation_by_session.pop(session_key, None)

        # Same SessionSource shape as _handle_agent_start so the gateway
        # derives the same session key as the running turn's.
        source = SessionSource(
            platform=Platform("domyn"),
            chat_id=conv_id,
            chat_name=conv_id,
            chat_type="dm",
            user_id=getattr(turn, "author", None) or getattr(event, "author", None),
            user_name=getattr(turn, "author", None) or getattr(event, "author", None),
        )

        # Kill detached background subagents before anything else — they burn
        # tokens with no one listening and /stop below cannot reach them.
        # Also arms the completion drop (by delegation id, or the
        # conversation-level fallback when the registry is unreadable) so
        # their late re-injections are swallowed by handle_message.
        self._interrupt_background_delegations(source, session_key, conv_id)

        # Ack the platform first so the FE gets its terminal event without
        # waiting for the gateway task to unwind (bounded at ~5s inside the
        # /stop dispatch).
        if turn is not None:
            # event_id must be unique per frame: the delegating transport records the
            # ids it sends and drops anything replayed back with one of them, so an
            # AGENT_END carrying the inbound AGENT_START's id is swallowed as an echo
            # and the cancel never acks (the turn then dies on a read timeout). The
            # correlation rides in metadata instead, as in the domyn/langgraph adapters.
            end_event = BaseEvent(
                event_type=ExecutionEventType.AGENT_END,
                author=turn.author,
                interaction_id=turn.interaction_id,
                turn_id=turn.turn_id,
                conversation_id=turn.conversation_id,
                content=[Part(text=self._STOP_MESSAGE)],
                metadata={"request_id": turn.event_id},
            )
            try:
                await self._send_traced(end_event)
                logger.info(
                    "domyn-adapter: sent canned AGENT_END for cancelled turn %s",
                    turn.event_id,
                )
            except Exception as exc:
                logger.warning("domyn-adapter: failed to send cancel AGENT_END - %s", exc)
        else:
            logger.warning("domyn-adapter: no active turn for cancelled session %s", session_key)

        # Dispatched even without an active turn — the turn is popped on the
        # final send while the gateway task may still be unwinding, and a
        # cold /stop is a harmless no-op reply.
        stop_msg = MessageEvent(
            text="/stop",
            message_type=MessageType.TEXT,
            source=source,
            message_id=None,
            timestamp=datetime.now(),
        )
        try:
            await self.handle_message(stop_msg)
            logger.info("domyn-adapter: dispatched /stop for cancelled session %s", session_key)
        except Exception as exc:
            logger.warning("domyn-adapter: /stop dispatch failed for %s - %s", session_key, exc)

    async def _handle_agent_start(
        self,
        event: Any,
        meta: dict[str, Any] | None = None,
        config_id: str | None = None,
    ) -> None:
        conv_id = getattr(event, "conversation_id", None)
        if not conv_id:
            logger.warning("domyn-adapter: AGENT_START missing conversation_id, dropping")
            return
        text = _extract_user_text(event)
        if not text:
            logger.warning("domyn-adapter: AGENT_START with no extractable text, dropping")
            return

        session_key = _build_session_key(self._channel_id, conv_id)

        # Re-delivery of the AGENT_START we are already running: dispatching again
        # would run the whole turn a second time and emit every one of its events
        # twice. Checked before the one-shot _CANCEL_NOTE below so a dropped
        # re-delivery doesn't consume it. DomynRelayClient._seen already covers the
        # relay's replay window; this is the backstop for anything outside it.
        #
        # Keyed on event_id, NOT on turn_id or "a turn is live": a clarify answer and
        # a queued follow-up are both legitimate second AGENT_STARTs for this session
        # and must still be dispatched.
        live_turn = self._turn_by_session.get(session_key)
        event_id = getattr(event, "event_id", None)
        if live_turn is not None and event_id and getattr(live_turn, "event_id", None) == event_id:
            logger.warning(
                "domyn-adapter: ignoring re-delivered AGENT_START %s for in-flight turn on %s",
                event_id,
                session_key,
            )
            return

        # First question after a user cancel: tell the model not to service
        # the cancelled request it still sees in the session history.
        if conv_id in self._cancelled_conversations:
            self._cancelled_conversations.discard(conv_id)
            text = f"{self._CANCEL_NOTE}\n\n{text}"

        self._turn_by_session[session_key] = event
        # Retain the inbound trace context so outbound frames for this turn —
        # emitted cross-loop by hermes hooks — re-join the same distributed trace.
        if meta:
            self._meta_by_session[session_key] = meta
        self._start_heartbeat(session_key)

        source = SessionSource(
            platform=Platform("domyn"),
            chat_id=conv_id,
            chat_name=conv_id,
            chat_type="dm",
            user_id=getattr(event, "author", None),
            user_name=getattr(event, "author", None),
        )
        msg = MessageEvent(
            text=text,
            message_type=MessageType.TEXT,
            source=source,
            message_id=getattr(event, "event_id", None),
            timestamp=datetime.now(),
        )

        # Reconcile pod settings against this turn's configuration before the
        # agent runs. Approval mode is only ever consulted DURING a turn, so
        # syncing here means the turn always executes under the configuration the
        # platform just pointed us at — including a configuration activated after
        # this pod spawned (new uid, carried on the turn). Awaited, not
        # fire-and-forget: the dangerous direction is a re-ENABLED approval
        # requirement landing after the turn already ran unguarded. Bounded and
        # fail-open inside sync(), so a slow/broken config endpoint delays the
        # turn briefly but never blocks it. Deliberately after the turn/heartbeat
        # bookkeeping above: a cancel arriving mid-fetch still finds a live turn.
        if self._config_syncer is not None:
            try:
                await self._config_syncer.sync(config_id)
            except Exception as exc:
                # ``ConfigSyncer.sync`` already fails open internally; this is the
                # backstop that makes "a settings sync never blocks a turn" a
                # property of the CALL SITE, independent of the syncer passed in.
                # CancelledError is a BaseException, so a genuine turn
                # cancellation still propagates rather than being swallowed here.
                logger.warning(
                    "domyn-adapter: config sync raised, serving turn anyway - %s",
                    type(exc).__name__,
                )
                logger.debug("domyn-adapter: config sync detail: %s", exc)

        await self.handle_message(msg)

    def _interrupt_background_delegations(
        self, source: SessionSource, session_key: str, conv_id: str
    ) -> None:
        """Interrupt this session's ``delegate_task(background=true)`` subagents.

        The async-delegation registry records each delegation under the
        GATEWAY session key (``get_current_session_key()`` at dispatch time),
        so we must rebuild that key with the same ``build_session_key``
        arguments the base adapter's ``handle_message`` uses — the
        adapter-internal key would silently match nothing.

        Before interrupting, arms the completion drop with the ids of ALL
        the session's registered delegations regardless of status — one that
        finished concurrently with the cancel already queued its completion
        event, so "0 interrupted" does not mean nothing arrives. When the
        registry is unreadable the drop degrades to conversation level.

        Hermes versions that predate ``interrupt_for_session`` (they only
        ship the global ``interrupt_all``) still record ``session_key`` and
        ``interrupt_fn`` on every delegation, so
        :meth:`_interrupt_session_records` replicates the session-scoped
        filter against the module's registry. ``interrupt_all`` is
        deliberately NOT used as a fallback — this gateway multiplexes many
        conversations, and a global sweep would kill other users' background
        work.
        """
        try:
            from tools import async_delegation
        except Exception as exc:
            logger.warning(
                "domyn-adapter: tools.async_delegation unavailable, "
                "background subagents (if any) not interrupted - %s",
                type(exc).__name__,
            )
            logger.debug(
                "domyn-adapter: tools.async_delegation unavailable, "
                "background subagents (if any) not interrupted - %s",
                exc,
            )
            self._drop_async_completion_conversations.add(conv_id)
            return
        try:
            gateway_key = build_session_key(
                source,
                group_sessions_per_user=self.config.extra.get("group_sessions_per_user", True),
                thread_sessions_per_user=self.config.extra.get("thread_sessions_per_user", False),
            )
            armed = self._collect_session_delegation_ids(async_delegation, gateway_key)
            if armed is None:
                self._drop_async_completion_conversations.add(conv_id)
            else:
                self._drop_async_completion_ids.update(armed)
            interrupt_for_session = getattr(async_delegation, "interrupt_for_session", None)
            if callable(interrupt_for_session):
                count = interrupt_for_session(session_key=gateway_key, reason="user_cancelled")
            else:
                count = self._interrupt_session_records(async_delegation, gateway_key)
        except Exception as exc:
            logger.warning(
                "domyn-adapter: background-delegation interrupt failed for %s - %s",
                session_key,
                type(exc).__name__,
            )
            logger.debug(
                "domyn-adapter: background-delegation interrupt failed for %s - %s",
                session_key,
                exc,
            )
            self._drop_async_completion_conversations.add(conv_id)
            return
        logger.info(
            "domyn-adapter: interrupted %d background delegation(s) for cancelled "
            "session %s (completion drop armed for %s)",
            count,
            session_key,
            f"{len(armed)} delegation id(s)" if armed is not None else "the whole conversation",
        )

    @staticmethod
    def _collect_session_delegation_ids(
        async_delegation: Any, gateway_key: str
    ) -> list[str] | None:
        """Ids of ALL delegations this session registered, or None if unreadable.

        Deliberately no status filter: a completed/interrupted record may
        still have its completion queued for re-injection, and arming an id
        whose completion was already delivered is harmless — each id
        re-injects at most once, so a stale mark is simply never hit.
        """
        records_lock = getattr(async_delegation, "_records_lock", None)
        records = getattr(async_delegation, "_records", None)
        if records_lock is None or records is None:
            return None
        with records_lock:
            return [
                str(r.get("delegation_id"))
                for r in records.values()
                if r.get("delegation_id") and str(r.get("session_key") or "") == gateway_key
            ]

    @staticmethod
    def _interrupt_session_records(async_delegation: Any, gateway_key: str) -> int:
        """Session-scoped interrupt for hermes without ``interrupt_for_session``.

        Behaviour-identical to the upstream helper: filter the module's
        ``_records`` registry down to this session's running delegations and
        call each record's ``interrupt_fn`` (both fields exist in every
        hermes version that has background delegation at all).
        """
        records_lock = getattr(async_delegation, "_records_lock", None)
        records = getattr(async_delegation, "_records", None)
        if records_lock is None or records is None:
            logger.warning(
                "domyn-adapter: async_delegation has neither interrupt_for_session "
                "nor a records registry — background subagents not interrupted"
            )
            return 0
        with records_lock:
            targets = [
                r
                for r in records.values()
                if r.get("status") == "running" and str(r.get("session_key") or "") == gateway_key
            ]
        count = 0
        for record in targets:
            fn = record.get("interrupt_fn")
            if callable(fn):
                try:
                    fn()
                    count += 1
                except Exception as exc:
                    logger.debug(
                        "domyn-adapter: interrupt_fn failed for %s - %s",
                        record.get("delegation_id"),
                        exc,
                    )
        return count

    # Stable prefix of tools.process_registry._format_async_delegation output
    # ("[ASYNC DELEGATION COMPLETE — ...]" / "[ASYNC DELEGATION BATCH
    # COMPLETE — ...]") — the only internal re-injections a cancel must drop.
    _ASYNC_COMPLETION_PREFIX = "[ASYNC DELEGATION"

    # =====================================================================
    # Outbound emits
    # =====================================================================

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> SendResult:
        """Emit the turn's final AGENT_END, or a RESPONSE while the turn stays open.

        - An interim narrative (``reply_to`` is None, not a clarify, and the
          session not flagged by ``_send_with_retry``) is not sent: it is
          surfaced as ``TOOL_START.thought`` instead.
        - A clarify prompt outside ``_send_with_retry`` is a ``RESPONSE``.
        - While the turn's background delegations or tool calls are pending,
          the send is a partial ``RESPONSE`` and the turn stays open.
        - Otherwise ``AGENT_END``, which closes the turn and stops its heartbeat.
        """
        self._log_gateway_call(
            "send",
            chat_id=chat_id,
            content=content,
            reply_to=reply_to,
            metadata=metadata,
        )
        session_key = _build_session_key(self._channel_id, chat_id)
        turn = self._turn_by_session.get(session_key)
        is_clarify = metadata and metadata.get("action") == "clarify"
        # A real/final response always comes through the base adapter's
        # _send_with_retry (which flags the session here); a hermes interim
        # narrative calls send() directly and is NOT flagged. This is the only
        # reliable way to tell them apart when reply_to is None — e.g. a
        # background-delegation completion re-enters as a synthetic internal
        # message with no message_id, so its final response arrives with
        # reply_to=None and must still be emitted, not suppressed.
        authoritative = session_key in self._authoritative_send_sessions

        if reply_to is None and not is_clarify and not authoritative:
            logger.debug(
                "domyn-adapter: send interim narrative for session %s (turn=%s)",
                session_key,
                getattr(turn, "event_id", None),
            )
            # Interim narrative — surfaced via TOOL_START.thought, not the wire.
            return SendResult(success=True, message_id="interim-noop")

        # Message content is PII — keep it out of prod/centralized logs (DEBUG only).
        logger.debug("EVENT CONTENT: %s", content)
        logger.debug("EVENT METADATA: %s", metadata)
        if turn is None:
            # Log the error without the content preview (PII).
            logger.warning(
                "domyn-adapter: send for %s with no prior AGENT_START",
                session_key,
            )
            logger.debug("domyn-adapter: no prior AGENT_START preview=%r", (content or "")[:100])
            return SendResult(success=False, error="no prior AGENT_START")

        # event_id is per-frame unique; let BaseEvent auto-generate.
        # Copying turn.event_id collides with TOOL_START (also fires
        # against the same turn) and confuses platforms that key on it.
        # The one exception is a retry of THIS logical send: _send_with_retry
        # allocates one id up front and every attempt reuses it, so a delivery
        # that reached the relay before failing is deduped by the consumer
        # instead of landing twice.
        # An authoritative send (final response, incl. a background-completion
        # re-entry whose synthetic message lacks a message_id) is terminal even
        # when reply_to is None → AGENT_END, not an interim RESPONSE.
        is_interim = reply_to is None and not authoritative
        # This turn dispatched background delegate_task calls that have not all
        # completed yet (post_tool_call incremented the session's counter via
        # mark_background_delegation; each completion re-entry decrements it in
        # handle_message). While any are in flight, this send ("working on it" /
        # a partial per-delegation report) is NOT the final word: emit it as a
        # non-terminal RESPONSE (is_partial=True) and keep the turn alive so
        # the last completion can deliver the real AGENT_END against this same
        # session.
        # The same holds for a tool call still awaiting its TOOL_END: hermes'
        # busy/steer ack arrives with reply_to set, so closing the turn on it
        # orphaned the call and left the run's own answer undeliverable.
        bg_pending = not is_interim and (
            self._bg_delegation_by_session.get(session_key, 0) > 0
            or self._has_pending_calls(session_key)
        )

        event = BaseEvent(
            event_type=(
                ExecutionEventType.RESPONSE
                if (is_interim or bg_pending)
                else ExecutionEventType.AGENT_END
            ),
            author=turn.author,
            interaction_id=turn.interaction_id,
            turn_id=turn.turn_id,
            conversation_id=turn.conversation_id,
            content=extract_parts(content) if content else [],
            is_partial=bg_pending,
        )
        # Retry of a logical send: reuse the id _send_with_retry pinned so the relay
        # sees one event, not one per attempt. Assigned after construction because
        # passing event_id=None explicitly would bypass BaseEvent's default factory.
        pinned_event_id = self._send_event_id_by_session.get(session_key)
        if pinned_event_id:
            event.event_id = pinned_event_id
        try:
            await self._send_traced(event)
        except Exception as exc:
            logger.warning("domyn-adapter: send_event failed - %s", exc)
            return SendResult(success=False, error=str(exc), retryable=True)
        # Keep the turn open across a background delegation so its completion can
        # re-enter and send the real final AGENT_END against this turn; popping
        # here would make that later send fail with "no prior AGENT_START".
        # The heartbeat lives exactly as long as the turn: it keeps ticking
        # through the delegation window and stops with the terminal emit.
        if not bg_pending:
            self._turn_by_session.pop(session_key, None)
            self._meta_by_session.pop(session_key, None)
            self._stop_heartbeat(session_key)
        return SendResult(success=True, message_id=event.event_id)

    async def _send_with_retry(
        self,
        chat_id: str,
        content: str,
        reply_to: str | None = None,
        metadata: Any = None,
        **kwargs: Any,
    ) -> SendResult:
        """Flag the session as *authoritative* and pin one ``event_id`` for the send.

        The base adapter routes every real/final response through this method,
        whereas hermes' interim narratives call ``send()`` directly. Marking the
        session here lets ``send()`` tell the two apart and emit a final
        response to the wire even when ``reply_to is None`` — which is the case
        for a background-delegation completion that re-enters as a synthetic
        internal message (no ``message_id`` → no reply anchor).

        This method also brackets the base class' retry loop, which re-invokes
        ``send()`` on a retryable failure. Allocating the ``event_id`` here makes
        every attempt of one logical send carry the same id, so a write that
        reached the relay before failing is deduped by the consumer rather than
        delivered twice (a fresh id per attempt is invisible to both the
        transport's event-id dedup and its turn filter).
        """
        session_key = self.session_key_for_chat(chat_id)
        self._authoritative_send_sessions.add(session_key)
        # setdefault: a nested/concurrent send for the same session keeps the id
        # already in flight rather than stealing it mid-retry.
        self._send_event_id_by_session.setdefault(session_key, f"event_{uuid.uuid4()}")
        try:
            return await super()._send_with_retry(
                chat_id=chat_id,
                content=content,
                reply_to=reply_to,
                metadata=metadata,
                **kwargs,
            )
        finally:
            self._authoritative_send_sessions.discard(session_key)
            self._send_event_id_by_session.pop(session_key, None)

    async def emit_visibility_tool_start(
        self,
        *,
        chat_id: str,
        tool_name: str,
        args: dict[str, Any],
        thought: str | None,
        call_id: str,
    ) -> None:
        """Push TOOL_START for a hermes built-in tool (visibility only).

        No pending future is registered and no TOOL_END round-trip is
        expected from the platform — the matching TOOL_END is emitted by
        ``emit_visibility_tool_end`` from the post_tool_call hook.
        """
        turn = self._turn_by_session.get(_build_session_key(self._channel_id, chat_id))
        if turn is None:
            return
        event = BaseEvent(
            event_type=ExecutionEventType.TOOL_START,
            author=turn.author,
            interaction_id=turn.interaction_id,
            turn_id=turn.turn_id,
            conversation_id=turn.conversation_id,
            # Marks the event as a report, not a request: the platform runner forwards it for
            # observability instead of executing a tool of its own that happens to share the
            # name. Stamped on all three visibility events, not just the one the runner acts
            # on, so the flag describes the event rather than one call site.
            metadata={METADATA_VISIBILITY_ONLY: True},
            action=ToolAction(
                name=tool_name,
                parameters=args or {},
                call_id=call_id,
                thought=thought,
            ),
        )
        try:
            await self._send_traced(event)
        except Exception as exc:
            logger.warning(
                "domyn-adapter: emit_visibility_tool_start %s failed - %s",
                tool_name,
                exc,
            )

    async def emit_visibility_tool_end(
        self,
        *,
        chat_id: str,
        tool_name: str,
        call_id: str,
        observation: Any,
        emitted_content: list[Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Push TOOL_END for a hermes built-in tool (companion to start).

        ``metadata`` is the tool's own ``ToolResult.metadata``, carried to the
        platform the same way a local tool's TOOL_END carries it. The
        visibility flag is applied last so a tool cannot overwrite it: it
        describes what the event *is*, and clearing it would make the runner
        execute a call nobody made.
        """
        turn = self._turn_by_session.get(_build_session_key(self._channel_id, chat_id))
        if turn is None:
            return
        event = BaseEvent(
            event_type=ExecutionEventType.TOOL_END,
            author=turn.author,
            interaction_id=turn.interaction_id,
            turn_id=turn.turn_id,
            conversation_id=turn.conversation_id,
            content=emitted_content or [],
            metadata={**(metadata or {}), METADATA_VISIBILITY_ONLY: True},
            action=ToolAction(
                name=tool_name,
                parameters={},
                call_id=call_id,
                observation=observation,
            ),
        )
        try:
            await self._send_traced(event)
        except Exception as exc:
            logger.warning(
                "domyn-adapter: emit_visibility_tool_end %s failed - %s",
                tool_name,
                exc,
            )

    async def emit_visibility_tool_error(
        self,
        *,
        chat_id: str,
        tool_name: str,
        call_id: str,
        error_message: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Push TOOL_ERROR for a hermes built-in tool that raised an exception.

        Carries the tool's own metadata on the same terms as the TOOL_END
        companion — a failed call is where it is most worth having.
        """

        turn = self._turn_by_session.get(_build_session_key(self._channel_id, chat_id))
        if turn is None:
            return
        event = BaseEvent(
            event_type=ExecutionEventType.TOOL_ERROR,
            author=turn.author,
            interaction_id=turn.interaction_id,
            turn_id=turn.turn_id,
            conversation_id=turn.conversation_id,
            error_code="tool_error",
            error_message=error_message,
            metadata={**(metadata or {}), METADATA_VISIBILITY_ONLY: True},
            action=ToolAction(
                name=tool_name,
                parameters={},
                call_id=call_id,
                observation=error_message,
            ),
        )
        try:
            await self._send_traced(event)
        except Exception as exc:
            logger.warning(
                "domyn-adapter: emit_visibility_tool_error %s failed - %s",
                tool_name,
                exc,
            )

    async def emit_llm_start(self, *, chat_id: str, model: str, call_id: str) -> None:
        """Push LLM_START for a hermes turn (visibility only — no round-trip)."""

        turn = self._turn_by_session.get(_build_session_key(self._channel_id, chat_id))
        if turn is None:
            return
        event = BaseEvent(
            event_type=ExecutionEventType.LLM_START,
            author=turn.author,
            interaction_id=turn.interaction_id,
            turn_id=turn.turn_id,
            conversation_id=turn.conversation_id,
            metadata={"model": model, "call_id": call_id},
        )
        try:
            await self._send_traced(event)
        except Exception as exc:
            logger.warning("domyn-adapter: emit_llm_start failed - %s", exc)

    async def emit_llm_end(
        self,
        *,
        chat_id: str,
        model: str,
        call_id: str,
        assistant_response: str,
    ) -> None:
        """Push LLM_END for a hermes turn (companion to emit_llm_start)."""
        turn = self._turn_by_session.get(_build_session_key(self._channel_id, chat_id))
        if turn is None:
            return
        event = BaseEvent(
            event_type=ExecutionEventType.LLM_END,
            author=turn.author,
            interaction_id=turn.interaction_id,
            turn_id=turn.turn_id,
            conversation_id=turn.conversation_id,
            metadata={"model": model, "call_id": call_id},
            content=extract_parts(assistant_response) if assistant_response else [],
        )
        try:
            await self._send_traced(event)
        except Exception as exc:
            logger.warning("domyn-adapter: emit_llm_end failed - %s", exc)

    # =====================================================================
    # Tool-call routing (real platform tools — round-trip via the relay)
    # =====================================================================

    async def call_tool(
        self,
        *,
        session_key: str,
        tool_name: str,
        args: dict[str, Any],
        thought: str | None = None,
        timeout: float = 120.0,
    ) -> str:
        """Send TOOL_START, await TOOL_END/TOOL_ERROR, return the observation as JSON."""
        turn = self._turn_by_session.get(session_key)
        if turn is None:
            return json.dumps({"error": "no active turn for session"})

        call_id = str(uuid.uuid4())
        loop = asyncio.get_running_loop()
        fut: asyncio.Future = loop.create_future()
        # Stash (future, loop) — _resolve_tool_call runs on the relay
        # client's loop, which may differ from this one, so we'll need
        # call_soon_threadsafe to wake the awaiter.
        self._pending_calls[call_id] = (fut, loop)
        self._session_by_call[call_id] = session_key

        start = BaseEvent(
            event_type=ExecutionEventType.TOOL_START,
            author=turn.author,
            interaction_id=turn.interaction_id,
            turn_id=turn.turn_id,
            conversation_id=turn.conversation_id,
            action=ToolAction(
                name=self._tool_rename_mapping.get(tool_name, tool_name),
                parameters=args,
                call_id=call_id,
                thought=thought,
            ),
        )
        logger.debug(
            "domyn-adapter: TOOL_START name=%s call_id=%s conversation_id=%s",
            self._tool_rename_mapping.get(tool_name, tool_name),
            call_id,
            turn.conversation_id,
        )
        try:
            try:
                await self._send_traced(start)
            except Exception as exc:
                logger.warning(
                    "domyn-adapter: TOOL_START send failed for %s call_id=%s - %s",
                    self._tool_rename_mapping.get(tool_name, tool_name),
                    call_id,
                    exc,
                )
                return json.dumps({"error": f"send failed: {exc}"})

            observation = await asyncio.wait_for(fut, timeout=timeout)
            logger.debug(
                "domyn-adapter: TOOL_END received name=%s call_id=%s",
                self._tool_rename_mapping.get(tool_name, tool_name),
                call_id,
            )
            return _serialise_observation(observation)
        except TimeoutError:
            logger.warning(
                "domyn-adapter: TOOL timeout name=%s call_id=%s after %.1fs",
                self._tool_rename_mapping.get(tool_name, tool_name),
                call_id,
                timeout,
            )
            return json.dumps(
                {
                    "error": f"Tool '{self._tool_rename_mapping.get(tool_name, tool_name)}' timed out after {timeout}s"
                }
            )
        except Exception as exc:
            logger.warning(
                "domyn-adapter: TOOL future raised name=%s call_id=%s - %s",
                self._tool_rename_mapping.get(tool_name, tool_name),
                call_id,
                exc,
            )
            return json.dumps({"error": str(exc)})
        finally:
            # Every exit drops this call's bookkeeping, cancellation included: a stale
            # _session_by_call entry would keep the turn open indefinitely.
            self._pending_calls.pop(call_id, None)
            self._session_by_call.pop(call_id, None)

    def _resolve_tool_call(self, event: Any) -> None:
        """Wake the awaiter in call_tool when TOOL_END/TOOL_ERROR arrives."""
        et_value = getattr(event.event_type, "value", event.event_type)
        call_id = getattr(getattr(event, "action", None), "call_id", None)
        if not call_id:
            logger.warning("domyn-adapter: %s with no call_id, dropping", et_value)
            return
        # Our own visibility-only frames come back through the relay's fan-out and
        # never had a pending future, so resolving them only logged a warning that
        # buried the real ones. Mirrors the runner's check, legacy prefix included.
        event_metadata = getattr(event, "metadata", None) or {}
        if event_metadata.get(METADATA_VISIBILITY_ONLY) or str(call_id).startswith("visibility"):
            return
        entry = self._pending_calls.pop(call_id, None)
        self._session_by_call.pop(call_id, None)
        if entry is None:
            logger.warning(
                "domyn-adapter: %s call_id=%s has no pending future (pending=%s)",
                et_value,
                call_id,
                sorted(self._pending_calls.keys()),
            )
            return
        fut, fut_loop = entry
        if fut.done():
            logger.warning(
                "domyn-adapter: %s call_id=%s future already resolved",
                et_value,
                call_id,
            )
            return
        if event.event_type == ExecutionEventType.TOOL_ERROR:
            msg = (
                getattr(event, "error_message", None)
                or f"platform tool error ({getattr(event, 'error_code', '')})"
            )
            self._resolve_future(fut, fut_loop, exc=RuntimeError(msg))
        else:
            observation = getattr(getattr(event, "action", None), "observation", None)
            self._resolve_future(fut, fut_loop, result=observation)

    def _fail_pending(self, reason: str) -> None:
        """Fail every in-flight tool call — called on disconnect."""
        for fut, fut_loop in self._pending_calls.values():
            if fut.done():
                continue
            fut_loop.call_soon_threadsafe(fut.set_exception, RuntimeError(reason))
        for task in self._heartbeat_by_session.values():
            if not task.done():
                task.cancel()
        self._heartbeat_by_session.clear()
        self._pending_calls.clear()
        self._session_by_call.clear()
        self._chat_id_by_hermes_session.clear()
        self._llm_by_session.clear()
        self._turn_by_session.clear()
        self._bg_delegation_by_session.clear()
        self._authoritative_send_sessions.clear()

    def _has_pending_calls(self, session_key: str) -> bool:
        """True while *session_key* has a tool call awaiting its TOOL_END."""
        # Snapshot: _resolve_tool_call mutates the map from the relay loop, and
        # iterating it live can raise "dictionary changed size during iteration".
        return session_key in list(self._session_by_call.values())

    def _fail_pending_for_session(self, session_key: str, reason: str) -> None:
        """Fail in-flight tool calls belonging to *session_key* only."""
        call_ids = [cid for cid, sk in self._session_by_call.items() if sk == session_key]
        for call_id in call_ids:
            entry = self._pending_calls.pop(call_id, None)
            self._session_by_call.pop(call_id, None)
            if entry is None:
                continue
            fut, fut_loop = entry
            if not fut.done():
                fut_loop.call_soon_threadsafe(fut.set_exception, RuntimeError(reason))

    @staticmethod
    def _resolve_future(
        fut: asyncio.Future,
        fut_loop: asyncio.AbstractEventLoop,
        *,
        result: Any = None,
        exc: BaseException | None = None,
    ) -> None:
        """Resolve a Future from a potentially different event loop.

        The future is bound to the loop where ``call_tool`` ran (a
        worker loop spawned by hermes' ``_run_async``). The relay
        receive loop, which calls us, runs in the gateway's loop.
        ``call_soon_threadsafe`` is the canonical cross-loop bridge.
        """

        def _apply() -> None:
            if fut.done():
                return
            if exc is not None:
                fut.set_exception(exc)
            else:
                fut.set_result(result)

        # Worker loop closed already — nothing waiting.
        with contextlib.suppress(RuntimeError):
            fut_loop.call_soon_threadsafe(_apply)

    # =====================================================================
    # Per-task bookkeeping (used by pre_tool_call / post_tool_call hooks)
    # =====================================================================

    def session_key_for_chat(self, chat_id: str) -> str:
        """Derive the adapter's internal session_key from a chat_id."""
        return _build_session_key(self._channel_id, chat_id)

    def mark_background_delegation(self, chat_id: str) -> None:
        """Count one more in-flight background delegation for this chat's turn.

        Read by :meth:`send`: while the counter is above zero the turn's
        wrap-up (and any per-delegation report) is emitted as a non-terminal
        RESPONSE (``is_partial=True``) instead of a final AGENT_END, and the
        turn is kept alive so each subagent result can re-enter and — once the
        last one lands — deliver the real final message.
        """
        if chat_id:
            key = self.session_key_for_chat(chat_id)
            self._bg_delegation_by_session[key] = self._bg_delegation_by_session.get(key, 0) + 1
            # Re-arm the coarse conversation-level drop (armed only when the
            # registry was unreadable at cancel time): a fresh legitimate
            # dispatch must have its completion delivered. Id-armed drops
            # need no re-arm — a new delegation always gets a new id.
            self._drop_async_completion_conversations.discard(chat_id)

    async def handle_message(self, event: MessageEvent) -> None:
        """Drop late background-delegation completions for cancelled turns.

        A background subagent interrupted by :meth:`_handle_cancel` (or one
        that finished concurrently with the cancel) still pushes a completion
        event that hermes' ``_async_delegation_watcher`` re-injects here as a
        synthetic ``internal`` message — which would forge a brand-new agent
        turn in a conversation the user already stopped. Swallow exactly
        those — matched by the delegation id armed at cancel time, with a
        conversation-level fallback for when the registry was unreadable —
        and let every other message pass through to the gateway.
        """
        self._log_gateway_call(
            "handle_message",
            chat_id=getattr(getattr(event, "source", None), "chat_id", None),
            internal=getattr(event, "internal", False),
            text=getattr(event, "text", None),
        )
        # A background-delegation completion re-enters as a synthetic internal
        # message (gateway ``_inject_watch_notification``): one fewer in
        # flight. When the last one lands the next authoritative send is the
        # real final word and must emit AGENT_END. Without this decrement, a
        # completion that lands while the turn is still busy gets drained as a
        # queued follow-up whose "first response" resend bypasses
        # ``_send_with_retry`` (interim-noop, counter not consumed), so the
        # follow-up's own final send still sees it pending and emits
        # ``RESPONSE is_partial=True`` — the turn is never closed and the
        # heartbeat keeps the platform session alive until the sandbox dies.
        if getattr(event, "internal", False):
            chat_id = getattr(getattr(event, "source", None), "chat_id", None)
            text = (getattr(event, "text", "") or "").lstrip()
            if text.startswith(self._ASYNC_COMPLETION_PREFIX):
                # The header line embeds the delegation id — "[ASYNC
                # DELEGATION (BATCH )COMPLETE — <id>]". Substring-match the
                # armed ids instead of parsing the line so a format drift
                # across hermes versions can't silently disarm the drop.
                header = text.split("\n", 1)[0]
                dropped = next((d for d in self._drop_async_completion_ids if d in header), None)
                if dropped is not None:
                    self._drop_async_completion_ids.discard(dropped)
                    logger.info(
                        "domyn-adapter: dropped completion of cancelled "
                        "background delegation %s (conversation %s)",
                        dropped,
                        chat_id,
                    )
                    return
                if chat_id in self._drop_async_completion_conversations:
                    logger.info(
                        "domyn-adapter: dropped background-delegation completion "
                        "for cancelled conversation %s",
                        chat_id,
                    )
                    return
            if chat_id:
                key = self.session_key_for_chat(chat_id)
                left = self._bg_delegation_by_session.get(key, 0)
                if left:
                    logger.debug(
                        "domyn-adapter: background-delegation completion for session %s — in flight %d → %d",
                        key,
                        left,
                        left - 1,
                    )
                if left <= 1:
                    self._bg_delegation_by_session.pop(key, None)
                else:
                    self._bg_delegation_by_session[key] = left - 1
        await super().handle_message(event)

    def record_task_chat(self, *, task_id: str, chat_id: str, thought: str | None = None) -> None:
        """Stash (chat_id, thought) for a registry task_id.

        The platform tool handler only receives ``task_id`` from
        ``registry.dispatch`` (not ``parent_agent``), so the
        pre_tool_call hook stores the chat_id under task_id here and the
        handler reads it back. The optional ``thought`` rides on
        ``ToolAction.thought`` so the platform can render *why* the tool
        was invoked.
        """
        if task_id and chat_id:
            self._chat_id_by_task[task_id] = chat_id
            if thought:
                self._thought_by_task[task_id] = thought

    def forget_task(self, *, task_id: str) -> None:
        """Drop the per-task stashes after the tool finishes."""
        if task_id:
            self._chat_id_by_task.pop(task_id, None)
            self._thought_by_task.pop(task_id, None)

    def thought_for_task(self, task_id: str) -> str | None:
        """Return the thought stashed by pre_tool_call, if any."""
        return self._thought_by_task.get(task_id)

    def record_visibility_call(self, *, task_id: str, call_id: str) -> None:
        """Pair a visibility TOOL_START's call_id with task_id for post_tool_call."""
        if task_id and call_id:
            self._visibility_call_id_by_task[task_id] = call_id

    def pop_visibility_call(self, task_id: str) -> str | None:
        """Return-and-clear the visibility call_id for *task_id*."""
        return self._visibility_call_id_by_task.pop(task_id, None) if task_id else None

    def record_llm_call(self, *, session_id: str, chat_id: str, call_id: str) -> None:
        """Stash (chat_id, call_id) for a hermes session_id ahead of LLM_START emission."""
        if session_id:
            self._llm_by_session[session_id] = (chat_id, call_id)

    def pop_llm_call(self, session_id: str) -> tuple[str, str] | None:
        """Return-and-clear the (chat_id, call_id) stashed for *session_id*."""
        return self._llm_by_session.pop(session_id, None) if session_id else None

    def record_hermes_session(self, *, session_id: str, chat_id: str) -> None:
        """Stash chat_id for a hermes session_id — called from on_session_start."""
        if session_id and chat_id:
            self._chat_id_by_hermes_session[session_id] = chat_id
