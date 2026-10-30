"""hermes-platform-gateway plugin — gateway-mode multi-conversation adapter.

Registers a Domyn platform adapter via ``ctx.register_platform``. The
adapter opens one WebSocket per worker to the Domyn relay and routes
each ``conversation_id`` to its own hermes session.
"""

import asyncio
import json
import os
import uuid
from collections.abc import Callable
from dataclasses import replace
from typing import Any

from domyn.agents.integrations.domyn_platform_sdk import PlatformSettings, relay_connection
from domyn.agents.integrations.hermes_platform_gateway.schema import convert_schema
from domyn.agents.logger import get_logger
from domyn.agents.tools.base import ToolResult

# NOTE: ``.client`` is imported lazily inside the functions that need it (not at
# module top level) because it imports ``hermes_cli`` — a pod-runtime-only
# package that isn't a pyproject dependency. Importing it here would make the
# whole plugin package un-importable in a bare environment (CI, the
# no-circular-imports check). Same reason ``.adapter`` (which needs ``gateway.*``)
# is imported lazily in ``_factory``.

logger = get_logger()


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_REQUIRED = ("DOMYN_SPACE_ID", "DOMYN_CHANNEL_ID")


# ---------------------------------------------------------------------------
# Helpers — pure functions, no adapter / hermes state
# ---------------------------------------------------------------------------


def _tool_result_is_error(result: Any) -> tuple[bool, str]:
    """Inspect a hermes tool result and return ``(is_error, error_message)``.

    Hermes built-in tools (e.g. terminal) return a JSON object of the form
    ``{"output": "...", "exit_code": int, "error": str|null}``.  A non-zero
    exit code or a non-null ``error`` field signals failure.
    """
    if result is None:
        return False, ""
    data: Any = result
    if isinstance(result, str):
        try:
            data = json.loads(result)
        except Exception:
            return False, ""
    if not isinstance(data, dict):
        return False, ""
    exit_code = data.get("exit_code")
    error_field = data.get("error")
    if (exit_code is not None and exit_code != 0) or error_field:
        message = data.get("output") or str(error_field) or "Tool returned an error"
        return True, str(message)
    return False, ""


def _as_tool_result(result: Any) -> ToolResult | None:
    """Parse a hermes tool result as a ``ToolResult``, or None when it isn't one.

    ``success`` is the discriminator: hermes built-ins that predate the
    ToolResult shape return their own JSON (``{"output": ..., "exit_code": ...}``)
    and must be left alone.
    """
    try:
        d = json.loads(result) if isinstance(result, str) else None
        if isinstance(d, dict) and "success" in d:
            return ToolResult.model_validate(d)
    except Exception:
        return None
    return None


def _extract_last_reasoning(conversation_history: Any) -> str:
    """Return the current iteration's reasoning trace, or ``""`` when absent.

    Walks ``conversation_history`` backwards until it finds the
    *latest* assistant message (or hits the user message that started
    the turn), and inspects only that one for a reasoning trace from
    any of these sources (in priority order):

    1. Explicit ``reasoning`` / ``reasoning_content`` field on the
       assistant message (DeepSeek, Qwen3 with vLLM reasoning parser,
       Moonshot, etc).
    2. Inline ``<think>…</think>`` / ``<thinking>…</thinking>`` blocks
       in ``content`` (Qwen3 default; provider passes the raw text
       through without stripping the tags).
    3. ``content`` text alongside ``tool_calls`` — the LLM's narrative
       about what it's about to do is the de-facto thought for that
       step (Qwen3 default chat-completions mode).

    Returns ``""`` when the latest assistant message has none of those.
    We deliberately do not fall back to earlier messages: their
    narratives belong to prior iterations and re-emitting them would
    show the same thought repeatedly in the UI.
    """
    if not conversation_history:
        return ""

    latest_assistant: dict[str, Any] | None = None
    for msg in reversed(list(conversation_history)):
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        if role == "user":
            break
        if role == "assistant":
            latest_assistant = msg
            break

    if latest_assistant is None:
        return ""

    # 1. Structured reasoning field.
    reasoning = latest_assistant.get("reasoning") or latest_assistant.get("reasoning_content")
    if isinstance(reasoning, str) and reasoning.strip():
        return reasoning
    if isinstance(reasoning, list):
        parts: list[str] = []
        for item in reasoning:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text") or item.get("thinking") or ""
                if text:
                    parts.append(str(text))
        if parts:
            return "\n".join(parts)

    content = latest_assistant.get("content")
    if not isinstance(content, str) or not content.strip():
        return ""

    # 2. Inline <think> / <thinking> tags.
    import re

    for pattern in (
        r"<think>(.*?)</think>",
        r"<thinking>(.*?)</thinking>",
        r"<thought>(.*?)</thought>",
        r"<reasoning>(.*?)</reasoning>",
    ):
        matches = re.findall(pattern, content, flags=re.DOTALL | re.IGNORECASE)
        if matches:
            stripped = [m.strip() for m in matches if m.strip()]
            if stripped:
                return "\n".join(stripped)

    # 3. Narrative content alongside tool_calls.
    if latest_assistant.get("tool_calls"):
        return content.strip()

    return ""


# ---------------------------------------------------------------------------
# Hermes runtime lookups — read-only access to gateway internals
# ---------------------------------------------------------------------------


def _runner_ref_or_none() -> Any:
    """Return the live ``GatewayRunner`` instance, or None."""
    try:
        from gateway.run import _gateway_runner_ref
    except Exception:
        return None
    return _gateway_runner_ref() if _gateway_runner_ref else None


def _chat_id_for_session_scan(session_id: str) -> str | None:
    """Scan _running_agents for the agent whose session_id matches *session_id*.

    Used only by the on_session_start hook, which fires after the agent is
    registered in _running_agents, so the exact attribute match is reliable.
    """
    runner = _runner_ref_or_none()
    running: dict[str, Any] = getattr(runner, "_running_agents", None) or {}
    for a in running.values():
        if getattr(a, "session_id", None) == session_id:
            chat_id = getattr(a, "_chat_id", None)
            return str(chat_id) if chat_id is not None else None
    return None


def _chat_id_and_thought_for_task(task_id: str) -> tuple[str | None, str]:
    """Return ``(chat_id, current_thought)`` for the AIAgent driving *task_id*.

    The thought is the LLM's current-step narrative, read from
    ``agent._session_messages`` via ``_extract_last_reasoning``. hermes
    appends the assistant message (content + tool_calls) to that list
    synchronously *before* it fires the ``pre_tool_call`` hook, so the
    narrative is reliably present when this runs — no cross-loop race,
    unlike the old interim-callback stash.

    Returns ``(None, "")`` when no agent matches.
    """
    runner = _runner_ref_or_none()
    if runner is None:
        return None, ""
    running = getattr(runner, "_running_agents", None) or {}
    for agent in running.values():
        if getattr(agent, "_current_task_id", None) != task_id:
            continue
        chat_id = getattr(agent, "_chat_id", None)
        # ``_session_messages`` is hermes' on-agent message list, mutated in
        # place as the turn progresses.
        thought = _extract_last_reasoning(getattr(agent, "_session_messages", None))
        logger.debug(
            "platform-gateway: task_id=%s chat_id=%s thought_chars=%d",
            task_id,
            chat_id,
            len(thought),
        )
        return chat_id, thought
    return None, ""


# ---------------------------------------------------------------------------
# Async dispatch — hooks fire on hermes' worker thread; outbound sends
# must reach the relay client's loop or websockets raises about loop
# affinity. ``_schedule_on_gateway_loop`` is the cross-loop bridge.
# ---------------------------------------------------------------------------


def _schedule_on_gateway_loop(coro: Any, *, label: str) -> None:
    """Fire-and-forget schedule of *coro* on the gateway's event loop."""
    runner = _runner_ref_or_none()
    gateway_loop = getattr(runner, "_gateway_loop", None) if runner else None
    if gateway_loop is not None and not gateway_loop.is_closed():
        try:
            asyncio.run_coroutine_threadsafe(coro, gateway_loop)
            return
        except Exception as exc:
            logger.warning("platform-gateway: schedule %s failed - %s", label, exc)
    # Same-loop fallback for tests / synchronous contexts.
    try:
        asyncio.get_event_loop().create_task(coro)
    except Exception as exc:
        logger.warning("platform-gateway: fallback %s dispatch failed - %s", label, exc)


# ---------------------------------------------------------------------------
# Refresh loop and tool handler factory — module-level so tests can stub
# the daemon thread and so ``RefreshLoop._refresh`` can rebuild handlers
# when new canvas tools appear.
# ---------------------------------------------------------------------------


def _start_refresh_loop(**kwargs: Any) -> None:
    """Indirection so tests can stub the daemon thread."""
    from domyn.agents.integrations.hermes_platform_gateway.client import RefreshLoop

    RefreshLoop(**kwargs).start()


def _build_config_syncer(**kwargs: Any) -> Any:
    """Indirection so tests can stub the syncer (and its network call)."""
    from domyn.agents.integrations.hermes_platform_gateway.client import ConfigSyncer

    return ConfigSyncer(**kwargs)


def _sync_config_at_boot(syncer: Any) -> None:
    """Reconcile approval mode once at register, before any turn is served.

    Unconditional (not gated on the tool-refresh interval): without it the pod
    would serve its first turn under whatever mode is on disk, and the per-turn
    sync in ``_handle_agent_start`` would only correct it from then on. Uses the
    pod's spawn-time configuration id — later turns pass their own.
    """
    syncer.sync_blocking()


def _make_tool_handler(
    adapter_slot: list[Any], tool_name: str, timeout: float
) -> Callable[..., Any]:
    """Build the async handler that bridges hermes' tool registry to the adapter.

    The registry only passes ``task_id`` to platform tool handlers (not
    ``parent_agent``), so we look up the chat_id via the per-task stash
    that ``pre_tool_call`` populated.
    """

    async def handler(args: dict, **kwargs: Any) -> str:
        adapter = adapter_slot[0]
        if adapter is None:
            return json.dumps({"error": "platform-gateway: adapter not ready"})
        task_id = kwargs.get("task_id") or ""
        chat_id = adapter._chat_id_by_task.get(task_id)
        if not chat_id:
            return json.dumps({"error": "platform-gateway: no chat_id for task_id"})
        session_key = adapter.session_key_for_chat(chat_id)
        thought = adapter.thought_for_task(task_id)
        return await adapter.call_tool(
            session_key=session_key,
            tool_name=tool_name,
            args=args,
            thought=thought,
            timeout=timeout,
        )

    return handler


# ---------------------------------------------------------------------------
# Plugin entry point
# ---------------------------------------------------------------------------


def register(ctx: Any) -> None:
    """Plugin entry — called once by hermes' plugin loader at startup."""
    from domyn.agents.integrations.hermes_platform_gateway.client import (
        _PLATFORM_TOOLSET,
        fetch_tools,
        hermes_tool_names,
        rename_colliding_tools,
    )

    missing = [v for v in _REQUIRED if not os.environ.get(v)]
    if missing:
        logger.warning(
            "platform-gateway: skipping registration — missing env vars: %s",
            ", ".join(missing),
        )
        return

    settings = PlatformSettings.from_env()
    configuration_id = settings.configuration_id
    timeout = float(os.environ.get("PLATFORM_TOOL_TIMEOUT", "120"))
    refresh_interval = float(os.environ.get("PLATFORM_TOOL_REFRESH_INTERVAL", "60"))
    agent_name = os.environ.get("DOMYN_AGENT_NAME")

    channel_id = settings.channel_id

    try:
        raw_tools = fetch_tools(settings, agent_name=agent_name)
    except Exception as exc:
        logger.warning("platform-gateway: could not fetch tools - %s", type(exc).__name__)
        logger.debug("platform-gateway: fetch tools detail: %s", exc)
        raw_tools = []

    # Check if any domyn tool collides with a hermes tool name
    reserved_tool_names = hermes_tool_names()
    raw_tools, tool_rename_mapping = rename_colliding_tools(raw_tools, reserved_tool_names)

    # Adapter is built lazily by the factory so the gateway controls
    # its lifecycle. We close over a one-slot list so the tool
    # handlers (registered now) can find the live adapter once the
    # factory has been called.
    adapter_slot: list[Any] = [None]

    try:
        # This pod's relay connection has never carried the configuration id.
        ws_url, headers = relay_connection(replace(settings, configuration_id=None))
    except Exception as exc:
        logger.warning(
            "platform-gateway: skipping registration — no relay connection - %s", type(exc).__name__
        )
        logger.debug("platform-gateway: relay connection detail: %s", exc)
        return

    # One syncer per pod, shared by the boot sync below and the per-turn sync in
    # the adapter, so both share the edge-trigger memo and can't double-write
    # config.yaml. ``default_config_id`` is the spawn-time id, used when a turn
    # carries none.
    config_syncer = _build_config_syncer(
        settings=settings,
        default_config_id=configuration_id,
    )
    _sync_config_at_boot(config_syncer)

    def _factory(config: Any) -> Any:
        from domyn.agents.integrations.hermes_platform_gateway.adapter import DomynPlatformAdapter
        from domyn.agents.integrations.hermes_platform_gateway.relay_client import DomynRelayClient

        def relay_factory(
            on_event: Callable[[Any, dict[str, Any] | None, str | None], Any],
        ) -> Any:
            return DomynRelayClient(ws_url=ws_url, headers=headers, on_event=on_event)

        adapter = DomynPlatformAdapter(
            config=config,
            channel_id=channel_id,
            relay_client_factory=relay_factory,
            tool_rename_mapping=tool_rename_mapping,
            config_syncer=config_syncer,
        )
        adapter_slot[0] = adapter
        return adapter

    def _check() -> bool:
        return all(os.environ.get(v) for v in _REQUIRED)

    ctx.register_platform(
        name="domyn",
        label="Domyn",
        adapter_factory=_factory,
        check_fn=_check,
        required_env=list(_REQUIRED),
        allowed_users_env="DOMYN_ALLOWED_USERS",
        allow_all_env="DOMYN_ALLOW_ALL_USERS",
    )

    # --- Tool registration ---
    # The live set of tool names hermes has registered for us. Handed to RefreshLoop
    # below, which adopts it and mutates it in place as tools come and go — never
    # copy it, or the tool hooks' ``is_platform`` check goes stale the first time
    # the tool list changes and platform tools get mistaken for hermes built-ins.
    registered_names: set[str] = set()
    for tool_def in raw_tools:
        name = tool_def.get("name")
        if not name:
            continue
        try:
            schema = convert_schema(tool_def)
        except Exception as exc:
            logger.warning(
                "platform-gateway: skipping tool '%s' - schema error: %s",
                name,
                type(exc).__name__,
            )
            logger.debug("platform-gateway: schema error detail for '%s': %s", name, exc)
            continue
        ctx.register_tool(
            name=name,
            toolset=_PLATFORM_TOOLSET,
            schema=schema,
            handler=_make_tool_handler(adapter_slot, name, timeout),
            is_async=True,
        )
        registered_names.add(name)
    logger.info(
        "platform-gateway: registered domyn adapter with %d platform tool(s): %s",
        len(registered_names),
        sorted(registered_names),
    )

    # --- Hooks ---
    # pre_tool_call bridges the AIAgent → tool-handler gap: hermes' tool
    # registry only passes ``task_id`` to the handler (not ``parent_agent``),
    # so we scan _running_agents for the matching agent and stash its
    # chat_id (+ current reasoning) under task_id. post_tool_call drops
    # the stash and emits a visibility TOOL_END for built-in tools.

    def _on_pre_tool_call(
        tool_name: str = "",
        args: Any = None,
        task_id: str = "",
        **_: Any,
    ) -> None:
        adapter = adapter_slot[0]
        if adapter is None or not task_id:
            return
        chat_id, thought = _chat_id_and_thought_for_task(task_id)
        if not thought:
            if args:
                cropped = (
                    {k: (str(v)[:10] + "..." if len(str(v)) > 10 else v) for k, v in args.items()}
                    if isinstance(args, dict)
                    else args
                )
                thought = f"Executing {tool_name} with args {cropped}..."
            else:
                thought = f"Executing {tool_name}..."
        if not chat_id:
            # _running_agents may not expose the agent via _current_task_id
            # (hermes version mismatch or attribute not set). Fall back to the
            # adapter's turn tracking — same approach as _resolve_chat_id.
            active_turns = list(adapter._turn_by_session.values())
            if len(active_turns) == 1:
                chat_id = getattr(active_turns[0], "conversation_id", None)
        if not chat_id:
            return

        is_platform = tool_name in registered_names
        logger.debug(
            "platform-gateway: pre_tool_call %s task_id=%s chat_id=%s is_platform=%s thought_chars=%d",
            tool_name,
            task_id,
            chat_id,
            is_platform,
            len(thought),
        )

        # Always stash chat_id (+ thought, possibly empty) — post_tool_call
        # reads this back regardless of branch. Empty thought rides on the
        # outbound ToolAction as null rather than a placeholder.
        adapter.record_task_chat(
            task_id=task_id,
            chat_id=chat_id,
            thought=thought,
        )

        if is_platform:
            # Real platform tool: the canonical TOOL_START goes out
            # from adapter.call_tool with a pending future. Done here.
            return

        # Built-in hermes tool: visibility-only TOOL_START. Generate a
        # fresh call_id, stash it so post_tool_call can pair the TOOL_END.
        call_id = f"visibility-{uuid.uuid4()}"
        adapter.record_visibility_call(task_id=task_id, call_id=call_id)
        _schedule_on_gateway_loop(
            adapter.emit_visibility_tool_start(
                chat_id=chat_id,
                tool_name=tool_name,
                args=args if isinstance(args, dict) else {},
                thought=thought or None,
                call_id=call_id,
            ),
            label=f"visibility TOOL_START {tool_name}",
        )

    def _on_post_tool_call(
        tool_name: str = "",
        task_id: str = "",
        result: Any = None,
        error: Any = None,
        **_: Any,
    ) -> None:
        adapter = adapter_slot[0]
        if adapter is None:
            return
        chat_id = adapter._chat_id_by_task.get(task_id)
        # A top-level delegate_task runs its subagents in the background and
        # returns {"status": "dispatched", "mode": "background", ...} immediately;
        # the consolidated result re-enters as a fresh turn later. Flag the
        # session so the turn's wrap-up goes out as a non-terminal RESPONSE
        # (is_partial=True) instead of a final AGENT_END. A synchronous fallback
        # returns {"results": [...]} (no "mode") and is left as a normal turn.
        if tool_name == "delegate_task" and chat_id:
            try:
                dispatch = json.loads(result) if isinstance(result, str) else result
            except Exception:
                dispatch = None
            if (
                isinstance(dispatch, dict)
                and dispatch.get("status") == "dispatched"
                and dispatch.get("mode") == "background"
            ):
                adapter.mark_background_delegation(chat_id)
        visibility_call_id = adapter.pop_visibility_call(task_id)
        adapter.forget_task(task_id=task_id)

        # Visibility TOOL_END/TOOL_ERROR only for built-in tools that emitted a
        # matching visibility TOOL_START. Platform tools have their own
        # round-trip from the relay; don't double-emit.
        if not visibility_call_id or tool_name in registered_names:
            return
        if not chat_id:
            return
        # Detect errors both from the explicit `error` kwarg (Python exception)
        # and from a non-zero exit_code inside the result JSON (subprocess error).
        result_is_error, result_error_msg = _tool_result_is_error(result)
        # Parsed before the branch, not inside the success one: a ToolResult that
        # carries `error` routes to TOOL_ERROR, and its metadata has to ride that
        # event too — a failed call is where metadata is most worth having.
        tool_result = _as_tool_result(result)
        metadata = (tool_result.metadata or None) if tool_result is not None else None
        if error is not None or result_is_error:
            error_message = str(error) if error is not None else result_error_msg
            _schedule_on_gateway_loop(
                adapter.emit_visibility_tool_error(
                    chat_id=chat_id,
                    tool_name=tool_name,
                    call_id=visibility_call_id,
                    error_message=error_message,
                    metadata=metadata,
                ),
                label=f"visibility TOOL_ERROR {tool_name}",
            )
        else:
            emitted_content = tool_result.emitted_content or None if tool_result else None
            observation: Any = tool_result.data if tool_result is not None else result
            _schedule_on_gateway_loop(
                adapter.emit_visibility_tool_end(
                    chat_id=chat_id,
                    tool_name=tool_name,
                    call_id=visibility_call_id,
                    observation=observation,
                    emitted_content=emitted_content,
                    metadata=metadata,
                ),
                label=f"visibility TOOL_END {tool_name}",
            )

    def _on_transform_tool_result(result: str = "", **_: Any) -> str | None:
        try:
            d = json.loads(result)
            if not isinstance(d, dict) or "success" not in d:
                return None
            tr = ToolResult.model_validate(d)
            data: str = str(tr.data)
            return data
        except Exception:
            return None

    ctx.register_hook("pre_tool_call", _on_pre_tool_call)
    ctx.register_hook("post_tool_call", _on_post_tool_call)
    ctx.register_hook("transform_tool_result", _on_transform_tool_result)

    # --- LLM hooks ---
    # pre_llm_call fires once per turn before the tool-calling loop. We use it
    # to emit LLM_START and stash a call_id so post_llm_call can emit the
    # matching LLM_END.  session_id is hermes' internal opaque ID — we resolve
    # it to a domyn chat_id by scanning _running_agents (same pattern as
    # _chat_id_and_thought_for_task but keyed on AIAgent.session_id).

    def _resolve_chat_id(session_id: str) -> str | None:
        """Resolve a domyn chat_id (= conversation_id) for *session_id*.

        Fast path: on_session_start stashes session_id → chat_id before the
        first pre_llm_call fires, so this is normally a plain dict lookup.

        Safety net: if on_session_start hasn't fired (older hermes, race on
        very first call), fall back to _turn_by_session when exactly one
        conversation is active on this channel.
        """
        adapter = adapter_slot[0]
        if adapter is None:
            return None
        chat_id = adapter._chat_id_by_hermes_session.get(session_id)
        if chat_id:
            return chat_id
        active_turns = list(adapter._turn_by_session.values())
        if len(active_turns) == 1:
            return getattr(active_turns[0], "conversation_id", None)
        logger.debug(
            "platform-gateway: _resolve_chat_id found %d active turns for session %s, skipping",
            len(active_turns),
            session_id,
        )
        return None

    def _on_pre_llm_call(
        session_id: str = "",
        model: str = "",
        **_: Any,
    ) -> None:
        adapter = adapter_slot[0]
        if adapter is None or not session_id:
            return
        chat_id = _resolve_chat_id(session_id)
        if not chat_id:
            return
        call_id = f"llm-{uuid.uuid4()}"
        adapter.record_llm_call(session_id=session_id, chat_id=chat_id, call_id=call_id)
        _schedule_on_gateway_loop(
            adapter.emit_llm_start(chat_id=chat_id, model=model, call_id=call_id),
            label=f"LLM_START {model}",
        )

    def _on_post_llm_call(
        session_id: str = "",
        model: str = "",
        assistant_response: str = "",
        **_: Any,
    ) -> None:
        adapter = adapter_slot[0]
        if adapter is None:
            return
        entry = adapter.pop_llm_call(session_id)
        if entry is None:
            return
        chat_id, call_id = entry
        _schedule_on_gateway_loop(
            adapter.emit_llm_end(
                chat_id=chat_id,
                model=model,
                call_id=call_id,
                assistant_response=assistant_response,
            ),
            label=f"LLM_END {model}",
        )

    ctx.register_hook("pre_llm_call", _on_pre_llm_call)
    ctx.register_hook("post_llm_call", _on_post_llm_call)

    def _on_session_start(
        session_id: str = "",
        platform: str = "",
        **_: Any,
    ) -> None:
        if platform != "domyn":
            return
        adapter = adapter_slot[0]
        if adapter is None or not session_id:
            return
        chat_id = _chat_id_for_session_scan(session_id)
        if chat_id:
            adapter.record_hermes_session(session_id=session_id, chat_id=chat_id)

    ctx.register_hook("on_session_start", _on_session_start)

    # --- Tool list refresh ---
    if refresh_interval > 0:
        _start_refresh_loop(
            ctx=ctx,
            handler_factory=lambda nm: _make_tool_handler(adapter_slot, nm, timeout),
            settings=settings,
            interval=refresh_interval,
            agent_name=agent_name,
            initial_names=registered_names,
            tool_rename_mapping=tool_rename_mapping,
        )
