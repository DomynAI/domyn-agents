"""PlatformCallbackHandler: langchain AsyncCallbackHandler that emits BaseEvent over a caller-provided closure."""

from __future__ import annotations

import functools
import time
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

from langchain_core.callbacks.base import AsyncCallbackHandler
from pydantic import TypeAdapter, ValidationError

from domyn.agents.core import BaseEvent, ExecutionEventType, Part, ToolAction
from domyn.agents.core.utils import extract_parts
from domyn.agents.integrations.langgraph.domyn_platform import PLATFORM_TOOL_TAG
from domyn.agents.integrations.langgraph.run_state import RunRegistry, RunState
from domyn.agents.logger import get_logger
from domyn.agents.tools import ToolResult

EmitFn = Callable[[BaseEvent], Awaitable[None]]
logger = get_logger()

# Serializes candidate metadata through the same annotation ``BaseEvent.metadata``
# carries, so the check below agrees with what pydantic will do on the wire
# rather than approximating it (``json.dumps`` would reject a datetime the field
# serializes fine).
_METADATA_ADAPTER: TypeAdapter[dict[str, Any]] = TypeAdapter(dict[str, Any])


def _command_messages(output: Any) -> list[Any]:
    """The messages carried by a LangGraph ``Command``'s state update, if any.

    A tool returning a ``Command`` never reaches ``_format_output``'s
    ToolMessage wrapping — ``Command`` is a ``ToolOutputMixin``, which
    langchain returns verbatim — so ``on_tool_end`` receives the ``Command``
    itself and the ToolMessage inside it has to be dug out here.

    Normalizing the update is delegated to ``Command._update_as_tuples()``:
    private, but it is what LangGraph itself applies the update with, so it
    covers every shape the graph accepts — a mapping, a sequence of
    ``(key, value)`` pairs, and an annotated state object (a pydantic model or
    dataclass) whose ``messages`` field would otherwise be invisible here.

    The presence check is duck-typed rather than an ``isinstance(output,
    Command)``: langgraph is an optional extra, and a tool is free to return an
    object of its own that is not a Command at all.
    """
    as_tuples = getattr(output, "_update_as_tuples", None)
    if not callable(as_tuples):
        return []
    messages: Any = None
    for key, value in as_tuples():
        if key == "messages":
            messages = value
    if messages is None:
        return []
    return list(messages) if isinstance(messages, list | tuple) else [messages]


def _wire_safe_metadata(metadata: dict[str, Any] | None, *, tool_name: str) -> dict[str, Any]:
    """``metadata`` minus anything that cannot be serialized onto the wire.

    A tool's metadata is free-form, so without this one unserializable object in
    it costs the whole event rather than just the metadata: the runtime
    serializes the BaseEvent outside its own try/except, and the error surfaces
    inside a ``@_safe`` callback, which swallows it. The event is the valuable
    part — drop only the keys that cannot travel, and say which.

    Any failure counts, not just ``PydanticSerializationError``: the question
    being asked is "does this survive the trip", and a custom serializer raising
    something else answers it the same way.
    """
    if not metadata:
        return {}
    try:
        _METADATA_ADAPTER.dump_json(metadata)
        return metadata
    except Exception:
        pass
    safe: dict[str, Any] = {}
    dropped: list[str] = []
    for key, value in metadata.items():
        try:
            _METADATA_ADAPTER.dump_json({key: value})
            safe[key] = value
        except Exception:
            dropped.append(key)
    # Key names only: the values are tool output, which belongs at debug.
    logger.warning(f"dropping unserializable metadata from tool {tool_name}: {sorted(dropped)}")
    return safe


def _unwrap_tool_message(output: Any) -> Any | None:
    """Return the ``ToolMessage`` a tool output carries, or None.

    Handles a bare ToolMessage and a ``Command`` wrapping one (the last
    message wins — with parallel updates the tool's own result is the one it
    appended last).
    """
    try:
        from langchain_core.messages import ToolMessage
    except ImportError:
        return None
    if isinstance(output, ToolMessage):
        return output
    for message in reversed(_command_messages(output)):
        if isinstance(message, ToolMessage):
            return message
    return None


def _parse_tool_result_dict(raw: dict[str, Any], *, require_data: bool) -> ToolResult | None:
    """Validate ``raw`` as a ToolResult (no extra fields, ``success`` required).

    ``require_data`` guards the channels where a dict is merely *found* — a
    plain dict return, or JSON parsed out of a tool's content — and any dict
    there could be domain data that happens to validate. An ``artifact`` is the
    opposite case: it is the field a tool fills in on purpose to hand us a
    structured result, so a metadata-only result is legitimate there and must
    not be dropped for having no ``data``.
    """
    if require_data and "data" not in raw:
        return None
    try:
        return ToolResult.model_validate(raw)
    except ValidationError:
        return None


def _message_text(message: Any) -> str:
    """The message's content as plain text.

    ``.text`` (the property — calling it is deprecated in langchain-core 1.0 and
    goes away in 2.0) rather than ``.content``, which may be a list of content
    blocks: stringifying that list would put a Python repr on the wire.
    """
    return str(message.text)


def _try_parse_tool_result(output: Any, *, message: Any | None = None) -> ToolResult | None:
    """Try to parse a tool output as a ToolResult strictly (no extra fields).

    Returns a ToolResult on success, None if output doesn't conform.
    Handles a ToolMessage — bare or wrapped in a LangGraph ``Command`` — by
    trying artifact (dict) then content (JSON string). ``message`` is the
    already-unwrapped ToolMessage when the caller has one, so a single output is
    not unwrapped twice.
    """
    if isinstance(output, ToolResult):
        return output
    if message is None:
        message = _unwrap_tool_message(output)
    if message is not None:
        if isinstance(message.artifact, dict):
            parsed = _parse_tool_result_dict(message.artifact, require_data=False)
            if parsed is not None:
                return parsed
        # An artifact that is not a ToolResult is domain data, not a failure:
        # fall through to the content the same as a tool that sets no artifact.
        output = _message_text(message)
    raw: dict[str, Any] | None = None
    if isinstance(output, dict):
        raw = output
    elif isinstance(output, str):
        import json

        try:
            parsed_json = json.loads(output)
            if isinstance(parsed_json, dict):
                raw = parsed_json
        except (json.JSONDecodeError, ValueError):
            return None
    if raw is None:
        return None
    return _parse_tool_result_dict(raw, require_data=True)


def _safe(method: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator: ensures no exception escapes an async callback method."""

    method_name = getattr(method, "__name__", repr(method))

    @functools.wraps(method)
    async def wrapper(self: PlatformCallbackHandler, *args: Any, **kwargs: Any) -> None:
        try:
            await method(self, *args, **kwargs)
        except Exception as exc:
            logger.warning(f"callback {method_name} raised; swallowing: {type(exc).__name__}")
            logger.debug(f"callback {method_name} detail: {exc}", exc_info=True)

    return wrapper


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def _is_part_list(value: Any) -> bool:
    """True when ``value`` is a non-empty list of ``Part`` objects."""
    return isinstance(value, list) and bool(value) and all(isinstance(p, Part) for p in value)


def _extract_thought_from_response(response: Any) -> str | None:
    """Mirror tool_use_thought._extract_thought for LangChain LLMResult.

    Priority (first non-empty wins):
    1. ``reasoning_content`` in additional_kwargs  — DeepSeek R1, Kimi, etc.
    2. ``thinking`` block in content list          — Anthropic extended thinking.
    3. Plain-text ``content`` when tool calls are also present — preamble text
       solicited by the system prompt (e.g. "Let me think step by step...").
    4. Synthetic fallback: human-readable tool call lines when no thought at all.
    """
    generations = (
        response.get("generations")
        if isinstance(response, dict)
        else getattr(response, "generations", None)
    )
    if not generations:
        return None

    for batch in generations:
        if not isinstance(batch, list):
            continue
        for gen in batch:
            if isinstance(gen, dict):
                continue
            message = getattr(gen, "message", None)
            if message is None:
                continue

            # 1. reasoning_content in additional_kwargs
            extra = getattr(message, "additional_kwargs", {}) or {}
            rc = extra.get("reasoning_content")
            if rc and isinstance(rc, str):
                return rc

            content = getattr(message, "content", None)
            tool_calls = getattr(message, "tool_calls", None)

            # 2. thinking block in structured content list
            if isinstance(content, list):
                for block in content:
                    block_type = block.get("type") if isinstance(block, dict) else None
                    if block_type in ("thinking", "reasoning"):
                        chunk = (
                            block.get("thinking") or block.get("reasoning") or block.get("text", "")
                        )
                        if chunk:
                            return chunk

            # 3. plain-text preamble when tool calls are present
            if tool_calls and isinstance(content, str) and content.strip():
                return content.strip()

            # 4. synthetic fallback
            if tool_calls:
                lines = []
                for tc in tool_calls:
                    name = (
                        tc.get("name", "tool")
                        if isinstance(tc, dict)
                        else getattr(tc, "name", "tool")
                    )
                    args = tc.get("args", {}) if isinstance(tc, dict) else getattr(tc, "args", {})
                    params_str = ", ".join(f"{k}={v}" for k, v in (args or {}).items())
                    lines.append(
                        f"Calling tool {name} with parameters: {params_str}"
                        if params_str
                        else f"Calling tool {name}"
                    )
                if lines:
                    return "\n".join(lines)

    return None


def _first_tool_call_id(response: Any) -> str | None:
    """Return the id of the first tool call in a LangChain LLMResult, if any.

    The turn-level preamble/reasoning is attached to this tool call (mirroring
    the core ``tool_use_thought`` planner, where only the first tool call falls
    back to the response-level thought).
    """
    generations = (
        response.get("generations")
        if isinstance(response, dict)
        else getattr(response, "generations", None)
    )
    if not generations:
        return None

    for batch in generations:
        if not isinstance(batch, list):
            continue
        for gen in batch:
            if isinstance(gen, dict):
                continue
            message = getattr(gen, "message", None)
            tool_calls = getattr(message, "tool_calls", None) if message is not None else None
            if not tool_calls:
                continue
            for tc in tool_calls:
                tc_id = tc.get("id") if isinstance(tc, dict) else getattr(tc, "id", None)
                if tc_id:
                    return str(tc_id)
    return None


def _extract_llm_text(response: Any) -> str:
    """Extract plain response text from a LangChain LLMResult.

    Skips thinking/reasoning blocks — those are handled by
    ``_extract_thought_from_response``.
    """
    generations = (
        response.get("generations")
        if isinstance(response, dict)
        else getattr(response, "generations", None)
    )
    if not generations:
        return ""

    text_parts: list[str] = []

    for batch in generations:
        if not isinstance(batch, list):
            continue
        for gen in batch:
            if isinstance(gen, dict):
                if gen.get("text"):
                    text_parts.append(gen["text"])
                continue

            message = getattr(gen, "message", None)
            content = getattr(message, "content", None) if message is not None else None

            if isinstance(content, list):
                for block in content:
                    block_type = block.get("type") if isinstance(block, dict) else None
                    if block_type == "text":
                        chunk = block.get("text", "")
                        if chunk:
                            text_parts.append(chunk)
            else:
                raw_text = getattr(gen, "text", None)
                if raw_text:
                    text_parts.append(raw_text)

    return "\n".join(text_parts)


class PlatformCallbackHandler(AsyncCallbackHandler):
    """Langchain async callback handler that emits ``BaseEvent`` directly.

    Each callback maps to a single :class:`ExecutionEventType` and constructs
    the BaseEvent inline; the caller-provided ``emit`` closure ships it to
    the wire.
    """

    def __init__(
        self,
        *,
        project_id: str,  # Author in BaseEvent
        emit: EmitFn,
        session_id: str | None = None,
        conversation_id: str | None = None,
        turn_id: str | None = None,
    ) -> None:
        super().__init__()
        self._project_id = project_id
        self._emit_fn = emit
        self._default_session_id = session_id
        # The conversation_id of the AGENT_START this handler serves. Authoritative
        # for every emitted event — not read from (unreliably propagated) run metadata.
        self._conversation_id = conversation_id
        # The turn_id of the AGENT_START this handler serves. Stamped onto every
        # emitted event (same rationale as conversation_id) so the platform can
        # correlate the whole trace with the turn.
        self._turn_id = turn_id
        self._registry = RunRegistry()
        # Tool run_ids tagged as platform tools. The bound adapter (``LangGraphAdapter``,
        # or the legacy ``Runtime``) owns their wire-level TOOL_START/TOOL_END, so we
        # skip registry tracking and emission on both ends.
        self._platform_tool_runs: set[UUID] = set()
        # Tool-name lookup keyed by run_id, populated at on_tool_start so
        # on_tool_end can rebuild a ToolAction with the matching name.
        self._tool_names: dict[UUID, str] = {}
        # LangGraph node-name lookup: metadata is only passed to on_chain_start,
        # not on_chain_end, so we store run_id → node_name here.
        self._node_names: dict[UUID, tuple[str, dict[str, Any] | None]] = {}
        # Reverse lookup: node_name → run_id. LangGraph propagates langgraph_node
        # through metadata to all child spans (LLM, tools, sub-chains), so we can
        # detect agent-style wrappers even when the LLM's direct parent_run_id is
        # an intermediate chain rather than the node run_id itself.
        self._node_run_by_name: dict[str, UUID] = {}
        # Chains that had child LLM/tool spans — treated as agent-style wrappers;
        # we suppress node-level TOOL_START/TOOL_END to avoid duplicate events.
        self._node_agent_children: set[UUID] = set()
        # Pending LLM reasoning/preamble keyed by the originating tool-call id.
        # The platform renders thoughts from action.thought on TOOL_START, so we
        # stash the turn's preamble against the tool call that carries it (set in
        # on_llm_end) and pop it when that tool starts. Keying by tool_call_id —
        # rather than a single mutable slot — avoids cross-attribution across
        # parallel branches and stale carry-over between turns.
        self._pending_thoughts: dict[str, str] = {}

    async def aclose(self) -> None:
        return None

    # ---- helpers ----

    async def _emit(self, event: BaseEvent) -> None:
        await self._emit_fn(event)

    def _resolve_session(self, metadata: dict[str, Any] | None) -> str | None:
        if metadata and metadata.get("session_id"):
            return metadata["session_id"]
        return self._default_session_id

    def _new_event(
        self,
        *,
        event_type: ExecutionEventType,
        metadata: dict[str, Any] | None = None,
        event_metadata: dict[str, Any] | None = None,
        content: list[Part] | None = None,
        action: ToolAction | None = None,
        error_code: str | None = None,
        error_message: str | None = None,
    ) -> BaseEvent:
        """Build one BaseEvent.

        The two metadata arguments are not interchangeable:

        - ``metadata`` is the *langchain run* metadata (``RunState.metadata``),
          read only to resolve the session id. It must not be put on the wire:
          the runtime folds the caller's ``run_config`` into it (PII, gated by
          ``VisibilityScope``) alongside the invocation ids, and langgraph adds
          its own node/step/checkpoint plumbing to every span.
        - ``event_metadata`` is what the event carries. It comes from a tool's
          own ``ToolResult.metadata`` — the same contract the core runner
          honours when it builds a TOOL_END (see ``Runner._execute_tool``).
        """
        return BaseEvent(
            event_type=event_type,
            author=self._project_id,
            event_id=str(uuid4()),
            timestamp=datetime.now(UTC).replace(tzinfo=None),
            interaction_id=self._resolve_session(metadata),
            content=content or [],
            action=action,
            error_code=error_code,
            error_message=error_message,
            conversation_id=self._conversation_id,
            turn_id=self._turn_id,
            metadata=event_metadata or {},
        )

    async def _maybe_emit_response(self, state: RunState) -> None:
        """If ``state`` is the trace root, emit a terminal RESPONSE event and clean up the subtree."""
        if state.parent_run_id is not None:
            return
        # Drop the whole subtree from the registry; we no longer surface per-step snapshots.
        self._registry.pop_trajectory(state.run_id)
        # If the state output is already a list of Parts, emit them verbatim.
        if _is_part_list(state.output):
            await self._emit(
                self._new_event(
                    event_type=ExecutionEventType.RESPONSE,
                    metadata=state.metadata,
                    content=list(state.output),
                )
            )
        else:
            if state.span_type == "llm":
                text = _extract_llm_text(state.output)
                thought = _extract_thought_from_response(state.output)
            else:
                text, thought = "", None
            if not text:
                text = _stringify(state.output)
            content: list[Part] = []
            if thought:
                content.append(Part(thought=thought))
            if text:
                content.extend(extract_parts(text))
            await self._emit(
                self._new_event(
                    event_type=ExecutionEventType.RESPONSE,
                    metadata=state.metadata,
                    content=content,
                )
            )

    # ---- chain callbacks (LangGraph nodes) ----

    @_safe
    async def on_chain_start(
        self,
        serialized: dict[str, Any] | None,
        inputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        node_name = (metadata or {}).get("langgraph_node")
        if not node_name:
            return
        # create_agent() graphs set ls_integration="langchain_create_agent" in config metadata.
        # Their model/tools nodes use trace=False on RunnableCallable, so LLM/tool callbacks
        # never fire — we detect agent-style graphs via this metadata key instead.
        if (metadata or {}).get("ls_integration") == "langchain_create_agent":
            return
        # LangGraph fires on_chain_start twice per node (outer pregel wrapper + inner function).
        # Only register the first (outer) call; skip inner sub-chains for the same node name.
        if node_name in self._node_run_by_name:
            return
        self._node_names[run_id] = (node_name, metadata)
        self._node_run_by_name[node_name] = run_id

    @_safe
    async def on_chain_end(
        self,
        outputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        # Trace-root chain whose output is already a list of Parts: surface it as
        # the terminal RESPONSE. The runnable's final output never flows through
        # an LLM/agent-finish callback (the LLM run is nested), so this is the
        # only place a Part-list result reaches _maybe_emit_response.
        if parent_run_id is None and _is_part_list(outputs):
            await self._maybe_emit_response(
                RunState(
                    run_id=run_id,
                    parent_run_id=None,
                    span_type="chain",
                    start_time=time.time(),
                    output=outputs,
                )
            )
            return
        entry = self._node_names.pop(run_id, None)
        if not entry:
            return
        node_name, metadata = entry
        self._node_run_by_name.pop(node_name, None)

        # If this node spawned real LLM or tool callbacks, they already emitted events — skip.
        is_agent_node = run_id in self._node_agent_children
        self._node_agent_children.discard(run_id)
        if is_agent_node:
            return

        # --- deterministic node: emit TOOL_START + TOOL_END with domain output ---
        observation = _stringify(
            outputs.get("result")
            if isinstance(outputs, dict) and outputs.get("result")
            else outputs
        )
        await self._emit(
            self._new_event(
                event_type=ExecutionEventType.TOOL_START,
                metadata=metadata,
                content=[Part(text=f"Running node: {node_name}")],
                action=ToolAction(name=node_name, parameters=None, call_id=str(run_id)),
            )
        )
        await self._emit(
            self._new_event(
                event_type=ExecutionEventType.TOOL_END,
                metadata=metadata,
                action=ToolAction(
                    name=node_name,
                    parameters=None,
                    call_id=str(run_id),
                    observation=observation,
                ),
            )
        )

    # ---- LLM callbacks ----

    @_safe
    async def on_llm_start(
        self,
        serialized: dict[str, Any] | None,
        prompts: list[str],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        node_name = (metadata or {}).get("langgraph_node")
        if node_name and node_name in self._node_run_by_name:
            self._node_agent_children.add(self._node_run_by_name[node_name])
        self._registry.start(
            run_id=run_id,
            parent_run_id=parent_run_id,
            span_type="llm",
            start_time=time.time(),
            inputs={"prompts": prompts},
            metadata=metadata or {},
        )
        await self._emit(
            self._new_event(
                event_type=ExecutionEventType.LLM_START,
                metadata=metadata,
                content=[Part(text="Invoking LLM for next action")],
            )
        )

    @_safe
    async def on_chat_model_start(
        self,
        serialized: dict[str, Any] | None,
        messages: list[list[Any]],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        prompts = [repr(msg) for group in messages for msg in group]
        await self.on_llm_start(
            serialized,
            prompts,
            run_id=run_id,
            parent_run_id=parent_run_id,
            metadata=metadata,
            **kwargs,
        )

    @_safe
    async def on_llm_end(
        self,
        response: Any,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        state, _ = self._registry.end(run_id=run_id, output=response, end_time=time.time())
        text = _extract_llm_text(response)
        thought = _extract_thought_from_response(response)
        # Stash the reasoning against the first tool call of this turn; on_tool_start
        # pops it by tool_call_id and renders it via action.thought (the platform
        # does not consume LLM_END Part(thought)).
        if thought:
            tc_id = _first_tool_call_id(response)
            if tc_id:
                self._pending_thoughts[tc_id] = thought
        content: list[Part] = []
        if thought:
            content.append(Part(thought=thought))
        if text and text.strip() != (thought or "").strip():
            content.append(Part(text=text))
        elif not thought:
            content.append(Part(text=text or "LLM call completed"))
        await self._emit(
            self._new_event(
                event_type=ExecutionEventType.LLM_END,
                metadata=state.metadata,
                content=content,
            )
        )
        await self._maybe_emit_response(state)

    @_safe
    async def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        state, _ = self._registry.error(run_id=run_id, error=error, end_time=time.time())
        await self._emit(
            self._new_event(
                event_type=ExecutionEventType.ERROR_OCCURRED,
                metadata=state.metadata if state else {},
                error_code=type(error).__name__,
                error_message=str(error),
            )
        )
        if state is not None:
            await self._maybe_emit_response(state)

    # ---- chain callbacks ----

    @_safe
    async def on_chain_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        # run_id is commonly unregistered here for create_agent()-style graphs
        # (on_chain_start early-returns for ls_integration=="langchain_create_agent"),
        # so `state` may legitimately be None - see RunRegistry.error()'s docstring.
        state, _ = self._registry.error(run_id=run_id, error=error, end_time=time.time())
        await self._emit(
            self._new_event(
                event_type=ExecutionEventType.ERROR_OCCURRED,
                metadata=state.metadata if state else {},
                error_code=type(error).__name__,
                error_message=str(error),
            )
        )
        if state is not None:
            await self._maybe_emit_response(state)

    # ---- tool callbacks ----

    @_safe
    async def on_tool_start(
        self,
        serialized: dict[str, Any] | None,
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        tags = kwargs.get("tags") or []
        node_name = (metadata or {}).get("langgraph_node")
        if PLATFORM_TOOL_TAG in tags:
            self._platform_tool_runs.add(run_id)
            if node_name and node_name in self._node_run_by_name:
                self._node_agent_children.add(self._node_run_by_name[node_name])
            return
        if node_name and node_name in self._node_run_by_name:
            self._node_agent_children.add(self._node_run_by_name[node_name])
        tool_name = (serialized or {}).get("name") or "tool"
        self._tool_names[run_id] = tool_name
        self._registry.start(
            run_id=run_id,
            parent_run_id=parent_run_id,
            span_type="tool",
            start_time=time.time(),
            inputs={"input": input_str},
            metadata=metadata or {},
        )
        parameters = kwargs.get("inputs") if isinstance(kwargs.get("inputs"), dict) else None
        text = f"Invoking tool: {tool_name}"
        if input_str:
            text = f"{text}\nInput: {input_str}"
        # Attach the preamble stashed against this tool call (by tool_call_id) as
        # the tool's thought — this is what the platform renders. Popping ensures
        # it is consumed exactly once and never bleeds onto an unrelated tool.
        tc_id = kwargs.get("tool_call_id")
        tool_thought = self._pending_thoughts.pop(tc_id, None) if tc_id else None
        await self._emit(
            self._new_event(
                event_type=ExecutionEventType.TOOL_START,
                metadata=metadata,
                content=[Part(text=text)],
                action=ToolAction(
                    name=tool_name,
                    parameters=parameters,
                    call_id=str(run_id),
                    thought=tool_thought,
                ),
            )
        )

    @_safe
    async def on_tool_end(
        self,
        output: Any,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        if run_id in self._platform_tool_runs:
            self._platform_tool_runs.discard(run_id)
            return
        state, _ = self._registry.end(run_id=run_id, output=output, end_time=time.time())
        tool_name = self._tool_names.pop(run_id, "tool")
        # Unwrapped once: the parse needs the message, and so does every
        # observation fallback below.
        message = _unwrap_tool_message(output)
        message_text = _message_text(message) if message is not None else None
        tool_result = _try_parse_tool_result(output, message=message)
        if tool_result is not None:
            event_type = (
                ExecutionEventType.TOOL_ERROR if tool_result.error else ExecutionEventType.TOOL_END
            )
            if tool_result.error:
                observation: Any = tool_result.error
            elif tool_result.data is not None:
                observation = tool_result.data
            else:
                # A metadata-only artifact is a legitimate result (see
                # _parse_tool_result_dict), and the answer the model saw is still
                # on the message. `is not None` rather than falsiness: a tool
                # returning 0, "" or [] as its data meant to.
                observation = message_text
            await self._emit(
                self._new_event(
                    event_type=event_type,
                    metadata=state.metadata,
                    # A tool's own metadata reaches the platform here and nowhere
                    # else — the same contract as a local tool's TOOL_END.
                    event_metadata=_wire_safe_metadata(tool_result.metadata, tool_name=tool_name),
                    content=tool_result.emitted_content if tool_result.emitted_content else [],
                    action=ToolAction(
                        name=tool_name,
                        parameters=None,
                        call_id=str(run_id),
                        observation=observation,
                    ),
                    error_code=tool_result.error_code,
                    error_message=tool_result.error,
                )
            )
        else:
            # A ToolMessage (bare, or inside a Command) that carries no ToolResult
            # still has the content the tool meant for the model; stringifying the
            # wrapper instead would put a Python repr on the wire as the observation.
            await self._emit(
                self._new_event(
                    event_type=ExecutionEventType.TOOL_END,
                    metadata=state.metadata,
                    action=ToolAction(
                        name=tool_name,
                        parameters=None,
                        call_id=str(run_id),
                        observation=_stringify(message_text if message is not None else output),
                    ),
                )
            )

    @_safe
    async def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        if run_id in self._platform_tool_runs:
            self._platform_tool_runs.discard(run_id)
            return
        state, _ = self._registry.error(run_id=run_id, error=error, end_time=time.time())
        tool_name = self._tool_names.pop(run_id, "tool")
        await self._emit(
            self._new_event(
                event_type=ExecutionEventType.TOOL_ERROR,
                metadata=state.metadata if state else {},
                action=ToolAction(name=tool_name, parameters=None, call_id=str(run_id)),
                error_code=type(error).__name__,
                error_message=str(error),
            )
        )

    # ---- retriever callbacks ----

    @_safe
    async def on_retriever_start(
        self,
        serialized: dict[str, Any] | None,
        query: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        retriever_name = (serialized or {}).get("name") or "retriever"
        self._tool_names[run_id] = retriever_name
        self._registry.start(
            run_id=run_id,
            parent_run_id=parent_run_id,
            span_type="retriever",
            start_time=time.time(),
            inputs={"query": query},
            metadata=metadata or {},
        )
        text = f"Invoking tool: {retriever_name}"
        if query:
            text = f"{text}\nInput: {query}"
        await self._emit(
            self._new_event(
                event_type=ExecutionEventType.TOOL_START,
                metadata=metadata,
                content=[Part(text=text)],
                action=ToolAction(
                    name=retriever_name,
                    parameters={"query": query},
                    call_id=str(run_id),
                ),
            )
        )

    @_safe
    async def on_retriever_end(
        self,
        documents: Any,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        docs_serializable = [
            {
                "page_content": getattr(d, "page_content", str(d)),
                "metadata": getattr(d, "metadata", {}),
            }
            for d in (documents or [])
        ]
        state, _ = self._registry.end(run_id=run_id, output=docs_serializable, end_time=time.time())
        retriever_name = self._tool_names.pop(run_id, "retriever")
        await self._emit(
            self._new_event(
                event_type=ExecutionEventType.TOOL_END,
                metadata=state.metadata,
                action=ToolAction(
                    name=retriever_name,
                    parameters=None,
                    call_id=str(run_id),
                    observation=docs_serializable,
                ),
            )
        )

    @_safe
    async def on_retriever_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        state, _ = self._registry.error(run_id=run_id, error=error, end_time=time.time())
        retriever_name = self._tool_names.pop(run_id, "retriever")
        await self._emit(
            self._new_event(
                event_type=ExecutionEventType.TOOL_ERROR,
                metadata=state.metadata if state else {},
                action=ToolAction(name=retriever_name, parameters=None, call_id=str(run_id)),
                error_code=type(error).__name__,
                error_message=str(error),
            )
        )

    # ---- agent callbacks ----

    @_safe
    async def on_agent_action(
        self,
        action: Any,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        state = self._registry.get(run_id)
        metadata = state.metadata if state else None
        tool_input = getattr(action, "tool_input", "")
        parameters: dict[str, Any] | None = (
            tool_input if isinstance(tool_input, dict) else {"input": tool_input}
        )
        await self._emit(
            self._new_event(
                event_type=ExecutionEventType.TOOL_START,
                metadata=metadata,
                action=ToolAction(
                    name=getattr(action, "tool", "") or "tool",
                    parameters=parameters,
                    call_id=str(run_id),
                    thought=getattr(action, "log", None) or None,
                ),
            )
        )

    @_safe
    async def on_agent_finish(
        self,
        finish: Any,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        state = self._registry.get(run_id)
        metadata = state.metadata if state else None
        return_values = getattr(finish, "return_values", {})
        await self._emit(
            self._new_event(
                event_type=ExecutionEventType.TOOL_END,
                metadata=metadata,
                action=ToolAction(
                    name="agent_finish",
                    parameters=None,
                    call_id=str(run_id),
                    observation=return_values,
                    message=getattr(finish, "log", None) or None,
                ),
            )
        )
        await self._maybe_emit_response(state)
