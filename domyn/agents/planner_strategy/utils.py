"""Shared utilities for planner strategies."""

import json
import re
from dataclasses import dataclass
from typing import Any, cast

from domyn.agents.core import ExecutionEventType, InteractionContext
from domyn.agents.llm import LLMMessage
from domyn.agents.logger import get_logger

# ---------------------------------------------------------------------------
# Forward-verbatim: shared prompt fragments
# ---------------------------------------------------------------------------
# Used by both React (SpecialAction in BasePrompting) and ToolUse
# (FunctionDeclaration in ToolUseContextManager) so the LLM sees a
# consistent description regardless of planner strategy.

FORWARD_VERBATIM_TOOL_NAME = "FORWARD_VERBATIM"

FORWARD_VERBATIM_DESCRIPTION = (
    "Forward a prior tool or sub-agent observation to the caller verbatim, "
    "by referencing its event_id. Use this terminal action only when the "
    "exact payload that will be emitted is a complete, user-ready natural "
    "language or markdown answer and is self-contained. If citations, links, "
    "or references must be resolved from sibling metadata, or the answer needs "
    "any rewriting, use the normal response action instead."
)

FORWARD_VERBATIM_SOURCE_ID_DESCRIPTION = (
    "The event_id of the prior source to forward: either a TOOL_END "
    "observation header or an eligible sub-agent AGENT_END marker "
    "(`Agent: <name> ended [event_id: ...]`). "
    "Copy the opaque value exactly from the header, including any prefix, "
    "dashes, or UUID characters. Do not repair, shorten, or regenerate it. "
    "Do not pass an empty string: if you cannot identify the source, respond "
    "in your own words instead."
)

FORWARD_VERBATIM_SOURCE_KEY_DESCRIPTION = (
    "Dot-path to the prose field inside a tool observation. The path MUST "
    "trace from the observation root to the prose leaf — every wrapper key "
    "included — and must resolve to a string of human-readable text, not to "
    "a container. Only use a leaf when that single resolved string is "
    "standalone: do not forward markdown that contains bare citation or "
    "reference markers (for example [0], [1], [^note], or [label][id]) whose "
    "definitions, URLs, or targets live in sibling fields such as sources, "
    "links, citations, or references. In that case, use the normal response "
    "action and rewrite the markdown with resolved links or without unresolved "
    "markers. Set source_key to the empty string only when the observation "
    "itself is already a plain string of prose; do not use a guessed key. For "
    "sub-agent AGENT_END sources, source_key MUST be the empty string — the "
    "sub already extracted its answer."
)

EVENT_ID_HEADER_FMT = "[event_id: {event_id}]"
"""Format string for the event_id header injected into tool observations.

Both React and ToolUse context managers use this so the LLM sees a
consistent header regardless of planner strategy.  Usage::

    header = EVENT_ID_HEADER_FMT.format(event_id=event.event_id)
"""

_EVENT_ID_HEADER_EXAMPLE = EVENT_ID_HEADER_FMT.format(event_id="...")

FORWARD_VERBATIM_MISS_UNRESOLVED = (
    "FORWARD_VERBATIM could not resolve source_id={source_id!r} / "
    "source_key={source_key!r}. Either the event_id does not match any "
    "prior TOOL_END/AGENT_END event, or the dot-path did not resolve on "
    f"the observation. Retry with the correct event_id from the "
    f"{_EVENT_ID_HEADER_EXAMPLE} header, or respond to the user in your "
    "own words without calling FORWARD_VERBATIM."
)
"""Internal diagnostic logged when resolve_verbatim returns not-found.

Usage::

    FORWARD_VERBATIM_MISS_UNRESOLVED.format(source_id=..., source_key=...)
"""

# ---------------------------------------------------------------------------
# Source_id format validation
# ---------------------------------------------------------------------------
# LLMs that lack an event_id header to copy have been observed inventing
# identifier-shape tokens for ``FORWARD_VERBATIM(source_id=...)``: the literal
# marker "Agent: Worker ended.", "tool_result_<tool_name>", or
# "TOOL_END[<tool_name>]". A loose regex rejects values containing spaces,
# punctuation, or brackets — fast first-line filter that catches the obvious
# phrase-shaped hallucinations before the history scan. Identifier-shape
# hallucinations (e.g. "tool_result_weather_report") still pass the
# filter and are caught later by the history miss + soft-retry path.

_VALID_SOURCE_ID_RE = re.compile(r"^[A-Za-z0-9_-]{4,}$")


def is_valid_source_id_format(source_id: str) -> bool:
    """Return ``True`` when *source_id* has identifier shape.

    Accepts anything alphanumeric plus ``-`` and ``_``, length >= 4. Real
    event_ids (``event_<uuid>`` or bare UUID) match. Test-fixture event_ids
    like ``evt-123`` match. English phrases, bracketed expressions, and
    formatted markers do not match.
    """
    return bool(_VALID_SOURCE_ID_RE.match(source_id))


FORWARD_VERBATIM_FEEDBACK_INVALID_FORMAT = (
    "FORWARD_VERBATIM was called with source_id={source_id!r}, which is not a "
    f"valid event_id. The event_id is the value inside the {_EVENT_ID_HEADER_EXAMPLE} "
    "header attached to a tool observation. If no such header is visible for "
    "the content you want to forward, do not call FORWARD_VERBATIM — respond "
    "to the user in your own words instead."
)
"""LLM-facing feedback when source_id has phrase shape (likely hallucinated)."""

FORWARD_VERBATIM_FEEDBACK_NOT_FOUND = (
    "FORWARD_VERBATIM: source_id={source_id!r} did not match any event_id in "
    f"the visible history. Copy the value from a {_EVENT_ID_HEADER_EXAMPLE} "
    "header exactly, or respond in your own words instead."
)
"""LLM-facing feedback when source_id has valid format but no matching event."""

FORWARD_VERBATIM_FEEDBACK_PATH_NOT_RESOLVED = (
    "FORWARD_VERBATIM: source_id={source_id!r} resolved correctly, but "
    "source_key={source_key!r} did not resolve inside that observation. "
    "Choose source_key by tracing the FULL dot-path from the observation root "
    "to the prose field, including every wrapper key."
)
"""LLM-facing feedback when source_key misses on an existing observation."""

FORWARD_VERBATIM_FEEDBACK_AGENT_END_SOURCE_KEY = (
    "FORWARD_VERBATIM: source_key={source_key!r} is not allowed for sub-agent "
    "AGENT_END sources. Either omit source_key (forward the sub's whole "
    "answer), or pick the original TOOL_END event_id and apply source_key on "
    "that source instead."
)
"""LLM-facing feedback when source_key is set on an AGENT_END source."""

FORWARD_VERBATIM_FEEDBACK_EMPTY_SOURCE_ID = (
    "FORWARD_VERBATIM requires a non-empty source_id. Copy the value inside "
    f"the {_EVENT_ID_HEADER_EXAMPLE} header of the observation you want to "
    "forward. If no such header is visible, do not call FORWARD_VERBATIM — "
    "respond in your own words instead."
)
"""LLM-facing feedback when source_id is empty."""

FORWARD_VERBATIM_FEEDBACK_HARD_FALLBACK = (
    "FORWARD_VERBATIM has failed multiple times and cannot be recovered for "
    "this request. Stop calling FORWARD_VERBATIM in this turn. Use the "
    "information already visible to you in the conversation history "
    "(prior tool observations and sub-agent answers) and respond to the "
    "user directly with a concise, helpful answer in your own words."
)
"""LLM-facing feedback used as a last-ditch nudge before a hard FV miss:
asks the LLM to abandon FORWARD_VERBATIM and respond directly with the
information it already has, instead of letting the turn die on AGENT_ERROR."""


@dataclass(frozen=True)
class VerbatimResolution:
    """Detailed FORWARD_VERBATIM lookup result used for LLM-facing recovery."""

    text: str = ""
    found: bool = False
    source_event_type: ExecutionEventType | None = None
    source_key_miss: bool = False


def resolve_verbatim(
    context: InteractionContext, source_id: str, source_key: str | None = None
) -> tuple[str, bool]:
    """Look up a tool/agent result in the interaction history.

    *source_id* is **required** — it must match the ``event_id`` of a
    ``TOOL_END`` or ``AGENT_END`` event in the current interaction or visible
    conversation history.

    Lookup spans both ``interaction_history`` and ``conversation_history``
    because a sub-agent's ``AGENT_END`` belongs to the sub's own
    ``interaction_id``, while the parent resumes with a fresh
    ``interaction_history`` filtered by the parent's ``interaction_id``.
    Without the conversation_history fallback, a parent calling
    ``FORWARD_VERBATIM`` on a sub's end would always miss.

    When *source_key* is given, ``extract_by_dot_path`` navigates into the
    observation to extract a nested field.  This is valid only for ``TOOL_END``
    sources.  ``AGENT_END`` sources are already post-extraction sub-agent
    outputs, so they must be forwarded with an empty ``source_key``.  When
    *source_key* is ``None`` or empty, the full observation is forwarded as-is
    (stringified if dict/list).

    Returns ``(text, found)`` — *found* is ``True`` when the observation was
    successfully resolved and should be flagged for forward-verbatim.
    """
    result = resolve_verbatim_detailed(context, source_id, source_key)
    return result.text, result.found


def resolve_verbatim_detailed(
    context: InteractionContext, source_id: str, source_key: str | None = None
) -> VerbatimResolution:
    """Resolve a FORWARD_VERBATIM source with miss diagnostics."""
    logger = get_logger()

    visible_history = [*context.interaction_history, *context.conversation_history]
    source_event = None
    for event in visible_history:
        if event.event_type not in (ExecutionEventType.TOOL_END, ExecutionEventType.AGENT_END):
            continue
        if event.event_id != source_id:
            continue
        source_event = event
        break

    if source_event is None:
        logger.warning(
            f"forward_verbatim: source_id={source_id!r} not found in interaction or conversation history"
        )
        return VerbatimResolution()

    if source_key and source_event.event_type == ExecutionEventType.AGENT_END:
        logger.warning(
            f"forward_verbatim: source_key={source_key!r} is not supported for "
            f"AGENT_END source event_id={source_event.event_id}; forward the "
            "sub-agent output with an empty source_key"
        )
        return VerbatimResolution(source_event_type=source_event.event_type)

    action = cast(Any, source_event.action)
    if action is None:
        logger.warning(
            f"forward_verbatim: source_id={source_id!r} not found in interaction or conversation history"
        )
        return VerbatimResolution(source_event_type=source_event.event_type)

    # Extract observation: TOOL_END uses action.observation, AGENT_END carries
    # its forwarded payload in content. ``or None`` so an AGENT_END with no
    # content falls through to the miss path below (empty payload == not found)
    # rather than silently forwarding an empty string.
    obs = action.observation
    if obs is None and source_event.event_type == ExecutionEventType.AGENT_END:
        obs = "".join(part.text for part in source_event.content if part.text) or None

    if obs is None:
        logger.warning(
            f"forward_verbatim: observation is None on source event "
            f"(event_id={source_event.event_id}, event_type={source_event.event_type})"
        )
        return VerbatimResolution(source_event_type=source_event.event_type)

    if source_key:
        resolved = extract_by_dot_path(obs, source_key)
        if resolved is None:
            logger.warning(
                f"forward_verbatim: source_key={source_key!r} did not resolve on observation of "
                f"event_id={source_event.event_id}"
            )
            return VerbatimResolution(
                source_event_type=source_event.event_type,
                source_key_miss=True,
            )
        obs = resolved

    text = json.dumps(obs, default=str) if isinstance(obs, dict | list) else str(obs)
    logger.info(
        f"forward_verbatim: forwarding observation from event_id={source_event.event_id} "
        f"(source_id={source_id!r}, source_key={source_key!r})"
    )
    return VerbatimResolution(
        text=text,
        found=True,
        source_event_type=source_event.event_type,
    )


def extract_by_dot_path(data: Any, path: str) -> Any:
    """Navigate *data* using a dot-separated *path* (e.g. ``"result.documents"``
    or ``"0.summary"`` for list-rooted observations).

    Returns the nested value, or ``None`` if any segment is missing or
    cannot be resolved against the current node.

    Numeric segments (e.g. ``"0"``, ``"3"``) are treated as list indices
    when the current node is a list — supports tools that return a list of
    records at the root (common for search-style MCP tools).

    If a value along the path is a JSON string, it is transparently decoded
    so that external tools returning serialised JSON (e.g. MCP servers) are
    handled correctly.
    """
    current = data
    for segment in path.split("."):
        if isinstance(current, str):
            try:
                current = json.loads(current)
            except (json.JSONDecodeError, TypeError):
                return None
        if not segment:
            return None
        if isinstance(current, list):
            try:
                idx = int(segment)
            except ValueError:
                return None
            if idx < 0 or idx >= len(current):
                return None
            current = current[idx]
            continue
        if not isinstance(current, dict) or segment not in current:
            return None
        current = current[segment]
    return current


def reorder_messages_around_tool_calls(messages: list[LLMMessage]) -> list[LLMMessage]:
    """Move messages wedged between a tool call and its observation to after it.

    Several LLM providers with native tool-calling (OpenAI in particular)
    reject a request when an assistant message carrying ``tool_calls`` is not
    immediately followed by the ``tool``-role response(s) for every one of
    those calls. That invariant can be violated by content injected mid-turn
    from outside the runner loop (see ``Runner.push_parts``): the resulting
    partial ``RESPONSE`` message is appended to history as soon as it is
    pushed, which can land — in event order — between a ``TOOL_START`` and
    its ``TOOL_END``/``TOOL_ERROR`` observation, e.g.::

        [assistant(tool_calls), assistant(injected partial), tool(observation), assistant(response)]

    This performs a stable reorder — no message is dropped or reordered
    globally — moving anything that lands before all matching ``tool``
    responses have arrived to right after them::

        [assistant(tool_calls), tool(observation), assistant(injected partial), assistant(response)]

    When a tool call's responses never show up in *messages* (e.g. the call
    is genuinely still in flight), the intervening messages are left where
    they are — there is nothing to move them after yet.
    """
    reordered: list[LLMMessage] = []
    i = 0
    n = len(messages)
    while i < n:
        message = messages[i]
        reordered.append(message)
        i += 1
        if message.role != "assistant" or not message.tool_calls:
            continue

        pending_ids = {tool_call.id for tool_call in message.tool_calls}
        deferred: list[LLMMessage] = []
        while i < n and pending_ids:
            candidate = messages[i]
            if candidate.role == "tool" and candidate.tool_call_id in pending_ids:
                reordered.append(candidate)
                pending_ids.discard(candidate.tool_call_id)
            else:
                deferred.append(candidate)
            i += 1
        reordered.extend(deferred)
    return reordered
