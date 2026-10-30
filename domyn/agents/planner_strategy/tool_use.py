"""Tool-use planner strategy for native tool-calling agentic loops.

Uses the LLM's native tool-calling API instead of prompt-engineered ReAct.
The LLM receives tool definitions as structured parameters and returns
tool_calls objects, eliminating the need for text parsing.

Response contract
-----------------
* ``use_stop=False`` (main / user-facing agent):
  - Content only (no tool calls) → final ``RESPONSE``, loop ends.
  - Content + tool calls in the same turn → content is treated as a
    user-visible preamble. In streaming mode it is emitted as
    ``RESPONSE_TOKEN`` events, but it is not copied into
    ``action.thought`` / ``action.message`` on the tool-call event.

* ``use_stop=True`` (sub-agent):
  - ``STOP(info)`` returns data to the parent agent (``AGENT_END``).
  - ``ASK_USER(text)`` asks the end-user for clarification (``RESPONSE`` with
    ``need_feedback=True``).
  - Content only (no tool calls) → implicit ``AGENT_END``.
"""

import json
import traceback
import uuid
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from pydantic import field_validator

from domyn.agents.core import (
    AGENT_ERROR_CODE,
    AGENT_ERROR_MESSAGE,
    METADATA_FORWARD_VERBATIM,
    METADATA_FORWARD_VERBATIM_SOURCE_ID,
    METADATA_FORWARD_VERBATIM_SOURCE_KEY,
    METADATA_TOOL_BATCH,
    AgentAction,
    BaseAgent,
    BaseEvent,
    ExecutionEventType,
    InteractionContext,
    LLMCallAction,
    Part,
    ToolAction,
)
from domyn.agents.core.utils import extract_parts
from domyn.agents.llm import DelegateLLMProvider
from domyn.agents.llm.base import LLMMessage, LLMResponse, LLMToolCall, is_context_overflow
from domyn.agents.logger import get_logger
from domyn.agents.planner_strategy.base import BasePlannerStrategy
from domyn.agents.planner_strategy.context_injection import (
    insert_turn_context,
    turn_context_messages,
)
from domyn.agents.planner_strategy.context_management.compaction import Compactor
from domyn.agents.planner_strategy.context_management.tool_use import ToolUseContextManager
from domyn.agents.planner_strategy.utils import (
    FORWARD_VERBATIM_FEEDBACK_AGENT_END_SOURCE_KEY,
    FORWARD_VERBATIM_FEEDBACK_EMPTY_SOURCE_ID,
    FORWARD_VERBATIM_FEEDBACK_HARD_FALLBACK,
    FORWARD_VERBATIM_FEEDBACK_INVALID_FORMAT,
    FORWARD_VERBATIM_FEEDBACK_NOT_FOUND,
    FORWARD_VERBATIM_FEEDBACK_PATH_NOT_RESOLVED,
    FORWARD_VERBATIM_MISS_UNRESOLVED,
    is_valid_source_id_format,
    resolve_verbatim_detailed,
)

# Number of soft-miss retries before falling back to the hard-miss terminal.
# Each retry injects a tool-role feedback message and re-iterates the planner
# so the LLM can recover by responding in its own words. The cap prevents
# pathological loops where the LLM keeps hallucinating the same garbage source_id.
MAX_FV_SOFT_MISS_RETRIES = 2

# Number of times to re-invoke the LLM verbatim (same messages) when a tool
# call's arguments fail to parse as JSON, before giving up.
MAX_TOOL_CALL_ARGS_RETRIES = 3


def _parse_tool_call_args(
    tool_calls: list[LLMToolCall] | None,
) -> tuple[dict[str, dict[str, Any]], bool]:
    """Parse each tool call's arguments once.

    Returns (call_id -> parsed args, whether every tool call parsed cleanly).
    A call whose arguments fail to parse gets {} in the returned mapping.
    """
    parsed, all_valid, _ = parse_tool_call_args_with_errors(tool_calls)
    return parsed, all_valid


# Raw control characters that JSON forbids inside a string literal.
_CONTROL_ESCAPES = {"\n": "\\n", "\r": "\\r", "\t": "\\t", "\b": "\\b", "\f": "\\f"}


def repair_control_characters(raw: str) -> str | None:
    """Escape raw control characters inside string literals, or ``None``.

    This is the one repair worth making automatically, because it cannot change
    what the arguments mean: a literal newline inside a JSON string is not
    ambiguous, it is *illegal*, and escaping it is the only reading there is.

    It is also the failure that actually happens. A model writing a file's
    contents escapes the quotes and forgets the newlines; over an HTML document
    that is a hundred chances to slip and one is enough. The other common shape —
    an unescaped ``"`` — is deliberately **not** repaired: where the string ends
    becomes a guess, and guessing at the boundaries of content that is about to
    be written to disk is worse than failing.
    """
    out: list[str] = []
    in_string = False
    escaped = False
    changed = False

    for char in raw:
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            elif char in _CONTROL_ESCAPES:
                out.append(_CONTROL_ESCAPES[char])
                changed = True
                continue
        elif char == '"':
            in_string = True
        out.append(char)

    return "".join(out) if changed else None


def tool_batches(agent: Any, tool_calls: list[LLMToolCall]) -> dict[str, dict[str, Any]]:
    """The ``METADATA_TOOL_BATCH`` of each call that belongs to a batch, by call id.

    A batch is a run of two or more consecutive calls to tools the agent has: those
    the planner yields one ``TOOL_START`` after another, with nothing in between
    that ends or diverts the step (a hand-off, STOP, FORWARD_VERBATIM, an unknown
    tool). The runner holds them until the last and runs them together; a call
    outside any batch is run on arrival, as before. Calls with no id cannot be told
    apart, so a response with one is not batched at all.
    """
    if any(not tc.id for tc in tool_calls):
        return {}
    runs: list[list[str]] = [[]]
    for tc in tool_calls:
        if agent.get_tool_by_name(tc.function.name):
            runs[-1].append(tc.id)
        elif runs[-1]:
            runs.append([])
    batches: dict[str, dict[str, Any]] = {}
    for run in runs:
        if len(run) < 2:
            continue
        batch_id = uuid.uuid4().hex
        for index, call_id in enumerate(run):
            batches[call_id] = {"id": batch_id, "index": index, "size": len(run)}
    return batches


def parse_tool_call_args_with_errors(
    tool_calls: list[LLMToolCall] | None,
) -> tuple[dict[str, dict[str, Any]], bool, list[str]]:
    """As above, plus what was wrong with each call that failed.

    The reasons exist to be handed back to the model. Retrying an identical
    request after a parse failure asks a model to spot a mistake it has no way
    of knowing it made — and it duly makes it again, three times, for thirty
    seconds. The error is the only thing that gives the retry a reason to differ.
    """
    parsed: dict[str, dict[str, Any]] = {}
    errors: list[str] = []
    all_valid = True
    for tc in tool_calls or []:
        try:
            parsed[tc.id] = json.loads(tc.function.arguments)
            continue
        except json.JSONDecodeError as exc:
            first_error = exc

        repaired = repair_control_characters(tc.function.arguments)
        if repaired is not None:
            try:
                parsed[tc.id] = json.loads(repaired)
                get_logger().info(
                    f"repaired unescaped control characters in the arguments to {tc.function.name}"
                )
                continue
            except json.JSONDecodeError:
                pass

        parsed[tc.id] = {}
        all_valid = False
        errors.append(f"{tc.function.name}: {first_error}")
    return parsed, all_valid, errors


def malformed_args_feedback(errors: list[str], *, truncated: bool = False) -> LLMMessage:
    """The message that makes a retry worth making.

    Two different failures need opposite advice, and telling them apart is the
    difference between a retry that can work and three that cannot:

    * **Truncated** — the provider said ``finish_reason: "length"``. Nothing was
      wrong with what the model wrote; there was simply more of it than the token
      limit allowed. Telling it to escape more carefully is useless, because the
      next attempt is cut at exactly the same place. The only way out is less
      content per call.
    * **Malformed** — the model got an escape wrong. Here naming the characters
      is what helps, since "produce valid JSON" is advice it believed it was
      already following.
    """
    if truncated:
        return LLMMessage(
            role="user",
            content=(
                "Your last tool call was cut off before it finished: the output hit "
                "the generation limit part-way through the arguments, so they could "
                "not be read. Nothing was wrong with the content itself — there was "
                "too much of it for one call. Split the work: write a first, smaller "
                "piece, then add the rest with further calls. Do not retry the same "
                "large call, it will be cut off at the same place."
            ),
        )

    detail = "; ".join(errors) or "the arguments were not valid JSON"
    return LLMMessage(
        role="user",
        content=(
            "Your last tool call could not be used: its arguments were not valid "
            f"JSON ({detail}). Make the same call again with valid JSON. If an "
            "argument contains a file's contents or a shell command, every "
            "newline, double quote and backslash inside it must be escaped. "
            "If the value is long, send it in several smaller calls rather than "
            "one large one — a long value is a long escape to get right."
        ),
    )


@dataclass(frozen=True)
class ForwardVerbatimMiss:
    """Recoverable ``FORWARD_VERBATIM`` miss.

    A successful ``FORWARD_VERBATIM`` dispatch returns a real ``BaseEvent``.
    Misses are not events yet: top-level planners may turn ``feedback`` into a
    tool-role diagnostic and retry, while sub-agents or exhausted retries later
    become a terminal hard-miss event.
    """

    feedback: str


def _handle_forward_verbatim(
    agent: BaseAgent,
    interaction_context: InteractionContext,
    tool_call: LLMToolCall,
    args: dict[str, Any],
    use_stop: bool,
    stop_tool_name: str,
) -> BaseEvent | ForwardVerbatimMiss:
    """Resolve a ``FORWARD_VERBATIM`` tool call.

    Diagnoses the call without making retry decisions — the planner owns
    retry policy and decides whether to yield the returned event, surface
    miss feedback as a soft retry, or build a terminal hard-miss event.

    Miss reasons surfaced via ``feedback``:
    - empty ``source_id``
    - phrase-shaped ``source_id`` (failed format check, likely hallucinated)
    - ``source_key`` set on an ``AGENT_END`` source (post-extraction prose
      cannot be navigated by dot-path)
    - ``source_key`` missing on an existing ``TOOL_END`` observation
    - valid format but no matching event in visible history
    """
    logger = get_logger()

    source_id = (args.get("source_id") or "").strip()
    source_key = args.get("source_key")

    if not source_id:
        logger.warning("FORWARD_VERBATIM called without source_id")
        return ForwardVerbatimMiss(feedback=FORWARD_VERBATIM_FEEDBACK_EMPTY_SOURCE_ID)

    if not is_valid_source_id_format(source_id):
        logger.warning(
            f"FORWARD_VERBATIM source_id={source_id!r} failed format check (likely hallucinated)"
        )
        return ForwardVerbatimMiss(
            feedback=FORWARD_VERBATIM_FEEDBACK_INVALID_FORMAT.format(source_id=source_id),
        )

    resolution = resolve_verbatim_detailed(interaction_context, source_id, source_key)
    text, found = resolution.text, resolution.found

    if not found:
        logger.warning(
            FORWARD_VERBATIM_MISS_UNRESOLVED.format(source_id=source_id, source_key=source_key)
        )
        # Distinguish the AGENT_END + source_key reject from a generic miss so
        # the LLM gets actionable feedback. ``resolve_verbatim`` returns the
        # same ``(text="", found=False)`` for both cases; reproduce the check
        # here only when the source actually exists with that ``event_id``.
        if source_key and _source_event_is_agent_end(interaction_context, source_id):
            return ForwardVerbatimMiss(
                feedback=FORWARD_VERBATIM_FEEDBACK_AGENT_END_SOURCE_KEY.format(
                    source_key=source_key
                ),
            )
        if source_key and resolution.source_key_miss:
            return ForwardVerbatimMiss(
                feedback=FORWARD_VERBATIM_FEEDBACK_PATH_NOT_RESOLVED.format(
                    source_id=source_id,
                    source_key=source_key,
                ),
            )
        return ForwardVerbatimMiss(
            feedback=FORWARD_VERBATIM_FEEDBACK_NOT_FOUND.format(source_id=source_id),
        )

    # Audit trail lives in metadata (not action.parameters) so it never
    # leaks into the next-turn LLM prompt via the context manager's
    # AGENT_END rendering which dumps parameters verbatim.
    terminal_metadata: dict[str, Any] = {
        METADATA_FORWARD_VERBATIM: True,
        METADATA_FORWARD_VERBATIM_SOURCE_ID: source_id,
        METADATA_FORWARD_VERBATIM_SOURCE_KEY: source_key,
    }
    if use_stop:
        return BaseEvent(
            event_type=ExecutionEventType.AGENT_END,
            author=agent.name,
            action=ToolAction(name=stop_tool_name, parameters=None),
            content=[Part(text=text)],  # Only what's in AGENT_END content goes into llm messages
            is_partial=False,
            metadata=terminal_metadata,
        )
    return BaseEvent(
        event_type=ExecutionEventType.RESPONSE,
        content=extract_parts(text),
        author=agent.name,
        is_partial=False,
        action=ToolAction(
            name=tool_call.function.name,
            parameters={},
        ),
        metadata=terminal_metadata,
    )


def _source_event_is_agent_end(context: InteractionContext, source_id: str) -> bool:
    """Look up a source event_id and report whether it is an ``AGENT_END``.

    Mirrors the lookup in :func:`resolve_verbatim` but answers a different
    question: did we miss because of an ``AGENT_END`` + ``source_key`` reject,
    or because the event_id genuinely did not match? Used only to choose the
    right LLM-facing feedback string on a miss.
    """
    visible = [*context.interaction_history, *context.conversation_history]
    for event in visible:
        if event.event_id == source_id:
            return event.event_type == ExecutionEventType.AGENT_END
    return False


def _hard_miss_event(agent: BaseAgent, use_stop: bool, stop_tool_name: str) -> BaseEvent:
    """Build the terminal event for an unrecoverable ``FORWARD_VERBATIM`` miss.

    Emits ``AGENT_END`` in both modes with ``error_code`` / ``error_message``
    populated so SDK consumers (e.g. a hosting service) can decide how to
    surface the failure — graceful response, retry from a higher layer, or
    propagate as error — instead of a hard ``ERROR_OCCURRED`` that closes the
    turn unconditionally.

    ``use_stop=True`` (sub-agent): action carries ``STOP(info=AGENT_ERROR_MESSAGE)``
    so the parent sees a normal sub-end and can decide recovery.

    ``use_stop=False`` (top-level): action left unset — top-level has no STOP
    wrapping. The error fields signal degraded termination.

    Reached only after both the soft-miss retry budget and the hard-fallback
    nudge have been exhausted.
    """
    action: ToolAction | None = None
    if use_stop:
        action = ToolAction(
            name=stop_tool_name,
            parameters=None,
        )
    return BaseEvent(
        event_type=ExecutionEventType.AGENT_END,
        event_id=str(uuid.uuid4()),
        timestamp=datetime.now(),
        author=agent.name,
        content=[Part(text=AGENT_ERROR_MESSAGE)],
        action=action,
        error_code=AGENT_ERROR_CODE,
        error_message=AGENT_ERROR_MESSAGE,
        is_partial=False,
    )


def _create_event(
    event_type: ExecutionEventType,
    content: list[Part],
    author: str,
    metadata: dict[str, Any] | None = None,
) -> BaseEvent:
    return BaseEvent(
        event_type=event_type,
        event_id=str(uuid.uuid4()),
        timestamp=datetime.now(),
        author=author,
        content=content,
        is_partial=False,
        metadata=metadata or {},
    )


class ToolUsePlannerStrategy(BasePlannerStrategy):
    """Native tool-calling planner strategy.

    Calls the LLM with tool definitions via the provider's ``tools`` parameter.
    Loops until the LLM returns a response with no tool calls.
    """

    type: str = "tool_use"
    name: str = "tool_use"
    context_manager: ToolUseContextManager = ToolUseContextManager()

    compactor: Compactor = Compactor()
    """Decides whether the conversation must be shortened to fit the window.

    Driven from here rather than hidden inside the context manager: compaction
    loses information and (with a summarizing strategy) takes seconds, so it has
    to be able to emit events — and only a generator can. With the default
    ``NoOpCompaction`` and no ``context_window`` on the provider it does
    nothing."""

    max_iterations: int = 25
    use_stop: bool = False
    parallel_tool_calls: bool = True
    """Whether the calls of one response that need no approval run at the same
    time. On, the calls are marked as a batch (``METADATA_TOOL_BATCH``) and the
    runner runs them together; off, each runs as it arrives, after the one before
    it. See ``Runner._run_tool_calls``."""

    @field_validator("context_manager", mode="wrap")
    @classmethod
    def validate_context_manager(cls, v: Any, handler: Any, info: Any) -> ToolUseContextManager:
        if isinstance(v, ToolUseContextManager):
            return v
        if isinstance(v, dict):
            return ToolUseContextManager(**v)
        if v is None:
            return ToolUseContextManager()
        try:
            return handler(v)
        except Exception:
            return ToolUseContextManager()

    async def execute(
        self,
        agent: BaseAgent,
        interaction_context: InteractionContext,
        *,
        skills_payload: str = "",
        instruction: str | None = None,
    ) -> AsyncGenerator[BaseEvent, None]:
        """Execute the native tool-use loop.

        1. Build messages from conversation history
        2. Call LLM with tool definitions
        3. If LLM returns tool calls, yield TOOL_START events
        4. If LLM returns no tool calls, yield final RESPONSE and stop
        5. Repeat until max_iterations
        """
        logger = get_logger()
        iteration = 0
        # Soft-miss budget for FORWARD_VERBATIM. Each retry injects a tool-role
        # feedback message so the LLM sees the diagnostic and can recover. Cap
        # prevents loops where the LLM keeps hallucinating the same garbage.
        # Only used when ``use_stop=False`` (top-level); sub-agents always emit
        # the hard miss so their parent decides recovery.
        fv_soft_miss_remaining = MAX_FV_SOFT_MISS_RETRIES
        # One last-ditch fallback after the soft-miss budget is spent: tell
        # the LLM to abandon FORWARD_VERBATIM and RESPOND directly. Avoids
        # ending the turn on AGENT_ERROR when the model just keeps hammering
        # FV with paths/event_ids the framework can't resolve.
        fv_hard_fallback_used = False
        # Carry-over messages from the previous iteration's soft-miss, prepended
        # to the next iteration's freshly-built context. We can't append these
        # to ``messages`` directly because ``messages`` is rebuilt each loop
        # from history; the FV miss feedback isn't in history (no ERROR_OCCURRED
        # was emitted), so it would otherwise be lost.
        fv_soft_miss_carry: list[LLMMessage] = []
        # A request the provider rejects as too long is compacted and retried
        # once per turn; this records that the once has been spent.
        overflow_retried = False

        while iteration < self.max_iterations:
            iteration += 1

            # Build messages from history
            messages = await self.context_manager.get_context(
                context=interaction_context, agent=agent, instruction=instruction
            )
            if fv_soft_miss_carry:
                messages.extend(fv_soft_miss_carry)

            # Shorten the conversation if it no longer fits. The logic lives in
            # the compactor; only the `yield`s stay here, because they can only
            # live in a generator. Nothing fires without a known window and a
            # strategy that acts.
            # Decisions already taken come back first: the message list is
            # rebuilt from events every turn, so without this it arrives with the
            # dropped text restored and a past compaction silently undoes itself.
            # Tool definitions and the turn's injected context are built first:
            # the threshold is checked on the whole request, but only the
            # conversation is compacted.
            tool_defs = self.context_manager.get_tool_definitions(agent, interaction_context)
            turn_context = await turn_context_messages(agent, interaction_context, skills_payload)
            messages = self.compactor.replay(messages, interaction_context.state, agent.name)
            compaction_plan = self.compactor.assess(
                messages,
                agent,
                tool_defs=tool_defs,
                turn_context=turn_context,
                state=interaction_context.state,
            )
            if compaction_plan.needed:
                yield self.compactor.start_event(compaction_plan, agent.name)
                messages, compaction_done = await self.compactor.run(
                    messages, compaction_plan, interaction_context.state, agent.name
                )
                yield compaction_done
            conversation = messages
            messages = insert_turn_context(list(conversation), turn_context)

            # The context measurement for LLM_START metadata: a front-end has no
            # other way to know it without rebuilding the context itself.
            usage_metadata = self.compactor.usage_metadata(
                messages, tool_defs, agent, interaction_context.state
            )

            # ── LLM invocation ────────────────────────────────────────────────
            response: LLMResponse | None = None
            parsed_tool_args: dict[str, dict[str, Any]] = {}

            if isinstance(agent.llm_provider, DelegateLLMProvider):
                # Bidirectional: runner executes LLM, sends back LLMResponse via asend()
                # TODO: LLM delegation is duplicated per planner (here, tool_use_thought); react
                # lacks it and fails with DelegateLLMProvider. Move it into the runner.
                response = yield BaseEvent(
                    event_type=ExecutionEventType.LLM_START,
                    action=LLMCallAction(
                        messages=messages,
                        tools=tool_defs,
                        stream=agent.stream,
                    ),
                    author=agent.name,
                    metadata=usage_metadata,
                )
                if response is None:
                    logger.error("DelegateLLMProvider received None from runner — aborting")
                    break
                yield _create_event(
                    event_type=ExecutionEventType.LLM_END,
                    content=[Part(text="LLM response received via delegate")],
                    author=agent.name,
                )
                parsed_tool_args, _ = _parse_tool_call_args(response.tool_calls)
            else:
                yield _create_event(
                    event_type=ExecutionEventType.LLM_START,
                    content=[Part(text="Invoking LLM with tools")],
                    author=agent.name,
                    metadata=usage_metadata,
                )
                try:
                    generation_complete = False
                    malformed_json_retries = 0
                    gave_up_on_malformed_json = False
                    while not generation_complete:
                        if agent.stream:
                            streamed_text_parts: list[str] = []
                            response = None
                            last_chunk = None
                            # Stream content as it arrives. If tool_calls follow in
                            # the same turn, this content is a user-visible preamble,
                            # not internal thought; do not copy it onto the tool event.
                            async for chunk in agent.llm_provider.generate_streaming(
                                messages=messages, tools=tool_defs if tool_defs else None
                            ):
                                last_chunk = chunk
                                # Drop only no-op (None/empty) chunks. Whitespace-only
                                # chunks (paragraph breaks, list separators) carry
                                # semantic meaning and must reach the consumer; the
                                # accumulator also depends on them so the final
                                # ``response.content`` keeps its formatting.
                                if chunk.content:
                                    streamed_text_parts.append(chunk.content)
                                    yield BaseEvent(
                                        event_type=ExecutionEventType.RESPONSE_TOKEN,
                                        content=[Part(text=chunk.content)],
                                        author=agent.name,
                                        is_partial=True,
                                    )
                                if chunk.tool_calls:
                                    response = chunk.model_copy(
                                        update={"content": "".join(streamed_text_parts)}
                                    )

                            if response is None:
                                accumulated = "".join(streamed_text_parts)
                                base = last_chunk or LLMResponse(
                                    content="", generated_tokens=0, execution_time=0.0
                                )
                                response = base.model_copy(
                                    update={"content": accumulated, "tool_calls": []}
                                )
                            else:
                                response = response.model_copy(
                                    update={"content": "".join(streamed_text_parts)}
                                )
                        else:
                            response = await agent.llm_provider.generate(
                                messages=messages,
                                tools=tool_defs if tool_defs else None,
                            )

                        if not response.content and not response.tool_calls:
                            logger.error("LLM response has no content and no tool calls")
                            messages.append(
                                LLMMessage(
                                    role="system",
                                    content="LLM response had no content and no tool calls, please generate a response to the final user.",
                                )
                            )
                        else:
                            (
                                parsed_tool_args,
                                args_valid,
                                parse_errors,
                            ) = parse_tool_call_args_with_errors(response.tool_calls)
                            if not args_valid:
                                malformed_json_retries += 1
                                if malformed_json_retries > MAX_TOOL_CALL_ARGS_RETRIES:
                                    gave_up_on_malformed_json = True
                                    generation_complete = True
                                else:
                                    # The retry is given the reason. Asking again
                                    # with an identical request expects the model
                                    # to notice a mistake it cannot see, and it
                                    # duly repeats it until the retries run out.
                                    messages.append(
                                        malformed_args_feedback(
                                            parse_errors,
                                            truncated=getattr(response, "truncated", False),
                                        )
                                    )
                                    logger.warning(
                                        (
                                            "Tool-call arguments were cut off by the "
                                            "generation limit"
                                            if getattr(response, "truncated", False)
                                            else "Malformed tool-call arguments"
                                        )
                                        + f", retrying LLM call ({malformed_json_retries}/"
                                        f"{MAX_TOOL_CALL_ARGS_RETRIES})"
                                    )
                                    logger.debug(
                                        "Malformed tool-call arguments payload: "
                                        f"{response.tool_calls!r}"
                                    )
                            else:
                                generation_complete = True

                except Exception as e:
                    # Rejected as too long: the estimate under-counted. Compact
                    # harder and go round again, once — a second rejection is an
                    # error like any other.
                    recovery = (
                        None
                        if overflow_retried or not is_context_overflow(e)
                        else self.compactor.overflow_plan(
                            conversation,
                            agent,
                            tool_defs=tool_defs,
                            turn_context=turn_context,
                            state=interaction_context.state,
                            error=e,
                        )
                    )
                    if recovery is not None:
                        overflow_retried = True
                        logger.warning("request rejected as too long: compacting, then retrying")
                        yield self.compactor.start_event(recovery, agent.name)
                        _, compaction_done = await self.compactor.run(
                            conversation, recovery, interaction_context.state, agent.name
                        )
                        yield compaction_done
                        continue
                    logger.error(f"LLM generation failed: {type(e).__name__}")
                    logger.debug(
                        f"LLM generation failed: {e}",
                        payload={"exception": traceback.format_exc()},
                    )
                    yield BaseEvent(
                        event_type=ExecutionEventType.ERROR_OCCURRED,
                        event_id=str(uuid.uuid4()),
                        timestamp=datetime.now(),
                        author=agent.name,
                        content=[Part(text=f"LLM error: {e}")],
                        error_message=str(e),
                        is_partial=False,
                    )
                    break

                yield _create_event(
                    event_type=ExecutionEventType.LLM_END,
                    content=[Part(text="LLM response received")],
                    author=agent.name,
                )

                if gave_up_on_malformed_json:
                    response_tool_calls = getattr(response, "tool_calls", None) or []
                    tool_names = [tc.function.name for tc in response_tool_calls]
                    logger.error(
                        "Tool-call arguments still malformed after "
                        f"{MAX_TOOL_CALL_ARGS_RETRIES} retries, giving up: "
                        f"tools={tool_names}"
                    )
                    logger.debug(f"Malformed tool-call arguments payload: {response_tool_calls!r}")
                    yield BaseEvent(
                        event_type=ExecutionEventType.ERROR_OCCURRED,
                        event_id=str(uuid.uuid4()),
                        timestamp=datetime.now(),
                        author=agent.name,
                        content=[
                            Part(
                                text="Tool-call arguments still malformed after "
                                f"{MAX_TOOL_CALL_ARGS_RETRIES} retries"
                            )
                        ],
                        error_message=(
                            "Malformed tool-call arguments after "
                            f"{MAX_TOOL_CALL_ARGS_RETRIES} retries"
                        ),
                        is_partial=False,
                    )
                    break

            # What the provider counted for this request: the base the next one is
            # measured from (see `Compactor.observe`).
            self.compactor.observe(
                getattr(response, "usage", None),
                usage_metadata,
                interaction_context.state,
                agent.name,
                getattr(agent, "llm_provider", None),
            )

            response_text = (getattr(response, "content", "") or "").strip()
            has_tool_calls = bool(getattr(response, "tool_calls", []))

            # ── Content only (no tool calls) → terminal ───────────────────────
            if response_text and not has_tool_calls:
                if self.use_stop:
                    yield BaseEvent(
                        event_type=ExecutionEventType.AGENT_END,
                        author=agent.name,
                        action=ToolAction(
                            name=self.context_manager.stop_tool_name,
                            parameters=None,
                        ),
                        content=[
                            Part(text=response_text)
                        ],  # Only what's in AGENT_END content goes into llm messages
                    )
                else:
                    yield BaseEvent(
                        event_type=ExecutionEventType.RESPONSE,
                        content=extract_parts(response_text),
                        author=agent.name,
                        is_partial=False,
                    )
                return

            # ── Content + tool calls → emit preamble before tools ─────────────
            # In non-streaming mode the preamble never reached the consumer as
            # RESPONSE_TOKEN, so emit it as a partial RESPONSE before the tool
            # events. In streaming mode the tokens already flowed through
            # RESPONSE_TOKEN; skip the duplicate emission.
            if response_text and has_tool_calls and not agent.stream and not self.use_stop:
                yield BaseEvent(
                    event_type=ExecutionEventType.RESPONSE,
                    content=extract_parts(response_text),
                    author=agent.name,
                    is_partial=True,
                )

            # ── Process tool calls ────────────────────────────────────────────
            if has_tool_calls:
                batches = (
                    tool_batches(agent, getattr(response, "tool_calls", None) or [])
                    if self.parallel_tool_calls
                    else {}
                )
                for tool_call in response.tool_calls:
                    args = parsed_tool_args.get(tool_call.id, {})

                    tool_name = tool_call.function.name
                    tool = agent.get_tool_by_name(tool_name)

                    if tool:
                        action = ToolAction(
                            call_id=tool_call.id,
                            name=tool_name,
                            parameters=args,
                        )
                        yield BaseEvent(
                            event_type=ExecutionEventType.TOOL_START,
                            author=agent.name,
                            action=action,
                            metadata=(
                                {METADATA_TOOL_BATCH: batches[tool_call.id]}
                                if tool_call.id in batches
                                else {}
                            ),
                        )
                    elif tool_name in [
                        a.name for a in interaction_context.handoff_targets_description
                    ]:
                        agent_action = AgentAction(
                            name=tool_name,
                            parameters=args,
                        )
                        yield BaseEvent(
                            event_type=ExecutionEventType.AGENT_CALL,
                            author=agent.name,
                            action=agent_action,
                        )
                        return
                    elif tool_name == self.context_manager.stop_tool_name and self.use_stop:
                        yield BaseEvent(
                            event_type=ExecutionEventType.AGENT_END,
                            author=agent.name,
                            action=ToolAction(
                                name=tool_name,
                                parameters=args,
                            ),
                            content=[
                                Part(text=args.get("info"))
                            ],  # Only what's in AGENT_END content goes into llm messages, "info" is the only parameter we expect on STOP
                        )
                        return
                    elif tool_name == self.context_manager.ask_user_tool_name and self.use_stop:
                        yield BaseEvent(
                            event_type=ExecutionEventType.RESPONSE,
                            content=extract_parts(args.get("text", "")),
                            author=agent.name,
                            need_feedback=True,
                            action=ToolAction(name=tool_name, parameters=args),
                        )
                        return
                    elif (
                        tool_name == self.context_manager.forward_verbatim_tool_name
                        and self.forward_verbatim_enabled
                    ):
                        # FORWARD_VERBATIM ends this planner step on hit. On miss
                        # the planner offers the LLM a bounded number of recovery
                        # attempts (top-level only) by injecting a tool-role
                        # diagnostic and re-iterating; once the budget is spent,
                        # it falls back to the hard-miss terminal.
                        fv_result = _handle_forward_verbatim(
                            agent,
                            interaction_context,
                            tool_call,
                            args,
                            use_stop=self.use_stop,
                            stop_tool_name=self.context_manager.stop_tool_name,
                        )
                        if isinstance(fv_result, BaseEvent):
                            yield fv_result
                            return
                        # Miss path. Sub-agents (use_stop=True) skip soft retry
                        # so the parent — which sees the resulting AGENT_END —
                        # owns recovery. Top-level agents retry up to
                        # ``MAX_FV_SOFT_MISS_RETRIES`` times before terminating.
                        if not self.use_stop and fv_soft_miss_remaining > 0:
                            # Native tool-use invariant: every assistant
                            # tool_call needs a paired tool-role reply. Carry
                            # both forward into the next iteration so the LLM
                            # sees what it tried and the diagnostic in turn.
                            fv_soft_miss_carry.append(
                                LLMMessage(
                                    role="assistant",
                                    content=None,
                                    tool_calls=[tool_call],
                                )
                            )
                            fv_soft_miss_carry.append(
                                LLMMessage(
                                    role="tool",
                                    tool_call_id=tool_call.id,
                                    name=tool_name,
                                    content=fv_result.feedback,
                                )
                            )
                            fv_soft_miss_remaining -= 1
                            logger.info(
                                f"FORWARD_VERBATIM soft-miss: retries remaining={fv_soft_miss_remaining}"
                            )
                            break  # break inner tool_calls loop → next planner iteration
                        # Soft-miss budget exhausted. Last-ditch fallback (top-level
                        # only): tell the LLM to abandon FORWARD_VERBATIM and
                        # RESPOND directly with information already in history,
                        # instead of ending on AGENT_ERROR. Sub-agents skip this
                        # so the parent owns recovery.
                        if not self.use_stop and not fv_hard_fallback_used:
                            fv_hard_fallback_used = True
                            fv_soft_miss_carry.append(
                                LLMMessage(
                                    role="assistant",
                                    content=None,
                                    tool_calls=[tool_call],
                                )
                            )
                            fv_soft_miss_carry.append(
                                LLMMessage(
                                    role="tool",
                                    tool_call_id=tool_call.id,
                                    name=tool_name,
                                    content=FORWARD_VERBATIM_FEEDBACK_HARD_FALLBACK,
                                )
                            )
                            logger.info(
                                "FORWARD_VERBATIM hard-fallback: instructing LLM "
                                "to abandon FV and RESPOND directly"
                            )
                            break  # break inner tool_calls loop → next planner iteration
                        # Hard miss: fallback already used or sub-agent.
                        yield _hard_miss_event(
                            agent,
                            use_stop=self.use_stop,
                            stop_tool_name=self.context_manager.stop_tool_name,
                        )
                        return
                    else:
                        action = ToolAction(
                            call_id=tool_call.id,
                            name=tool_name,
                            parameters=args,
                            observation=f"Tool '{tool_name}' not found",
                        )
                        yield BaseEvent(
                            event_type=ExecutionEventType.TOOL_START,
                            author=agent.name,
                            action=action,
                        )
                        yield BaseEvent(
                            event_type=ExecutionEventType.TOOL_ERROR,
                            author=agent.name,
                            action=action,
                        )
            elif response.content:
                # No tool calls — LLM is done, emit final response
                if self.use_stop:
                    # the content is the feedback
                    yield BaseEvent(
                        event_type=ExecutionEventType.AGENT_END,
                        author=agent.name,
                        action=ToolAction(
                            name=self.context_manager.stop_tool_name,
                            parameters=None,
                        ),
                        content=[
                            Part(text=response.content)
                        ],  # Only what's in AGENT_END content goes into llm messages
                    )

                else:
                    yield BaseEvent(
                        event_type=ExecutionEventType.RESPONSE,
                        content=extract_parts(response.content or ""),
                        author=agent.name,
                        is_partial=False,
                    )
                return

        # The while loop above can also end via an early `break` (LLM exception,
        # malformed tool-call arguments after retries exhausted) — only report
        # "max iterations reached" when that's actually why the loop ended.
        if iteration >= self.max_iterations:
            max_iterations_message = f"Reached maximum iterations ({self.max_iterations})"
            if self.use_stop:
                yield BaseEvent(
                    event_type=ExecutionEventType.AGENT_END,
                    author=agent.name,
                    action=ToolAction(
                        name=self.context_manager.stop_tool_name,
                        parameters=None,
                    ),
                    content=[
                        Part(text=max_iterations_message)
                    ],  # Only what's in AGENT_END content goes into llm messages
                )
                return

            yield BaseEvent(
                event_type=ExecutionEventType.RESPONSE,
                event_id=str(uuid.uuid4()),
                timestamp=datetime.now(),
                author=agent.name,
                content=extract_parts(max_iterations_message),
                is_partial=False,
            )
