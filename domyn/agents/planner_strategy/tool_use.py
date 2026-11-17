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
from collections import Counter
from collections.abc import AsyncGenerator, Collection
from dataclasses import dataclass
from datetime import datetime
from typing import Any, ClassVar

from pydantic import field_validator, model_validator

from domyn.agents.core import (
    AGENT_ERROR_CODE,
    AGENT_ERROR_MESSAGE,
    METADATA_FORWARD_VERBATIM,
    METADATA_FORWARD_VERBATIM_SOURCE_ID,
    METADATA_FORWARD_VERBATIM_SOURCE_KEY,
    METADATA_NOT_RUN,
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
from domyn.agents.llm.base import (
    LLMMessage,
    LLMResponse,
    LLMToolCall,
    TokenUsage,
    add_token_usage,
    is_context_overflow,
    token_usage_metadata,
)
from domyn.agents.logger import get_logger
from domyn.agents.planner_strategy.base import BasePlannerStrategy
from domyn.agents.planner_strategy.context_injection import (
    insert_turn_context,
    turn_context_messages,
)
from domyn.agents.planner_strategy.context_management.compaction import (
    Compactor,
    compaction_key,
)
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


def tool_batches(
    agent: Any,
    tool_calls: list[LLMToolCall],
    handoffs: Collection[str] = (),
    *,
    parallel_handoffs: bool = True,
    parallel_tools: bool = True,
    step_enders: Collection[str] = (),
) -> dict[str, dict[str, Any]]:
    """The ``METADATA_TOOL_BATCH`` of each call that belongs to a batch, by call id.

    A batch is a run of two or more consecutive calls the planner yields one after
    another, with nothing in between that ends the step (STOP, a question to the
    user, FORWARD_VERBATIM: ``step_enders``). The runner holds them until the last
    and runs them; a call outside any batch is run on arrival, as before. Calls
    are told apart by id, which ``ensure_call_ids`` gives the ones a provider sent
    without; should one still have none, nothing is batched.

    ``handoffs`` names the sub-agents the agent may call. A hand-off ends the
    planner's step, so a call after one that is not in a batch with it is never
    yielded, and the model is never told it did not run. Two kinds of batch keep
    every call of a response that names a sub-agent:

    * with ``parallel_handoffs``, the whole run is one batch and its sub-agents
      run at the same time, like its tools;
    * without it, the run from the first hand-off on is a batch marked
      ``sequential``: its calls run one at a time, in order, each sub-agent to its
      end before the next call starts. What comes before that hand-off is
      batched as it always was (``parallel_tools``).

    In a run with a sub-agent, a call to a tool the agent does not have joins it
    too (the runner answers it as not found): it would otherwise split the run,
    and what followed it would be lost. Without one it ends a run, as before.
    """
    if any(not tc.id for tc in tool_calls):
        return {}

    def kind(tc: LLMToolCall) -> str:
        name = tc.function.name
        if agent.get_tool_by_name(name):
            return "tool"
        if name in handoffs:
            return "agent"
        if name in step_enders:
            return "end"
        return "unknown"

    kinds = [kind(tc) for tc in tool_calls]
    names_an_agent = "agent" in kinds
    runs: list[list[int]] = [[]]
    for position, call_kind in enumerate(kinds):
        if call_kind in ("tool", "agent") or (call_kind == "unknown" and names_an_agent):
            runs[-1].append(position)
        elif runs[-1]:
            runs.append([])

    batches: dict[str, dict[str, Any]] = {}

    def stamp(run: list[int], sequential: bool = False) -> None:
        if len(run) < 2:
            return
        batch_id = uuid.uuid4().hex
        agents = Counter(tool_calls[p].function.name for p in run if kinds[p] == "agent")
        for index, position in enumerate(run):
            name = tool_calls[position].function.name
            batches[tool_calls[position].id] = {
                "id": batch_id,
                "index": index,
                "size": len(run),
                **({"sequential": True} if sequential else {}),
                # One of several calls to the same sub-agent: the copies never see
                # each other's work, so each keeps its own compaction record.
                **({"copy": True} if kinds[position] == "agent" and agents[name] > 1 else {}),
            }

    for run in runs:
        first_agent = next((i for i, p in enumerate(run) if kinds[p] == "agent"), None)
        if first_agent is None or parallel_handoffs:
            if parallel_tools:
                stamp(run)
            continue
        if parallel_tools:
            stamp(run[:first_agent])
        stamp(run[first_agent:], sequential=True)
    return batches


def ensure_call_ids(tool_calls: list[LLMToolCall] | None) -> None:
    """Give an id to each call of a response that has none, as some endpoints send
    them. Everything after keys a call by its id — its arguments, its batch, its
    result — so calls without one were read as one call: each ran with the last
    one's arguments, and a response naming two sub-agents lost the second. The id
    is ours, and goes back to the provider with the call and its result."""
    for tc in tool_calls or []:
        if not tc.id:
            tc.id = f"call_{uuid.uuid4().hex}"


def calls_not_run(
    agent_name: str,
    after: list[LLMToolCall],
    parsed: dict[str, dict[str, Any]],
    ender: str,
) -> list[BaseEvent]:
    """The calls a response makes after the one ending the step (``ender``): they
    do not run, but each is yielded marked ``METADATA_NOT_RUN`` and the runner gives
    it a result saying so. Left out, the model believed they had run."""
    reason = (
        f"the response ended with {ender} before this call. Call it again if it is still needed."
    )
    return [
        BaseEvent(
            event_type=ExecutionEventType.TOOL_START,
            author=agent_name,
            action=ToolAction(
                call_id=tc.id, name=tc.function.name, parameters=parsed.get(tc.id, {})
            ),
            metadata={METADATA_NOT_RUN: reason},
        )
        for tc in after
    ]


def parse_tool_call_args_with_errors(
    tool_calls: list[LLMToolCall] | None,
) -> tuple[dict[str, dict[str, Any]], bool, list[str]]:
    """As above, plus what was wrong with each call that failed.

    The reasons exist to be handed back to the model. Retrying an identical
    request after a parse failure asks a model to spot a mistake it has no way
    of knowing it made — and it duly makes it again, three times, for thirty
    seconds. The error is the only thing that gives the retry a reason to differ.
    """
    ensure_call_ids(tool_calls)
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


def _extract_thought(
    tool_call: LLMToolCall, response: LLMResponse | None, fallback_response: bool = False
) -> str | None:
    """Extract the thought/reasoning from an LLM response.

    Priority:
    1. ``reasoning_content`` — explicit reasoning field (DeepSeek R1, Kimi, etc.)
    2. First ``thinking`` block from ``thinking_blocks`` — Anthropic extended thinking
    3. ``content`` — plain text preamble emitted before tool calls (solicited via
       the system prompt instruction in ``ToolUseWithThoughtContextManager``).
       Only treated as thought when tool_calls are also present — otherwise it
       is the response to the user.
    """
    # Reading the "reasoning" field requires parsing the tool-call arguments.
    # Malformed arguments must NOT abort thought extraction — fall through to the
    # other thought sources instead of raising. The caller defaults malformed
    # arguments to {} on its own, so a parse failure here is not fatal and must
    # not turn into an orphan tool-error event (empty call_id, no paired
    # assistant tool_call) that strict providers reject with a 400.
    if tool_call:
        try:
            reasoning = json.loads(tool_call.function.arguments).get("reasoning")
        except json.JSONDecodeError:
            reasoning = None
        if reasoning:
            return reasoning.strip()
    if fallback_response and response:
        if response.reasoning_content:
            return response.reasoning_content.strip()
        if response.thinking_blocks:
            for block in response.thinking_blocks:
                if block.get("type") == "thinking" and block.get("thinking"):
                    return block.get("thinking", "").strip()
        if response.content and response.tool_calls:
            return response.content.strip()
    return None


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

    carries_thought: ClassVar[bool] = False
    """Whether the model's reasoning is carried with what it produced: on each
    action (``thought``), streamed as it arrives, and on the ``LLM_END`` of a
    direct answer. Set by ``ToolUseWithThoughtPlannerStrategy``; read by the one
    loop both strategies run (``execute``)."""

    max_iterations: int = 25
    use_stop: bool = False
    parallel_tool_calls: bool = True
    """Whether the calls of one response that need no approval run at the same
    time. On, the calls are marked as a batch (``METADATA_TOOL_BATCH``) and the
    runner runs them together; off, each runs as it arrives, after the one before
    it. See ``Runner._run_tool_calls``."""

    @model_validator(mode="after")
    def refuse_parallel_handoffs_alone(self) -> "ToolUsePlannerStrategy":
        # The batch is what runs sub-agents together: without it the option would
        # do nothing, silently.
        if self.parallel_handoffs and not self.parallel_tool_calls:
            raise ValueError("parallel_handoffs needs parallel_tool_calls")
        return self

    def batches(
        self, agent: Any, tool_calls: list[LLMToolCall], context: InteractionContext
    ) -> dict[str, dict[str, Any]]:
        """``tool_batches`` for this planner's options, shared by both loops."""
        step_enders = set()
        if self.use_stop:
            step_enders |= {
                self.context_manager.stop_tool_name,
                self.context_manager.ask_user_tool_name,
            }
        if self.forward_verbatim_enabled:
            step_enders.add(self.context_manager.forward_verbatim_tool_name)
        return tool_batches(
            agent,
            tool_calls,
            {a.name for a in context.handoff_targets_description},
            parallel_handoffs=self.parallel_handoffs,
            parallel_tools=self.parallel_tool_calls,
            step_enders=step_enders,
        )

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

    def _thought_of(
        self, tool_call: LLMToolCall, response: LLMResponse | None, *, first: bool
    ) -> str | None:
        """The reasoning behind ``tool_call``, when this strategy carries it
        (``carries_thought``); the response's own reasoning goes on the first call."""
        if not self.carries_thought:
            return None
        thought = _extract_thought(tool_call, response, fallback_response=first)
        return thought.strip() if thought else None

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

        The one loop of both tool-use strategies: what ``carries_thought`` adds
        (``ToolUseWithThoughtPlannerStrategy``) is read where it applies, so a
        change made here reaches both. It was a copy of this method once, and a
        fix made to one was found missing from the other more than once.
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
            # A copy of the agent called alongside another keeps its own.
            record = compaction_key(agent.name, interaction_context)
            messages = self.compactor.replay(messages, interaction_context.state, record)
            compaction_plan = self.compactor.assess(
                messages,
                agent,
                tool_defs=tool_defs,
                turn_context=turn_context,
                state=interaction_context.state,
                key=record,
            )
            if compaction_plan.needed:
                yield self.compactor.start_event(compaction_plan, agent.name)
                messages, compaction_done = await self.compactor.run(
                    messages, compaction_plan, interaction_context.state, agent.name, record
                )
                yield compaction_done
            conversation = messages
            messages = insert_turn_context(list(conversation), turn_context)

            # The context measurement for LLM_START metadata: a front-end has no
            # other way to know it without rebuilding the context itself.
            usage_metadata = self.compactor.usage_metadata(
                messages, tool_defs, agent, interaction_context.state, turn_context, key=record
            )

            # ── LLM invocation ────────────────────────────────────────────────
            response: LLMResponse | None = None
            parsed_tool_args: dict[str, dict[str, Any]] = {}
            gave_up_on_malformed_json = False
            # What this LLM_END covers: every call made for it, retries included.
            token_usage: TokenUsage | None = None

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
                parsed_tool_args, _ = _parse_tool_call_args(response.tool_calls)
                token_usage = add_token_usage(None, response)
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
                    empty_retries = 0
                    while not generation_complete:
                        if agent.stream:
                            streamed_text_parts: list[str] = []
                            response = None
                            last_chunk = None
                            streamed_thought = False
                            # Stream content as it arrives. If tool_calls follow in
                            # the same turn, this content is a user-visible preamble,
                            # not internal thought; do not copy it onto the tool event.
                            async for chunk in agent.llm_provider.generate_streaming(
                                messages=messages, tools=tool_defs if tool_defs else None
                            ):
                                last_chunk = chunk
                                # A model with native thinking emits its reasoning
                                # before the answer, and the provider surfaces it in
                                # a chunk of its own the moment it is complete.
                                # Forwarding it here keeps the order the user sees
                                # the same as the order the model produced: think,
                                # then answer. Waiting for LLM_END would put the
                                # reasoning underneath the answer it explains.
                                if (
                                    self.carries_thought
                                    and chunk.reasoning_content
                                    and not streamed_thought
                                ):
                                    streamed_thought = True
                                    yield BaseEvent(
                                        event_type=ExecutionEventType.RESPONSE_TOKEN,
                                        content=[Part(thought=chunk.reasoning_content.strip())],
                                        author=agent.name,
                                        is_partial=True,
                                    )
                                # Drop only no-op (None/empty) chunks. Whitespace-only
                                # chunks (paragraph breaks, list separators) carry
                                # semantic meaning and must reach the consumer; the
                                # accumulator also depends on them so the final
                                # ``response.content`` keeps its formatting.
                                if chunk.content:
                                    streamed_text_parts.append(chunk.content)
                                    # With thought, a sub-agent's text is its reasoning
                                    # or its report to the caller, never shown as typed.
                                    if not (self.carries_thought and self.use_stop):
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

                        token_usage = add_token_usage(token_usage, response)
                        if not response.content and not response.tool_calls:
                            logger.error("LLM response has no content and no tool calls")
                            empty_retries += 1
                            if empty_retries > MAX_TOOL_CALL_ARGS_RETRIES:
                                # Asked enough: go on with the empty answer rather than
                                # ask the same model the same thing for ever.
                                generation_complete = True
                                continue
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
                            key=record,
                        )
                    )
                    if recovery is not None:
                        overflow_retried = True
                        logger.warning("request rejected as too long: compacting, then retrying")
                        yield self.compactor.start_event(recovery, agent.name)
                        _, compaction_done = await self.compactor.run(
                            conversation, recovery, interaction_context.state, agent.name, record
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

            # What the provider counted for this request: the base the next one is
            # measured from (see `Compactor.observe`).
            self.compactor.observe(
                getattr(response, "usage", None),
                usage_metadata,
                interaction_context.state,
                record,
                getattr(agent, "llm_provider", None),
            )

            # With thought, reasoning is normally carried by the action it produced,
            # and `test_react_strategy_doesnt_emit_thought_in_llm_end` pins that: on a
            # tool-calling turn LLM_END stays empty so consumers see the thought
            # once, on TOOL_START. A turn that answers directly has no action to carry
            # it, so it is attached here, where it adds no duplication for anyone.
            final_thought = (
                _extract_thought(None, response, fallback_response=True)
                if self.carries_thought and not getattr(response, "tool_calls", None)
                else None
            )
            yield BaseEvent(
                event_type=ExecutionEventType.LLM_END,
                event_id=str(uuid.uuid4()),
                timestamp=datetime.now(),
                author=agent.name,
                content=[Part(text="LLM response received", thought=final_thought)],
                is_partial=False,
                metadata=token_usage_metadata(token_usage),
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
                        f"Malformed tool-call arguments after {MAX_TOOL_CALL_ARGS_RETRIES} retries"
                    ),
                    is_partial=False,
                )
                break

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
            # With thought, that text is the reasoning, carried by the first action.
            if (
                response_text
                and has_tool_calls
                and not agent.stream
                and not self.use_stop
                and not self.carries_thought
            ):
                yield BaseEvent(
                    event_type=ExecutionEventType.RESPONSE,
                    content=extract_parts(response_text),
                    author=agent.name,
                    is_partial=True,
                )

            # ── Process tool calls ────────────────────────────────────────────
            if has_tool_calls:
                batches = self.batches(
                    agent, getattr(response, "tool_calls", None) or [], interaction_context
                )
                for position, tool_call in enumerate(response.tool_calls):
                    args = parsed_tool_args.get(tool_call.id, {})
                    # The reasoning behind this call, carried by the action it
                    # produced (``carries_thought``). Extraction tolerates malformed
                    # arguments: keeping the call's id intact through TOOL_START keeps
                    # the assistant/tool pair valid for strict providers.
                    thought = self._thought_of(tool_call, response, first=position == 0)

                    tool_name = tool_call.function.name
                    tool = agent.get_tool_by_name(tool_name)

                    if tool:
                        action = ToolAction(
                            call_id=tool_call.id,
                            name=tool_name,
                            parameters=args,
                            thought=thought,
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
                            call_id=tool_call.id,
                            name=tool_name,
                            parameters=args,
                            thought=thought,
                            message=thought,
                        )
                        yield BaseEvent(
                            event_type=ExecutionEventType.AGENT_CALL,
                            author=agent.name,
                            action=agent_action,
                            metadata=(
                                {METADATA_TOOL_BATCH: batches[tool_call.id]}
                                if tool_call.id in batches
                                else {}
                            ),
                        )
                        if tool_call.id not in batches:
                            return
                    elif tool_name == self.context_manager.stop_tool_name and self.use_stop:
                        for not_run in calls_not_run(
                            agent.name,
                            (getattr(response, "tool_calls", None) or [])[position + 1 :],
                            parsed_tool_args,
                            tool_name,
                        ):
                            yield not_run
                        yield BaseEvent(
                            event_type=ExecutionEventType.AGENT_END,
                            author=agent.name,
                            action=ToolAction(
                                call_id=tool_call.id,
                                name=tool_name,
                                parameters=args,
                                thought=thought,
                            ),
                            content=[
                                Part(text=args.get("info"))
                            ],  # Only what's in AGENT_END content goes into llm messages, "info" is the only parameter we expect on STOP
                        )
                        return
                    elif tool_name == self.context_manager.ask_user_tool_name and self.use_stop:
                        for not_run in calls_not_run(
                            agent.name,
                            (getattr(response, "tool_calls", None) or [])[position + 1 :],
                            parsed_tool_args,
                            tool_name,
                        ):
                            yield not_run
                        yield BaseEvent(
                            event_type=ExecutionEventType.RESPONSE,
                            content=extract_parts(args.get("text", "")),
                            author=agent.name,
                            need_feedback=True,
                            action=ToolAction(name=tool_name, parameters=args, thought=thought),
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
                            for not_run in calls_not_run(
                                agent.name,
                                (getattr(response, "tool_calls", None) or [])[position + 1 :],
                                parsed_tool_args,
                                tool_name,
                            ):
                                yield not_run
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
                        if tool_call.id in batches:
                            # In a batch the runner answers it, after the last call
                            # of the batch: a result here would split the response.
                            yield BaseEvent(
                                event_type=ExecutionEventType.TOOL_START,
                                author=agent.name,
                                action=action,
                                metadata={METADATA_TOOL_BATCH: batches[tool_call.id]},
                            )
                            continue
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
