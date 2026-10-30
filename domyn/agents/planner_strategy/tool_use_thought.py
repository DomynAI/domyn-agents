"""ReAct-style tool-use planner strategy.

Extends ``ToolUsePlannerStrategy`` with explicit thought emission:
the model's reasoning (from ``reasoning_content``, ``thinking_blocks``,
or preamble ``content`` when accompanied by tool calls) is extracted and
surfaced as ``Part(thought=...)`` inside the ``LLM_END`` event, and propagated
into every action's ``thought`` field so that ``ToolUseContextManager``
correctly reconstructs the history.

When the model has no native thinking capability, the companion
``ToolUseWithThoughtContextManager`` instructs it to emit plain-text reasoning
as ``content`` before tool calls; that text is captured as the thought.

When content appears **without** tool calls it is the response to the user
(streamable), not a thought.
"""

import json
import traceback
import uuid
from collections.abc import AsyncGenerator
from datetime import datetime
from typing import Any

from domyn.agents.core import (
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
from domyn.agents.planner_strategy.context_injection import (
    insert_turn_context,
    turn_context_messages,
)
from domyn.agents.planner_strategy.context_management.tool_use import (
    ToolUseWithThoughtContextManager,
)
from domyn.agents.planner_strategy.tool_use import (
    MAX_FV_SOFT_MISS_RETRIES,
    MAX_TOOL_CALL_ARGS_RETRIES,
    ToolUsePlannerStrategy,
    _create_event,
    _handle_forward_verbatim,
    _hard_miss_event,
    _parse_tool_call_args,
    malformed_args_feedback,
    parse_tool_call_args_with_errors,
    tool_batches,
)
from domyn.agents.planner_strategy.utils import (
    FORWARD_VERBATIM_FEEDBACK_HARD_FALLBACK,
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


class ToolUseWithThoughtPlannerStrategy(ToolUsePlannerStrategy):
    """Native tool-calling strategy with explicit thought emission (ReAct-style).

    Uses ``ToolUseWithThoughtContextManager`` by default, which appends a reasoning
    instruction to the system prompt so that the LLM emits plain-text ``content``
    before every tool call.  That content — or ``reasoning_content`` /
    ``thinking_blocks`` for models with native extended thinking — is then
    extracted and emitted as ``Part(thought=...)`` inside the ``LLM_END`` event.

    Thought extraction priority (first non-empty wins):
    1. ``reasoning_content``  — DeepSeek R1, Kimi, etc.
    2. ``thinking_blocks``    — Anthropic extended thinking
    3. ``response.content``   — preamble text solicited by the system prompt
    """

    type: str = "tool_use_thought"
    name: str = "tool_use_thought"
    context_manager: ToolUseWithThoughtContextManager = ToolUseWithThoughtContextManager()

    async def execute(
        self,
        agent: BaseAgent,
        interaction_context: InteractionContext,
        *,
        skills_payload: str = "",
        instruction: str | None = None,
    ) -> AsyncGenerator[BaseEvent, None]:
        """Execute the ReAct-tool-use loop.

        Same as the parent loop but:
        - emits ``LLM_END`` with ``content=[Part(thought=<thought>)]`` when a
          thought is present so downstream consumers can display it.
        - passes the thought through ``ToolAction.thought`` (and ``AgentAction.thought``)
          so that ``ToolUseContextManager`` correctly reconstructs the history.
        """
        logger = get_logger()
        iteration = 0
        # See ``ToolUsePlannerStrategy.execute`` for soft-miss semantics.
        fv_soft_miss_remaining = MAX_FV_SOFT_MISS_RETRIES
        # Last-ditch fallback after soft-miss budget is spent: nudge the LLM
        # to drop FORWARD_VERBATIM and respond directly using history info,
        # avoiding AGENT_ERROR when the model can't recover the FV target.
        fv_hard_fallback_used = False
        fv_soft_miss_carry: list[LLMMessage] = []
        # See ``ToolUsePlannerStrategy.execute``: one compact-and-retry per turn.
        overflow_retried = False

        while iteration < self.max_iterations:
            iteration += 1

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
            gave_up_on_malformed_json = False

            if isinstance(agent.llm_provider, DelegateLLMProvider):
                # Bidirectional: runner executes LLM, sends back LLMResponse via asend()
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
                # thought extraction still applies to delegated responses
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
                    while not generation_complete:
                        if agent.stream:
                            streamed_text_parts: list[str] = []
                            response = None
                            last_chunk = None
                            streamed_thought = False
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
                                if chunk.reasoning_content and not streamed_thought:
                                    streamed_thought = True
                                    yield BaseEvent(
                                        event_type=ExecutionEventType.RESPONSE_TOKEN,
                                        content=[Part(thought=chunk.reasoning_content.strip())],
                                        author=agent.name,
                                        is_partial=True,
                                    )
                                # Drop only no-op (None/empty) chunks. Whitespace-only
                                # chunks (paragraph breaks, list separators) carry
                                # semantic meaning and must reach the consumer.
                                if chunk.content:
                                    streamed_text_parts.append(chunk.content)
                                    if not self.use_stop:
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
                                streamed_text_parts = []
                        else:
                            response = await agent.llm_provider.generate(
                                messages=messages,
                                tools=tool_defs if tool_defs else None,
                            )
                            if not response.content and not response.tool_calls:
                                logger.error("LLM response has no content and no tool calls")

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
                                # See tool_use.py: a retry without the reason is
                                # the same request again, and gets the same
                                # answer.
                                messages.append(
                                    malformed_args_feedback(
                                        parse_errors,
                                        truncated=getattr(response, "truncated", False),
                                    )
                                )
                                logger.warning(
                                    (
                                        "Tool-call arguments were cut off by the generation limit"
                                        if getattr(response, "truncated", False)
                                        else "Malformed tool-call arguments"
                                    )
                                    + ", retrying LLM call "
                                    f"({malformed_json_retries}/{MAX_TOOL_CALL_ARGS_RETRIES})"
                                )
                                logger.debug(
                                    "Malformed tool-call arguments payload: "
                                    f"{response.tool_calls!r}"
                                )
                        else:
                            generation_complete = True
                except Exception as e:
                    # See `ToolUsePlannerStrategy.execute`: compacted and retried
                    # once when the provider rejects the request as too long.
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

            # See `ToolUsePlannerStrategy.execute`: the base the next request is
            # measured from.
            self.compactor.observe(
                getattr(response, "usage", None),
                usage_metadata,
                interaction_context.state,
                agent.name,
                getattr(agent, "llm_provider", None),
            )

            # Reasoning is normally carried by the action it produced, and
            # `test_react_strategy_doesnt_emit_thought_in_llm_end` pins that: on a
            # tool-calling turn LLM_END stays empty so consumers see the thought
            # once, on TOOL_START.
            #
            # A turn that answers directly has no action to carry it, and used to
            # drop the reasoning entirely — precisely the turn where a user is
            # most likely to wonder how the model got there. So it is attached
            # here only when there is nothing else to attach it to, which adds no
            # duplication for anyone.
            final_thought = (
                None
                if getattr(response, "tool_calls", None)
                else _extract_thought(None, response, fallback_response=True)
            )
            yield BaseEvent(
                event_type=ExecutionEventType.LLM_END,
                event_id=str(uuid.uuid4()),
                timestamp=datetime.now(),
                author=agent.name,
                content=[Part(text="LLM response received", thought=final_thought)],
                is_partial=False,
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

            # ── Content only (no tool calls) → terminal (response to user) ────
            if response_text and not has_tool_calls:
                if self.use_stop:
                    yield BaseEvent(
                        event_type=ExecutionEventType.AGENT_END,
                        author=agent.name,
                        action=ToolAction(
                            name=self.context_manager.stop_tool_name, parameters=None
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

            # ── Process tool calls ────────────────────────────────────────────
            if has_tool_calls:
                batches = (
                    tool_batches(agent, getattr(response, "tool_calls", None) or [])
                    if self.parallel_tool_calls
                    else {}
                )
                for i, tool_call in enumerate(response.tool_calls):
                    # ── Extract and emit thought ──────────────────────────────────────
                    # _extract_thought tolerates malformed arguments (never raises);
                    # the malformed-arguments case is handled below where args default
                    # to {}. Keeping tool_call.id intact through TOOL_START means the
                    # assistant/tool message pair stays valid for strict providers —
                    # no orphan role=tool message (empty call_id / no preceding
                    # assistant tool_calls) that would 400.
                    thought = _extract_thought(tool_call, response, fallback_response=i == 0)
                    thought = thought.strip() if thought else None  # normalize empty to None

                    tool_name = tool_call.function.name
                    tool = agent.get_tool_by_name(tool_name)

                    args = parsed_tool_args.get(tool_call.id, {})

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
                        )
                        return
                    elif tool_name == self.context_manager.stop_tool_name and self.use_stop:
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
                        if not self.use_stop and fv_soft_miss_remaining > 0:
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
                            break
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
                            break
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
                if self.use_stop:
                    yield BaseEvent(
                        event_type=ExecutionEventType.AGENT_END,
                        author=agent.name,
                        action=ToolAction(
                            call_id=tool_call.id,
                            name=self.context_manager.stop_tool_name,
                            parameters=None,
                            thought=thought,
                        ),
                        content=[
                            Part(text=response.content)
                        ],  # Only what's in AGENT_END content goes into llm messages
                    )
                else:
                    yield BaseEvent(
                        event_type=ExecutionEventType.RESPONSE,
                        content=extract_parts(response.content),
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
                    content=[Part(text=max_iterations_message)],
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
