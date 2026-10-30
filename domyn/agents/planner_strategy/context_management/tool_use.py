"""Tool-use context manager for native tool-calling agentic loops.

Builds LLM messages in the OpenAI/Anthropic native tool-use format
instead of embedding tool descriptions in the system prompt.
"""

import copy
import json
import uuid
from typing import Any

from domyn.agents.core import (
    METADATA_CONVERTED_TO_AGENT_END,
    METADATA_EXTERNAL_PART_AGENT_VISIBLE,
    BaseAgent,
    BaseEvent,
    ExecutionEventType,
    InteractionContext,
)
from domyn.agents.core.utils import parts_to_text
from domyn.agents.decisions.tools import hidden_for
from domyn.agents.llm import LLMMessage, LLMToolCall, ToolCallFunction
from domyn.agents.logger import get_logger
from domyn.agents.planner_strategy.context_management.base import ContextManager
from domyn.agents.planner_strategy.utils import (
    EVENT_ID_HEADER_FMT,
    FORWARD_VERBATIM_DESCRIPTION,
    FORWARD_VERBATIM_SOURCE_ID_DESCRIPTION,
    FORWARD_VERBATIM_SOURCE_KEY_DESCRIPTION,
    FORWARD_VERBATIM_TOOL_NAME,
    reorder_messages_around_tool_calls,
)
from domyn.agents.tools import FunctionTool
from domyn.agents.tools.base import FunctionDeclaration, Param
from domyn.agents.tools.read_tool_result import (
    READ_TOOL_RESULT,
    ShownTexts,
    ToolResults,
    characters,
    read_back,
    read_call,
)

_FORWARD_VERBATIM_INSTRUCTION = (
    "FORWARD_VERBATIM is only for terminal pass-through. Before calling it, "
    "identify the exact payload that will be emitted: with an empty source_key, "
    "the whole observation; with a source_key, only that resolved leaf. That "
    "exact payload must be a complete, self-contained, user-ready natural "
    "language or markdown answer. Full inline markdown links like "
    "`[title](https://...)` are self-contained; bare citation or reference "
    "markers are not. NEVER use FORWARD_VERBATIM when the emitted payload "
    "contains markers such as `[0]`, `[1]`, `[^note]`, `[label][id]`, anchor "
    "links, 'see above', or `\\ref{}` whose target, URL, or definition lives "
    "outside that same payload. If the prose is wrapped inside a structured "
    "payload, source_key must trace from observation root to the prose leaf, "
    "including every wrapper key, and resolve to a string, not a container. "
    "Copy source_id and source_key exactly; do not guess names such as `text`, "
    "`summary`, `result`, or the tool name, and do not drop path segments like "
    "`sec-7`, `bucket_07`, or `z9`. "
    "When the resolved payload IS already the user-ready answer and passes "
    "every check above, USE FORWARD_VERBATIM rather than re-emitting the "
    "same prose in your own tokens. Do not paraphrase, do not wrap it in "
    "framing sentences like 'Here is the X' or '## Section', and do not "
    "re-type it line by line — that wastes tokens and risks introducing "
    "typos or dropped words. FORWARD_VERBATIM exists precisely so the "
    "exact bytes of the resolved payload reach the user unchanged. "
    "Do not use FORWARD_VERBATIM for partial answers, metadata-only "
    "observations (listings, IDs, status flags, raw records), when "
    "the user asked for a different format, length, transformation, "
    "or language, or for synthesis across multiple observations. "
    "In all these cases, respond directly instead."
)


class ToolUseContextManager(ContextManager):
    """Context manager that builds messages for native tool-calling APIs."""

    type: str = "tool_use"
    name: str = "tool_use"
    max_tool_result_chars: int = 20_000

    stop_tool_name: str = "STOP"
    ask_user_tool_name: str = "ASK_USER_MISSING_INFO"
    respond_tool_name: str = "RESPOND"
    forward_verbatim_tool_name: str = FORWARD_VERBATIM_TOOL_NAME
    forward_verbatim_instruction: str = _FORWARD_VERBATIM_INSTRUCTION
    info_start_pattern: str = "<INFO>"
    info_end_pattern: str = "</INFO>"
    user_role: str = "user"
    assistant_role: str = "assistant"
    user_info_key: str = "user_info"

    async def get_context(
        self,
        context: InteractionContext,
        agent: BaseAgent,
        *,
        instruction: str | None = None,
        **kwargs: Any,
    ) -> list[LLMMessage]:
        """Build message list from conversation history in native tool-use format.

        ``instruction`` replaces ``agent.instruction`` in the system prompt — the
        agent's instruction with its standing instructions merged in.
        """
        messages: list[LLMMessage] = []
        forward_verbatim_enabled = self.is_forward_verbatim_enabled(agent)
        # Whether a cut or a dropped result can say how to read it back.
        reads_back = self._has_read_tool_result(agent, context)

        # System message with agent instructions (no tool format descriptions)
        system_content = self._build_system_prompt(agent, context, instruction)
        messages.append(LLMMessage(role="system", content=system_content))

        # Convert conversation + interaction history to messages
        event_to_parse = _rendered_events(context)

        # A root's AGENT_START is built from its USER_INPUT (same interaction, same text):
        # only that start duplicates the request, any other start carries a real task.
        user_requests = {
            (e.interaction_id, self._extract_text(e))
            for e in event_to_parse
            if e.event_type == ExecutionEventType.USER_INPUT
        }

        # Where each event's messages start, so every message can be traced back
        # to the event it came from once the loop is done. Recorded as the first
        # statement of the body rather than stamped at the end of it: an early
        # `continue` would skip a trailing stamp, and there are already four of
        # them. Recorded here it cannot be missed by a fifth.
        boundaries: list[tuple[str | None, int]] = []

        # The text of each tool result as the model is shown it, which is what
        # `read_tool_result` reads too. What a streamable tool showed the user
        # while it ran reaches the model inside that tool's result, not as
        # assistant turns: those landed after the result and ended the request on
        # an assistant message, which some models answer with nothing.
        texts = ShownTexts()

        for event in event_to_parse:
            boundaries.append((event.event_id, len(messages)))

            match event.event_type:
                case ExecutionEventType.USER_INPUT:
                    messages.append(self._get_user_request_message(event))

                case ExecutionEventType.TOOL_START:
                    if event.author != agent.name:
                        # Foreign tool call (descendant/peer): do NOT render as
                        # an assistant tool_call — the consumer LLM reads its
                        # assistant turns as ITS OWN past actions and starts
                        # calling tools it does not have (ToolNotFound
                        # carousel). The paired TOOL_END/TOOL_ERROR below
                        # renders the whole call as attributed narration.
                        continue
                    if event.action:
                        thought = event.action.thought or ""
                        tool_call = LLMToolCall(
                            id=event.action.call_id,
                            type="function",
                            function=ToolCallFunction(
                                name=event.action.name,
                                arguments=json.dumps(
                                    event.action.parameters or {}, ensure_ascii=False
                                ),
                            ),
                        )
                        # Group consecutive TOOL_START events into one assistant message
                        # so parallel tool calls share a single assistant turn.
                        if (
                            messages
                            and messages[-1].role == "assistant"
                            and messages[-1].tool_calls
                        ):
                            messages[-1].tool_calls.append(tool_call)
                            # Merge thought if present
                            if thought:
                                existing = messages[-1].content or ""
                                messages[-1].content = (
                                    f"{existing}\n{thought}".strip() if existing else thought
                                )
                        else:
                            messages.append(
                                LLMMessage(
                                    role=self.assistant_role,
                                    tool_calls=[tool_call],
                                )
                            )

                case ExecutionEventType.TOOL_END | ExecutionEventType.TOOL_ERROR:
                    if event.action:
                        # Build the event_id header first so the truncation
                        # budget accounts for it — total content stays within
                        # max_tool_result_chars. The header lets the LLM
                        # reference this observation by event_id in a later
                        # FORWARD_VERBATIM call.
                        #
                        # Header exposure is controlled by the current
                        # planner only. If this event is already in the
                        # current agent's visible context, the agent may
                        # forward it when its own FV tool is enabled; a
                        # sub-agent's FV flag only controls what that sub can
                        # do inside its own planner.
                        header = (
                            EVENT_ID_HEADER_FMT.format(event_id=event.event_id) + "\n"
                            if event.event_id and forward_verbatim_enabled
                            else ""
                        )
                        body_budget = self.max_tool_result_chars - len(header)
                        if body_budget <= 0:
                            # Header alone meets/exceeds budget — drop header so
                            # observation content survives within the limit.
                            # NOTE: dropping the header breaks FORWARD_VERBATIM
                            # for this observation (no event_id for the LLM to
                            # reference). Surface it loudly so misconfigured
                            # max_tool_result_chars doesn't fail silently.
                            get_logger().warning(
                                f"max_tool_result_chars={self.max_tool_result_chars} is smaller "
                                f"than the event_id header ({len(header)} chars) for "
                                f"event_id={event.event_id}; dropping header — FORWARD_VERBATIM "
                                "will not be able to reference this tool result."
                            )
                            header = ""
                            body_budget = self.max_tool_result_chars
                        observation = _keep_start_and_end(
                            texts.text_of(event),
                            body_budget,
                            event.event_id if reads_back else None,
                        )
                        if event.author != agent.name:
                            # Foreign tool result: attributed user-role
                            # narration (its TOOL_START was skipped above).
                            # The event_id header stays inside so the consumer
                            # can still FORWARD_VERBATIM this observation.
                            args_str = json.dumps(event.action.parameters or {}, ensure_ascii=False)
                            messages.append(
                                LLMMessage(
                                    role=self.user_role,
                                    content=(
                                        f"Agent: {event.author} called tool "
                                        f"{event.action.name}({args_str})\n"
                                        f"{header}Result: {observation}"
                                    ),
                                )
                            )
                        else:
                            observation = header + observation
                            messages.append(
                                LLMMessage(
                                    role="tool",
                                    tool_call_id=event.action.call_id,
                                    name=event.action.name,
                                    content=observation,
                                    # Only a tool's result: the other tool-role
                                    # messages, a sub-agent's task or report,
                                    # have no result to read back.
                                    read_back=read_back(event) if reads_back else None,
                                )
                            )

                case ExecutionEventType.TOOL_FEEDBACK:
                    if event.action:
                        observation = str(event.action.observation or "")
                        messages.append(LLMMessage(role=self.user_role, content=observation))

                case ExecutionEventType.EXTERNAL_PART:
                    # Externally-injected content (e.g. from a channel) only
                    # reaches the model when explicitly marked visible to it —
                    # see METADATA_EXTERNAL_PART_AGENT_VISIBLE.
                    if event.metadata.get(METADATA_EXTERNAL_PART_AGENT_VISIBLE, False):
                        text = self._extract_text(event)
                        if text:
                            messages.append(LLMMessage(role=self.user_role, content=text))

                case ExecutionEventType.RESPONSE:
                    # Subagent responses converted to AGENT_END by the runner are
                    # rendered via the synthesized AGENT_END; skip the duplicate.
                    if event.metadata.get(METADATA_CONVERTED_TO_AGENT_END):
                        continue
                    if texts.take_chunk(event):
                        # A chunk a streamable tool yielded: shown inside its
                        # result (see `texts`).
                        continue
                    text = self._extract_text(event)
                    if text and event.author != agent.name:
                        # Foreign answer: as an assistant turn it would read as the consumer's own.
                        messages.append(
                            LLMMessage(
                                role=self.user_role,
                                content=f"Agent: {event.author} replied to the user: {text}",
                            )
                        )
                    elif text:
                        messages.append(LLMMessage(role=self.assistant_role, content=text))

                case ExecutionEventType.AGENT_END:
                    # AGENT_END is inbound feedback from a sub-agent to its
                    # parent, rendered as a self-contained STOP tool-call pair
                    # (see below).

                    info = self._extract_text(
                        event
                    )  # Only what's in AGENT_END content goes into llm messages
                    if not info:  # If content is none skip, keep this guard to avoid double assistant messages with events RESPONSE + AGENT_END (with no content)
                        continue
                    # event_id header surfaced when the caller is FV-capable
                    # (has the FORWARD_VERBATIM tool registered). Visibility
                    # has already been decided by the context; the emitting
                    # sub-agent's FV flag only controls whether that sub can
                    # call FORWARD_VERBATIM inside its own planner.
                    header = (
                        EVENT_ID_HEADER_FMT.format(event_id=event.event_id)
                        if event.event_id and forward_verbatim_enabled
                        else None
                    )
                    content = (
                        f"Agent: {event.author} ended"
                        f"{f' {header}' if header else ''}.\n"
                        f"Information from this agent's execution: {info}"
                    )
                    # Two renderings depending on the CONSUMER's relationship
                    # with the STOP tool (perspective-aware):
                    #
                    # 1. Consumer has ``use_stop`` (STOP is among its tools):
                    #    render a self-contained STOP tool-call pair (mirrors
                    #    how AGENT_CALL/AGENT_START close their pair). The
                    #    synthetic call_id only needs to match within the pair.
                    #    Arguments are replayed FAITHFULLY: a hardcoded "{}"
                    #    falsifies the in-context examples — agents re-read
                    #    STOP({}) as a successful call and imitate it, dropping
                    #    ``info`` (payload then silently vanishes via the
                    #    empty-content guard above). When the event has no
                    #    recorded args (framework-fabricated prose promotions),
                    #    reconstruct them as {"info": <content>} so every STOP
                    #    example in history is well-formed.
                    #
                    # 2. Consumer does NOT have STOP (e.g. top-level agent with
                    #    ``use_stop=false``): a STOP tool_use in its history is
                    #    pure bait — observed empirically: the agent calls the
                    #    undeclared STOP and burns an iteration on ToolNotFound.
                    #    Render an attributed user-role narration instead.
                    #    User role also keeps the list from ending on an
                    #    assistant turn (Anthropic strict prefill rejection),
                    #    same guarantee the pair provides.
                    consumer_has_stop = bool(getattr(agent.planner, "use_stop", False))
                    if consumer_has_stop:
                        stop_call_id = f"stop_{event.event_id or len(messages)}"
                        if (
                            event.action
                            and event.action.parameters
                            and event.action.name == self.stop_tool_name
                        ):
                            stop_args = json.dumps(event.action.parameters, ensure_ascii=False)
                        else:
                            stop_args = json.dumps({"info": info}, ensure_ascii=False)
                        messages.append(
                            LLMMessage(
                                role=self.assistant_role,
                                tool_calls=[
                                    LLMToolCall(
                                        id=stop_call_id,
                                        type="function",
                                        function=ToolCallFunction(
                                            name=self.stop_tool_name,
                                            arguments=stop_args,
                                        ),
                                    )
                                ],
                            )
                        )
                        messages.append(
                            LLMMessage(
                                role="tool",
                                tool_call_id=stop_call_id,
                                name=self.stop_tool_name,
                                content=content,
                            )
                        )
                    else:
                        messages.append(LLMMessage(role=self.user_role, content=content))

                case ExecutionEventType.AGENT_CALL:
                    # calling agent invoked a sub-agent
                    if event.author != agent.name:
                        # Foreign delegation (e.g. a child delegating to a
                        # grandchild): rendering it as an assistant tool_call
                        # makes the consumer believe IT can invoke that agent
                        # (ToolNotFound carousel). Skip — the callee's
                        # AGENT_START below already narrates author + task.
                        continue
                    tool_call = LLMToolCall(
                        id=event.action.call_id,
                        type="function",
                        function=ToolCallFunction(
                            name=event.action.name,
                            arguments=json.dumps(event.action.parameters or {}, ensure_ascii=False),
                        ),
                    )
                    messages.append(LLMMessage(role=self.assistant_role, tool_calls=[tool_call]))

                case ExecutionEventType.AGENT_START:
                    # called agent receiving its task assignment
                    task = event.content[0].text if event.content else ""
                    if not task and context.agent_input:
                        task = (
                            context.agent_input.content[0].text
                            if context.agent_input.content
                            else ""
                        )
                    msg_content = f"Agent: {event.author}\nTask assigned: {task}\n{self._get_user_info(event)}"
                    if messages and messages[-1].tool_calls:
                        # Close parent's tool_call pair (OpenAI strict: PAIRING + CONTIGUITY)
                        parent_call = messages[-1].tool_calls[0]
                        messages.append(
                            LLMMessage(
                                role="tool",
                                content=msg_content,
                                tool_call_id=parent_call.id,
                                name=parent_call.function.name,
                            )
                        )
                    elif (event.interaction_id, self._extract_text(event)) not in user_requests:
                        # Never skip on role alone: a re-called sub's task follows its user-role
                        # AGENT_END, a GLOBAL sub's follows the root's USER_INPUT.
                        messages.append(LLMMessage(role=self.user_role, content=msg_content))

        # Attribution, and it is bookkeeping only — `source_event_id` is excluded
        # from the wire format. It is what lets a compaction record *which*
        # events it dropped, so the decision survives this method running again
        # next turn and rebuilding everything from scratch. Without it a record
        # could only ever name tool results, and the conversation text — the part
        # that grows and that pruning cannot touch — would be unnameable.
        boundaries.append((None, len(messages)))
        for index in range(len(boundaries) - 1):
            event_id, start = boundaries[index]
            end = boundaries[index + 1][1]
            for message in messages[start:end]:
                if message.source_event_id is None:
                    message.source_event_id = event_id

        return reorder_messages_around_tool_calls(messages)

    def get_special_tools(
        self, use_stop: bool, forward_verbatim_enabled: bool
    ) -> list[FunctionTool]:
        special_tools: list[FunctionTool] = []

        if use_stop:
            _stop = FunctionTool(func=lambda: None, name=self.stop_tool_name, description="")
            _stop.set_declaration(
                FunctionDeclaration(
                    name=self.stop_tool_name,
                    description="When you have completed your task, use this tool to return the data you have found to the agent that invoked you. Pass all the information you deem relevant in the 'info' field. Don't provide the final answer to the user but use this tool to return to the previous agent first.",
                    parameters=[
                        Param(
                            name="info",
                            type="str",
                            description="The information to return to the calling agent",
                        )
                    ],
                )
            )
            _ask = FunctionTool(func=lambda: None, name=self.ask_user_tool_name, description="")
            _ask.set_declaration(
                FunctionDeclaration(
                    name=self.ask_user_tool_name,
                    description="Use this action to ask questions to the user if you can't proceed without more information. Don't use this to provide the final answer to the user, use the stop tool instead.",
                    parameters=[
                        Param(name="text", type="str", description="The question to ask the user")
                    ],
                )
            )
            special_tools.extend([_stop, _ask])

        if forward_verbatim_enabled:
            _fwd = FunctionTool(
                func=lambda: None, name=self.forward_verbatim_tool_name, description=""
            )
            _fwd.set_declaration(
                FunctionDeclaration(
                    name=self.forward_verbatim_tool_name,
                    description=FORWARD_VERBATIM_DESCRIPTION,
                    parameters=[
                        Param(
                            name="source_id",
                            type="str",
                            description=FORWARD_VERBATIM_SOURCE_ID_DESCRIPTION,
                        ),
                        Param(
                            name="source_key",
                            type="str",
                            description=FORWARD_VERBATIM_SOURCE_KEY_DESCRIPTION,
                        ),
                    ],
                )
            )
            special_tools.append(_fwd)

        return special_tools

    def get_tool_definitions(
        self, agent: BaseAgent, context: InteractionContext
    ) -> list[dict[str, Any]]:
        """Get native tool definitions for all agent tools and handoff targets.

        Every source ``tool_definition`` (from ``agent.tools`` and from special
        tools) is deep-copied at append time so the returned list owns dicts
        that are fully independent from the source objects. The subsequent
        ``reasoning`` injection therefore never touches
        ``tool.tool_definition`` / ``DelegateTool.tool_parameters`` /
        ``agent_desc.invocation_parameters``, even at the container level
        (no ``parameters`` key reassignment leaking across requests).
        """
        tool_defs: list[dict[str, Any]] = []

        # A decision model may have judged some of this agent's tools irrelevant
        # to this turn (`decisions/tools.py`). It is decided once per turn, in
        # the runner, and only ever *removes*: an empty set here — no model, no
        # decision, another turn's decision — is the full list.
        hidden = hidden_for(context)

        # Add regular tools (deep-copied to keep ``tool.tool_definition``
        # immutable across requests).
        for tool in agent.tools:
            if getattr(tool, "name", None) in hidden:
                continue
            if hasattr(tool, "tool_definition"):
                tool_defs.append(copy.deepcopy(tool.tool_definition))

        # An agent with tools, or agents to hand off to, can read back a result
        # cut in `get_context` or dropped by compaction: both give the call. No
        # agent lists the tool; `Agent.get_tool_by_name` resolves its name.
        if self._has_read_tool_result(agent, context):
            tool_defs.append(copy.deepcopy(READ_TOOL_RESULT.tool_definition))

        # Add explicit handoff targets as callable tools. The ``parameters`` slot
        # references ``agent_desc.invocation_parameters`` directly, so the
        # wrapper dict must also be built around a deep copy.
        for agent_desc in context.handoff_targets_description:
            agent_tool_def: dict[str, Any] = {
                "type": "function",
                "function": {
                    "name": agent_desc.name,
                    "description": agent_desc.description,
                    "parameters": copy.deepcopy(agent_desc.invocation_parameters)
                    if agent_desc.invocation_parameters
                    else {
                        "type": "object",
                        "properties": {},
                    },
                },
            }
            tool_defs.append(agent_tool_def)

        # special tools always append at bottom (deep-copied for the same
        # isolation reason as regular tools).
        use_stop: bool = getattr(agent.planner, "use_stop", False)
        for tools in self.get_special_tools(use_stop, self.is_forward_verbatim_enabled(agent)):
            tool_defs.append(copy.deepcopy(tools.tool_definition))

        # Inject ``reasoning`` directly on the now-owned copies. The dedup
        # guard is kept belt-and-braces for callers that pre-populate
        # ``required`` with ``reasoning``.
        for tool in tool_defs:
            params = tool["function"].setdefault("parameters", {"type": "object", "properties": {}})
            params.setdefault("properties", {})["reasoning"] = {
                "description": "Reasoning that brought the agent to choose this tool. Be concise.",
                "type": "string",
            }
            existing_required = params.get("required", [])
            if "reasoning" not in existing_required:
                params["required"] = [*existing_required, "reasoning"]

        return tool_defs

    def tool_results(self, agent: BaseAgent, context: InteractionContext) -> ToolResults:
        """The tool results *agent* sees in *context*, as this manager shows them
        to its model: what ``read_tool_result`` reads, in parts that reach the
        model whole."""
        return ToolResults(
            events=lambda: _rendered_events(context), budget=self._result_budget(agent)
        )

    def _result_budget(self, agent: BaseAgent) -> int:
        """What a result the runner records for *agent* keeps of
        ``max_tool_result_chars``: all of it, less the event-id header when
        `get_context` writes one."""
        if not self.is_forward_verbatim_enabled(agent):
            return self.max_tool_result_chars
        budget = self.max_tool_result_chars - len(_RUNNER_EVENT_ID_HEADER)
        return budget if budget > 0 else self.max_tool_result_chars

    @staticmethod
    def _has_read_tool_result(agent: BaseAgent, context: InteractionContext) -> bool:
        """Whether *agent* gets ``read_tool_result``: one with tools of its own or
        agents to hand off to, unless one of its tools has that name.

        An agent with neither sends no tools because of it: a model server without
        tool calling, such as a vLLM started without ``--enable-auto-tool-choice``,
        rejects a request with any, and such an agent has hardly a result to read
        back. One with a tool by that name keeps its own: two definitions with one
        name are a request providers reject, and a note giving the call would send
        the model to the other tool."""
        names = [getattr(tool, "name", None) for tool in agent.tools]
        if READ_TOOL_RESULT.name in names:
            return False
        return bool(names or context.handoff_targets_description)

    def _build_system_prompt(
        self, agent: BaseAgent, context: InteractionContext, instruction: str | None = None
    ) -> str:
        """Build a clean system prompt with agent instructions only."""
        parts: list[str] = []

        if instruction is None:
            instruction = getattr(agent, "instruction", None)
        if instruction:
            parts.append(instruction)

        if not parts:
            parts.append(f"You are {agent.name}. {agent.description}")

        if self.is_forward_verbatim_enabled(agent):
            parts.append(self.forward_verbatim_instruction)

        return "\n\n".join(parts)

    @staticmethod
    def _extract_text(event: Any) -> str:
        """Concatenate the renderable fields of an event's content parts."""
        return parts_to_text(getattr(event, "content", None))

    def _get_user_info(self, event: BaseEvent) -> str:
        info = ""
        if user_info := event.metadata.get(self.user_info_key, {}):
            if isinstance(user_info, dict):
                for k, v in user_info.items():
                    info += f"{k.upper()}: {v}\n"
            else:
                info = str(user_info)

        return f"{self.info_start_pattern}\n{info}{self.info_end_pattern}\n\n" if info else ""

    def _get_user_request_message(self, event: BaseEvent) -> LLMMessage:
        message = self._get_user_info(event)

        text = self._extract_text(event)
        if text:
            message += f"USER INPUT: {text}"

        return LLMMessage(role=self.user_role, content=message)


_REACT_THOUGHT_INSTRUCTION = """Never generate text when calling a tool. Instead, if you have any reasoning or thoughts that led you to call the tool, include them in the field dedicated to reasoning. Always include reasoning if you have it, even if it's just a few words.
The content field MUST be reserved for information you want to share with the user, and MUST NOT include any information about your thought process or tool calls."""


class ToolUseWithThoughtContextManager(ToolUseContextManager):
    """Context manager for ReAct-style tool-use loops.

    Identical to ``ToolUseContextManager`` but appends a reasoning instruction
    to the system prompt so that the LLM emits plain-text ``content`` before
    every tool call.  That content is then captured as the ``thought`` by
    ``ToolUseWithThoughtPlannerStrategy``.
    """

    type: str = "thought_tool_use"
    name: str = "thought_tool_use"
    thought_instruction: str = _REACT_THOUGHT_INSTRUCTION

    def _build_system_prompt(
        self, agent: BaseAgent, context: InteractionContext, instruction: str | None = None
    ) -> str:
        """Build system prompt with an added reasoning instruction."""
        base = super()._build_system_prompt(agent, context, instruction)
        return f"{base}\n\n{self.thought_instruction}"


def _rendered_events(context: InteractionContext) -> list[BaseEvent]:
    """The events a request is rendered from, in order and each once: the
    conversation, the input that started this agent when it carries an action,
    then this interaction, which can repeat events the conversation has."""
    events = list(context.conversation_history)
    if context.agent_input and context.agent_input.action:
        events.append(context.agent_input)
    events += context.interaction_history
    seen: set[str | None] = set()
    unique: list[BaseEvent] = []
    for event in events:
        if event.event_id not in seen:
            seen.add(event.event_id)
            unique.append(event)
    return unique


# The header `get_context` writes in front of a result the runner recorded, whose
# ids all have one shape (`Runner._ensure_event_identifiers`).
_RUNNER_EVENT_ID_HEADER = EVENT_ID_HEADER_FMT.format(event_id=f"event_{uuid.UUID(int=0)}") + "\n"

# Of what a cut result keeps, the share taken from its start. Less than half: the
# end is where tools put what matters most — the last error, the "and N more",
# the note saying where to read on.
_HEAD_SHARE = 0.4
# Below this, a budget spent on the notice leaves nothing worth reading around
# it, and a plain cut of the start shows more.
_MIN_KEPT_CHARS = 200


def _keep_start_and_end(text: str, budget: int, result_id: str | None) -> str:
    """``text`` in at most ``budget`` characters: its start, a notice, its end.

    The notice says which characters it leaves out, that they cannot be told
    from the rest and, given the result's id, the call that reads them. Both cuts
    land on a line boundary when one is near. A budget too small for both ends
    gets the plain cut of the start this used to make.
    """
    if len(text) <= budget:
        return text
    total = len(text)

    def notice(start: int, end: int) -> str:
        if not result_id:
            return f"\n[{characters(start, end, total)} are not shown here]\n"
        # Said in so many words: told only how much was missing, models took the
        # start and the end for the whole result and filled in the middle from
        # them, answering from the pattern of what they could see.
        return (
            f"\n[{characters(start, end, total)} are not shown here, and what they "
            "contain cannot be told from the rest. Before drawing any conclusion that "
            f"depends on them, read them with {read_call(result_id, start)}]\n"
        )

    # Neither number is ever wider than the total, so this is the most room it takes.
    room = budget - len(notice(total, total))
    if room < _MIN_KEPT_CHARS:
        return text[:budget] + "\n...(truncated)"
    head_size = int(room * _HEAD_SHARE)
    tail_size = room - head_size
    head = text[:head_size]
    cut = head.rfind("\n")
    if cut >= head_size // 2:
        head = head[: cut + 1]
    tail = text[-tail_size:]
    cut = tail.find("\n")
    if 0 <= cut < tail_size // 2:
        tail = tail[cut + 1 :]
    return head + notice(len(head), total - len(tail)) + tail
