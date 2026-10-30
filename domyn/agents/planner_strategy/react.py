"""React logic implementation for agents.

Avoids importing the concrete ``Agent`` class at runtime to prevent circular
imports (Agent -> planner_strategy -> react -> Agent). The ``Agent`` symbol is
only needed for type checking, so it's imported under ``TYPE_CHECKING``.
"""

import traceback
import uuid
from collections.abc import AsyncGenerator
from datetime import datetime
from typing import Any, Literal

from httpx import HTTPError, HTTPStatusError, TimeoutException
from pydantic import ConfigDict, Field, field_serializer, field_validator

from domyn.agents.core import (
    AGENT_ERROR_CODE,
    AGENT_ERROR_MESSAGE,
    METADATA_FORWARD_VERBATIM,
    METADATA_FORWARD_VERBATIM_SOURCE_ID,
    METADATA_FORWARD_VERBATIM_SOURCE_KEY,
    AgentAction,
    BaseAgent,
    BaseEvent,
    ExecutionEventType,
    InteractionContext,
    Part,
    ToolAction,
)
from domyn.agents.core.utils import extract_parts
from domyn.agents.llm import BaseLLMProvider, LLMMessage, StructuredOutput
from domyn.agents.logger import get_logger
from domyn.agents.planner_strategy import ContextManagementType
from domyn.agents.planner_strategy.base import BasePlannerStrategy
from domyn.agents.planner_strategy.context_injection import inject_turn_context
from domyn.agents.planner_strategy.context_management.base import ContextManager
from domyn.agents.planner_strategy.context_management.full_messages import FullMessageContext
from domyn.agents.planner_strategy.context_management.summary import (
    OldEventsSummaryContext,
    PreviousAgentsSummaryContext,
    PreviousInteractionsSummaryContext,
)
from domyn.agents.planner_strategy.prompting.data import ResponseType
from domyn.agents.planner_strategy.prompting.structural_tags_prompting import (
    StructuralTagMessagePrompting,
)
from domyn.agents.planner_strategy.utils import (
    FORWARD_VERBATIM_MISS_UNRESOLVED,
    resolve_verbatim,
)


def create_response_event(message: str, metadata: dict[str, Any] | None = None) -> BaseEvent:
    return BaseEvent(
        event_type=ExecutionEventType.RESPONSE,
        event_id=str(uuid.uuid4()),
        timestamp=datetime.now(),
        author="react_strategy",
        content=extract_parts(message),
        is_partial=False,
        # TODO: pass metadata through (metadata or {}); the caller at max-iterations passes
        # agent.name here, likely meant as author.
    )


def create_error_event(
    error: str,
    agent_name: str,
    metadata: dict[str, Any] | None = None,
    error_code: str | None = None,
) -> BaseEvent:
    return BaseEvent(
        event_type=ExecutionEventType.ERROR_OCCURRED,
        event_id=str(uuid.uuid4()),
        timestamp=datetime.now(),
        author=agent_name,
        content=[Part(text=error)],
        error_code=error_code,
        error_message=error,
        is_partial=False,
        # TODO: pass metadata through
    )


def create_event(event_type: ExecutionEventType, content: list[Part], author: str) -> BaseEvent:
    return BaseEvent(
        event_type=event_type,
        event_id=str(uuid.uuid4()),
        timestamp=datetime.now(),
        author=author,
        content=content,
        is_partial=False,
    )


def _default_context_manager() -> FullMessageContext:
    """The context react gets when none is given.

    Full messages in the structural-tags protocol its parser reads, with tool
    results narrowed to the agents this one can see.
    """
    return FullMessageContext(
        narrow_non_visible_agents=True,
        name="default_full_message_context",
        prompting=StructuralTagMessagePrompting(),
    )


class ReactPlannerStrategy(BasePlannerStrategy):
    """React (Reasoning and Acting) planner strategy implementation.

    The tool protocol travels in the prompt text, so it needs no native tool
    calls from the endpoint. ``ReactPlannerStrategy()`` alone is complete.
    """

    model_config = ConfigDict(use_enum_values=True)

    type: Literal["react"] = "react"
    use_stop: bool = False
    stream: bool = False
    context_manager: ContextManager = Field(default_factory=_default_context_manager)
    name: str = "react"
    _disable_structured_outputs: bool = False

    @field_validator("context_manager", mode="wrap")
    @classmethod
    def validate_context_manager(cls, v: Any, handler: Any, info: Any) -> ContextManager:
        if isinstance(v, ContextManager):
            return v

        if isinstance(v, dict):
            prompt_type = v.get("type")
            if prompt_type:
                try:
                    # Dispatch on context-manager type inline (no factory) to avoid circular imports
                    if prompt_type == ContextManagementType.FULL_MESSAGES.value:
                        return FullMessageContext(**v)
                    elif prompt_type == ContextManagementType.AGENT_SUMMARY.value:
                        return PreviousAgentsSummaryContext(**v)
                    elif prompt_type == ContextManagementType.OLD_SUMMARY.value:
                        return OldEventsSummaryContext(**v)
                    elif prompt_type == ContextManagementType.INTERACTIONS_SUMMARY.value:
                        return PreviousInteractionsSummaryContext(**v)
                    else:
                        raise ValueError(f"Unknown prompting type: {prompt_type}")
                except Exception as e:
                    get_logger().warning(
                        f"Warning: Could not deserialize prompt type '{prompt_type}': "
                        f"{type(e).__name__}"
                    )
                    get_logger().debug(f"Prompt type deserialization detail: {e}")

        if v is None:
            return _default_context_manager()

        try:
            return handler(v)
        except Exception:
            return _default_context_manager()

    @field_serializer("context_manager")
    def serialize_context_manager(self, value: Any) -> dict[str, Any] | Any:
        """Serialize context_manager to dict with automatic type detection."""
        if hasattr(value, "model_dump"):
            serialized = value.model_dump()
            return serialized
        return value

    async def execute(  # type: ignore
        self,
        agent: BaseAgent,
        interaction_context: InteractionContext,
        *,
        skills_payload: str = "",
        instruction: str | None = None,
    ) -> AsyncGenerator[BaseEvent, None]:
        """
        Execute the React logic: Reasoning and Acting in a loop.

        The React pattern works by:
        1. Thought: Reason about the current situation
        2. Action: Decide and execute an action
        3. Observation: Observe the result
        4. Repeat until done
        """

        iteration = 0

        while iteration < self.max_iterations:
            iteration += 1

            # Get next actions from LLM
            try:
                yield create_event(
                    event_type=ExecutionEventType.LLM_START,
                    content=[Part(text="Invoking LLM for next action")],
                    author=agent.name,
                )

                actions: list[BaseEvent] = []
                llm_ended = False
                async for event in self.next_actions(
                    agent,
                    interaction_context,
                    skills_payload=skills_payload,
                    instruction=instruction,
                ):
                    if event.is_partial:
                        # Stream partial token to caller immediately
                        yield event
                    else:
                        # First non-partial event marks end of LLM streaming phase
                        if not llm_ended:
                            llm_ended = True
                            yield create_event(
                                event_type=ExecutionEventType.LLM_END,
                                content=[Part(text="Invoking LLM for next action")],
                                author=agent.name,
                            )
                        actions.append(event)

                if not llm_ended:
                    yield create_event(
                        event_type=ExecutionEventType.LLM_END,
                        content=[Part(text="Invoking LLM for next action")],
                        author=agent.name,
                    )

                if not actions:
                    if self.verbose:
                        yield BaseEvent(
                            event_type=ExecutionEventType.RESPONSE,
                            content=[Part(thought="No more actions to execute, finishing")],
                            author=agent.name,
                        )
                    break

                # Execute actions and collect results
                for event in actions:
                    if (
                        event.action
                        and isinstance(event.action, ToolAction)
                        and (event.event_type != ExecutionEventType.TOOL_ERROR)
                    ):
                        tool = agent.get_tool_by_name(event.action.name)
                        if tool:
                            yield BaseEvent(
                                event_type=ExecutionEventType.TOOL_START,
                                author=agent.name,
                                action=event.action,
                            )
                        elif tool is None and event.event_type == ExecutionEventType.AGENT_END:
                            event.timestamp = datetime.now()
                            yield event
                            return
                        else:
                            if event.content:
                                yield BaseEvent(
                                    event_type=ExecutionEventType.RESPONSE,
                                    content=event.content,
                                    author=agent.name,
                                    action=event.action,
                                    need_feedback=event.need_feedback,
                                    is_partial=event.is_partial,
                                    metadata=event.metadata,
                                )
                            else:
                                action_copy = event.action.model_copy()
                                action_copy.observation = f"Tool {action_copy.name} not found"
                                yield BaseEvent(
                                    event_type=ExecutionEventType.TOOL_ERROR,
                                    author=agent.name,
                                    action=action_copy,
                                )
                    elif event.action and isinstance(event.action, AgentAction):
                        event.timestamp = datetime.now()
                        yield event
                    else:
                        event.timestamp = datetime.now()
                        yield event
                        if event.content and not event.action:
                            return

            except Exception as e:
                get_logger().error(f"Error in React iteration: {type(e).__name__}")
                get_logger().debug(
                    f"Error in React iteration {e}. Traceback: {traceback.format_exc()}"
                )
                yield create_error_event(
                    f"Error in React iteration {iteration}: {e}",
                    agent.name,
                    {"iteration": iteration},
                )
                break

        if iteration >= self.max_iterations:
            yield create_response_event(
                f"Reached maximum iterations ({self.max_iterations})", agent.name
            )

    async def next_actions(
        self,
        agent: BaseAgent,
        context: InteractionContext,
        *,
        skills_payload: str = "",
        instruction: str | None = None,
    ) -> AsyncGenerator[BaseEvent, None]:
        """Get the next actions from the LLM, yielding partial token events first.

        ``skills_payload`` is resolved once per :meth:`execute` call and passed
        down; this runs once per iteration. The default is an empty string rather
        than "resolve it here" on purpose — a self-resolving default would quietly
        reinstate the per-iteration cost the moment a caller forgot the argument.
        """

        messages = await self.context_manager.get_context(
            context=context,
            agent=agent,
            use_stop=self.use_stop,
            instruction=instruction,
        )

        messages = await inject_turn_context(messages, agent, context, skills_payload)

        structured_outputs = self.get_structured_outputs(
            agent=agent,
            context=context,
            use_stop=self.use_stop,
        )
        # TODO: add_generation_prompt is vLLM-only (and its default there) but reaches every
        # provider's payload; drop it or move it into VLLMProvider.
        # Collect LLM chunks, yielding each one as a partial event
        try:
            async for chunk in self.invoke_llm(
                agent.llm_provider,
                messages,
                structured_outputs=structured_outputs,
                stream=agent.stream,
                add_generation_prompt=True,
            ):
                if agent.stream:
                    if isinstance(chunk, str):
                        if not self.use_stop:
                            yield BaseEvent(
                                event_type=ExecutionEventType.RESPONSE_TOKEN,
                                content=[Part(text=chunk)],
                                author=agent.name,
                                is_partial=True,
                            )
                    elif isinstance(chunk, dict):
                        streamed_response = chunk.get("streamed_response", None)
                        response = chunk.get("full_response", None)
                else:
                    streamed_response = None
                    response = chunk
        except HTTPStatusError as exc:
            yield BaseEvent(
                event_type=ExecutionEventType.LLM_PROVIDER_ERROR,
                author=agent.llm_provider.name,
                error_code=str(exc.response.status_code),
                error_message=str(exc),
            )
            return
        except TimeoutException:
            yield BaseEvent(
                event_type=ExecutionEventType.LLM_PROVIDER_ERROR,
                author=agent.llm_provider.name,
                error_code=None,
                error_message="Request timed out",
            )
            return
        except HTTPError as exc:
            yield BaseEvent(
                event_type=ExecutionEventType.LLM_PROVIDER_ERROR,
                author=agent.llm_provider.name,
                error_code=None,
                error_message=str(exc),
            )
            return

        if not response:
            yield BaseEvent(
                event_type=ExecutionEventType.LLM_PROVIDER_ERROR,
                author=agent.llm_provider.name,
                error_code=None,
                error_message="No response received",
            )
            return

        parsed_actions = await self.context_manager.prompting.parse_llm_response(
            response, use_stop=self.use_stop
        )

        for parsed in parsed_actions:
            message_part = [Part(text=parsed.message)] if parsed.message else []
            if parsed.response_type == ResponseType.TOOL_AGENT_CALL:
                if parsed.action_name in [tool.name for tool in agent.tools]:
                    tool_action = ToolAction(
                        name=parsed.action_name,
                        parameters=parsed.action_arguments,
                        thought=parsed.thought,
                        message=parsed.message,
                    )
                    yield BaseEvent(
                        event_type=ExecutionEventType.TOOL_START,
                        author=agent.name,
                        action=tool_action,
                        content=message_part,
                        metadata={"streamed_response": streamed_response}
                        if streamed_response is not None
                        else {},
                    )
                # otherwise, a handoff to an agent
                elif parsed.action_name in [
                    agent.name for agent in context.handoff_targets_description
                ]:
                    agent_action = AgentAction(
                        name=parsed.action_name,
                        parameters=parsed.action_arguments,
                        thought=parsed.thought,
                        input_event=context.agent_input,
                        message=parsed.message,
                    )
                    yield BaseEvent(
                        event_type=ExecutionEventType.AGENT_CALL,
                        author=agent.name,
                        action=agent_action,
                        content=message_part,
                        metadata={"streamed_response": streamed_response}
                        if streamed_response is not None
                        else {},
                    )
                else:  # a tool that does not exist
                    yield BaseEvent(
                        event_type=ExecutionEventType.TOOL_START,
                        author=agent.name,
                        action=ToolAction(
                            name=parsed.action_name or "",
                            parameters=parsed.action_arguments,
                            thought=parsed.thought,
                            message=parsed.message,
                        ),
                        content=message_part,
                        metadata={"streamed_response": streamed_response}
                        if streamed_response is not None
                        else {},
                    )
            elif (parsed.response_type == ResponseType.STOP) and self.use_stop:
                stop_name = self.context_manager.prompting.stop_action.name
                info_text = (parsed.action_arguments or {}).get("info", "")
                stop_content = message_part or ([Part(text=info_text)] if info_text else [])
                yield BaseEvent(
                    event_type=ExecutionEventType.AGENT_END,
                    author=agent.name,
                    action=ToolAction(
                        name=stop_name,
                        parameters=parsed.action_arguments,
                        thought=parsed.thought,
                        message=parsed.message,
                    ),
                    content=stop_content,
                    metadata={"streamed_response": streamed_response}
                    if streamed_response is not None
                    else {},
                )
            elif (
                parsed.response_type == ResponseType.FORWARD_VERBATIM
                and self.forward_verbatim_enabled
            ):
                source_id = (parsed.action_arguments or {}).get("source_id", "")
                source_key = (parsed.action_arguments or {}).get("source_key")
                text, found = resolve_verbatim(context, source_id, source_key)
                if found:
                    terminal_metadata = {
                        METADATA_FORWARD_VERBATIM: True,
                        METADATA_FORWARD_VERBATIM_SOURCE_ID: source_id,
                        METADATA_FORWARD_VERBATIM_SOURCE_KEY: source_key,
                    }
                    if self.use_stop:
                        stop_name = self.context_manager.prompting.stop_action.name
                        yield BaseEvent(
                            event_type=ExecutionEventType.AGENT_END,
                            author=agent.name,
                            action=ToolAction(
                                name=stop_name,
                                parameters={"info": text},
                                thought=parsed.thought,
                            ),
                            content=[Part(text=text)],
                            is_partial=False,
                            metadata=terminal_metadata,
                        )
                    else:
                        yield BaseEvent(
                            event_type=ExecutionEventType.RESPONSE,
                            content=extract_parts(text),
                            author=agent.name,
                            action=ToolAction(
                                name=parsed.action_name,
                                thought=parsed.thought,
                                parameters={},
                            ),
                            metadata=terminal_metadata,
                        )
                else:
                    # Miss is a planner-level failure, not a tool failure:
                    # FORWARD_VERBATIM is a special action, not a registered
                    # tool. Emit ERROR_OCCURRED and terminate rather than
                    # retry — parity with the tool-use strategies and with
                    # the LLM-exception break path.
                    logger = get_logger()
                    observation = FORWARD_VERBATIM_MISS_UNRESOLVED.format(
                        source_id=source_id,
                        source_key=source_key,
                    )
                    logger.warning(observation)
                    yield create_error_event(
                        AGENT_ERROR_MESSAGE,
                        agent.name,
                        error_code=AGENT_ERROR_CODE,
                    )
            elif parsed.response_type == ResponseType.FORWARD_VERBATIM:
                yield create_error_event(
                    AGENT_ERROR_MESSAGE,
                    agent.name,
                    error_code=AGENT_ERROR_CODE,
                )
            elif parsed.response_type == ResponseType.RESPOND:
                event_metadata = (
                    {"streamed_response": streamed_response}
                    if streamed_response is not None
                    else {}
                )

                if agent.stream:
                    text = parsed.action_arguments.get(
                        next(
                            iter(
                                self.context_manager.prompting.response_action.parameters_schema[
                                    "properties"
                                ].keys()
                            )
                        ),
                        "",
                    )
                else:
                    text = parsed.action_arguments.get("text", "")

                yield BaseEvent(
                    event_type=ExecutionEventType.RESPONSE,
                    content=extract_parts(text) if parsed.action_arguments else [],
                    author=agent.name,
                    is_partial=parsed.action_arguments.get("is_partial", False),
                    need_feedback=parsed.action_arguments.get("need_feedback", False),
                    action=ToolAction(
                        name=parsed.action_name, thought=parsed.thought, parameters={}
                    ),
                    metadata=event_metadata,
                )
            elif parsed.response_type == ResponseType.ERROR:
                yield BaseEvent(
                    event_type=ExecutionEventType.TOOL_ERROR,
                    error_message=parsed.error,
                    author=agent.name,
                    action=ToolAction(
                        name=parsed.action_name or "",
                        parameters=parsed.action_arguments,
                        thought=parsed.thought,
                        message=parsed.message,
                        observation=parsed.error,
                    ),
                    error_code=parsed.action_name,
                    metadata={"streamed_response": streamed_response}
                    if streamed_response is not None
                    else {},
                )

    def get_structured_outputs(
        self,
        agent: BaseAgent,
        context: InteractionContext,
        use_stop: bool,
    ) -> StructuredOutput | None:
        """
        Get the structured output rules from the prompt builder using the first format supported with both the
        provider and the context manager.

        Args:
            agent (BaseAgent): The agent for which to get structured output rules.
            context (InteractionContext): The interaction context to use for generating structured output rules.
            use_stop (bool): Whether to use stop sequences when generating structured output rules.

        Returns:
            StructuredOutput | None: The structured output rules if supported by both the provider and the context manager,
            otherwise None.
        """
        structured_outputs: StructuredOutput | None = None
        supported_structured_outputs = agent.llm_provider.supported_structured_outputs
        if (not self._disable_structured_outputs) and supported_structured_outputs:
            for structured_output_type in supported_structured_outputs:
                structured_outputs = self.context_manager.prompting.get_structured_output_rules(
                    agent=agent,
                    context=context,
                    use_stop=self.use_stop,
                    structured_output_type=structured_output_type,
                    forward_verbatim_enabled=self.forward_verbatim_enabled,
                )
                if structured_outputs is not None:
                    break
        return structured_outputs

    async def invoke_llm(
        self,
        llm_provider: BaseLLMProvider,
        messages: list[LLMMessage],
        structured_outputs: StructuredOutput | None,
        stream: bool = False,
        **kwargs,
    ) -> AsyncGenerator[str, None]:
        """Async generator that yields LLMResponse chunks.

        In streaming mode each token is yielded as it arrives.
        In non-streaming mode a single LLMResponse is yielded.
        HTTP / timeout errors are raised so callers can handle them.
        """
        try:
            if stream:
                async for chunk in self.context_manager.prompting.parse_streaming_response(
                    llm_provider.generate_streaming(
                        messages, structured_outputs=structured_outputs, **kwargs
                    )
                ):
                    yield chunk
            else:
                response = await llm_provider.generate(
                    messages, structured_outputs=structured_outputs, **kwargs
                )
                yield response.content
        except HTTPStatusError as f:
            # Handle bad request or server error due to structured outputs
            if (structured_outputs is not None) and (
                (f.response.status_code == 400) or (f.response.status_code == 500)
            ):
                if stream:
                    async for chunk in self.context_manager.prompting.parse_streaming_response(
                        llm_provider.generate_streaming(messages, structured_outputs=None, **kwargs)
                    ):
                        yield chunk
                else:
                    response = await llm_provider.generate(
                        messages, structured_outputs=None, **kwargs
                    )
                    yield response.content

                self._disable_structured_outputs = True
            else:
                raise

    async def parse_llm_response(self, response: str) -> Any:
        """Parse the LLM output via the context manager's prompting layer."""
        return await self.context_manager.prompting.parse_llm_response(
            response, use_stop=self.use_stop
        )
