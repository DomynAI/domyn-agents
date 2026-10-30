import json

from pydantic import BaseModel, ConfigDict, field_serializer

from domyn.agents.agents import Agent
from domyn.agents.core import (
    METADATA_CONVERTED_TO_AGENT_END,
    BaseEvent,
    ExecutionEventType,
    InteractionContext,
)
from domyn.agents.llm import BaseLLMProvider, LLMMessage
from domyn.agents.planner_strategy.context_management.base import (
    MessageContextManager,
)
from domyn.agents.planner_strategy.prompting.messages_building import MessageBuilder


class EventsSummarizer(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    llm_provider: BaseLLMProvider
    summary_system_prompt: str = """You are an expert assistant summarizing interactions between agents, tools, and users in a multi-agent system. You are given a series of past events and you should summarize them in order to preserve all the important information for the agents to complete the task.
         Keep your summary very short. Give more space to open points and information that might be useful in future interactions, you can just mention briefly resolved interactions that won't be needed anymore. Just summarize the facts, don't add other considerations. Emphasize the open points and the last user request. """
    user_message_preamble: str = "Summarize the following interaction history between agents, tools, and users in a concise manner, preserving all important information:\n\n"

    async def summarize_history(self, history: list[BaseEvent], **kwargs) -> str:
        full_str = ""
        for event in history:
            full_str += "---EVENT---\n" + json.dumps(event.prompt_print()) + "---\n"

        messages = [
            LLMMessage(
                role="system",
                content=self.summary_system_prompt,
            ),
            LLMMessage(
                role="user",
                content=self.user_message_preamble + full_str,
            ),
        ]

        try:
            response = await self.llm_provider.generate(messages, **kwargs)
            return response.content
        except Exception as e:
            raise RuntimeError(f"Error invoking LLM: {e}") from e

    async def get_history_summary(self, history: list[BaseEvent], cache: dict, **kwargs) -> str:
        history_ids = str(tuple(event.event_id for event in history))
        if ("summaries" in cache) and (history_ids in cache["summaries"]):
            summary = cache["summaries"][history_ids]
        else:
            summary = await self.summarize_history(history, structured_outputs=None, **kwargs)
            if "summaries" not in cache:
                cache["summaries"] = {history_ids: summary}
            else:
                cache["summaries"] |= {history_ids: summary}
        return summary

    @field_serializer("llm_provider")
    def dump_provider(self, v: BaseLLMProvider) -> dict:
        return v.model_dump()


class PreviousInteractionsSummaryContext(MessageContextManager, EventsSummarizer):
    type: str = "interactions_summary"
    summary_system_prompt: str = """You are an expert assistant summarizing interactions between agents, and tools a multi-agent system. You are given a trajectory made of a series of past agents actions and tool calls and you should summarize it preserving the final answer while detailing how it was obtained.
Keep your summary very short, briefly mention what the agent did, what worked and what didn't and then state the final agent's answer. Just summarize the facts, don't add other considerations. Include all the specific data obtained by the tools.
Use the following template:

Obtained information:
    - <important facts>
    - <retrieved data>
    - <failed attempts>
    ...
Open points <IF ANY, OPTIONAL>:
    - <missing information>
    - <unresolved issues>
    - <needs for clarification>
    ...
Final Response: <final agent response>
"""
    events_in_history: list[ExecutionEventType] = [
        ExecutionEventType.USER_INPUT,
        ExecutionEventType.RESPONSE,
        ExecutionEventType.AGENT_CALL,
        ExecutionEventType.AGENT_START,
        ExecutionEventType.AGENT_END,
        ExecutionEventType.TOOL_END,
        ExecutionEventType.TOOL_ERROR,
    ]
    prompting: MessageBuilder

    async def get_context(
        self, context: "InteractionContext", agent: "Agent", use_stop: bool, **kwargs
    ) -> list[LLMMessage]:
        messages = [
            self.prompting.get_system_prompt_message(
                agent=agent, context=context, instruction=kwargs.get("instruction")
            )
        ]
        included_events = []
        current_response = []
        for event in context.conversation_history + context.interaction_history:
            if (event.event_id in included_events) or (
                event.event_type not in self.events_in_history
            ):
                continue
            included_events.append(event.event_id)
            if event.event_type == ExecutionEventType.USER_INPUT:
                if len(current_response) > 0:
                    summary = await self.get_history_summary(
                        history=current_response, cache=context.state, **kwargs
                    )
                    messages.append(
                        LLMMessage(
                            role="assistant", content="*AGENTS EXECUTION SUMMARY*\n" + summary
                        )
                    )
                messages.append(self.prompting.get_user_request_message(event))
                current_response = []
            else:
                current_response.append(event)
        # REMAINING EVENTS IN FULL
        for event in current_response:
            match event.event_type:
                case ExecutionEventType.USER_INPUT:
                    messages.append(self.prompting.get_user_request_message(event))

                case ExecutionEventType.RESPONSE:
                    # Subagent responses converted to AGENT_END by the runner are
                    # rendered via the synthesized AGENT_END; skip the duplicate.
                    if event.metadata.get(METADATA_CONVERTED_TO_AGENT_END):
                        continue
                    messages.append(self.prompting.get_response_message(event, use_stop=use_stop))

                case ExecutionEventType.AGENT_END:
                    messages.extend(self.prompting.get_agent_end_messages(event))

                case ExecutionEventType.AGENT_CALL:
                    messages.extend(self.prompting.get_agent_call_messages(event))

                case ExecutionEventType.AGENT_START:
                    if not event.content:
                        event = self.add_default_agent_start_content(event=event, context=context)
                    messages.extend(self.prompting.get_agent_start_messages(event))

                case ExecutionEventType.TOOL_END | ExecutionEventType.TOOL_FEEDBACK:
                    messages.extend(
                        self.prompting.get_tool_end_messages(
                            event,
                            forward_verbatim_enabled=self.is_forward_verbatim_enabled(agent),
                        )
                    )

                case ExecutionEventType.TOOL_ERROR:
                    messages.extend(self.prompting.get_tool_error_messages(event))
        return messages


class PreviousAgentsSummaryContext(MessageContextManager, EventsSummarizer):
    type: str = "agents_summary"
    min_events_for_summary: int = 5
    events_in_history: list[ExecutionEventType] = [
        ExecutionEventType.USER_INPUT,
        ExecutionEventType.RESPONSE,
        ExecutionEventType.AGENT_CALL,
        ExecutionEventType.AGENT_START,
        ExecutionEventType.AGENT_END,
        ExecutionEventType.TOOL_END,
        ExecutionEventType.TOOL_ERROR,
    ]

    async def get_context(
        self, context: "InteractionContext", agent: "Agent", use_stop: bool, **kwargs
    ) -> list[LLMMessage]:
        messages = [
            self.prompting.get_system_prompt_message(
                agent=agent, context=context, instruction=kwargs.get("instruction")
            )
        ]
        interaction_history_ids = [e.event_id for e in context.interaction_history]
        history_events = [
            event
            for event in context.conversation_history
            if (event.event_id not in interaction_history_ids)
            and (event.event_type in self.events_in_history)
        ]

        if len(history_events) > self.min_events_for_summary:
            summary = await self.get_history_summary(history_events, cache=context.state, **kwargs)
            messages.append(
                LLMMessage(role="user", content="*PREVIOUS INTERACTIONS SUMMARY*\n" + summary)
            )

            full_messages = context.interaction_history
        else:
            full_messages = history_events + context.interaction_history

        for event in full_messages:
            match event.event_type:
                case ExecutionEventType.USER_INPUT:
                    messages.append(self.prompting.get_user_request_message(event))

                case ExecutionEventType.RESPONSE:
                    # Subagent responses converted to AGENT_END by the runner are
                    # rendered via the synthesized AGENT_END; skip the duplicate.
                    if event.metadata.get(METADATA_CONVERTED_TO_AGENT_END):
                        continue
                    messages.append(self.prompting.get_response_message(event, use_stop=use_stop))

                case ExecutionEventType.AGENT_END:
                    messages.extend(self.prompting.get_agent_end_messages(event))

                case ExecutionEventType.AGENT_CALL:
                    messages.extend(self.prompting.get_agent_call_messages(event))

                case ExecutionEventType.AGENT_START:
                    if not event.content:
                        event = self.add_default_agent_start_content(event=event, context=context)
                    messages.extend(self.prompting.get_agent_start_messages(event))

                case ExecutionEventType.TOOL_END:
                    messages.extend(
                        self.prompting.get_tool_end_messages(
                            event,
                            forward_verbatim_enabled=self.is_forward_verbatim_enabled(agent),
                        )
                    )

                case ExecutionEventType.TOOL_ERROR:
                    messages.extend(self.prompting.get_tool_error_messages(event))
        return messages


class OldEventsSummaryContext(MessageContextManager, EventsSummarizer):
    type: str = "old_summary"
    events_in_history: list[ExecutionEventType] = [
        ExecutionEventType.USER_INPUT,
        ExecutionEventType.RESPONSE,
        ExecutionEventType.AGENT_CALL,
        ExecutionEventType.AGENT_START,
        ExecutionEventType.AGENT_END,
        ExecutionEventType.TOOL_END,
        ExecutionEventType.TOOL_ERROR,
    ]
    message_history_length: int = 8

    async def get_context(
        self, context: "InteractionContext", agent: "Agent", use_stop: bool, **kwargs
    ) -> list[LLMMessage]:
        messages = [
            self.prompting.get_system_prompt_message(
                agent=agent, context=context, instruction=kwargs.get("instruction")
            )
        ]
        added_event_ids = []
        added_events = []
        for event in context.conversation_history + context.interaction_history:
            if (event.event_id in added_event_ids) or (
                event.event_type not in self.events_in_history
            ):
                continue
            added_event_ids.append(event.event_id)
            added_events.append(event)

        full_messages_context = added_events[-self.message_history_length :]
        summarize_message_content = added_events[: -self.message_history_length]
        if len(summarize_message_content) > 0:
            summary = await self.summarize_history(summarize_message_content, **kwargs)
            messages.append(
                LLMMessage(role="user", content="*PREVIOUS INTERACTIONS SUMMARY*\n" + summary)
            )
        for event in full_messages_context:
            match event.event_type:
                case ExecutionEventType.USER_INPUT:
                    messages.append(self.prompting.get_user_request_message(event))

                case ExecutionEventType.RESPONSE:
                    # Subagent responses converted to AGENT_END by the runner are
                    # rendered via the synthesized AGENT_END; skip the duplicate.
                    if event.metadata.get(METADATA_CONVERTED_TO_AGENT_END):
                        continue
                    messages.append(self.prompting.get_response_message(event, use_stop=use_stop))

                case ExecutionEventType.AGENT_END:
                    messages.extend(self.prompting.get_agent_end_messages(event))

                case ExecutionEventType.AGENT_CALL:
                    messages.extend(self.prompting.get_agent_call_messages(event))

                case ExecutionEventType.AGENT_START:
                    if not event.content:
                        event = self.add_default_agent_start_content(event=event, context=context)
                    messages.extend(self.prompting.get_agent_start_messages(event))

                case ExecutionEventType.TOOL_END:
                    messages.extend(
                        self.prompting.get_tool_end_messages(
                            event,
                            forward_verbatim_enabled=self.is_forward_verbatim_enabled(agent),
                        )
                    )

                case ExecutionEventType.TOOL_ERROR:
                    messages.extend(self.prompting.get_tool_error_messages(event))
        return messages
