from domyn.agents.core import (
    METADATA_CONVERTED_TO_AGENT_END,
    METADATA_EXTERNAL_PART_AGENT_VISIBLE,
    BaseAgent,
    ExecutionEventType,
    InteractionContext,
)
from domyn.agents.core.utils import parts_to_text
from domyn.agents.llm import LLMMessage
from domyn.agents.planner_strategy.context_management.base import MessageContextManager


class FullMessageContext(MessageContextManager):
    type: str = "full_messages"
    narrow_non_visible_agents: bool

    async def get_context(
        self, context: InteractionContext, agent: BaseAgent, **kwargs
    ) -> list[LLMMessage]:
        visible_agent_names = [*context.visible_agent_names, agent.name]
        messages = [
            self.prompting.get_system_prompt_message(
                agent=agent, context=context, instruction=kwargs.get("instruction")
            )
        ]
        included_events = []
        for event in context.conversation_history + context.interaction_history:
            if event.event_id in included_events:
                continue
            else:
                included_events.append(event.event_id)

            match event.event_type:
                case ExecutionEventType.USER_INPUT:
                    messages.append(self.prompting.get_user_request_message(event))

                case ExecutionEventType.RESPONSE:
                    # Subagent responses converted to AGENT_END by the runner are
                    # rendered via the synthesized AGENT_END; skip the duplicate.
                    if event.metadata.get(METADATA_CONVERTED_TO_AGENT_END):
                        continue
                    messages.append(
                        self.prompting.get_response_message(event, use_stop=agent.planner.use_stop)
                    )

                case ExecutionEventType.AGENT_END:
                    messages.extend(self.prompting.get_agent_end_messages(event))

                case ExecutionEventType.AGENT_CALL:
                    messages.extend(self.prompting.get_agent_call_messages(event))

                case ExecutionEventType.AGENT_START:
                    if not event.content:
                        event = self.add_default_agent_start_content(event=event, context=context)
                    messages.extend(self.prompting.get_agent_start_messages(event))

                case ExecutionEventType.TOOL_END | ExecutionEventType.TOOL_FEEDBACK:
                    if (not self.narrow_non_visible_agents) or (
                        event.author in visible_agent_names
                    ):
                        messages.extend(
                            self.prompting.get_tool_end_messages(
                                event,
                                forward_verbatim_enabled=self.is_forward_verbatim_enabled(agent),
                            )
                        )

                case ExecutionEventType.TOOL_ERROR:
                    messages.extend(self.prompting.get_tool_error_messages(event))

                case ExecutionEventType.EXTERNAL_PART:
                    # Externally-injected content (e.g. from a channel) only
                    # reaches the model when explicitly marked visible to it —
                    # see METADATA_EXTERNAL_PART_AGENT_VISIBLE.
                    if event.metadata.get(METADATA_EXTERNAL_PART_AGENT_VISIBLE, False):
                        text = parts_to_text(event.content)
                        if text:
                            messages.append(LLMMessage(role=self.prompting.user_role, content=text))

        return messages
