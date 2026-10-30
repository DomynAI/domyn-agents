import json
import re
from collections.abc import AsyncGenerator
from typing import Self

from pydantic import Field, model_validator

from domyn.agents.core import BaseAgent, BaseEvent, InteractionContext
from domyn.agents.core.utils import parts_to_text
from domyn.agents.llm import LLMMessage, LLMResponse, StructuredOutput, StructuredOutputType
from domyn.agents.planner_strategy.prompting.base import BasePrompting
from domyn.agents.planner_strategy.prompting.messages_building import MessageBuilder
from domyn.agents.planner_strategy.prompting.utils import (
    partial_pattern_matching,
    remove_descriptions_from_schema,
    sanitize_format_string,
)
from domyn.agents.planner_strategy.utils import EVENT_ID_HEADER_FMT

_FORWARD_VERBATIM_REACT_INSTRUCTION = (
    "9. Use FORWARD_VERBATIM only for terminal pass-through when the exact "
    "payload that will be emitted is a complete, self-contained, user-ready "
    "natural language or markdown answer. With an empty source_key, that "
    "payload is the whole OBSERVATION; with a source_key, it is only the "
    "resolved leaf. Emit the FORWARD_VERBATIM action with source_id copied "
    "exactly from the OBSERVATION event_id header and source_key copied "
    "exactly from the observation's dot-path to the prose leaf. Do not guess "
    "source keys, drop path segments, or use names like text, summary, result, "
    "or the tool name unless they are literal visible keys on the full path. "
    "Do not use FORWARD_VERBATIM when the emitted payload contains bare "
    "citation or reference markers such as [0], [1], [^note], [label][id], "
    "anchor links, see-above pointers, or \\ref{} whose target, URL, or "
    "definition lives outside the same payload. Full inline markdown links "
    "like [title](https://...) are self-contained. For web-search style "
    "payloads such as {observation: markdown, sources: [...]}, do not forward "
    "observation alone because citation markers lose the sibling sources URLs "
    "downstream; use RESPOND (or STOP when returning to a caller) and rewrite "
    "the answer with resolved inline links or without unresolved markers. Do "
    "not use FORWARD_VERBATIM for partial answers, metadata-only observations "
    "(listings, IDs, status flags, raw records without prose), or when the "
    "user asked for a different format, length, language, transformation, or "
    "synthesis across multiple observations. In those cases, use RESPOND (or "
    "STOP when returning to a caller) with the necessary summary or adaptation."
)


class StructuralTagsPrompting(BasePrompting):
    type: str = "structural_tags"
    action_start_pattern: str = "<action>"
    action_end_pattern: str = "</action>"
    thought_start_pattern: str | None = "<thought>"
    thought_end_pattern: str | None = "</thought>"
    message_start_pattern: str | None = "<message>"
    message_end_pattern: str | None = "</message>"
    streaming_start_patterns: list[str] = Field(
        default=[
            "{action_start_pattern}",
            r'"{action_name_string}"\s*:\s*("{respond_action_name}"|"{partial_respond_action_name}")',
            r'"({content_parameter_name}|{content_partial_response_parameter_name})"\s*:\s*"',
        ]
    )
    streaming_end_patterns: list[str] = Field(default=["{action_end_pattern}", "}}", '"'])
    streaming_negative_lookaheads: list[str] = Field(default=["", "", "\\"])
    forward_verbatim_instruction: str = _FORWARD_VERBATIM_REACT_INSTRUCTION
    thought_min_len: int | None = None
    thought_max_len: int | None = None
    system_prompt_template: str = """You are an AI agent in a multi-agent system using the ReACT pattern. Your name is {name}. You are provided with previous interactions between a user and the agents of the system and your task is to choose the best next action to take based on the context provided and the instructions below.

{agent_instructions}

General Instructions:
1. You should think step by step before taking the next most appropriate action to solve the task. Provide your thoughts before an optional message and the action. You must strictly follow the format {thought_start}brief thoughts here{thought_end}{action_start}action_json{action_end}{message_start}optional message here{message_end}
2. You should specify the next action via a valid json with the following format between {{"{action_name_string}":"<action name>", "{parameters_string}":{{"<param1_key>":"<param1_value>", ...}} }}.
3. In addition to thought and action, you can optionally provide a message to log what you are doing between {message_start}your message here{message_end} tags. This message will be shown to the user but without the possibility to take actions on it. You can use it, for example, to provide partial results or explaining the sources you are inspecting.
4. If you are calling an agent, provide a focused, self-contained task description. Use the following format for the parameters {{"task":"<self-contained task description with all necessary details, scoped to what the sub-agent needs to do>"}}. Do not include the broader user request, your own reasoning, or context beyond what the sub-agent needs.
5. Make sure all the formatting is correct, including the json format, the <action> tags, and the action parameters otherwise the system will not be able to parse your response.
6. Your thoughts can be in English. Always respond in the language of the user's LATEST message. Ignore the user's name, timezone, or language of document contents when choosing response language.
7. Don't make up information, use the tools and agents available to you to get real information.
8. Whenever asked about capabilities of one of your subagents, ask them, as they will be better suited than you to reply.
{forward_verbatim_instruction}

Below you are provided with json schemas for the available tools and agents. Don't call tools or agents that are not explicitly listed.

<agents>:
{agents}
</agents>

<tools>
{tools}
{end_instruction}
</tools>
"""

    def get_structured_output_rules(
        self,
        agent: BaseAgent,
        context: InteractionContext,
        use_stop: bool,
        structured_output_type: StructuredOutputType,
        forward_verbatim_enabled: bool = True,
    ) -> StructuredOutput | None:
        match structured_output_type:
            case StructuredOutputType.LLGUIDANCE:
                tool_schemas = self.handle_duplicated_tool_names_schemas(agent=agent)
                agent_schemas = [
                    a.invocation_json_schema for a in context.handoff_targets_description
                ]
                if use_stop:
                    special_actions_schemas = [
                        self.partial_response_action.invocation_json_schema,
                        self.stop_action.invocation_json_schema,
                    ]
                else:
                    special_actions_schemas = [self.response_action.invocation_json_schema]
                if forward_verbatim_enabled:
                    special_actions_schemas.append(
                        self.forward_verbatim_action.invocation_json_schema
                    )
                action_schemas = tool_schemas + agent_schemas + special_actions_schemas
                action_schemas = [remove_descriptions_from_schema(a) for a in action_schemas]
                actions_json_schema = {
                    "anyOf": [{"type": "object"} | schema for schema in action_schemas]
                }
                actions_def = "schemas: %json " + json.dumps(actions_json_schema)
                if (self.thought_max_len is None) and (self.thought_min_len is None):
                    length_pattern = "*"
                elif self.thought_max_len is None:
                    length_pattern = f"{{ {self.thought_min_len}, }}"
                elif self.thought_min_len is None:
                    length_pattern = f"{{ 0,{self.thought_max_len} }}"
                else:
                    length_pattern = f"{{ {self.thought_min_len},{self.thought_max_len} }}"

                thought_pattern = f'thought[suffix="{self.thought_end_pattern}"]: "{self.thought_start_pattern}" /(.|\n){length_pattern}/'
                action_pattern = (
                    f'action:"{self.action_start_pattern}"schemas"{self.action_end_pattern}"'
                )
                message_pattern = f'message[suffix="{self.message_end_pattern}"]: "{self.message_start_pattern}"/(.|\n)*/'
                structured_output = StructuredOutput(
                    type=structured_output_type,
                    rules=(
                        "start: thought action message? \n"
                        + thought_pattern
                        + "\n"
                        + action_pattern
                        + "\n"
                        + message_pattern
                        + "\n"
                        + actions_def
                    ),
                )
            case _:
                structured_output = None
        return structured_output

    @model_validator(mode="after")
    def validate_patterns(self) -> Self:
        if not (self.action_start_pattern or self.action_end_pattern):
            raise ValueError(
                "At least one of 'action_start_pattern' or 'action_end_pattern' must be set"
            )
        if self.action_start_pattern is None:
            self.action_start_pattern = ""
        if self.action_end_pattern is None:
            self.action_end_pattern = ""
        if self.thought_start_pattern is None:
            self.thought_start_pattern = ""
        if self.thought_end_pattern is None:
            self.thought_end_pattern = ""

        self.streaming_start_patterns = [
            sanitize_format_string(s).format(
                action_start_pattern=self.action_start_pattern,
                action_name_string=self.action_name_string,
                respond_action_name=self.response_action.name,
                partial_respond_action_name=self.partial_response_action.name,
                content_parameter_name=next(
                    iter(self.response_action.parameters_schema["properties"].keys())
                ),
                content_partial_response_parameter_name=next(
                    iter(self.partial_response_action.parameters_schema["properties"].keys())
                ),
            )
            for s in self.streaming_start_patterns
        ]
        self.streaming_end_patterns = [
            sanitize_format_string(s).format(action_end_pattern=self.action_end_pattern)
            for s in self.streaming_end_patterns
        ]
        return self

    def format_system_prompt(
        self,
        tools_text: str,
        agents_descriptions: str,
        agent_instructions: str,
        end_instructions: str,
        agent_name: str,
        **kwargs,
    ) -> str:
        forward_verbatim_enabled = kwargs.pop("forward_verbatim_enabled", True)
        forward_verbatim_instruction = (
            self.forward_verbatim_instruction if forward_verbatim_enabled else ""
        )
        return self.system_prompt_template.format(
            tools=tools_text,
            agents=agents_descriptions,
            name=agent_name,
            agent_instructions=agent_instructions,
            end_instruction=end_instructions,
            action_start=self.action_start_pattern,
            action_end=self.action_end_pattern,
            thought_start=self.thought_start_pattern,
            thought_end=self.thought_end_pattern,
            message_start=self.message_start_pattern,
            message_end=self.message_end_pattern,
            action_name_string=self.action_name_string,
            parameters_string=self.action_parameters_string,
            forward_verbatim_instruction=forward_verbatim_instruction,
            **kwargs,
        )

    @staticmethod
    def get_regex_pattern(start_pattern: str | None, end_pattern: str | None) -> str:
        if start_pattern and end_pattern:
            regex = rf"{start_pattern}(.*?){end_pattern}"
        elif start_pattern:
            regex = rf"{start_pattern}(.*)"
        elif end_pattern:
            regex = rf"(.*){end_pattern}"
        else:
            regex = "^(?!)$"
        return regex

    def extract_actions_from_response(self, response: str) -> list[str]:
        regex = self.get_regex_pattern(
            start_pattern=self.action_start_pattern, end_pattern=self.action_end_pattern
        )
        action_matches = re.findall(regex, response, re.IGNORECASE)
        return action_matches

    def extract_thoughts_from_response(self, response: str) -> str | None:
        thought_pattern = self.get_regex_pattern(
            start_pattern=self.thought_start_pattern, end_pattern=self.thought_end_pattern
        )
        thought_input_match = re.search(thought_pattern, response, re.IGNORECASE | re.DOTALL)
        thoughts = thought_input_match.group(1).strip() if thought_input_match else ""
        return thoughts

    def extract_message_from_response(self, response: str) -> str | None:
        if self.message_start_pattern and self.message_end_pattern:
            message_pattern = self.get_regex_pattern(
                start_pattern=self.message_start_pattern, end_pattern=self.message_end_pattern
            )
            message_match = re.search(message_pattern, response, re.IGNORECASE | re.DOTALL)
            message = message_match.group(1).strip() if message_match else None
            return message
        return None

    @property
    def no_actions_error(self) -> str:
        return f'No actions found in the response. Make sure to provide an action in the format {self.thought_start_pattern}your thoughts here{self.thought_end_pattern}{self.action_start_pattern}{{"{self.action_name_string}":"tool_name","{self.action_parameters_string}":{{"arg1":"value1","arg2":"value2"}} }}{self.action_end_pattern}'

    @property
    def unparsable_action_error(self) -> str:
        return f'Unparsable action. You must follow the following format in your response {self.thought_start_pattern}your thoughts here{self.thought_end_pattern}{self.action_start_pattern}{{"{self.action_name_string}":"tool_name","{self.action_parameters_string}":{{"arg1":"value1","arg2":"value2"}} }}{self.action_end_pattern}'

    def format_invalid_json_error(self, action_arguments: str) -> str:
        return f'Invalid JSON {action_arguments}. You must provide a valid JSON inside tags like {self.action_start_pattern}{{{self.action_name_string}:"tool_name", {self.action_parameters_string}:{{"arg1":"value1","arg2":"value2"}} }}{self.action_end_pattern}'

    def parse_streaming_response(
        self, generator: AsyncGenerator[LLMResponse, None]
    ) -> AsyncGenerator[str | dict[str, str], None]:
        return partial_pattern_matching(
            generator=generator,
            start_pattern_conditions=self.streaming_start_patterns,
            end_pattern_conditions=self.streaming_end_patterns,
            end_pattern_negative_lookaheads=self.streaming_negative_lookaheads,
        )


class StructuralTagMessagePrompting(StructuralTagsPrompting, MessageBuilder):
    type: str = "structural_tags"

    def get_system_prompt_message(
        self, agent: BaseAgent, context: InteractionContext, instruction: str | None = None
    ) -> LLMMessage:
        system_prompt = LLMMessage(
            role="system",
            content=self.get_system_prompt(agent=agent, context=context, instruction=instruction),
        )

        return system_prompt

    def get_thought_string(self, event: BaseEvent) -> str:
        if event and event.action and event.action.thought:
            thought_string = (
                f"{self.thought_start_pattern}{event.action.thought}{self.thought_end_pattern}"
            )
        else:
            thought_string = f"{self.thought_start_pattern}{self.thought_end_pattern}"
        return thought_string

    def get_message_string(self, event: BaseEvent) -> str:
        if event and event.action and event.action.message:
            message_string = (
                f"{self.message_start_pattern}{event.action.message}{self.message_end_pattern}"
            )
        else:
            message_string = ""
        return message_string

    def get_action_string(self, event: BaseEvent) -> str:
        action = {
            self.action_name_string: event.action.name
            if event.action is not None
            else event.author,
            self.action_parameters_string: event.action.parameters
            if event.action is not None
            else {},
        }
        action_string = f"{self.action_start_pattern}{json.dumps(action)}{self.action_end_pattern}"
        return action_string

    def get_response_message(self, event: BaseEvent, use_stop: bool) -> LLMMessage:
        name = self.response_action.name if use_stop else self.partial_response_action.name
        thought = self.get_thought_string(event)
        content = json.dumps(parts_to_text(event.content), ensure_ascii=False)
        return LLMMessage(
            role=self.assistant_role,
            content=f'Agent: {event.author}\n\n{thought}{self.action_start_pattern}{{"{self.action_name_string}":"{name}", "{self.action_parameters_string}":{{"text":{content}, "need_feedback": {event.need_feedback} }}{self.action_end_pattern}',
        )

    def get_agent_call_messages(self, event: BaseEvent) -> list[LLMMessage]:
        thought = self.get_thought_string(event)
        message = self.get_message_string(event)
        action = self.get_action_string(event)

        call_message = LLMMessage(
            role=self.assistant_role,
            content=f"Agent: {event.author}\n\n{thought}{action}{message}",
        )
        feedback_message = LLMMessage(
            role=self.feedback_role,
            content=f"STARTING AGENT: {event.action.name if event.action is not None else None}",
        )
        return [call_message, feedback_message]

    def get_agent_start_messages(self, event: BaseEvent) -> list[LLMMessage]:
        task = event.content[0].text if event.content else ""
        feedback_message = LLMMessage(
            role=self.feedback_role,
            content=f"STARTING AGENT: {event.action.name if event.action is not None else None}\n\nTask assigned: {task}",
        )
        return [feedback_message]

    def get_agent_end_messages(self, event: BaseEvent) -> list[LLMMessage]:
        thought = self.get_thought_string(event)
        message = self.get_message_string(event)
        action = self.get_action_string(event)
        stop_action = LLMMessage(
            role=self.assistant_role,
            content=f"Agent: {event.author}\n\n{thought}{action}{message}",
        )
        if event.error_code and event.error_message:
            obs_text = f"ERROR ({event.error_code}): {event.error_message}"
        else:
            answer = parts_to_text(event.content)
            obs_text = (
                answer
                if answer
                else f"Task completed. Ending {event.author} execution. Returning to calling agent."
            )
        observation = LLMMessage(
            role=self.feedback_role,
            content=f"OBSERVATION:\n\n {obs_text}",
        )
        return [stop_action, observation]

    def get_tool_end_messages(
        self, event: BaseEvent, forward_verbatim_enabled: bool = True
    ) -> list[LLMMessage]:
        thought = self.get_thought_string(event)
        message = self.get_message_string(event)
        action = self.get_action_string(event)

        action = LLMMessage(
            role=self.assistant_role,
            content=f"Agent: {event.author}\n\n{thought}{action}{message}",
        )
        obs = (
            json.dumps(event.action.observation)
            if isinstance(event.action.observation, dict | list)
            else event.action.observation
        )
        # The event_id header lets the LLM reference this observation by
        # event_id in a later FORWARD_VERBATIM call.
        if event.event_id and forward_verbatim_enabled:
            header = EVENT_ID_HEADER_FMT.format(event_id=event.event_id)
            obs_content = f"OBSERVATION {header}:\n\n Tool response: {obs}"
        else:
            obs_content = f"OBSERVATION:\n\n Tool response: {obs}"
        observation = LLMMessage(
            role=self.feedback_role,
            content=obs_content,
        )
        return [action, observation]

    def get_tool_error_messages(self, event: BaseEvent) -> list[LLMMessage]:
        messages = []
        if event.action is not None:
            if event.action.observation:
                thought = self.get_thought_string(event)
                message = self.get_message_string(event)
                action = self.get_action_string(event)

                messages.append(
                    LLMMessage(
                        role=self.assistant_role,
                        content=f"Agent: {event.author}\n\n{thought}{action}{message}",
                    )
                )

                messages.append(
                    LLMMessage(
                        role=self.feedback_role,
                        content=f"Agent: {event.author}\n\nTool error: {event.action.observation}",
                    ),
                )
            elif ("response" in event.action.parameters) and ("error" in event.action.parameters):
                response = event.action.parameters["response"]
                error = event.action.parameters["error"]
                messages.append(
                    LLMMessage(
                        role=self.assistant_role,
                        content=f"Agent: {event.author}\n\n{response}",
                    ),
                )
                messages.append(
                    LLMMessage(
                        role=self.feedback_role,
                        content=f"Agent: {event.author}\n\nError:{error}",
                    )
                )
        else:
            messages.append(
                LLMMessage(
                    role=self.assistant_role,
                    content=f"Agent: {event.author}\n\n{json.dumps(event.action.parameters)}",
                ),
            )
        return messages

    def add_start_message(
        self,
        messages: list[LLMMessage],
        agent: BaseAgent,
    ) -> list[LLMMessage]:
        start = f"Agent: {agent.name}\n\n"
        messages.append(LLMMessage(role=self.assistant_role, content=start))
        return messages
