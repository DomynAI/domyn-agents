import copy
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
)
from domyn.agents.planner_strategy.utils import EVENT_ID_HEADER_FMT


class JSONPrompting(BasePrompting):
    type: str = "json_schema"
    streaming_start_patterns: list[str] = Field(
        default=[
            r'"action"\s*:\s*',
            r'"{action_name_string}"\s*:\s*("{respond_action_name}"|"{partial_respond_action_name}")',
            r'"({content_parameter_name}|{content_partial_response_parameter_name})"\s*:\s*"',
        ]
    )
    streaming_end_patterns: list[str] = Field(default=["}", "}}", '"'])
    streaming_negative_lookaheads: list[str] = Field(default=["", "", "\\"])
    system_prompt_template: str = """You are an AI agent in a multi-agent system using the ReACT pattern. Your name is {name}. You are provided with previous interactions between a user and the agents of the system and your task is to choose the best next action to take based on the context provided and the instructions below.

{agent_instructions}

General Instructions:
1. You should think step by step before taking the next most appropriate action to solve the task. Provide your thoughts before an optional message and the action. You must strictly follow the JSON format {{"thought": <brief thoughts here>, "action":{{ "{action_name_string}": <action_name>, "{parameters_string}":<action_parameters>}}, "message": <optional message here>}}
2. You should specify the next action via the <action_name> and the <action_parameters> which should also be a valid JSON, i.e. "{parameters_string}":{{"<param1_key>":"<param1_value>", ...}} }}.
3. In addition to thought and action, you can optionally provide a message to log what you are doing associated to the "message" key. This message will be shown to the user but without the possibility to take actions on it. You can use it, for example, to provide partial results or explaining the sources you are inspecting.
4. If you are calling an agent, provide a focused, self-contained task description. Use the following format for the parameters {{"task":"<self-contained task description with all necessary details, scoped to what the sub-agent needs to do>"}}. Do not include the broader user request, your own reasoning, or context beyond what the sub-agent needs.
5. Make sure all the formatting is correct, including the json format, the <action> tags, and the action parameters otherwise the system will not be able to parse your response.
6. Your thoughts can be in English. Always respond in the language of the user's LATEST message. Ignore the user's name, timezone, or language of document contents when choosing response language.
7. Don't make up information, use the tools and agents available to you to get real information.
8. Whenever asked about capabilities of one of your subagents, ask them, as they will be better suited than you to reply.

Below you are provided with json schemas for the available tools, agents, and special actions. Don't call tools or agents that are not explicitly listed.

Available Agents:
{agents}

Available Tools:
{tools}

Special actions:
{end_instruction}
"""

    def get_structured_output_rules(
        self,
        agent: BaseAgent,
        context: InteractionContext,
        use_stop: bool,
        structured_output_type: StructuredOutputType,
        forward_verbatim_enabled: bool = True,
    ) -> StructuredOutput | None:
        # TODO: strict JSON-schema mode needs every property in required; optional tool params get
        # the schema rejected and react drops structured output. Emit them as required + nullable.
        tool_schemas = self.handle_duplicated_tool_names_schemas(agent=agent)
        agent_schemas = [a.invocation_json_schema for a in context.handoff_targets_description]
        if use_stop:
            special_actions_schemas = [
                self.partial_response_action.invocation_json_schema,
                self.stop_action.invocation_json_schema,
            ]
        else:
            special_actions_schemas = [self.response_action.invocation_json_schema]
        if forward_verbatim_enabled:
            special_actions_schemas.append(self.forward_verbatim_action.invocation_json_schema)
        action_schemas = tool_schemas + agent_schemas + special_actions_schemas
        action_schemas = [remove_descriptions_from_schema(a) for a in action_schemas]
        actions_json_schema = {"anyOf": [{"type": "object"} | schema for schema in action_schemas]}
        step_schema = {
            "type": "object",
            "properties": {
                "thought": {"type": "string"},
                "action": actions_json_schema,
                "message": {"type": "string"},
            },
            "required": ["thought", "action"],
        }
        match structured_output_type:
            case StructuredOutputType.JSON_SCHEMA:
                step_schema = copy.deepcopy(step_schema)
                step_schema |= {"additionalProperties": False}
                step_schema["required"] = [
                    "thought",
                    "action",
                    "message",
                ]  # Strict mode: every property must be in required.
                step_schema["properties"]["action"] |= {"additionalProperties": False}
                for action in step_schema["properties"]["action"]["anyOf"]:
                    action |= {"additionalProperties": False}
                    action["required"] = [self.action_name_string, self.action_parameters_string]
                    action["properties"][self.action_parameters_string] |= {
                        "additionalProperties": False,
                        "type": "object",
                    }
                structured_output = StructuredOutput(type=structured_output_type, rules=step_schema)
            case StructuredOutputType.LLGUIDANCE:
                structured_output = StructuredOutput(
                    type=structured_output_type, rules="start: %json " + json.dumps(step_schema)
                )
            case _:
                structured_output = None
        return structured_output

    @model_validator(mode="after")
    def validate_patterns(self) -> Self:
        self.streaming_start_patterns = [
            s.format(
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
        return self.system_prompt_template.format(
            tools=tools_text,
            agents=agents_descriptions,
            name=agent_name,
            agent_instructions=agent_instructions,
            end_instruction=end_instructions,
            action_name_string=self.action_name_string,
            parameters_string=self.action_parameters_string,
            **kwargs,
        )

    def extract_actions_from_response(self, response: str) -> list[str]:
        # TODO: one action per step (the step schema has a single action); structural tags allow
        # several.
        m = re.search(r'"action"\s*:\s*\{', response)
        if not m:
            return []
        start_brace = m.end() - 1  # position of the first {
        depth = 1
        i = m.end()
        while i < len(response) and depth:
            c = response[i]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
            i += 1
        if depth != 0:
            return []  # unmatched braces
        return [response[start_brace:i]]

    def extract_thoughts_from_response(self, response: str) -> str | None:
        pattern = r'"thought"\s*:\s*"((?:\\.|[^"\\])*)"'
        thought_match = re.search(pattern, response)
        if thought_match:
            return thought_match.group(1)
        return None

    def extract_message_from_response(self, response: str) -> str | None:
        pattern = r'"message"\s*:\s*"((?:\\.|[^"\\])*)"'
        message_match = re.search(pattern, response)
        if message_match:
            return message_match.group(1)
        return None

    @property
    def no_actions_error(self) -> str:
        return f'No actions found in the response. Make sure to provide a response in the format {{"thought":<your thoughts here>, "action":{{ "{self.action_name_string}":"tool_name", "{self.action_parameters_string}":{{"arg1":"value1","arg2":"value2"}}, "message":<optional message here>}}'

    @property
    def unparsable_action_error(self) -> str:
        return f'Unparsable action. You must strictly adhere to the following JSON format in your response  {{"thought":<your thoughts here>, "action":{{ "{self.action_name_string}":"tool_name", "{self.action_parameters_string}":{{"arg1":"value1","arg2":"value2"}}, "message":<optional message here>}}'

    def format_invalid_json_error(self, action_arguments: str) -> str:
        return f'Invalid JSON {action_arguments}. You must provide a valid JSON for the action like {{ "{self.action_name_string}":"tool_name", "{self.action_parameters_string}":{{"arg1":"value1","arg2":"value2"}} }}'

    def parse_streaming_response(
        self, generator: AsyncGenerator[LLMResponse, None]
    ) -> AsyncGenerator[str | dict[str, str], None]:
        return partial_pattern_matching(
            generator=generator,
            start_pattern_conditions=self.streaming_start_patterns,
            end_pattern_conditions=self.streaming_end_patterns,
            end_pattern_negative_lookaheads=self.streaming_negative_lookaheads,
        )


class JSONMessagePrompting(JSONPrompting, MessageBuilder):
    type: str = "json_schema"

    def get_system_prompt_message(
        self, agent: BaseAgent, context: InteractionContext, instruction: str | None = None
    ) -> LLMMessage:
        system_prompt = LLMMessage(
            role="system",
            content=self.get_system_prompt(agent=agent, context=context, instruction=instruction),
        )

        return system_prompt

    @staticmethod
    def get_thought_string(event: BaseEvent) -> str:
        thought_value = event.action.thought if event.action is not None else ""
        thought = f'"{thought_value}"' if thought_value else ""
        return thought

    @staticmethod
    def get_message_dict_piece(event: BaseEvent) -> str:
        message = (
            event.action.message if event.action is not None and event.action.message else None
        )
        message_piece = f', "message":"{message}"' if message else ""
        return message_piece

    def get_action_string(self, event: BaseEvent) -> str:
        if event.action is None:
            return f'"{self.action_name_string}": "{event.author}", "{self.action_parameters_string}": {{}}'
        return f'"{self.action_name_string}": "{event.action.name}", "{self.action_parameters_string}": {json.dumps(event.action.parameters)}'

    def get_response_message(self, event: BaseEvent, use_stop: bool) -> LLMMessage:
        # The message is assembled by string formatting, so it is not always valid JSON: an
        # empty thought, the doubled braces around the action and Python's `True`/`False`
        # produce text a strict parser rejects.
        thought = self.get_thought_string(event)
        name = self.response_action.name if use_stop else self.partial_response_action.name
        content = json.dumps(parts_to_text(event.content), ensure_ascii=False)
        action = f'{{ "{self.action_name_string}":"{name}", "{self.action_parameters_string}":{{"text":{content}, "need_feedback": {event.need_feedback} }} }}'
        return LLMMessage(
            role=self.assistant_role,
            content=f'Agent: {event.author}\n\n{{"thought":{thought}, "action":{{ {action} }} }}',
        )

    def get_agent_call_messages(self, event: BaseEvent) -> list[LLMMessage]:
        thought = self.get_thought_string(event)
        action = self.get_action_string(event)
        message_piece = self.get_message_dict_piece(event)
        call_message = LLMMessage(
            role=self.assistant_role,
            content=f'Agent: {event.author}\n\n{{"thought": {thought}, "action": {{ {action} }}{message_piece} }}',
        )
        feedback_message = LLMMessage(
            role=self.feedback_role,
            content=f"STARTING AGENT: {event.action.name if event.action is not None else None}\n\n",
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
        message_piece = self.get_message_dict_piece(event)
        action = self.get_action_string(event)
        stop_action = LLMMessage(
            role=self.assistant_role,
            content=f'Agent: {event.author}\n\n{{"thought": {thought}, "action": {{ {action} }}{message_piece} }}',
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
        message_piece = self.get_message_dict_piece(event)
        action = self.get_action_string(event)

        action = LLMMessage(
            role=self.assistant_role,
            content=f'Agent: {event.author}\n\n{{"thought": {thought}, "action": {{ {action} }}{message_piece} }}',
        )
        obs = (
            json.dumps(event.action.observation)
            if isinstance(event.action.observation, dict | list)
            else event.action.observation
        )
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
                action = self.get_action_string(event)
                message_piece = self.get_message_dict_piece(event)

                messages.append(
                    LLMMessage(
                        role=self.assistant_role,
                        content=f'Agent: {event.author}\n\n{{"thought": {thought}, "action": {{ {action} }}{message_piece} }}',
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
            fallback_error = event.error_message or "Tool execution failed"
            messages.append(
                LLMMessage(
                    role=self.assistant_role,
                    content=f"Agent: {event.author}\n\n{fallback_error}",
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
