import json
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from typing import Any

from pydantic import BaseModel, Field

from domyn.agents.core import AgentDescription, BaseAgent, InteractionContext
from domyn.agents.llm import LLMResponse, StructuredOutput, StructuredOutputType
from domyn.agents.planner_strategy.prompting.data import (
    ActionErrors,
    ParsedAction,
    ResponseType,
    SpecialAction,
)
from domyn.agents.planner_strategy.utils import (
    FORWARD_VERBATIM_DESCRIPTION,
    FORWARD_VERBATIM_SOURCE_ID_DESCRIPTION,
    FORWARD_VERBATIM_SOURCE_KEY_DESCRIPTION,
    FORWARD_VERBATIM_TOOL_NAME,
)


class BasePrompting(ABC, BaseModel):
    type: str
    name: str | None = Field(default=None, description="Unique name of the prompting instance")
    response_action: SpecialAction = SpecialAction(
        name="RESPOND",
        description='Use this action to respond to the user. The "text" field must be markdown that can be shown directly to the user — never raw JSON. Set "need_feedback" to true when you need clarification.',
        parameters_schema={
            "properties": {
                "text": {"type": "string"},
                "need_feedback": {"type": "boolean"},
            },
            "required": ["text", "need_feedback"],
        },
    )
    forward_verbatim_action: SpecialAction = SpecialAction(
        name=FORWARD_VERBATIM_TOOL_NAME,
        description=FORWARD_VERBATIM_DESCRIPTION,
        parameters_schema={
            "properties": {
                "source_id": {
                    "type": "string",
                    "description": FORWARD_VERBATIM_SOURCE_ID_DESCRIPTION,
                },
                "source_key": {
                    "type": "string",
                    "description": FORWARD_VERBATIM_SOURCE_KEY_DESCRIPTION,
                },
            },
            "required": ["source_id", "source_key"],
        },
    )
    # used when use_stop=True
    partial_response_action: SpecialAction = SpecialAction(
        name="ASK_USER_MISSING_INFO",
        description="Use this action to ask questions to the user if you can't proceed without more information. Don't use this to provide the final answer to the user, use the stop tool instead.",
        parameters_schema={
            "properties": {
                "text": {"type": "string"},
            },
            "required": ["text"],
        },
    )
    stop_action: SpecialAction = SpecialAction(
        name="STOP",
        description="When you have completed your task, use this tool to return the data you have found to the agent that invoked you. Pass all the information you deem relevant in the 'info' field. Don't provide the final answer to the user but use this tool to return to the previous agent first.",
        parameters_schema={
            "properties": {
                # info is a string: strict structured outputs reject additionalProperties: true.
                "info": {
                    "type": "string",
                    "description": "The information to return to the calling agent",
                }
            },
            "required": ["info"],
        },
    )
    system_prompt_template: str
    system_prompt_suffix: str | None = None
    action_name_string: str = "name"
    action_parameters_string: str = "arguments"

    @staticmethod
    def handle_duplicated_tool_names_schemas(agent: BaseAgent) -> list[dict[str, Any]]:
        tool_names: list[str] = [t.name for t in agent.tools]
        duplicated_tool_names = {t for t in tool_names if tool_names.count(t) > 1}
        tool_schemas = [
            tool.namespaced_invocation_json_schema
            if tool.name in duplicated_tool_names
            else tool.invocation_json_schema
            for tool in agent.tools
        ]
        return tool_schemas

    @staticmethod
    def get_agent_descriptions_text(handoff_targets: list[AgentDescription]) -> str:
        if handoff_targets:
            agents_text = ""
            for a in handoff_targets:
                agents_text += f"-{json.dumps(a.invocation_json_schema)}\n"
        else:
            agents_text = "No sub-agents available"
        return agents_text

    @staticmethod
    def get_agent_instructions_text(agent: BaseAgent, instruction: str | None = None) -> str:
        """``instruction`` replaces ``agent.instruction`` when given: the agent's
        instruction with its standing instructions merged in."""
        text = agent.instruction if instruction is None else instruction
        return f"Goal-specific instructions:\n{text}" if text else ""

    def get_end_instructions_text(
        self, use_stop: bool, forward_verbatim_enabled: bool = True
    ) -> str:
        if use_stop:
            end_instruction = (
                f"-{json.dumps(self.partial_response_action.invocation_json_schema)}\n"
            )
            end_instruction += f"-{json.dumps(self.stop_action.invocation_json_schema)}\n"
        else:
            end_instruction = f"-{json.dumps(self.response_action.invocation_json_schema)}\n"
        if forward_verbatim_enabled:
            end_instruction += (
                f"-{json.dumps(self.forward_verbatim_action.invocation_json_schema)}\n"
            )
        return end_instruction

    @abstractmethod
    def format_system_prompt(
        self,
        tools_text: str,
        agents_descriptions: str,
        agent_instructions: str,
        end_instructions: str,
        agent_name: str,
        **kwargs,
    ) -> str: ...

    def get_system_prompt(
        self,
        agent: BaseAgent,
        context: InteractionContext,
        instruction: str | None = None,
        **kwargs,
    ) -> str:
        handoff_target_descriptions = list(context.handoff_targets_description)

        tools = "-" + "\n-".join(self.get_tools_str(agent))
        agents_descriptions = self.get_agent_descriptions_text(
            handoff_targets=handoff_target_descriptions
        )
        agents_instructions = self.get_agent_instructions_text(agent=agent, instruction=instruction)
        forward_verbatim_enabled = bool(getattr(agent.planner, "forward_verbatim_enabled", True))
        end_instructions = self.get_end_instructions_text(
            use_stop=agent.planner.use_stop,
            forward_verbatim_enabled=forward_verbatim_enabled,
        )
        system_prompt = self.format_system_prompt(
            tools_text=tools,
            agents_descriptions=agents_descriptions,
            agent_name=agent.name,
            agent_instructions=agents_instructions,
            end_instructions=end_instructions,
            forward_verbatim_enabled=forward_verbatim_enabled,
            **kwargs,
        )
        if self.system_prompt_suffix:
            system_prompt += f"\n {self.system_prompt_suffix}"
        return system_prompt

    def get_tools_str(self, agent: BaseAgent) -> list[str]:
        if not hasattr(agent, "tools") or not isinstance(agent.tools, list):
            return []

        tool_names: list[str] = [t.name for t in agent.tools]
        duplicated_tool_names = {t for t in tool_names if tool_names.count(t) > 1}

        available_tools: list[str] = []
        for tool in agent.tools:
            if tool.name in duplicated_tool_names:
                tool_json_schema = tool.namespaced_invocation_json_schema
            else:
                tool_json_schema = tool.invocation_json_schema
            tool_name = (
                tool_json_schema.get("properties", {})
                .get(self.action_name_string, {})
                .get("const", tool.name)
            )
            tool_description = tool_json_schema.get("description", "")
            tool_parameters = tool_json_schema.get("properties", {}).get(
                self.action_parameters_string, {}
            )
            # Some models expect the schema under parameters here, though calls use arguments.
            tool_desc = {
                "type": "function",
                "function": {
                    "name": tool_name,
                    "description": tool_description,
                    "parameters": tool_parameters,
                },
            }  # Declared under "parameters", while calls carry them under "arguments"
            available_tools.append(json.dumps(tool_desc))
        return available_tools

    @abstractmethod
    def get_structured_output_rules(
        self,
        agent: BaseAgent,
        context: InteractionContext,
        use_stop: bool,
        structured_output_type: StructuredOutputType,
        forward_verbatim_enabled: bool = True,
    ) -> StructuredOutput | None: ...

    @abstractmethod
    def extract_actions_from_response(self, response: str) -> list[str]: ...

    @abstractmethod
    def extract_thoughts_from_response(self, response: str) -> str | None: ...

    @abstractmethod
    def extract_message_from_response(self, response: str) -> str | None: ...

    def parse_actions(self, response: str) -> list[dict[str, dict[str, Any]]]:
        action_matches = self.extract_actions_from_response(response=response)
        parsed = []
        for action in action_matches:
            try:
                # Try to parse as JSON
                # clean up garbage before json
                action = "{" + "{".join(action.split("{")[1:])
                # clean up garbage after json
                action = "}".join(action.split("}")[:-1]) + "}"
                action = action.replace("True", "true").replace("False", "false")
                action = json.loads(action)
                action_name = action.get(self.action_name_string)
                action_arguments = action.get(self.action_parameters_string, {})
                parsed.append(
                    {
                        self.action_name_string: action_name,
                        self.action_parameters_string: action_arguments,
                    }
                )
            except json.JSONDecodeError as e:
                action_name = ActionErrors.JSONDecodeError
                action_arguments = {"action": action_matches, "error": str(e)}
                parsed.append(
                    {
                        self.action_name_string: action_name,
                        self.action_parameters_string: action_arguments,
                    }
                )
        if not parsed:
            action_name = ActionErrors.UNPARSABLE_ACTIONS
            action_arguments = None
            parsed.append(
                {
                    self.action_name_string: action_name,
                    self.action_parameters_string: action_arguments,
                }
            )
        return parsed

    @property
    @abstractmethod
    def no_actions_error(self) -> str: ...

    @property
    @abstractmethod
    def unparsable_action_error(self) -> str: ...

    @abstractmethod
    def format_invalid_json_error(self, action_arguments: str) -> str: ...

    @abstractmethod
    def parse_streaming_response(
        self, generator: AsyncGenerator[LLMResponse, None]
    ) -> AsyncGenerator[str | dict[str, str], None]: ...

    async def parse_llm_response(self, response: str, use_stop: bool) -> list[ParsedAction]:
        """Parse the LLM response to extract thoughts and actions"""

        extracted_actions = self.parse_actions(response)
        thought_input = self.extract_thoughts_from_response(response)
        message = self.extract_message_from_response(response)
        parsed_actions = []

        if len(extracted_actions) == 0:
            return [
                ParsedAction(
                    action_name=None,
                    response_type=ResponseType.ERROR,
                    action_arguments={
                        "response": response,
                        "error": self.no_actions_error,
                    },
                    thought=thought_input,
                    message=message,
                )
            ]

        for actions in extracted_actions:
            action_name = actions.get(self.action_name_string)
            action_arguments = actions.get(self.action_parameters_string, {})
            error = None
            match action_name:
                case self.response_action.name | self.partial_response_action.name:
                    if not all(
                        k in action_arguments
                        for k in self.partial_response_action.parameters_schema["required"]
                    ):
                        if use_stop:
                            name = self.response_action.name
                            parameters_schema = self.response_action.parameters_schema
                        else:
                            name = self.partial_response_action.name
                            parameters_schema = self.partial_response_action.parameters_schema
                        action_arguments = {
                            "response": response,
                            "error": f'Missing "text" in the field {self.action_parameters_string} for {name} action. The correct format is {parameters_schema}',
                        }
                        action_type = ResponseType.ERROR
                    else:
                        if use_stop:
                            action_arguments["need_feedback"] = True
                        else:
                            partial = action_arguments.pop("need_feedback", False)
                            if isinstance(partial, str):
                                partial = partial.lower() == "true"
                            action_arguments["need_feedback"] = partial

                            need_feedback = action_arguments.pop("need_feedback", False)
                            if isinstance(need_feedback, str):
                                need_feedback = need_feedback.lower() == "true"
                            action_arguments["need_feedback"] = need_feedback

                        action_type = ResponseType.RESPOND

                case self.forward_verbatim_action.name:
                    action_type = ResponseType.FORWARD_VERBATIM
                case self.stop_action.name:
                    action_type = ResponseType.STOP
                case ActionErrors.UNPARSABLE_ACTIONS:
                    action_type = ResponseType.ERROR
                    action_arguments = {
                        "response": response,
                        "error": self.unparsable_action_error,
                    }
                    error = self.unparsable_action_error
                case ActionErrors.JSONDecodeError:
                    action_type = ResponseType.ERROR
                    action_arguments = {
                        "response": response,
                    }
                    error = self.format_invalid_json_error(action_arguments=action_arguments)

                case _:
                    action_type = ResponseType.TOOL_AGENT_CALL

            parsed_actions.append(
                ParsedAction(
                    action_name=action_name,
                    response_type=action_type,
                    action_arguments=action_arguments,
                    thought=thought_input,
                    message=message,
                    error=error,
                )
            )
        return parsed_actions
