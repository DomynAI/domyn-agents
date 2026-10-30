import json
import textwrap
from unittest.mock import MagicMock

import pytest
from jsonschema import Draft202012Validator

from domyn.agents import BaseEvent, ExecutionEventType, Part
from domyn.agents.agents import Agent
from domyn.agents.core import InteractionContext
from domyn.agents.llm import BaseLLMProvider, StructuredOutputType
from domyn.agents.planner_strategy import JSONPrompting
from domyn.agents.planner_strategy.prompting.base import BasePrompting
from domyn.agents.planner_strategy.prompting.structural_tags_prompting import (
    StructuralTagsPrompting,
)
from domyn.agents.tools import ConfigParam, FunctionTool


@pytest.fixture
def structural_tags_prompting() -> StructuralTagsPrompting:
    return StructuralTagsPrompting(type="base", name="BasePromptingTest")


@pytest.fixture
def json_schema_prompting() -> JSONPrompting:
    return JSONPrompting(type="base", name="BasePromptingTest")


@pytest.fixture
def agent_build_single():
    return Agent(
        name="BaseAgentTest",
        description="A test agent.",
        instruction="You are a test agent.",
        llm_provider=MagicMock(BaseLLMProvider),
    )


@pytest.fixture
def agent_build_one_sub_one_tool(agent_build_single):
    def function_to_test(self, param1: str, param2: list[str], param3: float, key: str = ""):
        """Example tool for prompting tests.

        Args:
            param1 (str): the param1 from doc.
            param2 (List[str]): the param2 from doc.
            param3 (float): the param3 from doc.
            key (str): The plain text content of the email. Is a config param


        Returns:
            bool: True if ok, False otherwise.
        """
        return True

    tool = FunctionTool(
        name="TestTool",
        func=function_to_test,
        config_params=[ConfigParam(type="str", name="key", value="test")],
    )

    return Agent(
        name="MultiAgentTest",
        description="A test agent with a sub agent and 1 tool.",
        instruction="You are a test agent.",
        llm_provider=MagicMock(BaseLLMProvider),
        sub_agents=[agent_build_single],
        tools=[tool],
    )


def test_base_prompting_respond_always_simple():
    """RESPOND action always has the simple schema (text + need_feedback)."""
    prompting = StructuralTagsPrompting(type="base", name="test")
    respond_props = prompting.response_action.parameters_schema["properties"]
    assert "text" in respond_props
    assert "need_feedback" in respond_props
    assert "source_id" not in respond_props
    assert "source_key" not in respond_props


def test_base_prompting_forward_verbatim_action_has_source_params():
    """forward_verbatim_action always carries source_id / source_key."""
    prompting = StructuralTagsPrompting(type="base", name="test")
    fwd_props = prompting.forward_verbatim_action.parameters_schema["properties"]
    assert "source_id" in fwd_props
    assert "source_key" in fwd_props


def _has_description_key(obj) -> bool:
    """Recursively check if any dict in obj has a 'description' key."""
    if isinstance(obj, dict):
        if "description" in obj:
            return True
        return any(_has_description_key(v) for v in obj.values())
    if isinstance(obj, list):
        return any(_has_description_key(item) for item in obj)
    return False


def _structured_output_action_names(schema: dict) -> list[str]:
    action_schemas = schema.get("anyOf") or schema["properties"]["action"]["anyOf"]
    return [s["properties"]["name"]["const"] for s in action_schemas]


def test_grammar_schema_no_descriptions(structural_tags_prompting, agent_build_single):
    """grammar JSON schema must not contain any 'description'
    fields (remove_descriptions_from_schema applied)."""
    prompting = StructuralTagsPrompting(type="base", name="test")
    grammar = prompting.get_structured_output_rules(
        agent=agent_build_single,
        context=InteractionContext(
            conversation_id="test_conversation_id",
            user_id="test_user_id",
            interaction_id="test_interaction_id",
            agent_input=BaseEvent(
                event_type=ExecutionEventType.USER_INPUT,
                content=[Part(text="Test user input")],
                author="user",
            ),
        ),
        use_stop=False,
        structured_output_type=StructuredOutputType.LLGUIDANCE,
    )
    assert grammar is not None
    assert "%json " in grammar.rules
    schema = json.loads(grammar.rules.split("%json ")[1])
    assert not _has_description_key(schema), (
        "Grammar schema contains 'description' fields — "
        "remove_descriptions_from_schema not applied correctly"
    )


def test_get_structured_output_grammar(structural_tags_prompting, agent_build_single):
    grammar = structural_tags_prompting.get_structured_output_rules(
        agent=agent_build_single,
        context=InteractionContext(
            conversation_id="test_conversation_id",
            user_id="test_user_id",
            interaction_id="test_interaction_id",
            agent_input=BaseEvent(
                event_type=ExecutionEventType.USER_INPUT,
                content=[Part(text="Test user input")],
                author="user",
            ),
        ),
        use_stop=False,
        structured_output_type=StructuredOutputType.LLGUIDANCE,
    )
    assert grammar is not None
    assert "%json " in grammar.rules
    schema = json.loads(grammar.rules.split("%json ")[1])
    Draft202012Validator.check_schema(schema)
    # use_stop=False → RESPOND + FORWARD_VERBATIM (default on)
    action_names = [s["properties"]["name"]["const"] for s in schema["anyOf"]]
    assert action_names == ["RESPOND", "FORWARD_VERBATIM"]
    respond_schema = schema["anyOf"][0]
    respond_arg_props = respond_schema["properties"]["arguments"]["properties"]
    assert "text" in respond_arg_props
    assert "need_feedback" in respond_arg_props

    grammar = structural_tags_prompting.get_structured_output_rules(
        agent=agent_build_single,
        context=InteractionContext(
            conversation_id="test_conversation_id",
            user_id="test_user_id",
            interaction_id="test_interaction_id",
            agent_input=BaseEvent(
                event_type=ExecutionEventType.USER_INPUT,
                content=[Part(text="Test user input")],
                author="user",
            ),
        ),
        use_stop=True,
        structured_output_type=StructuredOutputType.LLGUIDANCE,
    )
    assert grammar is not None
    # use_stop=True → ASK_USER_MISSING_INFO + STOP + FORWARD_VERBATIM (default on)
    schema = json.loads(grammar.rules.split("%json ")[1])
    Draft202012Validator.check_schema(schema)
    action_names = [s["properties"]["name"]["const"] for s in schema["anyOf"]]
    assert action_names == ["ASK_USER_MISSING_INFO", "STOP", "FORWARD_VERBATIM"]

    structural_tags_prompting.action_start_pattern = "<test_action>"
    structural_tags_prompting.action_end_pattern = "</test_action>"
    structural_tags_prompting.thought_start_pattern = "<test_thought>"
    structural_tags_prompting.thought_end_pattern = "</test_thought>"
    structural_tags_prompting.message_start_pattern = "<test_message>"
    structural_tags_prompting.message_end_pattern = "</test_message>"

    grammar = structural_tags_prompting.get_structured_output_rules(
        agent=agent_build_single,
        context=InteractionContext(
            conversation_id="test_conversation_id",
            user_id="test_user_id",
            interaction_id="test_interaction_id",
            agent_input=BaseEvent(
                event_type=ExecutionEventType.USER_INPUT,
                content=[Part(text="Test user input")],
                author="user",
            ),
        ),
        use_stop=True,
        structured_output_type=StructuredOutputType.LLGUIDANCE,
    )
    assert grammar is not None
    # Same shape as previous assertion but after custom patterns are applied.
    schema = json.loads(grammar.rules.split("%json ")[1])
    Draft202012Validator.check_schema(schema)
    action_names = [s["properties"]["name"]["const"] for s in schema["anyOf"]]
    assert action_names == ["ASK_USER_MISSING_INFO", "STOP", "FORWARD_VERBATIM"]
    assert "<test_action>" in grammar.rules
    assert "<test_thought>" in grammar.rules
    assert "<test_message>" in grammar.rules


@pytest.mark.parametrize("prompting_cls", [StructuralTagsPrompting, JSONPrompting])
def test_forward_verbatim_can_be_disabled_in_structured_outputs(prompting_cls, agent_build_single):
    prompting = prompting_cls(type="base", name="test")
    context = InteractionContext(
        conversation_id="test_conversation_id",
        user_id="test_user_id",
        interaction_id="test_interaction_id",
        agent_input=BaseEvent(
            event_type=ExecutionEventType.USER_INPUT,
            content=[Part(text="Test user input")],
            author="user",
        ),
    )

    grammar = prompting.get_structured_output_rules(
        agent=agent_build_single,
        context=context,
        use_stop=False,
        structured_output_type=StructuredOutputType.LLGUIDANCE,
        forward_verbatim_enabled=False,
    )
    assert grammar is not None
    schema = json.loads(grammar.rules.split("%json ")[1])
    Draft202012Validator.check_schema(schema)
    action_names = _structured_output_action_names(schema)
    assert action_names == ["RESPOND"]

    grammar = prompting.get_structured_output_rules(
        agent=agent_build_single,
        context=context,
        use_stop=True,
        structured_output_type=StructuredOutputType.LLGUIDANCE,
        forward_verbatim_enabled=False,
    )
    assert grammar is not None
    schema = json.loads(grammar.rules.split("%json ")[1])
    Draft202012Validator.check_schema(schema)
    action_names = _structured_output_action_names(schema)
    assert action_names == ["ASK_USER_MISSING_INFO", "STOP"]


def test_get_structured_output_grammar_agent_one_tool_one(
    structural_tags_prompting, agent_build_one_sub_one_tool
):
    grammar = structural_tags_prompting.get_structured_output_rules(
        agent=agent_build_one_sub_one_tool,
        context=InteractionContext(
            conversation_id="test_conversation_id",
            user_id="test_user_id",
            interaction_id="test_interaction_id",
            agent_input=BaseEvent(
                event_type=ExecutionEventType.USER_INPUT,
                content=[Part(text="Test user input")],
                author="user",
            ),
        ),
        use_stop=False,
        structured_output_type=StructuredOutputType.LLGUIDANCE,
    )
    assert grammar is not None
    assert "%json " in grammar.rules
    schema = json.loads(grammar.rules.split("%json ")[1])
    Draft202012Validator.check_schema(schema)
    action_names = [s["properties"]["name"]["const"] for s in schema["anyOf"]]
    assert "function_to_test" in action_names
    assert "RESPOND" in action_names
