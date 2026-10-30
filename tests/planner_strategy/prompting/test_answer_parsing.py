import pytest

from domyn.agents.planner_strategy import JSONPrompting
from domyn.agents.planner_strategy.prompting.data import ParsedAction, ResponseType
from domyn.agents.planner_strategy.prompting.structural_tags_prompting import (
    StructuralTagsPrompting,
)


@pytest.fixture
def structural_tags_prompting() -> StructuralTagsPrompting:
    return StructuralTagsPrompting(type="base", name="BasePromptingTest")


@pytest.fixture
def json_schema_prompting() -> JSONPrompting:
    return JSONPrompting(type="base", name="BasePromptingTest")


structural_tags_answer_parsing_testdata = [
    (
        'Agent: CustomerServiceAgent\n\n<thought>The requester asked for the status of their orders. To answer correctly, I first need to identify the user by their name. However, the user\'s name was not provided. Therefore, I need to ask the user for their name so I can retrieve their user ID and then get the status of their orders.</thought><action>{"name": "RESPOND", "arguments": {"text": "To check the status of your orders, I need your full name. Could you provide it, please?", "need_feedback": true}}</action><message>I\'m asking the user for their full name to identify them and proceed with the request.</message>',
        [
            ParsedAction(
                action_name="RESPOND",
                response_type=ResponseType.RESPOND,
                action_arguments={
                    "text": "To check the status of your orders, I need your full name. Could you provide it, please?",
                    "need_feedback": True,
                },
                message="I'm asking the user for their full name to identify them and proceed with the request.",
                thought="The requester asked for the status of their orders. To answer correctly, I first need to identify the user by their name. However, the user's name was not provided. Therefore, I need to ask the user for their name so I can retrieve their user ID and then get the status of their orders.",
            )
        ],
    ),
    (
        'Agent: CustomerServiceAgent\n\n<thought>The customer provided the name "john smith". Now I need to use the tool to extract the user\'s ID from this name before proceeding with the order request.</thought><action>{"name": "base-group.get_user_id_from_name", "arguments": {"name_of_the_user": "john smith"}}</action><message>Looking up your user ID using the name provided: John Smith.</message>',
        [
            ParsedAction(
                action_name="base-group.get_user_id_from_name",
                response_type=ResponseType.TOOL_AGENT_CALL,
                action_arguments={"name_of_the_user": "john smith"},
                message="Looking up your user ID using the name provided: John Smith.",
                thought='The customer provided the name "john smith". Now I need to use the tool to extract the user\'s ID from this name before proceeding with the order request.',
            )
        ],
    ),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("response, expected_parsed", structural_tags_answer_parsing_testdata)
async def test_structural_tags_answer_parsing(
    structural_tags_prompting: StructuralTagsPrompting,
    response: str,
    expected_parsed: list[ParsedAction],
):
    parsed = await structural_tags_prompting.parse_llm_response(response, use_stop=False)
    assert parsed == expected_parsed


json_schema_answer_parsing_testdata = [
    (
        'Agent: CustomerServiceAgent\n\n{"thought": "The user is asking for the status of their orders. To assist, I first need to identify the user by their name to retrieve their user ID. However, the user has not provided their name. I need to ask for clarification.", "action": { "name": "RESPOND", "arguments": { "text": "To check the status of your orders, I need your full name. Could you provide it, please?", "need_feedback": true } } }',
        [
            ParsedAction(
                action_name="RESPOND",
                response_type=ResponseType.RESPOND,
                action_arguments={
                    "text": "To check the status of your orders, I need your full name. Could you provide it, please?",
                    "need_feedback": True,
                },
                thought="The user is asking for the status of their orders. To assist, I first need to identify the user by their name to retrieve their user ID. However, the user has not provided their name. I need to ask for clarification.",
                message=None,
            )
        ],
    ),
    (
        'Agent: CustomerServiceAgent\n\n{\n  "thought": "The user has provided their full name, \'John Smith\'. I will now use the tool \'get_user_id_from_name\' to retrieve the user ID associated with this name. Once I have the user ID, I can proceed to query the OrderAgent for the status of their orders.",\n  "action": {\n    "name": "get_user_id_from_name",\n    "arguments": {\n      "name_of_the_user": "John Smith"\n    }\n  },\n  "message": "Retrieving your user ID based on the name you provided."\n}',
        [
            ParsedAction(
                action_name="get_user_id_from_name",
                response_type=ResponseType.TOOL_AGENT_CALL,
                action_arguments={"name_of_the_user": "John Smith"},
                thought="The user has provided their full name, 'John Smith'. I will now use the tool 'get_user_id_from_name' to retrieve the user ID associated with this name. Once I have the user ID, I can proceed to query the OrderAgent for the status of their orders.",
                message="Retrieving your user ID based on the name you provided.",
            )
        ],
    ),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("response, expected_parsed", json_schema_answer_parsing_testdata)
async def test_json_schema_answer_parsing(
    json_schema_prompting: JSONPrompting, response: str, expected_parsed: list[ParsedAction]
):
    parsed = await json_schema_prompting.parse_llm_response(response, use_stop=False)
    assert parsed == expected_parsed
