import json

from domyn.agents.core import (
    Action,
    AgentAction,
    BaseAction,
    BaseEvent,
    ExecutionEventType,
    StopAction,
    ToolAction,
)


def test_serializ_base_action():
    b = BaseAction(
        name="test",
        thought="thought",
        observation="observation",
        message="message",
        parameters={"param": "param_value"},
    )
    b_to_json = b.model_dump_json()
    b_to_dict = b.model_dump()

    assert json.loads(b_to_json) == b_to_dict

    assert Action.validate_json(b_to_json) == b
    assert Action.validate_python(b_to_dict) == b
    assert Action.validate_python(json.loads(b_to_json)) == b


def test_serializ_tool_action():
    t = ToolAction(
        name="test_tool",
        thought="thought",
        observation={"resp": 1},
        message="message",
        parameters={"param": "param_value"},
    )
    t_to_json = t.model_dump_json()
    t_to_dict = t.model_dump()

    assert json.loads(t_to_json) == t_to_dict

    assert Action.validate_json(t_to_json) == t
    assert Action.validate_python(t_to_dict) == t
    assert Action.validate_python(json.loads(t_to_json)) == t


def test_serializ_agent_action():
    a = AgentAction(
        name="test_agent",
        thought="thought",
        observation=None,
        message="message",
        parameters={"param": "param_value"},
        input_event=BaseEvent(event_type=ExecutionEventType.AGENT_START, author="another_agent"),
    )
    a_to_json = a.model_dump_json()
    a_to_dict = a.model_dump()

    # json.loads leaves timestamp as a str, so it isn't compared

    assert Action.validate_json(a_to_json) == a
    assert Action.validate_python(a_to_dict) == a
    assert Action.validate_python(json.loads(a_to_json)) == a


def test_serializ_stop_action():
    s = StopAction(
        name="stop_action", thought=None, observation=None, message="Stopping now", parameters=None
    )
    s_to_json = s.model_dump_json()
    s_to_dict = s.model_dump()

    assert json.loads(s_to_json) == s_to_dict

    assert Action.validate_json(s_to_json) == s
    assert Action.validate_python(s_to_dict) == s
    assert Action.validate_python(json.loads(s_to_json)) == s
