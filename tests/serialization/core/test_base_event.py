import json
import random
from json import JSONEncoder

from domyn.agents.core import BaseAction, BaseEvent, ExecutionEventType, Part


def test_serialization_base_event():
    event = BaseEvent(
        event_type=random.choice(list(ExecutionEventType)),
        author="test_author",
        interaction_id="test_interaction_id",
        content=[Part(text="This is a test part."), Part(code="print('Hello, World!')")],
        action=BaseAction(name="test_action", parameters={"param1": {"key": "value"}}),
    )

    # Serialize to dict
    event_dict = event.model_dump()

    # Deserialize back to BaseEvent
    deserialized_event = BaseEvent.model_validate(event_dict)

    assert event == deserialized_event

    # json
    event_json = event.model_dump_json()
    deserialized_event_from_json = BaseEvent.model_validate_json(event_json)
    assert event == deserialized_event_from_json
    assert BaseEvent.model_validate(json.loads(event_json)) == event
