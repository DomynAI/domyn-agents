from enum import StrEnum, auto
from typing import Any

from pydantic import BaseModel


class ResponseType(StrEnum):
    RESPOND = auto()
    FORWARD_VERBATIM = auto()
    STOP = auto()
    TOOL_AGENT_CALL = auto()
    ERROR = auto()


class ActionErrors(StrEnum):
    JSONDecodeError = "JSONDecodeError"
    UNPARSABLE_ACTIONS = "UNPARSABLE_RESPONSE"


class ParsedAction(BaseModel):
    action_name: str | None
    response_type: ResponseType
    action_arguments: dict[str, Any] | None
    thought: str | None
    message: str | None
    error: str | None = None


class SpecialAction(BaseModel):
    name: str
    description: str
    parameters_schema: dict
    parameters_string: str = "arguments"
    action_name_string: str = "name"

    @property
    def invocation_json_schema(self):
        schema = {
            "properties": {
                self.action_name_string: {"const": self.name, "type": "string"},
                self.parameters_string: self.parameters_schema,
            },
            "required": [self.parameters_string],
            "description": self.description,
        }
        return schema
