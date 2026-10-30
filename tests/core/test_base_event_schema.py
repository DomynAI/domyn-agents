"""The shape of ``BaseEvent`` on the wire.

``BaseEvent`` is what crosses process boundaries — the relay, SSE, the channels,
the conversation store — so a field whose type changes breaks whoever reads it on
the other side, and nothing in the model itself would notice. This pins the JSON
schema (types, defaults, required fields, nested models) to a committed copy.

Titles and descriptions are left out: rewording a docstring is not a contract
change. Enum values may be added (a new event type does not break a reader that
ignores it) but never removed.

After an intended change, regenerate the copy and commit it with the change:

    DOMYN_UPDATE_SCHEMA=1 uv run pytest tests/core/test_base_event_schema.py --no-cov
"""

import json
import os
from pathlib import Path
from typing import Any

from domyn.agents.core import BaseEvent

SCHEMA_FILE = Path(__file__).with_name("base_event_schema.json")
_PROSE = {"title", "description"}


def _shape(schema: Any) -> Any:
    if isinstance(schema, dict):
        return {k: _shape(v) for k, v in schema.items() if k not in _PROSE}
    if isinstance(schema, list):
        return [_shape(v) for v in schema]
    return schema


def _without_enums(schema: Any) -> Any:
    if isinstance(schema, dict):
        return {k: _without_enums(v) for k, v in schema.items() if k != "enum"}
    if isinstance(schema, list):
        return [_without_enums(v) for v in schema]
    return schema


def _enums(schema: Any, path: str = "") -> dict[str, list[Any]]:
    found: dict[str, list[Any]] = {}
    if isinstance(schema, dict):
        for k, v in schema.items():
            if k == "enum":
                found[path] = v
            else:
                found.update(_enums(v, f"{path}/{k}"))
    elif isinstance(schema, list):
        for i, v in enumerate(schema):
            found.update(_enums(v, f"{path}/{i}"))
    return found


def test_the_wire_shape_of_base_event_is_unchanged() -> None:
    actual = _shape(BaseEvent.model_json_schema())
    if os.environ.get("DOMYN_UPDATE_SCHEMA"):
        SCHEMA_FILE.write_text(json.dumps(actual, indent=2, sort_keys=True) + "\n")
    expected = json.loads(SCHEMA_FILE.read_text())

    assert _without_enums(actual) == _without_enums(expected), (
        "BaseEvent's schema changed. If it is intended, regenerate "
        f"{SCHEMA_FILE.name} (see this module's docstring) and commit it."
    )
    actual_enums = _enums(actual)
    for where, values in _enums(expected).items():
        removed = set(values) - set(actual_enums.get(where, []))
        assert not removed, f"enum values removed at {where}: {sorted(removed)}"
