"""Convert platform tool definitions to hermes JSON Schema format."""

from typing import Any

from domyn.agents.tools.schema import python_type_to_json_type


def convert_schema(tool_def: dict[str, Any]) -> dict[str, Any]:
    """Convert a platform tool definition to hermes JSON Schema format.

    A field is required unless is_required is explicitly False or a default is present.
    """
    name = tool_def["name"]
    description = tool_def.get("description", "")
    param_list: list[dict[str, Any]] = tool_def.get("parameters") or []

    properties: dict[str, dict[str, Any]] = {}
    required: list[str] = []

    for p in param_list:
        p_name = p.get("name")
        if not p_name:
            continue

        prop: dict[str, Any] = dict(python_type_to_json_type(p.get("type", "str")))

        p_desc = p.get("description", "")
        if p_desc:
            prop["description"] = p_desc

        if "default" in p:
            prop["default"] = p["default"]
        elif p.get("is_required", True) is not False:
            required.append(p_name)

        properties[p_name] = prop

    schema: dict[str, Any] = {
        "name": name,
        "description": description,
        "parameters": {
            "type": "object",
            "properties": properties,
        },
    }
    if required:
        schema["parameters"]["required"] = required

    return schema
