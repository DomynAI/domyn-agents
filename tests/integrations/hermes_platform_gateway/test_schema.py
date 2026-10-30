from domyn.agents.integrations.hermes_platform_gateway import convert_schema


def test_required_field_appears_in_required_list():
    tool = {
        "name": "send_email",
        "description": "Send an email",
        "parameters": [
            {"name": "to", "type": "str", "is_required": True, "description": "Recipient"},
        ],
    }
    schema = convert_schema(tool)
    assert schema["name"] == "send_email"
    assert schema["description"] == "Send an email"
    assert schema["parameters"]["properties"]["to"] == {
        "type": "string",
        "description": "Recipient",
    }
    assert schema["parameters"]["required"] == ["to"]


def test_is_required_false_excluded_from_required():
    tool = {
        "name": "search",
        "parameters": [
            {"name": "query", "type": "str", "is_required": True},
            {"name": "limit", "type": "int", "is_required": False},
        ],
    }
    schema = convert_schema(tool)
    assert "query" in schema["parameters"]["required"]
    assert "limit" not in schema["parameters"].get("required", [])


def test_field_with_default_excluded_from_required():
    tool = {
        "name": "foo",
        "parameters": [
            {"name": "retries", "type": "int", "default": 3},
        ],
    }
    schema = convert_schema(tool)
    assert "required" not in schema["parameters"]
    assert schema["parameters"]["properties"]["retries"]["default"] == 3


def test_unknown_type_produces_unconstrained_schema():
    tool = {"name": "foo", "parameters": [{"name": "x", "type": "uuid"}]}
    schema = convert_schema(tool)
    assert "type" not in schema["parameters"]["properties"]["x"]


def test_all_type_mappings():
    params = [
        {"name": "a", "type": "str"},
        {"name": "b", "type": "int"},
        {"name": "c", "type": "float"},
        {"name": "d", "type": "bool"},
        {"name": "e", "type": "list"},
        {"name": "f", "type": "dict"},
    ]
    schema = convert_schema({"name": "t", "parameters": params})
    props = schema["parameters"]["properties"]
    assert props["a"]["type"] == "string"
    assert props["b"]["type"] == "integer"
    assert props["c"]["type"] == "number"
    assert props["d"]["type"] == "boolean"
    assert props["e"]["type"] == "array"
    assert props["f"]["type"] == "object"


def test_list_with_item_type():
    tool = {"name": "foo", "parameters": [{"name": "tags", "type": "list[str]"}]}
    schema = convert_schema(tool)
    prop = schema["parameters"]["properties"]["tags"]
    assert prop["type"] == "array"
    assert prop["items"] == {"type": "string"}


def test_list_capitalised():
    tool = {"name": "foo", "parameters": [{"name": "ids", "type": "List[int]"}]}
    schema = convert_schema(tool)
    prop = schema["parameters"]["properties"]["ids"]
    assert prop["type"] == "array"
    assert prop["items"] == {"type": "integer"}


def test_list_of_dicts():
    tool = {"name": "foo", "parameters": [{"name": "rows", "type": "list[dict]"}]}
    schema = convert_schema(tool)
    prop = schema["parameters"]["properties"]["rows"]
    assert prop["type"] == "array"
    assert prop["items"]["type"] == "object"


def test_nested_list():
    tool = {"name": "foo", "parameters": [{"name": "matrix", "type": "list[list[str]]"}]}
    schema = convert_schema(tool)
    prop = schema["parameters"]["properties"]["matrix"]
    assert prop == {"type": "array", "items": {"type": "array", "items": {"type": "string"}}}


def test_dict_with_value_type():
    tool = {"name": "foo", "parameters": [{"name": "counts", "type": "dict[str, int]"}]}
    schema = convert_schema(tool)
    prop = schema["parameters"]["properties"]["counts"]
    assert prop["type"] == "object"
    assert prop["additionalProperties"] == {"type": "integer"}


def test_dict_any_value():
    tool = {"name": "foo", "parameters": [{"name": "data", "type": "dict[str, Any]"}]}
    schema = convert_schema(tool)
    prop = schema["parameters"]["properties"]["data"]
    assert prop["type"] == "object"


def test_nullable_union():
    tool = {"name": "foo", "parameters": [{"name": "note", "type": "str | None"}]}
    schema = convert_schema(tool)
    prop = schema["parameters"]["properties"]["note"]
    assert "anyOf" in prop
    types_in_any_of = [s.get("type") for s in prop["anyOf"]]
    assert "string" in types_in_any_of
    assert "null" in types_in_any_of


def test_no_parameters_produces_empty_properties():
    tool = {"name": "ping", "description": "Ping", "parameters": []}
    schema = convert_schema(tool)
    assert schema["parameters"]["properties"] == {}
    assert "required" not in schema["parameters"]


def test_parameter_without_name_is_skipped():
    tool = {
        "name": "foo",
        "parameters": [
            {"type": "str"},  # no name
            {"name": "ok", "type": "str", "is_required": True},
        ],
    }
    schema = convert_schema(tool)
    assert list(schema["parameters"]["properties"].keys()) == ["ok"]
