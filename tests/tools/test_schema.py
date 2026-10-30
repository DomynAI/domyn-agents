"""Tests for the tool schema helpers: Python type strings, and MCP JSON Schemas."""

import typing
from typing import Any

import pytest

from domyn.agents.tools.schema import (
    MAX_INLINED_SIZE,
    as_tool_parameters,
    inline_schema_refs,
    json_type_to_python_type,
    python_type_str_to_python_type,
    python_type_to_json_type,
    split_union_types,
)

# test mapping json types


def test_split_union_types():
    assert split_union_types("str | int | None") == ["str", "int", "None"]
    assert split_union_types("List[str] | None") == ["List[str]", "None"]
    assert split_union_types("Dict[str, int] | float") == ["Dict[str, int]", "float"]
    assert split_union_types("int") == ["int"]
    assert split_union_types("List[Dict[str, int | float]] | None") == [
        "List[Dict[str, int | float]]",
        "None",
    ]
    assert split_union_types("Dict[str, List[int | None]] | str | None") == [
        "Dict[str, List[int | None]]",
        "str",
        "None",
    ]
    assert split_union_types("List[str|int]|None") == ["List[str|int]", "None"]
    assert split_union_types("List[str|int] | None") == ["List[str|int]", "None"]


@pytest.mark.parametrize(
    "json_type, expected",
    [
        ("integer", {"type": "integer"}),
        (["string", "number"], {"anyOf": [{"type": "string"}, {"type": "number"}]}),
        (["boolean", "null"], {"anyOf": [{"type": "boolean"}, {"type": "null"}]}),
    ],
)
def test_json_type_to_python_type_reads_back(json_type, expected):
    """The two directions agree: a JSON type named in Python converts back to it."""
    assert python_type_to_json_type(json_type_to_python_type(json_type)) == expected


def test_python_type_to_json_type():
    assert python_type_to_json_type("str") == {"type": "string"}
    assert python_type_to_json_type("int") == {"type": "integer"}
    assert python_type_to_json_type("float") == {"type": "number"}
    assert python_type_to_json_type("bool") == {"type": "boolean"}
    assert python_type_to_json_type("List[str]") == {
        "type": "array",
        "items": {"type": "string"},
    }
    assert python_type_to_json_type("list[Any]") == {"type": "array", "items": {}}
    assert python_type_to_json_type("List[Any]") == {"type": "array", "items": {}}
    assert python_type_to_json_type("list") == {"type": "array", "items": {}}
    assert python_type_to_json_type("List") == {"type": "array", "items": {}}
    assert python_type_to_json_type("dict") == {"type": "object", "additionalProperties": {}}
    assert python_type_to_json_type("Dict") == {"type": "object", "additionalProperties": {}}
    assert python_type_to_json_type("List|int") == {
        "anyOf": [{"type": "array", "items": {}}, {"type": "integer"}]
    }
    assert python_type_to_json_type("dict[int, str]") == {
        "type": "object",
        "additionalProperties": {"type": "string"},
    }
    assert python_type_to_json_type("Dict[Any, Any]") == {
        "type": "object",
        "additionalProperties": {},
    }
    assert python_type_to_json_type("dict[str, Any]") == {
        "type": "object",
        "additionalProperties": {},
    }
    assert python_type_to_json_type("Dict[str, Any]") == {
        "type": "object",
        "additionalProperties": {},
    }
    assert python_type_to_json_type("List[int]") == {
        "type": "array",
        "items": {"type": "integer"},
    }
    assert python_type_to_json_type("list[float]") == {
        "type": "array",
        "items": {"type": "number"},
    }

    assert python_type_to_json_type("List[str|int]") == {
        "type": "array",
        "items": {"anyOf": [{"type": "string"}, {"type": "integer"}]},
    }
    assert python_type_to_json_type("str | None") == {
        "anyOf": [{"type": "string"}, {"type": "null"}]
    }
    assert python_type_to_json_type("int | float") == {
        "anyOf": [{"type": "integer"}, {"type": "number"}]
    }
    assert python_type_to_json_type("str | float | None") == {
        "anyOf": [{"type": "string"}, {"type": "number"}, {"type": "null"}]
    }
    assert python_type_to_json_type("something") == {}
    assert python_type_to_json_type("Dict[str, int]") == {
        "type": "object",
        "additionalProperties": {"type": "integer"},
    }
    assert python_type_to_json_type("dict[str, float]") == {
        "type": "object",
        "additionalProperties": {"type": "number"},
    }

    assert python_type_to_json_type("List[Dict[str, int]]") == {
        "type": "array",
        "items": {
            "type": "object",
            "additionalProperties": {"type": "integer"},
        },
    }

    assert python_type_to_json_type("Dict[str, int | None]") == {
        "type": "object",
        "additionalProperties": {"anyOf": [{"type": "integer"}, {"type": "null"}]},
    }

    assert python_type_to_json_type("Dict[str, List[float|str]]") == {
        "type": "object",
        "additionalProperties": {
            "type": "array",
            "items": {"anyOf": [{"type": "number"}, {"type": "string"}]},
        },
    }

    assert python_type_to_json_type("Dict[str, Dict[str, int]]") == {
        "type": "object",
        "additionalProperties": {
            "type": "object",
            "additionalProperties": {"type": "integer"},
        },
    }

    assert python_type_to_json_type("Dict[str, dict[str, int]]") == {
        "type": "object",
        "additionalProperties": {
            "type": "object",
            "additionalProperties": {"type": "integer"},
        },
    }

    assert python_type_to_json_type("Dict[str, List[Dict[str, float]] ]") == {
        "type": "object",
        "additionalProperties": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": {"type": "number"},
            },
        },
    }

    assert python_type_to_json_type("list[str] | None") == {
        "anyOf": [
            {
                "type": "array",
                "items": {"type": "string"},
            },
            {"type": "null"},
        ]
    }

    assert python_type_to_json_type("list[str|int] | None") == {
        "anyOf": [
            {
                "type": "array",
                "items": {"anyOf": [{"type": "string"}, {"type": "integer"}]},
            },
            {"type": "null"},
        ]
    }

    assert python_type_to_json_type("dict[str, str] | None") == {
        "anyOf": [
            {
                "type": "object",
                "additionalProperties": {"type": "string"},
            },
            {"type": "null"},
        ]
    }

    assert python_type_to_json_type("Dict[str, str|int] | None") == {
        "anyOf": [
            {
                "type": "object",
                "additionalProperties": {"anyOf": [{"type": "string"}, {"type": "integer"}]},
            },
            {"type": "null"},
        ]
    }

    assert python_type_to_json_type("List[str | Dict[str, int] | float]") == {
        "type": "array",
        "items": {
            "anyOf": [
                {"type": "string"},
                {
                    "type": "object",
                    "additionalProperties": {"type": "integer"},
                },
                {"type": "number"},
            ]
        },
    }


def test_python_type_str_to_python_type_base_types():
    assert python_type_str_to_python_type("str") is str
    assert python_type_str_to_python_type("int") is int
    assert python_type_str_to_python_type("float") is float
    assert python_type_str_to_python_type("bool") is bool
    assert python_type_str_to_python_type("list") is list
    assert python_type_str_to_python_type("dict") is dict
    assert python_type_str_to_python_type("bytes") is bytes
    assert python_type_str_to_python_type("None") is type(None)
    assert python_type_str_to_python_type("Any") is typing.Any


def test_python_type_str_to_python_type_generic_list():
    assert python_type_str_to_python_type("list[str]") == list[str]
    assert python_type_str_to_python_type("List[int]") == list[int]
    assert python_type_str_to_python_type("list[float]") == list[float]
    assert python_type_str_to_python_type("list[list[str]]") == list[list[str]]


def test_python_type_str_to_python_type_generic_dict():
    assert python_type_str_to_python_type("dict[str, int]") == dict[str, int]
    assert python_type_str_to_python_type("Dict[str, float]") == dict[str, float]


def test_python_type_str_to_python_type_optional():
    result = python_type_str_to_python_type("Optional[str]")
    args = typing.get_args(result)
    assert str in args
    assert type(None) in args


def test_python_type_str_to_python_type_pep604_union():
    result = python_type_str_to_python_type("str | None")
    args = typing.get_args(result)
    assert str in args
    assert type(None) in args


def test_python_type_str_to_python_type_union():
    result = python_type_str_to_python_type("Union[str, int]")
    args = typing.get_args(result)
    assert str in args
    assert int in args


def test_python_type_str_to_python_type_unknown_falls_back_to_str():
    assert python_type_str_to_python_type("uuid") is str
    assert python_type_str_to_python_type("MyCustomType") is str


def walk(node: Any):
    """Yield every dict in a schema, at any depth (a substring check would match `#/$defs/`)."""
    if isinstance(node, dict):
        yield node
        for value in node.values():
            yield from walk(value)
    elif isinstance(node, list):
        for item in node:
            yield from walk(item)


def keys_anywhere(schema: Any) -> set[str]:
    return {key for node in walk(schema) for key in node}


def size(node: Any) -> int:
    if isinstance(node, dict):
        return 1 + sum(size(v) for v in node.values())
    if isinstance(node, list):
        return 1 + sum(size(v) for v in node)
    return 1


def all_connected(n: int) -> dict[str, Any]:
    """``n`` definitions that each reference every one of them."""
    names = [f"D{i}" for i in range(n)]
    return {
        "type": "object",
        "properties": {"root": {"$ref": "#/$defs/D0"}},
        "$defs": {
            name: {
                "type": "object",
                "properties": {f"to_{other}": {"$ref": f"#/$defs/{other}"} for other in names},
            }
            for name in names
        },
    }


class TestTheDefinitionsAreInlined:
    def test_a_ref_becomes_its_definition(self):
        resolved = inline_schema_refs(
            {
                "properties": {"a": {"$ref": "#/$defs/A"}},
                "$defs": {"A": {"type": "string", "maxLength": 3}},
            }
        )

        assert resolved["properties"]["a"] == {"type": "string", "maxLength": 3}

    def test_the_definition_block_is_dropped(self):
        resolved = inline_schema_refs(
            {"properties": {"a": {"$ref": "#/$defs/A"}}, "$defs": {"A": {"type": "string"}}}
        )

        assert "$defs" not in keys_anywhere(resolved)
        assert "$ref" not in keys_anywhere(resolved)

    def test_draft_07_definitions_resolve_too(self):
        """Not every MCP server writes the same JSON Schema draft."""
        resolved = inline_schema_refs(
            {
                "properties": {"a": {"$ref": "#/definitions/A"}},
                "definitions": {"A": {"type": "integer"}},
            }
        )

        assert resolved["properties"]["a"] == {"type": "integer"}
        assert "definitions" not in keys_anywhere(resolved)

    def test_a_ref_inside_items_resolves(self):
        """The shape that started this: an array of objects."""
        resolved = inline_schema_refs(
            {
                "properties": {
                    "attachments": {"type": "array", "items": {"$ref": "#/$defs/Attachment"}}
                },
                "$defs": {
                    "Attachment": {
                        "type": "object",
                        "properties": {"content": {"type": "string"}},
                        "required": ["content"],
                    }
                },
            }
        )

        items = resolved["properties"]["attachments"]["items"]
        assert items["properties"]["content"] == {"type": "string"}
        assert items["required"] == ["content"]

    def test_a_ref_inside_any_of_resolves(self):
        resolved = inline_schema_refs(
            {
                "properties": {"a": {"anyOf": [{"$ref": "#/$defs/A"}, {"type": "null"}]}},
                "$defs": {"A": {"type": "string"}},
            }
        )

        assert resolved["properties"]["a"]["anyOf"] == [{"type": "string"}, {"type": "null"}]

    def test_a_definition_that_refs_another_definition_resolves(self):
        resolved = inline_schema_refs(
            {
                "properties": {"a": {"$ref": "#/$defs/Outer"}},
                "$defs": {
                    "Outer": {"type": "object", "properties": {"inner": {"$ref": "#/$defs/Inner"}}},
                    "Inner": {"type": "boolean"},
                },
            }
        )

        assert resolved["properties"]["a"]["properties"]["inner"] == {"type": "boolean"}

    def test_the_root_pointer_resolves(self):
        resolved = inline_schema_refs({"type": "object", "properties": {"self": {"$ref": "#"}}})

        assert resolved["properties"]["self"]["type"] == "object"
        assert "$ref" not in keys_anywhere(resolved)

    def test_the_root_pointer_does_not_bring_the_definition_block_along(self):
        resolved = inline_schema_refs(
            {
                "type": "object",
                "properties": {"self": {"$ref": "#"}, "a": {"$ref": "#/$defs/A"}},
                "$defs": {"A": {"type": "string"}},
            }
        )

        assert "$defs" not in keys_anywhere(resolved)

    def test_a_boolean_definition_is_inlined_as_is(self):
        resolved = inline_schema_refs(
            {"properties": {"a": {"$ref": "#/$defs/Anything"}}, "$defs": {"Anything": True}}
        )

        assert resolved["properties"]["a"] is True


class TestTheSiblingsSurvive:
    def test_a_description_next_to_a_ref_is_kept(self):
        """JSON Schema 2020-12 lets keywords sit next to a $ref."""
        resolved = inline_schema_refs(
            {
                "properties": {"a": {"$ref": "#/$defs/A", "description": "what it is for"}},
                "$defs": {"A": {"type": "string"}},
            }
        )

        assert resolved["properties"]["a"] == {"type": "string", "description": "what it is for"}

    def test_a_sibling_wins_over_the_definition(self):
        resolved = inline_schema_refs(
            {
                "properties": {"a": {"$ref": "#/$defs/A", "description": "local"}},
                "$defs": {"A": {"type": "string", "description": "shared"}},
            }
        )

        assert resolved["properties"]["a"]["description"] == "local"

    def test_siblings_survive_an_unresolvable_ref(self):
        resolved = inline_schema_refs({"properties": {"a": {"$ref": "#/$defs/Gone", "title": "A"}}})

        assert resolved["properties"]["a"] == {"title": "A"}


class TestAnUnresolvableRefIsReported:
    def test_a_missing_definition_is_dropped_and_reported(self):
        seen: list[str] = []

        resolved = inline_schema_refs(
            {"properties": {"a": {"$ref": "#/$defs/Missing"}}, "$defs": {"Other": {}}},
            on_unresolved=seen.append,
        )

        assert "$ref" not in keys_anywhere(resolved)
        assert seen == ["#/$defs/Missing"]

    def test_a_remote_ref_is_dropped_and_reported(self):
        """The model cannot fetch it either, so a dangling pointer helps nobody."""
        seen: list[str] = []

        resolved = inline_schema_refs(
            {"properties": {"a": {"$ref": "https://example.com/schema.json"}}},
            on_unresolved=seen.append,
        )

        assert "$ref" not in keys_anywhere(resolved)
        assert seen == ["https://example.com/schema.json"]

    def test_nothing_is_reported_when_everything_resolves(self):
        seen: list[str] = []

        inline_schema_refs(
            {"properties": {"a": {"$ref": "#/$defs/A"}}, "$defs": {"A": {"type": "string"}}},
            on_unresolved=seen.append,
        )

        assert seen == []

    def test_no_callback_is_tolerated(self):
        assert inline_schema_refs({"a": {"$ref": "#/$defs/Missing"}}) == {"a": {}}

    def test_a_pointer_is_reported_once_however_often_it_occurs(self):
        seen: list[str] = []

        inline_schema_refs(
            {
                "properties": {
                    "a": {"$ref": "#/$defs/Missing"},
                    "b": {"type": "array", "items": {"$ref": "#/$defs/Missing"}},
                }
            },
            on_unresolved=seen.append,
        )

        assert seen == ["#/$defs/Missing"]

    def test_an_unused_definition_is_not_reported(self):
        """The definition blocks are dropped, so what nothing points at is never walked."""
        seen: list[str] = []

        inline_schema_refs(
            {
                "properties": {"a": {"type": "string"}},
                "$defs": {"Unused": {"properties": {"x": {"$ref": "#/$defs/Gone"}}}},
            },
            on_unresolved=seen.append,
        )

        assert seen == []


class TestRecursionTerminates:
    def test_a_self_referencing_definition_does_not_loop(self):
        resolved = inline_schema_refs(
            {
                "properties": {"node": {"$ref": "#/$defs/Node"}},
                "$defs": {
                    "Node": {
                        "type": "object",
                        "properties": {"child": {"$ref": "#/$defs/Node"}},
                    }
                },
            }
        )

        assert resolved["properties"]["node"]["properties"]["child"] == {
            "type": "object",
            "description": "Node",
        }

    def test_mutually_recursive_definitions_do_not_loop(self):
        resolved = inline_schema_refs(
            {
                "properties": {"a": {"$ref": "#/$defs/A"}},
                "$defs": {
                    "A": {"type": "object", "properties": {"b": {"$ref": "#/$defs/B"}}},
                    "B": {"type": "object", "properties": {"a": {"$ref": "#/$defs/A"}}},
                },
            }
        )

        assert resolved["properties"]["a"]["properties"]["b"]["properties"]["a"] == {
            "type": "object",
            "description": "A",
        }

    def test_the_same_definition_twice_on_different_branches_expands_twice(self):
        """The guard is per path, not per document: reuse is not recursion."""
        resolved = inline_schema_refs(
            {
                "properties": {"a": {"$ref": "#/$defs/A"}, "b": {"$ref": "#/$defs/A"}},
                "$defs": {"A": {"type": "string"}},
            }
        )

        assert resolved["properties"]["a"] == resolved["properties"]["b"] == {"type": "string"}


class TestTheSizeIsCapped:
    def test_definitions_that_all_reference_each_other_stay_small(self):
        """Eight of them used to inline to 6 MB, after seconds of CPU."""
        resolved = inline_schema_refs(all_connected(8))

        assert size(resolved) < 2 * MAX_INLINED_SIZE
        assert "$ref" not in keys_anywhere(resolved)

    def test_the_cut_is_reported_once_with_a_count(self):
        cuts: list[int] = []

        inline_schema_refs(all_connected(8), on_truncated=cuts.append)

        assert len(cuts) == 1
        assert cuts[0] > 0

    def test_nothing_is_reported_under_the_cap(self):
        cuts: list[int] = []

        inline_schema_refs(all_connected(3), on_truncated=cuts.append)

        assert cuts == []

    def test_what_is_cut_is_the_deepest_not_the_last_parameter(self):
        """Breadth first: a deep first parameter cannot starve a shallow second one."""
        chain = {
            f"L{i}": {"type": "object", "properties": {"next": {"$ref": f"#/$defs/L{i + 1}"}}}
            for i in range(20)
        }
        chain["L20"] = {"type": "string"}
        resolved = inline_schema_refs(
            {
                "properties": {"deep": {"$ref": "#/$defs/L0"}, "flat": {"$ref": "#/$defs/Flat"}},
                "$defs": {**chain, "Flat": {"type": "object", "properties": {"x": {}, "y": {}}}},
            },
            # Depth first, the chain (30) would leave no room for `flat` (5).
            max_size=34,
        )

        assert resolved["properties"]["flat"] == {
            "type": "object",
            "properties": {"x": {}, "y": {}},
        }
        assert resolved["properties"]["deep"]["properties"]["next"]["type"] == "object"

    def test_a_large_body_does_not_eat_the_budget(self):
        """Only what inlining adds counts: a big schema still gets its small refs."""
        many = {f"p{i}": {"type": "string", "enum": ["a", "b"]} for i in range(20)}
        resolved = inline_schema_refs(
            {
                "properties": {**many, "a": {"$ref": "#/$defs/A"}},
                "$defs": {"A": {"type": "string"}},
            },
            max_size=5,
        )

        assert resolved["properties"]["a"] == {"type": "string"}

    def test_a_long_enum_reused_by_several_parameters_is_inlined_everywhere(self):
        zones = {"type": "string", "enum": [f"Zone/{i}" for i in range(400)]}
        resolved = inline_schema_refs(
            {
                "properties": {p: {"$ref": "#/$defs/TimeZone"} for p in ("a", "b", "c")},
                "$defs": {"TimeZone": zones},
            }
        )

        assert all(resolved["properties"][p] == zones for p in ("a", "b", "c"))

    def test_a_cut_definition_keeps_its_type(self):
        resolved = inline_schema_refs(
            {
                "properties": {"a": {"$ref": "#/$defs/Colour", "description": "which one"}},
                "$defs": {"Colour": {"type": "string", "enum": ["red", "green", "blue"]}},
            },
            max_size=4,
        )

        assert resolved["properties"]["a"] == {"type": "string", "description": "which one"}


class TestTheResultIsAlwaysFresh:
    def test_a_schema_without_refs_is_copied_not_returned(self):
        """MCP schemas come from a shared cache and the caller annotates them."""
        source = {"properties": {"a": {"type": "string"}}}

        resolved = inline_schema_refs(source)

        assert resolved == source
        assert resolved is not source
        assert resolved["properties"]["a"] is not source["properties"]["a"]

    def test_annotating_the_result_leaves_the_source_alone(self):
        source = {"properties": {"a": {"$ref": "#/$defs/A"}}, "$defs": {"A": {"type": "string"}}}

        resolved = inline_schema_refs(source)
        resolved["properties"]["a"]["description"] = "added later"

        assert source["$defs"]["A"] == {"type": "string"}

    def test_a_list_is_copied(self):
        source = {"required": ["a"]}

        assert inline_schema_refs(source)["required"] is not source["required"]


class TestTheOddInputsAreTolerated:
    def test_an_empty_schema_survives(self):
        assert inline_schema_refs({}) == {}

    def test_a_non_dollar_ref_key_is_left_alone(self):
        assert inline_schema_refs({"properties": {"ref": {"type": "string"}}}) == {
            "properties": {"ref": {"type": "string"}}
        }

    def test_a_non_string_ref_is_not_treated_as_a_pointer(self):
        assert inline_schema_refs({"properties": {"$ref": 3}}) == {"properties": {"$ref": 3}}

    def test_an_escaped_pointer_token_resolves(self):
        resolved = inline_schema_refs(
            {"properties": {"a": {"$ref": "#/$defs/a~1b"}}, "$defs": {"a/b": {"type": "string"}}}
        )

        assert resolved["properties"]["a"] == {"type": "string"}


class TestAsToolParameters:
    def test_an_array_without_items_gets_them(self):
        """Chat Completions validators reject it: "array schema missing items"."""
        parameters = as_tool_parameters({"properties": {"tags": {"type": "array"}}})

        assert parameters["properties"]["tags"] == {"type": "array", "items": {}}

    def test_arrays_get_items_at_any_depth(self):
        parameters = as_tool_parameters(
            {
                "properties": {
                    "rows": {"type": "array", "items": {"type": "array"}},
                    "maybe": {"anyOf": [{"type": "array"}, {"type": "null"}]},
                    "nested": {"type": "object", "properties": {"inner": {"type": "array"}}},
                    "nullable": {"type": ["array", "null"]},
                }
            }
        )
        properties = parameters["properties"]

        assert properties["rows"]["items"] == {"type": "array", "items": {}}
        assert properties["maybe"]["anyOf"][0] == {"type": "array", "items": {}}
        assert properties["nested"]["properties"]["inner"]["items"] == {}
        assert properties["nullable"]["items"] == {}

    def test_declared_items_are_left_alone(self):
        parameters = as_tool_parameters(
            {"properties": {"to": {"type": "array", "items": {"type": "string"}}}}
        )

        assert parameters["properties"]["to"] == {"type": "array", "items": {"type": "string"}}

    def test_draft_07_tuple_items_are_followed(self):
        parameters = as_tool_parameters(
            {"properties": {"pair": {"type": "array", "items": [{"type": "array"}, {}]}}}
        )

        assert parameters["properties"]["pair"]["items"][0] == {"type": "array", "items": {}}

    def test_data_that_looks_like_an_array_schema_is_not_touched(self):
        """`default`, `enum`, `examples` hold values, not schemas."""
        parameters = as_tool_parameters(
            {
                "properties": {
                    "shape": {
                        "type": "object",
                        "default": {"type": "array"},
                        "examples": [{"type": "array"}],
                    }
                }
            }
        )

        assert parameters["properties"]["shape"]["default"] == {"type": "array"}
        assert parameters["properties"]["shape"]["examples"] == [{"type": "array"}]

    def test_a_parameter_named_like_a_keyword_is_still_a_parameter(self):
        parameters = as_tool_parameters(
            {"properties": {"type": {"type": "array"}, "items": {"type": "string"}}}
        )

        assert parameters["properties"]["type"] == {"type": "array", "items": {}}
        assert parameters["properties"]["items"] == {"type": "string"}

    def test_top_level_combinators_are_dropped(self):
        """Anthropic rejects them in input_schema, and one such tool fails every turn."""
        parameters = as_tool_parameters(
            {
                "type": "object",
                "properties": {"a": {"type": "string"}, "b": {"type": "string"}},
                "anyOf": [{"required": ["a"]}, {"required": ["b"]}],
                "oneOf": [{}],
                "allOf": [{}],
                "not": {},
                "enum": [{}],
            }
        )

        assert parameters == {
            "type": "object",
            "properties": {"a": {"type": "string"}, "b": {"type": "string"}},
            "required": [],
        }

    def test_nested_combinators_are_kept(self):
        parameters = as_tool_parameters(
            {"properties": {"a": {"anyOf": [{"type": "string"}, {"type": "null"}]}}}
        )

        assert parameters["properties"]["a"]["anyOf"] == [{"type": "string"}, {"type": "null"}]

    def test_the_dialect_declaration_is_dropped(self):
        parameters = as_tool_parameters({"$schema": "https://json-schema.org/draft/2020-12/schema"})

        assert "$schema" not in parameters

    def test_the_missing_keys_are_filled_in(self):
        assert as_tool_parameters({}) == {"type": "object", "properties": {}, "required": []}

    def test_the_input_is_left_alone(self):
        source = {"properties": {"tags": {"type": "array"}}, "anyOf": [{}]}

        parameters = as_tool_parameters(source)
        parameters["properties"]["tags"]["description"] = "added later"

        assert source == {"properties": {"tags": {"type": "array"}}, "anyOf": [{}]}
