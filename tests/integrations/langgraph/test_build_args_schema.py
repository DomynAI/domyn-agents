import pytest
from pydantic import ValidationError

from domyn.agents.integrations.langgraph.domyn_platform import _build_args_schema


def _make_param(name: str, type_str: str, **kwargs):
    return {"name": name, "type": type_str, **kwargs}


def test_list_str_field_validates_list():
    model = _build_args_schema("t", [_make_param("tags", "list[str]")])
    instance = model(tags=["a", "b"])
    assert instance.tags == ["a", "b"]


def test_list_str_field_rejects_scalar():
    model = _build_args_schema("t", [_make_param("tags", "list[str]")])
    with pytest.raises(ValidationError):
        model(tags="not-a-list")


def test_dict_str_int_field_validates():
    model = _build_args_schema("t", [_make_param("counts", "dict[str, int]")])
    instance = model(counts={"k": 1})
    assert instance.counts == {"k": 1}


def test_optional_field_accepts_none():
    model = _build_args_schema("t", [_make_param("note", "Optional[str]")])
    instance = model(note=None)
    assert instance.note is None


def test_optional_field_accepts_value():
    model = _build_args_schema("t", [_make_param("note", "Optional[str]")])
    instance = model(note="hello")
    assert instance.note == "hello"


def test_pep604_nullable_accepts_none():
    model = _build_args_schema("t", [_make_param("note", "str | None")])
    instance = model(note=None)
    assert instance.note is None


def test_pep604_nullable_accepts_value():
    model = _build_args_schema("t", [_make_param("note", "str | None")])
    instance = model(note="hello")
    assert instance.note == "hello"


def test_unknown_type_falls_back_to_str():
    model = _build_args_schema("t", [_make_param("uid", "uuid")])
    instance = model(uid="some-value")
    assert instance.uid == "some-value"
