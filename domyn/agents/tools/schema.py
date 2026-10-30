"""JSON Schema helpers shared by the tools.

A function's parameters are Python type strings: :func:`python_type_to_json_type` turns one
into a schema, :func:`python_type_str_to_python_type` into a Python type, and
:func:`json_type_to_python_type` names the one an MCP parameter's ``type`` stands for.

An MCP tool arrives with a JSON Schema instead. :func:`inline_schema_refs` replaces each
local ``$ref`` with its definition once, when the tool is loaded: the react path nests the
schema where ``#/$defs/...`` no longer resolves, and on vLLM the schema reaches the model as
text. :func:`as_tool_parameters` then adapts a copy to what the hosted providers accept.
"""

import re
from collections import deque
from collections.abc import Callable
from copy import deepcopy
from typing import Any


def split_union_types(type_str: str) -> list[str]:
    """Split a union type string into its component types considering also nested cases like List[int|str] | None."""

    parts = []
    current_part = []
    bracket_level = 0

    for char in type_str:
        if char == "[":
            bracket_level += 1
            current_part.append(char)
        elif char == "]":
            bracket_level -= 1
            current_part.append(char)
        elif (
            char == "|" and bracket_level == 0
        ):  # If we are at top level all the character before is a part
            parts.append("".join(current_part).strip())
            current_part = []
        else:  # If not a top level just add the character to the current part
            current_part.append(char)
    if current_part:
        parts.append("".join(current_part).strip())

    return parts


# Base Python type names and their JSON Schema types, both ways.
_PY_TO_JSON_TYPE = {
    "str": "string",
    "int": "integer",
    "float": "number",
    "bool": "boolean",
    "None": "null",
}
_JSON_TO_PY_TYPE = {json: py for py, json in _PY_TO_JSON_TYPE.items()} | {
    "array": "list",
    "object": "dict",
}


def json_type_to_python_type(json_type: Any) -> str:
    """Python type name for a JSON Schema ``type``, ``"any"`` when it has none.

    ``type`` may be a list (``["string", "number"]``), valid JSON Schema that some MCP
    servers emit: it becomes a union, ``"str | float"``, which
    :func:`python_type_to_json_type` reads back.
    """
    members = json_type if isinstance(json_type, list) else [json_type]
    names: list[str] = []
    for t in members:
        name = _JSON_TO_PY_TYPE.get(t) if isinstance(t, str) else None
        if name is None:
            return "any"
        if name not in names:
            names.append(name)
    # Alone "null" is no usable type; in a union it makes the parameter nullable.
    if names in ([], ["None"]):
        return "any"
    return " | ".join(names)


def python_type_to_json_type(type_str: str) -> Any:
    type_str = type_str.strip()

    # --- Literal ---
    # TODO: support Literal (emit enum). The declaration keeps only "Literal", not its values, so
    # the parameter gets no type.

    # List/list (bare or parametrized)
    m = re.fullmatch(r"(?:List|list)(?:\[(.+)\])?", type_str)
    if m:
        inner = m.group(1)
        item_type = python_type_to_json_type(inner.strip()) if inner else {}
        return {"type": "array", "items": item_type}

    # Dict/dict (bare or parametrized)
    m = re.fullmatch(r"(?:Dict|dict)(?:\[(?:([^,]+),(.+))\])?$", type_str)
    if m:
        key_type = m.group(1)
        if key_type is None:
            return {"type": "object", "additionalProperties": {}}
        value_type = m.group(2).strip()
        additional_properties = python_type_to_json_type(value_type)
        return {"type": "object", "additionalProperties": additional_properties}

    # Unions written with '|'
    if "|" in type_str:
        parts = split_union_types(type_str)
        types = []
        for p in parts:
            t = python_type_to_json_type(p)
            types.append(t)

        if len(types) > 1:
            return {"anyOf": types}
        return {"type": types}

    # Base types
    t = _PY_TO_JSON_TYPE.get(type_str)
    if t is None:
        return {}
    return {"type": t}


_BASE_PY_TYPES: dict[str, Any] = {
    "str": str,
    "int": int,
    "float": float,
    "bool": bool,
    "list": list,
    "dict": dict,
    "tuple": tuple,
    "set": set,
    "bytes": bytes,
    "None": type(None),
    "Any": Any,
}


def _split_generic_args(args_str: str) -> list[str]:
    """Bracket-aware comma split for generic type arguments."""
    parts: list[str] = []
    depth = 0
    current: list[str] = []
    for ch in args_str:
        if ch == "[":
            depth += 1
            current.append(ch)
        elif ch == "]":
            depth -= 1
            current.append(ch)
        elif ch == "," and depth == 0:
            parts.append("".join(current).strip())
            current = []
        else:
            current.append(ch)
    if current:
        parts.append("".join(current).strip())
    return [p for p in parts if p]


def python_type_str_to_python_type(type_str: str) -> Any:
    """Convert a Python type annotation string to an actual Python type.

    Handles base types, list[X], dict[K,V], Optional[X], Union[X,Y], and X | Y.
    Falls back to str for unknown or unparseable type strings.
    """
    type_str = type_str.strip()

    if type_str in _BASE_PY_TYPES:
        return _BASE_PY_TYPES[type_str]

    # list[X] / List[X]
    m = re.fullmatch(r"(?:List|list)\[(.+)\]", type_str)
    if m:
        return list[python_type_str_to_python_type(m.group(1).strip())]

    # dict[K, V] / Dict[K, V]
    m = re.fullmatch(r"(?:Dict|dict)\[(.+)\]", type_str)
    if m:
        args = _split_generic_args(m.group(1))
        if len(args) == 2:
            return dict[
                python_type_str_to_python_type(args[0]),
                python_type_str_to_python_type(args[1]),
            ]
        return dict

    # Optional[X]
    m = re.fullmatch(r"Optional\[(.+)\]", type_str)
    if m:
        return python_type_str_to_python_type(m.group(1).strip()) | None

    # Union[X, Y, ...]
    m = re.fullmatch(r"Union\[(.+)\]", type_str)
    if m:
        args = _split_generic_args(m.group(1))
        types = [python_type_str_to_python_type(a) for a in args]
        if not types:
            return str
        result: Any = types[0]
        for t in types[1:]:
            result = result | t
        return result

    # X | Y (PEP 604)
    if "|" in type_str:
        parts = split_union_types(type_str)
        if len(parts) > 1:
            types = [python_type_str_to_python_type(p) for p in parts]
            result = types[0]
            for t in types[1:]:
                result = result | t
            return result

    return str


_DEFINITION_BLOCKS = ("$defs", "definitions")

# JSON values inlining may add (each object, array and scalar is one). Mutually referencing
# definitions grow factorially without it; Google's create_event adds about 110, and the
# cap keeps the worst case near 60 KB.
MAX_INLINED_SIZE = 3_000

# Keywords holding a subschema, a list or a map of them; anything else (`enum`, `default`…)
# is data, even when it looks like a schema.
_SUBSCHEMA_KEYWORDS = frozenset(
    {
        "items",
        "additionalItems",
        "additionalProperties",
        "unevaluatedItems",
        "unevaluatedProperties",
        "contains",
        "propertyNames",
        "not",
        "if",
        "then",
        "else",
    }
)
_SUBSCHEMA_LIST_KEYWORDS = frozenset({"allOf", "anyOf", "oneOf", "prefixItems"})
_SUBSCHEMA_MAP_KEYWORDS = frozenset(
    {"properties", "patternProperties", "dependentSchemas", *_DEFINITION_BLOCKS}
)

# Rejected by Anthropic and/or OpenAI at the top level of a tool's parameters.
_TOP_LEVEL_UNSUPPORTED = ("allOf", "anyOf", "oneOf", "not", "enum")


def _unescape_pointer_token(token: str) -> str:
    """Undo RFC 6901 escaping (``~1`` is ``/``, ``~0`` is ``~``)."""
    return token.replace("~1", "/").replace("~0", "~")


def _lookup(root: Any, ref: str) -> Any:
    """Resolve a local JSON pointer against ``root``; ``None`` if it does not resolve."""
    if not ref.startswith("#/"):
        return None

    node = root
    for raw_token in ref[2:].split("/"):
        token = _unescape_pointer_token(raw_token)
        if isinstance(node, dict):
            if token not in node:
                return None
            node = node[token]
        elif isinstance(node, list):
            try:
                node = node[int(token)]
            except (ValueError, IndexError):
                return None
        else:
            return None
    return node


def _size(node: Any) -> int:
    """The number of JSON values in ``node``, itself included."""
    count = 0
    pending = [node]
    while pending:
        current = pending.pop()
        count += 1
        if isinstance(current, dict):
            pending.extend(current.values())
        elif isinstance(current, list):
            pending.extend(current)
    return count


def inline_schema_refs(
    schema: Any,
    *,
    on_unresolved: Callable[[str], None] | None = None,
    on_truncated: Callable[[int], None] | None = None,
    max_size: int = MAX_INLINED_SIZE,
) -> Any:
    """Return a copy of ``schema`` with every local ``$ref`` replaced by its definition.

    Follows ``$defs`` and ``definitions``, lets keywords next to a ``$ref`` win over the
    definition's, and drops a ``$ref`` that does not resolve (``on_unresolved`` is called
    once per pointer). A recursive ref, or one past the ``max_size`` values inlining may
    add, becomes a placeholder; refs expand breadth first, so the deepest are cut, and
    ``on_truncated`` gets the count. The result shares nothing with the input.
    """
    body = schema
    if isinstance(schema, dict):
        body = {k: v for k, v in schema.items() if k not in _DEFINITION_BLOCKS}
    inliner = _Inliner(schema, body, on_unresolved, max_size)
    result = inliner.run()
    if inliner.truncated and on_truncated is not None:
        on_truncated(inliner.truncated)
    return result


class _Inliner:
    """One breadth-first pass copying a schema and expanding its refs within a budget."""

    def __init__(
        self,
        root: Any,
        body: Any,
        on_unresolved: Callable[[str], None] | None,
        budget: int,
    ) -> None:
        self._root = root
        self._body = body
        self._on_unresolved = on_unresolved
        self._budget = budget
        self._reported: set[str] = set()
        self._sizes: dict[str, int] = {}
        # (container, key) slot to fill, the source node, the refs open on its path.
        self._queue: deque[tuple[Any, Any, Any, frozenset[str]]] = deque()
        self.truncated = 0

    def run(self) -> Any:
        holder: list[Any] = [None]
        self._queue.append((holder, 0, self._body, frozenset()))
        while self._queue:
            container, key, node, open_refs = self._queue.popleft()
            container[key] = self._copy(node, open_refs)
        return holder[0]

    def _copy(self, node: Any, open_refs: frozenset[str]) -> Any:
        """Copy one node, queueing its children rather than descending."""
        if isinstance(node, list):
            items: list[Any] = [None] * len(node)
            for index, item in enumerate(node):
                self._queue.append((items, index, item, open_refs))
            return items
        if not isinstance(node, dict):
            return node
        if isinstance(node.get("$ref"), str):
            return self._expand(node, open_refs)
        return self._copy_entries({k: (v, open_refs) for k, v in node.items()})

    def _copy_entries(self, entries: dict[str, tuple[Any, frozenset[str]]]) -> dict[str, Any]:
        copy: dict[str, Any] = {}
        for key, (value, open_refs) in entries.items():
            copy[key] = None
            self._queue.append((copy, key, value, open_refs))
        return copy

    def _expand(self, node: dict[str, Any], open_refs: frozenset[str]) -> Any:
        # One layer per step down a chain of refs, each with the open refs of its children.
        layers = [({k: v for k, v in node.items() if k != "$ref"}, open_refs)]
        ref: Any = node["$ref"]
        while isinstance(ref, str):
            target = self._resolve_pointer(ref)
            if target is None:
                self._report(ref)
                break
            if ref in open_refs or not self._afford(ref, target):
                layers.append((_placeholder(ref, target), open_refs))
                break
            open_refs = open_refs | {ref}
            if not isinstance(target, dict):
                return self._copy(target, open_refs)
            layers.append(({k: v for k, v in target.items() if k != "$ref"}, open_refs))
            ref = target.get("$ref")

        # Deepest first for the key order; shallower layers then overwrite its values.
        entries: dict[str, tuple[Any, frozenset[str]]] = {}
        for keywords, layer_refs in reversed(layers):
            for key, value in keywords.items():
                entries[key] = (value, layer_refs)
        return self._copy_entries(entries)

    def _resolve_pointer(self, ref: str) -> Any:
        return self._body if ref == "#" else _lookup(self._root, ref)

    def _afford(self, ref: str, target: Any) -> bool:
        if ref not in self._sizes:
            self._sizes[ref] = _size(target)
        if self._sizes[ref] > self._budget:
            self.truncated += 1
            return False
        self._budget -= self._sizes[ref]
        return True

    def _report(self, ref: str) -> None:
        if ref not in self._reported:
            self._reported.add(ref)
            if self._on_unresolved is not None:
                self._on_unresolved(ref)


def _placeholder(ref: str, target: Any) -> dict[str, Any]:
    """What stands in for a definition that is not expanded: its type and name."""
    declared = target.get("type") if isinstance(target, dict) else None
    return {
        "type": declared if isinstance(declared, (str, list)) else "object",
        "description": ref.rsplit("/", 1)[-1],
    }


def as_tool_parameters(schema: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of ``schema`` that hosted providers accept as a tool's parameters.

    Arrays get ``items: {}`` (Chat Completions validators require it), the top-level
    combinators providers reject and ``$schema`` are dropped, and ``type``,
    ``properties`` and ``required`` are filled in.
    """
    parameters: dict[str, Any] = _with_array_items(schema)
    for keyword in (*_TOP_LEVEL_UNSUPPORTED, "$schema"):
        parameters.pop(keyword, None)
    parameters.setdefault("type", "object")
    parameters.setdefault("properties", {})
    parameters.setdefault("required", [])
    return parameters


def _with_array_items(schema: Any) -> Any:
    """Copy ``schema``, giving every array schema in it an ``items``."""
    if not isinstance(schema, dict):
        return deepcopy(schema)

    copy: dict[str, Any] = {}
    for key, value in schema.items():
        if key in _SUBSCHEMA_KEYWORDS:
            # `items` may be a list of schemas (draft-07 tuples).
            copy[key] = (
                [_with_array_items(v) for v in value]
                if isinstance(value, list)
                else _with_array_items(value)
            )
        elif key in _SUBSCHEMA_LIST_KEYWORDS and isinstance(value, list):
            copy[key] = [_with_array_items(v) for v in value]
        elif key in _SUBSCHEMA_MAP_KEYWORDS and isinstance(value, dict):
            copy[key] = {name: _with_array_items(v) for name, v in value.items()}
        else:
            copy[key] = deepcopy(value)

    declared = schema.get("type")
    if (declared == "array" or (isinstance(declared, list) and "array" in declared)) and (
        "items" not in copy
    ):
        copy["items"] = {}
    return copy
