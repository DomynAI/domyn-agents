"""Base classes for tools in the domyn agents framework."""

import inspect
import re
import types
from collections.abc import Callable
from dataclasses import dataclass, field, fields
from functools import reduce
from operator import or_
from typing import (  # type: ignore[attr-defined]
    Annotated,
    Any,
    ClassVar,
    Literal,
    Self,
    _AnnotatedAlias,
)

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    ValidationError,
    field_validator,
    model_serializer,
)
from pydantic.dataclasses import dataclass as pydantic_dataclass

from domyn.agents.core import Part
from domyn.agents.core.utils import dict_to_part

_TOOL_NAME_SANITIZE_RE: re.Pattern[str] = re.compile(r"[^a-zA-Z0-9_-]")


def sanitize_tool_name(name: str) -> str:
    """Normalize a tool/agent name to the charset accepted by strict LLM providers.

    Replaces every character outside ``[a-zA-Z0-9_-]`` with an underscore.
    Applied ONLY at the provider boundary (request building) — the canonical
    name everywhere else (registry, execution events, frontend) stays the
    original. Idempotent: re-applying it on an already-sanitized name is a
    no-op, so replaying tool calls from conversation history is safe.

    Empty / non-str names are rejected so the error points back at the source
    field. Upper length is intentionally not enforced: both Anthropic and
    OpenAI accept up to 128 chars (verified empirically against the OpenAI
    chat-completions API — the docs' 64 figure is outdated, and the 128 limit
    holds in both strict and non-strict mode). Truncating client-side would
    silently drop information, so overlong names are left for the server.
    """
    if not isinstance(name, str):
        raise ValueError(f"tool/agent name must be str, got {type(name).__name__}")
    if not name:
        raise ValueError("tool/agent name cannot be empty")
    return _TOOL_NAME_SANITIZE_RE.sub("_", name)


def build_sanitized_name_map(
    tools: list[dict[str, Any]] | None, logger: Any = None
) -> dict[str, str]:
    """Map each tool's sanitized name back to its original, built per-request.

    Providers sanitize tool names on the way out (request payload) and use this
    map to restore the original when the model echoes the name back in a tool
    call, so the runner keeps dispatching on the canonical (original) name.

    Distinct originals collapsing to the same sanitized id make the reverse
    mapping ambiguous; warn (not raise — a request-time raise would break a
    live call) so the offending pair surfaces in logs.
    """
    mapping: dict[str, str] = {}
    for tool in tools or []:
        original = tool.get("function", {}).get("name", "")
        if not original:
            continue
        key = sanitize_tool_name(original)
        if key in mapping and mapping[key] != original and logger is not None:
            logger.warning(
                f"Tool name collision after sanitization: {mapping[key]!r} and {original!r} "
                f"both map to {key!r}; the reverse mapping back to the original will be ambiguous."
            )
        mapping[key] = original
    return mapping


def _sanitize_function_tool_defs(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return a copy of chat-completions tool defs with ``function.name`` sanitized.

    Strict tool-calling validates names against the provider charset
    (``^[a-zA-Z0-9_-]{1,128}$``). Sanitize at the provider boundary; the
    original is restored from the response via ``build_sanitized_name_map``.
    """
    sanitized = []
    for tool in tools:
        func = tool.get("function", {})
        raw_name = func.get("name", "")
        sanitized.append(
            {
                **tool,
                "function": {
                    **func,
                    "name": sanitize_tool_name(raw_name) if raw_name else raw_name,
                },
            }
        )
    return sanitized


def _sanitize_message_tool_call_names(messages: list[dict[str, Any]]) -> None:
    """Sanitize replayed ``tool_calls`` names in serialized messages in place.

    Replayed assistant ``tool_calls`` must reference the same (sanitized)
    names sent in the tool definitions. ``sanitize_tool_name`` is idempotent,
    so already-valid names are untouched.
    """
    for msg in messages:
        for tc in msg.get("tool_calls") or []:
            func = tc.get("function")
            if func and func.get("name"):
                func["name"] = sanitize_tool_name(func["name"])


def sanitize_tool_choice(tool_choice: Any) -> Any:
    """Sanitize the tool name carried by a forced ``tool_choice``, if any.

    A dict ``tool_choice`` that names a specific tool must use the same
    sanitized name as the tool definitions in the payload, otherwise the
    forced choice won't match (and violates the provider charset). Handles
    both OpenAI (``{"type":"function","function":{"name":...}}``) and Anthropic
    (``{"type":"tool","name":...}``) shapes. String forms (``"auto"``/``"any"``/
    ``"required"``/``"none"``/``"tool"``) and dicts without a name pass through
    unchanged. Non-mutating: returns a shallow copy.
    """
    if not isinstance(tool_choice, dict):
        return tool_choice
    func = tool_choice.get("function")
    if isinstance(func, dict) and func.get("name"):
        return {**tool_choice, "function": {**func, "name": sanitize_tool_name(func["name"])}}
    if tool_choice.get("name"):
        return {**tool_choice, "name": sanitize_tool_name(tool_choice["name"])}
    return tool_choice


def sanitize_tool_names(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    tool_choice: Any,
    logger: Any = None,
) -> tuple[list[dict[str, Any]] | None, Any, dict[str, str]]:
    """Sanitize every tool name across a chat-completions request, in one call.

    Single boundary entry point shared by all chat-completions providers
    (openai, open_router, cerebras, vllm). Anthropic uses a different wire
    shape and sanitizes inline.

    - mutates ``messages`` in place (replayed assistant ``tool_calls``)
    - returns a sanitized copy of ``tools`` and ``tool_choice``
    - returns the reverse map (sanitized → original) so the caller restores the
      canonical name from the response

    Idempotent: already-valid names are untouched, so a provider whose backend
    doesn't enforce the charset pays nothing for being routed through here.
    """
    _sanitize_message_tool_call_names(messages)
    name_map = build_sanitized_name_map(tools, logger=logger)
    if tools:
        tools = _sanitize_function_tool_defs(tools)
    if tool_choice is not None:
        tool_choice = sanitize_tool_choice(tool_choice)
    return tools, tool_choice, name_map


# Parameters the runtime injects, never shown to the model and never part of a
# tool's schema. A tool declares one to receive it; tools that do not are called
# without it.
FUNCTION_RESERVED_PARAMS = {
    "self",
    "cls",
    "run_config",
    "state",
    "conversation_id",
    # Per-call metadata always passed through (REST headers / MCP _meta),
    # unlike run_config it is never gated by VisibilityScope.
    "call_metadata",
    # The tool results the calling agent sees, as its model is shown them: what
    # `read_tool_result` reads a cut or dropped result back from.
    "tool_results",
}


# sentinel to express missing default in Param
class _Missing:
    _instance: "_Missing | None" = None

    def __new__(cls) -> "_Missing":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "<MISSING>"

    def __bool__(self) -> bool:
        return False


_MISSING = _Missing()


@pydantic_dataclass
class Param:
    type: str
    name: str | None = None
    description: str | None = None
    is_required: bool = True
    default: Any = _MISSING

    @model_serializer(mode="plain")
    def _serialize(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for f in fields(self):
            value = getattr(self, f.name)
            if f.name == "default" and isinstance(value, _Missing):
                continue  # omit default if it's the sentinel _MISSING
            out[f.name] = value
        return out


@pydantic_dataclass
class ConfigParam(Param):
    """Configuration parameter definition."""

    type: str | None = None
    value: Any = None
    is_secret: bool = False


class BaseTool(BaseModel):
    """Base class for all tools in the domyn agents framework."""

    name_string: ClassVar[str] = "name"
    parameters_string: ClassVar[str] = "arguments"
    # register subclasses
    _registry: ClassVar[dict[str, type["BaseTool"]]] = {}
    type: Literal["BaseTool"] = "BaseTool"
    group: str | None = Field(
        default=None, description="Group name for the tool (used for categorization)"
    )
    metadata: dict[str, Any] | None = Field(
        default_factory=dict, description="Additional metadata for the tool"
    )
    uid: str | None = Field(default=None, description="Optional unique identifier for the tool")
    timeout_s: float | None = Field(
        default=None,
        description="Seconds this tool may run before the turn gives up on it. None takes the "
        "runner's ceiling, 0 removes it. On a streaming tool it bounds the wait for each "
        "chunk, not the whole call.",
    )

    # auto-register sub types
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        tag = getattr(cls, "type", None)
        if tag is not None:
            BaseTool._registry[tag] = cls

    @classmethod
    def from_config(cls, config: dict[str, Any], save_in_registry: bool = False) -> "BaseTool":
        """
        Create a tool instance from configuration dictionary.

        This method should be overridden by subclasses to provide
        their specific configuration logic.
        """
        raise NotImplementedError(f"from_config not implemented for {cls.__name__}")

    @classmethod
    def read_from_registry_and_configure(cls, config: dict[str, Any]) -> "BaseTool":
        raise NotImplementedError(
            f"read_from_registry_and_configure not implemented for {cls.__name__}"
        )

    def required_param_names(self) -> set[str]:
        """Return the names of parameters this tool's schema marks as required."""
        raise NotImplementedError(f"required_param_names not implemented for {type(self).__name__}")


class BaseToolAdapter:
    """Custom adapter for BaseTool that uses from_config for dictionary inputs."""

    def __init__(self) -> None:
        # Create a union type of all registered tool types
        tool_types = reduce(or_, (BaseTool, *BaseTool._registry.values()))
        self._adapter: TypeAdapter[Any] = TypeAdapter(
            Annotated[tool_types, Field(discriminator="type")]
        )

    def validate_python(
        self,
        obj: Any,
        *,
        strict: bool | None = None,
        from_attributes: bool | None = None,
        context: dict[str, Any] | None = None,
        save_in_registry: bool = False,
    ) -> "BaseTool":
        """Validate Python object, using from_config for dictionaries."""
        # Case 1: Already a BaseTool or subclass - return as-is
        if isinstance(obj, BaseTool):
            return obj

        # Case 2: Dictionary - determine tool type and use its from_config method
        if isinstance(obj, dict):
            tool_type = obj.get("type")
            tool_reference = obj.get("reference")

            # Handle reference logic
            if not tool_type and not tool_reference:
                raise ValueError("Tool type or reference must be specified in the config")

            if tool_reference:
                from domyn.agents.registry import get_tool_registry

                referenced_tool = get_tool_registry().get_tool(tool_reference)
                if referenced_tool is None:
                    raise ValueError(f"Referenced tool '{tool_reference}' not found in registry")
                else:
                    tool_type = referenced_tool.type

            if tool_type is None:
                raise ValueError("Tool type cannot be None")

            tool_class = BaseTool._registry.get(tool_type)
            if tool_class is None:
                raise ValueError(f"Unknown tool type: {tool_type}")

            if tool_reference:
                return tool_class.read_from_registry_and_configure(obj)
            # Do not register a tool reconfigured from registry

            # Use the specific tool's from_config method in case of custom logic
            tool = tool_class.from_config(obj, save_in_registry=False)

            if save_in_registry:
                from domyn.agents.registry import get_tool_registry

                get_tool_registry().register_tool(tool)

            return tool

        # Case 3: Other types - use standard TypeAdapter validation
        result = self._adapter.validate_python(
            obj, strict=strict, from_attributes=from_attributes, context=context
        )
        if not isinstance(result, BaseTool):
            raise ValueError(f"Expected BaseTool, got {type(result)}")
        return result

    def from_config(self, config: dict[str, Any], *, save_in_registry: bool = False) -> "BaseTool":
        """Create tool from config with optional registry saving."""
        return self.validate_python(config, save_in_registry=save_in_registry)

    def validate_json(
        self, json_str: str, *, strict: bool | None = None, context: dict[str, Any] | None = None
    ) -> "BaseTool":
        result = self._adapter.validate_json(json_str, strict=strict, context=context)
        if not isinstance(result, BaseTool):
            raise ValueError(f"Expected BaseTool, got {type(result)}")
        return result

    def model_json_schema(self, *, mode: str = "validation") -> dict[str, Any]:
        schema = self._adapter.json_schema(mode=mode)
        if not isinstance(schema, dict):
            raise ValueError(f"Expected dict schema, got {type(schema)}")
        return schema


Tool = BaseToolAdapter()


@dataclass
class FunctionDeclaration:
    """Function declaration structure."""

    name: str
    description: str
    parameters: list[Param]
    returns: Param | None = None

    @classmethod
    def from_callable(
        cls, callable_func: Callable, ignore_params: list[str]
    ) -> "FunctionDeclaration":
        """Create a FunctionDeclaration from a Python callable."""
        name = callable_func.__name__

        # extract signature
        sig = inspect.signature(callable_func)

        # check napoleon docstring
        parse_napoleon = _parse_napoleon_docstring(callable_func.__doc__ or "")

        if parse_napoleon:
            # Defaults aren't expressible in Google-style docstrings — read them from
            # the signature and merge so napoleon-driven Params still carry is_required/default.
            signature_defaults = {
                p.name: p.default
                for p in sig.parameters.values()
                if p.default is not inspect.Parameter.empty
            }

            # Types come from the signature when the docstring does not state
            # one. Google style makes the type optional — `path: what it is` is
            # as idiomatic as `path (str): what it is` — and taking the
            # docstring as the only source meant `def f(path: str)` produced a
            # schema with no type at all. The model was then told a parameter's
            # name and description and nothing else, and duly sent `"30"` for an
            # integer. The signature is authoritative about types; the docstring
            # about meaning. A docstring that *does* declare a type still wins,
            # so nothing that used to work changes.
            signature_types = {
                param.name: param.type for param in _create_params_from_signature(sig)
            }

            list_parameters = []
            for arg in parse_napoleon["args"]:
                param = Param(**arg)
                if param.type in (None, "", "any"):
                    param.type = signature_types.get(param.name, param.type)
                if param.name in signature_defaults:
                    param.is_required = False
                    param.default = signature_defaults[param.name]
                if param.name not in ignore_params:
                    list_parameters.append(param)

            # check if there is a mismatch
            joined_params = (
                {c.name for c in _create_params_from_signature(sig)} - set(ignore_params)
            ).difference({param.name for param in list_parameters})
            if joined_params:
                raise ValueError("There is mismatch between docstring and signature of function.")
            return cls(
                name=name,
                description=parse_napoleon["description"],
                parameters=list_parameters,
                returns=Param(**parse_napoleon["returns"]) if parse_napoleon["returns"] else None,
            )
        else:
            description = callable_func.__doc__ or f"Function {name}"

            # Parse function signature
            parameters = list(
                filter(lambda x: x.name not in ignore_params, _create_params_from_signature(sig))
            )

            return cls(name=name, description=description, parameters=parameters)


class ToolResult(BaseModel):
    """Result of a tool execution."""

    model_config = ConfigDict(extra="forbid")

    success: bool
    data: Any = None
    error: str | None = None
    error_code: str | None = None
    state: dict | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    emitted_content: list[Part] | dict | str | None = []

    @classmethod
    def from_result_dict(
        cls,
        value: Any,
        *,
        result_key_fallback: bool = False,
        default_state: dict | None = None,
    ) -> Self | None:
        """Cast a dict into a ToolResult, recognizing two shapes in order.

        1. A *ToolResult-shaped* dict — validated by the model itself
           (``success`` required, extra keys forbidden) — is cast directly.
        2. If ``result_key_fallback`` is set, a legacy dict carrying a
           ``"result"`` key has that value lifted into ``data`` (with optional
           ``metadata``/``state``); ``default_state`` is used when ``"state"``
           is absent.

        Returns ``None`` when neither shape matches, so callers can fall back to
        their own wrapping logic.
        """
        try:
            return cls(**value)
        except (ValidationError, TypeError):
            pass

        if result_key_fallback and isinstance(value, dict) and "result" in value:
            return cls(
                success=True,
                data=value["result"],
                metadata=value.get("metadata", {}),
                state=value.get("state", default_state),
            )
        return None

    @field_validator("emitted_content", mode="before")
    @classmethod
    def process_emitted_content(cls, v: Any) -> list[Part]:
        if not v:
            return []
        if isinstance(v, dict):
            return [dict_to_part(v)]
        if isinstance(v, list):
            if all(isinstance(item, Part) for item in v):
                return list(v)
            if all(isinstance(item, dict) for item in v):
                return [dict_to_part(item) for item in v]
            return [Part(text=str(v))]
        return [Part(text=str(v))]


class UpdateToolResult(BaseModel):
    """Update of a tool execution."""

    data: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)


def __get_param_type(param_type):
    if isinstance(param_type, types.UnionType):
        return str(param_type)
    return param_type.__name__


def _create_params_from_signature(sig: inspect.Signature) -> list[Param]:
    params = []
    for param_name, param in sig.parameters.items():
        if param_name in FUNCTION_RESERVED_PARAMS:
            continue

        param_type = param.annotation if param.annotation != inspect.Parameter.empty else None
        if isinstance(param_type, _AnnotatedAlias):
            param_type = param_type.__origin__

        has_default = param.default is not inspect.Parameter.empty
        params.append(
            Param(
                type=__get_param_type(param_type) if param_type else "any",
                name=param_name,
                description=param.annotation.__metadata__[0]
                if isinstance(param.annotation, _AnnotatedAlias)
                else f"Parameter {param_name}",
                is_required=not has_default,
                default=param.default if has_default else _MISSING,
            )
        )
    return params


def _parse_napoleon_docstring(docstring: str) -> dict[str, Any] | None:
    """Parse a Napoleon/Google-style docstring.

    Returns a dict with keys:
      - description: str
      - args: list[{"name", "type", "description"}]
      - returns: {"type", "description"} or None
      - raises: list[{"type", "description"}]
    Returns None when no Args/Arguments/Parameters/Returns/Raises section is found.
    """
    if not docstring or not docstring.strip():
        return None

    # Strip each line (relative indentation is not preserved)
    text = "\n".join([d.strip() for d in docstring.split("\n")])
    lines = text.splitlines()

    # Recognized section headers (Args, Arguments, Parameters, Returns, Raises)
    header_re = re.compile(r"^(Args|Arguments|Parameters|Returns|Raises):\s*$", re.IGNORECASE)
    sections = {"description": []}
    current = "description"
    for line in lines:
        m = header_re.match(line)
        if m:
            key = m.group(1).lower()
            if key in ("arguments", "parameters"):
                key = "args"
            sections[key] = []
            current = key
        else:
            sections.setdefault(current, []).append(line)

    # Without a known header it is not a Napoleon docstring, as far as this parser goes
    recognized_headers = set(sections.keys()) - {"description"}
    if not recognized_headers:
        return None

    result: dict[str, Any] = {
        "description": "\n".join(
            [desc.rstrip() for desc in sections.get("description", [])]
        ).strip(),
        "args": [],
        "returns": None,
        "raises": [],
    }

    # --- parse Args (Google-style) ---

    def fix_args(typ: str) -> str:
        if not typ:
            return "any"
        typ = typ.strip()
        # Only replace commas that are not inside brackets (e.g. "str, optional"
        # but NOT "list[dict[str, str]]")
        result_chars = []
        depth = 0
        for ch in typ:
            if ch in "([":
                depth += 1
                result_chars.append(ch)
            elif ch in ")]":
                depth -= 1
                result_chars.append(ch)
            elif ch == "," and depth == 0:
                result_chars.append("|")
            else:
                result_chars.append(ch)
        typ = "".join(result_chars)
        typ = typ.replace("optional", "None")
        typ = re.sub(r"\s{2,}", " ", typ)  # normalize multiple whitespaces
        return " | ".join(map(str.strip, typ.split("|"))) if "|" in typ else typ

    if "args" in sections:
        body = "\n".join(sections["args"])
        param_pat = re.compile(
            r"^\s*(?P<name>[A-Za-z_]\w*)\s*(?:\((?P<type>[^)]+)\))?\s*:\s*(?P<desc>.*?)(?=(?:\n^\s*[A-Za-z_]\w*\s*(?:\([^)]+\))?\s*:)|\Z)",
            re.MULTILINE | re.DOTALL,
        )
        for m in param_pat.finditer(body):
            name = m.group("name")
            typ = fix_args((m.group("type") or "").strip())
            desc = m.group("desc").strip().replace("\n", " ").strip()
            result["args"].append({"name": name, "type": typ or "any", "description": desc})

    # --- parse Returns ---
    if "returns" in sections:
        body = "\n".join(sections["returns"]).strip()
        if body:
            m = re.match(r"^\s*([^\:]+?)\s*:\s*(.+)$", body, re.DOTALL)
            if m:
                typ = m.group(1).strip()
                desc = m.group(2).strip().replace("\n", " ").strip()
                result["returns"] = {"type": typ, "description": desc}
            else:
                result["returns"] = {"type": "", "description": body.replace("\n", " ").strip()}

    # --- parse Raises ---
    if "raises" in sections:
        body = "\n".join(sections["raises"])
        exc_pat = re.compile(
            r"^\s*(?P<type>[\w\.]+)\s*:\s*(?P<desc>.*?)(?=(?:\n^\s*[\w\.]+\s*:)|\Z)",
            re.MULTILINE | re.DOTALL,
        )
        for m in exc_pat.finditer(body):
            typ = m.group("type").strip()
            desc = m.group("desc").strip().replace("\n", " ").strip()
            result["raises"].append({"type": typ, "description": desc})

    return result
