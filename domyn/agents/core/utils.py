import asyncio
import inspect
import json
import re
from collections.abc import AsyncGenerator, Callable
from importlib import machinery, util
from typing import TYPE_CHECKING, Any, Self

from pydantic import BaseModel, model_validator

from domyn.agents.core import GuardrailResult, Part

if TYPE_CHECKING:
    from domyn.agents.core.interfaces import BaseAgent


def load_function_from_file(name: str, path: str) -> Callable:

    spec = util.spec_from_file_location(name, path, loader=machinery.SourceFileLoader(name, path))
    _module = util.module_from_spec(spec)
    spec.loader.exec_module(_module)
    _module = getattr(_module, name)
    if not _module:
        raise ValueError(f"Function {name} not found in {path}")
    return _module


def validate_parameter_keys(fn: Callable, mandatory_keys: set[str]) -> None:
    sig = inspect.signature(fn)
    param_keys = set(sig.parameters.keys())
    if not mandatory_keys.issubset(param_keys):
        missing_keys = mandatory_keys - param_keys
        raise ValueError(f"Function {fn.__name__} is missing expected parameters: {missing_keys}.")


async def execute_async(fn: Callable, **kwargs):
    if not inspect.iscoroutinefunction(fn):
        context = await asyncio.to_thread(fn, **kwargs)
    else:
        context = await fn(**kwargs)
    return context


def wrap_as_async_generator(fn: Callable) -> Callable[..., AsyncGenerator]:
    if inspect.isasyncgenfunction(fn):
        return fn
    elif inspect.iscoroutinefunction(fn):

        async def inner(*args, **kwargs):
            yield await fn(*args, **kwargs)

        inner.__name__ = getattr(fn, "__name__", "inner")
        return inner
    else:

        async def inner(*args, **kwargs):
            yield await asyncio.to_thread(fn, *args, **kwargs)

        inner.__name__ = getattr(fn, "__name__", "inner")
        return inner


def serialize_callable(value: Any) -> dict[str, Any]:
    if hasattr(value, "__module__") and hasattr(value, "__name__"):
        return {
            "type": "function",
            "module": value.__module__,
            "name": value.__name__,
            "qualname": getattr(value, "__qualname__", value.__name__),
            "doc": value.__doc__,
            "path": inspect.getfile(value) if inspect.isfunction(value) else None,
        }
    elif hasattr(value, "__class__"):
        return {
            "type": "callable_object",
            "class": value.__class__.__name__,
            "module": value.__class__.__module__,
            "repr": repr(value),
            "doc": getattr(value, "__doc__", None),
            "path": inspect.getfile(value) if inspect.isfunction(value) else None,
        }
    else:
        return {"type": "unknown_callable", "repr": repr(value), "str": str(value)}


SerializedCallable = dict[str, Any]


def extract_parts(
    text: str,
    html_tag_re: re.Pattern[str] = re.compile(r"<html>(.*?)</html>", re.DOTALL | re.IGNORECASE),
) -> list[Part]:
    parts: list[Part] = []
    cursor = 0
    for match in html_tag_re.finditer(text):
        if match.start() > cursor:
            segment = text[cursor : match.start()].strip()
            if segment:
                parts.append(Part(text=segment))
        parts.append(Part(html=match.group(0).strip()))
        cursor = match.end()
    if cursor < len(text):
        segment = text[cursor:].strip()
        if segment:
            parts.append(Part(text=segment))
    return parts if parts else [Part(text=text)]


def dict_to_part(v: dict) -> Part:
    """Read a dict emitted by a tool or a hook as a Part: a guardrail result, a Part's own
    fields, or else the dict itself as metadata."""
    try:
        return Part(guardrail_result=GuardrailResult(**v))
    except Exception:
        pass
    try:
        part = Part(**v)
        if any(val is not None for val in part.model_dump().values()):
            return part
    except Exception:
        pass
    return Part(metadata=v)


def parts_to_text(parts: list[Part] | None) -> str:
    """Concatenate the renderable fields of each Part, in list order.

    For every part, the present values among ``text`` and ``html`` are
    concatenated, followed by ``rich_text`` serialized with ``json.dumps``.
    Parts are processed in order and all chunks are joined with a single space.
    Empty/None fields are skipped.
    """
    if not parts:
        return ""
    chunks: list[str] = []
    for part in parts:
        for value in (part.text, part.html):
            if value:
                chunks.append(value)
        if part.rich_text:
            chunks.append(json.dumps(part.rich_text, ensure_ascii=False))
    return " ".join(chunks)


# A snapshot of an agent tree: its agents, what each can call and hand off to, and every
# tool's declaration. Hosts store it alongside a configuration they run, so its fields
# are a stored format: change them with care.


class SystemToolDescription(BaseModel):
    name: str
    description: str
    parameters: dict[str, Any] | None

    def __str__(self) -> str:
        return f"---Tool {self.name}---\n description: {self.description}\n parameters:{self.parameters}\n---\n"


class SystemAgentDescription(BaseModel):
    name: str
    description: str
    tools: list[str]
    subagents: list[str]

    def __str__(self) -> str:
        return f"---Agent {self.name}---\n description: {self.description}\n tools:{self.tools}\n subagents:{self.subagents}\n---\n"


class SystemDescription(BaseModel):
    agents: dict[str, SystemAgentDescription]
    tools: dict[str, SystemToolDescription]

    @model_validator(mode="after")
    def check_maps(self) -> Self:
        for agent in self.agents:
            for subagent_name in self.agents[agent].subagents:
                if subagent_name not in self.agents:
                    raise ValueError(
                        f"Subagent '{subagent_name}' used by agent '{self.agents[agent].name}' is not defined in the system agents."
                    )
            for tool in self.agents[agent].tools:
                if tool not in self.tools:
                    name_match = any(tool == t.name for t in self.tools.values())
                    if not name_match:
                        raise ValueError(
                            f"Tool '{tool}' used by agent '{self.agents[agent].name}' is not defined in the system tools."
                        )
        return self

    def __str__(self) -> str:
        agents_str = "Available agents:\n" + "\n".join([str(self.agents[a]) for a in self.agents])
        tools_str = "Available tools:\n" + "\n".join([str(self.tools[t]) for t in self.tools])
        return agents_str + "\n" + tools_str


def _describe_tool(tool: Any) -> SystemToolDescription:
    return SystemToolDescription(
        name=tool.declaration.name,
        description=tool.declaration.description,
        parameters={
            p.name: {"description": p.description, "type": p.type}
            for p in tool.declaration.parameters
        },
    )


# TODO: tools are keyed by name, so same-named tools on different agents overwrite each other.
def get_system_description_from_root(root_agent: "BaseAgent") -> SystemDescription:
    """Describe the agent tree under ``root_agent``: every agent reached through
    ``sub_agents``, once each, and every tool any of them declares."""
    tools_descriptions = {t.declaration.name: _describe_tool(t) for t in root_agent.tools}
    agents_descriptions = {}
    to_check = [root_agent]
    checked: list[Any] = []
    while to_check:
        agent = to_check.pop(-1)
        if agent in checked:
            continue
        to_check.extend(agent.sub_agents)
        checked.append(agent)

        agents_descriptions[agent.name] = SystemAgentDescription(
            name=agent.name,
            description=agent.description,
            tools=[t.declaration.name for t in agent.tools],
            subagents=[a.name for a in agent.sub_agents],
        )
        for tool in agent.tools:
            tools_descriptions[tool.declaration.name] = _describe_tool(tool)

    return SystemDescription(tools=tools_descriptions, agents=agents_descriptions)
