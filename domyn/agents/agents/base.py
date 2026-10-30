"""Base classes for agents in the domyn agents framework."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from domyn.agents.logger import get_logger


@dataclass
class FunctionCall:
    """Function call representation."""

    name: str
    args: dict[str, Any] = field(default_factory=dict)


@dataclass
class Content:
    role: str  # "user", "assistant", "system", "tool"
    parts: list[str | FunctionCall] = field(default_factory=list)


class ContextVisibility(Enum):
    """Context/history visibility levels for agent flows."""

    GLOBAL = "global"
    PEERS = "peers"
    STANDARD = "standard"


class HandoffScope(Enum):
    """Handoff/EXECUTE scope: which agents an agent may invoke via handoff.

    Orthogonal to ``ContextVisibility`` (READ scope). While the callable set
    defaults to the agent's explicit ``sub_agents`` list, DIRECT_PEERS/GLOBAL resolve
    it against the live agent tree at runtime — letting framework-provided agents
    reach user-defined agents unknown at definition time.
    """

    EXPLICIT = "explicit"
    DIRECT_PEERS = "direct_peers"
    GLOBAL = "global"


def normalize_legacy_visibility(config: dict[str, Any]) -> dict[str, Any]:
    """Map the pre-split ``handoff_visibility`` key from stored legacy configs to
    ``context_visibility`` + ``handoff_scope``, preserving the coupled behaviour
    (``standard`` -> ``explicit`` scope). Explicit new keys win.
    """
    legacy = config.get("handoff_visibility")
    if legacy is None:
        return config

    value = legacy.value if isinstance(legacy, ContextVisibility) else str(legacy).lower()
    config = dict(config)
    del config["handoff_visibility"]
    config.setdefault("context_visibility", value)
    config.setdefault(
        "handoff_scope",
        {"standard": "explicit", "peers": "direct_peers"}.get(value, value),
    )

    get_logger().warning(
        f"Deprecated 'handoff_visibility' in agent config {config.get('name')!r}: "
        "replace it with 'context_visibility' (READ) and 'handoff_scope' (EXECUTE)."
    )
    return config


__all__ = [
    "Content",
    "ContextVisibility",
    "FunctionCall",
    "HandoffScope",
    "normalize_legacy_visibility",
]
