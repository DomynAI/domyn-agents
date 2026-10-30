"""Visibility scope governing state access and run_config (PII) propagation.

``VisibilityScope`` is a progressive trust ladder shared by any subsystem that
must decide how much of the caller's ``run_config`` (which carries PII such as
``user_id``, ``space_id`` and auth tokens) to expose to a downstream consumer.

It is used by:
- MCP tools (:class:`domyn.agents.tools.mcp.mcp_tool.MCPTool`) to gate the ``_meta``
  they send to a remote MCP server, and to scope shared state.
- Delegated agents (:class:`domyn.agents.agents.delegate_agent.DelegateAgent`) to
  gate the ``run_config`` forwarded to an external runner.

The enum owns only the ``run_config`` gate. State-scoping semantics (per-tool vs
per-group vs global) are consumer-specific and interpreted where they matter.
"""

from enum import Enum
from typing import Any


class VisibilityScope(Enum):
    """Progressive trust levels governing state scope and run_config (PII) access.

    - ISOLATED: state per consumer (e.g. tool_name key), no run_config.
    - SCOPED: state per group (e.g. MCP server URL), no run_config. Consumers in
      the same group share state.
    - TRUSTED: state per group + run_config (PII).
    - GLOBAL: full shared state + run_config (PII).
    """

    ISOLATED = "isolated"
    SCOPED = "scoped"
    TRUSTED = "trusted"
    GLOBAL = "global"

    @property
    def grants_run_config(self) -> bool:
        """Whether this scope is allowed to receive run_config (PII)."""
        return self in (VisibilityScope.TRUSTED, VisibilityScope.GLOBAL)

    def filter_run_config(self, run_config: dict[str, Any] | None) -> dict[str, Any]:
        """Return a copy of ``run_config`` when this scope grants PII access, else ``{}``."""
        return dict(run_config or {}) if self.grants_run_config else {}
