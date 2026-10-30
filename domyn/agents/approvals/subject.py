"""Turning a call into the thing that gets approved.

A tool declares which of its arguments carry the risk, through
``FunctionTool.approval_subject``:

* ``["command"]`` — build the subject from that argument
* ``lambda **kw: ...`` — full control, for tools whose risk is a derived value
  (the *host* of a URL rather than the URL, the *parent directory* of a path)
* unset — fall back to every argument

The fallback is safe and nearly useless: a subject containing every argument
almost never matches a second time, so the user is asked every time. That is the
right default. A tool that wants standing approvals has to say what it is
willing to be approved *for*, and having to say it out loud is a feature.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from domyn.agents.logger import get_logger

if TYPE_CHECKING:
    from domyn.agents.approvals.models import ApprovalRequest

# Parameters the planner adds to a tool's schema for its own purposes, which are
# no part of what the user is approving. They must never reach a subject, for two
# independent reasons:
#
# * The value differs on every call — ``reasoning`` is the model's thought about
#   this invocation — so a subject containing one can never match twice, and a
#   remembered approval would be dead on arrival.
# * The gate and the approval see different parameter sets. The gate builds its
#   subject from the raw action, while ``_build_tool_needs_feedback`` strips
#   these before showing the call to the user, so the approved subject comes back
#   without them. Filtering in both directions is what makes the two agree.
#
# Kept separate from ``FUNCTION_RESERVED_PARAMS`` (``run_config``, ``state``, …):
# those are injected by the runtime at call time and never appear in an action's
# parameters at all, whereas these are declared in the schema the model answers.
INJECTED_PARAMS: frozenset[str] = frozenset({"reasoning"})


def visible_parameters(parameters: dict[str, Any]) -> dict[str, Any]:
    """The parameters a subject may be built from: the call minus planner noise."""
    return {key: value for key, value in parameters.items() if key not in INJECTED_PARAMS}


def compute_subject(tool: Any, parameters: dict[str, Any]) -> str:
    """Describe what this call will do, in one line.

    Never shortened: it is the grant key, and two calls that differ past a cut
    would share one grant.

    Never raises: a tool with a broken ``approval_subject`` falls back to the
    full-parameter form, which asks more often rather than less.
    """
    spec = getattr(tool, "approval_subject", None)
    name = getattr(tool, "name", "tool")
    # Applied before every branch below, the declared ones included: a tool whose
    # approval_subject names an injected parameter would otherwise reintroduce
    # exactly the per-call variance this filter exists to remove.
    parameters = visible_parameters(parameters)

    if callable(spec):
        try:
            return _flat(str(spec(**parameters)))
        except Exception as exc:
            get_logger().warning(
                f"approval: approval_subject for '{name}' raised {type(exc).__name__}, "
                "falling back to the full parameters"
            )
            get_logger().debug(f"approval_subject detail: {exc}")

    elif isinstance(spec, list | tuple) and spec:
        parts = [f"{key}={parameters[key]}" for key in spec if key in parameters]
        if parts:
            return _flat(f"{name} {' '.join(parts)}")
        # Declared arguments that are not in this call: rather than approve a
        # subject that describes nothing, fall through to the full form.
        get_logger().debug(f"approval: none of {list(spec)} present in the call to '{name}'")

    if not parameters:
        return name
    rendered = " ".join(f"{key}={parameters[key]}" for key in sorted(parameters))
    return _flat(f"{name} {rendered}")


def _flat(text: str) -> str:
    return " ".join(str(text).split())


def compute_parts(tool: Any, parameters: dict[str, Any]) -> list[str] | None:
    """The commands this call runs, as subjects; ``None`` if the tool does not say.

    Never raises: on failure no parts, so only an exact grant covers the call.
    """
    spec = getattr(tool, "approval_parts", None)
    if not callable(spec):
        return None
    try:
        return [_flat(part) for part in spec(**parameters)]
    except Exception as exc:
        get_logger().warning(
            f"approval: approval_parts for '{getattr(tool, 'name', 'tool')}' raised "
            f"{type(exc).__name__}; only an exact grant can cover the call"
        )
        get_logger().debug(f"approval_parts detail: {exc}")
        return []


def build_request(
    tool: Any,
    parameters: dict[str, Any],
    *,
    agent_name: str | None = None,
    user_id: str | None = None,
    conversation_id: str | None = None,
) -> ApprovalRequest:
    """Assemble the request a policy judges.

    ``parameters`` must be the ones that will actually execute. Building this
    from a proposal that is later edited — the ``PARAMS_UPDATE`` flow lets a
    human change arguments before approving — would approve one thing and run
    another.
    """
    from domyn.agents.approvals.models import ApprovalRequest

    return ApprovalRequest(
        tool_name=getattr(tool, "name", "tool"),
        subject=compute_subject(tool, parameters),
        parameters=parameters,
        parts=compute_parts(tool, parameters),
        agent_name=agent_name,
        user_id=user_id,
        conversation_id=conversation_id,
    )
