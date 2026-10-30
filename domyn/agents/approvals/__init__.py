"""Approval policy: deciding which tool calls need a human, and remembering.

The runner already knows how to stop a turn and wait for a person — that is
``TOOL_NEEDS_FEEDBACK``/``TOOL_FEEDBACK``, and none of it changes here. What was
missing is everything *in front of* that gate: a way to say "yes, and stop
asking me about this one", and a way to say no without a human at all.

The split is the same one used for turn continuation: the runner owns the
mechanism, a policy owns the judgement.

* :mod:`~domyn.agents.approvals.models` — what a decision is about
* :mod:`~domyn.agents.approvals.subject` — turning a call into an approvable thing
* :mod:`~domyn.agents.approvals.store` — where standing grants live
* :mod:`~domyn.agents.approvals.policy` — the order the rules are consulted in
"""

from domyn.agents.approvals.models import (
    ApprovalDecision,
    ApprovalGrant,
    ApprovalRequest,
    ApprovalScope,
    ApprovalVerdict,
    is_already_approved,
)
from domyn.agents.approvals.policy import HARDLINE_DENY, ApprovalPolicy, hardline_match
from domyn.agents.approvals.store import (
    STATE_KEY,
    ApprovalStore,
    ConversationApprovalStore,
    FileApprovalStore,
    InMemoryApprovalStore,
)
from domyn.agents.approvals.subject import (
    INJECTED_PARAMS,
    build_request,
    compute_parts,
    compute_subject,
    visible_parameters,
)

__all__ = [
    "HARDLINE_DENY",
    "INJECTED_PARAMS",
    "STATE_KEY",
    "ApprovalDecision",
    "ApprovalGrant",
    "ApprovalPolicy",
    "ApprovalRequest",
    "ApprovalScope",
    "ApprovalStore",
    "ApprovalVerdict",
    "ConversationApprovalStore",
    "FileApprovalStore",
    "InMemoryApprovalStore",
    "build_request",
    "compute_parts",
    "compute_subject",
    "hardline_match",
    "is_already_approved",
    "visible_parameters",
]
