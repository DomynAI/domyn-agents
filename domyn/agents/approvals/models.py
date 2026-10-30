"""What an approval decision is about.

The whole design turns on one question: *what exactly is being approved?*

"Approve ``run_command``" grants everything the shell can do. "Approve this
exact invocation" never matches again, so the user is asked every single time
and stops reading the prompt — which is worse than not asking, because it
manufactures consent. Neither is usable.

So a grant is about a **subject**: a normalised string derived from the
risk-bearing arguments of one call. For a shell tool that is the program and its
subcommand (``git status``), for a file writer the path, for a fetcher the host.
Which arguments those are is knowledge only the tool has, so the tool declares
it — see :attr:`ApprovalRequest.subject` and ``FunctionTool.approval_subject``.

The subject is also what gets shown to the user. That is a requirement, not a
convenience: if the prompt says ``git status`` and something else runs, the
approval is theatre.
"""

from __future__ import annotations

from collections.abc import Iterable
from enum import StrEnum
from fnmatch import fnmatch
from typing import Any, Literal

from pydantic import BaseModel, Field


class ApprovalDecision(StrEnum):
    """The three answers a policy can give.

    ``ASK`` is a real answer, not a failure to decide: it means "this needs a
    human", which is the correct outcome for anything a policy is not confident
    about. Everything that goes wrong — an unreachable model, a malformed rule,
    an unknown tool — resolves to ``ASK`` rather than ``ALLOW``.
    """

    ALLOW = "allow"
    DENY = "deny"
    ASK = "ask"


class ApprovalScope(StrEnum):
    """How long a grant lives, and therefore where it is stored.

    Ordered from narrowest to widest. The width of a grant is the user's choice
    and never a policy's: a policy may skip a prompt for one call, only a human
    can create a standing rule.
    """

    ONCE = "once"
    """Not stored at all. The next identical call asks again."""

    CONVERSATION = "conversation"
    """Lives on the interaction state, so it dies with the conversation. This is
    the useful default for a session: approve ``git status`` once and stop being
    asked, without that decision silently following the user into tomorrow's
    work on a different project."""

    USER = "user"
    """Persisted for this user across conversations and projects."""

    PROJECT = "project"
    """Per project, in the user's store: a repo never carries its own grants."""

    GLOBAL = "global"
    """Managed centrally — a platform deployment deciding for its tenants."""


class ApprovalGrant(BaseModel):
    """A standing permission, and how widely it reaches.

    The two kinds are kept apart rather than treated as one glob, and that is a
    correctness matter, not tidiness. ``ls *.py`` is a perfectly ordinary command
    to approve exactly once; stored as a pattern it would silently also permit
    ``ls anything.py``. A permission must never be wider than the string the user
    read, unless they said so.
    """

    value: str
    kind: Literal["exact", "pattern"] = "exact"

    user_id: str | None = None
    """Who granted it. Part of the grant's identity, not decoration: a store
    shared by several users must never let one user's permission answer for
    another, and that has to hold for dedupe and revocation too, not only for
    matching."""

    # Carried for the human, not for matching. A grant that outlives the session
    # has to be reviewable later, and ``value`` alone ("*::run_command::git
    # status") is a key, not an explanation. Optional because a grant read back
    # from an older file will not have them.
    subject: str | None = None
    granted_at: str | None = None

    def matches(self, key: str) -> bool:
        if self.kind == "exact":
            return key == self.value
        return fnmatch(key, self.value)

    def same_as(self, other: ApprovalGrant) -> bool:
        """Whether two grants are the same permission held by the same person.

        Compares what is enforced and ignores the rest: re-approving something
        must not accumulate near-duplicate entries differing only by timestamp,
        because a list nobody can read is a list nobody revokes.
        """
        return (
            self.value == other.value and self.kind == other.kind and self.user_id == other.user_id
        )


class ApprovalRequest(BaseModel):
    """One call, described in the terms a policy needs to judge it."""

    tool_name: str
    subject: str
    """Normalised, human-readable description of what will happen. Grants match
    on this, and it is what the user is shown."""

    parameters: dict[str, Any] = Field(default_factory=dict)
    agent_name: str | None = None
    user_id: str | None = None
    conversation_id: str | None = None
    parts: list[str] | None = None
    """The commands the call runs, when its tool can tell (``FunctionTool.approval_parts``).
    A pattern covers the call only if it covers every part; with no parts, no pattern
    does. ``None``: the subject is the one part."""

    def pattern_key(self, pattern: str) -> str:
        """A pattern, scoped to the same agent and tool as an exact grant.

        Without the same prefix a pattern would reach across tools and agents,
        which is the opposite of what a user picking "and commands like this"
        means.
        """
        return f"{self.agent_name or '*'}::{self.tool_name}::{pattern}"

    def part_key(self, part: str) -> str:
        """A part, keyed like the patterns it must match."""
        return f"{self.agent_name or '*'}::{self.tool_name}::{part}"

    def key(self) -> str:
        """Identity of a grant.

        The agent is part of it deliberately. Without it, approving a command
        for the root agent would silently extend that permission to every
        sub-agent it delegates to, including ones the user never saw.
        """
        return f"{self.agent_name or '*'}::{self.tool_name}::{self.subject}"


def is_already_approved(request: ApprovalRequest, grants: Iterable[ApprovalGrant]) -> bool:
    """Whether standing grants cover a call.

    An exact grant covers only the identical call. Patterns must cover every
    part: ``git *`` does not cover ``git status && rm x``; ``git *`` plus
    ``make *`` cover ``git pull && make``.
    """
    grants = list(grants)
    key = request.key()
    if any(grant.kind == "exact" and grant.value == key for grant in grants):
        return True
    patterns = [grant for grant in grants if grant.kind == "pattern"]
    parts = [request.subject] if request.parts is None else request.parts
    return bool(parts) and all(
        # `+ " "` lets `$ make *` match a bare `make`, never `makeself`.
        any(
            p.matches(request.part_key(part)) or p.matches(f"{request.part_key(part)} ")
            for p in patterns
        )
        for part in parts
    )


class ApprovalVerdict(BaseModel):
    """A decision plus why, so it can be logged, shown and argued with."""

    decision: ApprovalDecision
    reason: str | None = None
    rule: str | None = None
    """What produced the decision — a stored grant, a deny rule, a hook name.
    Carried so a surprising outcome can be traced to the thing that caused it."""

    @classmethod
    def ask(cls, reason: str | None = None, rule: str | None = None) -> ApprovalVerdict:
        return cls(decision=ApprovalDecision.ASK, reason=reason, rule=rule)

    @classmethod
    def allow(cls, reason: str | None = None, rule: str | None = None) -> ApprovalVerdict:
        return cls(decision=ApprovalDecision.ALLOW, reason=reason, rule=rule)

    @classmethod
    def deny(cls, reason: str | None = None, rule: str | None = None) -> ApprovalVerdict:
        return cls(decision=ApprovalDecision.DENY, reason=reason, rule=rule)

    @property
    def is_allow(self) -> bool:
        return self.decision is ApprovalDecision.ALLOW

    @property
    def is_deny(self) -> bool:
        return self.decision is ApprovalDecision.DENY
