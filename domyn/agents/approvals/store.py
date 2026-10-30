"""Where standing approvals live.

Split from the policy because *deciding* and *remembering* are different
concerns with different lifetimes, and because the useful part of this feature
is that the two can be mixed: the same rules with a different memory gives
"remember for this conversation" and "remember for this machine" without the
policy knowing which it got.

Three backends ship here. ``InMemoryApprovalStore`` lasts as long as the process.
``ConversationApprovalStore`` writes to the interaction state, which the
conversation manager already persists and already isolates per conversation —
so "approved here, ask again in a new conversation" costs nothing to build and,
more importantly, cannot leak between conversations by construction.
``FileApprovalStore`` writes to disk, which is what makes "remember always"
mean anything.

Every store must implement ``revoke``. A file of standing permissions is a
security artefact, and the thing that makes it one is not the writing but the
taking back: a permission that can only be created is a decision the user cannot
change their mind about.

Note that no store is wired to disk by default. A library that started writing
permissions under a user's home because it was imported would be making that
choice on their behalf; the CLI opts in explicitly.
"""

import os
import tempfile
from abc import ABC, abstractmethod
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from domyn.agents.approvals.models import (
    ApprovalGrant,
    ApprovalRequest,
    ApprovalScope,
    is_already_approved,
)
from domyn.agents.logger import get_logger

# Key under which conversation-scoped grants live on the interaction state.
STATE_KEY = "approved_subjects"


class ApprovalStore(ABC):
    """A place standing approvals are kept, for one scope."""

    scope: ApprovalScope

    @abstractmethod
    async def granted(self, context: Any = None) -> list[ApprovalGrant]:
        """Every standing permission this store holds for the caller."""

    @abstractmethod
    async def store(
        self, grant: ApprovalGrant, request: ApprovalRequest, context: Any = None
    ) -> None:
        """Remember a permission. Only ever called on an explicit human choice."""

    async def is_granted(self, request: ApprovalRequest, context: Any = None) -> bool:
        """Whether this call is already approved (see ``is_already_approved``)."""
        return is_already_approved(request, await self.granted(context))

    async def grant(
        self, request: ApprovalRequest, context: Any = None, pattern: str | None = None
    ) -> None:
        """Remember this call, or a pattern the user chose to cover it with."""
        now = datetime.now(UTC).isoformat(timespec="seconds")
        entry = (
            ApprovalGrant(
                value=request.pattern_key(pattern),
                kind="pattern",
                subject=pattern,
                user_id=request.user_id,
                granted_at=now,
            )
            if pattern
            else ApprovalGrant(
                value=request.key(),
                kind="exact",
                subject=request.subject,
                user_id=request.user_id,
                granted_at=now,
            )
        )
        await self.store(entry, request, context)

    async def revoke(self, grant: ApprovalGrant, context: Any = None) -> bool:
        """Drop one grant. Returns whether it was there to drop.

        Required, not optional. A store that can create a standing permission and
        cannot remove one is a one-way door, and pattern grants make that door
        wide: "and anything matching ``git *``" is a decision a user must be able
        to take back without editing files by hand or wiping everything.
        """
        raise NotImplementedError

    async def revoke_all(self, context: Any = None) -> None:
        """Drop every grant in this store."""
        raise NotImplementedError


class InMemoryApprovalStore(ApprovalStore):
    """Grants for the life of the process.

    Keyed per user as well as per subject: one process serves many users in a
    server deployment, and a permission one of them gave must not answer for
    another. That is a correctness property, not a nicety.
    """

    scope = ApprovalScope.USER

    def __init__(self) -> None:
        self._granted: dict[str, list[ApprovalGrant]] = {}

    async def granted(self, context: Any = None, user_id: str | None = None) -> list[ApprovalGrant]:
        return list(self._granted.get(user_id or "*", []))

    async def is_granted(self, request: ApprovalRequest, context: Any = None) -> bool:
        return is_already_approved(request, self._granted.get(request.user_id or "*", []))

    async def store(
        self, grant: ApprovalGrant, request: ApprovalRequest, context: Any = None
    ) -> None:
        self._granted.setdefault(request.user_id or "*", []).append(grant)

    async def revoke(self, grant: ApprovalGrant, context: Any = None) -> bool:
        removed = False
        for user, grants in self._granted.items():
            kept = [g for g in grants if not g.same_as(grant)]
            removed = removed or len(kept) != len(grants)
            self._granted[user] = kept
        return removed

    async def revoke_all(self, context: Any = None) -> None:
        self._granted.clear()


class ConversationApprovalStore(ApprovalStore):
    """Grants that live and die with one conversation.

    Stored on ``context.state``, which the conversation manager persists and
    scopes per conversation. Nothing here has to enforce that isolation, which
    is the point: an isolation rule that is re-implemented is an isolation rule
    that eventually has a hole in it.

    Grants are also keyed by ``user_id``: a conversation can have several users.
    """

    scope = ApprovalScope.CONVERSATION

    async def is_granted(self, request: ApprovalRequest, context: Any = None) -> bool:
        mine = [g for g in await self.granted(context) if g.user_id == request.user_id]
        return is_already_approved(request, mine)

    async def granted(self, context: Any = None) -> list[ApprovalGrant]:
        state = getattr(context, "state", None)
        if not isinstance(state, dict):
            return []
        stored = state.get(STATE_KEY)
        if not isinstance(stored, list):
            return []
        # Stored as plain dicts: the conversation state is serialised by the
        # conversation manager and has to survive a round trip through JSON.
        out: list[ApprovalGrant] = []
        for item in stored:
            try:
                out.append(ApprovalGrant.model_validate(item))
            except Exception:
                continue
        return out

    async def store(
        self, grant: ApprovalGrant, request: ApprovalRequest, context: Any = None
    ) -> None:
        state = getattr(context, "state", None)
        if not isinstance(state, dict):
            return
        existing = await self.granted(context)
        if any(g.same_as(grant) for g in existing):
            return
        state[STATE_KEY] = [*(g.model_dump() for g in existing), grant.model_dump()]

    async def revoke(self, grant: ApprovalGrant, context: Any = None) -> bool:
        state = getattr(context, "state", None)
        if not isinstance(state, dict):
            return False
        existing = await self.granted(context)
        kept = [g for g in existing if not g.same_as(grant)]
        if len(kept) == len(existing):
            return False
        state[STATE_KEY] = [g.model_dump() for g in kept]
        return True

    async def revoke_all(self, context: Any = None) -> None:
        state = getattr(context, "state", None)
        if isinstance(state, dict):
            state.pop(STATE_KEY, None)


class FileApprovalStore(ApprovalStore):
    """Grants persisted to a file, so they outlive the process.

    This is the store that makes "remember always" mean anything, and it is also
    the one that needs the most care, because it writes down permissions that
    nobody will look at again unless something goes wrong.

    Hence, deliberately:

    * **The file is inspectable and hand-editable.** YAML, one readable record per
      grant, carrying the subject and the date. A permissions file that can only
      be read by the program that wrote it cannot be audited.
    * **Owner-only, and written atomically.** A half-written file is a file that
      parses as fewer grants than it should — which fails safe here (more prompts)
      but would fail open in any store where absence meant "allow".
    * **Unreadable means empty, never fatal.** A corrupt file costs extra prompts;
      crashing the CLI on startup because of it would push people towards
      deleting the whole thing, and a truncated file must not be interpreted
      generously.
    * **Grants record which user gave them** and are matched on that, exactly as
      ``InMemoryApprovalStore`` does. One file per machine is the normal case, but
      "normal case" is not a security argument.

    Two limits worth stating rather than discovering. Concurrent sessions do
    read-modify-write, so two granting at the same instant can lose one of the
    two grants — annoying, never permissive. And a policy holding both this and
    ``InMemoryApprovalStore`` for the same scope will only ever write to whichever
    comes first in its list; do not configure both.
    """

    def __init__(self, path: str | Path, scope: ApprovalScope = ApprovalScope.USER) -> None:
        self.path = Path(path)
        self.scope = scope

    def _load(self) -> list[ApprovalGrant]:
        if not self.path.is_file():
            return []
        try:
            raw = yaml.safe_load(self.path.read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError) as exc:
            get_logger().warning(
                f"approval: ignoring unreadable {self.scope} grants at {self.path}: "
                f"{type(exc).__name__}"
            )
            get_logger().debug(f"approval file detail: {exc}")
            return []
        entries = raw.get("grants") if isinstance(raw, dict) else None
        if not isinstance(entries, list):
            return []
        out: list[ApprovalGrant] = []
        for item in entries:
            try:
                out.append(ApprovalGrant.model_validate(item))
            except Exception:
                # One malformed record must not discard the rest of the file: the
                # others are permissions the user gave and expects to still hold.
                continue
        return out

    def _save(self, grants: list[ApprovalGrant]) -> None:
        payload = {
            "version": 1,
            "grants": [g.model_dump(exclude_none=True) for g in grants],
        }
        # Everything is inside the try, including creating the directory and the
        # temporary file. A read-only parent raises from `mkstemp`, before a
        # single byte is written, and an unwritable config directory must cost the
        # user a warning rather than the session they were in the middle of.
        tmp: str | None = None
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            # Written to a sibling and renamed: os.replace is atomic within a
            # filesystem, so a reader either sees the old file or the new one,
            # never a partial list of permissions.
            fd, tmp = tempfile.mkstemp(dir=self.path.parent, prefix=f".{self.path.name}.")
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                yaml.safe_dump(payload, handle, sort_keys=False)
            os.chmod(tmp, 0o600)
            os.replace(tmp, self.path)
        except (OSError, yaml.YAMLError) as exc:
            if tmp is not None:
                Path(tmp).unlink(missing_ok=True)
            get_logger().warning(
                f"approval: could not write {self.scope} grants to {self.path}: "
                f"{type(exc).__name__}"
            )
            get_logger().debug(f"approval write detail: {exc}")

    async def granted(self, context: Any = None, user_id: str | None = None) -> list[ApprovalGrant]:
        """Every grant in the file, or only one user's when asked.

        Listing everything is the default because that is what an audit wants:
        a review that silently hid other people's standing permissions would be
        worse than no review.
        """
        grants = self._load()
        if user_id is None:
            return grants
        return [g for g in grants if g.user_id == user_id]

    async def is_granted(self, request: ApprovalRequest, context: Any = None) -> bool:
        grants = await self.granted(context, user_id=request.user_id)
        return is_already_approved(request, grants)

    async def store(
        self, grant: ApprovalGrant, request: ApprovalRequest, context: Any = None
    ) -> None:
        existing = self._load()
        if any(g.same_as(grant) for g in existing):
            return
        self._save([*existing, grant])

    async def revoke(self, grant: ApprovalGrant, context: Any = None) -> bool:
        existing = self._load()
        kept = [g for g in existing if not g.same_as(grant)]
        if len(kept) == len(existing):
            return False
        self._save(kept)
        return True

    async def revoke_all(self, context: Any = None) -> None:
        # Removed rather than emptied: an absent file is unambiguous, and leaving
        # `grants: []` behind invites the question of whether something failed.
        self.path.unlink(missing_ok=True)
