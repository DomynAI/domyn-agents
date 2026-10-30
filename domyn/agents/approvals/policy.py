"""Deciding whether a call needs a human.

The order the rules are consulted in *is* the security model, so it is written
once, here, and nothing else is allowed to reorder it:

1. **Hardline deny** — never bypassable, not by a stored grant, not by a policy
   hook, not by a permissive mode.
2. **Deny rules** — the user's own blocklist.
3. **Stored grant** — the user already said yes to this exact subject.
4. **Policy hooks** — whatever else has an opinion. First decisive one wins.
5. **Ask** — the default, and where everything uncertain lands.

Deny beats allow at every level. A rule that can be talked out of denying is not
a rule.

On the hardline list: it is a backstop against accidents and confused models,
**not** a security boundary. Anything matching on a string can be worked around
by an adversary who wants to — different quoting, an equivalent command, a
script. The boundary is the sandbox the tool runs in; this list only means the
obvious catastrophes cannot happen by mistake, including in permissive modes.
Saying so plainly matters, because a blocklist that is believed to be a boundary
is how people end up running an agent as root with ``--yolo``.
"""

from __future__ import annotations

import fnmatch
import re
from typing import TYPE_CHECKING, Any

from domyn.agents.approvals.models import ApprovalRequest, ApprovalVerdict
from domyn.agents.logger import get_logger

if TYPE_CHECKING:
    from domyn.agents.approvals.store import ApprovalStore

# The atomic groups `(?>…X)` stop at the first X: what matches after a later X
# matches after it too, and retrying every X is quadratic on long subjects.

# A root or a home as a whole argument: `/`, `/*`, `~`, `~/`, `$HOME`, `"${HOME}"/*`.
_ROOT_OR_HOME = r"""["']?(?:/|~|\$\{?HOME\}?)/?\*?["']?/?\*?(?:[\s;&|)`]|$)"""
# Inside one command.
_IN_CMD = r"[^;&|\n]"
# A pipeline segment that runs curl or wget, up to its pipe.
_FETCHED = r"(?:^|\|)(?>[^|]*?\b(?:curl|wget)\b)[^|]*\|\s*(?:sudo\s+)?"

# Subjects that are refused whatever the configuration says. Deliberately short:
# a long list invites the belief that it is exhaustive. These are the ones with
# no legitimate unattended use and no way back once they run.
HARDLINE_DENY: tuple[str, ...] = (
    # `rm -r` of a root or a home, in one command (not `rm -rf x && cd ~`).
    rf"(?:^|[;&|\n])(?>{_IN_CMD}*?\brm\b)"
    rf"(?>{_IN_CMD}*?\s(?:-[a-zA-Z]*[rR][a-zA-Z]*|--recursive)\b){_IN_CMD}*\s" + _ROOT_OR_HOME,
    # Overwriting a block device or a filesystem.
    r"\bmkfs(\.\w+)?\b",
    r"^(?>.*?\bdd\b).*\bof=/dev/",
    # Downloaded code piped into a shell, or into an interpreter given only flags,
    # which reads its program from stdin (not `| python -m json.tool`).
    _FETCHED + r"(?:ba|z|k|)sh\b",
    _FETCHED + r"(?:python[0-9.]*|perl|ruby|node)(?:\s+-[^\s;&|)]*)*\s*(?:$|[;&|)])",
    # Fork bomb.
    r"^(?>.*?:\(\)\s*\{).*\|\s*:\s*&.*\}\s*;\s*:",
)

_HARDLINE = tuple(re.compile(pattern) for pattern in HARDLINE_DENY)


def hardline_match(subject: str) -> str | None:
    """The hardline pattern this subject trips, if any."""
    for pattern, source in zip(_HARDLINE, HARDLINE_DENY, strict=True):
        if pattern.search(subject):
            return source
    return None


class ApprovalPolicy:
    """Resolves one call to a verdict, in the fixed order above.

    Holds no state of its own beyond its configuration: the stores hold the
    memory, the hooks hold the judgement. That keeps this class small enough to
    be read in one sitting, which for the thing that decides what may run is
    worth more than any feature it could grow.
    """

    def __init__(
        self,
        *,
        stores: list[ApprovalStore] | None = None,
        deny_rules: list[str] | None = None,
        enforce_hardline: bool = True,
    ) -> None:
        self.stores = stores or []
        self.deny_rules = deny_rules or []
        self.enforce_hardline = enforce_hardline

    async def resolve(
        self,
        request: ApprovalRequest,
        *,
        context: Any = None,
        hook_verdicts: list[ApprovalVerdict] | None = None,
    ) -> ApprovalVerdict:
        """Decide. Never raises: an exception here would either block a turn or,
        far worse, be caught somewhere that treats failure as permission."""
        # Each part too: a deny rule `$ git push*` must catch `cd x && git push`.
        checked = [request.subject, *(request.parts or [])]
        if self.enforce_hardline:
            for subject in checked:
                matched = hardline_match(subject)
                if matched is not None:
                    get_logger().warning(
                        f"approval: hardline rule blocked {request.tool_name} (pattern {matched!r})"
                    )
                    return ApprovalVerdict.deny(
                        reason="This is on the list of operations that are never run automatically.",
                        rule=f"hardline:{matched}",
                    )

        for rule in self.deny_rules:
            if any(fnmatch.fnmatch(subject, rule) for subject in checked):
                return ApprovalVerdict.deny(
                    reason=f"Blocked by a deny rule ({rule}).", rule=f"deny:{rule}"
                )

        for store in self.stores:
            try:
                # Awaited, and it must stay awaited: an un-awaited coroutine is
                # truthy, so dropping the `await` here silently allows every
                # gated call and raises nothing for the `except` below to catch.
                if await store.is_granted(request, context):
                    return ApprovalVerdict.allow(
                        reason="Already approved.", rule=f"grant:{store.scope}"
                    )
            except Exception as exc:
                # A broken store must not become a broken gate. Falling through
                # costs an extra prompt; failing the other way costs the point
                # of having a gate.
                get_logger().warning(f"approval: store {store.scope} failed: {type(exc).__name__}")
                get_logger().debug(f"approval store detail: {exc}")

        # Hooks are consulted after the stores so a standing grant keeps working
        # even when a policy hook is misconfigured, and before the default so a
        # policy can still be the thing that decides.
        for verdict in hook_verdicts or []:
            if verdict.decision is not None and not _is_ask(verdict):
                return verdict

        return ApprovalVerdict.ask()

    async def remember(
        self,
        request: ApprovalRequest,
        scope: Any,
        context: Any = None,
        pattern: str | None = None,
    ) -> bool:
        """Store a grant in whichever store owns ``scope``.

        Returns whether anything was stored, so a caller can tell the user the
        truth when the scope they picked has nowhere to live.
        """
        for store in self.stores:
            if store.scope == scope:
                try:
                    await store.grant(request, context, pattern=pattern)
                except Exception as exc:
                    get_logger().warning(
                        f"approval: could not store grant in {scope}: {type(exc).__name__}"
                    )
                    get_logger().debug(f"approval grant detail: {exc}")
                    return False
                return True
        return False

    def available_scopes(self) -> list[Any]:
        """Scopes a user can actually pick, i.e. the ones with a store behind
        them. Offering a choice that silently does nothing is worse than not
        offering it."""
        return [store.scope for store in self.stores]


def _is_ask(verdict: ApprovalVerdict) -> bool:
    from domyn.agents.approvals.models import ApprovalDecision

    return verdict.decision is ApprovalDecision.ASK
