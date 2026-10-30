"""Showing and removing standing approvals.

Kept apart from the REPL because the interesting part is not the printing: it is
that a user can *find out* what they have agreed to. Approving is a decision made
in a hurry, under a prompt, while waiting for an answer; reviewing it happens
later, deliberately, and needs the whole list in one place with the widest and
most dangerous entries visible rather than buried.

Two presentation rules follow from that, and both are deliberate:

* **Pattern grants are marked.** ``git status`` and ``git *`` are one character
  apart in a list and worlds apart in what they permit. The one that matches
  future commands nobody has read yet is the one worth revoking, so it is called
  out rather than left to be inferred from a quoted string.
* **Empty is stated, not implied.** "You have no standing approvals" is
  information. A blank response looks like a command that failed.

Indices are assigned over the whole list, narrowest scope first, so ``revoke 3``
means one unambiguous thing. They are positional and change as the list changes —
which is why revoking prints back what it removed rather than only confirming.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from domyn.agents.approvals import ApprovalGrant, ApprovalPolicy, ApprovalStore

if TYPE_CHECKING:
    from pathlib import Path

    from rich.console import Console

    from domyn.agents.interfaces.cli.theme import Theme

# Narrowest first, matching the order the prompt offers them in. A scope the
# policy reports that is not here still gets listed, under its own name: hiding a
# standing permission because this module has not been taught about its scope
# would be the worst possible failure mode for an audit view.
_SCOPE_ORDER = ("conversation", "project", "user", "global")

_SCOPE_BLURB: dict[str, str] = {
    "conversation": "until this conversation ends",
    "project": "in this directory, from now on",
    "user": "everywhere, from now on",
    "global": "for everyone",
}


@dataclass(frozen=True)
class Entry:
    """One standing permission, with where it lives and how to remove it."""

    index: int
    scope: str
    grant: ApprovalGrant
    store: ApprovalStore

    @property
    def label(self) -> str:
        """What the user approved, in the words they were shown.

        Falls back to the stored key for grants written before the subject was
        recorded — unhelpful to read, but it must still be revocable.
        """
        return self.grant.subject or self.grant.value


async def collect(policy: ApprovalPolicy, *, context: object = None) -> list[Entry]:
    """Every standing grant the policy can see, numbered from 1.

    A store that fails to answer is skipped with its scope still counted as
    present, on the same reasoning as the policy itself: a broken store must not
    take the rest of the review down with it.

    Async because a store may be backed by something remote — a database rather
    than this process or a local file — and the review has to wait for it.
    """
    entries: list[Entry] = []
    ordered = sorted(
        policy.stores,
        key=lambda s: (
            _SCOPE_ORDER.index(str(s.scope)) if str(s.scope) in _SCOPE_ORDER else len(_SCOPE_ORDER)
        ),
    )
    for store in ordered:
        try:
            grants = await store.granted(context)
        except Exception:
            continue
        for grant in grants:
            entries.append(
                Entry(index=len(entries) + 1, scope=str(store.scope), grant=grant, store=store)
            )
    return entries


def render(
    console: Console,
    theme: Theme,
    entries: list[Entry],
    *,
    paths: dict[str, Path] | None = None,
) -> None:
    """Print the list, grouped by scope."""
    if not entries:
        console.print(
            "  No standing approvals — every tool call that needs one will ask.",
            style=theme.style("meta"),
            highlight=False,
        )
        return

    patterns = sum(1 for e in entries if e.grant.kind == "pattern")
    console.print()
    for scope in dict.fromkeys(e.scope for e in entries):
        blurb = _SCOPE_BLURB.get(scope, scope)
        console.print(f"  {scope}  ", style=theme.style("accent_bold"), end="")
        console.print(blurb, style=theme.style("meta"), highlight=False)
        for entry in (e for e in entries if e.scope == scope):
            mark = "  matches anything like it" if entry.grant.kind == "pattern" else ""
            # highlight=False, or rich's number highlighter repaints the index and
            # it competes with the subject for attention.
            console.print(
                f"   {entry.index})  ", style=theme.style("meta"), end="", highlight=False
            )
            console.print(entry.label, style=theme.style("approval"), end="", highlight=False)
            if mark:
                console.print(mark, style=theme.style("warning"), highlight=False)
            else:
                console.print(
                    f"   {entry.grant.granted_at or ''}",
                    style=theme.style("meta"),
                    highlight=False,
                )
        if paths and scope in paths:
            console.print(f"      {paths[scope]}", style=theme.style("meta"), highlight=False)
    console.print()
    if patterns:
        # Said once, at the bottom, where someone deciding what to remove is
        # looking: a pattern grant is the only entry here that can permit
        # something the user has never seen.
        console.print(
            f"  {patterns} of these match by pattern, so they cover calls you have not seen.",
            style=theme.style("warning"),
            highlight=False,
        )
    console.print(
        "  /approvals revoke <n> to remove one · /approvals clear to remove all",
        style=theme.style("meta"),
        highlight=False,
    )


def parse_selector(argument: str, total: int) -> tuple[list[int], str | None]:
    """Turn ``revoke``'s argument into indices, or explain why it cannot.

    Rejects anything it does not fully understand instead of acting on the part
    it does: "revoke 2 4" quietly removing only entry 2 would leave the user
    believing a permission is gone when it is not.
    """
    raw = argument.strip()
    if not raw:
        return [], "which one? /approvals revoke <n>"
    wanted: list[int] = []
    for token in raw.replace(",", " ").split():
        if not token.isdigit():
            return [], f"{token!r} is not a number from the list"
        number = int(token)
        if not 1 <= number <= total:
            return [], f"there is no {number} in the list (1-{total})"
        if number not in wanted:
            wanted.append(number)
    return wanted, None
