"""The bank's rule on onboarding decisions: code, not an instruction to the model.

``kyc_policy`` is a ``pre_tool_approval`` hook on ``save_memo``. Before a memo is saved it
screens, itself, every party behind the company in the registry (the company, its
directors, its shareholders down the chain), whatever the agents said about them:

- approving is refused when a party is on the internal watchlist, or matches a listed
  party closely with nothing that tells them apart (same birth date, or an entity with the
  same name): that case is rejected or escalated, never approved;
- approving with possible matches still open asks the analyst, saying which ones they must
  clear before confirming;
- anything else (rejecting, escalating, approving with nobody on any list) is left to the
  analyst's usual confirmation.

So the analyst confirms every memo in the terminal: the rule decides what the agent may
not do, the analyst signs what it does.
"""

import os
import sys

from domyn.agents.approvals import ApprovalVerdict

# The loader runs each file on its own, so this one finds its neighbours itself.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from registry import parties
from screening import find_candidates

CAUTIOUS = ("reject", "escalate")
SAME_NAME = 0.95


def _findings(company: str) -> tuple[list[str], list[str]]:
    """What the lists say about the company's parties: (blocking, to clear)."""
    blocking, to_clear = [], []
    for party in parties(company):
        for score, entry, listed in find_candidates(party["name"], party["kind"]):
            what = f"{party['name']} ~ {listed} ({entry.source}, {entry.entry_id})"
            born = party.get("birth_date", "")
            same_person = bool(born) and any(
                born.startswith(d) or d.startswith(born) for d in entry.birth_dates
            )
            if entry.source == "Internal watchlist":
                blocking.append(f"{what}: on the bank's own watchlist")
            elif party["kind"] == "entity" and score >= SAME_NAME:
                blocking.append(f"{what}: same name")
            elif same_person:
                blocking.append(f"{what}: same birth date")
            else:
                to_clear.append(what)
    return blocking, to_clear


def kyc_policy(tool, tool_args):
    if tool.name != "save_memo":
        return None
    if tool_args.get("decision", "") in CAUTIOUS:
        return None  # the cautious way: the analyst confirms it as usual

    blocking, to_clear = _findings(tool_args.get("company", ""))
    if blocking:
        return ApprovalVerdict.deny(
            rule="kyc_policy",
            reason="The bank's rule refuses to approve: "
            + "; ".join(blocking)
            + ". Reject or escalate instead.",
        )
    if to_clear:
        return ApprovalVerdict.ask(
            rule="kyc_policy",
            reason="Possible matches an analyst must clear before approving: "
            + "; ".join(to_clear),
        )
    return None  # no party on any list: the analyst confirms it as usual
