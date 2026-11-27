"""Saving the KYC memo: the one action of the example, ruled on by policy.py first."""

import re
from datetime import date
from pathlib import Path

MEMOS = Path(__file__).parent / "memos"
DECISIONS = ("approve", "approve_with_conditions", "reject", "escalate")
RATINGS = ("low", "medium", "high")


def save_memo(company: str, decision: str, risk_rating: str, memo: str) -> str:
    """Save the KYC memo of an onboarding and record its decision.

    ``decision`` is one of approve, approve_with_conditions, reject, escalate;
    ``risk_rating`` one of low, medium, high. ``memo`` is the memo itself, in markdown:
    what was checked, what was found, and why the decision follows from it. The bank's
    rules check the decision before it is saved, and an analyst confirms it.
    """
    if decision not in DECISIONS:
        return f"Unknown decision {decision!r}: use one of {', '.join(DECISIONS)}."
    if risk_rating not in RATINGS:
        return f"Unknown risk rating {risk_rating!r}: use one of {', '.join(RATINGS)}."
    MEMOS.mkdir(exist_ok=True)
    slug = re.sub(r"[^a-z0-9]+", "-", company.lower()).strip("-")
    path = MEMOS / f"{slug}.md"
    header = (
        f"# KYC memo: {company}\n\n"
        f"- Date: {date.today().isoformat()}\n"
        f"- Decision: **{decision.replace('_', ' ')}**\n"
        f"- Risk rating: **{risk_rating}**\n\n"
    )
    path.write_text(header + memo.strip() + "\n", encoding="utf-8")
    return f"Saved the memo of {company} to memos/{path.name}: {decision}, {risk_rating} risk."
