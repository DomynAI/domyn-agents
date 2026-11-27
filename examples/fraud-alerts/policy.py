"""The bank's rules on what the agent may do alone, and the audit log of what it did.

``fraud_policy`` is a ``pre_tool_approval`` hook: it runs before the approval gate of each
card action and answers allow, deny, or ask a person. It reads the facts from the bank's
services, never from the agent's arguments, so the agent cannot talk its way past it:

- blocking a card is allowed: it protects the customer and can be undone;
- closing an alert as a false alarm is allowed when its risk score is under the bank's
  threshold (50, changed from the console), and asks an analyst otherwise;
- releasing a held payment always asks an analyst, who can change the amount; an amount
  larger than the payment is refused, even one the analyst typed.

Each verdict, with its rule and reason, is recorded in the bank's services, where the
console shows it beside the action it ruled on.

``audit`` is a listener on every action that ran: it appends one line to the audit log.
"""

import contextlib
import json
import os
import tempfile
from datetime import UTC, datetime

import httpx

from domyn.agents.approvals import ApprovalVerdict

SERVICES_URL = os.getenv("SERVICES_URL", "http://localhost:8001")
AUDIT_LOG = os.path.join(tempfile.gettempdir(), "domyn_fraud_audit.jsonl")


async def _get(path: str) -> dict:
    async with httpx.AsyncClient(base_url=SERVICES_URL, timeout=10) as client:
        response = await client.get(path)
        response.raise_for_status()
        return response.json()


async def _record(alert_id: str, action: str, verdict: ApprovalVerdict) -> ApprovalVerdict:
    """Record the verdict in the bank's services, for the audit and the console, then return
    it. A record that fails does not change the verdict."""
    with contextlib.suppress(httpx.HTTPError):
        async with httpx.AsyncClient(base_url=SERVICES_URL, timeout=5) as client:
            await client.post(
                "/engine/verdicts",
                json={
                    "alert_id": alert_id,
                    "action": action,
                    "decision": str(verdict.decision),
                    "rule": verdict.rule or "",
                    "reason": verdict.reason or "",
                },
            )
    return verdict


async def fraud_policy(tool, tool_args):
    alert_id = str(tool_args.get("alert_id", ""))

    if tool.name == "block_card":
        verdict = ApprovalVerdict.allow(
            rule="Blocking a card is always allowed",
            reason="it protects the customer, and can be undone",
        )
        return await _record(alert_id, tool.name, verdict)

    if tool.name == "close_alert":
        threshold = (await _get("/engine/policy"))["high_risk"]
        risk = (await _get(f"/api/alerts/{alert_id}"))["risk_score"]
        rule = f"Closing an alert asks an analyst from a risk score of {threshold}"
        if risk < threshold:
            verdict = ApprovalVerdict.allow(
                rule=rule, reason=f"risk score {risk} is under {threshold}"
            )
        else:
            verdict = ApprovalVerdict.ask(
                rule=rule, reason=f"risk score {risk} is {threshold} or more"
            )
        return await _record(alert_id, tool.name, verdict)

    if tool.name == "release_payment":
        payment = await _get(f"/api/payments/{tool_args['payment_id']}")
        if float(tool_args["amount"]) > payment["amount"]:
            verdict = ApprovalVerdict.deny(
                rule="Never more than the payment holds",
                reason=f"{float(tool_args['amount']):,.2f} is more than the {payment['amount']:,.2f} held",
            )
        else:
            verdict = ApprovalVerdict.ask(
                rule="Releasing money always asks an analyst",
                reason="the analyst can change the amount before approving it",
            )
        return await _record(alert_id, tool.name, verdict)

    return None  # not a rule of ours: the tool's own approval applies


def audit(tool, result):
    entry = {
        "at": datetime.now(UTC).isoformat(timespec="seconds"),
        "action": tool.name,
        "success": result.success,
        "result": str(result.data if result.success else result.error),
    }
    with open(AUDIT_LOG, "a", encoding="utf-8") as log:
        log.write(json.dumps(entry) + "\n")
