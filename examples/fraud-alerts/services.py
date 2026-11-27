"""The bank's services the agent works with: alerts, customers, transactions and card actions.

A plain FastAPI app with made-up data, standing in for the systems a bank already has. The
same app is also served as an MCP server: ``FastMCP.from_fastapi`` turns each route into a
tool, named by its ``operation_id``, so ``config.yaml`` can give the routes to the agent
without a line of wrapper code.

- ``/api/...``: the REST API, which ``policy.py`` reads to check an alert's risk score.
- ``/mcp``: the same routes as MCP tools, which the agent calls.
- ``/engine/...``: the fraud engine's side, which raises new alerts for the console. Not
  an MCP tool: the agent works alerts, it does not raise them.

See README.md to run it.
"""

import os
from datetime import UTC, datetime, timedelta

from fastapi import APIRouter, FastAPI, HTTPException
from fastmcp import FastMCP
from pydantic import BaseModel

PORT = 8001
HOST = os.getenv("LISTEN_HOST", "127.0.0.1")  # 0.0.0.0 in Docker, to be reachable from the agent
NOW = datetime(2026, 11, 20, 14, 0)


def _ago(minutes: int) -> str:
    return (NOW - timedelta(minutes=minutes)).isoformat(timespec="minutes")


# --- Made-up data -------------------------------------------------------------------------

CUSTOMERS = {
    "C-201": {
        "name": "Giulia Bianchi",
        "since": "2015",
        "segment": "retail",
        "card_id": "K-2011",
        "home_country": "IT",
        "devices": ["iPhone (2023)"],
        "travel_notice": None,
    },
    "C-202": {
        "name": "Marco Ferri",
        "since": "2019",
        "segment": "retail",
        "card_id": "K-2021",
        "home_country": "IT",
        "devices": ["Pixel (2024)"],
        "travel_notice": None,
    },
    "C-203": {
        "name": "Elena Russo",
        "since": "2011",
        "segment": "private",
        "card_id": "K-2031",
        "home_country": "IT",
        "devices": ["iPad (2022)"],
        "travel_notice": None,
    },
    "C-204": {
        "name": "Rossi Arredamenti Srl",
        "since": "2008",
        "segment": "small business",
        "card_id": "K-2041",
        "home_country": "IT",
        "devices": ["Windows PC (2021)"],
        "travel_notice": None,
    },
    "C-205": {
        "name": "Luca Conti",
        "since": "2017",
        "segment": "retail",
        "card_id": "K-2051",
        "home_country": "IT",
        "devices": ["iPhone (2024)"],
        "travel_notice": "Tokyo, 18 to 30 November",
    },
}

CARDS = {
    customer["card_id"]: {"customer_id": cid, "status": "active"}
    for cid, customer in CUSTOMERS.items()
}

TRANSACTIONS = {
    "C-201": [
        *(
            {
                "at": _ago(30 - i),
                "merchant": m,
                "country": "US",
                "channel": "online",
                "amount": a,
                "status": "approved",
            }
            for i, (m, a) in enumerate(
                [
                    ("DIGITALGIFTS.COM", 1.00),
                    ("APP-STORE-X", 1.50),
                    ("DONATE-NOW.ORG", 2.00),
                    ("GAMEKEYS.NET", 1.00),
                    ("E-VOUCHER.IO", 1.99),
                    ("STREAMPASS.TV", 2.49),
                ]
            )
        ),
        {
            "at": _ago(12),
            "merchant": "ELECTRONICS-OUTLET.COM",
            "country": "US",
            "channel": "online",
            "amount": 480.00,
            "status": "declined: over online limit",
        },
        {
            "at": _ago(60 * 26),
            "merchant": "Esselunga Milano",
            "country": "IT",
            "channel": "chip",
            "amount": 63.20,
            "status": "approved",
        },
    ],
    "C-202": [
        {
            "at": _ago(45),
            "merchant": "Bar Centrale Bologna",
            "country": "IT",
            "channel": "chip",
            "amount": 3.80,
            "status": "approved",
        },
        {
            "at": _ago(20),
            "merchant": "SINGAPORE AIRLINES",
            "country": "SG",
            "channel": "online",
            "amount": 712.00,
            "status": "approved",
        },
        {
            "at": _ago(60 * 24 * 40),
            "merchant": "SINGAPORE AIRLINES",
            "country": "SG",
            "channel": "online",
            "amount": 655.00,
            "status": "approved",
        },
    ],
    "C-203": [
        {
            "at": _ago(50),
            "merchant": "Password reset (online banking)",
            "country": "NG",
            "channel": "online banking",
            "amount": 0,
            "status": "completed",
        },
        {
            "at": _ago(40),
            "merchant": "New device registered: Android (unknown)",
            "country": "NG",
            "channel": "online banking",
            "amount": 0,
            "status": "completed",
        },
        {
            "at": _ago(35),
            "merchant": "New payee added: J. Okafor",
            "country": "NG",
            "channel": "online banking",
            "amount": 0,
            "status": "completed",
        },
        {
            "at": _ago(30),
            "merchant": "Transfer to J. Okafor",
            "country": "NG",
            "channel": "online banking",
            "amount": 9500.00,
            "status": "held: P-3031",
        },
    ],
    "C-204": [
        {
            "at": _ago(25),
            "merchant": "Transfer to Legnami Veneto Spa",
            "country": "IT",
            "channel": "online banking",
            "amount": 12000.00,
            "status": "held: P-3041",
        },
        *(
            {
                "at": _ago(60 * 24 * d),
                "merchant": "Transfer to Legnami Veneto Spa",
                "country": "IT",
                "channel": "online banking",
                "amount": a,
                "status": "approved",
            }
            for d, a in [(30, 7800.00), (61, 8100.00), (92, 7650.00)]
        ),
    ],
    "C-205": [
        {
            "at": _ago(60 * 3),
            "merchant": "Lawson Shinjuku",
            "country": "JP",
            "channel": "chip",
            "amount": 14.30,
            "status": "approved",
        },
        {
            "at": _ago(15),
            "merchant": "Bicocca Parking Milano",
            "country": "IT",
            "channel": "online",
            "amount": 6.00,
            "status": "approved",
        },
    ],
}

METRICS = {
    "C-201": {
        "avg_monthly_spend": 1150.0,
        "online_share": 0.20,
        "countries_90d": ["IT"],
        "max_single_payment_90d": 240.0,
    },
    "C-202": {
        "avg_monthly_spend": 1900.0,
        "online_share": 0.45,
        "countries_90d": ["IT", "SG"],
        "max_single_payment_90d": 712.0,
    },
    "C-203": {
        "avg_monthly_spend": 4200.0,
        "online_share": 0.10,
        "countries_90d": ["IT"],
        "max_single_payment_90d": 1800.0,
    },
    "C-204": {
        "avg_monthly_spend": 26000.0,
        "online_share": 0.90,
        "countries_90d": ["IT"],
        "max_single_payment_90d": 8100.0,
    },
    "C-205": {
        "avg_monthly_spend": 1600.0,
        "online_share": 0.30,
        "countries_90d": ["IT", "JP"],
        "max_single_payment_90d": 380.0,
    },
}

PAYMENTS = {
    "P-3031": {"customer_id": "C-203", "payee": "J. Okafor", "amount": 9500.00, "status": "held"},
    "P-3041": {
        "customer_id": "C-204",
        "payee": "Legnami Veneto Spa",
        "amount": 12000.00,
        "status": "held",
    },
}

# What the fraud engine raised. alerts.py publishes these to Kafka.
ALERTS = {
    "A-1001": {
        "type": "card_testing",
        "customer_id": "C-201",
        "risk_score": 88,
        "summary": "Six online charges under 3 EUR in six minutes, then a 480 EUR attempt.",
    },
    "A-1002": {
        "type": "impossible_travel",
        "customer_id": "C-202",
        "risk_score": 35,
        "summary": "Card used in Bologna, then in Singapore 25 minutes later.",
    },
    "A-1003": {
        "type": "account_takeover",
        "customer_id": "C-203",
        "risk_score": 92,
        "summary": "Password reset, new device and new payee, then a 9,500 EUR transfer.",
    },
    "A-1004": {
        "type": "unusual_payment",
        "customer_id": "C-204",
        "risk_score": 61,
        "summary": "A 12,000 EUR transfer, 48% above the largest of the last 90 days.",
    },
    "A-1005": {
        "type": "impossible_travel",
        "customer_id": "C-205",
        "risk_score": 70,
        "summary": "Card used in Tokyo, then in Milan three hours later.",
    },
}
for _alert in ALERTS.values():
    _alert["status"] = "open"


# --- The REST API ------------------------------------------------------------------------

api = FastAPI(title="Bank services")


def _find(table: dict, key: str, what: str) -> dict:
    if key not in table:
        raise HTTPException(status_code=404, detail=f"No {what} {key}.")
    return table[key]


@api.get("/api/alerts/{alert_id}", operation_id="get_alert")
def get_alert(alert_id: str) -> dict:
    """An alert raised by the fraud engine: its type, customer, risk score and status."""
    return {"alert_id": alert_id, **_find(ALERTS, alert_id, "alert")}


@api.get("/api/customers/{customer_id}", operation_id="get_customer")
def get_customer(customer_id: str) -> dict:
    """A customer's profile: name, segment, card, home country, known devices, travel notice."""
    return {"customer_id": customer_id, **_find(CUSTOMERS, customer_id, "customer")}


@api.get("/api/customers/{customer_id}/transactions", operation_id="get_transactions")
def get_transactions(customer_id: str) -> list[dict]:
    """A customer's recent transactions and online banking events, newest first."""
    _find(CUSTOMERS, customer_id, "customer")
    return sorted(TRANSACTIONS.get(customer_id, []), key=lambda t: t["at"], reverse=True)


@api.get("/api/customers/{customer_id}/metrics", operation_id="get_spending_metrics")
def get_spending_metrics(customer_id: str) -> dict:
    """A customer's spending over the last 90 days: average, online share, countries, largest."""
    return {"customer_id": customer_id, **_find(METRICS, customer_id, "customer")}


@api.get("/api/payments/{payment_id}", operation_id="get_payment")
def get_payment(payment_id: str) -> dict:
    """A held payment: customer, payee, amount and status."""
    return {"payment_id": payment_id, **_find(PAYMENTS, payment_id, "payment")}


class BlockCard(BaseModel):
    alert_id: str
    reason: str


class CloseAlert(BaseModel):
    note: str


class ReleasePayment(BaseModel):
    alert_id: str
    amount: float
    note: str


@api.post("/api/cards/{card_id}/block", operation_id="block_card")
def block_card(card_id: str, body: BlockCard) -> dict:
    """Block a card now. The alert is closed as confirmed fraud."""
    card = _find(CARDS, card_id, "card")
    alert = _find(ALERTS, body.alert_id, "alert")
    card["status"] = "blocked"
    alert["status"] = "closed: fraud confirmed"
    return {
        "card_id": card_id,
        "status": "blocked",
        "alert_id": body.alert_id,
        "reason": body.reason,
    }


@api.post("/api/alerts/{alert_id}/close", operation_id="close_alert")
def close_alert(alert_id: str, body: CloseAlert) -> dict:
    """Close an alert as a false alarm, with a note saying why."""
    alert = _find(ALERTS, alert_id, "alert")
    alert["status"] = "closed: false alarm"
    return {"alert_id": alert_id, "status": alert["status"], "note": body.note}


@api.post("/api/payments/{payment_id}/release", operation_id="release_payment")
def release_payment(payment_id: str, body: ReleasePayment) -> dict:
    """Release a held payment, in full or in part. The alert is closed as a false alarm."""
    payment = _find(PAYMENTS, payment_id, "payment")
    if body.amount > payment["amount"]:
        raise HTTPException(status_code=400, detail="More than the payment holds.")
    payment["status"] = "released" if body.amount == payment["amount"] else "partly released"
    _find(ALERTS, body.alert_id, "alert")["status"] = "closed: false alarm"
    return {
        "payment_id": payment_id,
        "released": body.amount,
        "alert_id": body.alert_id,
        "note": body.note,
    }


# --- The fraud engine: new alerts, from the console ------------------------------------------
#
# Each scenario writes what the agent will find (transactions, a held payment) before the
# alert, so the alert and the customer's data tell the same story.

SCENARIOS = {
    "card_testing": {
        "label": "Card testing",
        "card": True,
        "risk_score": 85,
    },
    "travel_online": {
        "label": "Impossible travel: an airline ticket bought online",
        "card": True,
        "risk_score": 35,
    },
    "travel_card_present": {
        "label": "Impossible travel: the card used in two countries",
        "card": True,
        "risk_score": 80,
    },
    "account_takeover": {
        "label": "Account takeover",
        "card": False,
        "risk_score": 90,
        "amount": 6500.0,
    },
    "payment_regular_payee": {
        "label": "Large transfer to a regular supplier",
        "card": False,
        "risk_score": 60,
        "amount": 15000.0,
    },
    "payment_new_payee": {
        "label": "Large transfer to a new payee",
        "card": False,
        "risk_score": 75,
        "amount": 15000.0,
    },
}


def _new_id(table: dict, prefix: str, first: int) -> str:
    numbers = [int(k.split("-")[1]) for k in table]
    return f"{prefix}-{max([*numbers, first - 1]) + 1}"


def _record(
    customer_id: str,
    minutes: int,
    merchant: str,
    country: str,
    channel: str,
    amount: float,
    status: str = "approved",
) -> None:
    TRANSACTIONS.setdefault(customer_id, []).append(
        {
            "at": _ago(minutes),
            "merchant": merchant,
            "country": country,
            "channel": channel,
            "amount": amount,
            "status": status,
        }
    )


def _hold(customer_id: str, payee: str, amount: float) -> str:
    payment_id = _new_id(PAYMENTS, "P", 3031)
    PAYMENTS[payment_id] = {
        "customer_id": customer_id,
        "payee": payee,
        "amount": amount,
        "status": "held",
    }
    return payment_id


def _write_scenario(scenario: str, customer_id: str, amount: float) -> str:
    """Write the scenario's data for the customer; return the alert's summary."""
    metrics = METRICS[customer_id]
    if scenario == "card_testing":
        for i, (merchant, charge) in enumerate(
            [
                ("GIFTCARDS-24.COM", 1.00),
                ("PIXEL-GAMES.IO", 1.49),
                ("CHARITY-RUN.ORG", 2.00),
                ("TOPUP-NOW.NET", 1.00),
                ("MOVIEPASS.TV", 2.99),
            ]
        ):
            _record(customer_id, 9 - i, merchant, "US", "online", charge)
        _record(
            customer_id,
            3,
            "LUXWATCH-OUTLET.COM",
            "US",
            "online",
            620.00,
            "declined: over online limit",
        )
        return "Five online charges under 3 EUR in five minutes, then a 620 EUR attempt."
    if scenario == "travel_online":
        _record(customer_id, 60 * 24 * 55, "LUFTHANSA", "DE", "online", 410.00)
        _record(customer_id, 40, "Caffè Duomo Milano", "IT", "chip", 2.60)
        _record(customer_id, 10, "LUFTHANSA", "DE", "online", 389.00)
        metrics["countries_90d"] = sorted({*metrics["countries_90d"], "DE"})
        return "Card used in Milan, then in Germany 30 minutes later."
    if scenario == "travel_card_present":
        _record(customer_id, 50, "Farmacia Centrale Milano", "IT", "chip", 18.40)
        _record(customer_id, 12, "ATM Bucharest Unirii", "RO", "chip", 300.00)
        _record(customer_id, 8, "Mega Image Bucharest", "RO", "chip", 214.70)
        return "Card used at a pharmacy in Milan, then at an ATM in Bucharest 38 minutes later."
    if scenario == "account_takeover":
        for minutes, event in [
            (45, "Password reset (online banking)"),
            (38, "New device registered: Android (unknown)"),
            (33, "New payee added: D. Popescu"),
        ]:
            _record(customer_id, minutes, event, "RO", "online banking", 0, "completed")
        payment_id = _hold(customer_id, "D. Popescu", amount)
        _record(
            customer_id,
            28,
            "Transfer to D. Popescu",
            "RO",
            "online banking",
            amount,
            f"held: {payment_id}",
        )
        return f"Password reset, new device and new payee, then a {amount:,.0f} EUR transfer."
    if scenario == "payment_regular_payee":
        payee = "Officine Brambilla Srl"
        history = [round(amount * share, 2) for share in (0.62, 0.70, 0.66)]
        for days, past in zip((28, 59, 90), history, strict=True):
            _record(
                customer_id, 60 * 24 * days, f"Transfer to {payee}", "IT", "online banking", past
            )
        metrics["max_single_payment_90d"] = max(metrics["max_single_payment_90d"], *history)
        payment_id = _hold(customer_id, payee, amount)
        _record(
            customer_id,
            20,
            f"Transfer to {payee}",
            "IT",
            "online banking",
            amount,
            f"held: {payment_id}",
        )
        above = amount / metrics["max_single_payment_90d"] - 1
        return f"A {amount:,.0f} EUR transfer, {above:.0%} above the largest of the last 90 days."
    # payment_new_payee
    payee = "Global Trade Partners Ltd"
    payment_id = _hold(customer_id, payee, amount)
    _record(customer_id, 32, f"New payee added: {payee}", "IT", "online banking", 0, "completed")
    _record(
        customer_id,
        20,
        f"Transfer to {payee}",
        "GB",
        "online banking",
        amount,
        f"held: {payment_id}",
    )
    return f"A {amount:,.0f} EUR transfer to a payee added 12 minutes earlier."


engine = APIRouter()

# The bank's rules, as the console sets them, and what they decided. policy.py reads the
# first and writes the second: the rules are the bank's code, their record its audit.
POLICY = {"high_risk": 50}
VERDICTS: list[dict] = []


class Policy(BaseModel):
    high_risk: int


@engine.get("/engine/policy")
def get_policy() -> dict:
    """The settings of the bank's rules."""
    return POLICY


@engine.put("/engine/policy")
def set_policy(body: Policy) -> dict:
    """Change the risk score from which closing an alert needs an analyst."""
    if not 0 <= body.high_risk <= 100:
        raise HTTPException(status_code=422, detail="The threshold is between 0 and 100.")
    POLICY["high_risk"] = body.high_risk
    return POLICY


class Verdict(BaseModel):
    alert_id: str
    action: str
    decision: str  # allow, deny or ask
    rule: str
    reason: str


@engine.post("/engine/verdicts")
def record_verdict(body: Verdict) -> dict:
    """Record what a rule decided, for the audit and the console."""
    VERDICTS.append({"at": datetime.now(UTC).isoformat(), **body.model_dump()})
    return {"recorded": len(VERDICTS)}


@engine.get("/engine/verdicts")
def verdicts(after: int = 0) -> list[dict]:
    """The rules' decisions, oldest first, from the ``after``-th on."""
    return VERDICTS[after:]


@engine.get("/engine/scenarios")
def scenarios() -> dict:
    """The scenarios a new alert can follow, and the customers it can be about."""
    customers = [
        {
            "customer_id": cid,
            "name": c["name"],
            "segment": c["segment"],
            "card_blocked": CARDS[c["card_id"]]["status"] == "blocked",
        }
        for cid, c in CUSTOMERS.items()
    ]
    return {"scenarios": SCENARIOS, "customers": customers}


class NewAlert(BaseModel):
    scenario: str
    customer_id: str
    risk_score: int
    amount: float | None = None


@engine.post("/engine/alerts")
def raise_alert(body: NewAlert) -> dict:
    """Raise a new alert: write the scenario's data, then the alert. Publishing it is the caller's."""
    scenario = SCENARIOS.get(body.scenario)
    if scenario is None:
        raise HTTPException(status_code=404, detail=f"No scenario {body.scenario}.")
    customer = _find(CUSTOMERS, body.customer_id, "customer")
    if scenario["card"] and CARDS[customer["card_id"]]["status"] == "blocked":
        raise HTTPException(
            status_code=409,
            detail=f"The card of {customer['name']} is blocked: it cannot make transactions.",
        )
    if not 0 <= body.risk_score <= 100:
        raise HTTPException(status_code=422, detail="The risk score is between 0 and 100.")
    amount = body.amount or scenario.get("amount", 0.0)
    if "amount" in scenario and amount <= 0:
        raise HTTPException(status_code=422, detail="The amount must be more than 0.")
    summary = _write_scenario(body.scenario, body.customer_id, amount)
    alert_id = _new_id(ALERTS, "A", 1001)
    kind = body.scenario.split("_")[0]
    ALERTS[alert_id] = {
        "type": {
            "card": "card_testing",
            "travel": "impossible_travel",
            "account": "account_takeover",
            "payment": "unusual_payment",
        }[kind],
        "customer_id": body.customer_id,
        "risk_score": body.risk_score,
        "summary": summary,
        "status": "open",
    }
    return {"alert_id": alert_id, **ALERTS[alert_id]}


# --- The same routes as MCP tools ---------------------------------------------------------

mcp_app = FastMCP.from_fastapi(api, name="bank").http_app(path="/mcp")
app = FastAPI(
    title="Bank services", routes=[*mcp_app.routes, *api.routes], lifespan=mcp_app.lifespan
)
app.include_router(engine)  # after the MCP server is built from `api`: not a tool


if __name__ == "__main__":
    import uvicorn

    print(f"Bank services on http://localhost:{PORT}", flush=True)
    uvicorn.run(app, host=HOST, port=PORT, log_level="warning")
