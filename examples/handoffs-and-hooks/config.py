"""A multi-agent system: a front desk that hands each request to a specialist, plus hooks.

To the front desk's model each sub-agent is one more tool, described by its
``description``. ``context_visibility=PEERS`` lets ``BillingAgent`` read what ``OrderAgent``
found, and ``handoff_scope=DIRECT_PEERS`` lets ``OrderAgent`` pass a refund straight to
``BillingAgent``. The hooks: ``refund_policy`` (``PRE_TOOL_APPROVAL``) decides some refunds
without asking you, ``audit`` (``POST_TOOL_CALL``, a listener) prints every tool call. See
README.md to run it.
"""

import os

from domyn.agents import (
    Agent,
    ApprovalType,
    ContextVisibility,
    FunctionTool,
    HandoffScope,
    OpenRouterProvider,
)
from domyn.agents.approvals import ApprovalVerdict
from domyn.agents.core.decorators import hook
from domyn.agents.hooks import HookPhase

# A stand-in for the shop's database.
CUSTOMERS = {"anna rossi": {"customer_id": "C-17", "orders": ["2001", "2002", "2003"]}}
ORDERS = {
    "2001": {"items": ["moka pot"], "status": "delivered", "tracking": "IT-55012"},
    "2002": {"items": ["milk frother"], "status": "delivered", "tracking": "IT-55377"},
    "2003": {"items": ["espresso machine"], "status": "shipped", "tracking": "IT-55940"},
}
PAYMENTS = {
    "2001": {"amount": 35.00, "status": "paid"},
    "2002": {"amount": 89.00, "status": "paid"},
    "2003": {"amount": 649.00, "status": "paid"},
}


@hook(trigger=HookPhase.PRE_TOOL_APPROVAL)
def refund_policy(tool, tool_args):
    """Allow refunds up to 50 EUR, refuse those over 500 EUR, ask a person in between."""
    amount = PAYMENTS.get(tool_args["order_id"], {}).get("amount", 0)
    if amount <= 50:
        return ApprovalVerdict.allow(reason="refunds up to 50 EUR need no approval")
    if amount > 500:
        return ApprovalVerdict.deny(reason="refunds over 500 EUR go through the back office")
    return None  # no opinion: the person decides


@hook(trigger=HookPhase.POST_TOOL_CALL, is_listener=True)
def audit(tool, result):
    print(f"[audit] {tool.name} -> {result.data if result.success else result.error}")


def find_customer(full_name: str) -> dict | str:
    """Find a customer by full name: their customer id and the ids of their orders."""
    customer = CUSTOMERS.get(full_name.strip().lower())
    return customer or f"No customer is called {full_name}."


def get_order(order_id: str) -> dict | str:
    """Look up an order: the items in it, its delivery status and its tracking number."""
    order = ORDERS.get(order_id)
    return {"order_id": order_id, **order} if order else f"There is no order {order_id}."


def get_payment(order_id: str) -> dict | str:
    """Look up the payment of an order: the amount charged and whether it was refunded."""
    payment = PAYMENTS.get(order_id)
    return {"order_id": order_id, **payment} if payment else f"No payment for order {order_id}."


def refund_payment(order_id: str) -> str:
    """Refund the payment of an order in full."""
    payment = PAYMENTS.get(order_id)
    if payment is None:
        return f"No payment for order {order_id}."
    if payment["status"] == "refunded":
        return f"The payment of order {order_id} was already refunded."
    payment["status"] = "refunded"
    return f"Refunded {payment['amount']:.2f} EUR on order {order_id}."


llm = OpenRouterProvider(
    model_name=os.environ["VLLM_MODEL"],
    url=os.environ["VLLM_URL"],
    api_key=os.getenv("VLLM_API_KEY"),
)

billing_agent = Agent(
    name="BillingAgent",
    description="Handles payments and refunds. Does not know where a parcel is.",
    instruction=(
        "You handle payments. Refund only the orders the customer asks about: check the "
        "payment first, and say the amount refunded. A refund is due when the order arrived "
        "damaged or never arrived. If a refund is refused, say why and do not retry it."
    ),
    llm_provider=llm,
    tools=[
        FunctionTool(func=get_payment, hooks=[audit]),
        FunctionTool(
            func=refund_payment,
            approval_type=ApprovalType.USER_CONFIRM,
            hooks=[refund_policy, audit],
        ),
    ],
    context_visibility=ContextVisibility.PEERS,
    stream=True,
)

order_agent = Agent(
    name="OrderAgent",
    description="Knows where a customer's orders are: items, delivery status, tracking number.",
    instruction=(
        "You answer questions about orders and deliveries. If the customer wants their money "
        "back, establish what happened to the order, then hand the refund to BillingAgent."
    ),
    llm_provider=llm,
    tools=[FunctionTool(func=get_order, hooks=[audit])],
    handoff_scope=HandoffScope.DIRECT_PEERS,
    stream=True,
)

front_desk = Agent(
    name="FrontDesk",
    description="The front desk of an online coffee shop.",
    instruction=(
        "You are the front desk of an online coffee shop. Find the customer by name first. "
        "Then hand each request to the right specialist with the ids it needs, and answer "
        "the customer from what the specialist found. Never answer about an order or a "
        "payment yourself."
    ),
    llm_provider=llm,
    tools=[FunctionTool(func=find_customer, hooks=[audit])],
    sub_agents=[order_agent, billing_agent],
    stream=True,
)
