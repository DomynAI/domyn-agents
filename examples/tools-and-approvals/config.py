"""A shop's support agent with three tools, built in Python.

Any function is a tool: its docstring is the description the model reads, and its type
annotations are the schema of its arguments. ``refund_order`` waits for a yes/no before it
runs (``USER_CONFIRM``), ``change_shipping_address`` shows its arguments for you to edit
first (``PARAMS_UPDATE``), and ``refund_order`` keeps what it refunded in the conversation
state. See README.md to run it.
"""

from domyn.agents import Agent, ApprovalType, OpenRouterProvider
from domyn.agents.tools import FunctionTool, ToolResult

# A stand-in for the shop's database.
ORDERS = {
    "1001": {
        "status": "delivered",
        "items": ["coffee grinder"],
        "total": 89.90,
        "address": "Corso Como 5, Milan",
    },
    "1002": {
        "status": "processing",
        "items": ["espresso cups (x6)", "milk jug"],
        "total": 42.50,
        "address": "Via Po 12, Turin",
    },
    "1003": {
        "status": "shipped",
        "items": ["moka pot"],
        "total": 35.00,
        "address": "Piazza Navona 3, Rome",
    },
}


def get_order(order_id: str) -> dict | str:
    """Look up an order: its status, the items in it, the total paid and the shipping address."""
    order = ORDERS.get(order_id)
    if order is None:
        return f"There is no order {order_id}."
    return {"order_id": order_id, **order}


def change_shipping_address(order_id: str, new_address: str) -> str:
    """Change where an order is shipped. Only possible while the order is still processing."""
    order = ORDERS.get(order_id)
    if order is None:
        return f"There is no order {order_id}."
    if order["status"] != "processing":
        return f"Order {order_id} is already {order['status']}: its address cannot change."
    order["address"] = new_address
    return f"Order {order_id} will be shipped to {new_address}."


def refund_order(order_id: str, reason: str, state: dict) -> ToolResult:
    """Refund an order in full. Give the reason the customer stated."""
    # `state` is the conversation state, and is not shown to the model. What the tool
    # returns in `ToolResult.state` is kept for the rest of the conversation.
    refunded = state.get("refunded_orders", [])
    if order_id not in ORDERS:
        return ToolResult(success=False, error=f"There is no order {order_id}.")
    if order_id in refunded:
        return ToolResult(success=False, error=f"Order {order_id} has already been refunded.")
    total = ORDERS[order_id]["total"]
    return ToolResult(
        success=True,
        data=f"Refunded {total:.2f} EUR on order {order_id} ({reason}).",
        state={**state, "refunded_orders": [*refunded, order_id]},
    )


agent = Agent(
    name="SupportAgent",
    description="Answers customers' questions about their orders.",
    instruction=(
        "You are the support agent of an online coffee shop. Look an order up before "
        "answering about it, and before changing or refunding it. Answer in a few sentences."
    ),
    llm_provider=OpenRouterProvider(model_name="nvidia/nemotron-3-ultra-550b-a55b:free"),
    tools=[
        FunctionTool(func=get_order),
        FunctionTool(func=change_shipping_address, approval_type=ApprovalType.PARAMS_UPDATE),
        FunctionTool(func=refund_order, approval_type=ApprovalType.USER_CONFIRM),
    ],
    stream=True,
)
