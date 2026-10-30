"""The tools of the skills example, loaded by ``config.yaml`` by ``path`` and function name.

The data is made up, and always the same for the same id: a tool that answered differently
on every call would make a model mistake hard to tell from a tool mistake.
"""

import hashlib
from datetime import date, timedelta


def _pick(seed: str, options: list[str]) -> str:
    """Choose an option deterministically from an id, so a rerun is reproducible."""
    digest = hashlib.sha256(seed.encode("utf-8")).digest()
    return options[digest[0] % len(options)]


def _amount(seed: str) -> str:
    digest = hashlib.sha256(seed.encode("utf-8")).digest()
    return f"€{10 + digest[1] % 90}.00"


def get_user_id_from_name(name_of_the_user: str) -> str | None:
    """Look up the customer's user id from their full name. Returns None when no customer matches."""
    if name_of_the_user.strip().lower() in {"", "unknown", "user", "customer"}:
        return None
    digest = hashlib.sha256(name_of_the_user.strip().lower().encode("utf-8")).digest()
    return str(1000 + int.from_bytes(digest[:2], "big") % 9000)


def get_customer_orders(user_id: str) -> list[str]:
    """List every order id belonging to a user id."""
    digest = hashlib.sha256(user_id.encode("utf-8")).digest()
    count = 1 + digest[0] % 3
    return [str(2000 + int.from_bytes(digest[i : i + 2], "big") % 8000) for i in range(count)]


async def get_order_info(order_id: str) -> dict[str, str]:
    """Retrieve the status, tracking number and order date of a single order."""
    status = _pick(order_id, ["processing", "shipped", "delivered"])
    digest = hashlib.sha256(order_id.encode("utf-8")).digest()
    return {
        "order_id": order_id,
        "status": status,
        "tracking_number": str(100000000 + int.from_bytes(digest[:4], "big") % 899999999),
        "order_date": (date.today() - timedelta(days=digest[2] % 20)).isoformat(),
    }


def get_payment_info(order_id: str) -> dict[str, str]:
    """Retrieve the payment status and amount charged for an order."""
    return {
        "payment_id": f"pay_{order_id}",
        "order_id": order_id,
        "status": _pick(f"payment:{order_id}", ["completed", "pending", "failed"]),
        "amount": _amount(order_id),
    }


def refund_payment(order_id: str, refund_time: str) -> dict[str, str]:
    """Start a refund for the payment on an order. Only call this once the refund policy allows it."""
    # `refund_time` is a config param: config.yaml sets it, and the model never sees it.
    return {
        "payment_id": f"pay_{order_id}",
        "order_id": order_id,
        "status": "refund_pending",
        "amount": _amount(order_id),
        "expected_in": refund_time,
    }
