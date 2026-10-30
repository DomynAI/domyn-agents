"""Tool functions the loader tests point at by ``path``, so they need nothing from ``examples/``."""


async def get_order_info(order_id: str) -> dict[str, str]:
    """Retrieve information about a specific order."""
    return {"order_id": order_id, "status": "shipped", "tracking_number": "123456789"}


def get_customer_orders(user_id: str) -> list[str]:
    """Get a list of all the order IDs for a specific user ID."""
    return ["1001", "1002"]


def get_payment_info(order_id: str) -> dict[str, str]:
    """Retrieve information about a specific payment."""
    return {"payment_id": order_id, "status": "completed", "amount": "$42.00"}


def refund_payment(order_id: str) -> dict[str, str]:
    """Initiate a refund for a specific payment."""
    return {"payment_id": order_id, "status": "refund_pending", "amount": "$42.00"}


def get_user_id_from_name(name_of_the_user: str) -> str | None:
    """Tool to extract user ID from the name of the user. If not found return None."""
    return None if name_of_the_user.lower() == "unknown" else "1234"
