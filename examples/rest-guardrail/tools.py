"""The tool of the guardrail example, loaded by ``config.yaml`` by ``path`` and function name."""

CUSTOMERS = {
    "mario bianchi": {"email": "mario.bianchi@example.com", "orders": ["1001", "1004"]},
    "jane smith": {"email": "jane.smith@example.com", "orders": ["1002"]},
}


def get_customer(full_name: str) -> dict | str:
    """Find a customer by full name: their email address and the ids of their orders."""
    customer = CUSTOMERS.get(full_name.strip().lower())
    return customer or f"No customer is called {full_name}."
