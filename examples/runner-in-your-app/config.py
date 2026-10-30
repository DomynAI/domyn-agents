"""The agent ``app.py`` runs: a shop assistant with two tools and long-term memory.

The memory is ``AGENTIC``: the agent gets a ``remember`` tool and decides what to keep.
What it keeps outlives the conversation, and is recalled in every new one.
"""

import os
import tempfile

from domyn.agents import Agent, ApprovalType, FunctionTool, OpenRouterProvider
from domyn.agents.memory.managers import FilesystemMemoryManager, MemoryExecutionMode


def get_order(order_id: str) -> str:
    """Look up the status of an order."""
    return f"Order {order_id}: a bag of decaf beans, delivered on 2 October, 14.90 EUR."


def refund_order(order_id: str) -> str:
    """Refund an order in full."""
    return f"Refunded order {order_id}."


agent = Agent(
    name="ShopAssistant",
    description="The assistant of an online coffee shop.",
    instruction=(
        "You are the assistant of an online coffee shop. When the customer tells you "
        "something about themselves, keep it with the remember tool. Answer briefly."
    ),
    llm_provider=OpenRouterProvider(model_name="nvidia/nemotron-3-ultra-550b-a55b:free"),
    tools=[
        FunctionTool(func=get_order),
        FunctionTool(func=refund_order, approval_type=ApprovalType.USER_CONFIRM),
    ],
    memory_manager=FilesystemMemoryManager(
        file_path=os.path.join(tempfile.gettempdir(), "domyn_example_memory.jsonl"),
        execution_mode=MemoryExecutionMode.AGENTIC,
    ),
    stream=True,
)
