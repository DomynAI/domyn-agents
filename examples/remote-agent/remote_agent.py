"""The remote side of the example: a relay, and an agent served on it.

- **A relay** on ws://localhost:8765 (``mock_relay.py``), through which the two sides talk.
- **A remote agent**, ``Roastery``: an ordinary ``Agent`` with its own tools and model,
  served on the relay's ``roastery`` channel. In a real setup another team would run it, on
  another machine; the front desk of ``config.yaml`` reaches it through the relay only.

See README.md to run it.
"""

import asyncio
import contextlib

from mock_relay import relay_handler
from websockets.asyncio.server import serve

from domyn.agents import Agent, FunctionTool, OpenRouterProvider
from domyn.agents.adapters import DomynAgentAdapter
from domyn.agents.channels import DomynRelayChannel
from domyn.agents.server import AgentServer

STOCK = {"ethiopian yirgacheffe": 12, "colombian huila": 0, "decaf brazil": 30}
ROASTS = {
    "ethiopian yirgacheffe": "Monday",
    "colombian huila": "Thursday",
    "decaf brazil": "Friday",
}


def get_stock(bean: str) -> str:
    """How many 1 kg bags of a bean are in stock."""
    bags = STOCK.get(bean.strip().lower())
    return f"No bean is called {bean}." if bags is None else f"{bags} bags of {bean} in stock."


def get_next_roast(bean: str) -> str:
    """The day a bean is roasted next."""
    day = ROASTS.get(bean.strip().lower())
    return f"No bean is called {bean}." if day is None else f"{bean} is roasted next {day}."


roastery = Agent(
    name="Roastery",
    description="Knows the beans in stock and the roasting schedule.",
    instruction=(
        f"You run the roastery. The beans are: {', '.join(STOCK)}. Answer from your tools only."
    ),
    llm_provider=OpenRouterProvider(model_name="nvidia/nemotron-3-ultra-550b-a55b:free"),
    tools=[FunctionTool(func=get_stock), FunctionTool(func=get_next_roast)],
)


async def main() -> None:
    async with serve(relay_handler, "localhost", 8765):
        channel = DomynRelayChannel(
            ws_url="ws://localhost:8765/relay/v1/ws", headers={"channel-id": "roastery"}
        )
        print("Relay on ws://localhost:8765, Roastery on its channel 'roastery'. Ctrl+C stops.")
        try:
            await AgentServer(channel=channel, adapter=DomynAgentAdapter(root=roastery)).serve()
        finally:
            await channel.close()


if __name__ == "__main__":
    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(main())
