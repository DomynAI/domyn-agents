"""An agent that names no planner runs on ``tool_use_thought``, built in code or from a config."""

from collections.abc import Awaitable, Callable

import pytest

from domyn.agents import Agent
from domyn.agents.llm import VLLMProvider
from domyn.agents.loader import load_application_from_config_new
from domyn.agents.planner_strategy import ToolUseWithThoughtPlannerStrategy

_URL = "http://mock_url/v1/chat/completions"


async def _in_code() -> Agent:
    provider = VLLMProvider(name="p", model_name="m", url=_URL)
    return Agent(name="A", description="d", instruction="i", llm_provider=provider)


async def _from_config() -> Agent:
    config = {
        "version": "1.0",
        "entrypoint": "A",
        "llm_providers": [{"name": "p", "type": "vllm", "model_name": "m", "url": _URL}],
        "agents": [{"name": "A", "description": "d", "instruction": "i", "llm_provider": "p"}],
    }
    return await load_application_from_config_new(config, mcp_init=False)


@pytest.mark.asyncio
@pytest.mark.parametrize("build", [_in_code, _from_config], ids=["in_code", "from_config"])
async def test_an_agent_that_names_no_planner_runs_on_tool_use_thought(
    build: Callable[[], Awaitable[Agent]],
) -> None:
    agent = await build()
    assert isinstance(agent.planner, ToolUseWithThoughtPlannerStrategy)
