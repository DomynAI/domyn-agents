"""The memory recall starts with the turn's other loads, and is not done twice.

What can go wrong: a recall that waits for the other loads instead of running beside
them, a planner that searches again what the runner already searched, and a
failing manager that takes the turn down with it.
"""

import asyncio

import pytest

from domyn.agents import Agent, Runner
from domyn.agents.core import BaseEvent, ExecutionEventType, Part
from domyn.agents.memory.managers import FilesystemMemoryManager


class Recording(FilesystemMemoryManager):
    """A real manager that records its searches and what else was loading then."""

    searches: int = 0
    loads_running: bool | None = None

    async def recall_page(self, *args, **kwargs):
        object.__setattr__(self, "searches", self.searches + 1)
        if self.loads_running is None:
            others = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
            object.__setattr__(self, "loads_running", bool(others))
        return await super().recall_page(*args, **kwargs)


class Failing(FilesystemMemoryManager):
    """Raises on the first recall, the runner's: a manager that breaks its contract."""

    calls: int = 0

    async def recall_block(self, agent, context):
        object.__setattr__(self, "calls", self.calls + 1)
        if self.calls == 1:
            raise RuntimeError("the memory went away")
        return ""


def agent_with(manager, llm_provider, seen: list[str]) -> Agent:
    agent = Agent(
        name="MainAgent",
        description="main agent description",
        instruction="main agent instruction",
        llm_provider=llm_provider,
        memory_manager=manager,
    )

    async def execute(context, **kwargs):
        # What the planner does before its first request.
        seen.append(await agent.memory_manager.recall_block(agent, context))
        yield BaseEvent(
            event_type=ExecutionEventType.RESPONSE, author="MainAgent", content=[Part(text="done")]
        )

    agent._execute = execute
    return agent


async def run_turn(runner: Runner, agent: Agent, text: str) -> list[BaseEvent]:
    return [
        event
        async for event in runner.run(
            conversation_id="c1",
            application_name="TestApplication",
            user_id="test_user",
            root=agent,
            user_input=BaseEvent(
                event_type=ExecutionEventType.USER_INPUT,
                author="user",
                content=[Part(text=text)],
                turn_id="turn_1",
            ),
        )
    ]


@pytest.mark.asyncio
async def test_the_recall_runs_beside_the_other_loads_and_the_planner_reuses_it(
    tmp_path, runner, llm_provider
):
    manager = Recording(file_path=str(tmp_path / "memory.md"))
    (tmp_path / "memory.md").write_text("# Memory\n\n- The user drinks tea every morning\n")
    seen: list[str] = []

    await run_turn(runner, agent_with(manager, llm_provider, seen), "what tea do I like?")

    assert manager.searches == 1
    assert manager.loads_running
    assert "drinks tea" in seen[0]


@pytest.mark.asyncio
async def test_a_failing_recall_does_not_fail_the_turn(tmp_path, runner, llm_provider):
    seen: list[str] = []
    agent = agent_with(Failing(file_path=str(tmp_path / "m.md")), llm_provider, seen)

    events = await run_turn(runner, agent, "hello")

    assert [e.event_type for e in events if e.event_type == ExecutionEventType.ERROR_OCCURRED] == []
    assert any(e.event_type == ExecutionEventType.RESPONSE for e in events)
