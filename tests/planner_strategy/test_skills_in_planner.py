"""The planner is handed a skills payload; it never resolves one.

Resolution moved out: the runner loads a turn's skills
(``tests/runner/test_skills_loading.py``) and the agent selects and renders them
(``tests/agents/test_skills_in_agent.py``). What is left here is the half only
the real loop can show — that the payload survives every iteration of it, and
that the planner does not go looking for skills on the agent behind its back.

Both tool-use planners are parametrized. They run one loop now, but
``ToolUseWithThoughtPlannerStrategy.execute`` was a verbatim copy of its parent's
for long, and a change applied to one and not the other shipped as a bug.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from domyn.agents.llm import LLMResponse, LLMToolCall, ToolCallFunction
from domyn.agents.planner_strategy import ToolUsePlannerStrategy, ToolUseWithThoughtPlannerStrategy
from domyn.agents.skills import (
    SkillContent,
    SkillDescriptor,
    SkillIndex,
    SkillNotFoundError,
    SkillQuery,
    SkillSource,
    SkillsRuntime,
)
from tests.planner_strategy.test_tool_use import (
    make_context,
    make_mock_agent,
    make_mock_tool,
)

PLANNERS = [ToolUsePlannerStrategy, ToolUseWithThoughtPlannerStrategy]

PAYLOAD = (
    "Available skills. They hold instructions for kinds of work, and you must use them. Before starting a task, check this list and read every skill relevant to the task in full with skill_view, before doing anything else. "
    "While working, whenever a step might make use of a skill (a kind of work comes up that one of them covers), read it before taking that step. Follow every skill you read:\n"
    "- sql-style: how we write sql"
)


class TripwireSource(SkillSource):
    """Records any access. The planner must make none of its own."""

    type: str = "tripwire"
    calls: int = 0

    async def index(self, query: SkillQuery) -> SkillIndex:
        self.calls += 1
        return SkillIndex(
            skills=[SkillDescriptor(name="sql-style", description="how we write sql")]
        )

    async def fetch(self, ref, query) -> SkillContent:
        self.calls += 1
        raise SkillNotFoundError(ref.name)


def _two_iteration_llm() -> AsyncMock:
    """An LLM that calls a tool once, then answers — i.e. two loop iterations.

    The turn ends on a response carrying content and no tool calls; that is what
    makes this exactly two trips through the loop, which is the point of the
    every-iteration assertion below.
    """
    return AsyncMock(
        side_effect=[
            LLMResponse(
                content="",
                generated_tokens=1,
                execution_time=0.1,
                tool_calls=[
                    LLMToolCall(
                        id="call_1",
                        function=ToolCallFunction(
                            name="search", arguments='{"query": "x", "reasoning": "r"}'
                        ),
                    )
                ],
            ),
            LLMResponse(
                content="done",
                generated_tokens=1,
                execution_time=0.1,
                tool_calls=[],
            ),
        ]
    )


def _agent_with(skills: SkillsRuntime | None) -> MagicMock:
    agent = make_mock_agent(tools=[make_mock_tool("search")])
    agent.planner = MagicMock()
    agent.planner.use_stop = False
    agent.llm_provider = AsyncMock()
    agent.llm_provider.generate = _two_iteration_llm()
    if skills is not None:
        agent.skills = skills
    return agent


async def _drive(planner, agent, context, **kwargs) -> list[list]:
    """Run the turn; return the message list seen by each LLM call."""
    async for _ in planner.execute(agent, context, **kwargs):
        pass
    return [
        call.kwargs.get("messages", call.args[0] if call.args else [])
        for call in agent.llm_provider.generate.call_args_list
    ]


@pytest.mark.parametrize("planner_cls", PLANNERS)
class TestThePayloadItWasHanded:
    @pytest.mark.asyncio
    async def test_payload_is_present_on_every_iteration(self, planner_cls) -> None:
        """Handed in once, but injected into each rebuilt message list.

        ``messages`` is rebuilt from history on every trip through the loop, so a
        payload inserted only on the first one silently disappears the moment the
        agent calls a tool.
        """
        agent = _agent_with(None)

        message_lists = await _drive(
            planner_cls(use_stop=False), agent, make_context(), skills_payload=PAYLOAD
        )

        assert len(message_lists) == 2, "expected a two-iteration turn"
        for i, messages in enumerate(message_lists):
            assert messages[0].role == "system"
            assert (messages[0].content or "").endswith(PAYLOAD), (
                f"iteration {i} lost the skills block from the end of the system prompt"
            )
            assert all("sql-style" not in (m.content or "") for m in messages[1:])

    @pytest.mark.asyncio
    async def test_no_payload_injects_nothing(self, planner_cls) -> None:
        """The default is the common case: most agents declare no skills."""
        agent = _agent_with(None)

        message_lists = await _drive(planner_cls(use_stop=False), agent, make_context())

        assert len(message_lists) == 2
        for messages in message_lists:
            assert "sql-style" not in " ".join(m.content or "" for m in messages)
