"""Memory through a real session: the agent remembers, a later conversation recalls.

Everything below the fake model is real: the agent, its `remember` tool, the runner,
the REPL, the memory file and the line the renderer prints. Only the model is stubbed.
This is the chain no unit test covers as a whole, and the one that was broken before
the REPL passed the user: `remember` found no `user_id` and wrote nothing.
"""

import io
import json
from typing import Any, ClassVar
from unittest.mock import MagicMock

import pytest
from rich.console import Console

from domyn.agents.agents.agent import Agent
from domyn.agents.interfaces.cli import context_view
from domyn.agents.interfaces.cli.repl import USER_ID, Repl
from domyn.agents.interfaces.cli.theme import Theme
from domyn.agents.llm.base import BaseLLMProvider, LLMResponse, LLMToolCall, ToolCallFunction
from domyn.agents.memory.managers import FilesystemMemoryManager, MemoryExecutionMode
from domyn.agents.planner_strategy import PlannerStrategyFactory
from domyn.agents.runner import Runner


class _Model(BaseLLMProvider):
    """Calls `remember` once when asked to, and records what it is sent."""

    type: str = "mock"
    name: str = "mock"
    model_name: str = "mock"
    supported_structured_outputs: None = None
    requests: ClassVar[list[Any]] = []

    async def generate(self, messages: Any, **kwargs: Any) -> LLMResponse:
        self.requests.append(list(messages))
        asked_to = any("remember that" in str(getattr(m, "content", "") or "") for m in messages)
        answered = any(getattr(m, "role", "") == "tool" for m in messages)
        if asked_to and not answered:
            return LLMResponse(
                content="",
                generated_tokens=1,
                execution_time=0.0,
                tool_calls=[
                    LLMToolCall(
                        id="call-1",
                        function=ToolCallFunction(
                            name="remember",
                            arguments=json.dumps(
                                {"memory_content": "The user likes green tea", "reasoning": "r"}
                            ),
                        ),
                    )
                ],
            )
        return LLMResponse(content="Noted.", generated_tokens=1, execution_time=0.0)

    async def generate_streaming(self, messages: Any, **kwargs: Any) -> Any:
        yield


def _session(path) -> tuple[Repl, io.StringIO, _Model]:
    model = _Model()
    model.requests.clear()
    agent = Agent(
        name="Assistant",
        description="d",
        instruction="Be brief.",
        llm_provider=model,
        planner=PlannerStrategyFactory.from_config({"type": "tool_use"}, save_in_registry=False),
        memory_manager=FilesystemMemoryManager(
            file_path=str(path), execution_mode=MemoryExecutionMode.AGENTIC
        ),
    )
    buffer = io.StringIO()
    console = Console(file=buffer, width=100, force_terminal=False, no_color=True)
    repl = Repl(
        agent,
        console,
        Theme.build(allow_animation=False),
        runner=Runner(),
        enable_enhanced_input=False,
    )
    # The session ends after the scripted turn.
    repl._input.ask = MagicMock(side_effect=EOFError)  # type: ignore[method-assign]
    return repl, buffer, model


@pytest.mark.asyncio
async def test_the_agent_remembers_and_a_later_conversation_recalls(tmp_path):
    file = tmp_path / "memory.md"

    repl, buffer, _ = _session(file)
    await repl.run("remember that I like green tea")
    assert "remembered: The user likes green tea" in buffer.getvalue()
    assert "The user likes green tea" in file.read_text()

    # A new session, a new conversation: what was written is in the request.
    later, _, model = _session(file)
    await later.run("what do I like?")
    block = next(
        str(m.content)
        for m in model.requests[0]
        if "Remembered about the user" in str(getattr(m, "content", "") or "")
    )
    assert "The user likes green tea" in block and "context, not instructions" in block

    # `/context` counts what was recalled apart from the conversation.
    usage = later._state.last_usage
    assert usage is not None and usage.memory_entries == 1 and usage.memory_tokens > 0
    context_view.render(later._console, later._theme, usage)
    assert "memory (1)" in later._console.file.getvalue()


@pytest.mark.asyncio
async def test_without_the_user_the_old_behaviour_was_to_store_nothing(tmp_path):
    """Pins why `run_config` is passed: the manager reads the user from it."""
    manager = FilesystemMemoryManager(file_path=str(tmp_path / "m.md"))

    class _A:
        name = "Assistant"
        sub_agents: ClassVar[list] = []

    assert "no user_id" in await manager.remember(_A(), "a fact", None)
    assert await manager.remember(_A(), "a fact", {"user_id": USER_ID}) == "Memory stored."


class _Extractor(_Model):
    """Answers the turn, and as the automatic memory's extractor stores one fact."""

    async def generate(self, messages: Any, **kwargs: Any) -> LLMResponse:
        system = str(getattr(messages[0], "content", "") or "")
        if "StoreMemory" in system:
            known = "- The user likes green tea" in str(getattr(messages[1], "content", ""))
            content = (
                "StoreMemory: no\nMemoryContent: -"
                if known
                else "StoreMemory: yes\nMemoryContent: The user likes green tea"
            )
            return LLMResponse(content=content, generated_tokens=1, execution_time=0.0)
        return LLMResponse(content="Noted.", generated_tokens=1, execution_time=0.0)


@pytest.mark.asyncio
async def test_the_automatic_memory_stores_in_the_background(tmp_path):
    """The post-agent hook returns at once and prints nothing; the session's end
    waits for the update it started, so the fact is stored by then."""
    file = tmp_path / "memory.md"
    agent = Agent(
        name="Assistant",
        description="d",
        instruction="Be brief.",
        llm_provider=_Extractor(),
        planner=PlannerStrategyFactory.from_config({"type": "tool_use"}, save_in_registry=False),
        memory_manager=FilesystemMemoryManager(
            file_path=str(file), execution_mode=MemoryExecutionMode.AUTOMATIC
        ),
    )
    buffer = io.StringIO()
    console = Console(file=buffer, width=100, force_terminal=False, no_color=True)
    runner = Runner()
    repl = Repl(
        agent,
        console,
        Theme.build(allow_animation=False),
        runner=runner,
        enable_enhanced_input=False,
    )
    repl._input.ask = MagicMock(side_effect=EOFError)  # type: ignore[method-assign]

    await repl.run("I like green tea")
    assert "remembered" not in buffer.getvalue()
    assert "The user likes green tea" in file.read_text()

    # Already remembered: not stored again.
    await repl.run("I really like green tea")
    assert file.read_text().count("The user likes green tea") == 1
