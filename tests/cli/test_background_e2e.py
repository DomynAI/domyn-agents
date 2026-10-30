"""End-to-end: a command left running, seen and stopped from the session.

Everything below the fake LLM is real: the ``shell`` bundle with a local backend,
the ``Runner`` persisting conversation state, the REPL and its ``/ps`` and
``/stop``. The unit tests of ``processes_view`` hand it a backend directly, which
is how ``Repl._backend()`` went on reading an attribute nothing set — ``/ps``
never asked the machine and ``/stop`` stopped nothing — with every test green.
"""

import io
import json
from pathlib import Path
from typing import Any

import pytest
from rich.console import Console

from domyn.agents import Runner
from domyn.agents.agents import Agent
from domyn.agents.interfaces.cli import Theme
from domyn.agents.interfaces.cli.repl import Repl
from domyn.agents.llm import BaseLLMProvider, LLMResponse, LLMToolCall, ToolCallFunction
from domyn.agents.planner_strategy import PlannerStrategyFactory
from domyn.agents.tools.builtin import bundle, processes


class _StartsInTheBackground(BaseLLMProvider):
    """Starts one command with ``background=true``, then answers."""

    type: str = "mock"
    name: str = "mock-llm"
    model_name: str = "mock-model"
    supported_structured_outputs: None = None

    async def generate(self, messages: Any, **kwargs: Any) -> LLMResponse:
        if any(getattr(m, "role", "") == "tool" for m in messages):
            return LLMResponse(content="Started.", generated_tokens=1, execution_time=0.0)
        arguments = {"command": "sleep 300", "background": True, "reasoning": "asked to"}
        return LLMResponse(
            content="",
            generated_tokens=1,
            execution_time=0.0,
            tool_calls=[
                LLMToolCall(
                    id="call-1",
                    function=ToolCallFunction(name="run_command", arguments=json.dumps(arguments)),
                )
            ],
        )

    async def generate_streaming(self, messages: Any, **kwargs: Any) -> Any:
        yield await self.generate(messages, **kwargs)


@pytest.fixture
def repl(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Repl, io.StringIO]:
    # TERM-to-KILL grace is waiting this test does not exercise.
    monkeypatch.setattr(processes, "DEFAULT_GRACE_S", 0)
    built = bundle("shell", root=tmp_path)
    agent = Agent(
        name="Shell",
        description="runs commands",
        instruction="Use the tools available to you.",
        llm_provider=_StartsInTheBackground(),
        planner=PlannerStrategyFactory.from_config(
            {"type": "tool_use_thought"}, save_in_registry=False
        ),
        tools=list(built.tools),
        builtin_tools=[built],
        stream=False,
    )
    out = io.StringIO()
    session = Repl(
        agent,
        Console(file=out, width=200, no_color=True),
        Theme.build(allow_animation=False),
        runner=Runner(),
        enable_enhanced_input=False,
    )
    session._state.auto_approve = True
    return session, out


async def _handle(session: Repl) -> str:
    known = processes.registry((await session._conversation_state()).state)
    assert len(known) == 1, "the started command must be in the conversation state"
    return next(iter(known))


class TestABackgroundCommandFromTheSession:
    @pytest.mark.asyncio
    async def test_the_session_reaches_the_agents_shell(
        self, repl: tuple[Repl, io.StringIO]
    ) -> None:
        session, _ = repl
        assert session._backend() is not None

    @pytest.mark.asyncio
    async def test_ps_sees_it_running_and_stop_ends_it(
        self, repl: tuple[Repl, io.StringIO]
    ) -> None:
        session, out = repl
        await session._run_turn("start sleep 300 in the background")
        handle = await _handle(session)

        await session._handle_slash("/ps")
        listing = out.getvalue().split("/ps", 1)[-1]
        assert handle in listing
        assert "running" in listing
        assert "could not be checked" not in listing

        await session._handle_slash(f"/stop {handle}")
        assert f"Stopped {handle}" in out.getvalue()

        # Asked of the machine, not of the record /stop just wrote.
        record = processes.registry((await session._conversation_state()).state)[handle]
        probe = await session._backend().run(processes.build_probe(record), timeout=30)
        assert not processes.parse_report(probe.stdout, record, from_cursor=True).running
