"""Several sub-agents in one response, against a real model. Skipped unless asked.

    DOMYN_LIVE_LLM_URL=https://host/v1/chat/completions \
    DOMYN_LIVE_LLM_MODEL=deepseek-ai/DeepSeek-V4.1-Flash \
    DOMYN_LIVE_LLM_API_KEY=... \
    uv run pytest tests/runner/test_parallel_handoffs_live.py -q --no-cov -n0

`test_parallel_handoffs.py` scripts the model, so it proves what the runner does
with a response naming two sub-agents; it cannot prove that a model writes one,
nor that a real endpoint accepts the requests the runner builds from it — the
calls of one assistant turn, each answered once, through a pause, a restart and
a message instead of the answer. These do, on any OpenAI-compatible endpoint.

The prompts ask for two independent things on purpose, and the instructions are
blunt, so that a capable model calls both specialists at once and calls the
tools it is told to. A model that does not is reported as such, not as a runner
failure. Each test is a few model calls: seconds, and a few cents at most.
"""

# No `from __future__ import annotations`: builds real agents and tools.
import asyncio
import os
import time
from pathlib import Path
from typing import Any

import pytest
from pydantic import Field

from domyn.agents.agents import Agent
from domyn.agents.conversation_manager import SqliteConversationManager
from domyn.agents.core import (
    METADATA_BRANCH,
    METADATA_TOOL_BATCH,
    ApprovalType,
    BaseEvent,
    ExecutionEventType,
    FeedbackAction,
    Part,
)
from domyn.agents.llm.base import LLMMessage, LLMResponse
from domyn.agents.llm.vllm import VLLMProvider
from domyn.agents.planner_strategy import PlannerStrategyFactory
from domyn.agents.runner import Runner
from domyn.agents.tools.function_tool import FunctionTool

URL = os.environ.get("DOMYN_LIVE_LLM_URL")
MODEL = os.environ.get("DOMYN_LIVE_LLM_MODEL", "")
API_KEY = os.environ.get("DOMYN_LIVE_LLM_API_KEY")

pytestmark = pytest.mark.skipif(
    not (URL and MODEL),
    reason="set DOMYN_LIVE_LLM_URL and DOMYN_LIVE_LLM_MODEL to run the live tests",
)

APP = "parallel_handoffs_live"
USER = "user_1"
PLANNERS = ["tool_use", "tool_use_thought"]
# Long enough for the two tool calls of a branch to overlap visibly, short enough
# to keep a test in seconds.
TOOL_SECONDS = 2.0


class _Recorded(VLLMProvider):
    """The real endpoint, keeping every request it is sent."""

    # Any, so the list is the caller's and not a copy.
    requests: Any = Field(default_factory=list)

    async def generate(self, messages: list[LLMMessage], **kwargs: Any) -> LLMResponse:
        self.requests.append(list(messages))
        return await super().generate(messages, **kwargs)


def _llm() -> _Recorded:
    return _Recorded(
        name="live",
        model_name=MODEL,
        url=URL or "",
        api_key=API_KEY,
        generation_params={"temperature": 0.0, "max_tokens": 2000},
    )


class _Log:
    """When each tool started and ended, by name."""

    def __init__(self) -> None:
        self.started: dict[str, float] = {}
        self.ended: dict[str, float] = {}

    def overlap(self, a: str, b: str) -> bool:
        return self.started[a] < self.ended[b] and self.started[b] < self.ended[a]


def _named(func: Any, name: str) -> Any:
    """The function under ``name``: the name a model calls a tool by is its
    function's, and each specialist's tool needs its own."""
    func.__name__ = func.__qualname__ = name
    return func


def _lookup(log: _Log, label: str) -> FunctionTool:
    async def lookup(topic: str) -> str:
        """Look a topic up in the archive. Takes a moment.

        Args:
            topic: what to look up.
        """
        log.started[label] = time.monotonic()
        await asyncio.sleep(TOOL_SECONDS)
        log.ended[label] = time.monotonic()
        return f"Archive entry on {topic}: {label} checked it."

    return FunctionTool(func=_named(lookup, f"lookup_{label.lower()}"))


def _save_note(saved: list[str], label: str) -> FunctionTool:
    async def save_note(text: str) -> str:
        """Save a note for the user.

        Args:
            text: the note.
        """
        saved.append(f"{label}: {text}")
        return "Saved."

    return FunctionTool(
        func=_named(save_note, f"save_note_{label.lower()}"),
        approval_type=ApprovalType.USER_CONFIRM,
    )


def _planner(planner: str, **options: Any) -> Any:
    return PlannerStrategyFactory.from_config({"type": planner, **options}, save_in_registry=False)


def _specialist(
    name: str, subject: str, planner: str, tools: list[FunctionTool], must: str
) -> tuple[Agent, _Recorded]:
    llm = _llm()
    agent = Agent(
        name=name,
        description=f"Answers questions about {subject}. Give it the question.",
        instruction=(
            f"You answer questions about {subject}. {must} Then answer in one short sentence."
        ),
        llm_provider=llm,
        planner=_planner(planner),
        tools=tools,
    )
    return agent, llm


def _team(
    planner: str, historian_tools: list[FunctionTool], geographer_tools: list[FunctionTool]
) -> tuple[Agent, _Recorded, _Recorded, _Recorded]:
    must = (
        f"You MUST call your tool {historian_tools[0].name} exactly once before answering."
        if historian_tools
        else ""
    )
    historian, historian_llm = _specialist(
        "Historian", "the history of cities", planner, historian_tools, must
    )
    must = (
        f"You MUST call your tool {geographer_tools[0].name} exactly once before answering."
        if geographer_tools
        else ""
    )
    geographer, geographer_llm = _specialist(
        "Geographer", "the geography of cities", planner, geographer_tools, must
    )
    llm = _llm()
    root = Agent(
        name="Coordinator",
        description="Talks to the user and delegates to specialists.",
        instruction=(
            "You coordinate specialists and never answer from your own knowledge. "
            "When a request has parts for different specialists, call ALL of them "
            "in the same response, in parallel, then combine their answers."
        ),
        llm_provider=llm,
        planner=_planner(planner, parallel_handoffs=True),
        sub_agents=[historian, geographer],
    )
    return root, llm, historian_llm, geographer_llm


QUESTION = "Tell me one historical fact about Lisbon and one fact about Lisbon's geography."


def _ask(text: str = QUESTION) -> BaseEvent:
    return BaseEvent(
        event_type=ExecutionEventType.USER_INPUT, author="user", content=[Part(text=text)]
    )


def _approve(tool: str) -> BaseEvent:
    return BaseEvent(
        event_type=ExecutionEventType.TOOL_FEEDBACK,
        author="user",
        action=FeedbackAction(
            name=tool, parameters=[], approval_type=ApprovalType.USER_CONFIRM, approve=True
        ),
    )


async def _run(
    runner: Runner, root: Agent, conversation: str, user_input: BaseEvent | None = None
) -> list[BaseEvent]:
    return [
        event
        async for event in runner.run(
            application_name=APP,
            user_id=USER,
            conversation_id=conversation,
            root=root,
            user_input=user_input or _ask(),
        )
    ]


def _assert_no_error(events: list[BaseEvent]) -> None:
    errors = [
        e
        for e in events
        if e.event_type
        in (ExecutionEventType.ERROR_OCCURRED, ExecutionEventType.LLM_PROVIDER_ERROR)
        or e.error_code
    ]
    assert not errors, [(e.author, e.error_code, e.error_message) for e in errors]


def _assert_called_together(events: list[BaseEvent]) -> None:
    together = [
        e.action.name  # type: ignore[union-attr]
        for e in events
        if e.event_type == ExecutionEventType.AGENT_CALL and e.metadata.get(METADATA_TOOL_BATCH)
    ]
    assert sorted(together) == ["Geographer", "Historian"], (
        f"the model did not call both specialists in one response: {together}"
    )


def _assert_paired(requests: list[list[LLMMessage]]) -> None:
    """Every call in every request is answered exactly once, right after its turn —
    what a strict endpoint checks before it accepts a request."""
    for request in requests:
        for position, message in enumerate(request):
            if message.role != "assistant" or not message.tool_calls:
                continue
            calls = [tc.id for tc in message.tool_calls]
            answers = []
            for after in request[position + 1 :]:
                if after.role != "tool":
                    break
                answers.append(after.tool_call_id)
            assert sorted(answers) == sorted(calls), (calls, answers)


def _answered(events: list[BaseEvent]) -> str:
    answers = [
        e
        for e in events
        if e.event_type == ExecutionEventType.RESPONSE
        and e.author == "Coordinator"
        and not e.is_partial
    ]
    assert answers, "the coordinator did not answer; the run ended with: " + repr(
        [(e.event_type.value, e.author, getattr(e.action, "name", None)) for e in events[-6:]]
    )
    return " ".join(p.text for p in answers[-1].content if p.text)


class TestLive:
    @pytest.mark.parametrize("planner", PLANNERS)
    @pytest.mark.asyncio
    async def test_two_specialists_work_at_the_same_time(self, planner: str) -> None:
        log = _Log()
        root, llm, historian, geographer = _team(
            planner, [_lookup(log, "Historian")], [_lookup(log, "Geographer")]
        )

        events = await _run(Runner(), root, f"live_together_{planner}")

        _assert_no_error(events)
        _assert_called_together(events)
        assert log.overlap("Historian", "Geographer"), "the two lookups took turns"
        assert "Lisbon" in _answered(events)
        # The coordinator is asked twice: before its specialists and after both.
        assert len(llm.requests) == 2
        _assert_paired(llm.requests + historian.requests + geographer.requests)

    @pytest.mark.parametrize("planner", PLANNERS)
    @pytest.mark.asyncio
    async def test_one_waits_for_approval_and_resumes_after_a_restart(
        self, planner: str, tmp_path: Path
    ) -> None:
        saved: list[str] = []
        root, llm, historian, geographer = _team(planner, [_save_note(saved, "Historian")], [])
        path = str(tmp_path / "conversations.db")
        conversation = f"live_restart_{planner}"

        first = await _run(
            Runner(conversation_manager=SqliteConversationManager(path)), root, conversation
        )

        _assert_no_error(first)
        _assert_called_together(first)
        question = first[-1]
        assert question.event_type == ExecutionEventType.TOOL_NEEDS_FEEDBACK
        assert question.metadata[METADATA_BRANCH] == "Historian"
        assert not saved
        # The other one finished while this one waited.
        assert any(
            e.event_type == ExecutionEventType.AGENT_END and e.author == "Geographer" for e in first
        )

        second = await _run(
            Runner(conversation_manager=SqliteConversationManager(path)),
            root,
            conversation,
            _approve("save_note_historian"),
        )

        _assert_no_error(second)
        assert len(saved) == 1
        assert "Lisbon" in _answered(second)
        _assert_paired(llm.requests + historian.requests + geographer.requests)

    @pytest.mark.parametrize("planner", PLANNERS)
    @pytest.mark.asyncio
    async def test_two_wait_and_are_asked_one_at_a_time(self, planner: str) -> None:
        saved: list[str] = []
        root, llm, historian, geographer = _team(
            planner, [_save_note(saved, "Historian")], [_save_note(saved, "Geographer")]
        )
        runner = Runner()
        conversation = f"live_two_{planner}"

        first = await _run(runner, root, conversation)
        _assert_no_error(first)
        _assert_called_together(first)
        asked_first = first[-1].metadata[METADATA_BRANCH]
        asked_second = "Geographer" if asked_first == "Historian" else "Historian"

        second = await _run(
            runner, root, conversation, _approve(f"save_note_{asked_first.lower()}")
        )
        _assert_no_error(second)
        assert second[-1].event_type == ExecutionEventType.TOOL_NEEDS_FEEDBACK
        assert second[-1].metadata[METADATA_BRANCH] == asked_second
        assert len(saved) == 1

        third = await _run(
            runner, root, conversation, _approve(f"save_note_{asked_second.lower()}")
        )
        _assert_no_error(third)
        assert len(saved) == 2
        assert "Lisbon" in _answered(third)
        _assert_paired(llm.requests + historian.requests + geographer.requests)

    @pytest.mark.parametrize("planner", PLANNERS)
    @pytest.mark.asyncio
    async def test_a_message_instead_of_the_answer_reaches_the_one_waiting(
        self, planner: str
    ) -> None:
        saved: list[str] = []
        root, llm, historian, geographer = _team(planner, [_save_note(saved, "Historian")], [])
        runner = Runner()
        conversation = f"live_message_{planner}"
        first = await _run(runner, root, conversation)
        _assert_no_error(first)
        _assert_called_together(first)

        second = await _run(
            runner, root, conversation, _ask("Do not save anything, just answer the question.")
        )

        _assert_no_error(second)
        assert not saved
        assert "Lisbon" in _answered(second)
        # The real endpoint accepted every request: the waiting call has one result.
        _assert_paired(llm.requests + historian.requests + geographer.requests)
