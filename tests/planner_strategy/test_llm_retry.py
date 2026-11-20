"""A transient provider failure is asked again instead of ending the turn.

Measured on our gateway: a stream the peer closed half-way (``RemoteProtocolError``) ended the turn
with ``ERROR_OCCURRED``, and in an unattended run that is the whole task lost. The planner now
repeats the same request, up to ``max_llm_retries``, and continues with the answer.
"""

# No `from __future__ import annotations`: builds real tools.
from typing import Any

import httpx
import pytest
from pydantic import PrivateAttr

from domyn.agents import Agent, BaseEvent, ExecutionEventType, Part, Runner
from domyn.agents.llm import BaseLLMProvider, LLMResponse
from domyn.agents.llm import retry as retry_module
from domyn.agents.planner_strategy import PlannerStrategyFactory


def _flaky(error: Exception, failures: int) -> type[BaseLLMProvider]:
    """A provider that raises ``error`` for the first ``failures`` calls, then answers."""

    class _Provider(BaseLLMProvider):
        type: str = "mock"
        name: str = "mock"
        model_name: str = "mock"
        supported_structured_outputs: None = None
        calls: int = 0
        _seen: list = PrivateAttr(default_factory=list)

        async def generate(self, messages: Any, **kwargs: Any) -> LLMResponse:
            self.calls += 1
            self._seen.append(messages)
            if self.calls <= failures:
                raise error
            return LLMResponse(content="Done.", generated_tokens=1, execution_time=0.0)

        async def generate_streaming(self, messages: Any, **kwargs: Any) -> Any:
            self.calls += 1
            if self.calls <= failures:
                raise error
            yield LLMResponse(content="Done.", generated_tokens=1, execution_time=0.0)

    return _Provider


def _status_error(code: int, headers: dict[str, str] | None = None) -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "https://example.test/v1/chat/completions")
    return httpx.HTTPStatusError(
        f"{code}",
        request=request,
        response=httpx.Response(code, request=request, headers=headers or {}),
    )


def _agent(
    provider: BaseLLMProvider, planner: str, *, stream: bool = False, retries: int = 3
) -> Agent:
    return Agent(
        name="A",
        description="d",
        instruction="i",
        llm_provider=provider,
        stream=stream,
        planner=PlannerStrategyFactory.from_config(
            {"type": planner, "max_llm_retries": retries}, save_in_registry=False
        ),
        tools=[],
    )


async def _run(agent: Agent) -> list[BaseEvent]:
    return [
        event
        async for event in Runner().run(
            application_name="app",
            user_id="u",
            root=agent,
            user_input=BaseEvent(
                event_type=ExecutionEventType.USER_INPUT, author="user", content=[Part(text="go")]
            ),
            conversation_id="c1",
        )
    ]


@pytest.fixture(autouse=True)
def _no_waiting(monkeypatch: pytest.MonkeyPatch) -> None:
    """The waits are the point of ``retry.py``'s own tests; here they would only slow the suite."""
    monkeypatch.setattr(retry_module, "backoff_delay", lambda attempt, **_: 0.0)


class TestAskingAgain:
    @pytest.mark.parametrize("planner", ["tool_use", "tool_use_thought"])
    @pytest.mark.asyncio
    async def test_a_rate_limit_is_retried_and_the_turn_finishes(self, planner: str) -> None:
        provider = _flaky(_status_error(429), failures=2)()

        events = await _run(_agent(provider, planner))

        assert provider.calls == 3
        assert not any(e.event_type == ExecutionEventType.ERROR_OCCURRED for e in events)
        assert any(e.event_type == ExecutionEventType.RESPONSE for e in events), [
            e.event_type for e in events
        ]

    @pytest.mark.asyncio
    async def test_a_stream_that_breaks_is_asked_again(self) -> None:
        """The measured failure: the peer closes the stream half-way."""
        provider = _flaky(httpx.RemoteProtocolError("peer closed connection"), failures=1)()

        events = await _run(_agent(provider, "tool_use_thought", stream=True))

        assert provider.calls == 2
        assert not any(e.event_type == ExecutionEventType.ERROR_OCCURRED for e in events)

    @pytest.mark.asyncio
    async def test_the_providers_retry_after_is_read(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Whatever the provider asked for is what the planner waits (up to the cap)."""
        import asyncio

        waited: list[float] = []

        async def _record(seconds: float) -> None:
            waited.append(seconds)

        monkeypatch.setattr(asyncio, "sleep", _record)
        provider = _flaky(_status_error(503, {"Retry-After": "7"}), failures=1)()

        await _run(_agent(provider, "tool_use_thought"))

        assert 7.0 in waited


class TestWhenNotTo:
    @pytest.mark.parametrize("planner", ["tool_use", "tool_use_thought"])
    @pytest.mark.asyncio
    async def test_a_refused_request_is_not_retried(self, planner: str) -> None:
        provider = _flaky(_status_error(400), failures=1)()

        events = await _run(_agent(provider, planner))

        assert provider.calls == 1
        assert any(e.event_type == ExecutionEventType.ERROR_OCCURRED for e in events)

    @pytest.mark.parametrize("planner", ["tool_use", "tool_use_thought"])
    @pytest.mark.asyncio
    async def test_the_budget_is_respected_and_then_the_turn_ends(self, planner: str) -> None:
        provider = _flaky(_status_error(502), failures=99)()

        events = await _run(_agent(provider, planner, retries=2))

        assert provider.calls == 3, "one call and two retries"
        assert any(e.event_type == ExecutionEventType.ERROR_OCCURRED for e in events)

    @pytest.mark.asyncio
    async def test_a_retry_does_not_burn_a_step(self) -> None:
        """The retry asks the same request again; a small step budget must still leave room.

        Without the iteration given back, two failures inside a two-step budget would end the turn
        on "Reached maximum iterations" before the answer could arrive.
        """
        agent = _agent(_flaky(_status_error(429), failures=2)(), "tool_use_thought", retries=3)
        agent.planner.max_iterations = 2

        events = await _run(agent)

        assert not any("Reached maximum iterations" in str(e.content) for e in events), (
            "the retry spent the step budget"
        )
