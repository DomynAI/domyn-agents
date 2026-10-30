"""Tests for telling "cut off" apart from "wrong".

The failure that cost a session: an agent asked for an HTML dashboard, the
generation hit `max_tokens: 2000` part-way through the `content` argument, and
the truncated JSON surfaced as "Malformed tool-call arguments". Everyone —
including the model, which was told to escape more carefully — went looking for
a mistake that had never been made, and all three retries were cut off at
exactly the same place.

The provider had been saying so the whole time. `finish_reason: "length"` was in
every response and no provider read it.
"""

# No `from __future__ import annotations`: builds real tools.
import json
from typing import Any

import httpx
import pytest
import respx

from domyn.agents.llm import CustomLLMProvider, LLMResponse
from domyn.agents.planner_strategy.tool_use import malformed_args_feedback

URL = "http://llm.test/v1/chat/completions"


class TestTheResponseCarriesWhyItStopped:
    def test_truncated_is_recognised(self) -> None:
        assert LLMResponse(
            content="", generated_tokens=1, execution_time=0.0, finish_reason="length"
        ).truncated

    @pytest.mark.parametrize("reason", ["stop", "tool_calls", None])
    def test_a_normal_finish_is_not(self, reason: str | None) -> None:
        assert not LLMResponse(
            content="", generated_tokens=1, execution_time=0.0, finish_reason=reason
        ).truncated


class TestProvidersReportIt:
    @pytest.mark.asyncio
    async def test_the_non_streaming_path(self) -> None:
        body = {
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "half a sen"},
                    "finish_reason": "length",
                }
            ],
            "usage": {},
        }
        with respx.mock:
            respx.post(URL).mock(return_value=httpx.Response(200, json=body))
            provider = CustomLLMProvider(model_name="m", url=URL, api_key="k")
            response = await provider.generate(messages=[])
        assert response.truncated

    @pytest.mark.asyncio
    async def test_the_streaming_path(self) -> None:
        """Where it matters most: a tool call cut off mid-argument."""
        args = '{"path":"d.html","content":"<html> and then it stops'
        sse = (
            "data: "
            + json.dumps(
                {
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": "c1",
                                        "type": "function",
                                        "function": {"name": "write_file", "arguments": args},
                                    }
                                ]
                            },
                            "finish_reason": None,
                        }
                    ]
                }
            )
            + "\n\n"
            'data: {"choices":[{"index":0,"delta":{},"finish_reason":"length"}]}\n\n'
            "data: [DONE]\n\n"
        )
        with respx.mock:
            respx.post(URL).mock(return_value=httpx.Response(200, text=sse))
            provider = CustomLLMProvider(model_name="m", url=URL, api_key="k")
            chunks = [c async for c in provider.generate_streaming(messages=[])]

        with_calls = [c for c in chunks if c.tool_calls]
        assert with_calls, "the partial tool call should still arrive"
        assert with_calls[-1].truncated, "and it should say it was cut off"


class TestTheAdviceIsOpposite:
    """Two failures, two remedies. Giving the wrong one guarantees three wasted
    retries."""

    def test_truncation_says_send_less(self) -> None:
        content = str(malformed_args_feedback([], truncated=True).content).lower()
        assert "cut off" in content
        assert "split" in content

    def test_truncation_does_not_blame_the_escaping(self) -> None:
        """Nothing was wrong with what the model wrote — there was too much of
        it. Telling it to escape better sends it back into the same wall."""
        content = str(malformed_args_feedback([], truncated=True).content).lower()
        assert "escape" not in content

    def test_truncation_says_not_to_repeat_the_same_call(self) -> None:
        content = str(malformed_args_feedback([], truncated=True).content).lower()
        assert "same place" in content or "do not retry" in content

    def test_a_real_syntax_error_still_gets_the_escaping_advice(self) -> None:
        content = str(malformed_args_feedback(["write_file: Expecting ','"])).lower()
        assert "escaped" in content
        assert "cut off" not in content


class TestTheCliRaisesTheCeiling:
    def test_the_default_is_no_longer_two_thousand(self) -> None:
        """2000 is fine for a chat reply and passed easily by a tool argument
        carrying a file."""
        from domyn.agents.interfaces.cli import config as cfg

        assert cfg.ProviderConfig().max_tokens >= 8000

    @pytest.mark.parametrize("kind", ["vllm", "custom", "openai"])
    def test_it_reaches_every_provider(self, kind: str) -> None:
        """The providers disagree on where generation settings live: most take
        `generation_params`, `custom` merges a free-form `body`. A ceiling that
        only lands on some of them is a ceiling nobody can rely on."""
        from domyn.agents.interfaces.cli import config as cfg
        from domyn.agents.llm import LLMProviderFactory

        provider_config = cfg.ProviderConfig(
            type=kind, model_name="m", url="http://x", max_tokens=12345
        )
        config_dict = cfg.configuration_v1_dict_from(
            provider_config, cfg.AgentConfig(), cfg.CompactionConfig()
        )
        provider = LLMProviderFactory.from_config(config_dict["llm_providers"][0])
        settings = getattr(provider, "generation_params", None) or getattr(provider, "body", {})
        assert settings["max_tokens"] == 12345

    def test_it_is_visible(self) -> None:
        """An invisible ceiling is one nobody thinks to raise."""
        from domyn.agents.interfaces.cli import config as cfg

        provider = cfg.ProviderConfig(type="openai", model_name="m")
        config_dict = cfg.configuration_v1_dict_from(
            provider, cfg.AgentConfig(), cfg.CompactionConfig()
        )
        providers = cfg.describe_configuration_v1(config_dict).sections[0]
        assert "max tokens" in dict(providers.items[0].fields)


@pytest.mark.parametrize("planner", ["tool_use", "tool_use_thought"])
@pytest.mark.asyncio
async def test_the_planner_passes_the_right_advice(planner: str) -> None:
    """End to end: a truncated tool call must produce the "send less" message,
    not the "escape better" one."""
    from pydantic import PrivateAttr

    from domyn.agents import Agent, BaseEvent, ExecutionEventType, Part, Runner
    from domyn.agents.llm import BaseLLMProvider, LLMToolCall, ToolCallFunction
    from domyn.agents.planner_strategy import PlannerStrategyFactory
    from domyn.agents.tools import FunctionTool

    class P(BaseLLMProvider):
        type: str = "m"
        name: str = "m"
        model_name: str = "m"
        supported_structured_outputs: None = None
        _n: int = PrivateAttr(default=0)
        _seen: list = PrivateAttr(default_factory=list)

        async def generate(self, messages: Any, **kwargs: Any) -> LLMResponse:
            self._n += 1
            self._seen.append([str(getattr(m, "content", "") or "") for m in messages])
            if self._n == 1:
                return LLMResponse(
                    content="",
                    generated_tokens=1,
                    execution_time=0.0,
                    finish_reason="length",
                    tool_calls=[
                        LLMToolCall(
                            id="c1",
                            function=ToolCallFunction(
                                name="write_file", arguments='{"path":"a","content":"cut off'
                            ),
                        )
                    ],
                )
            return LLMResponse(content="Done.", generated_tokens=1, execution_time=0.0)

        async def generate_streaming(self, messages: Any, **kwargs: Any) -> Any:
            yield

    def write_file(path: str, content: str) -> str:
        """Write.

        Args:
            path: Where.
            content: What.
        """
        return "ok"

    provider = P()
    agent = Agent(
        name="A",
        description="d",
        instruction="i",
        llm_provider=provider,
        planner=PlannerStrategyFactory.from_config({"type": planner}, save_in_registry=False),
        tools=[FunctionTool(func=write_file)],
    )
    async for _ in Runner().run(
        application_name="a",
        user_id="u",
        root=agent,
        user_input=BaseEvent(
            event_type=ExecutionEventType.USER_INPUT, author="user", content=[Part(text="go")]
        ),
        conversation_id="c1",
    ):
        pass

    assert len(provider._seen) >= 2, "it should have retried"
    second = " ".join(provider._seen[1]).lower()
    assert "cut off" in second
    assert "must be escaped" not in second
