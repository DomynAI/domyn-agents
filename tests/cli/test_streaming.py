"""Tests for streaming, and for keeping reasoning out of the stream.

What the user should see is: the model's thinking as a block, then the answer
arriving word by word. Getting there needed three things that were each broken
or absent, so each has a test here.

The provider-level test uses ``respx`` against a real SSE body rather than a
mock generator: the bug it guards was in parsing and in *when* the accumulated
reasoning gets yielded, which a stubbed provider cannot reproduce.
"""

import json

import httpx
import pytest
import respx

from domyn.agents.llm import CustomLLMProvider

URL = "http://llm.test/v1/chat/completions"


def _sse(*deltas: dict[str, object]) -> str:
    lines = []
    for delta in deltas:
        payload = {"choices": [{"index": 0, "delta": delta, "finish_reason": None}]}
        lines.append(f"data: {json.dumps(payload)}\n\n")
    lines.append('data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n')
    lines.append("data: [DONE]\n\n")
    return "".join(lines)


def _provider() -> CustomLLMProvider:
    return CustomLLMProvider(model_name="m", url=URL, api_key="k")


async def _collect(body: str) -> list:
    with respx.mock:
        respx.post(URL).mock(return_value=httpx.Response(200, text=body))
        return [chunk async for chunk in _provider().generate_streaming(messages=[])]


class TestReasoningReachesTheConsumer:
    @pytest.mark.asyncio
    async def test_it_is_surfaced_even_without_tool_calls(self) -> None:
        """It used to be accumulated and then thrown away unless the model also
        called a tool — so a plain answer lost its reasoning entirely."""
        chunks = await _collect(
            _sse(
                {"reasoning_content": "The user greets me."},
                {"content": "Hi"},
                {"content": " there"},
            )
        )
        assert any(chunk.reasoning_content for chunk in chunks)

    @pytest.mark.asyncio
    async def test_it_arrives_before_the_answer(self) -> None:
        """Held to the end it would print underneath the answer it explains."""
        chunks = await _collect(_sse({"reasoning_content": "Thinking..."}, {"content": "Hi"}))
        first_reasoning = next(i for i, c in enumerate(chunks) if c.reasoning_content)
        first_content = next(i for i, c in enumerate(chunks) if c.content)
        assert first_reasoning < first_content

    @pytest.mark.asyncio
    async def test_it_is_not_mixed_into_the_streamed_text(self) -> None:
        """Reasoning in the answer stream would print it as if it were the
        answer, which is exactly what the two-level transcript avoids."""
        chunks = await _collect(_sse({"reasoning_content": "secret thinking"}, {"content": "Hi"}))
        assert "".join(c.content for c in chunks) == "Hi"

    @pytest.mark.asyncio
    async def test_it_is_sent_once_however_many_deltas_carry_it(self) -> None:
        chunks = await _collect(
            _sse(
                {"reasoning_content": "first "},
                {"reasoning_content": "second"},
                {"content": "Hi"},
                {"content": " there"},
            )
        )
        with_reasoning = [c for c in chunks if c.reasoning_content]
        assert len(with_reasoning) == 1
        assert with_reasoning[0].reasoning_content == "first second"

    @pytest.mark.asyncio
    async def test_a_plain_stream_is_unaffected(self) -> None:
        chunks = await _collect(_sse({"content": "Hi"}, {"content": " there"}))
        assert not any(c.reasoning_content for c in chunks)
        assert "".join(c.content for c in chunks) == "Hi there"


def _agent_from(provider: object, agent: object) -> object:
    """Build a real Agent the way the default config now does: project the
    wizard's typed fields onto a ConfigurationV1 dict, then load it."""
    from domyn.agents.interfaces.cli.config import CompactionConfig, configuration_v1_dict_from
    from domyn.agents.loader import ConfigurationV1

    config_dict = configuration_v1_dict_from(provider, agent, CompactionConfig())
    return ConfigurationV1.from_dict(config_dict).agents[0]


class TestTheCliTurnsStreamingOn:
    def test_it_is_the_default(self) -> None:
        """Waiting in silence and then getting a paragraph reads as slow even
        when it is not."""
        from domyn.agents.interfaces.cli import config as cfg

        assert cfg.AgentConfig().stream is True

    def test_the_agent_is_built_with_it(self) -> None:
        from domyn.agents.interfaces.cli import config as cfg

        provider = cfg.ProviderConfig(type="custom", model_name="m", url="http://x")
        assert _agent_from(provider, cfg.AgentConfig()).stream is True

    def test_it_can_be_turned_off(self) -> None:
        """A gateway that does not implement SSE has to be survivable."""
        from domyn.agents.interfaces.cli import config as cfg

        provider = cfg.ProviderConfig(type="custom", model_name="m", url="http://x")
        assert _agent_from(provider, cfg.AgentConfig(stream=False)).stream is False

    def test_it_is_reported_by_slash_config(self) -> None:
        from domyn.agents.interfaces.cli import config as cfg

        provider = cfg.ProviderConfig(type="custom", model_name="m", url="http://x")
        config_dict = cfg.configuration_v1_dict_from(
            provider, cfg.AgentConfig(), cfg.CompactionConfig()
        )
        agents = next(
            s for s in cfg.describe_configuration_v1(config_dict).sections if s.title == "Agents"
        )
        assert dict(agents.items[0].fields)["stream"] == "True"


class TestTheRendererSeparatesThemOnScreen:
    """The point of all of the above: thinking as a block, answer streamed."""

    def _render(self, events: list) -> str:
        import io

        from rich.console import Console

        from domyn.agents.interfaces.cli import Theme
        from domyn.agents.interfaces.cli.renderer import EventRenderer

        buffer = io.StringIO()
        renderer = EventRenderer(Console(file=buffer, width=100), Theme.build())
        renderer.begin_turn()
        for event in events:
            renderer.render(event)
        renderer.finish()
        return buffer.getvalue()

    def _events(self) -> list:
        from domyn.agents import BaseEvent, ExecutionEventType, Part

        token = ExecutionEventType.RESPONSE_TOKEN
        return [
            BaseEvent(event_type=token, author="a", content=[Part(thought="I should greet.")]),
            BaseEvent(event_type=token, author="a", content=[Part(text="Hi")]),
            BaseEvent(event_type=token, author="a", content=[Part(text=" there")]),
            BaseEvent(
                event_type=ExecutionEventType.RESPONSE, author="a", content=[Part(text="Hi there")]
            ),
        ]

    def test_the_thought_is_indented_and_the_answer_is_not(self) -> None:
        lines = [ln for ln in self._render(self._events()).splitlines() if ln.strip()]
        thought = next(ln for ln in lines if "I should greet." in ln)
        answer = next(ln for ln in lines if "Hi there" in ln)
        assert thought.startswith("  ")
        assert answer.startswith("⏺")

    def test_the_thought_comes_first(self) -> None:
        output = self._render(self._events())
        assert output.index("I should greet.") < output.index("Hi there")

    def test_the_streamed_answer_is_printed_once(self) -> None:
        assert self._render(self._events()).count("Hi there") == 1

    def test_a_mid_stream_thought_does_not_land_inside_the_answer(self) -> None:
        """A thought arriving after tokens have started must close the stream
        first, or it is appended to a half-printed sentence."""
        from domyn.agents import BaseEvent, ExecutionEventType, Part

        token = ExecutionEventType.RESPONSE_TOKEN
        output = self._render(
            [
                BaseEvent(event_type=token, author="a", content=[Part(text="Partial")]),
                BaseEvent(event_type=token, author="a", content=[Part(thought="Hmm.")]),
            ]
        )
        assert "PartialHmm." not in output
        assert "  Hmm." in output
