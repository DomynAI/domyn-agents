"""`max_tokens_share`: the reply's ceiling never claims more than its share of the window.

A model whose longest reply is its whole window is declared with `max_tokens` equal
to the window (the leaderboard's Flint: 262,144 and 262,144). Sent as it is, every
request asks for the prompt plus the whole window, which a server checking prompt
plus output refuses whatever the prompt. Lowered when the provider is built if the
window is declared, and again whenever the window is assigned later.
"""

import contextlib
import json
import logging
from typing import Any

import httpx
import pytest
import respx

from domyn.agents.llm import LLMMessage
from domyn.agents.llm import base as llm_base
from domyn.agents.llm.anthropic import AnthropicProvider
from domyn.agents.llm.custom import CustomLLMProvider
from domyn.agents.llm.openai import OpenAIProvider
from domyn.agents.llm.vllm import VLLMProvider

WINDOW = 262_144


def _vllm(**fields: Any) -> VLLMProvider:
    return VLLMProvider(
        **{"url": "http://vllm.test/v1/chat/completions", "model_name": "m"} | fields
    )


class TestAtConstruction:
    def test_a_declared_window_halves_a_ceiling_as_large_as_it(self) -> None:
        provider = _vllm(context_window=WINDOW, generation_params={"max_tokens": WINDOW})
        assert provider.generation_params["max_tokens"] == WINDOW // 2
        assert provider.max_reply_tokens == WINDOW // 2

    def test_a_ceiling_within_the_share_is_kept(self) -> None:
        provider = _vllm(context_window=WINDOW, generation_params={"max_tokens": 16_000})
        assert provider.max_reply_tokens == 16_000

    def test_exactly_the_share_is_kept(self) -> None:
        provider = _vllm(context_window=WINDOW, generation_params={"max_tokens": WINDOW // 2})
        assert provider.max_reply_tokens == WINDOW // 2

    def test_no_window_lowers_nothing(self) -> None:
        provider = _vllm(generation_params={"max_tokens": WINDOW})
        assert provider.max_reply_tokens == WINDOW

    def test_body_too(self) -> None:
        provider = CustomLLMProvider(
            url="http://custom.test/v1/chat/completions",
            model_name="m",
            context_window=WINDOW,
            body={"max_tokens": WINDOW},
        )
        assert provider.body["max_tokens"] == WINDOW // 2

    def test_every_spelling(self) -> None:
        params = {
            "max_tokens": WINDOW,
            "max_completion_tokens": WINDOW,
            "max_output_tokens": WINDOW,
        }
        provider = _vllm(context_window=WINDOW, generation_params=params)
        assert all(provider.generation_params[k] == WINDOW // 2 for k in params)

    def test_other_values_are_left_alone(self) -> None:
        provider = _vllm(
            context_window=WINDOW, generation_params={"max_tokens": True, "temperature": 2}
        )
        assert provider.generation_params == {"max_tokens": True, "temperature": 2}

    def test_the_dicts_given_are_not_changed(self) -> None:
        params = {"max_tokens": WINDOW}
        body = {"max_tokens": WINDOW}
        _vllm(context_window=WINDOW, generation_params=params)
        CustomLLMProvider(
            url="http://custom.test/v1/chat/completions",
            model_name="m",
            context_window=WINDOW,
            body=body,
        )
        assert params == body == {"max_tokens": WINDOW}

    def test_the_share_is_configurable(self) -> None:
        provider = _vllm(
            context_window=WINDOW, generation_params={"max_tokens": WINDOW}, max_tokens_share=0.25
        )
        assert provider.max_reply_tokens == WINDOW // 4

    @pytest.mark.parametrize("share", [0, -0.5, 1.5])
    def test_the_share_is_a_fraction_of_the_window(self, share: float) -> None:
        with pytest.raises(ValueError):
            _vllm(max_tokens_share=share)


class TestWhenTheWindowArrivesLater:
    def test_assigning_the_window_lowers_it(self) -> None:
        provider = _vllm(generation_params={"max_tokens": WINDOW})
        provider.context_window = WINDOW
        assert provider.max_reply_tokens == WINDOW // 2

    def test_a_smaller_window_lowers_it_again(self) -> None:
        provider = _vllm(context_window=WINDOW, generation_params={"max_tokens": WINDOW})
        provider.context_window = 100_000
        assert provider.max_reply_tokens == 50_000

    def test_clearing_the_window_changes_nothing(self) -> None:
        provider = _vllm(context_window=WINDOW, generation_params={"max_tokens": WINDOW})
        provider.context_window = None
        assert provider.max_reply_tokens == WINDOW // 2

    def test_other_assignments_change_nothing(self) -> None:
        provider = _vllm(generation_params={"max_tokens": WINDOW})
        provider.vision = True
        assert provider.max_reply_tokens == WINDOW


class TestTheLog:
    def test_once(self, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(llm_base, "_lowered_max_tokens", set())
        with caplog.at_level(logging.INFO, logger="domyn.agents"):
            for _ in range(3):
                _vllm(context_window=WINDOW, generation_params={"max_tokens": WINDOW})
        lines = [r.message for r in caplog.records if "lowered from" in r.message]
        assert len(lines) == 1
        assert f"{WINDOW} to {WINDOW // 2}" in lines[0]


class TestOnTheWire:
    """What a provider sends, captured before the (failed) answer."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("provider", "key"),
        [
            (_vllm(generation_params={"max_tokens": WINDOW}), "max_tokens"),
            (
                OpenAIProvider(
                    model_name="gpt-4.1", api_key="k", generation_params={"max_tokens": WINDOW}
                ),
                "max_output_tokens",
            ),
            (
                AnthropicProvider(
                    model_name="m", api_key="k", generation_params={"max_tokens": WINDOW}
                ),
                "max_tokens",
            ),
        ],
        ids=["vllm", "openai", "anthropic"],
    )
    async def test_a_window_set_later_reaches_the_request(self, provider: Any, key: str) -> None:
        provider.context_window = WINDOW
        bodies: list[dict[str, Any]] = []

        def capture(request: httpx.Request) -> httpx.Response:
            bodies.append(json.loads(request.content))
            return httpx.Response(500, json={"error": "stop here"})

        with respx.mock(assert_all_called=False) as router:
            router.route().mock(side_effect=capture)
            with contextlib.suppress(Exception):
                await provider.generate([LLMMessage(role="user", content="hi")])

        assert bodies[-1][key] == WINDOW // 2
