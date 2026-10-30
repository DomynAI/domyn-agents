"""Where a provider's context window comes from when nobody declares it.

The response shapes are the ones the servers send, recorded live: vLLM 0.26's
``/v1/models`` (``max_model_len``), LiteLLM 1.104's (``max_input_tokens``),
OpenRouter's (``top_provider.context_length``), Anthropic's ``/v1/models/{id}``.
"""

from types import SimpleNamespace

import httpx
import pytest
import respx

from domyn.agents.llm import (
    AnthropicProvider,
    CerebrasProvider,
    CustomLLMProvider,
    OpenAIProvider,
    OpenRouterProvider,
    VLLMProvider,
)
from domyn.agents.llm import context_window as cw

VLLM_URL = "http://vllm.test/v1/chat/completions"
VLLM_MODELS = "http://vllm.test/v1/models"


def _vllm(**kwargs: object) -> VLLMProvider:
    return VLLMProvider(model_name="Qwen/Qwen3-VL-32B-Instruct", url=VLLM_URL, **kwargs)  # type: ignore[arg-type]


def _listing(*entries: dict) -> httpx.Response:
    return httpx.Response(200, json={"object": "list", "data": list(entries)})


class TestWhatAModelsEntrySays:
    @pytest.mark.parametrize(
        ("entry", "window"),
        [
            ({"id": "m", "max_model_len": 32768, "owned_by": "vllm"}, 32768),
            ({"id": "m", "max_input_tokens": 65536, "max_output_tokens": 65536}, 65536),
            (
                {"id": "m", "context_length": 131072, "top_provider": {"context_length": 40960}},
                40960,
            ),
            ({"id": "m", "context_length": 131072}, 131072),
            (
                {"id": "m", "max_model_len": 400_000, "max_input_tokens": 272_000},
                272_000,
            ),
            ({"id": "m", "object": "model", "owned_by": "vllm"}, None),
            ({"id": "m", "max_model_len": None, "max_input_tokens": True}, None),
        ],
        ids=[
            "vllm",
            "litellm",
            "openrouter routed",
            "openrouter max",
            "both",
            "router strips it",
            "junk",
        ],
    )
    def test_each_server_spelling(self, entry: dict, window: int | None) -> None:
        assert cw.window_from_model_entry(entry) == window

    @pytest.mark.parametrize(
        ("endpoint", "listing"),
        [
            ("http://h/v1/chat/completions", "http://h/v1/models"),
            ("https://api.openai.com/v1/responses", "https://api.openai.com/v1/models"),
            (
                "https://openrouter.ai/api/v1/chat/completions",
                "https://openrouter.ai/api/v1/models",
            ),
            ("http://h/generate", None),
            (None, None),
        ],
    )
    def test_the_listing_is_found_beside_the_endpoint_never_guessed(
        self, endpoint: str | None, listing: str | None
    ) -> None:
        assert cw.models_url(endpoint) == listing


class TestResolvingTheWindow:
    @pytest.mark.asyncio
    @respx.mock
    async def test_vllm_reports_its_max_model_len(self) -> None:
        route = respx.get(VLLM_MODELS).mock(
            return_value=_listing({"id": "Qwen/Qwen3-VL-32B-Instruct", "max_model_len": 32768})
        )
        provider = _vllm(api_key="k")

        assert await cw.resolve_context_window(provider) == 32768
        assert provider.context_window == 32768
        assert route.calls.last.request.headers["authorization"] == "Bearer k"

    @pytest.mark.asyncio
    @respx.mock
    async def test_a_declared_window_is_never_overridden_nor_asked(self) -> None:
        route = respx.get(VLLM_MODELS).mock(return_value=_listing({"id": "x", "max_model_len": 1}))
        provider = _vllm(context_window=8192)

        assert await cw.resolve_context_window(provider) == 8192
        assert not route.called

    @pytest.mark.asyncio
    @respx.mock
    async def test_another_models_entry_is_not_this_ones(self) -> None:
        respx.get(VLLM_MODELS).mock(return_value=_listing({"id": "other", "max_model_len": 4096}))
        provider = _vllm()

        assert await cw.resolve_context_window(provider) is None
        assert provider.context_window is None

    @pytest.mark.asyncio
    @respx.mock
    @pytest.mark.parametrize(
        "response",
        [
            httpx.Response(403, json={"error": "route not allowed"}),
            httpx.Response(200, text="<html>"),
        ],
        ids=["forbidden", "not json"],
    )
    async def test_a_server_that_cannot_answer_leaves_it_unknown(
        self, response: httpx.Response
    ) -> None:
        respx.get(VLLM_MODELS).mock(return_value=response)
        provider = _vllm()

        assert await cw.resolve_context_window(provider) is None
        assert provider.context_window is None

    @pytest.mark.asyncio
    @respx.mock
    async def test_an_unreachable_server_does_not_fail_the_turn(self) -> None:
        respx.get(VLLM_MODELS).mock(side_effect=httpx.ConnectError("down"))
        assert await cw.resolve_context_window(_vllm()) is None

    @pytest.mark.asyncio
    @respx.mock
    async def test_one_question_per_endpoint_and_model_while_the_answer_is_fresh(self) -> None:
        """A host that builds its providers per request must not ask per turn —
        including when the answer was "nothing"."""
        route = respx.get(VLLM_MODELS).mock(return_value=_listing({"id": "other"}))
        for _ in range(3):
            assert await cw.resolve_context_window(_vllm()) is None
        assert route.call_count == 1

        route.mock(
            return_value=_listing({"id": "Qwen/Qwen3-VL-32B-Instruct", "max_model_len": 32768})
        )
        cw._answers.clear()  # what expiry does after CACHE_SECONDS
        assert await cw.resolve_context_window(_vllm()) == 32768

    @pytest.mark.asyncio
    @respx.mock
    @pytest.mark.parametrize(
        ("reply", "kept_for"),
        [
            (lambda request: _listing({"id": "other"}), cw.CACHE_SECONDS),
            (lambda request: httpx.Response(503), cw.RETRY_SECONDS),
            (httpx.ConnectError("down"), cw.RETRY_SECONDS),
        ],
        ids=["answered-without-a-window", "refused", "unreachable"],
    )
    async def test_a_failed_question_is_asked_again_sooner_than_an_answer(
        self, monkeypatch: pytest.MonkeyPatch, reply: object, kept_for: float
    ) -> None:
        """An answer, "no window" included, holds for five minutes; a failure says
        nothing about the window, so it is asked again after one."""
        now = [1_000.0]
        monkeypatch.setattr(cw, "time", SimpleNamespace(monotonic=lambda: now[0]))
        route = respx.get(VLLM_MODELS).mock(side_effect=reply)

        await cw.resolve_context_window(_vllm())
        now[0] += kept_for - 1
        await cw.resolve_context_window(_vllm())
        assert route.call_count == 1, "asked again before the answer expired"
        now[0] += 2
        await cw.resolve_context_window(_vllm())
        assert route.call_count == 2, "not asked again once it expired"

    @pytest.mark.asyncio
    @respx.mock
    async def test_custom_endpoints_ask_with_their_own_headers(self) -> None:
        route = respx.get(VLLM_MODELS).mock(
            return_value=_listing({"id": "m", "max_input_tokens": 65536})
        )
        provider = CustomLLMProvider(model_name="m", url=VLLM_URL, headers={"X-Key": "secret"})

        assert await cw.resolve_context_window(provider) == 65536
        assert route.calls.last.request.headers["x-key"] == "secret"

    @pytest.mark.asyncio
    @respx.mock
    async def test_openrouter_reports_the_window_requests_are_routed_to(self) -> None:
        respx.get("https://openrouter.ai/api/v1/models").mock(
            return_value=_listing(
                {
                    "id": "qwen/qwen3-32b",
                    "context_length": 131072,
                    "top_provider": {"context_length": 40960},
                }
            )
        )
        provider = OpenRouterProvider(model_name="qwen/qwen3-32b", api_key="k")

        assert await cw.resolve_context_window(provider) == 40960


class TestAWindowStatedInARejection:
    """Kept per endpoint and model for the life of the process: the provider's own word,
    so it caps what anything else says."""

    @pytest.mark.asyncio
    @respx.mock
    @pytest.mark.parametrize(
        ("declared", "reported", "kept"),
        [(None, 131072, 40960), (150_000, None, 40960), (None, 32768, 32768), (8192, None, 8192)],
        ids=[
            "reported-too-large",
            "declared-too-large",
            "reported-smaller",
            "declared-smaller-is-a-choice",
        ],
    )
    async def test_it_caps_what_the_provider_declares_or_reports(
        self, declared: int | None, reported: int | None, kept: int
    ) -> None:
        respx.get(VLLM_MODELS).mock(
            return_value=_listing({"id": "Qwen/Qwen3-VL-32B-Instruct", "max_model_len": reported})
        )
        cw.remember_stated_window(_vllm(), 40960)

        assert await cw.resolve_context_window(_vllm(context_window=declared)) == kept

    @pytest.mark.asyncio
    @respx.mock
    async def test_it_belongs_to_one_endpoint_and_model(self) -> None:
        respx.get(VLLM_MODELS).mock(return_value=_listing())
        respx.get("http://other.test/v1/models").mock(return_value=_listing())
        cw.remember_stated_window(_vllm(), 40960)

        other_model = VLLMProvider(model_name="Qwen/Qwen3-32B", url=VLLM_URL)
        other_endpoint = VLLMProvider(
            model_name="Qwen/Qwen3-VL-32B-Instruct", url="http://other.test/v1/chat/completions"
        )
        assert await cw.resolve_context_window(other_model) is None
        assert await cw.resolve_context_window(other_endpoint) is None

    @pytest.mark.asyncio
    @respx.mock
    async def test_a_larger_one_stated_later_does_not_raise_it(self) -> None:
        """Behind a gateway with several deployments, a request too long for the
        smallest is refused on that one too."""
        respx.get(VLLM_MODELS).mock(return_value=_listing())
        cw.remember_stated_window(_vllm(), 32768)
        cw.remember_stated_window(_vllm(), 40960)

        assert await cw.resolve_context_window(_vllm()) == 32768


class TestHostedApisThatCannotBeAsked:
    """Against a catalogue of the tests' own: the shipped one is regenerated by every
    release, and a test pinned to its numbers would fail the release one of them changes."""

    @pytest.fixture(autouse=True)
    def catalogue(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            cw,
            "_catalogue",
            lambda: {"openai": {"gpt-4o": 128_000}, "anthropic": {"claude-haiku-4-5": 200_000}},
        )

    @pytest.mark.asyncio
    @respx.mock
    async def test_openai_reports_nothing_and_the_catalogue_answers(self) -> None:
        respx.get("https://api.openai.com/v1/models").mock(
            return_value=_listing({"id": "gpt-4o", "object": "model", "owned_by": "system"})
        )
        provider = OpenAIProvider(model_name="gpt-4o", api_key="k")

        assert await cw.resolve_context_window(provider) == 128_000
        assert provider.context_window == 128_000

    def test_a_dated_snapshot_has_its_aliases_window(self) -> None:
        assert cw.catalogue_window("openai", "gpt-4o-2024-08-06") == 128_000

    @pytest.mark.asyncio
    @respx.mock
    async def test_anthropic_answers_per_model(self) -> None:
        route = respx.get("https://api.anthropic.com/v1/models/claude-haiku-4-5").mock(
            return_value=httpx.Response(
                200, json={"id": "claude-haiku-4-5", "max_input_tokens": 200_000}
            )
        )
        provider = AnthropicProvider(model_name="claude-haiku-4-5", api_key="k")

        assert await cw.resolve_context_window(provider) == 200_000
        assert route.calls.last.request.headers["x-api-key"] == "k"

    @pytest.mark.asyncio
    @respx.mock
    async def test_anthropic_falls_back_to_the_catalogue_when_it_cannot_be_asked(self) -> None:
        respx.get("https://api.anthropic.com/v1/models/claude-haiku-4-5").mock(
            return_value=httpx.Response(401, json={"type": "error"})
        )
        provider = AnthropicProvider(model_name="claude-haiku-4-5", api_key="expired")

        assert await cw.resolve_context_window(provider) == 200_000

    @pytest.mark.asyncio
    @respx.mock
    async def test_an_id_the_catalogue_does_not_know_stays_unknown(self) -> None:
        respx.get("https://api.openai.com/v1/models").mock(return_value=_listing())
        provider = OpenAIProvider(model_name="gpt-from-the-future", api_key="k")

        assert await cw.resolve_context_window(provider) is None


class TestTheShippedCatalogue:
    def test_every_provider_that_reads_one_finds_windows_in_it(self) -> None:
        """What a refresh may change is the numbers, never whether a provider has any."""
        shipped = cw._catalogue()
        for provider in (OpenAIProvider, AnthropicProvider, CerebrasProvider):
            models = shipped[provider.context_window_catalog]
            assert models and all(cw.positive_int(window) for window in models.values())
