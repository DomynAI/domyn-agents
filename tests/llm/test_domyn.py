"""Tests for DomynProvider: the models of a platform space, called through the platform SDK."""

import json
from collections.abc import AsyncIterator
from typing import Any

import pytest

from domyn.agents.integrations import domyn_platform_sdk
from domyn.agents.integrations.domyn_platform_sdk import PlatformSDKNotInstalledError
from domyn.agents.llm import DomynProvider, LLMMessage, LLMProviderFactory, LLMResponse

_ENV = ("DOMYN_API_KEY", "DOMYN_SPACE_ID", "DOMYN_BASE_URL", "DOMYN_CONFIGURATION_ID")


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in _ENV:
        monkeypatch.delenv(name, raising=False)


def _provider(**kwargs: Any) -> DomynProvider:
    defaults: dict[str, Any] = {
        "model_name": "qwen",
        "api_key": "key-1",
        "space_id": "space-1",
        "base_url": "https://p.example.com",
    }
    return DomynProvider(**{**defaults, **kwargs})


USER = [LLMMessage(role="user", content="hi")]


class Recorder:
    """Stands in for the SDK seam and records the settings and payload it was given."""

    def __init__(self) -> None:
        self.settings: Any = None
        self.payload: dict[str, Any] = {}
        self.completion: dict[str, Any] = {}
        self.chunks: list[dict[str, Any]] = []

    async def chat_completion(self, settings: Any, payload: dict[str, Any]) -> dict[str, Any]:
        self.settings, self.payload = settings, payload
        return self.completion

    async def stream_chat_completion(
        self, settings: Any, payload: dict[str, Any]
    ) -> AsyncIterator[dict[str, Any]]:
        self.settings, self.payload = settings, payload
        for chunk in self.chunks:
            yield chunk


@pytest.fixture
def seam(monkeypatch: pytest.MonkeyPatch) -> Recorder:
    recorder = Recorder()
    monkeypatch.setattr(domyn_platform_sdk, "chat_completion", recorder.chat_completion)
    monkeypatch.setattr(
        domyn_platform_sdk, "stream_chat_completion", recorder.stream_chat_completion
    )
    return recorder


def _delta(**delta: Any) -> dict[str, Any]:
    return {"choices": [{"delta": delta}]}


class TestInit:
    def test_defaults(self) -> None:
        provider = _provider()

        assert provider.type == "domyn"
        assert provider.name == "qwen"
        assert provider.configuration_id is None
        assert provider.body == {}

    def test_every_connection_field_falls_back_to_its_variable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("DOMYN_API_KEY", "env-key")
        monkeypatch.setenv("DOMYN_SPACE_ID", "env-space")
        monkeypatch.setenv("DOMYN_BASE_URL", "env.example.com")
        monkeypatch.setenv("DOMYN_CONFIGURATION_ID", "env-cfg")

        provider = DomynProvider(model_name="qwen")

        assert provider.settings.api_key == "env-key"
        assert provider.settings.space_id == "env-space"
        assert provider.settings.base_url == "env.example.com"
        assert provider.settings.configuration_id == "env-cfg"

    def test_an_argument_beats_the_variable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DOMYN_SPACE_ID", "env-space")

        assert _provider(space_id="mine").space_id == "mine"

    def test_url_is_read_as_the_base_url(self) -> None:
        provider = DomynProvider(model_name="qwen", url="https://via-url.example.com")

        assert provider.base_url == "https://via-url.example.com"

    def test_the_api_key_is_not_in_the_repr(self) -> None:
        assert "key-1" not in repr(_provider())

    def test_the_factory_builds_it_from_a_config_dict(self) -> None:
        provider = LLMProviderFactory.from_config(
            {
                "type": "domyn",
                "name": "default_provider",
                "model_name": "qwen",
                "url": "https://p.example.com",
                "space_id": "space-1",
                "api_key": "k",
                "context_window": 100_000,
            }
        )

        assert isinstance(provider, DomynProvider)
        assert provider.name == "default_provider"
        assert provider.context_window == 100_000


class TestGenerate:
    @pytest.mark.asyncio
    async def test_the_request_goes_to_the_sdk_with_the_connection_fields(
        self, seam: Recorder
    ) -> None:
        seam.completion = {"choices": [{"message": {"content": "hello"}}]}

        await _provider(configuration_id="cfg").generate(USER)

        assert seam.settings.api_key == "key-1"
        assert seam.settings.space_id == "space-1"
        assert seam.settings.base_url == "https://p.example.com"
        assert seam.settings.configuration_id == "cfg"

    @pytest.mark.asyncio
    async def test_the_payload_names_the_model_and_merges_the_extras(self, seam: Recorder) -> None:
        seam.completion = {"choices": [{"message": {"content": "x"}}]}
        provider = _provider(generation_params={"max_tokens": 10}, body={"temperature": 0.5})

        await provider.generate(USER, top_p=0.9)

        assert seam.payload == {
            "model": "qwen",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 10,
            "temperature": 0.5,
            "top_p": 0.9,
        }

    @pytest.mark.asyncio
    async def test_the_response_is_parsed(self, seam: Recorder) -> None:
        seam.completion = {
            "choices": [
                {
                    "message": {
                        "content": "hello",
                        "reasoning_content": "thinking",
                        "tool_calls": [
                            {
                                "id": "c1",
                                "type": "function",
                                "function": {"name": "add", "arguments": '{"a": 1}'},
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 3},
        }
        tools = [{"type": "function", "function": {"name": "add", "parameters": {}}}]

        response = await _provider().generate(USER, tools=tools, tool_choice="auto")

        assert seam.payload["tools"] == tools
        assert seam.payload["tool_choice"] == "auto"
        assert response.content == "hello"
        assert response.reasoning_content == "thinking"
        assert response.finish_reason == "tool_calls"
        assert response.generated_tokens == 3
        assert response.usage["prompt_tokens"] == 5
        assert [(c.id, c.function.name, c.function.arguments) for c in response.tool_calls] == [
            ("c1", "add", '{"a": 1}')
        ]

    @pytest.mark.asyncio
    async def test_an_empty_content_is_an_empty_string(self, seam: Recorder) -> None:
        seam.completion = {"choices": [{"message": {"content": None}}]}

        response = await _provider().generate(USER)

        assert response.content == ""
        assert response.tool_calls == []

    @pytest.mark.asyncio
    async def test_without_the_sdk_it_says_how_to_install_it(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(domyn_platform_sdk, "sdk_installed", lambda: False)

        with pytest.raises(PlatformSDKNotInstalledError, match=r"domyn-platform-sdk>="):
            await _provider().generate(USER)


class TestStreaming:
    @staticmethod
    async def _collect(provider: DomynProvider, **kwargs: Any) -> list[LLMResponse]:
        return [r async for r in provider.generate_streaming(USER, **kwargs)]

    @pytest.mark.asyncio
    async def test_content_arrives_a_chunk_at_a_time(self, seam: Recorder) -> None:
        seam.chunks = [_delta(content="\n"), _delta(content="hel"), _delta(content="lo")]

        responses = await self._collect(_provider())

        assert [r.content for r in responses] == ["hel", "lo"]
        assert [r.generated_tokens for r in responses] == [1, 2]

    @pytest.mark.asyncio
    async def test_the_payload_names_the_model_for_a_stream_too(self, seam: Recorder) -> None:
        seam.chunks = [_delta(content="x")]

        await self._collect(_provider(body={"temperature": 0.1}))

        assert seam.payload["model"] == "qwen"
        assert seam.payload["temperature"] == 0.1
        assert seam.settings.space_id == "space-1"

    @pytest.mark.asyncio
    async def test_reasoning_comes_first_in_a_chunk_of_its_own(self, seam: Recorder) -> None:
        seam.chunks = [
            _delta(reasoning_content="think "),
            _delta(reasoning="hard"),
            _delta(content="answer"),
        ]

        responses = await self._collect(_provider())

        assert responses[0].content == ""
        assert responses[0].reasoning_content == "think hard"
        assert responses[1].content == "answer"

    @pytest.mark.asyncio
    async def test_tool_calls_are_assembled_into_one_closing_response(self, seam: Recorder) -> None:
        seam.chunks = [
            _delta(
                tool_calls=[
                    {"index": 0, "id": "c1", "function": {"name": "add", "arguments": '{"a"'}}
                ]
            ),
            _delta(tool_calls=[{"index": 0, "function": {"arguments": ": 1}"}}]),
            {
                "choices": [{"delta": {}, "finish_reason": "tool_calls"}],
                "usage": {"completion_tokens": 4},
            },
        ]

        responses = await self._collect(_provider())

        assert len(responses) == 1
        final = responses[0]
        assert [
            (c.id, c.function.name, json.loads(c.function.arguments)) for c in final.tool_calls
        ] == [("c1", "add", {"a": 1})]
        assert final.finish_reason == "tool_calls"
        assert final.usage == {"completion_tokens": 4}

    @pytest.mark.asyncio
    async def test_the_usage_gets_a_closing_chunk_of_its_own(self, seam: Recorder) -> None:
        seam.chunks = [_delta(content="hi"), {"choices": [], "usage": {"completion_tokens": 1}}]

        responses = await self._collect(_provider())

        assert [r.content for r in responses] == ["hi", ""]
        assert responses[-1].usage == {"completion_tokens": 1}

    @pytest.mark.asyncio
    async def test_a_chunk_without_choices_is_skipped(self, seam: Recorder) -> None:
        seam.chunks = [{"choices": []}, _delta(content="ok")]

        assert [r.content for r in await self._collect(_provider())] == ["ok"]

    @pytest.mark.asyncio
    async def test_without_the_sdk_it_says_how_to_install_it(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(domyn_platform_sdk, "sdk_installed", lambda: False)

        with pytest.raises(PlatformSDKNotInstalledError):
            await self._collect(_provider())
