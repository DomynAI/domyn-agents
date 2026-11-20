"""A message with an image, as each provider sends it.

Messages are built in one shape, OpenAI's chat format (`image_message`): the
OpenAI-compatible endpoints take it as it is, Anthropic and OpenAI's Responses API
convert it at their boundary.
"""

import base64
import json
from typing import Any

import httpx
import pytest
import respx

from domyn.agents.llm import (
    AnthropicProvider,
    LLMMessage,
    OpenAIProvider,
    VLLMProvider,
)
from domyn.agents.llm.base import image_message

VLLM_URL = "http://vllm.test/v1/chat/completions"
CHAT_ANSWER = {"choices": [{"message": {"content": "Red."}}], "usage": {"completion_tokens": 1}}
PNG = b"\x89PNG not really"
DATA_URL = "data:image/png;base64," + base64.b64encode(PNG).decode("ascii")


def _question() -> LLMMessage:
    return image_message("What colour?", PNG, "image/png")


def _body(route: respx.Route) -> dict[str, Any]:
    return json.loads(route.calls.last.request.read())  # type: ignore[no-any-return]


def test_the_message_is_a_text_part_and_a_data_url() -> None:
    message = _question()

    assert message.role == "user"
    assert message.content == [
        {"type": "text", "text": "What colour?"},
        {"type": "image_url", "image_url": {"url": DATA_URL}},
    ]


class TestOpenAICompatibleEndpoints:
    @pytest.mark.asyncio
    @respx.mock
    async def test_vllm_sends_it_as_it_is(self) -> None:
        route = respx.post(VLLM_URL).mock(return_value=httpx.Response(200, json=CHAT_ANSWER))
        message = _question()

        response = await VLLMProvider(url=VLLM_URL, model_name="m").generate([message])

        assert response.content == "Red."
        assert _body(route)["messages"] == [{"role": "user", "content": message.content}]


class TestAnthropic:
    @pytest.mark.asyncio
    @respx.mock
    async def test_a_data_url_becomes_a_base64_image_block(self) -> None:
        route = respx.post("https://api.anthropic.com/v1/messages").mock(
            return_value=httpx.Response(
                200,
                json={
                    "content": [{"type": "text", "text": "Red."}],
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                },
            )
        )

        await AnthropicProvider(model_name="claude", api_key="k").generate([_question()])

        assert _body(route)["messages"] == [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What colour?"},
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": base64.b64encode(PNG).decode("ascii"),
                        },
                    },
                ],
            }
        ]

    def test_an_address_becomes_a_url_image_block(self) -> None:
        message = LLMMessage(
            role="user",
            content=[{"type": "image_url", "image_url": {"url": "https://example.com/a.png"}}],
        )

        _, messages = AnthropicProvider(model_name="claude", api_key="k")._convert_messages(
            [message]
        )

        assert messages[0]["content"] == [
            {"type": "image", "source": {"type": "url", "url": "https://example.com/a.png"}}
        ]

    def test_an_image_part_without_an_address_is_left_as_written(self) -> None:
        """It used to stop the request with an error about `None`."""
        part = {"type": "image_url", "image_url": {}}

        _, messages = AnthropicProvider(model_name="claude", api_key="k")._convert_messages(
            [LLMMessage(role="user", content=[part])]
        )

        assert messages[0]["content"] == [part]


class TestOpenAI:
    def test_the_parts_take_the_names_of_the_responses_api(self) -> None:
        serialized = OpenAIProvider._serialize_messages([_question()])

        assert serialized == [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "What colour?"},
                    {"type": "input_image", "image_url": DATA_URL},
                ],
            }
        ]

    def test_a_part_of_another_kind_is_left_as_written(self) -> None:
        part = {"type": "input_file", "file_id": "f"}

        serialized = OpenAIProvider._serialize_messages([LLMMessage(role="user", content=[part])])

        assert serialized[0]["content"] == [part]

    def test_a_plain_message_is_unchanged(self) -> None:
        serialized = OpenAIProvider._serialize_messages([LLMMessage(role="user", content="Hi")])

        assert serialized == [{"role": "user", "content": "Hi"}]
