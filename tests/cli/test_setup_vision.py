"""`domyn setup` finding out whether the model reads images.

It sends the provider a small image and writes `vision: true` on the provider
entry when the request is accepted, which is what mounts `read_image`. It is the
last step of the model section and asks nothing, so no scripted answer is for it.
"""

import base64
import io
import json
import logging
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import httpx
import pytest
import respx
from PIL import Image

from domyn.agents.interfaces.cli import Theme, setup
from domyn.agents.interfaces.cli import config as cfg
from domyn.agents.interfaces.cli import select as select_mod
from domyn.agents.interfaces.cli.prompt import GoBack

URL = "http://vllm.test:8000/v1/chat/completions"
ANSWER = {"choices": [{"message": {"content": "Red."}}], "usage": {"completion_tokens": 1}}
VLLM = ["vllm", "m", URL, "", "", ""]
"""Provider, model, URL, key, max tokens, context window; the built-in skills follow."""

# Captured before the CLI conftest replaces it for every other wizard test.
_REAL_KEEP = setup._keep


@pytest.fixture(autouse=True)
def isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cfg, "GLOBAL_DIR", tmp_path / "home" / ".domyn")
    monkeypatch.setattr(select_mod, "_interactive", lambda theme: False)
    for name in ("OPENAI_API_KEY", "LLM_API_KEY", "VLLM_API_KEY"):
        monkeypatch.setenv(name, "")
        monkeypatch.delenv(name)


@pytest.fixture
def theme() -> Theme:
    return Theme.build(allow_animation=False)


@pytest.fixture
def console() -> MagicMock:
    return MagicMock()


def _run(
    console: MagicMock,
    theme: Theme,
    answers: list[Any],
    existing: dict[str, Any] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    with patch("builtins.input", side_effect=answers):
        result = setup.run_setup(console, theme, existing, **kwargs)
    assert result is not None
    return result


def _printed(console: MagicMock) -> str:
    return " ".join(str(c.args[0]) for c in console.print.call_args_list if c.args)


class TestTheProbe:
    @respx.mock
    def test_an_accepted_image_is_no_rejection(self) -> None:
        route = respx.post(URL).mock(return_value=httpx.Response(200, json=ANSWER))

        assert (
            setup._image_rejection(cfg.ProviderConfig(type="vllm", model_name="m", url=URL)) is None
        )

        body = json.loads(route.calls.last.request.read())
        assert body["max_tokens"] == 16
        text, picture = body["messages"][0]["content"]
        assert text["type"] == "text"
        encoded = picture["image_url"]["url"].removeprefix("data:image/png;base64,")
        # Not a single pixel: some image processors refuse an image smaller than a patch.
        assert Image.open(io.BytesIO(base64.b64decode(encoded))).size == (64, 64)

    @respx.mock
    def test_a_refusal_is_reported_with_what_the_server_said(self) -> None:
        respx.post(URL).mock(
            return_value=httpx.Response(400, text='{"error": "m is not a multimodal model"}')
        )

        rejection = setup._image_rejection(cfg.ProviderConfig(type="vllm", model_name="m", url=URL))

        assert rejection is not None
        assert rejection.startswith("HTTP 400")
        assert "not a multimodal model" in rejection

    @pytest.mark.parametrize(
        ("body", "shown"),
        [
            pytest.param(
                {"error": {"message": "m is not a multimodal model", "type": "BadRequestError"}},
                "m is not a multimodal model",
                id="openai, litellm, openrouter, recent vllm",
            ),
            pytest.param(
                {
                    "type": "error",
                    "error": {"type": "invalid_request_error", "message": "images not allowed"},
                },
                "images not allowed",
                id="anthropic",
            ),
            pytest.param(
                {"object": "error", "message": "m is not a multimodal model", "code": 400},
                "m is not a multimodal model",
                id="older vllm",
            ),
            pytest.param({"detail": "Unsupported content"}, "Unsupported content", id="fastapi"),
            pytest.param({"error": "images not supported"}, "images not supported", id="plain"),
        ],
    )
    @respx.mock
    def test_a_refusal_shows_the_message_not_the_body(self, body: Any, shown: str) -> None:
        respx.post(URL).mock(return_value=httpx.Response(400, json=body))

        rejection = setup._image_rejection(cfg.ProviderConfig(type="vllm", model_name="m", url=URL))

        assert rejection == f"HTTP 400: {shown}"

    @pytest.mark.parametrize(
        ("text", "shown"),
        [
            pytest.param("Bad Request:\r\n  no images", "Bad Request: no images", id="not json"),
            pytest.param('{"code": 400}', '{"code": 400}', id="json without a message"),
            pytest.param('["no", "images"]', '["no", "images"]', id="json not an object"),
        ],
    )
    @respx.mock
    def test_a_body_with_no_message_is_shown_on_one_line(self, text: str, shown: str) -> None:
        respx.post(URL).mock(return_value=httpx.Response(400, text=text))

        rejection = setup._image_rejection(cfg.ProviderConfig(type="vllm", model_name="m", url=URL))

        assert rejection == f"HTTP 400: {shown}"

    @respx.mock
    def test_a_refusal_logs_nothing(self, caplog: pytest.LogCaptureFixture) -> None:
        """The provider logs a refused request as an error, printed outside the wizard's
        console: a row going back did not erase, so the question before it was left on
        screen twice. The refusal is said on the wizard's own line instead."""
        respx.post(URL).mock(return_value=httpx.Response(400, text="not multimodal"))
        library_logger = logging.getLogger("domyn.agents")
        level = library_logger.level
        caplog.set_level(logging.DEBUG)

        setup._image_rejection(cfg.ProviderConfig(type="vllm", model_name="m", url=URL))

        assert [r for r in caplog.records if r.name.startswith("domyn.agents")] == []
        # Only for the probe: the library logs as before once it is over.
        assert library_logger.level == level

    @respx.mock
    def test_an_endpoint_that_does_not_answer_is_named_by_what_happened(self) -> None:
        respx.post(URL).mock(side_effect=httpx.ConnectError("refused"))

        rejection = setup._image_rejection(cfg.ProviderConfig(type="vllm", model_name="m", url=URL))

        assert rejection == "ConnectError"

    def test_a_provider_that_cannot_be_built_is_a_rejection_not_a_crash(self) -> None:
        """OpenAI without a key: the provider refuses to be built."""
        rejection = setup._image_rejection(cfg.ProviderConfig(type="openai", model_name="m"))

        assert rejection is not None

    @respx.mock
    def test_the_key_is_sent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("VLLM_API_KEY", "sk-probe")
        monkeypatch.setenv("LLM_API_KEY", "sk-probe")
        route = respx.post(URL).mock(return_value=httpx.Response(200, json=ANSWER))

        setup._image_rejection(cfg.ProviderConfig(type="vllm", model_name="m", url=URL))

        assert route.calls.last.request.headers["authorization"] == "Bearer sk-probe"


class TestWhatSetupSaves:
    @respx.mock
    def test_a_model_that_reads_images_is_saved_as_one(
        self, console: MagicMock, theme: Theme, tmp_path: Path
    ) -> None:
        respx.post(URL).mock(return_value=httpx.Response(200, json=ANSWER))

        result = _run(console, theme, [*VLLM, ""], path=tmp_path / "config.yaml")

        assert result["llm_providers"][0]["vision"] is True
        assert "Vision: enabled" in _printed(console)

    @respx.mock
    def test_a_model_that_refuses_them_is_not(
        self, console: MagicMock, theme: Theme, tmp_path: Path
    ) -> None:
        respx.post(URL).mock(return_value=httpx.Response(400, text="no images here"))

        result = _run(console, theme, [*VLLM, ""], path=tmp_path / "config.yaml")

        assert "vision" not in result["llm_providers"][0]
        assert "Vision: not supported by this model (HTTP 400: no images here)" in _printed(console)

    @respx.mock
    def test_changing_the_model_asks_again(
        self, console: MagicMock, theme: Theme, tmp_path: Path
    ) -> None:
        """`/model` runs the wizard for the provider alone, and the answer is the new model's."""
        respx.post(URL).mock(return_value=httpx.Response(200, json=ANSWER))
        first = _run(console, theme, [*VLLM, ""], path=tmp_path / "config.yaml")
        respx.post(URL).mock(return_value=httpx.Response(400, text="no images here"))

        second = _run(
            console,
            theme,
            ["vllm", "text-only", URL, "", "", ""],
            first,
            path=tmp_path / "config.yaml",
            provider_only=True,
        )

        assert "vision" not in second["llm_providers"][0]

    @respx.mock
    def test_a_kept_model_is_asked_too(
        self,
        console: MagicMock,
        theme: Theme,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A config written before `read_image` says nothing about images, and keeping
        its model must not keep it that way."""
        respx.post(URL).mock(return_value=httpx.Response(400, text="no images here"))
        before = _run(console, theme, [*VLLM, ""], path=tmp_path / "config.yaml")
        assert "vision" not in before["llm_providers"][0]
        monkeypatch.setattr(setup, "_keep", _REAL_KEEP)
        respx.post(URL).mock(return_value=httpx.Response(200, json=ANSWER))

        after = _run(console, theme, [""] * 8, before, path=tmp_path / "config.yaml")

        assert after["llm_providers"][0]["model_name"] == "m"
        assert after["llm_providers"][0]["vision"] is True

    @respx.mock
    def test_going_back_over_it_sends_nothing_again(
        self, console: MagicMock, theme: Theme, tmp_path: Path
    ) -> None:
        """Ctrl-B at the question after it goes back to the context window; coming
        forward passes the check again, with the same settings."""
        route = respx.post(URL).mock(return_value=httpx.Response(200, json=ANSWER))

        result = _run(console, theme, [*VLLM, GoBack(), "", ""], path=tmp_path / "config.yaml")

        assert route.call_count == 1
        assert result["llm_providers"][0]["vision"] is True

    @respx.mock
    def test_going_back_to_change_the_key_asks_again(
        self, console: MagicMock, theme: Theme, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The key is not among the provider's settings, only the name of its variable,
        so an answer remembered by the settings alone kept a wrong key's refusal after
        going back to type the right one."""
        key_env = cfg.ProviderConfig(type="vllm").default_key_env()
        # Setup puts a pasted key into the environment; this makes it undone after the test.
        monkeypatch.setenv(key_env, "")
        monkeypatch.delenv(key_env)

        def answer(request: httpx.Request) -> httpx.Response:
            if request.headers.get("authorization") == "Bearer good-key":
                return httpx.Response(200, json=ANSWER)
            return httpx.Response(401, text="bad key")

        route = respx.post(URL).mock(side_effect=answer)

        result = _run(
            console,
            theme,
            [
                *VLLM[:3],
                "bad-key",
                "",
                "",
                # The question after the check, then the context window and max tokens.
                GoBack(),
                GoBack(),
                GoBack(),
                "good-key",
                "",
                "",
                "",
            ],
            path=tmp_path / "config.yaml",
        )

        assert [call.request.headers["authorization"] for call in route.calls] == [
            "Bearer bad-key",
            "Bearer good-key",
        ]
        assert result["llm_providers"][0]["vision"] is True

    def test_a_setup_without_a_model_asks_nothing(
        self,
        console: MagicMock,
        theme: Theme,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(
            setup, "_image_rejection", lambda provider: pytest.fail("there is no model to ask")
        )

        result = _run(console, theme, ["vllm", "", URL, "", "", "", ""], path=tmp_path / "c.yaml")

        assert "vision" not in result["llm_providers"][0]
        assert "Vision:" not in _printed(console)
