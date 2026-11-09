"""`domyn setup` for a vLLM or custom endpoint: URL, then key, then the models it lists."""

import io
from pathlib import Path
from unittest.mock import patch

import httpx
import pytest
import respx
from rich.console import Console

from domyn.agents.interfaces.cli import Theme, setup
from domyn.agents.interfaces.cli import config as cfg
from domyn.agents.interfaces.cli import select as select_mod

# Taken before `tests/cli/conftest.py` replaces it for every other test.
_REAL_DISCOVERY = setup._discover_endpoint_models

MODELS_URL = "https://gw.example.com/v1/models"
LISTING = {
    "object": "list",
    "data": [
        {"id": "qwen3-32b", "object": "model", "max_model_len": 32768},
        {"id": "gpt-oss-120b", "object": "model"},
    ],
}


@pytest.fixture(autouse=True)
def isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cfg, "GLOBAL_DIR", tmp_path / "home" / ".domyn")
    monkeypatch.setattr(cfg, "PROJECT_DIR", tmp_path / "project" / ".domyn")
    for name in ("LLM_API_KEY", "VLLM_API_KEY"):
        monkeypatch.setenv(name, "")
        monkeypatch.delenv(name)
    monkeypatch.setattr(select_mod, "_interactive", lambda theme: False)
    monkeypatch.setattr(setup, "_discover_endpoint_models", _REAL_DISCOVERY)


@pytest.fixture
def theme() -> Theme:
    return Theme.build(allow_animation=False)


def _run(theme: Theme, tmp_path: Path, answers: list[str]) -> tuple[dict | None, str]:
    buffer = io.StringIO()
    with patch("builtins.input", side_effect=answers):
        result = setup.run_setup(
            Console(file=buffer, width=200), theme, path=tmp_path / "config.yaml"
        )
    return result, buffer.getvalue()


class TestTheModelsAreListed:
    @respx.mock
    def test_a_listed_model_is_picked_with_the_key_just_given(
        self, theme: Theme, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        route = respx.get(MODELS_URL).mock(return_value=httpx.Response(200, json=LISTING))

        # provider, url, key, model 2 of the list, max tokens, context window, (rest)
        result, shown = _run(
            theme,
            tmp_path,
            ["custom", "https://gw.example.com/v1", "sk-test", "2", "", "", "", ""],
        )

        assert result is not None
        provider = result["llm_providers"][0]
        assert provider["model_name"] == "gpt-oss-120b"
        assert provider["url"] == "https://gw.example.com/v1/chat/completions"
        assert provider["api_key"] == "${LLM_API_KEY}"
        assert route.calls.last.request.headers["authorization"] == "Bearer sk-test"
        assert "32,768 tokens" in shown

    @respx.mock
    def test_vllm_lists_them_too(self, theme: Theme, tmp_path: Path) -> None:
        respx.get(MODELS_URL).mock(return_value=httpx.Response(200, json=LISTING))

        result, _ = _run(
            theme, tmp_path, ["vllm", "https://gw.example.com", "", "1", "", "", "", ""]
        )

        assert result is not None
        assert result["llm_providers"][0]["model_name"] == "qwen3-32b"


class TestAFailedListingFallsBackToTyping:
    @respx.mock
    @pytest.mark.parametrize(
        ("status", "hint"), [(401, "check the API key"), (404, "check the URL")]
    )
    def test_the_failure_says_what_to_check(
        self, theme: Theme, tmp_path: Path, status: int, hint: str
    ) -> None:
        respx.get(MODELS_URL).mock(return_value=httpx.Response(status))

        result, shown = _run(
            theme,
            tmp_path,
            ["custom", "https://gw.example.com/v1", "sk-bad", "typed-model", "", "", "", ""],
        )

        assert result is not None
        assert result["llm_providers"][0]["model_name"] == "typed-model"
        assert f"HTTP {status}" in shown
        assert hint in shown

    @respx.mock
    def test_an_empty_listing_asks_for_the_name(self, theme: Theme, tmp_path: Path) -> None:
        respx.get(MODELS_URL).mock(return_value=httpx.Response(200, json={"data": []}))

        result, shown = _run(
            theme,
            tmp_path,
            ["custom", "https://gw.example.com/v1", "", "typed-model", "", "", "", ""],
        )

        assert result is not None
        assert result["llm_providers"][0]["model_name"] == "typed-model"
        assert "lists no models" in shown
