"""Tests for the setup wizard writing a real ConfigurationV1 file."""

from pathlib import Path
from typing import Any, ClassVar
from unittest.mock import MagicMock, patch

import pytest
import yaml

from domyn.agents.integrations.domyn_platform_sdk import (
    PlatformSDKNotInstalledError,
    PlatformSettings,
)
from domyn.agents.interfaces.cli import Theme, setup
from domyn.agents.interfaces.cli import config as cfg
from domyn.agents.interfaces.cli import prompt as prompt_mod
from domyn.agents.interfaces.cli import select as select_mod


@pytest.fixture(autouse=True)
def isolated_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Redirect config and credentials into a temp dir, as tests/cli/test_cli_config.py does."""
    monkeypatch.setattr(cfg, "GLOBAL_DIR", tmp_path / "home" / ".domyn")
    monkeypatch.setattr(cfg, "PROJECT_DIR", tmp_path / "project" / ".domyn")
    # setenv first so the original value (or its absence) is always recorded:
    # delenv on an unset variable records nothing, and a key the wizard then
    # writes to os.environ would survive into every later test.
    for name in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "LLM_API_KEY"):
        monkeypatch.setenv(name, "")
        monkeypatch.delenv(name)
    return tmp_path


@pytest.fixture(autouse=True)
def force_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests run without a TTY, which is exactly the numbered-prompt fallback path."""
    monkeypatch.setattr(select_mod, "_interactive", lambda theme: False)


@pytest.fixture
def console() -> MagicMock:
    return MagicMock()


@pytest.fixture
def theme() -> Theme:
    return Theme.build(allow_animation=False)


class TestAFreshWizardRun:
    def test_writes_a_configurationv1_shaped_file(
        self, console: MagicMock, theme: Theme, tmp_path: Path
    ) -> None:
        target = tmp_path / "config.yaml"
        with patch(
            "builtins.input",
            side_effect=["openai", "gpt-4o-mini", "", "", "", "", ""],
        ):
            result = setup.run_setup(console, theme, path=target)

        assert result is not None
        assert target.is_file()
        on_disk = yaml.safe_load(target.read_text())
        assert on_disk == result
        assert "entrypoint" in on_disk
        assert "provider" not in on_disk
        assert on_disk["entrypoint"] == "Assistant"
        assert on_disk["agents"][0]["llm_provider"] == "default_provider"
        assert on_disk["agents"][0]["planner_strategy"] == "tool_use_thought"
        assert on_disk["llm_providers"][0]["model_name"] == "gpt-4o-mini"

    def test_the_planner_is_never_a_prompted_field(
        self, console: MagicMock, theme: Theme, tmp_path: Path
    ) -> None:
        target = tmp_path / "config.yaml"
        with patch(
            "builtins.input",
            side_effect=["openai", "m", "", "", "", "", ""],
        ):
            result = setup.run_setup(console, theme, path=target)

        assert result is not None
        planner_strategies = result["planner_strategies"]
        assert planner_strategies == [
            {
                "name": "tool_use_thought",
                "type": "tool_use_thought",
                "compactor": {"strategy": "two_phase"},
            }
        ]

    def test_a_secret_is_stored_in_credentials_not_in_the_yaml(
        self, console: MagicMock, theme: Theme, tmp_path: Path
    ) -> None:
        target = tmp_path / "config.yaml"
        with patch(
            "builtins.input",
            side_effect=["openai", "m", "", "sk-super-secret", "", "", ""],
        ):
            result = setup.run_setup(console, theme, path=target)

        assert result is not None
        assert "sk-super-secret" not in target.read_text()
        assert result["llm_providers"][0]["api_key"] == "${OPENAI_API_KEY}"
        assert cfg.read_credentials()["OPENAI_API_KEY"] == "sk-super-secret"

    def test_a_self_hosted_provider_without_a_url_is_reported_incomplete(
        self, console: MagicMock, theme: Theme, tmp_path: Path
    ) -> None:
        target = tmp_path / "config.yaml"
        with patch(
            "builtins.input",
            side_effect=["vllm", "m", "", "", "", "", ""],
        ):
            result = setup.run_setup(console, theme, path=target)

        assert result is not None  # still saved
        assert cfg.is_configuration_v1_usable(result) is False

    def test_aborting_saves_nothing(self, console: MagicMock, theme: Theme, tmp_path: Path) -> None:
        target = tmp_path / "config.yaml"
        with patch("builtins.input", side_effect=EOFError):
            result = setup.run_setup(console, theme, path=target)

        assert result is None
        assert not target.is_file()


def _domyn_answers(
    url: str = "https://p.example.com",
    space: str = "space-1",
    configuration: str = "",
    *,
    key: str | None = "",
    model: str = "qwen",
) -> list[str]:
    """What the wizard is asked for Domyn, in order. ``key=None``: it is already set, so not asked."""
    asked_key = [] if key is None else [key]
    return ["domyn", url, space, configuration, *asked_key, model, "", "", "A", "i", ""]


class TestTheDomynProvider:
    """Domyn is the first choice, and the one a fresh setup starts on."""

    @pytest.fixture(autouse=True)
    def clean_domyn_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for name in ("DOMYN_API_KEY", "DOMYN_SPACE_ID", "DOMYN_BASE_URL"):
            monkeypatch.setenv(name, "")
            monkeypatch.delenv(name)

    def test_it_is_the_first_provider_offered(self) -> None:
        assert next(iter(cfg.PROVIDER_HINTS)) == "domyn"

    def test_it_is_what_a_fresh_setup_starts_on(self) -> None:
        assert cfg.ProviderConfig().type == "domyn"
        assert cfg.current_settings_from(None)[0].type == "domyn"

    def test_pressing_enter_on_the_provider_choice_picks_it(
        self, console: MagicMock, theme: Theme
    ) -> None:
        with patch("builtins.input", side_effect=[""]):
            assert setup._choose_provider(console, theme, cfg.ProviderConfig().type) == "domyn"

    def test_the_wizard_asks_for_the_platform_url_space_and_configuration(
        self, console: MagicMock, theme: Theme, tmp_path: Path
    ) -> None:
        target = tmp_path / "config.yaml"
        with patch("builtins.input", side_effect=_domyn_answers(configuration="cfg-1")):
            result = setup.run_setup(console, theme, path=target)

        assert result is not None
        provider = result["llm_providers"][0]
        assert provider["type"] == "domyn"
        assert provider["model_name"] == "qwen"
        assert provider["url"] == "https://p.example.com"
        assert provider["space_id"] == "space-1"
        assert provider["configuration_id"] == "cfg-1"
        assert cfg.is_configuration_v1_usable(result) is True

    def test_the_key_goes_to_the_credentials_file_under_its_conventional_name(
        self, console: MagicMock, theme: Theme, tmp_path: Path
    ) -> None:
        target = tmp_path / "config.yaml"
        with patch("builtins.input", side_effect=_domyn_answers(key="dk-secret")):
            result = setup.run_setup(console, theme, path=target)

        assert result is not None
        assert "dk-secret" not in target.read_text()
        assert result["llm_providers"][0]["api_key"] == "${DOMYN_API_KEY}"
        assert cfg.read_credentials()["DOMYN_API_KEY"] == "dk-secret"

    def test_the_environment_pre_fills_the_url_and_the_space(
        self, console: MagicMock, theme: Theme, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("DOMYN_BASE_URL", "env.example.com")
        monkeypatch.setenv("DOMYN_SPACE_ID", "env-space")
        target = tmp_path / "config.yaml"
        with patch("builtins.input", side_effect=_domyn_answers("", "")):
            result = setup.run_setup(console, theme, path=target)

        assert result is not None
        provider = result["llm_providers"][0]
        assert provider["url"] == "env.example.com"
        assert provider["space_id"] == "env-space"

    def test_a_value_from_the_environment_is_named_not_shown(
        self, console: MagicMock, theme: Theme, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("DOMYN_BASE_URL", "secret-host.example.com")
        monkeypatch.setenv("DOMYN_SPACE_ID", "secret-space")
        with patch("builtins.input", side_effect=_domyn_answers("", "")):
            setup.run_setup(console, theme, path=tmp_path / "config.yaml")

        printed = _printed(console)
        assert "Platform URL: found in DOMYN_BASE_URL" in printed
        assert "Space id: found in DOMYN_SPACE_ID" in printed
        assert "secret-host" not in printed
        assert "secret-space" not in printed

    def test_typing_a_value_beats_the_environment(
        self, console: MagicMock, theme: Theme, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("DOMYN_BASE_URL", "env.example.com")
        monkeypatch.setenv("DOMYN_SPACE_ID", "env-space")
        with patch(
            "builtins.input", side_effect=_domyn_answers("typed.example.com", "typed-space")
        ):
            result = setup.run_setup(console, theme, path=tmp_path / "config.yaml")

        assert result is not None
        provider = result["llm_providers"][0]
        assert (provider["url"], provider["space_id"]) == ("typed.example.com", "typed-space")

    def test_a_variable_that_is_not_set_is_said(
        self, console: MagicMock, theme: Theme, tmp_path: Path
    ) -> None:
        with patch("builtins.input", side_effect=_domyn_answers()):
            setup.run_setup(console, theme, path=tmp_path / "config.yaml")

        printed = _printed(console)
        assert "DOMYN_BASE_URL is not set" in printed
        assert "DOMYN_SPACE_ID is not set" in printed

    def test_a_saved_value_is_offered_as_the_current_one(
        self, console: MagicMock, theme: Theme, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("DOMYN_BASE_URL", "env.example.com")
        existing = {
            "entrypoint": "A",
            "agents": [{"name": "A", "llm_provider": "p"}],
            "llm_providers": [
                {
                    "name": "p",
                    "type": "domyn",
                    "model_name": "m",
                    "url": "saved.example.com",
                    "space_id": "saved",
                }
            ],
        }
        with patch("builtins.input", side_effect=["", "", "", "", "", "m", "", "", "A", "i", ""]):
            result = setup.run_setup(console, theme, existing, path=tmp_path / "config.yaml")

        assert result is not None
        assert result["llm_providers"][0]["url"] == "saved.example.com"
        assert "found in DOMYN_BASE_URL" not in _printed(console)

    def test_a_missing_sdk_is_said_when_it_is_chosen(
        self, console: MagicMock, theme: Theme, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(setup, "sdk_installed", lambda: False)
        with patch("builtins.input", side_effect=_domyn_answers(space="s")):
            setup.run_setup(console, theme, path=tmp_path / "config.yaml")

        assert "domyn-platform-sdk>=" in _printed(console)

    def test_without_a_space_it_is_reported_incomplete(
        self, console: MagicMock, theme: Theme, tmp_path: Path
    ) -> None:
        with patch("builtins.input", side_effect=_domyn_answers(space="")):
            result = setup.run_setup(console, theme, path=tmp_path / "config.yaml")

        assert result is not None
        assert cfg.is_configuration_v1_usable(result) is False
        assert "space id is still required" in _printed(console)

    def test_the_settings_round_trip_through_the_saved_file(
        self, console: MagicMock, theme: Theme, tmp_path: Path
    ) -> None:
        with patch("builtins.input", side_effect=_domyn_answers(configuration="cfg-1")):
            result = setup.run_setup(console, theme, path=tmp_path / "config.yaml")

        assert result is not None
        provider = cfg.current_settings_from(result)[0]
        assert (provider.type, provider.space_id, provider.configuration_id) == (
            "domyn",
            "space-1",
            "cfg-1",
        )

    def test_another_provider_writes_no_space(
        self, console: MagicMock, theme: Theme, tmp_path: Path
    ) -> None:
        with patch(
            "builtins.input",
            side_effect=["openai", "gpt-4o-mini", "", "", "", "", "A", "i", ""],
        ):
            result = setup.run_setup(console, theme, path=tmp_path / "config.yaml")

        assert result is not None
        assert "space_id" not in result["llm_providers"][0]


def _printed(console: MagicMock) -> str:
    return " ".join(str(call.args[0]) for call in console.print.call_args_list if call.args)


class TestDiscoveringTheSpacesModels:
    """With a key, a URL and a space, the wizard offers the space's models as a list to pick from."""

    MODELS: ClassVar[list[dict[str, str]]] = [
        {"id": "qwen", "type": "vllm", "root": "Qwen/Qwen3-32B"},
        {"id": "gpt", "type": "openai", "root": "gpt-4o"},
    ]

    @pytest.fixture(autouse=True)
    def a_key_and_the_sdk(self, monkeypatch: pytest.MonkeyPatch) -> list[PlatformSettings]:
        for name in ("DOMYN_SPACE_ID", "DOMYN_BASE_URL"):
            monkeypatch.setenv(name, "")
            monkeypatch.delenv(name)
        monkeypatch.setenv("DOMYN_API_KEY", "dk-env")
        monkeypatch.setattr(setup, "sdk_installed", lambda: True)
        asked: list[PlatformSettings] = []

        async def fake_list_models(settings: PlatformSettings) -> list[dict[str, str]]:
            asked.append(settings)
            return self.MODELS

        monkeypatch.setattr(setup, "list_models", fake_list_models)
        return asked

    def test_the_list_is_asked_with_the_connection_just_given(
        self,
        console: MagicMock,
        theme: Theme,
        tmp_path: Path,
        a_key_and_the_sdk: list[PlatformSettings],
    ) -> None:
        with patch(
            "builtins.input", side_effect=_domyn_answers(configuration="cfg-1", key=None, model="1")
        ):
            setup.run_setup(console, theme, path=tmp_path / "config.yaml")

        assert a_key_and_the_sdk == [
            PlatformSettings(
                api_key="dk-env",
                space_id="space-1",
                base_url="https://p.example.com",
                configuration_id="cfg-1",
            )
        ]

    def test_a_model_is_picked_by_its_number(
        self, console: MagicMock, theme: Theme, tmp_path: Path
    ) -> None:
        with patch("builtins.input", side_effect=_domyn_answers(key=None, model="2")):
            result = setup.run_setup(console, theme, path=tmp_path / "config.yaml")

        assert result is not None
        assert result["llm_providers"][0]["model_name"] == "gpt"

    def test_each_model_shows_what_is_behind_it(self, console: MagicMock, theme: Theme) -> None:
        with (
            patch("builtins.input", side_effect=["1"]),
            patch.object(setup, "select", wraps=setup.select) as picker,
        ):
            setup._choose_domyn_model(
                console, theme, {"qwen": "vllm Qwen/Qwen3-32B", "gpt": ""}, current=""
            )

        assert picker.call_args.kwargs["descriptions"] == {"qwen": "vllm Qwen/Qwen3-32B"}
        assert picker.call_args.args[3] == ["qwen", "gpt", setup._OTHER_MODEL]

    def test_a_name_that_is_not_on_the_list_can_be_typed(
        self, console: MagicMock, theme: Theme, tmp_path: Path
    ) -> None:
        # 1. qwen, 2. gpt, 3. other
        with patch(
            "builtins.input",
            side_effect=[
                *_domyn_answers(key=None)[:4],
                "3",
                "typed-model",
                *["", "", "A", "i", ""],
            ],
        ):
            result = setup.run_setup(console, theme, path=tmp_path / "config.yaml")

        assert result is not None
        assert result["llm_providers"][0]["model_name"] == "typed-model"

    def test_a_failed_lookup_says_so_and_asks_for_the_name(
        self, console: MagicMock, theme: Theme, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def refuse(settings: PlatformSettings) -> list[dict[str, str]]:
            raise ConnectionError("down")

        monkeypatch.setattr(setup, "list_models", refuse)
        with patch("builtins.input", side_effect=_domyn_answers(key=None, model="typed-model")):
            result = setup.run_setup(console, theme, path=tmp_path / "config.yaml")

        assert result is not None
        assert result["llm_providers"][0]["model_name"] == "typed-model"
        assert "Could not list the models (ConnectionError)" in _printed(console)

    def test_an_http_error_names_its_status(
        self, console: MagicMock, theme: Theme, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class NotFoundError(Exception):
            response = MagicMock(status_code=404)

        async def missing(settings: PlatformSettings) -> list[dict[str, str]]:
            raise NotFoundError()

        monkeypatch.setattr(setup, "list_models", missing)
        with patch("builtins.input", side_effect=_domyn_answers(key=None, model="typed-model")):
            setup.run_setup(console, theme, path=tmp_path / "config.yaml")

        assert "Could not list the models (NotFoundError, HTTP 404)" in _printed(console)

    def test_an_sdk_that_is_too_old_is_named_with_the_way_to_fix_it(
        self, console: MagicMock, theme: Theme, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def too_old(settings: PlatformSettings) -> list[dict[str, str]]:
            raise PlatformSDKNotInstalledError("0.36.3")

        monkeypatch.setattr(setup, "list_models", too_old)
        with patch("builtins.input", side_effect=_domyn_answers(key=None, model="typed-model")):
            result = setup.run_setup(console, theme, path=tmp_path / "config.yaml")

        assert result is not None
        assert result["llm_providers"][0]["model_name"] == "typed-model"
        printed = _printed(console)
        assert "version 0.36.3 is installed" in printed
        assert "uv pip install" in printed

    def test_an_empty_space_asks_for_the_name(
        self, console: MagicMock, theme: Theme, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def nothing(settings: PlatformSettings) -> list[dict[str, str]]:
            return []

        monkeypatch.setattr(setup, "list_models", nothing)
        with patch("builtins.input", side_effect=_domyn_answers(key=None, model="typed-model")):
            result = setup.run_setup(console, theme, path=tmp_path / "config.yaml")

        assert result is not None
        assert result["llm_providers"][0]["model_name"] == "typed-model"

    def test_what_the_lookup_is_missing_is_named(
        self, console: MagicMock, theme: Theme, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("DOMYN_API_KEY")
        with patch("builtins.input", side_effect=_domyn_answers(space="", key="", model="m")):
            setup.run_setup(console, theme, path=tmp_path / "config.yaml")

        assert "Models: listing them needs the space id, an api key" in _printed(console)

    @pytest.mark.parametrize("missing", ["url", "space", "key", "sdk"])
    def test_nothing_is_asked_without_what_it_takes(
        self,
        console: MagicMock,
        theme: Theme,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        a_key_and_the_sdk: list[PlatformSettings],
        missing: str,
    ) -> None:
        if missing == "key":
            monkeypatch.delenv("DOMYN_API_KEY")
        if missing == "sdk":
            monkeypatch.setattr(setup, "sdk_installed", lambda: False)
        answers = _domyn_answers(
            url="" if missing == "url" else "https://p.example.com",
            space="" if missing == "space" else "space-1",
            key="" if missing == "key" else None,
            model="typed-model",
        )
        with patch("builtins.input", side_effect=answers):
            result = setup.run_setup(console, theme, path=tmp_path / "config.yaml")

        assert a_key_and_the_sdk == []
        assert result is not None
        assert result["llm_providers"][0]["model_name"] == "typed-model"


class TestUsableDomynConfig:
    def _raw(self, **provider: object) -> dict[str, Any]:
        return {
            "entrypoint": "A",
            "agents": [{"name": "A", "llm_provider": "p"}],
            "llm_providers": [{"name": "p", "type": "domyn", "model_name": "m", **provider}],
        }

    def test_the_url_and_the_space_are_needed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("DOMYN_SPACE_ID", raising=False)

        assert cfg.is_configuration_v1_usable(self._raw()) is False
        assert cfg.is_configuration_v1_usable(self._raw(url="https://p.example.com")) is False
        assert cfg.is_configuration_v1_usable(self._raw(url="https://p", space_id="s")) is True

    def test_the_space_variable_stands_in_for_a_missing_space(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("DOMYN_SPACE_ID", "env-space")

        assert cfg.is_configuration_v1_usable(self._raw(url="https://p")) is True


class TestMaxTokensAndContextWindow:
    def test_a_declared_value_round_trips_into_the_saved_config(
        self, console: MagicMock, theme: Theme, tmp_path: Path
    ) -> None:
        target = tmp_path / "config.yaml"
        with patch(
            "builtins.input",
            side_effect=["openai", "m", "", "", "8000", "32000", ""],
        ):
            result = setup.run_setup(console, theme, path=target)

        assert result is not None
        provider_dict = result["llm_providers"][0]
        assert provider_dict["generation_params"]["max_tokens"] == 8000
        assert provider_dict["context_window"] == 32000

    def test_an_empty_context_window_is_left_for_the_provider_to_report(
        self, console: MagicMock, theme: Theme, tmp_path: Path
    ) -> None:
        target = tmp_path / "config.yaml"
        with patch(
            "builtins.input",
            side_effect=["openai", "m", "", "", "", "", ""],
        ):
            result = setup.run_setup(console, theme, path=target)

        assert result is not None
        provider_dict = result["llm_providers"][0]
        assert provider_dict["generation_params"]["max_tokens"] == cfg.ProviderConfig().max_tokens
        assert "context_window" not in provider_dict

    def test_auto_is_shown_as_the_current_value_and_enter_keeps_it(
        self, console: MagicMock, theme: Theme
    ) -> None:
        with patch.object(setup, "_ask", return_value="auto") as ask:
            assert setup._ask_context_window(console, theme, None) is None

        assert ask.call_args.args[3] == "auto"

    def test_a_typed_150000_is_kept_and_shown_as_a_number(
        self, console: MagicMock, theme: Theme, tmp_path: Path
    ) -> None:
        with patch.object(setup, "_ask", return_value="150000") as ask:
            assert setup._ask_context_window(console, theme, cfg.DEFAULT_CONTEXT_WINDOW) == 150000
        assert ask.call_args.args[3] == "150000"

        target = tmp_path / "config.yaml"
        with patch(
            "builtins.input",
            side_effect=["openai", "m", "", "", "", "150000", "A", "i", "", ""],
        ):
            result = setup.run_setup(console, theme, path=target)

        assert result is not None
        assert result["llm_providers"][0]["context_window"] == 150000

    def test_typing_auto_resets_a_declared_window(self, console: MagicMock, theme: Theme) -> None:
        with patch.object(setup, "_ask", return_value="AUTO"):
            assert setup._ask_context_window(console, theme, 32000) is None

    def test_non_numeric_input_is_rejected_and_re_prompted(
        self, console: MagicMock, theme: Theme, tmp_path: Path
    ) -> None:
        target = tmp_path / "config.yaml"
        with patch(
            "builtins.input",
            side_effect=["openai", "m", "", "", "not-a-number", "4096", "", ""],
        ):
            result = setup.run_setup(console, theme, path=target)

        assert result is not None
        assert result["llm_providers"][0]["generation_params"]["max_tokens"] == 4096

    def test_a_saved_value_pre_fills_the_next_run(
        self, console: MagicMock, theme: Theme, tmp_path: Path
    ) -> None:
        target = tmp_path / "config.yaml"
        with patch(
            "builtins.input",
            side_effect=["openai", "m", "", "", "8000", "32000", ""],
        ):
            first = setup.run_setup(console, theme, path=target)

        # Bare-Enter on every prompt keeps whatever was already there.
        with patch(
            "builtins.input",
            side_effect=["", "", "", "", "", "", ""],
        ):
            second = setup.run_setup(console, theme, first, path=target)

        assert second is not None
        provider_dict = second["llm_providers"][0]
        assert provider_dict["generation_params"]["max_tokens"] == 8000
        assert provider_dict["context_window"] == 32000


class TestProviderOnly:
    def test_agent_fields_survive_a_provider_only_run(
        self, console: MagicMock, theme: Theme, tmp_path: Path
    ) -> None:
        target = tmp_path / "config.yaml"
        with patch("builtins.input", side_effect=["openai", "m1", "", "", "", "", ""]):
            first = setup.run_setup(console, theme, path=target)
        assert first is not None
        # Setup no longer asks for them, so they are hand-edited.
        first["entrypoint"] = first["agents"][0]["name"] = "MyAgent"
        first["agents"][0]["instruction"] = "My instruction"

        with patch("builtins.input", side_effect=["anthropic", "m2", "", "", "", ""]):
            second = setup.run_setup(console, theme, first, path=target, provider_only=True)

        assert second is not None
        assert second["agents"][0]["name"] == "MyAgent"
        assert second["agents"][0]["instruction"] == "My instruction"
        assert second["llm_providers"][0]["type"] == "anthropic"
        assert second["llm_providers"][0]["model_name"] == "m2"


class TestRenderConfig:
    def test_no_configuration_yet_is_reported_gracefully(
        self, console: MagicMock, theme: Theme
    ) -> None:
        setup.render_config(console, theme, None)
        printed = " ".join(str(c.args[0]) for c in console.print.call_args_list if c.args)
        assert "No configuration yet" in printed

    def test_renders_the_current_settings(
        self, console: MagicMock, theme: Theme, tmp_path: Path
    ) -> None:
        target = tmp_path / "config.yaml"
        with patch(
            "builtins.input",
            side_effect=["openai", "gpt-4o-mini", "", "", "", "", ""],
        ):
            result = setup.run_setup(console, theme, path=target)

        setup.render_config(console, theme, result)
        printed = " ".join(str(c.args[0]) for c in console.print.call_args_list if c.args)
        assert "gpt-4o-mini" in printed
        assert "planner_strategy" in printed


class TestTextFields:
    """Free-text fields: a line editor on a terminal, plain ``input()`` on a pipe.

    Bare ``input()`` does no editing without ``readline``, so ←/→ typed
    ``^[[D`` into the answer instead of moving the cursor.
    """

    def test_falls_back_to_input_without_a_tty(self, console: MagicMock, theme: Theme) -> None:
        with patch("builtins.input", return_value=" gpt-4o "):
            assert setup._ask(console, theme, "model") == "gpt-4o"
        assert console.print.call_args.args[0] == "  model: "

    @pytest.mark.asyncio
    async def test_the_editor_asks_for_its_own_thread(
        self, theme: Theme, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``/model`` reaches the fields from inside the REPL's loop, like the picker."""
        import prompt_toolkit
        from rich.console import Console

        calls: list[dict[str, Any]] = []

        class StubSession:
            def prompt(self, message: object, **kwargs: Any) -> str:
                calls.append({"message": message, **kwargs})
                return "typed"

        monkeypatch.setattr(prompt_toolkit, "PromptSession", StubSession)
        monkeypatch.setattr(prompt_mod, "interactive_terminal", lambda: True)

        result = setup._ask(Console(), theme, "paste key", password=True)

        assert result == "typed"
        assert calls and calls[0]["is_password"] is True
        # Without it the prompt raises inside a running loop — ``/model``'s case.
        assert calls[0]["in_thread"] is True
        # The editor draws the label itself, or its redraw would overwrite it.
        assert "paste key" in "".join(
            text for _, text in calls[0]["message"].__pt_formatted_text__()
        )

    def test_the_editor_opens_with_the_current_value_and_enter_keeps_it(
        self, theme: Theme, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import prompt_toolkit
        from rich.console import Console

        class StubSession:
            def prompt(self, message: object, **kwargs: Any) -> str:
                return str(kwargs["default"])

        monkeypatch.setattr(prompt_toolkit, "PromptSession", StubSession)
        monkeypatch.setattr(prompt_mod, "interactive_terminal", lambda: True)

        assert setup._ask(Console(), theme, "model", "gpt-4o", default="d") == "gpt-4o"

    def test_an_erased_field_in_the_editor_gives_the_default(
        self, theme: Theme, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The only way back to a default once a value was saved."""
        import prompt_toolkit
        from rich.console import Console

        class StubSession:
            def prompt(self, message: object, **kwargs: Any) -> str:
                return "  "

        monkeypatch.setattr(prompt_toolkit, "PromptSession", StubSession)
        monkeypatch.setattr(prompt_mod, "interactive_terminal", lambda: True)

        assert setup._ask(Console(), theme, "url", "https://gw", default="") == ""
        assert setup._ask(Console(), theme, "max", "32000", default="16000") == "16000"

    def test_a_pipe_shows_the_current_value_and_enter_keeps_it(
        self, console: MagicMock, theme: Theme
    ) -> None:
        """Nothing can be pre-filled without a terminal, so Enter keeps what is shown."""
        with patch("builtins.input", return_value=" "):
            assert setup._ask(console, theme, "model", "gpt-4o", default="d") == "gpt-4o"
        assert console.print.call_args.args[0] == "  model: [gpt-4o] "

        with patch("builtins.input", return_value="other"):
            assert setup._ask(console, theme, "model", "gpt-4o") == "other"

    def test_ctrl_c_in_the_editor_reaches_the_caller(
        self, theme: Theme, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import prompt_toolkit
        from rich.console import Console

        class StubSession:
            def prompt(self, message: object, **kwargs: Any) -> str:
                raise KeyboardInterrupt

        monkeypatch.setattr(prompt_toolkit, "PromptSession", StubSession)
        monkeypatch.setattr(prompt_mod, "interactive_terminal", lambda: True)

        with pytest.raises(KeyboardInterrupt):
            setup._ask(Console(), theme, "model")

    def test_the_api_key_is_asked_for_hidden(
        self, console: MagicMock, theme: Theme, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        asked: list[tuple[str, bool]] = []
        answers = iter(["gpt-4o-mini", "", "", "", "", "", ""])

        def fake_ask(
            console: object,
            theme: object,
            label: str,
            current: str = "",
            *,
            default: str = "",
            password: bool = False,
        ) -> str:
            asked.append((label, password))
            return next(answers) or current

        monkeypatch.setattr(setup, "_ask", fake_ask)
        with patch("builtins.input", return_value="openai"):
            setup.run_setup(console, theme, path=tmp_path / "config.yaml")

        hidden = [label for label, password in asked if password]
        assert hidden == ["Paste key (empty = skip)"]

    def test_ctrl_c_in_a_field_aborts_without_saving(
        self, console: MagicMock, theme: Theme, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def interrupted(*args: object, **kwargs: object) -> str:
            raise KeyboardInterrupt

        monkeypatch.setattr(setup, "_ask", interrupted)
        target = tmp_path / "config.yaml"
        with patch("builtins.input", return_value="openai"):
            assert setup.run_setup(console, theme, path=target) is None
        assert not target.exists()


def _erasing(*labels: str) -> Any:
    """A line editor where the fields whose label starts with one of ``labels`` are
    erased, and every other is left as it opened."""

    def read_line(console: object, message: str, *, prefill: str = "", **kwargs: object) -> str:
        return "" if message.strip().startswith(labels) else prefill

    return read_line


class TestErasingAFieldGivesItsDefault:
    """On a re-run every field opens with its saved value; erased, it is the default."""

    @pytest.fixture(autouse=True)
    def clean_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for name in ("DOMYN_API_KEY", "DOMYN_SPACE_ID", "DOMYN_BASE_URL", "ANTHROPIC_API_KEY"):
            monkeypatch.setenv(name, "")
            monkeypatch.delenv(name)
        monkeypatch.setattr(setup, "sdk_installed", lambda: False)

    def test_the_provider_url_max_tokens_and_window_go_back_to_their_defaults(
        self, console: MagicMock, theme: Theme, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        existing = cfg.configuration_v1_dict_from(
            cfg.ProviderConfig(
                type="anthropic",
                model_name="m",
                url="https://gw.example.com/v1/messages",
                max_tokens=32000,
                context_window=64000,
            ),
            cfg.AgentConfig(),
            cfg.CompactionConfig(),
        )
        monkeypatch.setattr(setup, "read_line", _erasing("URL", "Max tokens", "Context window"))
        with patch("builtins.input", return_value=""):
            result = setup.run_setup(console, theme, existing, path=tmp_path / "config.yaml")

        assert result is not None
        provider = result["llm_providers"][0]
        assert (provider["type"], provider["model_name"]) == ("anthropic", "m")
        assert "url" not in provider
        assert provider["generation_params"]["max_tokens"] == cfg.ProviderConfig().max_tokens
        assert "context_window" not in provider

    def test_the_domyn_configuration_id_goes_back_to_the_last_released(
        self, console: MagicMock, theme: Theme, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        existing = cfg.configuration_v1_dict_from(
            cfg.ProviderConfig(
                type="domyn",
                model_name="qwen",
                url="https://p.example.com",
                space_id="space-1",
                configuration_id="cfg-1",
            ),
            cfg.AgentConfig(),
            cfg.CompactionConfig(),
        )
        monkeypatch.setattr(setup, "read_line", _erasing("Configuration id"))
        with patch("builtins.input", return_value=""):
            result = setup.run_setup(console, theme, existing, path=tmp_path / "config.yaml")

        assert result is not None
        provider = result["llm_providers"][0]
        assert (provider["url"], provider["space_id"]) == ("https://p.example.com", "space-1")
        assert "configuration_id" not in provider

    def test_an_erased_platform_setting_falls_back_to_its_variable(
        self, console: MagicMock, theme: Theme, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("DOMYN_SPACE_ID", "env-space")
        monkeypatch.setattr(setup, "read_line", _erasing("Space id"))

        assert (
            setup._ask_setting(console, theme, "Space id", "saved-space", "DOMYN_SPACE_ID")
            == "env-space"
        )

    def test_a_saved_platform_setting_is_kept_as_it_opened(
        self, console: MagicMock, theme: Theme, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("DOMYN_SPACE_ID", "env-space")
        monkeypatch.setattr(setup, "read_line", _erasing())

        assert (
            setup._ask_setting(console, theme, "Space id", "saved-space", "DOMYN_SPACE_ID")
            == "saved-space"
        )


FIELDS = ["openai", "m", "", "", "", ""]
"""Answers up to and including the context window; the skills question follows."""


def _skill_names() -> list[str]:
    from domyn.agents.skills.catalog import list_builtin_skills

    return [skill.name for skill in list_builtin_skills()]


def _number(name: str) -> str:
    """The number the picker prints for a skill or a group: every row counts, groups too."""
    from domyn.agents.skills.catalog import skill_groups

    rows = select_mod._TickList(_skill_names(), skill_groups()).rows
    return str(next(i for i, row in enumerate(rows, start=1) if row.label == name))


def _listed(config: dict[str, Any]) -> list[str] | None:
    skills = config["agents"][0].get("skills")
    return None if skills is None else skills["source"]["paths"]


class TestChoosingTheSkills:
    def test_a_first_setup_installs_every_builtin_and_lists_each_by_absolute_path(
        self, console: MagicMock, theme: Theme, tmp_path: Path
    ) -> None:
        with patch("builtins.input", side_effect=[*FIELDS, ""]):
            result = setup.run_setup(console, theme, path=tmp_path / "config.yaml")

        assert result is not None
        paths = _listed(result)
        assert paths is not None
        install_dir = cfg.skills_install_dir()
        assert all(Path(p).is_absolute() and Path(p).is_dir() for p in paths)
        assert all(Path(p).parent.parent == install_dir for p in paths)
        assert sorted(Path(p).name for p in paths) == sorted(_skill_names())

    def test_the_skills_are_where_the_config_says(
        self, console: MagicMock, theme: Theme, tmp_path: Path
    ) -> None:
        with patch("builtins.input", side_effect=[*FIELDS, _number("charts")]):
            result = setup.run_setup(console, theme, path=tmp_path / "config.yaml")

        assert result is not None
        install_dir = cfg.skills_install_dir()
        assert _listed(result) == [str(install_dir / "visualization" / "charts")]
        assert (install_dir / "visualization" / "charts" / "SKILL.md").is_file()
        assert (install_dir / "resources" / "seaborn_style.py").is_file()
        assert not (install_dir / "data").exists()

    def test_choosing_none_empties_the_directory_and_lists_nothing(
        self, console: MagicMock, theme: Theme, tmp_path: Path
    ) -> None:
        with patch("builtins.input", side_effect=[*FIELDS, "none"]):
            result = setup.run_setup(console, theme, path=tmp_path / "config.yaml")

        assert result is not None
        assert _listed(result) is None
        assert list(cfg.skills_install_dir().iterdir()) == []

    def test_a_second_setup_starts_with_the_current_choice_ticked(
        self, console: MagicMock, theme: Theme, tmp_path: Path
    ) -> None:
        target = tmp_path / "config.yaml"
        with patch("builtins.input", side_effect=[*FIELDS, _number("charts")]):
            first = setup.run_setup(console, theme, path=target)
        assert first is not None

        # Empty answers all the way: keep every value, and keep what is ticked.
        with patch("builtins.input", side_effect=[""] * 7):
            second = setup.run_setup(console, theme, first, path=target)

        assert second is not None
        assert _listed(second) == _listed(first)

    def test_unticking_a_skill_removes_it_from_the_directory_and_the_config(
        self, console: MagicMock, theme: Theme, tmp_path: Path
    ) -> None:
        target = tmp_path / "config.yaml"
        both = f"{_number('charts')},{_number('brainstorming')}"
        with patch("builtins.input", side_effect=[*FIELDS, both]):
            first = setup.run_setup(console, theme, path=target)
        assert first is not None
        assert len(_listed(first) or []) == 2

        with patch("builtins.input", side_effect=[*[""] * 6, _number("brainstorming")]):
            second = setup.run_setup(console, theme, first, path=target)

        assert second is not None
        assert [Path(p).name for p in _listed(second) or []] == ["brainstorming"]
        install_dir = cfg.skills_install_dir()
        assert not (install_dir / "visualization").exists()
        assert not (install_dir / "resources").exists()

    def test_a_provider_only_run_leaves_the_skills_alone(
        self, console: MagicMock, theme: Theme, tmp_path: Path
    ) -> None:
        target = tmp_path / "config.yaml"
        with patch("builtins.input", side_effect=[*FIELDS, _number("charts")]):
            first = setup.run_setup(console, theme, path=target)
        assert first is not None

        with patch("builtins.input", side_effect=["anthropic", "m2", "", "", "", ""]):
            second = setup.run_setup(console, theme, first, path=target, provider_only=True)

        assert second is not None
        assert _listed(second) == _listed(first)
        assert (cfg.skills_install_dir() / "visualization" / "charts").is_dir()

    def test_aborting_the_picker_saves_nothing(
        self, console: MagicMock, theme: Theme, tmp_path: Path
    ) -> None:
        target = tmp_path / "config.yaml"
        with patch("builtins.input", side_effect=[*FIELDS, EOFError]):
            assert setup.run_setup(console, theme, path=target) is None
        assert not target.exists()

    def test_an_unusable_answer_three_times_gives_up_instead_of_looping(
        self, console: MagicMock, theme: Theme, tmp_path: Path
    ) -> None:
        target = tmp_path / "config.yaml"
        with patch("builtins.input", side_effect=[*FIELDS, "x", "y", "99"]):
            assert setup.run_setup(console, theme, path=target) is None
        assert not target.exists()


class TestWhatTheRewriteWouldDelete:
    @pytest.fixture
    def stray(self) -> Path:
        directory = cfg.skills_install_dir() / "mine"
        directory.mkdir(parents=True)
        (directory / "SKILL.md").write_text("my own skill")
        return directory

    def test_declining_leaves_the_directory_and_the_config_as_they_were(
        self, stray: Path, console: MagicMock, theme: Theme, tmp_path: Path
    ) -> None:
        with patch("builtins.input", side_effect=[*FIELDS, "", ""]):  # keep ticks, default "no"
            result = setup.run_setup(console, theme, path=tmp_path / "config.yaml")

        assert result is not None
        assert _listed(result) is None
        assert (stray / "SKILL.md").read_text() == "my own skill"

    def test_it_is_named_before_it_is_deleted(
        self, stray: Path, console: MagicMock, theme: Theme, tmp_path: Path
    ) -> None:
        with patch("builtins.input", side_effect=[*FIELDS, "", ""]):
            setup.run_setup(console, theme, path=tmp_path / "config.yaml")

        printed = " ".join(str(call.args[0]) for call in console.print.call_args_list if call.args)
        assert "mine" in printed

    def test_agreeing_replaces_it_with_the_choice(
        self, stray: Path, console: MagicMock, theme: Theme, tmp_path: Path
    ) -> None:
        with patch("builtins.input", side_effect=[*FIELDS, _number("charts"), "y"]):
            result = setup.run_setup(console, theme, path=tmp_path / "config.yaml")

        assert result is not None
        assert not stray.exists()
        assert len(_listed(result) or []) == 1

    def test_nothing_is_asked_when_there_is_nothing_foreign(
        self, console: MagicMock, theme: Theme, tmp_path: Path
    ) -> None:
        target = tmp_path / "config.yaml"
        with patch("builtins.input", side_effect=[*FIELDS, _number("charts")]):
            first = setup.run_setup(console, theme, path=target)
        assert first is not None

        # Exactly the answers of a run without the confirmation: one more would be
        # unread, one fewer would end the script with StopIteration.
        with patch("builtins.input", side_effect=[""] * 7) as answers:
            setup.run_setup(console, theme, first, path=target)
        assert answers.call_count == 7


class TestWhatTheConfigAlreadyHolds:
    def test_a_path_elsewhere_is_kept_next_to_the_installed_ones(
        self, console: MagicMock, theme: Theme, tmp_path: Path
    ) -> None:
        mine = tmp_path / "team-skills" / "review"
        existing = cfg.configuration_v1_dict_from(
            cfg.ProviderConfig(type="openai", model_name="m"),
            cfg.AgentConfig(
                name="A",
                instruction="i",
                skills={"source": {"type": "local_dir", "paths": [str(mine)]}},
            ),
            cfg.CompactionConfig(),
        )

        with patch("builtins.input", side_effect=[*[""] * 6, _number("charts")]):
            result = setup.run_setup(console, theme, existing, path=tmp_path / "config.yaml")

        assert result is not None
        paths = _listed(result)
        assert paths is not None
        assert paths[-1] == str(mine)
        assert len(paths) == 2

    def test_a_source_that_is_not_a_local_directory_is_left_alone(
        self, console: MagicMock, theme: Theme, tmp_path: Path
    ) -> None:
        remote = {"source": {"type": "remote", "url": "https://skills.example"}}
        existing = cfg.configuration_v1_dict_from(
            cfg.ProviderConfig(type="openai", model_name="m"),
            cfg.AgentConfig(name="A", instruction="i", skills=remote),
            cfg.CompactionConfig(),
        )

        # No answer for the skills question: it is not asked.
        with patch("builtins.input", side_effect=[""] * 6):
            result = setup.run_setup(console, theme, existing, path=tmp_path / "config.yaml")

        assert result is not None
        assert result["agents"][0]["skills"] == remote
        assert not cfg.skills_install_dir().exists()

    def test_a_symlinked_directory_is_reported_and_the_config_unchanged(
        self, console: MagicMock, theme: Theme, tmp_path: Path
    ) -> None:
        target = tmp_path / "my-skills-repo"
        target.mkdir()
        (target / "precious.md").write_text("x")
        install_dir = cfg.skills_install_dir()
        install_dir.parent.mkdir(parents=True, exist_ok=True)
        install_dir.symlink_to(target)

        with patch("builtins.input", side_effect=[*FIELDS, ""]):
            result = setup.run_setup(console, theme, path=tmp_path / "config.yaml")

        assert result is not None
        assert _listed(result) is None
        assert (target / "precious.md").read_text() == "x"


class TestTheNumberedPickerOnARealConsole:
    """A `MagicMock` console prints nothing, which hid that rich reads `[x]` as a tag
    and drew the list with no tick boxes at all."""

    def test_the_boxes_are_drawn_and_a_description_with_brackets_survives(
        self, theme: Theme
    ) -> None:
        import io

        from rich.console import Console

        buffer = io.StringIO()
        console = Console(file=buffer, width=200)
        with patch("builtins.input", return_value=""):
            chosen = select_mod.multiselect(
                console,
                theme,
                "pick",
                ["a", "b"],
                selected=["b"],
                descriptions={"a": "see [docs] for more"},
            )

        shown = buffer.getvalue()
        assert chosen == ["b"]
        assert "[ ] 1) a" in shown
        assert "[x] 2) b" in shown
        assert "see [docs] for more" in shown
        assert "ticked ones" in shown

    def test_the_single_choice_prompt_shows_its_default(self, theme: Theme) -> None:
        """`[no]` was read as a rich tag and the prompt printed as "choice :", so the
        default of a question that deletes files was not on screen."""
        import io

        from rich.console import Console

        buffer = io.StringIO()
        with patch("builtins.input", return_value=""):
            answer = select_mod.select(
                Console(file=buffer, width=100), theme, "ok?", ["no", "yes"], current="no"
            )

        assert answer == "no"
        assert "choice [no]:" in buffer.getvalue()


GROUPS = {"a": ["a1", "a2", "a3"], "b": ["b1", "b2"]}
OPTIONS = ["a1", "a2", "a3", "b1", "b2"]


def _ticks(selected: tuple[str, ...] = ()) -> Any:
    return select_mod._TickList(OPTIONS, GROUPS, selected)


def _row(ticks: Any, label: str) -> int:
    return next(i for i, row in enumerate(ticks.rows) if row.label == label)


class TestTheTickList:
    """What both pickers are drawn from, with no terminal."""

    def test_a_group_row_comes_first_with_its_options_indented_below_it(self) -> None:
        rows = _ticks().rows

        assert [(r.label, r.is_group, r.indent) for r in rows] == [
            ("a", True, 0),
            ("a1", False, 1),
            ("a2", False, 1),
            ("a3", False, 1),
            ("b", True, 0),
            ("b1", False, 1),
            ("b2", False, 1),
        ]

    def test_ticking_a_group_ticks_all_its_options_and_nothing_else(self) -> None:
        ticks = _ticks()

        ticks.toggle(_row(ticks, "a"))

        assert ticks.selected() == ["a1", "a2", "a3"]

    def test_ticking_a_group_again_clears_it(self) -> None:
        ticks = _ticks()
        ticks.toggle(_row(ticks, "a"))

        ticks.toggle(_row(ticks, "a"))

        assert ticks.selected() == []

    def test_ticking_a_partly_ticked_group_ticks_the_rest(self) -> None:
        ticks = _ticks(("a2",))

        ticks.toggle(_row(ticks, "a"))

        assert ticks.selected() == ["a1", "a2", "a3"]

    def test_one_option_toggles_alone_and_the_group_follows(self) -> None:
        ticks = _ticks()
        group = _row(ticks, "a")
        assert ticks.state(group) == "none"

        ticks.toggle(_row(ticks, "a1"))
        assert ticks.state(group) == "some"
        assert ticks.selected() == ["a1"]

        ticks.toggle(_row(ticks, "a2"))
        ticks.toggle(_row(ticks, "a3"))
        assert ticks.state(group) == "all"

        ticks.toggle(_row(ticks, "a2"))
        assert ticks.state(group) == "some"
        assert ticks.selected() == ["a1", "a3"]

    def test_a_group_is_derived_so_it_starts_from_the_options_ticked(self) -> None:
        ticks = _ticks(("b1", "b2", "a1"))

        assert ticks.state(_row(ticks, "a")) == "some"
        assert ticks.state(_row(ticks, "b")) == "all"

    def test_only_options_are_ever_returned(self) -> None:
        ticks = _ticks()
        ticks.toggle(_row(ticks, "a"))
        ticks.toggle(_row(ticks, "b"))

        assert ticks.selected() == OPTIONS
        assert "a" not in ticks.ticked and "b" not in ticks.ticked

    def test_selected_follows_the_order_the_options_were_given(self) -> None:
        ticks = select_mod._TickList(["z", "m", "a"], None, ["a", "z"])

        assert ticks.selected() == ["z", "a"]

    def test_an_option_in_no_group_is_still_reachable(self) -> None:
        ticks = select_mod._TickList([*OPTIONS, "loose"], GROUPS)

        assert ticks.rows[-1].label == "loose" and not ticks.rows[-1].is_group
        ticks.toggle(len(ticks.rows) - 1)
        assert ticks.selected() == ["loose"]

    def test_all_and_none(self) -> None:
        ticks = _ticks()

        ticks.toggle_all()
        assert ticks.selected() == OPTIONS
        ticks.toggle_all()
        assert ticks.selected() == []

    def test_without_groups_it_is_the_flat_list_it_was(self) -> None:
        ticks = select_mod._TickList(OPTIONS)

        assert [r.label for r in ticks.rows] == OPTIONS
        assert not any(r.is_group or r.indent for r in ticks.rows)

    @pytest.mark.parametrize(
        "groups",
        [
            {"a1": ["a2"]},  # a group named like an option
            {"a": ["nope"]},  # an unknown member
            {"a": ["a1"], "b": ["a1"]},  # an option in two groups
        ],
    )
    def test_a_group_that_does_not_make_sense_is_refused(
        self, groups: dict[str, list[str]]
    ) -> None:
        with pytest.raises(ValueError):
            select_mod._TickList(OPTIONS, groups)

    def test_an_empty_group_is_not_drawn(self) -> None:
        ticks = select_mod._TickList(OPTIONS, {"empty": [], **GROUPS})

        assert "empty" not in [r.label for r in ticks.rows]


class TestDrawingGroups:
    @staticmethod
    def _lines(theme: Theme, ticks: Any, index: int = 0) -> list[str]:
        rows = select_mod.multi_choice_rows(
            theme, "pick", ticks, index, descriptions={"a": "3 skills", "a1": "first"}
        )
        return "".join(text for _, text in rows).splitlines()

    @staticmethod
    def _plain(theme: Theme) -> Theme:
        """The same theme on a terminal that cannot draw the Unicode boxes."""
        import dataclasses

        return dataclasses.replace(
            theme, capabilities=dataclasses.replace(theme.capabilities, unicode=False)
        )

    @staticmethod
    def _unicode(theme: Theme) -> Theme:
        import dataclasses

        return dataclasses.replace(
            theme, capabilities=dataclasses.replace(theme.capabilities, unicode=True)
        )

    def test_the_three_states_are_drawn(self, theme: Theme) -> None:
        ticks = _ticks(("a1", "b1", "b2"))

        lines = self._lines(self._unicode(theme), ticks)

        assert any("\u25d0 a" in line for line in lines)  # some of a
        assert any("\u25c9 b" in line for line in lines)  # all of b
        assert any("\u25c9 a1" in line for line in lines)
        assert any("\u25ef a2" in line for line in lines)

    def test_an_option_is_indented_further_than_its_group(self, theme: Theme) -> None:
        lines = self._lines(self._unicode(theme), _ticks())
        group = next(line for line in lines if " a  " in line or line.rstrip().endswith(" a"))
        option = next(line for line in lines if " a1" in line)

        def box_column(line: str) -> int:
            return min(line.index(glyph) for glyph in "\u25c9\u25d0\u25ef" if glyph in line)

        assert box_column(option) > box_column(group)

    def test_a_group_carries_its_description_like_an_option(self, theme: Theme) -> None:
        lines = self._lines(self._unicode(theme), _ticks())

        assert any(" a " in line and "3 skills" in line for line in lines)
        assert any(" a1" in line and "first" in line for line in lines)

    def test_a_plain_terminal_gets_ascii_boxes(self, theme: Theme) -> None:
        ticks = _ticks(("a1", "b1", "b2"))

        text = "\n".join(self._lines(self._plain(theme), ticks))

        assert "[-] a" in text and "[x] b" in text and "[ ] a2" in text
        assert not any(glyph in text for glyph in "\u25c9\u25d0\u25ef")

    def test_the_cursor_marks_the_row_it_is_on(self, theme: Theme) -> None:
        ticks = _ticks()

        lines = self._lines(self._unicode(theme), ticks, index=_row(ticks, "a2"))

        marked = [line for line in lines if line.lstrip().startswith("\u276f")]
        assert len(marked) == 1 and "a2" in marked[0]


class TestTheNumberedAnswerWithGroups:
    @pytest.mark.parametrize(
        ("answer", "expected"),
        [
            ("1", ["a1", "a2", "a3"]),  # a group's number takes all its options
            ("5", ["b1", "b2"]),
            ("2,4", ["a1", "a3"]),  # two options of one group
            ("1-3", ["a1", "a2", "a3"]),  # a range covers a group and what is in it
            ("3-6", ["a2", "a3", "b1", "b2"]),  # row 5 is the group b: all of b comes with it
            ("1,5", ["a1", "a2", "a3", "b1", "b2"]),
            ("all", OPTIONS),
            ("none", []),
        ],
    )
    def test_numbers_name_rows_groups_included(self, answer: str, expected: list[str]) -> None:
        assert select_mod.parse_numbers(answer, _ticks()) == expected

    @pytest.mark.parametrize("answer", ["0", "8", "x", "1-", "3-1", "2,,3"])
    def test_what_is_not_a_row_is_unusable(self, answer: str) -> None:
        assert select_mod.parse_numbers(answer, _ticks()) is None

    def test_the_printed_list_numbers_every_row_and_shows_a_groups_state(
        self, theme: Theme
    ) -> None:
        import io

        from rich.console import Console

        buffer = io.StringIO()
        with patch("builtins.input", return_value=""):
            chosen = select_mod.multiselect(
                Console(file=buffer, width=120),
                theme,
                "pick",
                OPTIONS,
                selected=["a1", "b1", "b2"],
                groups=GROUPS,
                descriptions={"a": "3 skills"},
            )

        shown = buffer.getvalue()
        assert chosen == ["a1", "b1", "b2"]
        assert "[-] 1) a  — 3 skills" in shown
        assert "[x] 2) a1" in shown and "[ ] 3) a2" in shown
        assert "[x] 5) b" in shown

    def test_an_answer_that_is_a_group_number_selects_the_group(self, theme: Theme) -> None:
        with patch("builtins.input", return_value="5"):
            chosen = select_mod.multiselect(
                MagicMock(), theme, "pick", OPTIONS, groups=GROUPS, max_tries=1
            )

        assert chosen == ["b1", "b2"]

    def test_three_bad_answers_give_up(self, theme: Theme) -> None:
        with patch("builtins.input", side_effect=["99", "x", "0"]):
            chosen = select_mod.multiselect(
                MagicMock(), theme, "pick", OPTIONS, groups=GROUPS, max_tries=3
            )

        assert chosen is None


class TestChoosingAGroupInSetup:
    def _group_skills(self, group: str) -> list[str]:
        from domyn.agents.skills.catalog import skill_groups

        return skill_groups()[group]

    def test_a_group_number_installs_all_its_skills(
        self, console: MagicMock, theme: Theme, tmp_path: Path
    ) -> None:
        with patch("builtins.input", side_effect=[*FIELDS, _number("productivity")]):
            result = setup.run_setup(console, theme, path=tmp_path / "config.yaml")

        assert result is not None
        listed = sorted(Path(p).name for p in _listed(result) or [])
        assert listed == sorted(self._group_skills("productivity"))
        assert len(listed) == 4

    def test_one_skill_of_a_group_lists_only_that_one(
        self, console: MagicMock, theme: Theme, tmp_path: Path
    ) -> None:
        with patch("builtins.input", side_effect=[*FIELDS, _number("brainstorming")]):
            result = setup.run_setup(console, theme, path=tmp_path / "config.yaml")

        assert result is not None
        assert [Path(p).name for p in _listed(result) or []] == ["brainstorming"]

    def test_a_group_and_a_loose_skill_from_another_group_combine(
        self, console: MagicMock, theme: Theme, tmp_path: Path
    ) -> None:
        answer = f"{_number('productivity')},{_number('charts')}"
        with patch("builtins.input", side_effect=[*FIELDS, answer]):
            result = setup.run_setup(console, theme, path=tmp_path / "config.yaml")

        assert result is not None
        assert len(_listed(result) or []) == 5

    def test_a_rerun_with_a_whole_group_chosen_keeps_it_on_an_empty_answer(
        self, console: MagicMock, theme: Theme, tmp_path: Path
    ) -> None:
        target = tmp_path / "config.yaml"
        with patch("builtins.input", side_effect=[*FIELDS, _number("productivity")]):
            first = setup.run_setup(console, theme, path=target)
        assert first is not None

        with patch("builtins.input", side_effect=[""] * 7):
            second = setup.run_setup(console, theme, first, path=target)

        assert second is not None
        assert _listed(second) == _listed(first)

    def test_the_picker_offers_every_category_as_a_group(
        self, console: MagicMock, theme: Theme, tmp_path: Path
    ) -> None:
        import io

        from rich.console import Console

        from domyn.agents.skills.catalog import skill_groups

        buffer = io.StringIO()
        with patch("builtins.input", side_effect=[*FIELDS, ""]):
            setup.run_setup(Console(file=buffer, width=160), theme, path=tmp_path / "config.yaml")

        shown = buffer.getvalue()
        for group, members in skill_groups().items():
            noun = "skill" if len(members) == 1 else "skills"
            assert f"{group}  — {len(members)} {noun}" in shown


class TestAVLLMUrlIsCompleted:
    """vLLM is posted to at its `url` as given: setup turns a server's base URL into the
    chat completions endpoint, so the first message is not a 404."""

    @pytest.mark.parametrize(
        ("typed", "saved"),
        [
            ("http://host:8000", "http://host:8000/v1/chat/completions"),
            ("http://host:8000/", "http://host:8000/v1/chat/completions"),
            ("http://host:8000/v1", "http://host:8000/v1/chat/completions"),
            ("http://host:8000/v1/", "http://host:8000/v1/chat/completions"),
            ("http://host:8000/v1/chat/completions", "http://host:8000/v1/chat/completions"),
            ("https://gw/serve/chat/completions", "https://gw/serve/chat/completions"),
            ("https://gw/vllm/qwen?key=1", "https://gw/vllm/qwen/v1/chat/completions?key=1"),
        ],
    )
    def test_the_saved_url(
        self, console: MagicMock, theme: Theme, tmp_path: Path, typed: str, saved: str
    ) -> None:
        with patch("builtins.input", side_effect=["vllm", "m", typed, "", "", "", ""]):
            result = setup.run_setup(console, theme, path=tmp_path / "config.yaml")

        assert result is not None
        assert result["llm_providers"][0]["url"] == saved

    def test_another_provider_keeps_its_url_as_typed(
        self, console: MagicMock, theme: Theme, tmp_path: Path
    ) -> None:
        with patch(
            "builtins.input",
            side_effect=["custom", "m", "http://host:8000", "", "", "", ""],
        ):
            result = setup.run_setup(console, theme, path=tmp_path / "config.yaml")

        assert result is not None
        assert result["llm_providers"][0]["url"] == "http://host:8000"

    def test_the_completed_url_is_shown_without_its_credentials(
        self, theme: Theme, tmp_path: Path
    ) -> None:
        import io

        from rich.console import Console

        buffer = io.StringIO()
        with patch(
            "builtins.input",
            side_effect=["vllm", "m", "http://user:pw@host:8000", "", "", "", ""],
        ):
            setup.run_setup(Console(file=buffer, width=160), theme, path=tmp_path / "config.yaml")

        assert "URL: http://***@host:8000/v1/chat/completions" in buffer.getvalue()
        assert "pw@" not in buffer.getvalue()
