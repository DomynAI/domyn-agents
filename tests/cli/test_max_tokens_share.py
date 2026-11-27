"""`max_tokens_share` in the CLI's configuration.

The provider lowers a `max_tokens` that would claim more than its share of the window
(`BaseLLMProvider.max_tokens_share`, tested in `tests/llm`). Here: the setting
round-trips through the wizard, `/config` shows it, and a window the CLI fixes after
the provider is built (asked of the endpoint, or the 150k default) lowers it too, at
startup and on a reload.
"""

from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import yaml

from domyn.agents.interfaces.cli import config as cfg
from domyn.agents.llm import LLMMessage, LLMProviderFactory
from domyn.agents.llm.base import DEFAULT_MAX_TOKENS_SHARE, BaseLLMProvider
from domyn.agents.planner_strategy import Compactor

FLINT = 262_144


def _written(**fields: Any) -> dict[str, Any]:
    provider = cfg.ProviderConfig(
        type="vllm", url="http://vllm.test/v1/chat/completions", model_name="flint", **fields
    )
    return cfg.configuration_v1_dict_from(provider, cfg.AgentConfig(), cfg.CompactionConfig())


def _provider(written: dict[str, Any]) -> BaseLLMProvider:
    return LLMProviderFactory.from_config(written["llm_providers"][0])


async def _applied(reported: int | None = None, **fields: Any) -> BaseLLMProvider:
    provider = _provider(_written(**fields))
    with patch.object(type(provider), "fetch_context_window", return_value=reported):
        await cfg.apply_context_window(provider)
    return provider


class TestTheWindowTheSessionFixes:
    def test_a_declared_window_lowers_it_when_the_provider_is_built(self) -> None:
        provider = _provider(_written(context_window=FLINT, max_tokens=FLINT))
        assert provider.max_reply_tokens == FLINT // 2

    @pytest.mark.asyncio
    async def test_a_window_the_endpoint_reports(self) -> None:
        provider = await _applied(reported=FLINT, max_tokens=FLINT)
        assert provider.max_reply_tokens == FLINT // 2

    @pytest.mark.asyncio
    async def test_the_default_window(self) -> None:
        """Nobody declared the window and nothing answered: 150k is the session's."""
        provider = await _applied(max_tokens=100_000)
        assert provider.context_window == cfg.DEFAULT_CONTEXT_WINDOW
        assert provider.max_reply_tokens == cfg.DEFAULT_CONTEXT_WINDOW // 2

    @pytest.mark.asyncio
    async def test_the_cli_default_is_left_alone(self) -> None:
        provider = await _applied(reported=FLINT)
        assert provider.max_reply_tokens == cfg.ProviderConfig().max_tokens

    @pytest.mark.asyncio
    async def test_the_reserve_follows_so_the_prompt_keeps_half(self) -> None:
        provider = await _applied(reported=FLINT, max_tokens=FLINT)

        class _Agent:
            llm_provider = provider
            name = "A"

        plan = Compactor().assess([LLMMessage(role="user", content="hi")], _Agent())
        assert plan.threshold_tokens == FLINT // 2

    def test_the_file_is_not_rewritten(self) -> None:
        written = _written(context_window=FLINT, max_tokens=FLINT)
        _provider(written)
        assert written["llm_providers"][0]["generation_params"]["max_tokens"] == FLINT


class TestTheSetting:
    def test_the_default_is_not_written(self) -> None:
        assert "max_tokens_share" not in _written()["llm_providers"][0]

    def test_another_share_round_trips_and_reaches_the_provider(self) -> None:
        written = _written(max_tokens_share=0.25, context_window=FLINT, max_tokens=FLINT)
        assert written["llm_providers"][0]["max_tokens_share"] == 0.25
        provider_config, _, _ = cfg.current_settings_from(written)
        assert provider_config.max_tokens_share == 0.25
        assert _provider(written).max_reply_tokens == FLINT // 4

    @pytest.mark.parametrize("value", [0, 1.5, -1, "half", True])
    def test_the_wizard_opens_on_a_value_out_of_range(
        self, value: Any, caplog: pytest.LogCaptureFixture
    ) -> None:
        written = _written()
        written["llm_providers"][0]["max_tokens_share"] = value
        with caplog.at_level("WARNING", logger="domyn.agents"):
            provider_config, _, _ = cfg.current_settings_from(written)
        assert provider_config.max_tokens_share == DEFAULT_MAX_TOKENS_SHARE
        assert any("max_tokens_share" in r.message for r in caplog.records)

    def test_loading_refuses_it(self) -> None:
        written = _written()
        written["llm_providers"][0]["max_tokens_share"] = 1.5
        with pytest.raises(ValueError):
            _provider(written)

    def test_shown_by_config(self) -> None:
        described = cfg.describe_configuration_v1(_written(max_tokens_share=0.25))
        fields = [f for s in described.sections for i in s.items for f in i.fields]
        assert ("max tokens share", "0.25 of the window") in fields


class TestAReloadedConfiguration:
    @staticmethod
    def _save(path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(_written(max_tokens=100_000)), encoding="utf-8")

    @pytest.mark.asyncio
    async def test_the_cli_configuration_gets_the_window_again(self) -> None:
        """A config edit rebuilds the agent (`load_session_agent`): it gets the default
        window startup gives it, and the ceiling that window lowers."""
        from domyn.agents.cli.commands.run.command import SessionOptions, load_session_agent

        path = cfg.config_path()
        self._save(path)
        with patch("domyn.agents.llm.vllm.VLLMProvider.fetch_context_window", return_value=None):
            agent, _ = await load_session_agent(path, SessionOptions())
        assert agent.llm_provider.context_window == cfg.DEFAULT_CONTEXT_WINDOW
        assert agent.llm_provider.max_reply_tokens == cfg.DEFAULT_CONTEXT_WINDOW // 2

    @pytest.mark.asyncio
    async def test_another_file_is_left_to_the_runner(self, tmp_path: Path) -> None:
        from domyn.agents.cli.commands.run.command import SessionOptions, load_session_agent

        path = tmp_path / "other.yaml"
        self._save(path)
        agent, _ = await load_session_agent(path, SessionOptions())
        assert agent.llm_provider.context_window is None
        assert agent.llm_provider.max_reply_tokens == 100_000
