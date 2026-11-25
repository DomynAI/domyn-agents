"""`domyn run-headless`: one task, nobody to answer, no follow-up, no setup read or needed."""

import os
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx
import yaml
from typer.testing import CliRunner

from domyn.agents import BaseEvent, ExecutionEventType, Part, Runner
from domyn.agents.cli.cli import app
from domyn.agents.cli.commands.run_headless import command as run_headless
from domyn.agents.cli.commands.run_headless.command import (
    DEFAULT_INSTRUCTIONS,
    REVIEW_PROMPT,
    run_headless_config,
)
from domyn.agents.instructions import load_declared
from domyn.agents.interfaces.cli import config as cli_config
from domyn.agents.interfaces.cli.state import CliState
from domyn.agents.loader import load_application_from_file
from domyn.agents.planner_strategy.context_management.budget import CONTEXT_USAGE_METADATA_KEY
from domyn.agents.planner_strategy.context_management.compaction import DEFAULT_RESERVE_TOKENS
from tests.cli.test_repl_session import _repl

ENDPOINT = "http://127.0.0.1:8787/v1"


def _config(**given: Any) -> dict[str, Any]:
    values: dict[str, Any] = {
        "model": "the-model",
        "url": ENDPOINT,
        "api_key_env": None,
        "context_window": None,
        "max_tokens": None,
        "temperature": None,
        "reasoning_effort": None,
        "max_iterations": 1000,
        "instructions": DEFAULT_INSTRUCTIONS,
        "workspace": "/",
    }
    return run_headless_config(**{**values, **given})


class TestTheConfiguration:
    def test_the_agent_does_not_stream(self) -> None:
        """A reply that is not streamed has no read timeout, so a long generation is not cut."""
        assert _config()["agents"][0]["stream"] is False

    def test_what_is_not_given_is_left_to_the_endpoint(self) -> None:
        provider = _config()["llm_providers"][0]
        assert provider["url"] == f"{ENDPOINT}/chat/completions"
        assert "context_window" not in provider
        assert "api_key" not in provider
        assert provider["generation_params"] == {"max_tokens": None, "temperature": None}
        assert _config()["planner_strategies"][0]["compactor"] == {"strategy": "two_phase"}

    def test_the_model_limits_and_effort_are_passed_in(self) -> None:
        config = _config(
            context_window=1_048_576,
            max_tokens=393_216,
            reasoning_effort="high",
            api_key_env="OPENAI_API_KEY",
        )
        provider = config["llm_providers"][0]
        assert provider["context_window"] == 1_048_576
        assert provider["generation_params"]["max_tokens"] == 393_216
        assert provider["generation_params"]["reasoning_effort"] == "high"
        # The variable's name, never the key itself.
        assert provider["api_key"] == "${OPENAI_API_KEY}"
        # The reserve is left to the provider's reply ceiling, after the share.
        assert "reserve_tokens" not in config["planner_strategies"][0]["compactor"]

    def test_the_tools_reach_the_machine_it_runs_on(self) -> None:
        bundle = _config()["tool_bundles"][0]
        assert bundle["bundles"] == ["files", "shell"]
        assert bundle["backend"] == "local"
        assert bundle["workspace"] == "/"
        assert bundle["isolate_environment"] is False

    def test_no_instructions_means_none_declared(self) -> None:
        assert "standing_instructions" not in _config(instructions=None)["agents"][0]

    @pytest.mark.asyncio
    async def test_it_loads_with_the_instructions_and_no_file_behind_them(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("RUN_HEADLESS_TEST_KEY", "sk-test")
        path = tmp_path / "config.yaml"
        config = _config(max_tokens=4000, api_key_env="RUN_HEADLESS_TEST_KEY")
        path.write_text(yaml.safe_dump(config))

        agent = await load_application_from_file(str(path))

        assert agent.llm_provider.api_key == "sk-test"  # type: ignore[attr-defined]
        # Passed explicitly as None, so the provider's 0.7 is not merged in: sent as null.
        assert agent.llm_provider.generation_params["temperature"] is None
        assert agent.planner.max_iterations == 1000  # type: ignore[attr-defined]
        loaded = await load_declared(agent)
        assert loaded is not None
        assert "Do not ask questions." in loaded.agent[0].text
        assert loaded.agent[0].path is None


class TestCompactionFollowsTheEffectiveMaxTokens:
    """The reserve is the reply ceiling after `max_tokens_share`, as everywhere in the
    CLI: a `--max-tokens` as large as the window compacts at half of it. Written into
    the config as a `reserve_tokens`, it reserved the whole window and every request
    over the tenth floor (4,000 here) was compacted.

    Through the runner, against the endpoint the config names, so what is checked is
    whether a turn of the loaded agent compacts, not only the threshold it computes.
    The sizes leave room for what every request carries besides the message (tool
    definitions, instructions: about 6,000 tokens).
    """

    WINDOW = 40_000

    async def _turn(self, tmp_path: Path, text: str, **given: Any) -> list[BaseEvent]:
        path = tmp_path / "config.yaml"
        path.write_text(yaml.safe_dump(_config(context_window=self.WINDOW, **given)))
        agent = await load_application_from_file(str(path))
        answer = {
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "done"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"completion_tokens": 1},
        }
        user_input = BaseEvent(
            event_type=ExecutionEventType.USER_INPUT, author="user", content=[Part(text=text)]
        )
        with respx.mock:
            respx.post(f"{ENDPOINT}/chat/completions").mock(
                return_value=httpx.Response(200, json=answer)
            )
            return [
                event
                async for event in Runner().run(
                    application_name="test",
                    user_id="u",
                    user_input=user_input,
                    root=agent,
                    conversation_id="c1",
                )
            ]

    @staticmethod
    def _compacted(events: list[BaseEvent]) -> bool:
        return any(e.event_type == ExecutionEventType.CONTEXT_COMPACTION_START for e in events)

    @staticmethod
    def _threshold(events: list[BaseEvent]) -> int:
        start = next(e for e in events if e.event_type == ExecutionEventType.LLM_START)
        return int(start.metadata[CONTEXT_USAGE_METADATA_KEY]["threshold_tokens"])

    def test_no_reserve_is_written_into_the_config(self) -> None:
        compactor = _config(context_window=self.WINDOW, max_tokens=self.WINDOW)[
            "planner_strategies"
        ][0]["compactor"]
        assert "reserve_tokens" not in compactor

    @pytest.mark.asyncio
    async def test_under_half_the_window_it_does_not_compact(self, tmp_path: Path) -> None:
        # ~8,000 tokens of request: over the tenth floor, under half the window.
        events = await self._turn(tmp_path, "x" * 24_000, max_tokens=self.WINDOW)
        assert self._threshold(events) == self.WINDOW // 2
        assert not self._compacted(events)

    @pytest.mark.asyncio
    async def test_past_half_the_window_it_compacts(self, tmp_path: Path) -> None:
        # ~22,000 tokens of request: past half the window, under the window.
        events = await self._turn(tmp_path, "x" * 66_000, max_tokens=self.WINDOW)
        assert self._threshold(events) == self.WINDOW // 2
        assert self._compacted(events)

    @pytest.mark.asyncio
    async def test_a_smaller_max_tokens_is_reserved_as_it_is(self, tmp_path: Path) -> None:
        events = await self._turn(tmp_path, "x" * 24_000, max_tokens=4_000)
        assert self._threshold(events) == self.WINDOW - 4_000
        assert not self._compacted(events)

    @pytest.mark.asyncio
    async def test_without_max_tokens_the_default_reserve_applies(self, tmp_path: Path) -> None:
        under = await self._turn(tmp_path, "x" * 45_000)
        assert self._threshold(under) == self.WINDOW - DEFAULT_RESERVE_TOKENS
        assert not self._compacted(under)
        # ~25,000 tokens: past the window less the default reserve.
        assert self._compacted(await self._turn(tmp_path, "x" * 75_000))


@pytest.fixture
def started(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """What `start_session` was called with, in place of running the agent."""
    seen: dict[str, Any] = {}

    async def fake_start_session(agent: Any, theme: Any, **kwargs: Any) -> int:
        seen.update(kwargs, agent=agent, home=cli_config.home_dir())
        return 0

    monkeypatch.setattr(run_headless, "start_session", fake_start_session)
    monkeypatch.setenv("RUN_HEADLESS_TEST_KEY", "sk-test")
    return seen


def _invoke(*args: str, endpoint: bool = True) -> Any:
    given = ["--model_url", ENDPOINT, "--api-key-env", "RUN_HEADLESS_TEST_KEY"] if endpoint else []
    return CliRunner().invoke(app, ["run-headless", *given, *args], catch_exceptions=False)


class TestTheCommand:
    def test_one_turn_auto_approved_in_a_folder_of_its_own(
        self, tmp_path: Path, started: dict[str, Any]
    ) -> None:
        state = tmp_path / "state"
        result = _invoke("--model_name", "m", "--state-dir", str(state), "do it")

        assert result.exit_code == 0, result.output
        assert started["initial_prompt"] == "do it"
        assert started["auto_approve"] is True
        assert started["single_turn"] is True
        assert started["steering"] is False
        # One review turn after the task's, unless --review-turns says otherwise.
        assert list(started["follow_ups"]) == [REVIEW_PROMPT]
        # The state folder plays ~/.domyn: nothing of a setup is read.
        assert started["home"] == state.resolve()
        written = yaml.safe_load((state / "config.yaml").read_text())
        assert written["llm_providers"][0]["api_key"] == "${RUN_HEADLESS_TEST_KEY}"
        assert written["llm_providers"][0]["url"] == f"{ENDPOINT}/chat/completions"

    def test_no_reasoning_effort_is_sent_unless_given(
        self, tmp_path: Path, started: dict[str, Any]
    ) -> None:
        state = tmp_path / "state"
        assert _invoke("--model_name", "m", "--state-dir", str(state), "x").exit_code == 0
        params = yaml.safe_load((state / "config.yaml").read_text())["llm_providers"][0][
            "generation_params"
        ]
        assert "reasoning_effort" not in params

    @pytest.mark.parametrize(("given", "sent"), [("high", "high"), ("default", None)])
    def test_a_reasoning_effort_given_is_sent_and_default_sends_none(
        self, tmp_path: Path, started: dict[str, Any], given: str, sent: str | None
    ) -> None:
        state = tmp_path / "state"
        result = _invoke(
            "--model_name",
            "m",
            "--state-dir",
            str(state),
            "--reasoning-effort",
            given,
            "x",
        )
        assert result.exit_code == 0, result.output
        params = yaml.safe_load((state / "config.yaml").read_text())["llm_providers"][0][
            "generation_params"
        ]
        assert params.get("reasoning_effort") == sent
        assert ("reasoning_effort" in params) is (sent is not None)

    def test_the_task_can_come_from_a_file(self, tmp_path: Path, started: dict[str, Any]) -> None:
        task = tmp_path / "task.md"
        task.write_text("from the file")
        state = str(tmp_path / "state")
        result = _invoke("--model_name", "m", "--state-dir", state, "--prompt-file", str(task))
        assert result.exit_code == 0, result.output
        assert started["initial_prompt"] == "from the file"

    def test_no_skill_is_installed_or_loaded(self, tmp_path: Path, started: dict[str, Any]) -> None:
        state = tmp_path / "state"
        assert _invoke("--model_name", "m", "--state-dir", str(state), "x").exit_code == 0
        assert "skills" not in yaml.safe_load((state / "config.yaml").read_text())["agents"][0]
        assert not (state / "builtin-skills").exists()

    def test_there_is_no_option_to_install_the_built_in_skills(
        self, tmp_path: Path, started: dict[str, Any]
    ) -> None:
        result = _invoke("--model_name", "m", "--skills", "--state-dir", str(tmp_path / "s"), "x")
        assert result.exit_code == 2
        assert not started

    def test_no_endpoint_is_refused_whatever_the_environment_says(
        self, tmp_path: Path, started: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OPENAI_BASE_URL", ENDPOINT)
        result = _invoke(
            "--model_name", "m", "--state-dir", str(tmp_path / "s"), "x", endpoint=False
        )
        assert result.exit_code == 2
        assert not started

    def test_without_api_key_env_no_key_is_sent(
        self, tmp_path: Path, started: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-not-ours")
        state = tmp_path / "state"
        result = _invoke(
            "--model_url",
            ENDPOINT,
            "--model_name",
            "m",
            "--state-dir",
            str(state),
            "x",
            endpoint=False,
        )
        assert result.exit_code == 0, result.output
        assert (
            "api_key" not in yaml.safe_load((state / "config.yaml").read_text())["llm_providers"][0]
        )

    def test_a_key_variable_that_is_not_set_is_refused(
        self, tmp_path: Path, started: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("RUN_HEADLESS_TEST_KEY")
        result = _invoke("--model_name", "m", "--state-dir", str(tmp_path / "s"), "x")
        assert result.exit_code == 2
        assert not started

    def test_the_setup_credentials_are_taken_out_of_the_environment(
        self, tmp_path: Path, started: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # What startup put in from ~/.domyn/credentials before the command ran.
        monkeypatch.setenv("SETUP_ONLY_KEY", "from-credentials")
        monkeypatch.setattr(
            cli_config, "_credentials_in_environment", {"SETUP_ONLY_KEY": "from-credentials"}
        )
        result = _invoke("--model_name", "m", "--state-dir", str(tmp_path / "s"), "x")
        assert result.exit_code == 0, result.output
        assert "SETUP_ONLY_KEY" not in os.environ
        # A key the environment itself holds stays.
        assert os.environ["RUN_HEADLESS_TEST_KEY"] == "sk-test"


def _config_file(tmp_path: Path, model: str = "from-file") -> Path:
    path = tmp_path / "my-config.yaml"
    path.write_text(
        yaml.safe_dump(_config(model=model, api_key_env="RUN_HEADLESS_TEST_KEY", instructions=None))
    )
    return path


class TestAConfigFile:
    def test_it_is_run_as_written(self, tmp_path: Path, started: dict[str, Any]) -> None:
        path = _config_file(tmp_path)
        state = tmp_path / "state"
        result = _invoke("--config", str(path), "--state-dir", str(state), "do it", endpoint=False)

        assert result.exit_code == 0, result.output
        assert started["agent"].llm_provider.model_name == "from-file"
        # Copied unchanged into the state folder, which the session uses as its project.
        assert (state / "config.yaml").read_text() == path.read_text()
        assert started["config_file"] == state.resolve() / "config.yaml"
        assert started["home"] == state.resolve()
        # Nothing is built or installed beside it.
        assert not (state / "builtin-skills").exists()
        # The session is still one unattended turn.
        assert started["auto_approve"] is True
        assert started["single_turn"] is True
        assert started["steering"] is False

    @pytest.mark.parametrize(
        "option",
        [
            ["--model_name", "m"],
            ["--model_url", ENDPOINT],
            ["--workspace", "/tmp"],
            # An option's default given explicitly is still an option given.
            ["--reasoning-effort", "default"],
        ],
    )
    def test_the_options_that_build_a_configuration_are_refused_with_it(
        self, tmp_path: Path, started: dict[str, Any], option: list[str]
    ) -> None:
        path = _config_file(tmp_path)
        result = _invoke(
            "--config", str(path), *option, "--state-dir", str(tmp_path / "s"), "x", endpoint=False
        )
        assert result.exit_code == 2
        assert option[0] in result.output
        assert not started

    def test_a_missing_file_is_refused(self, tmp_path: Path, started: dict[str, Any]) -> None:
        result = _invoke(
            "--config",
            str(tmp_path / "nope.yaml"),
            "--state-dir",
            str(tmp_path / "s"),
            "x",
            endpoint=False,
        )
        assert result.exit_code == 2
        assert not started

    def test_the_setup_credentials_stay_for_the_variables_it_names(
        self, tmp_path: Path, started: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("SETUP_ONLY_KEY", "from-credentials")
        monkeypatch.setattr(
            cli_config, "_credentials_in_environment", {"SETUP_ONLY_KEY": "from-credentials"}
        )
        path = _config_file(tmp_path)
        result = _invoke(
            "--config", str(path), "--state-dir", str(tmp_path / "s"), "x", endpoint=False
        )
        assert result.exit_code == 0, result.output
        assert os.environ["SETUP_ONLY_KEY"] == "from-credentials"

    def test_review_turns_zero_ends_with_the_task(
        self, tmp_path: Path, started: dict[str, Any]
    ) -> None:
        result = _invoke(
            "--model_name", "m", "--state-dir", str(tmp_path / "s"), "--review-turns", "0", "x"
        )
        assert result.exit_code == 0, result.output
        assert list(started["follow_ups"]) == []

    def test_without_it_the_endpoint_and_model_are_needed(
        self, tmp_path: Path, started: dict[str, Any]
    ) -> None:
        result = _invoke(
            "--model_name", "m", "--state-dir", str(tmp_path / "s"), "x", endpoint=False
        )
        assert result.exit_code == 2
        assert "--config" in result.output
        assert not started


class TestTheRepl:
    @pytest.mark.asyncio
    async def test_a_single_turn_session_reads_no_input(self) -> None:
        repl, _ = _repl("never read")
        repl._single_turn = True
        assert await repl.run(initial_prompt="the task") == 0
        assert repl._state.turns == 1
        assert repl._input._session._lines == ["never read"]  # type: ignore[attr-defined]

    @pytest.mark.asyncio
    async def test_follow_ups_are_sent_as_further_turns_then_the_session_ends(self) -> None:
        from collections import deque

        repl, _ = _repl("never read")
        repl._single_turn = True
        repl._follow_ups = deque(["review it"])
        assert await repl.run(initial_prompt="the task") == 0
        assert repl._state.turns == 2
        assert repl._input._session._lines == ["never read"]  # type: ignore[attr-defined]

    @pytest.mark.asyncio
    async def test_a_prompt_starting_with_a_slash_is_the_task(self) -> None:
        repl, _ = _repl()
        repl._single_turn = True
        assert await repl.run(initial_prompt="/app/data.csv has a bad row: fix it") == 0
        assert repl._state.turns == 1

    def test_auto_approve_given_is_not_remembered(self) -> None:
        from io import StringIO

        from rich.console import Console

        from domyn.agents.interfaces.cli import Theme
        from domyn.agents.interfaces.cli.repl import Repl
        from tests.cli.test_repl_session import _agent

        repl = Repl(
            _agent(),
            Console(file=StringIO()),
            Theme.build(allow_animation=False),
            enable_enhanced_input=False,
            auto_approve=True,
        )
        assert repl._prefs.auto_approve is True
        assert CliState.load().auto_approve is False
