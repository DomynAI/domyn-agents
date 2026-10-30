"""Tests for the CLI configuration helpers.

`~/.domyn/config.yaml` is a plain ``ConfigurationV1`` file; the wizard's typed
structs (`ProviderConfig`/`AgentConfig`/`CompactionConfig`) exist only as its
working state on the way to `configuration_v1_dict_from`. These tests exercise
that projection, the credentials store, and the small pure helpers around them.
"""

from pathlib import Path
from typing import ClassVar
from unittest.mock import patch

import httpx
import pytest

from domyn.agents.interfaces.cli import config as cfg
from domyn.agents.llm import LLMProviderFactory
from domyn.agents.tools.builtin import BUNDLES, BuiltinBundleConfig


@pytest.fixture(autouse=True)
def isolated_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Redirect config and credentials into a temp dir.

    Without this, running the suite would read — or worse, write — the developer's
    real ~/.domyn.
    """
    monkeypatch.setattr(cfg, "GLOBAL_DIR", tmp_path / "home" / ".domyn")
    monkeypatch.setattr(cfg, "PROJECT_DIR", tmp_path / "project" / ".domyn")
    for name in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "LLM_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    return tmp_path


class TestConfigLocation:
    def test_project_config_wins_over_global(self) -> None:
        """Project-local wins so a repo can pin its own model without touching
        the user's global setup."""
        (cfg.GLOBAL_DIR).mkdir(parents=True)
        (cfg.GLOBAL_DIR / cfg.CONFIG_FILENAME).write_text("entrypoint: Global\n")
        (cfg.PROJECT_DIR).mkdir(parents=True)
        (cfg.PROJECT_DIR / cfg.CONFIG_FILENAME).write_text("entrypoint: Project\n")

        raw = cfg.read_raw_config()
        assert raw is not None
        assert raw["entrypoint"] == "Project"

    def test_missing_config_returns_none(self) -> None:
        assert cfg.read_raw_config() is None

    def test_corrupt_config_is_ignored_not_raised(self) -> None:
        """A broken config must not lock the user out of the CLI."""
        path = cfg.config_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("not: valid: yaml: [", encoding="utf-8")
        assert cfg.read_raw_config() is None


class TestSecrets:
    def test_credentials_file_is_owner_only(self) -> None:
        path = cfg.write_credential("OPENAI_API_KEY", "sk-x")
        assert oct(path.stat().st_mode)[-3:] == "600"

    def test_environment_wins_over_file(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """CI and containers inject keys via env; that must take precedence."""
        cfg.write_credential("OPENAI_API_KEY", "from-file")
        provider = cfg.ProviderConfig(type="openai", model_name="m")
        assert cfg.resolve_api_key(provider) == "from-file"
        monkeypatch.setenv("OPENAI_API_KEY", "from-env")
        assert cfg.resolve_api_key(provider) == "from-env"

    def test_missing_key_resolves_to_none(self) -> None:
        assert cfg.resolve_api_key(cfg.ProviderConfig(type="openai", model_name="m")) is None

    def test_writing_a_second_key_keeps_the_first(self) -> None:
        cfg.write_credential("OPENAI_API_KEY", "a")
        cfg.write_credential("ANTHROPIC_API_KEY", "b")
        stored = cfg.read_credentials()
        assert stored["OPENAI_API_KEY"] == "a"
        assert stored["ANTHROPIC_API_KEY"] == "b"

    def test_describe_reports_key_presence_not_value(self) -> None:
        cfg.write_credential("OPENAI_API_KEY", "sk-secret")
        provider = cfg.ProviderConfig(type="openai", model_name="m")
        config_dict = cfg.configuration_v1_dict_from(
            provider, cfg.AgentConfig(), cfg.CompactionConfig()
        )
        description = cfg.describe_configuration_v1(config_dict)
        assert "sk-secret" not in str(description)
        provider_fields = dict(description.sections[0].items[0].fields)
        assert provider_fields["context window"] == cfg.AUTO_CONTEXT_WINDOW
        assert "set" in provider_fields["api key"]

    def test_a_key_found_at_write_time_is_referenced_by_name_not_value(self) -> None:
        """The YAML must never hold the literal secret — only the variable name,
        resolved fresh on every load."""
        cfg.write_credential("OPENAI_API_KEY", "sk-super-secret")
        provider = cfg.ProviderConfig(type="openai", model_name="m")
        config_dict = cfg.configuration_v1_dict_from(
            provider, cfg.AgentConfig(), cfg.CompactionConfig()
        )
        assert config_dict["llm_providers"][0]["api_key"] == "${OPENAI_API_KEY}"

    def test_no_key_omits_the_field_entirely(self) -> None:
        """Referencing an unresolvable ${VAR} would make the file fail to load
        outright (env-var interpolation raises on a missing variable) — a
        missing key must instead surface as an auth error at call time."""
        provider = cfg.ProviderConfig(type="openai", model_name="m")
        config_dict = cfg.configuration_v1_dict_from(
            provider, cfg.AgentConfig(), cfg.CompactionConfig()
        )
        assert "api_key" not in config_dict["llm_providers"][0]

    def test_credentials_survive_a_readonly_home(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Reading credentials from an unreadable location must not raise."""
        monkeypatch.setattr(cfg, "GLOBAL_DIR", Path("/proc/nonexistent-domyn"))
        assert cfg.read_credentials() == {}

    @pytest.mark.parametrize(
        ("provider_type", "expected_env"),
        [
            ("domyn", "DOMYN_API_KEY"),
            ("openai", "OPENAI_API_KEY"),
            ("anthropic", "ANTHROPIC_API_KEY"),
            ("open_router", "OPENROUTER_API_KEY"),
            ("cerebras", "CEREBRAS_API_KEY"),
            ("vllm", "VLLM_API_KEY"),
            ("custom", "LLM_API_KEY"),
        ],
    )
    def test_each_provider_uses_its_conventional_env_var(
        self, provider_type: str, expected_env: str
    ) -> None:
        assert cfg.ProviderConfig(type=provider_type).default_key_env() == expected_env

    def test_explicit_api_key_env_overrides_the_convention(self) -> None:
        provider = cfg.ProviderConfig(type="openai", api_key_env="MY_OWN_VAR")
        assert provider.default_key_env() == "MY_OWN_VAR"


class TestPromptDisambiguation:
    """`domyn run "a question"` must be read as a prompt, not a filename."""

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ("agent.py", True),
            ("config.yaml", True),
            ("config.yml", True),
            ("what is the weather?", False),
            ("hello", False),
        ],
    )
    def test_detection(self, value: str, expected: bool) -> None:
        from domyn.agents.cli.commands.run.command import _looks_like_agent_file

        assert _looks_like_agent_file(Path(value)) is expected

    def test_existing_file_wins_even_without_extension(self, tmp_path: Path) -> None:
        target = tmp_path / "agentfile"
        target.write_text("x", encoding="utf-8")
        from domyn.agents.cli.commands.run.command import _looks_like_agent_file

        assert _looks_like_agent_file(target) is True


class TestIsConfigurationV1UsableWithoutAnExplicitEntrypoint:
    """Without `entrypoint:`, the raw dict is unordered — dependency reordering
    only happens later, inside the loader. So falling back to `agents[-1]` is
    only safe when it's the *only* agent; with more than one, which is the
    root is genuinely unknown at this point and must not be guessed."""

    @staticmethod
    def _raw(agents: list[dict], entrypoint: str | None = None) -> dict:
        raw: dict = {
            "agents": agents,
            "llm_providers": [{"name": "p", "type": "openai", "model_name": "m"}],
        }
        if entrypoint is not None:
            raw["entrypoint"] = entrypoint
        return raw

    def test_explicit_entrypoint_resolves_the_named_agent_among_several(self) -> None:
        raw = self._raw(
            agents=[
                {"name": "A", "llm_provider": "missing"},
                {"name": "B", "llm_provider": "p"},
            ],
            entrypoint="B",
        )
        assert cfg.is_configuration_v1_usable(raw) is True

    def test_a_single_agent_still_falls_back_without_an_explicit_entrypoint(self) -> None:
        raw = self._raw(agents=[{"name": "Solo", "llm_provider": "p"}])
        assert cfg.is_configuration_v1_usable(raw) is True

    def test_multiple_agents_without_an_explicit_entrypoint_is_unusable(self) -> None:
        raw = self._raw(
            agents=[
                {"name": "A", "llm_provider": "p"},
                {"name": "B", "llm_provider": "p"},
            ]
        )
        assert cfg.is_configuration_v1_usable(raw) is False


class TestCurrentSettingsFromToleratesAHandEditedToolBundle:
    """`tool_bundles` entries only carry a guaranteed `bundles` key when
    written by `configuration_v1_dict_from`; `config.yaml` is plain,
    hand-editable YAML, so a bundle entry missing it must not raise out of
    the setup wizard. It reads as what the loader mounts for it — every
    bundle, `BuiltinBundleConfig`'s default — or the wizard's checklist would
    start empty and a re-run would take the agent's tools away."""

    def test_a_tool_bundle_entry_missing_bundles_does_not_raise(self) -> None:
        raw = {
            "entrypoint": "A",
            "agents": [{"name": "A", "llm_provider": "p", "tool_bundles": ["custom"]}],
            "llm_providers": [{"name": "p", "type": "openai", "model_name": "m"}],
            "tool_bundles": [{"name": "custom"}],  # no "bundles" key
        }
        _, agent, _ = cfg.current_settings_from(raw)
        assert agent.tools == BuiltinBundleConfig(name="custom").bundles


class TestAnAgentsToolsAndBundlesShareOneRow:
    """`/config` showed `tools (none)` beside `tool_bundles cli_tools` for an
    agent whose only tools come from a bundle, which reads as an agent with
    none. One row: the tool names, each tagged with its built-in bundle."""

    @staticmethod
    def _fields(agent: dict, bundles: list[dict] | None = None) -> dict[str, str]:
        raw = {
            "version": "1.0",
            "entrypoint": "A",
            "agents": [{"name": "A", **agent}],
            "tool_bundles": bundles or [],
        }
        sections = cfg.describe_configuration_v1(raw).sections
        return dict(next(s for s in sections if s.title == "Agents").items[0].fields)

    @classmethod
    def _tools_field(cls, agent: dict, bundles: list[dict] | None = None) -> str:
        return cls._fields(agent, bundles)["tools"]

    @classmethod
    def _bundles_field(cls, agent: dict, bundles: list[dict] | None = None) -> str:
        return cls._fields(agent, bundles)["tool_bundles"]

    def test_bundle_tools_are_listed_by_name_tagged_with_their_bundle(self) -> None:
        tools = self._tools_field(
            {"tool_bundles": ["cli_tools"]},
            [{"name": "cli_tools", "bundles": ["files", "shell"]}],
        ).split(", ")
        assert "read_file(files)" in tools
        assert "edit_file(files)" in tools
        assert "run_command(shell)" in tools
        assert "cli_tools" not in "".join(tools)

    def test_a_single_bundle_lists_only_its_own_tools(self) -> None:
        tools = self._tools_field({"tool_bundles": ["b"]}, [{"name": "b", "bundles": ["shell"]}])
        assert "run_command(shell)" in tools
        assert "read_file" not in tools

    def test_own_tools_come_first_by_plain_name(self) -> None:
        tools = self._tools_field(
            {"tools": ["a"], "tool_bundles": ["b"]}, [{"name": "b", "bundles": ["shell"]}]
        ).split(", ")
        assert tools[0] == "a"
        assert tools[1] == "run_command(shell)"

    def test_an_entry_that_cannot_be_resolved_contributes_nothing(self) -> None:
        assert self._tools_field({"tool_bundles": ["missing"]}) == "(none)"
        assert (
            self._tools_field({"tool_bundles": ["b"]}, [{"name": "b", "bundles": ["nonsense"]}])
            == "(none)"
        )
        assert self._tools_field({"tool_bundles": ["b"]}, [{"name": "b"}]) == "(none)"

    def test_neither_is_none(self) -> None:
        assert self._tools_field({}) == "(none)"
        assert self._bundles_field({}) == "(none)"

    def test_the_bundles_row_names_each_bundle_with_its_group(self) -> None:
        assert (
            self._bundles_field(
                {"tool_bundles": ["cli_tools"]},
                [{"name": "cli_tools", "bundles": ["files", "shell"]}],
            )
            == "files(cli_tools), shell(cli_tools)"
        )

    def test_each_group_tags_its_own_bundles(self) -> None:
        assert (
            self._bundles_field(
                {"tool_bundles": ["a", "b"]},
                [{"name": "a", "bundles": ["files"]}, {"name": "b", "bundles": ["shell"]}],
            )
            == "files(a), shell(b)"
        )

    def test_a_group_that_cannot_be_resolved_is_none_on_the_bundles_row(self) -> None:
        assert self._bundles_field({"tool_bundles": ["missing"]}) == "(none)"
        assert self._bundles_field({"tool_bundles": ["b"]}, [{"name": "b"}]) == "(none)"


class TestOnlyBundlesTheAgentHasAreDescribed:
    """`current_settings_from` once filled a missing bundle with the wizard's
    defaults (files, shell), described as if mounted, for a file whose agent
    declares no `tool_bundles` at all."""

    @staticmethod
    def _sections(raw: dict) -> dict[str, cfg.ConfigSection]:
        return {s.title: s for s in cfg.describe_configuration_v1(raw).sections}

    def test_an_agent_without_bundles_has_no_builtin_tools(self) -> None:
        sections = self._sections(
            {"version": "1.0", "entrypoint": "A", "agents": [{"name": "A", "llm_provider": "p"}]}
        )
        assert dict(sections["Agents"].items[0].fields)["tools"] == "(none)"
        assert "Tool Bundles" not in sections

    def test_an_unresolvable_bundle_name_counts_as_none(self) -> None:
        sections = self._sections(
            {
                "version": "1.0",
                "entrypoint": "A",
                "agents": [{"name": "A", "tool_bundles": ["missing"]}],
            }
        )
        assert dict(sections["Agents"].items[0].fields)["tools"] == "(none)"
        assert "Tool Bundles" not in sections

    def test_a_declared_bundle_is_described(self) -> None:
        raw = cfg.configuration_v1_dict_from(
            cfg.ProviderConfig(type="openai", model_name="m"),
            cfg.AgentConfig(tools=list(BUNDLES)),
            cfg.CompactionConfig(),
        )
        sections = self._sections(raw)
        tools = dict(sections["Agents"].items[0].fields)["tools"]
        assert "read_file(files)" in tools
        assert "run_command(shell)" in tools
        bundle = dict(sections["Tool Bundles"].items[0].fields)
        assert bundle["sandbox"] == "local"
        assert bundle["isolate_environment"] == "True"


class TestTheAgentsTools:
    """Own tools come first, then those of its bundles."""

    BUNDLE: ClassVar[dict] = {"name": "b", "bundles": ["files", "shell"]}

    @classmethod
    def _tools(cls, agent: dict, bundles: list[dict] | None = None) -> str:
        raw = {
            "version": "1.0",
            "entrypoint": "A",
            "agents": [{"name": "A", **agent}],
            "tool_bundles": bundles or [],
        }
        agents = next(s for s in cfg.describe_configuration_v1(raw).sections if s.title == "Agents")
        return dict(agents.items[0].fields)["tools"]

    def test_an_agent_with_only_its_own_tools_lists_them(self) -> None:
        assert self._tools({"tools": ["a", "b"]}) == "a, b"

    def test_own_tools_precede_the_bundle_tools(self) -> None:
        assert self._tools({"tools": ["a"], "tool_bundles": ["b"]}, [self.BUNDLE]).startswith(
            "a, read_file(files)"
        )


class TestAnAgentsMcpServersAreListedHoweverTheyAreDeclared:
    """An agent names a server declared at the top level, or defines one inline.
    The row only read the inline form, so an agent referencing its servers by
    name showed `(none)` above a populated MCP Servers section."""

    @staticmethod
    def _describe(agent_servers: list | None, top: list[dict] | None = None):  # type: ignore[no-untyped-def]
        agent: dict = {"name": "A"}
        if agent_servers is not None:
            agent["mcp_servers"] = agent_servers
        raw = {"version": "1.0", "entrypoint": "A", "agents": [agent], "mcp_servers": top or []}
        description = cfg.describe_configuration_v1(raw)
        sections = {s.title: s for s in description.sections}
        row = dict(sections["Agents"].items[0].fields)["mcp_servers"]
        listed = (
            [i.name for i in sections["MCP Servers"].items] if "MCP Servers" in sections else []
        )
        return row, listed

    TOP = (
        {"name": "svelte_docs", "url": "https://a.example/mcp"},
        {"name": "slack", "url": "https://b.example/mcp"},
    )

    def test_servers_referenced_by_name_are_listed(self) -> None:
        row, listed = self._describe(["svelte_docs", "slack"], list(self.TOP))
        assert row == "svelte_docs, slack"
        assert listed == ["svelte_docs", "slack"]  # each described once

    def test_inline_servers_are_listed(self) -> None:
        row, _ = self._describe([{"name": "own", "url": "https://c.example/mcp"}])
        assert row == "own"

    def test_the_two_forms_mix_in_declaration_order(self) -> None:
        row, _ = self._describe(
            ["slack", {"name": "own", "url": "https://c.example/mcp"}], [self.TOP[1]]
        )
        assert row == "slack, own"

    def test_none_declared(self) -> None:
        assert self._describe(None)[0] == "(none)"


class TestThePlannerIsPinned:
    """The CLI default must be native tool calling.

    Not ReAct, which needs the model to reply in a `<thought>/<action>`
    format; a plain chat model does not, and loops on "Unparsable action"
    instead of answering.

    `tool_use_thought` rather than plain `tool_use`: same native tool
    calling, plus the reasoning surfaced as `Part(thought=...)` so the CLI
    can show what the agent is thinking before each tool call.

    Nothing prompts for it, and `configuration_v1_dict_from` always writes it —
    see `CLI_PLANNER`.
    """

    def test_it_is_always_written(self) -> None:
        provider = cfg.ProviderConfig(type="custom", model_name="m", url="http://x")
        config_dict = cfg.configuration_v1_dict_from(
            provider, cfg.AgentConfig(), cfg.CompactionConfig()
        )
        assert config_dict["planner_strategies"] == [
            {
                "name": cfg.CLI_PLANNER,
                "type": cfg.CLI_PLANNER,
                "compactor": {"strategy": "two_phase"},
            }
        ]
        assert config_dict["agents"][0]["planner_strategy"] == cfg.CLI_PLANNER


class TestTheContextWindowTheProviderEndsUpWith:
    """One window: declared, else what the provider reports, else the default.

    Before the default existed, a self-hosted model declared no window, so no
    threshold existed and compaction could never fire — the common case here.
    Asking the server first means the default is only the last resort.
    """

    def _provider(self, **kwargs: object) -> object:
        provider_config = cfg.ProviderConfig(type="vllm", url="http://x", **kwargs)  # type: ignore[arg-type]
        config_dict = cfg.configuration_v1_dict_from(
            provider_config, cfg.AgentConfig(), cfg.CompactionConfig()
        )
        return LLMProviderFactory.from_config(config_dict["llm_providers"][0])

    def test_a_declared_window_is_used(self) -> None:
        provider = self._provider(model_name="whatever", context_window=32_768)
        assert provider.context_window == 32_768

    def test_max_model_len_is_accepted_as_the_same_thing(self) -> None:
        """The name vLLM uses on the command line that decides it, so the name
        whoever knows the number will reach for."""
        config = cfg.ProviderConfig.model_validate(
            {"type": "vllm", "url": "http://x", "model_name": "m", "max_model_len": 65_536}
        )
        assert config.context_window == 65_536

    def test_an_undeclared_window_is_not_written_so_the_provider_can_be_asked(self) -> None:
        """A written default would be a declaration, and nothing would ever ask."""
        assert self._provider(model_name="m").context_window is None

    @pytest.mark.asyncio
    async def test_a_typed_150000_is_a_declaration_not_auto(self) -> None:
        provider_config = cfg.ProviderConfig(
            type="vllm", url="http://x", model_name="m", context_window=cfg.DEFAULT_CONTEXT_WINDOW
        )
        written = cfg.configuration_v1_dict_from(
            provider_config, cfg.AgentConfig(), cfg.CompactionConfig()
        )
        assert written["llm_providers"][0]["context_window"] == cfg.DEFAULT_CONTEXT_WINDOW
        read_back, _, _ = cfg.current_settings_from(written)
        assert read_back.context_window == cfg.DEFAULT_CONTEXT_WINDOW

        provider = LLMProviderFactory.from_config(written["llm_providers"][0])
        with patch.object(type(provider), "fetch_context_window", side_effect=AssertionError):
            await cfg.apply_context_window(provider)
        assert provider.context_window == cfg.DEFAULT_CONTEXT_WINDOW

    @pytest.mark.asyncio
    async def test_a_declared_window_beats_what_the_provider_reports(self) -> None:
        provider = self._provider(model_name="m", context_window=8_192)
        with patch.object(type(provider), "fetch_context_window", side_effect=AssertionError):
            await cfg.apply_context_window(provider)
        assert provider.context_window == 8_192

    @pytest.mark.asyncio
    async def test_what_the_provider_reports_beats_the_default(self) -> None:
        provider = self._provider(model_name="reported")
        with patch.object(type(provider), "fetch_context_window", return_value=40_960):
            await cfg.apply_context_window(provider)
        assert provider.context_window == 40_960

    @pytest.mark.asyncio
    async def test_the_default_applies_when_nothing_answers(self) -> None:
        """The whole point of the default: without it a provider that reports
        nothing has no threshold, and `assess` says "window unknown" forever."""
        from domyn.agents.llm import LLMMessage
        from domyn.agents.planner_strategy import Compactor

        provider = self._provider(model_name="silent")
        with patch.object(
            type(provider), "fetch_context_window", side_effect=httpx.ConnectError("down")
        ):
            await cfg.apply_context_window(provider)
        assert provider.context_window == cfg.DEFAULT_CONTEXT_WINDOW

        class _Agent:
            llm_provider = provider
            name = "A"

        plan = Compactor().assess([LLMMessage(role="user", content="x" * 2_000_000)], _Agent())
        assert plan.needed is True, plan.reason

    def test_the_fallback_is_large_enough_to_not_compact_a_128k_deployment_early(self) -> None:
        """The direction of the damage decides the value: assuming too small
        compacts a 128k model needlessly, every session."""
        assert cfg.DEFAULT_CONTEXT_WINDOW > 128_000


class TestChoosingHowTheSessionShortensItself:
    """The config surface. Without it `two_phase` would exist and be unreachable,
    which is the failure this feature has already had once."""

    def _compactor(self, **compaction: object) -> object:
        from domyn.agents.planner_strategy.context_management.compaction import build_compactor

        return build_compactor(cfg.CompactionConfig(**compaction))  # type: ignore[arg-type]

    def test_the_default_summarizes_when_pruning_is_not_enough(self) -> None:
        assert self._compactor().strategy_name == "two_phase"

    def test_pruning_only_can_be_asked_for(self) -> None:
        """For anyone who wants no model call on this path at all."""
        assert self._compactor(strategy="prune_only").strategy_name == "prune_only"

    def test_it_can_be_turned_off_entirely(self) -> None:
        compactor = self._compactor(strategy="noop")
        assert compactor.active is False
        assert compactor.strategy_name is None

    def test_a_typo_falls_back_instead_of_killing_the_session(self) -> None:
        """One mistyped line of configuration should not stop someone working;
        the warning is what gets it fixed."""
        assert self._compactor(strategy="prune_onyl").strategy_name == "two_phase"

    def test_the_reserve_can_be_widened(self) -> None:
        assert self._compactor(reserve_tokens=32_000).policy.reserve_tokens == 32_000

    def test_the_protected_tail_can_be_widened(self) -> None:
        assert self._compactor(keep_recent_tokens=40_000).policy.keep_recent_tokens == 40_000

    def test_omitted_values_keep_the_framework_defaults(self) -> None:
        """`None` means "not configured", not "zero" — a zero reserve would make
        the threshold the window itself and every request a coin flip."""
        from domyn.agents.planner_strategy import CompactionPolicy

        compactor = self._compactor()
        assert compactor.policy.reserve_tokens == CompactionPolicy().reserve_tokens
        assert compactor.policy.keep_recent_tokens == CompactionPolicy().keep_recent_tokens


class TestDescribeConfigurationV1IsComprehensiveButNeverLeaksASecret:
    """`config.yaml` is a full, open-ended `ConfigurationV1` file — nothing
    stops a hand-edited one from holding a literal secret instead of an
    `${ENV_VAR}` reference (an MCP `headers` bearer token, an
    `oauth_providers` `client_secret`, a second provider's `api_key`, or
    credentials embedded in a URL). `describe_configuration_v1` is what
    `/config` renders instead of the raw file, so every section it adds
    beyond the entrypoint's own summary must show everything *except* the
    secret value itself."""

    @staticmethod
    def _raw() -> dict:
        return {
            "version": "1.0",
            "entrypoint": "Main",
            "llm_providers": [
                {"name": "p1", "type": "openai", "model_name": "gpt-4o"},
                {
                    "name": "p2",
                    "type": "custom",
                    "model_name": "local-model",
                    "url": "https://user:s3cr3t@internal.example.com/v1",
                    "api_key": "sk-hand-typed-literal",
                    "headers": {"Authorization": "Bearer abc123"},
                },
            ],
            "agents": [
                {
                    "name": "Main",
                    "llm_provider": "p1",
                    "planner_strategy": "react",
                    "skills": {"source": {"type": "local_dir", "paths": ["./skills"]}},
                },
                {
                    # A source that is not a directory, carrying content that
                    # must never reach `/config`.
                    "name": "Helper",
                    "llm_provider": "p1",
                    "skills": {
                        "source": {
                            "type": "memory",
                            "skills": {"s": "---\nname: s\n---\ninternal-skill-body"},
                        }
                    },
                },
            ],
            "mcp_servers": [
                {
                    "name": "search",
                    "url": "https://mcp.example.com",
                    "headers": {"X-Api-Key": "top-secret-header-value"},
                }
            ],
            "oauth_providers": [
                {
                    "name": "gh",
                    "client_id": "public-client-id",
                    "client_secret": "literal-oauth-secret",
                }
            ],
            "tools": [
                {
                    "type": "mcp",
                    "tool_name": "get_me",
                    "server_url": "https://api.githubcopilot.com/mcp/",
                    "use_oauth": True,
                    "headers": {"Authorization": "Bearer mcp-tool-header-secret"},
                }
            ],
            "metadata": {"internal_note": "do not commit this"},
        }

    def _description(self, raw: dict | None = None) -> cfg.ConfigDescription:
        return cfg.describe_configuration_v1(raw if raw is not None else self._raw())

    @staticmethod
    def _field(
        description: cfg.ConfigDescription, section_title: str, item_name: str, field_label: str
    ) -> str:
        section = next(s for s in description.sections if s.title == section_title)
        item = next(i for i in section.items if i.name == item_name)
        return dict(item.fields)[field_label]

    def test_a_second_providers_hardcoded_key_is_never_shown(self) -> None:
        description = self._description()
        assert self._field(description, "Providers", "p2", "api key") == (
            "(hardcoded value — not shown)"
        )
        assert "sk-hand-typed-literal" not in str(description)

    def test_a_second_providers_headers_show_names_not_values(self) -> None:
        description = self._description()
        assert self._field(description, "Providers", "p2", "headers") == "Authorization"
        assert "abc123" not in str(description)

    def test_a_url_with_embedded_credentials_is_scrubbed(self) -> None:
        description = self._description()
        assert "s3cr3t" not in str(description)
        assert "internal.example.com" in self._field(description, "Providers", "p2", "url")

    def test_mcp_server_header_values_are_never_shown(self) -> None:
        description = self._description()
        assert self._field(description, "MCP Servers", "search", "headers") == "X-Api-Key"
        assert "top-secret-header-value" not in str(description)

    def test_oauth_client_id_is_shown_but_the_secret_is_not(self) -> None:
        description = self._description()
        assert self._field(description, "OAuth Providers", "gh", "client_id") == "public-client-id"
        assert self._field(description, "OAuth Providers", "gh", "client_secret") == (
            "(hardcoded value — not shown)"
        )
        assert "literal-oauth-secret" not in str(description)

    def test_an_interpolated_secret_reports_the_env_var_name_not_the_value(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("GH_CLIENT_SECRET", "whatever")
        raw = self._raw()
        raw["oauth_providers"][0]["client_secret"] = "${GH_CLIENT_SECRET}"
        description = self._description(raw)
        assert self._field(description, "OAuth Providers", "gh", "client_secret") == (
            "GH_CLIENT_SECRET (set)"
        )

    def test_metadata_keys_show_names_only(self) -> None:
        description = self._description()
        assert "internal_note" in self._field(description, "Metadata", "", "keys")
        assert "do not commit this" not in str(description)

    def test_a_leftover_skills_dirs_key_is_listed_like_any_other(self) -> None:
        """Skills live on the agent now; an old `metadata.skills_dirs` is just a key."""
        raw = self._raw()
        raw["metadata"]["skills_dirs"] = ["./old"]
        description = self._description(raw)
        assert self._field(description, "Metadata", "", "keys") == "internal_note, skills_dirs"
        assert "./old" not in str(description)

    def test_every_agent_shows_its_skills(self) -> None:
        description = self._description()
        assert self._field(description, "Agents", "Main", "skills") == "local_dir: ./skills"
        assert self._field(description, "Agents", "Helper", "skills") == "memory"

    def test_a_skill_sources_content_is_never_shown(self) -> None:
        assert "internal-skill-body" not in str(self._description())

    def test_an_agent_without_skills_says_none(self) -> None:
        raw = self._raw()
        del raw["agents"][1]["skills"]
        assert self._field(self._description(raw), "Agents", "Helper", "skills") == "(none)"

    @pytest.mark.parametrize(
        ("skills", "expected"),
        [
            (None, "(absent)"),
            ({}, "(absent)"),
            ({"selector": {"type": "index"}}, "(unrecognised)"),
            ({"source": {"paths": ["./x"]}}, "(unrecognised)"),
            ({"source": {"type": "local_dir"}}, "local_dir (no paths)"),
            ({"source": {"type": "local_dir", "paths": ["./a", "/b"]}}, "local_dir: ./a, /b"),
            ({"source": {"type": "remote", "headers": {"Authorization": "x"}}}, "remote"),
        ],
    )
    def test_skills_summary(self, skills: object, expected: str) -> None:
        assert cfg._skills_summary(skills, absent="(absent)") == expected

    def test_an_mcp_tool_entry_is_labeled_by_tool_name_not_name(self) -> None:
        """MCPTool (and DelegateTool) have no `name` field — their raw YAML
        key is `tool_name`. This used to render as `tool[?]` for every entry,
        indistinguishable from one another."""
        description = self._description()
        assert self._field(description, "Tools", "get_me", "tool_name") == "get_me"
        assert self._field(description, "Tools", "get_me", "server_url") == (
            "https://api.githubcopilot.com/mcp/"
        )
        assert self._field(description, "Tools", "get_me", "use_oauth") == "True"
        assert self._field(description, "Tools", "get_me", "headers") == "Authorization"
        assert "mcp-tool-header-secret" not in str(description)

    def test_an_agents_own_mcp_server_is_named_on_the_agent_and_detailed_in_its_section(
        self,
    ) -> None:
        """An agent-declared MCP server (not also listed at the top level)
        must still show its full detail somewhere — just not inline on the
        agent, which would need a fourth indentation level."""
        raw = self._raw()
        raw["agents"][0]["mcp_servers"] = [
            {
                "name": "agent_only",
                "url": "https://agent-only.example.com",
                "headers": {"Authorization": "Bearer agent-only-secret"},
            }
        ]
        description = self._description(raw)
        assert self._field(description, "Agents", "Main", "mcp_servers") == "agent_only"
        assert self._field(description, "MCP Servers", "agent_only", "headers") == ("Authorization")
        assert "agent-only-secret" not in str(description)

    def test_nothing_in_the_whole_summary_ever_contains_a_planted_secret(self) -> None:
        """Belt and suspenders: no single row assertion can miss a leak the
        way scanning the entire rendered output can. `str()` on the
        dataclass walks every nested section/item/field via its default
        `repr`, so this still covers the whole structure."""
        output = str(self._description())
        for planted_secret in (
            "sk-hand-typed-literal",
            "abc123",
            "s3cr3t",
            "top-secret-header-value",
            "literal-oauth-secret",
            "do not commit this",
            "mcp-tool-header-secret",
        ):
            assert planted_secret not in output


class TestTheBrowserSettingsSurviveTheWizard:
    """The wizard rewrites the whole file, so anything not read back is lost on
    every `domyn setup` and `/model` — which is what happened to `allowed_env`."""

    @staticmethod
    def _round_trip(agent: cfg.AgentConfig) -> cfg.AgentConfig:
        provider = cfg.ProviderConfig(type="openai", model_name="m")
        raw = cfg.configuration_v1_dict_from(provider, agent, cfg.CompactionConfig())
        return cfg.current_settings_from(raw)[1]

    def test_browser_options_and_allowed_env_come_back(self) -> None:
        agent = cfg.AgentConfig(
            tools=["files", "shell", "browser"],
            allowed_env=["GITHUB_TOKEN"],
            browser=cfg.BrowserOptions(engine="lightpanda", allowed_domains=["example.com"]),
        )
        back = self._round_trip(agent)
        assert back.tools == ["files", "shell", "browser"]
        assert back.allowed_env == ["GITHUB_TOKEN"]
        assert back.browser == agent.browser

    def test_only_what_was_chosen_is_written(self) -> None:
        provider = cfg.ProviderConfig(type="openai", model_name="m")
        agent = cfg.AgentConfig(tools=["browser"])
        raw = cfg.configuration_v1_dict_from(provider, agent, cfg.CompactionConfig())
        assert "browser" not in raw["tool_bundles"][0]
        agent.browser = cfg.BrowserOptions(allow_private_urls=True)
        raw = cfg.configuration_v1_dict_from(provider, agent, cfg.CompactionConfig())
        assert raw["tool_bundles"][0]["browser"] == {"allow_private_urls": True}

    def test_a_broken_browser_block_does_not_stop_the_wizard(self) -> None:
        raw = {
            "entrypoint": "A",
            "agents": [{"name": "A", "llm_provider": "p", "tool_bundles": ["b"]}],
            "llm_providers": [{"name": "p", "type": "openai", "model_name": "m"}],
            "tool_bundles": [{"name": "b", "bundles": ["browser"], "browser": {"typo": 1}}],
        }
        assert cfg.current_settings_from(raw)[1].browser == cfg.BrowserOptions()

    def test_config_shows_the_fences_and_never_the_cdp_url(self) -> None:
        provider = cfg.ProviderConfig(type="openai", model_name="m")
        agent = cfg.AgentConfig(
            tools=["browser"], browser=cfg.BrowserOptions(cdp_url="wss://host/?token=s3cret")
        )
        raw = cfg.configuration_v1_dict_from(provider, agent, cfg.CompactionConfig())
        description = cfg.describe_configuration_v1(raw)
        assert "s3cret" not in str(description)
        bundle = next(s for s in description.sections if s.title == "Tool Bundles").items[0]
        fields = dict(bundle.fields)
        assert fields["browser cdp"] == "set"
        assert fields["browser allowed_domains"] == "(any public site)"
