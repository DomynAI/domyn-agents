"""Configuring a decision model from the wizard, instead of by hand.

Until this existed the only way to switch one on was editing `config.yaml`,
which made the feature reachable only by someone who had read the source — and
for a while `/model` then deleted what they had written.

The tests that matter most here are the two round trips: what the wizard writes
has to load through the real loader and produce a runtime, and what it reads
back has to be what it would write again. A projection that only agrees with
itself is how a config key gets renamed with the suite still green.
"""

from unittest.mock import MagicMock, patch

import httpx
import pytest
import respx
import yaml

from domyn.agents import load_application_from_file
from domyn.agents.decisions import DecisionMode, SystemOneProvider, runtime_of
from domyn.agents.interfaces.cli import Theme, setup
from domyn.agents.interfaces.cli import select as select_mod
from domyn.agents.interfaces.cli.config import (
    DECISION_LEVELS,
    DEFAULT_DECISION_MODEL,
    DEFAULT_DECISION_URL,
    DecisionsChoice,
    configuration_v1_dict_from,
    current_decisions_from,
    current_settings_from,
)

ENDPOINT = "http://localhost:28080"
MODEL = "Cloudflare/clef"


def base() -> dict:
    return {
        "version": "1.0",
        "entrypoint": "Assistant",
        "llm_providers": [
            {
                "name": "default_provider",
                "type": "vllm",
                "model_name": "m",
                "url": "http://x/v1/chat/completions",
                "context_window": 200_000,
            }
        ],
        "agents": [
            {
                "name": "Assistant",
                "description": "d",
                "instruction": "i",
                "llm_provider": "default_provider",
            }
        ],
    }


def written(raw: dict, decisions: DecisionsChoice | None) -> dict:
    provider, agent, compaction = current_settings_from(raw)
    return configuration_v1_dict_from(provider, agent, compaction, raw, decisions)


def choice(level: str) -> DecisionsChoice:
    return DecisionsChoice(level=level, base_url=ENDPOINT, model=MODEL)


class TestWhatTheWizardWrites:
    def test_a_level_writes_a_provider_and_an_agent_block(self):
        config = written(base(), choice("active"))
        assert config["decision_providers"][0]["base_url"] == ENDPOINT
        assert config["decision_providers"][0]["type"] == "system_one"
        assert config["agents"][0]["decisions"]["provider"] == "decision_model"

    def test_the_model_name_is_written_as_given(self):
        """It is the name the *server* exposes, and deriving it is how a config
        ends up asking for `clef` from something serving `Cloudflare/clef`."""
        config = written(base(), choice("shadow"))
        assert config["decision_providers"][0]["model"] == MODEL

    @pytest.mark.parametrize("level", [name for name in DECISION_LEVELS if name != "no"])
    def test_every_level_round_trips_to_itself(self, level):
        config = written(base(), choice(level))
        assert current_decisions_from(config).level == level

    def test_choosing_no_removes_what_an_earlier_run_wrote(self):
        """A wizard that can only switch something on is a wizard people edit
        the file to undo."""
        configured = written(base(), choice("active"))
        off = written(configured, DecisionsChoice(level="no"))
        assert "decision_providers" not in off
        assert "decisions" not in off["agents"][0]

    def test_the_endpoint_and_model_survive_switching_off_and_on(self):
        configured = written(base(), choice("active"))
        current = current_decisions_from(configured)
        assert (current.base_url, current.model) == (ENDPOINT, MODEL)


class TestModelDoesNotTouchIt:
    """`/model` runs the wizard with `provider_only`, which never asks about
    decisions — so it passes `None`, and `None` has to mean "leave it"."""

    def test_passing_nothing_keeps_an_existing_decision_model(self):
        configured = written(base(), choice("active, approvals too"))
        after = written(configured, None)
        assert after["decision_providers"] == configured["decision_providers"]
        assert after["agents"][0]["decisions"] == configured["agents"][0]["decisions"]

    def test_passing_nothing_does_not_invent_one(self):
        assert "decision_providers" not in written(base(), None)

    def test_a_hand_written_use_case_the_wizard_never_offers_is_kept(self):
        """The wizard owns this block only when it asked. `/model` must not
        quietly narrow a config someone tuned by hand."""
        raw = base()
        raw["decision_providers"] = [
            {"name": "mine", "type": "system_one", "base_url": ENDPOINT, "model": MODEL}
        ]
        raw["agents"][0]["decisions"] = {
            "provider": "mine",
            "sensitivity": {"mode": "active", "on_unknown": "block"},
            "tools": {"mode": "active", "min_confidence": 0.95},
        }
        after = written(raw, None)
        assert after["agents"][0]["decisions"]["tools"]["min_confidence"] == 0.95
        assert after["agents"][0]["decisions"]["sensitivity"]["on_unknown"] == "block"


class TestReadingAFileBackIn:
    def test_a_file_with_no_decision_model_opens_on_no(self):
        assert current_decisions_from(base()).level == "no"

    def test_nothing_at_all_opens_on_the_defaults(self):
        assert current_decisions_from(None).level == "no"

    def test_a_hand_written_active_use_case_reads_as_active(self):
        """A file the four levels cannot express still has to open the prompt
        on something true, rather than on "no" over a model that is running."""
        raw = base()
        raw["decision_providers"] = [{"name": "mine", "type": "system_one", "model": MODEL}]
        raw["agents"][0]["decisions"] = {"provider": "mine", "tools": {"mode": "active"}}
        assert current_decisions_from(raw).level == "active"

    def test_an_active_approval_gate_reads_as_the_level_that_hands_it_over(self):
        raw = base()
        raw["decision_providers"] = [{"name": "mine", "type": "system_one", "model": MODEL}]
        raw["agents"][0]["decisions"] = {"provider": "mine", "approvals": {"mode": "active"}}
        assert current_decisions_from(raw).level == "active, approvals too"

    def test_a_provider_declared_but_used_nowhere_reads_as_no(self):
        raw = base()
        raw["decision_providers"] = [{"name": "mine", "type": "system_one", "model": MODEL}]
        assert current_decisions_from(raw).level == "no"


class TestTheFileTheWizardWroteActuallyLoads:
    """The half no projection test can reach: YAML on disk, through the real
    loader, into a runtime. These are the lines that decide whether the feature
    switches on at all."""

    async def built(self, tmp_path, monkeypatch, level: str):
        monkeypatch.setenv("VLLM_API_KEY", "not-a-real-key")
        config = written(base(), choice(level))
        path = tmp_path / "config.yaml"
        path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        return await load_application_from_file(str(path))

    @pytest.mark.asyncio
    async def test_the_provider_is_built_from_what_was_written(self, tmp_path, monkeypatch):
        runtime = runtime_of(await self.built(tmp_path, monkeypatch, "active"))
        assert isinstance(runtime.provider, SystemOneProvider)
        assert runtime.provider.model == MODEL

    @pytest.mark.asyncio
    async def test_active_means_active_on_the_agent(self, tmp_path, monkeypatch):
        runtime = runtime_of(await self.built(tmp_path, monkeypatch, "active"))
        assert runtime.mode("skills") is DecisionMode.ACTIVE
        assert runtime.mode("instructions") is DecisionMode.ACTIVE

    @pytest.mark.asyncio
    async def test_active_still_leaves_the_approval_gate_measuring(self, tmp_path, monkeypatch):
        """The level below the last one. Allowing removes a human from the
        loop, so it is not what "active" buys."""
        runtime = runtime_of(await self.built(tmp_path, monkeypatch, "active"))
        assert runtime.mode("approvals") is DecisionMode.SHADOW

    @pytest.mark.asyncio
    async def test_the_last_level_hands_the_gate_over(self, tmp_path, monkeypatch):
        runtime = runtime_of(await self.built(tmp_path, monkeypatch, "active, approvals too"))
        assert runtime.mode("approvals") is DecisionMode.ACTIVE

    @pytest.mark.asyncio
    async def test_shadow_acts_nowhere(self, tmp_path, monkeypatch):
        runtime = runtime_of(await self.built(tmp_path, monkeypatch, "shadow"))
        assert all(
            runtime.mode(use_case) is DecisionMode.SHADOW
            for use_case in ("approvals", "skills", "instructions")
        )

    @pytest.mark.asyncio
    async def test_what_the_wizard_never_offered_stays_off(self, tmp_path, monkeypatch):
        runtime = runtime_of(await self.built(tmp_path, monkeypatch, "active, approvals too"))
        assert runtime.mode("tools") is DecisionMode.OFF
        assert runtime.mode("context_ladder") is DecisionMode.OFF
        assert runtime.mode("sensitivity") is DecisionMode.OFF


class TestTheEndpointProbe:
    """Checked with a real request, not a socket connect.

    The two mistakes that actually happen here are an endpoint that is not
    up and a model name the server does not expose. A connect sees the first
    and cannot see the second, and the second is the quieter failure: the
    config looks right and every turn silently stops asking.
    """

    @respx.mock
    def test_an_answering_endpoint_is_no_problem(self):
        respx.post(f"{ENDPOINT}/v1/systemone").mock(
            return_value=httpx.Response(200, json={"answers": {"q": {"type": "noul", "noul": 0.9}}})
        )
        assert setup._decision_endpoint_answers(ENDPOINT, MODEL) is None

    @respx.mock
    def test_a_wrong_model_name_is_reported_with_what_the_server_said(self):
        """The server names the model it does serve, and that line is the
        whole fix — so it has to reach the screen.

        Status and body of a server asked for a model it does not serve: it
        answers **404**, not the 400 this test first assumed, and names the one
        it does. A socket connect would have called the same configuration
        healthy."""
        respx.post(f"{ENDPOINT}/v1/systemone").mock(
            return_value=httpx.Response(
                404,
                text='{"detail":"model \'clef\' is not served here; '
                "available: 'Cloudflare/clef'\"}",
            )
        )
        problem = setup._decision_endpoint_answers(ENDPOINT, "clef")
        assert problem is not None
        assert "404" in problem
        assert "Cloudflare/clef" in problem

    @respx.mock
    def test_an_unreachable_endpoint_says_so_without_raising(self):
        respx.post(f"{ENDPOINT}/v1/systemone").mock(side_effect=httpx.ConnectError("refused"))
        problem = setup._decision_endpoint_answers(ENDPOINT, MODEL)
        assert problem is not None
        assert ENDPOINT in problem

    @respx.mock
    def test_a_long_error_body_is_cut_rather_than_flooding_the_screen(self):
        respx.post(f"{ENDPOINT}/v1/systemone").mock(
            return_value=httpx.Response(500, text="x" * 5_000)
        )
        problem = setup._decision_endpoint_answers(ENDPOINT, MODEL)
        assert problem is not None and len(problem) < 300

    @respx.mock
    def test_a_trailing_slash_does_not_double_up(self):
        route = respx.post(f"{ENDPOINT}/v1/systemone").mock(
            return_value=httpx.Response(200, json={"answers": {}})
        )
        setup._decision_endpoint_answers(f"{ENDPOINT}/", MODEL)
        assert route.called


class TestThePromptItself:
    def theme(self):
        return Theme.build(allow_animation=False)

    @pytest.fixture(autouse=True)
    def numbered_prompts(self, monkeypatch):
        """No TTY in a test, which is the numbered fallback — the same path a
        pipe takes."""
        monkeypatch.setattr(select_mod, "_interactive", lambda theme: False)

    def test_keeping_the_default_asks_for_no_endpoint(self, monkeypatch):
        """Choosing "no" must not then interrogate someone about a URL they
        just said they did not want."""
        monkeypatch.setattr(
            setup, "_decision_endpoint_answers", lambda *a: pytest.fail("probed on 'no'")
        )
        with patch("builtins.input", side_effect=[""]):
            answer = setup._ask_decisions(MagicMock(), self.theme(), DecisionsChoice())
        assert answer.level == "no"

    @respx.mock
    def test_choosing_a_level_asks_for_the_endpoint_and_the_model(self):
        respx.post(f"{ENDPOINT}/v1/systemone").mock(
            return_value=httpx.Response(200, json={"answers": {}})
        )
        with patch("builtins.input", side_effect=["3", ENDPOINT, MODEL, ""]):
            answer = setup._ask_decisions(MagicMock(), self.theme(), DecisionsChoice())
        assert (answer.level, answer.base_url, answer.model) == ("active", ENDPOINT, MODEL)

    @respx.mock
    def test_an_unreachable_endpoint_is_still_saved(self):
        """It may come up after setup, exactly as agent-browser may be
        installed after it."""
        respx.post(f"{ENDPOINT}/v1/systemone").mock(side_effect=httpx.ConnectError("refused"))
        console = MagicMock()
        with patch("builtins.input", side_effect=["2", ENDPOINT, MODEL, ""]):
            answer = setup._ask_decisions(console, self.theme(), DecisionsChoice())
        assert answer.level == "shadow"
        printed = " ".join(str(call) for call in console.print.call_args_list)
        assert "saved anyway" in printed

    def test_erased_fields_go_back_to_the_default_endpoint_and_model(self, monkeypatch):
        """A changed endpoint opens the field; erasing it is the way back."""
        monkeypatch.setattr(setup, "_decision_endpoint_answers", lambda *a: None)
        monkeypatch.delenv("DECISIONS_API_KEY", raising=False)
        seen: list[str] = []

        def erasing(console, message, *, prefill="", **kwargs):
            seen.append(prefill)
            return ""

        monkeypatch.setattr(setup, "read_line", erasing)
        current = DecisionsChoice(base_url="http://other:9000", model="other/model")
        with patch("builtins.input", side_effect=["3"]):
            answer = setup._ask_decisions(MagicMock(), self.theme(), current)

        assert "http://other:9000" in seen and "other/model" in seen
        assert (answer.base_url, answer.model) == (DEFAULT_DECISION_URL, DEFAULT_DECISION_MODEL)

    def test_aborting_the_choice_aborts_the_wizard(self):
        with (
            patch.object(setup, "select", lambda *a, **k: None),
            pytest.raises(KeyboardInterrupt),
        ):
            setup._ask_decisions(MagicMock(), self.theme(), DecisionsChoice())


class TestTheApiKey:
    def theme(self):
        return Theme.build(allow_animation=False)

    @pytest.fixture(autouse=True)
    def numbered_prompts(self, monkeypatch):
        monkeypatch.setattr(select_mod, "_interactive", lambda theme: False)
        monkeypatch.delenv("DECISIONS_API_KEY", raising=False)

    @respx.mock
    def test_the_probe_sends_the_key_as_a_bearer_token(self):
        route = respx.post(f"{ENDPOINT}/v1/systemone").mock(
            return_value=httpx.Response(200, json={"answers": {}})
        )
        setup._decision_endpoint_answers(ENDPOINT, MODEL, "s3cret")
        assert route.calls.last.request.headers["authorization"] == "Bearer s3cret"

    @respx.mock
    def test_the_probe_sends_no_header_without_a_key(self):
        route = respx.post(f"{ENDPOINT}/v1/systemone").mock(
            return_value=httpx.Response(200, json={"answers": {}})
        )
        setup._decision_endpoint_answers(ENDPOINT, MODEL)
        assert "authorization" not in route.calls.last.request.headers

    @respx.mock
    def test_a_pasted_key_goes_to_credentials_and_the_yaml_names_the_variable(
        self, monkeypatch, tmp_path
    ):
        monkeypatch.setattr(setup, "credentials_path", lambda: tmp_path / "credentials")
        saved = {}
        monkeypatch.setattr(
            setup,
            "write_credential",
            lambda name, value: saved.setdefault(name, value) and tmp_path,
        )
        route = respx.post(f"{ENDPOINT}/v1/systemone").mock(
            return_value=httpx.Response(200, json={"answers": {}})
        )
        with patch("builtins.input", side_effect=["3", ENDPOINT, MODEL, "s3cret"]):
            answer = setup._ask_decisions(MagicMock(), self.theme(), DecisionsChoice())
        assert saved == {"DECISIONS_API_KEY": "s3cret"}
        assert route.calls.last.request.headers["authorization"] == "Bearer s3cret"
        provider = written(base(), answer)["decision_providers"][0]
        assert provider["api_key"] == "${DECISIONS_API_KEY}"
        assert "s3cret" not in str(written(base(), answer))

    def test_the_reference_is_read_back(self):
        raw = written(base(), DecisionsChoice(level="shadow", api_key_env="MY_KEY"))
        assert current_decisions_from(raw).api_key_env == "MY_KEY"

    def test_no_key_writes_no_api_key(self):
        assert "api_key" not in written(base(), choice("shadow"))["decision_providers"][0]
