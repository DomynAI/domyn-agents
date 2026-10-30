"""Declaring a decision model in YAML.

What raises and what degrades is the point of these: a configuration mistake
surfaces at load time, because a silently disabled feature someone believes
they configured is worse than a failed start.
"""

import pytest

from domyn.agents.decisions import DecisionMode, Provenance
from domyn.agents.decisions.config import (
    DecisionProviderConfig,
    DecisionsConfig,
    UseCaseConfig,
)


class TestTheFlatYAMLForm:
    def test_each_use_case_is_its_own_key(self):
        config = DecisionsConfig.from_dict(
            {"provider": "local_clef", "approvals": {"mode": "active", "min_confidence": 0.85}}
        )
        assert config.provider == "local_clef"
        assert config.use_cases["approvals"].mode is DecisionMode.ACTIVE
        assert config.policies()["approvals"].threshold == 0.85

    def test_an_empty_use_case_block_is_off(self):
        config = DecisionsConfig.from_dict({"provider": "p", "approvals": None})
        assert config.use_cases["approvals"].mode is DecisionMode.OFF

    def test_a_missing_threshold_takes_the_use_cases_default(self):
        config = DecisionsConfig.from_dict({"provider": "p", "approvals": {"mode": "shadow"}})
        assert config.policies()["approvals"].threshold == 0.8


class TestAConfigurationMistakeIsLoud:
    def test_an_unknown_use_case_is_refused(self):
        with pytest.raises(ValueError, match="Unknown decision use case"):
            DecisionsConfig.from_dict({"provider": "p", "aprovals": {"mode": "active"}})

    def test_the_error_lists_the_known_ones(self):
        with pytest.raises(ValueError, match="approvals"):
            DecisionsConfig.from_dict({"provider": "p", "nonsense": {"mode": "active"}})

    def test_an_unknown_provider_name_is_refused(self):
        config = DecisionsConfig.from_dict({"provider": "typo", "approvals": {"mode": "off"}})
        with pytest.raises(ValueError, match="not found in configuration"):
            config.build({"local_clef": object()})

    def test_an_out_of_range_threshold_is_refused(self):
        with pytest.raises(ValueError):
            UseCaseConfig(mode="active", min_confidence=1.5)

    def test_an_unknown_mode_is_refused(self):
        with pytest.raises(ValueError):
            UseCaseConfig(mode="sometimes")

    def test_a_typo_in_a_key_is_refused_rather_than_ignored(self):
        with pytest.raises(ValueError):
            UseCaseConfig(mode="active", min_confidances=0.9)


class TestNoProviderMeansTodaysBehaviour:
    def test_a_block_without_a_provider_builds_nothing(self):
        config = DecisionsConfig.from_dict({"approvals": {"mode": "active"}})
        assert config.build({}) is None


class TestBuildingAProvider:
    def test_the_type_selects_the_backend(self):
        entry = DecisionProviderConfig(
            name="local_clef",
            type="system_one",
            base_url="http://localhost:28080",
            model="Cloudflare/clef",
        )
        provider = entry.build()
        assert provider.provenance is Provenance.NATIVE
        assert provider.model == "Cloudflare/clef"
        assert provider.url == "http://localhost:28080/v1/systemone"

    def test_extra_keys_reach_the_backend(self):
        entry = DecisionProviderConfig(
            name="x",
            type="system_one",
            base_url="http://h",
            model="m",
            max_state_tokens=73728,
            timeout_s=1.5,
        )
        provider = entry.build()
        assert provider.max_state_tokens == 73728
        assert provider.timeout_s == 1.5

    def test_an_unknown_type_raises(self):
        with pytest.raises(ValueError, match="Unknown decision provider type"):
            DecisionProviderConfig(name="x", type="clef-ish").build()


class TestTheWholeRuntime:
    def test_it_comes_out_configured(self):
        provider = DecisionProviderConfig(
            name="local_clef", type="system_one", base_url="http://h", model="m"
        ).build()
        config = DecisionsConfig.from_dict(
            {"provider": "local_clef", "approvals": {"mode": "active", "min_confidence": 0.9}}
        )
        runtime = config.build({"local_clef": provider})

        assert runtime is not None
        assert runtime.mode("approvals") is DecisionMode.ACTIVE
        assert runtime.threshold("approvals") == 0.9
        # Everything not named stays off.
        assert runtime.mode("skills") is DecisionMode.OFF
