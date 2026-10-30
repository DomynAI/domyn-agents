"""`/model` and `domyn setup` must edit the config file, not rewrite it.

The wizard builds its half of the file from its own prompts. For a long time it
emitted exactly those keys and nothing else, which made every run a silent
delete of anything hand-written beside them. Measured on a config carrying a
decision model: the file came back without one, with no warning, and the
feature looked broken rather than erased.
"""

from domyn.agents.interfaces.cli.config import (
    configuration_v1_dict_from,
    current_settings_from,
)


def rebuild(raw: dict) -> dict:
    """What the wizard would write over this file."""
    provider, agent, compaction = current_settings_from(raw)
    return configuration_v1_dict_from(provider, agent, compaction, raw)


def base(**extra) -> dict:
    return {
        "version": "1.0",
        "entrypoint": "Assistant",
        "llm_providers": [
            {
                "name": "default_provider",
                "type": "vllm",
                "model_name": "m",
                "url": "http://x/v1/chat/completions",
                "context_window": 256_000,
            }
        ],
        "agents": [
            {
                "name": "Assistant",
                "description": "d",
                "instruction": "i",
                "llm_provider": "default_provider",
                **extra.pop("agent_extra", {}),
            }
        ],
        **extra,
    }


class TestItKeepsWhatItDidNotWrite:
    def test_a_decision_provider_survives(self):
        raw = base(
            decision_providers=[
                {"name": "local_clef", "type": "system_one", "model": "Cloudflare/clef"}
            ]
        )
        assert rebuild(raw)["decision_providers"] == raw["decision_providers"]

    def test_an_agents_decision_block_survives(self):
        raw = base(
            agent_extra={
                "decisions": {
                    "provider": "local_clef",
                    "approvals": {"mode": "active", "min_confidence": 0.85},
                }
            }
        )
        rebuilt = rebuild(raw)["agents"][0]
        assert rebuilt["decisions"]["approvals"]["min_confidence"] == 0.85

    def test_mcp_servers_survive(self):
        """Not a decisions-specific fix: every hand-added key was being lost."""
        raw = base(mcp_servers=[{"name": "s", "url": "https://example.com/mcp"}])
        assert rebuild(raw)["mcp_servers"] == raw["mcp_servers"]

    def test_hooks_and_extra_tools_survive(self):
        raw = base(
            hooks=[{"name": "h", "path": "x.py", "type": "listener", "trigger": "pre_agent_call"}],
            tools=[{"name": "t", "type": "function", "path": "x.py"}],
        )
        rebuilt = rebuild(raw)
        assert rebuilt["hooks"] == raw["hooks"]
        assert rebuilt["tools"] == raw["tools"]

    def test_a_second_agent_survives(self):
        raw = base()
        raw["agents"].append(
            {"name": "Helper", "description": "d", "instruction": "i", "llm_provider": "p"}
        )
        names = [a["name"] for a in rebuild(raw)["agents"]]
        assert names == ["Assistant", "Helper"]

    def test_an_unknown_key_on_the_entrypoint_agent_survives(self):
        raw = base(agent_extra={"memory_manager": {"type": "filesystem"}})
        assert rebuild(raw)["agents"][0]["memory_manager"] == {"type": "filesystem"}


class TestItStillOwnsItsOwnHalf:
    def test_the_provider_is_replaced_not_merged(self):
        """The whole point of `/model`: the new model has to win."""
        raw = base()
        provider, agent, compaction = current_settings_from(raw)
        provider.model_name = "a-different-model"
        rebuilt = configuration_v1_dict_from(provider, agent, compaction, raw)
        assert rebuilt["llm_providers"][0]["model_name"] == "a-different-model"
        assert len(rebuilt["llm_providers"]) == 1

    def test_the_entrypoint_agents_instruction_is_the_wizards(self):
        raw = base()
        provider, agent, compaction = current_settings_from(raw)
        agent.instruction = "a new instruction"
        rebuilt = configuration_v1_dict_from(provider, agent, compaction, raw)
        assert rebuilt["agents"][0]["instruction"] == "a new instruction"

    def test_the_entrypoint_agent_is_not_duplicated(self):
        raw = base(agent_extra={"decisions": {"provider": "p"}})
        assert len(rebuild(raw)["agents"]) == 1

    def test_without_an_existing_file_nothing_is_invented(self):
        raw = base(decision_providers=[{"name": "x", "type": "none"}])
        provider, agent, compaction = current_settings_from(raw)
        fresh = configuration_v1_dict_from(provider, agent, compaction)
        assert "decision_providers" not in fresh
