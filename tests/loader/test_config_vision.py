"""`vision` on a provider entry is what gives an agent `read_image`."""

from pathlib import Path
from typing import Any

from domyn.agents.loader import ConfigurationV1


def _config(tmp_path: Path, provider: dict[str, Any], bundles: list[str]) -> dict[str, Any]:
    return {
        "version": "1.0",
        "entrypoint": "Reader",
        "llm_providers": [
            {"name": "p", "type": "vllm", "model_name": "m", "url": "http://test.com", **provider}
        ],
        "tool_bundles": [{"name": "local", "bundles": bundles, "workspace": str(tmp_path)}],
        "agents": [
            {
                "name": "Reader",
                "description": "Reads",
                "instruction": "Read",
                "llm_provider": "p",
                "tool_bundles": ["local"],
            }
        ],
    }


def _tool_names(config: dict[str, Any]) -> list[str]:
    return [tool.name for tool in ConfigurationV1.from_dict(config).agents[0].tools]


def test_a_provider_that_reads_images_gives_the_agent_read_image(tmp_path: Path) -> None:
    config = ConfigurationV1.from_dict(_config(tmp_path, {"vision": True}, ["files"]))
    agent = config.agents[0]

    assert agent.llm_provider.vision is True
    assert "read_image" in [tool.name for tool in agent.tools]


def test_a_provider_that_says_nothing_does_not(tmp_path: Path) -> None:
    assert "read_image" not in _tool_names(_config(tmp_path, {}, ["files"]))


def test_not_without_the_files_bundle(tmp_path: Path) -> None:
    assert "read_image" not in _tool_names(_config(tmp_path, {"vision": True}, ["shell"]))
