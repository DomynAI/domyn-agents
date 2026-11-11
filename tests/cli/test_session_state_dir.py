"""`domyn run` keeps the tools' state in ``home_dir()``, beside ``conversations.db``:
the project's folder, else ``~/.domyn``."""

from pathlib import Path
from typing import Any

import pytest

from domyn.agents.cli.commands.run import command as run_command_module


@pytest.mark.asyncio
async def test_the_session_agent_is_loaded_with_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = tmp_path / "config.yaml"
    config.write_text(
        "version: '1.0'\nentrypoint: A\n"
        "llm_providers: [{name: p, type: vllm, model_name: m, url: 'http://x'}]\n"
        "agents: [{name: A, llm_provider: p}]\n",
        encoding="utf-8",
    )
    seen: dict[str, Any] = {}

    async def _load(path: str, **kwargs: Any) -> Any:
        seen.update(kwargs)
        raise ValueError("stop here")

    monkeypatch.setattr(run_command_module, "load_application_from_file", _load)
    monkeypatch.setattr(run_command_module, "home_dir", lambda: tmp_path / "state")

    with pytest.raises(ValueError, match="stop here"):
        await run_command_module.load_session_agent(config, run_command_module.SessionOptions())

    assert seen["state_dir"] == str(tmp_path / "state")
