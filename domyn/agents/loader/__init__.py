import asyncio
import os
import re
from collections.abc import Iterable
from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from domyn.agents.agents.agent import Agent
from domyn.agents.runner import _return_subagents

_ENV_VAR_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def interpolate_env_vars(value: Any, file_path: str) -> Any:
    """Recursively resolve ${VAR_NAME} references in string values against os.environ."""
    if isinstance(value, str):

        def replace(match: re.Match[str]) -> str:
            name = match.group(1)
            resolved = os.environ.get(name)
            if resolved is None:
                raise ValueError(
                    f"Environment variable '{name}' referenced as '${{{name}}}' "
                    f"in {file_path} is not set"
                )
            return resolved

        return _ENV_VAR_PATTERN.sub(replace, value)
    if isinstance(value, dict):
        return {k: interpolate_env_vars(v, file_path) for k, v in value.items()}
    if isinstance(value, list):
        return [interpolate_env_vars(v, file_path) for v in value]
    return value


def load_from_file(file_path: str) -> dict[str, Any]:
    """Load configuration from a YAML file.

    String values may reference environment variables with ${VAR_NAME}
    syntax, resolved once at load time.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {file_path}")

    with open(path, encoding="utf-8") as f:
        config: dict[str, Any] = yaml.safe_load(f)

    interpolated: dict[str, Any] = interpolate_env_vars(config, file_path)
    return interpolated


async def load_application_from_config_new(
    config: dict[str, Any],
    mcp_init: bool = True,
    *,
    extra_workspace_roots: Iterable[str] = (),
) -> Agent:
    version = config.get("version", "1.0")
    match version:
        case "1.0":
            from domyn.agents.loader.config import ConfigurationV1

            cfg = ConfigurationV1.from_dict(config, extra_workspace_roots=extra_workspace_roots)
        case _:
            raise ValueError(f"Unsupported configuration version: {version}")

    if not cfg.entrypoint and cfg.agents:
        cfg.entrypoint = cfg.agents[-1].name

    # Resolve the entrypoint agent
    agent = next(
        filter(lambda a: a.name == cfg.entrypoint, cfg.agents),
        None,
    )
    if agent is None:
        raise ValueError(f"Agent '{cfg.entrypoint}' not found in configuration")

    tasks = []
    for sub_agent in _return_subagents(agent):
        if mcp_init and hasattr(sub_agent, "initialize_mcp_tools"):
            tasks.append(sub_agent.initialize_mcp_tools())

    await asyncio.gather(*tasks)

    return agent


async def load_application_from_file(
    file_path: str, *, extra_workspace_roots: Iterable[str] = ()
) -> Agent:
    """Load application configuration from a YAML file. Return the root agent."""
    config = load_from_file(file_path)
    return await load_application_from_config_new(
        config, extra_workspace_roots=extra_workspace_roots
    )


load_application_from_config = load_application_from_config_new


# Re-exported on first use: the defining modules import this package, so naming
# them here at import time would be a cycle.
_LAZY_EXPORTS = {
    "ConfigurationV1": "domyn.agents.loader.config",
}

if TYPE_CHECKING:
    from domyn.agents.loader.config import ConfigurationV1 as ConfigurationV1


def __getattr__(name: str) -> Any:
    if name in _LAZY_EXPORTS:
        value = getattr(import_module(_LAZY_EXPORTS[name]), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
