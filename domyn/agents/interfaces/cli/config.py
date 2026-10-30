"""Persisted CLI configuration, so ``domyn run`` works without an agent file.

Without this, the CLI is only usable by someone who already has a Python agent
module — which is a poor first five minutes. With it, ``domyn run`` boots a
default agent from ``config.yaml`` and every setting can be changed later, from
the CLI or from inside a session.

**API keys are never written to the YAML.** Config files get committed, shared and
pasted into issues; keys go to a separate credentials file with ``0600``
permissions, or are read from an environment variable. The YAML only ever holds
the *name* of the variable.

No default model names are shipped on purpose: hardcoded model identifiers go
stale, and a wrong default is worse than a prompt asking for one.
"""

import copy
import hashlib
import os
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, overload
from urllib.parse import urlsplit, urlunsplit

import yaml
from dotenv import dotenv_values, find_dotenv
from pydantic import AliasChoices, BaseModel, Field, ValidationError
from rich.console import Console

from domyn.agents.approvals import (
    ApprovalPolicy,
    ApprovalScope,
    ConversationApprovalStore,
    FileApprovalStore,
)
from domyn.agents.instructions.discovery import AUTODISCOVERY_DIRS
from domyn.agents.interfaces.cli.theme import Theme
from domyn.agents.llm import BaseLLMProvider
from domyn.agents.llm.context_window import resolve_context_window
from domyn.agents.loader import interpolate_env_vars
from domyn.agents.logger import get_logger
from domyn.agents.planner_strategy.context_management.compaction import CompactionConfig
from domyn.agents.skills import LocalDirSkillSource
from domyn.agents.skills.catalog import BUILTIN_SKILLS_ROOT
from domyn.agents.skills.layout import listed_path_is_linked
from domyn.agents.tools.builtin import BUNDLE_TOOL_NAMES, DEFAULT_BUNDLES, BrowserOptions

CONFIG_FILENAME = "config.yaml"
CREDENTIALS_FILENAME = "credentials"

PROJECT_DIR = Path(".domyn")
GLOBAL_DIR = Path.home() / ".domyn"

# The planner the CLI runs on, deliberately fixed rather than configurable.
#
# `tool_use_thought` is native tool calling plus the model's reasoning surfaced
# as `Part(thought=...)`, which is what lets the transcript show *why* a tool is
# being called. Plain `tool_use` drops that entirely; `react` needs
# `<thought>/<action>` replies that a normal chat model does not produce, and
# loops on "Unparsable action". `tool_use_thought` is also the framework
# default, the planner an `Agent` gets when it names none.
#
# It is not a config field because it was one, and that turned out to be a trap:
# the wizard wrote the value of the day into config.yaml, and a config written
# once pins that choice forever — changing the default in code never reaches
# anyone who has already run setup. Nothing asks the user about the planner,
# so nothing should freeze their answer to a question they were never asked.
#
# A stale `planner:` left in an old config is ignored: pydantic drops unknown
# keys, so those files heal themselves on the next run.
CLI_PLANNER = "tool_use_thought"

# Per provider: the environment variable conventionally holding its key, and a
# pointer to where a user finds valid model names. Deliberately no model list.
# Order matters: it is the order `domyn setup` offers them in, so Domyn comes first.
PROVIDER_HINTS: dict[str, dict[str, str]] = {
    "domyn": {
        "env": "DOMYN_API_KEY",
        "models": "the models of your space's configuration on the Domyn platform",
    },
    "openai": {"env": "OPENAI_API_KEY", "models": "https://platform.openai.com/docs/models"},
    "anthropic": {
        "env": "ANTHROPIC_API_KEY",
        "models": "https://docs.claude.com/en/docs/about-claude/models",
    },
    "open_router": {"env": "OPENROUTER_API_KEY", "models": "https://openrouter.ai/models"},
    "cerebras": {"env": "CEREBRAS_API_KEY", "models": "https://inference-docs.cerebras.ai/models"},
    "vllm": {"env": "VLLM_API_KEY", "models": "your vLLM deployment"},
    "custom": {"env": "LLM_API_KEY", "models": "your endpoint"},
}

# Providers whose endpoint we cannot guess.
REQUIRES_URL = ("domyn", "vllm", "custom")

_CHAT_COMPLETIONS = "/chat/completions"


def vllm_chat_url(url: str) -> str:
    """The chat completions endpoint for a vLLM server URL typed into ``domyn setup``.

    ``VLLMProvider`` posts to its ``url`` as given, so a server's base URL would be a
    404 on the first message. ``http://host:8000`` and ``http://host:8000/v1`` both
    become ``http://host:8000/v1/chat/completions``; a URL already ending in
    ``/chat/completions`` is kept as written, so a gateway serving it under another
    prefix still works. Query and fragment are kept.

    The CLI's only: the provider itself takes its ``url`` verbatim, for the services
    that build it in code.
    """
    parts = urlsplit(url.strip())
    path = parts.path.rstrip("/")
    if path.endswith(_CHAT_COMPLETIONS):
        return url
    path += _CHAT_COMPLETIONS if path.endswith("/v1") else "/v1" + _CHAT_COMPLETIONS
    return urlunsplit(parts._replace(path=path))


DEFAULT_CONTEXT_WINDOW = 150_000
"""Window assumed when nobody declares one and the provider cannot be asked.

Without a number every context policy is inert, which meant compaction could
never fire on a self-hosted model — the common case here. So the CLI's own
configuration asks the provider first (see `apply_context_window`) and, when
that yields nothing, assumes one. The choice of value follows the direction of
the damage:

- **Assume too large** (real model smaller): we compact late or not at all and
  the provider rejects the request — which then states the real window, and it
  is adopted.
- **Assume too small** (real model larger): we compact earlier than necessary.
  Cheap and safe.

150k sits above the common 128k deployments — so those keep their full window
instead of being compacted early — while still catching a runaway session before
a 200k+ model would reject it.
"""


async def apply_context_window(provider: Any) -> None:
    """Give the CLI configuration's provider a window: declared, else asked of the
    provider (`resolve_context_window`), else `DEFAULT_CONTEXT_WINDOW`.

    Before the first request, so the session's first message is already measured
    against the window the server reports.

    Kept for the session: the provider holds a window from here on, so nothing is
    asked again until `/model` or the next start, a question that failed included.
    150k when the real window is smaller costs one rejection, which states it; when
    it is larger, compaction comes earlier than it needs to.
    """
    if not isinstance(provider, BaseLLMProvider):
        return
    if await resolve_context_window(provider) is None:
        provider.context_window = DEFAULT_CONTEXT_WINDOW


AUTO_CONTEXT_WINDOW = "auto"
"""A provider with no ``context_window``: asked at session start, else
``DEFAULT_CONTEXT_WINDOW`` (`apply_context_window`). The one spelling of it, typed
and shown by the wizard and shown by ``/config``, which describes the file;
``/context`` shows the window the session got."""


class ProviderConfig(BaseModel):
    """LLM provider settings. Never holds a secret."""

    type: str = "domyn"
    model_name: str = ""
    url: str | None = None
    space_id: str | None = None
    """Space whose configuration holds the model. Only the ``domyn`` provider reads it."""
    configuration_id: str | None = None
    """Configuration to take the model from; the space's last released one when empty. ``domyn`` only."""
    api_key_env: str | None = None

    context_window: int | None = Field(
        default=None,
        validation_alias=AliasChoices("context_window", "max_model_len"),
    )
    """Total token window of the model, when the user knows it.

    ``max_model_len`` is accepted as a spelling of the same thing: that is what
    vLLM calls it on the command line that decides it, so it is the name whoever
    knows the number will reach for.

    Left empty, the provider is asked (vLLM reports its ``max_model_len``), and
    :data:`DEFAULT_CONTEXT_WINDOW` is assumed when it cannot answer. Never
    inferred from a model name on a self-hosted endpoint: there the window is how
    the server was started, not a property of the weights.
    """

    max_tokens: int = 16000
    """Ceiling on one generation.

    The framework default is 2000, which is fine for a chat reply and far too
    low for an agent: writing an HTML page or a module into a tool argument
    passes it easily, and the generation is then cut off *mid-JSON-string*. The
    provider reports ``finish_reason: "length"``, the arguments do not parse, and
    the failure surfaces as "malformed tool-call arguments" — sending everyone
    looking for an escaping mistake that was never made. That cost a session to
    find, hence the comment.
    """

    def hints(self) -> dict[str, str]:
        return PROVIDER_HINTS.get(self.type, {})

    def default_key_env(self) -> str:
        return self.api_key_env or self.hints().get("env", "LLM_API_KEY")


class AgentConfig(BaseModel):
    """The default agent built when no agent file is given.

    The planner is **not** a field here: the CLI pins ``tool_use_thought`` and
    offers no way to change it. See :data:`CLI_PLANNER`.
    """

    name: str = "Assistant"
    description: str = "A general-purpose assistant"
    instruction: str = ""

    tools: list[str] = Field(default_factory=list)
    """Built-in tool bundles to mount. Empty means a conversation-only agent.

    Empty by default, so a config that names no bundle reads back as what it is.
    Whether the CLI agent gets tools is decided by ``domyn setup``, which starts
    with every bundle in ``BUNDLES`` ticked: an agent that can only talk is not
    useful for the work people open a terminal to do. Nothing here bypasses the
    approval gate: writing, editing and running commands all stop and ask, and
    the answer can be remembered for the conversation. See
    ``domyn.agents.tools.builtin``."""

    sandbox: str = "local"
    """Where `run_command` runs. Only ``local`` exists today — on this machine,
    with no isolation. Named now so an isolated backend can be selected later
    without a config migration, and so what you are running under is visible in
    `/config` rather than implied."""

    isolate_environment: bool = True
    """Give each conversation its own virtualenv under
    ``.domyn/conversations/<id>/venv``, so what the agent installs stays there.

    On by default because the alternative is what the CLI used to do: inherit
    whatever environment it was launched from, which under ``uv run domyn`` is
    the framework's own — an agent could break its own runtime with a
    ``pip install``. Turn it off to let commands use the ambient environment,
    which is what you want if the agent is meant to work *on* that project."""

    workspace: str | None = None
    """Root the tools may not leave. Defaults to the directory the CLI was
    started in, which is what makes ``cd`` into a project the way you scope an
    agent to it."""

    allowed_env: list[str] = Field(default_factory=list)
    """Extra environment variable names ``run_command``'s subprocess may see,
    beyond the fixed safe set (``PATH``, ``HOME``, etc.). Empty by default —
    secrets loaded into the CLI's own environment (from ``.env`` or
    ``~/.domyn/credentials``) are not visible to shell commands unless named
    here."""

    browser: BrowserOptions = Field(default_factory=BrowserOptions)
    """How the ``browser`` bundle launches and fences its browser. Only read
    when ``browser`` is in ``tools``, and kept when it is not, so turning
    browsing off and on again does not lose an allowlist."""

    stream: bool = True
    """Print the answer as it is generated rather than when it is finished.

    On by default because the difference is the whole feel of the tool: waiting
    in silence for ten seconds and then getting a paragraph reads as slow even
    when it is not. Every shipped provider implements ``generate_streaming``.
    Set it to ``false`` if a gateway in front of the model does not."""
    skills: dict[str, Any] | None = None
    standing_instructions: dict[str, Any] | None = None
    """The agent's ``standing_instructions:`` mapping as written, like ``skills``.

    ``domyn setup`` fills it with the files it found under the path the user gave;
    kept raw so a source setup did not write survives a re-run."""


PROJECTS_DIRNAME = "projects"
BUILTIN_SKILLS_DIRNAME = "builtin-skills"
"""Where ``domyn setup`` installs the built-in skills, in ``~/.domyn`` or a project.

Not ``skills``: started from ``$HOME``, ``./.domyn/skills`` (autodiscovery) is
``~/.domyn/skills``, and the folder setup rewrites from scratch must never be one the
user's own skills are read from."""

_PROJECT_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
# `_project_approvals_path` keeps a folder per working directory in the same place,
# named `<folder>-<12 hex>`. A project named like one would share its folder.
_APPROVALS_FOLDER_NAME = re.compile(r".*-[0-9a-f]{12}")

_active_project: Path | None = None
"""The folder of the project selected with ``--project``, for the rest of the process."""


PROJECT_DIR_VAR = "DOMYN_PROJECT_DIR"
"""Set to the selected project's folder; its config names its own files with it."""
PROJECT_DIR_REF = f"${{{PROJECT_DIR_VAR}}}"


class ProjectError(ValueError):
    """A project name that cannot be used, or a project that does not exist."""


def projects_dir() -> Path:
    return GLOBAL_DIR / PROJECTS_DIRNAME


def project_dir(name: str) -> Path:
    """``~/.domyn/projects/<name>``: a folder laid out like ``~/.domyn`` itself.

    Only the name is checked here, not that the folder exists: `domyn setup` creates it.
    """
    if not _PROJECT_NAME.fullmatch(name) or _APPROVALS_FOLDER_NAME.fullmatch(name):
        raise ProjectError(
            f"Invalid project name {name!r}: use only letters, digits, '.', '_' and '-', "
            "starting with a letter or digit (and not ending in '-' followed by 12 hex "
            "digits, a form domyn keeps for its own folders)."
        )
    return projects_dir() / name


def list_projects() -> list[str]:
    """The projects that have a configuration, by name."""
    if not projects_dir().is_dir():
        return []
    return sorted(
        child.name
        for child in projects_dir().iterdir()
        if child.is_dir() and (child / CONFIG_FILENAME).is_file()
    )


def is_project_path(value: str) -> bool:
    """Whether a ``--project`` value is a path rather than a name.

    A path has a ``/`` in it or starts with ``~`` or ``.`` (``./alpha``, ``../alpha``,
    ``~/work/alpha``, ``/srv/alpha``) — none of which a name may have — so the two
    readings never overlap: ``alpha`` is always ``~/.domyn/projects/alpha``.
    """
    return "/" in value or os.sep in value or value.startswith(("~", "."))


@overload
def use_project(name: str, *, must_exist: bool = True) -> Path: ...


@overload
def use_project(name: None, *, must_exist: bool = True) -> None: ...


@overload
def use_project(name: str | None, *, must_exist: bool = True) -> Path | None: ...


def use_project(name: str | None, *, must_exist: bool = True) -> Path | None:
    """Make ``name`` the project every config lookup of this process reads, and return
    its folder (``None`` for no project).

    A project is a folder that stands in for ``~/.domyn``: its ``config.yaml``, the
    skills setup installs (``skills/``), ``conversations.db``, ``credentials`` and
    approvals. Nothing is read from ``~/.domyn`` as a fallback. Its instructions are
    the files its config lists, as everywhere.

    ``name`` is a project under ``~/.domyn/projects``, or the path of a project folder
    anywhere else (`is_project_path`), read against the working directory.

    ``must_exist`` refuses a project with no configuration yet, which is what every
    command but `domyn setup` wants: a typo should not open the wizard on a new one.
    ``None`` selects no project, and removes `PROJECT_DIR_VAR` from the environment:
    the agent's shell reads it to tell which configuration the session runs, so a
    value inherited from the user's shell or a ``.env`` would point it at a project
    the session is not using. A config file run by name that uses the variable is run
    with ``--project <its folder>`` instead.
    """
    global _active_project
    if name is None:
        _active_project = None
        os.environ.pop(PROJECT_DIR_VAR, None)
        return None
    by_path = is_project_path(name)
    folder = Path(name).expanduser().resolve() if by_path else project_dir(name)
    if by_path and folder.exists() and not folder.is_dir():
        raise ProjectError(f"{folder} is a file, not a project folder.")
    if must_exist and not (folder / CONFIG_FILENAME).is_file():
        if by_path:
            raise ProjectError(
                f"There is no project at {folder}: it has no {CONFIG_FILENAME}. "
                f"To create one there, run `domyn setup --project {name}`."
            )
        known = list_projects()
        listed = (
            f" Existing projects: {', '.join(known)}." if known else " There are no projects yet."
        )
        raise ProjectError(
            f"There is no project named {name!r}. "
            f"To create it, run `domyn setup --project {name}`.{listed}"
        )
    _active_project = folder
    # What the project's config names its own files with. Set over any value already
    # in the environment: `--project` is the choice, and a stale variable left in a
    # shell must not point this config at another project's skills.
    os.environ[PROJECT_DIR_VAR] = str(folder)
    return folder


def active_project() -> Path | None:
    return _active_project


def describe_project(folder: Path) -> str:
    """``name (location)``, the home folder written ``~``: how a project is shown."""
    return f"{folder.name} ({project_location(folder)})"


def project_location(folder: Path) -> str:
    """Where a project is, the home folder written ``~``."""
    try:
        return f"~/{folder.relative_to(Path.home()).as_posix()}"
    except ValueError:
        return str(folder)


def home_dir() -> Path:
    """The folder that plays ``~/.domyn``: the active project's, else ``~/.domyn``."""
    return _active_project or GLOBAL_DIR


def config_dir(*, prefer_project: bool = True) -> Path:
    """Where config lives: the ``--project`` folder when one is selected, else
    ``./.domyn`` when present, else the home one.

    Project-local wins so a repo can pin its own model without touching the
    user's global setup.
    """
    if _active_project is not None:
        return _active_project
    if prefer_project and (PROJECT_DIR / CONFIG_FILENAME).is_file():
        return PROJECT_DIR
    return GLOBAL_DIR


def _config_paths(raw: dict[str, Any]) -> Iterable[list[Any]]:
    """Every list of paths a config reads files from: each agent's ``local_dir``
    skills and standing instructions. Yielded for rewriting in place."""
    for agent in raw.get("agents") or []:
        if not isinstance(agent, dict):
            continue
        for key in ("skills", "standing_instructions"):
            runtime = agent.get(key)
            source = runtime.get("source") if isinstance(runtime, dict) else None
            if isinstance(source, dict) and source.get("type") == "local_dir":
                paths = source.get("paths")
                if isinstance(paths, list):
                    yield paths


def in_project_terms(raw: dict[str, Any], folder: Path | None) -> dict[str, Any]:
    """``raw`` with every skills and instructions path inside ``folder`` written as
    ``${DOMYN_PROJECT_DIR}/...``: how a project names its own files, so the folder
    still works moved, renamed or exported. A copy; ``raw`` is not changed.

    Anything else is left as written — an absolute path elsewhere, and a relative one,
    which means the folder domyn runs in (the ``.domyn/skills`` and ``.agents/skills``
    autodiscovery entries).
    """
    written = copy.deepcopy(raw)
    if folder is None:
        return written
    root = folder.expanduser().absolute()
    for paths in _config_paths(written):
        for i, entry in enumerate(paths):
            path = Path(str(entry)).expanduser()
            if path.is_absolute() and path.absolute().is_relative_to(root):
                inside = path.absolute().relative_to(root).as_posix()
                paths[i] = PROJECT_DIR_REF if inside == "." else f"{PROJECT_DIR_REF}/{inside}"
    return written


def expand_project_dir(raw: dict[str, Any], folder: Path | None) -> dict[str, Any]:
    """The inverse of `in_project_terms`: ``${DOMYN_PROJECT_DIR}`` replaced with
    ``folder`` in the skills and instructions paths, and nothing else interpolated.

    For `domyn setup`, which compares the listed paths with the folders it installs
    to; every other reader interpolates the variable as the loader does.
    """
    expanded = copy.deepcopy(raw)
    if folder is None:
        return expanded
    for paths in _config_paths(expanded):
        for i, entry in enumerate(paths):
            if isinstance(entry, str) and entry.startswith(PROJECT_DIR_REF):
                paths[i] = str(folder) + entry[len(PROJECT_DIR_REF) :]
    return expanded


def config_path(*, prefer_project: bool = True) -> Path:
    return config_dir(prefer_project=prefer_project) / CONFIG_FILENAME


def credentials_path() -> Path:
    """``credentials`` in the active project's folder, else in ``~/.domyn``: the file
    setup writes to. Reading falls back to ``~/.domyn`` (`credential_files`)."""
    return home_dir() / CREDENTIALS_FILENAME


def credential_files() -> list[Path]:
    """The credentials files a key is read from, most specific first: the active
    project's, then ``~/.domyn``'s. A key the project does not set is the global one,
    so a key kept at home serves every project, and a project overrides it by setting
    its own."""
    files = [credentials_path()]
    global_file = GLOBAL_DIR / CREDENTIALS_FILENAME
    if global_file != files[0]:
        files.append(global_file)
    return files


def read_all_credentials() -> dict[str, str]:
    """Every key `credential_files` hold, the project's winning over the global ones."""
    values: dict[str, str] = {}
    for path in reversed(credential_files()):
        values.update(read_credentials(path))
    return values


_credentials_in_environment: dict[str, str] = {}
"""What `load_credentials_into_environment` put in ``os.environ``, to take back."""


def load_credentials_into_environment() -> None:
    """Expose the credentials files' entries (`read_all_credentials`) as environment
    variables.

    A variable already set wins (``setdefault``), so a real environment variable or a
    ``.env`` entry always beats the files. Called once at startup, before any option is
    parsed, and again once ``--project`` is known: what the first call put in from
    ``~/.domyn/credentials`` is taken back then, so a key the project sets replaces the
    global one. Only what this function itself set is removed, and only while it
    still holds the value it was given.
    """
    for name, value in _credentials_in_environment.items():
        if os.environ.get(name) == value:
            del os.environ[name]
    _credentials_in_environment.clear()
    for name, value in read_all_credentials().items():
        if name not in os.environ:
            os.environ[name] = value
            _credentials_in_environment[name] = value


def read_credentials(path: Path | None = None) -> dict[str, str]:
    """Read ``KEY=value`` pairs from a credentials file, the active one by default."""
    path = path or credentials_path()
    if not path.is_file():
        return {}
    values: dict[str, str] = {}
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, _, value = stripped.partition("=")
            values[key.strip()] = value.strip()
    except OSError as exc:
        get_logger().warning(f"cannot read credentials: {type(exc).__name__}")
    return values


def write_credential(name: str, value: str) -> Path:
    """Store one secret, with owner-only permissions.

    Kept out of the YAML on purpose (see module docstring). ``chmod`` is applied
    before writing so the secret is never briefly world-readable.
    """
    path = credentials_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    existing = read_credentials()
    existing[name] = value

    path.touch(mode=0o600, exist_ok=True)
    os.chmod(path, 0o600)
    path.write_text(
        "# domyn credentials — keep private\n"
        + "".join(f"{key}={val}\n" for key, val in sorted(existing.items())),
        encoding="utf-8",
    )
    return path


def resolve_api_key(provider: ProviderConfig) -> str | None:
    """Find the key: environment first, then the credentials files (`credential_files`).

    Environment wins so CI and containers can inject a key without touching disk.
    """
    name = provider.default_key_env()
    from_env = os.environ.get(name)
    if from_env:
        return from_env
    return read_all_credentials().get(name)


@dataclass(frozen=True)
class FoundKey:
    """An API key `find_api_key` found, and where."""

    value: str
    source: str
    """Where it was found, for the person reading setup's output."""
    path: Path | None = None
    """The credentials file it was read from; ``None`` for the environment or a ``.env``,
    which setup cannot change."""


def find_api_key(name: str) -> FoundKey | None:
    """Look for the variable ``name`` the way `domyn setup` reports it, most specific
    first: the environment (a ``.env`` in the folder domyn runs in included), then the
    active project's ``credentials``, then ``~/.domyn/credentials`` — the order a
    session reads them in (`credential_files`).
    """
    value = os.environ.get(name)
    # What the loader put in from a credentials file is not the environment's: the
    # files below are where it came from, and say so.
    if value and _credentials_in_environment.get(name) != value:
        dotenv = find_dotenv(usecwd=True)
        if dotenv and dotenv_values(dotenv).get(name) == value:
            return FoundKey(value, f"{name}, from {dotenv}")
        return FoundKey(value, f"{name}, in the environment")
    for path in credential_files():
        if value := read_credentials(path).get(name):
            return FoundKey(value, f"{name}, in {path}", path)
    return None


# ---------------------------------------------------------------------------
# ConfigurationV1: what the default config actually is
# ---------------------------------------------------------------------------
#
# ~/.domyn/config.yaml is a plain ConfigurationV1 file
# (`domyn.agents.loader.config`), loaded through the same
# `load_application_from_file` as any `.yaml` handed to `domyn run` explicitly.
# `ProviderConfig`/`AgentConfig`/`CompactionConfig` are not the persisted
# format — they're the wizard's own typed working state on its way to being
# projected into a `ConfigurationV1` dict by `configuration_v1_dict_from`.

CLI_DEFAULT_PROVIDER_NAME = "default_provider"
CLI_DEFAULT_TOOL_BUNDLE_NAME = "cli_tools"


CLI_DEFAULT_DECISION_PROVIDER_NAME = "decision_model"

#: What the wizard offers, and what each level switches on. The order is the
#: order of increasing consequence, which is also the order the prompt lists.
#:
#: `approvals` is the use case the measurements were taken on and the one a
#: reader expects first, yet it is the only level of its own: everywhere else a
#: wrong answer changes what the model is *shown*, and the turn carries on; at
#: the gate, allowing removes a human confirmation. Something that takes a
#: person out of the loop should be chosen on its own line rather than arrive
#: with a bundle, so "active" leaves the gate measuring and the level after it
#: is what hands the gate over.
DECISION_LEVELS: dict[str, dict[str, Any]] = {
    "no": {},
    "shadow": {
        "approvals": {"mode": "shadow"},
        "skills": {"mode": "shadow"},
        "instructions": {"mode": "shadow"},
    },
    "active": {
        "approvals": {"mode": "shadow"},
        "skills": {"mode": "active"},
        "instructions": {"mode": "active"},
    },
    "active, approvals too": {
        "approvals": {"mode": "active"},
        "skills": {"mode": "active"},
        "instructions": {"mode": "active"},
    },
}

#: Shown beside each level. Says what it does, not how good it is.
DECISION_LEVEL_BLURB: dict[str, str] = {
    "no": "no decision model",
    "shadow": "ask and record, change nothing",
    "active": "pick skills and instructions; the approval gate only records",
    "active, approvals too": "also let it allow or deny a tool call",
}

DEFAULT_DECISION_URL = "http://localhost:28080"
DEFAULT_DECISION_MODEL = "Cloudflare/clef"
DEFAULT_DECISION_KEY_ENV = "DECISIONS_API_KEY"


class DecisionsChoice(BaseModel):
    """The wizard's working state for a decision model.

    `level` is a key of `DECISION_LEVELS`; `"no"` means the file should carry
    no decision model at all, which is how the wizard turns one off again.
    """

    level: str = "no"
    base_url: str = DEFAULT_DECISION_URL
    model: str = DEFAULT_DECISION_MODEL
    #: Name of the variable holding the endpoint's key; the key itself is never
    #: in the choice, nor in the YAML (which gets ``${NAME}``).
    api_key_env: str | None = None

    @property
    def enabled(self) -> bool:
        return self.level != "no" and bool(self.base_url) and bool(self.model)


def current_decisions_from(raw: dict[str, Any] | None) -> DecisionsChoice:
    """What the file already says, as the wizard's own state.

    A file written by hand can say things the four levels cannot express — a
    use case the wizard never offers, a threshold someone tuned. That is read
    back as the closest level so the prompt opens on something true, and
    `configuration_v1_dict_from` is what decides whether to overwrite it.
    """
    if not raw:
        return DecisionsChoice()
    providers = [p for p in (raw.get("decision_providers") or []) if isinstance(p, dict)]
    if not providers:
        return DecisionsChoice()
    entry = providers[0]

    agent_dict, _ = _entrypoint_agent_and_provider(raw)
    declared = agent_dict.get("decisions")
    modes: dict[str, str] = {}
    if isinstance(declared, dict):
        for key, value in declared.items():
            if key != "provider" and isinstance(value, dict):
                modes[key] = str(value.get("mode", "off"))

    if modes.get("approvals") == "active":
        level = "active, approvals too"
    elif any(mode == "active" for mode in modes.values()):
        level = "active"
    elif any(mode == "shadow" for mode in modes.values()):
        level = "shadow"
    else:
        level = "no"

    return DecisionsChoice(
        level=level,
        base_url=str(entry.get("base_url") or DEFAULT_DECISION_URL),
        model=str(entry.get("model") or DEFAULT_DECISION_MODEL),
        api_key_env=_env_reference(entry.get("api_key")),
    )


def _env_reference(value: Any) -> str | None:
    """``NAME`` from a ``${NAME}`` reference, else ``None`` (absent or literal)."""
    match = re.fullmatch(r"\$\{(\w+)\}", value) if isinstance(value, str) else None
    return match.group(1) if match else None


def configuration_v1_dict_from(
    provider: ProviderConfig,
    agent: AgentConfig,
    compaction: CompactionConfig,
    existing: dict[str, Any] | None = None,
    decisions: DecisionsChoice | None = None,
) -> dict[str, Any]:
    """Project the CLI's prompted fields onto a ConfigurationV1-shaped dict.

    The one place this mapping is written, so the wizard's own round-trip
    (see `current_settings_from`, its reverse) can only ever agree with what
    it writes.

    ``existing`` is what the file said before, and **everything the wizard does
    not own is carried across from it**. Without that this function is a
    rewrite, not an edit: it emits exactly the keys it knows about, so a
    `/model` would silently delete anything hand-written beside them —
    `decision_providers`, `mcp_servers`, extra `tools`, a second agent. That is
    not a hypothetical; it was measured on a config with a decision model in it,
    and the file came back without one. Omitting ``existing`` keeps the old
    behaviour for callers that genuinely want a fresh file.
    """
    provider_dict: dict[str, Any] = {
        "name": CLI_DEFAULT_PROVIDER_NAME,
        "type": provider.type,
        "model_name": provider.model_name,
    }
    if provider.url:
        provider_dict["url"] = provider.url
    if provider.type == "domyn":
        if provider.space_id:
            provider_dict["space_id"] = provider.space_id
        if provider.configuration_id:
            provider_dict["configuration_id"] = provider.configuration_id
    # Written only when declared: left out, the provider is asked at session
    # start (`apply_context_window`), which a written default would prevent.
    if provider.context_window is not None:
        provider_dict["context_window"] = provider.context_window
    # Only set when a key actually resolves right now (env, or the credentials
    # file — injected into the environment at CLI startup by
    # `_load_credentials` in `cli/main.py`). Omitting it otherwise matters:
    # `${VAR_NAME}` interpolation (`domyn.agents.loader.load_from_file`) raises
    # immediately for an unset variable, and this file must still load — with
    # no key configured, an agent that fails at call time with an auth error
    # is recoverable; a CLI that cannot start at all is not. The cost is that
    # exporting the variable *after* saving needs another `domyn setup`/
    # `/model` to be picked up, unlike `resolve_api_key`, which re-checks
    # fresh every time it's called — an accepted, smaller trade-off.
    if resolve_api_key(provider):
        provider_dict["api_key"] = f"${{{provider.default_key_env()}}}"
    provider_dict["generation_params"] = {"max_tokens": provider.max_tokens}
    provider_dict["body"] = {"max_tokens": provider.max_tokens}

    agent_dict: dict[str, Any] = {
        "name": agent.name,
        "description": agent.description,
        "instruction": agent.instruction,
        "llm_provider": CLI_DEFAULT_PROVIDER_NAME,
        "planner_strategy": CLI_PLANNER,
        "stream": agent.stream,
    }

    if agent.skills:
        agent_dict["skills"] = agent.skills
    if agent.standing_instructions:
        agent_dict["standing_instructions"] = agent.standing_instructions

    config: dict[str, Any] = {
        "version": "1.0",
        "entrypoint": agent.name,
        "llm_providers": [provider_dict],
        "planner_strategies": [
            {
                "name": CLI_PLANNER,
                "type": CLI_PLANNER,
                "compactor": compaction.model_dump(exclude_none=True),
            }
        ],
        "agents": [agent_dict],
    }

    if agent.tools:
        config["tool_bundles"] = [
            {
                "name": CLI_DEFAULT_TOOL_BUNDLE_NAME,
                "bundles": list(agent.tools),
                "workspace": agent.workspace,
                "sandbox": agent.sandbox,
                "isolate_environment": agent.isolate_environment,
                "allowed_env": list(agent.allowed_env),
            }
        ]
        # Only what differs from the defaults, so the file says what was chosen.
        browser = agent.browser.model_dump(exclude_defaults=True)
        if browser:
            config["tool_bundles"][0]["browser"] = browser
        agent_dict["tool_bundles"] = [CLI_DEFAULT_TOOL_BUNDLE_NAME]

    kept = _keep_unowned(config, existing)
    return _apply_decisions(kept, decisions)


def _apply_decisions(config: dict[str, Any], decisions: DecisionsChoice | None) -> dict[str, Any]:
    """Write the decision model the wizard was told about, or leave it alone.

    ``None`` means the wizard never asked — a ``provider_only`` run, which is
    what ``/model`` is — and then this must not touch a thing. That is not a
    nicety: ``decision_providers`` is otherwise carried across by
    `_keep_unowned` precisely because the wizard does not own it, and owning it
    unconditionally would hand `/model` back the deletion that function exists
    to prevent.

    When it did ask, the answer is authoritative in both directions: a level
    writes the provider and the agent's block, and ``"no"`` removes both, or
    the wizard could turn a model on and never off again.
    """
    if decisions is None:
        return config

    agent_dict = config["agents"][0]
    if not decisions.enabled:
        config.pop("decision_providers", None)
        agent_dict.pop("decisions", None)
        return config

    provider: dict[str, Any] = {
        "name": CLI_DEFAULT_DECISION_PROVIDER_NAME,
        "type": "system_one",
        "base_url": decisions.base_url,
        "model": decisions.model,
    }
    if decisions.api_key_env:
        provider["api_key"] = "${" + decisions.api_key_env + "}"
    config["decision_providers"] = [provider]
    agent_dict["decisions"] = {
        "provider": CLI_DEFAULT_DECISION_PROVIDER_NAME,
        **{
            use_case: dict(settings)
            for use_case, settings in DECISION_LEVELS[decisions.level].items()
        },
    }
    return config


#: What the wizard builds from its own prompts and may therefore replace whole.
#: Everything else in the file belongs to whoever wrote it.
_WIZARD_OWNS: frozenset[str] = frozenset(
    {"version", "entrypoint", "llm_providers", "planner_strategies", "agents", "tool_bundles"}
)

#: Same, one level down, for the entrypoint agent the wizard rebuilds.
_WIZARD_OWNS_ON_AGENT: frozenset[str] = frozenset(
    {
        "name",
        "description",
        "instruction",
        "llm_provider",
        "planner_strategy",
        "stream",
        "skills",
        "standing_instructions",
        "tool_bundles",
    }
)


def _keep_unowned(config: dict[str, Any], existing: dict[str, Any] | None) -> dict[str, Any]:
    """Carry across everything the wizard did not write itself.

    Two levels, because the two losses look different to whoever suffers them:
    a dropped top-level key takes a whole facility with it, and a dropped key on
    the entrypoint agent takes the setting that pointed the agent at one.
    """
    if not existing:
        return config

    for key, value in existing.items():
        if key not in _WIZARD_OWNS:
            config.setdefault(key, value)

    previous_agents = existing.get("agents")
    if not isinstance(previous_agents, list) or not previous_agents:
        return config
    rebuilt = config["agents"][0]
    # The agent the wizard rebuilt is the previous entrypoint, found as the wizard
    # read it rather than by name: setup may have renamed it.
    previous_entrypoint, _ = _entrypoint_agent_and_provider(existing)
    for key, value in previous_entrypoint.items():
        if key not in _WIZARD_OWNS_ON_AGENT:
            rebuilt.setdefault(key, value)

    # Agents the wizard never asked about: a multi-agent file edited by hand is
    # still a file someone wants to keep.
    others = [
        other
        for other in previous_agents
        if isinstance(other, dict)
        and other is not previous_entrypoint
        and other.get("name") != rebuilt.get("name")
    ]
    config["agents"].extend(others)
    return config


def _entrypoint_agent_and_provider(raw: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """The entrypoint agent's raw dict, and its provider's raw dict.

    Shared lookup for `current_settings_from` and `describe_configuration_v1` —
    both need to find the same agent and provider inside the file, and having
    it in one place is what keeps them from finding two different ones.
    """
    agents = raw.get("agents") or []
    # Same fallback as the loader.
    # TODO: an unknown entrypoint falls back to agents[0] here; the loader raises.
    entrypoint_name = raw.get("entrypoint") or (agents[-1].get("name") if agents else None)
    agent_dict = next(
        (a for a in agents if a.get("name") == entrypoint_name),
        agents[0] if agents else {},
    )
    providers = {p.get("name"): p for p in raw.get("llm_providers", []) if isinstance(p, dict)}
    provider_dict = providers.get(agent_dict.get("llm_provider"), {})
    return agent_dict, provider_dict


def _browser_options(raw: Any) -> BrowserOptions:
    """The bundle's ``browser`` block, or the defaults when it cannot be read.

    Lenient on purpose, unlike the loader: this only pre-fills the wizard, and a
    wizard that cannot start is no way to fix the file. The loader still refuses
    the same block when the agent is built.
    """
    if not raw:
        return BrowserOptions()
    try:
        return BrowserOptions.model_validate(raw)
    except ValidationError as exc:
        get_logger().warning(
            f"ignoring an invalid browser block in the config: {type(exc).__name__}"
        )
        get_logger().debug(f"browser block detail: {exc}")
        return BrowserOptions()


def current_settings_from(
    raw: dict[str, Any] | None,
) -> tuple[ProviderConfig, AgentConfig, CompactionConfig]:
    """Pull the wizard's editable fields back out of a ConfigurationV1 dict.

    The reverse of `configuration_v1_dict_from`, used to pre-fill the wizard's
    prompts and to preserve whatever a ``provider_only`` run does not touch.
    ``max_tokens``/``context_window`` round-trip here for the same reason
    ``model_name``/``url`` do: the wizard prompts for them, so the prompt
    should show what was there before rather than the class default.
    """
    if not raw:
        return ProviderConfig(), AgentConfig(), CompactionConfig()

    agent_dict, provider_dict = _entrypoint_agent_and_provider(raw)
    max_tokens = (provider_dict.get("generation_params") or provider_dict.get("body") or {}).get(
        "max_tokens", ProviderConfig().max_tokens
    )
    provider = ProviderConfig(
        type=provider_dict.get("type", ProviderConfig().type),
        model_name=provider_dict.get("model_name", ""),
        url=provider_dict.get("url"),
        space_id=provider_dict.get("space_id"),
        configuration_id=provider_dict.get("configuration_id"),
        max_tokens=max_tokens,
        context_window=provider_dict.get("context_window"),
    )

    bundles = {b.get("name"): b for b in raw.get("tool_bundles", []) if isinstance(b, dict)}
    bundle_names = agent_dict.get("tool_bundles") or []
    bundle = bundles.get(bundle_names[0]) if bundle_names else None
    default_agent = AgentConfig()
    # A missing `bundles` key mounts what `BuiltinBundleConfig` defaults it to.
    tools = list(bundle.get("bundles", DEFAULT_BUNDLES) or []) if bundle else default_agent.tools
    agent = AgentConfig(
        name=agent_dict.get("name", default_agent.name),
        description=agent_dict.get("description", default_agent.description),
        instruction=agent_dict.get("instruction", default_agent.instruction),
        tools=tools,
        sandbox=bundle.get("sandbox", default_agent.sandbox) if bundle else default_agent.sandbox,
        isolate_environment=bundle.get("isolate_environment", default_agent.isolate_environment)
        if bundle
        else default_agent.isolate_environment,
        workspace=bundle.get("workspace", default_agent.workspace)
        if bundle
        else default_agent.workspace,
        # Read back because the wizard rewrites the whole file: a field read
        # here is a field every `domyn setup` and `/model` silently dropped.
        allowed_env=list(bundle.get("allowed_env") or []) if bundle else default_agent.allowed_env,
        browser=_browser_options(bundle.get("browser") if bundle else None),
        stream=agent_dict.get("stream", default_agent.stream),
        skills=agent_dict.get("skills", default_agent.skills),
        standing_instructions=agent_dict.get(
            "standing_instructions", default_agent.standing_instructions
        ),
    )

    planner_strategies = raw.get("planner_strategies") or []
    planner_dict = next(
        (p for p in planner_strategies if isinstance(p, dict) and p.get("name") == CLI_PLANNER),
        {},
    )
    compaction = CompactionConfig(**(planner_dict.get("compactor") or {}))
    return provider, agent, compaction


_URL_USERINFO_RE = re.compile(r"^(\w+://)[^/@]+@")


def _scrub_url(url: str | None) -> str | None:
    """Strip any ``user:pass@`` userinfo from a URL before it's displayed.

    A URL is otherwise safe and useful to show whole — it's the one field
    besides an outright secret where a credential can hide, since nothing
    stops a hand-written ``https://user:token@host`` from being valid.
    """
    if not url:
        return url
    return _URL_USERINFO_RE.sub(r"\1***@", url)


def _header_names(headers: Any) -> str:
    """Header *names* only — a header's value can be a bearer token."""
    if not isinstance(headers, dict) or not headers:
        return "(none)"
    return ", ".join(sorted(str(k) for k in headers))


def _secret_field_state(raw_value: Any) -> str:
    """Describe a secret-typed field's *state* without ever showing its value.

    ``${VAR_NAME}`` interpolation — the shape the wizard writes, and the
    documented convention for hand-written configs — reports the variable
    name and whether it currently resolves (environment, then the credentials
    file). A non-empty literal is flagged as hardcoded rather than echoed:
    that's precisely the case this function exists for. Absent/empty is
    reported as missing.
    """
    if isinstance(raw_value, str):
        stripped = raw_value.strip()
        if stripped.startswith("${") and stripped.endswith("}") and len(stripped) > 3:
            env_name = stripped[2:-1]
            resolved = bool(os.environ.get(env_name) or read_all_credentials().get(env_name))
            return f"{env_name} ({'set' if resolved else 'missing'})"
        if stripped:
            return "(hardcoded value — not shown)"
    return "(missing)"


def _skills_summary(skills: Any, *, absent: str) -> str:
    """An agent's ``skills:`` mapping as its source type, plus paths for ``local_dir``.

    Nothing else from the mapping is shown. A source's settings are open-ended —
    a third-party one can carry a token or headers — so, like everything else
    here, this shows structure and never values. Paths are shown because they
    are where the skills live, and ``/skills`` and the file tools expose them
    anyway.
    """
    if not isinstance(skills, dict) or not skills:
        return absent
    source = skills.get("source")
    if not isinstance(source, dict) or not source.get("type"):
        return "(unrecognised)"
    source_type = str(source["type"])
    if source_type != LocalDirSkillSource.model_fields["type"].default:
        return source_type
    paths = ", ".join(str(p) for p in source.get("paths") or [])
    return f"{source_type}: {paths}" if paths else f"{source_type} (no paths)"


@dataclass
class ConfigItem:
    """One named element in a ``Configuration`` section — a provider, an
    agent, … — printed as its name, then its fields indented under it.

    ``name == ""`` is the one exception, used by the metadata section (which
    has no natural per-element name): its fields print directly under the
    section title instead of under an empty name line.
    """

    name: str
    fields: list[tuple[str, str]]


@dataclass
class ConfigSection:
    """One named collection under ``Configuration`` — e.g. all providers."""

    title: str
    items: list[ConfigItem]


@dataclass
class ConfigDescription:
    """Everything ``/config`` and ``domyn setup --show`` render.

    ``path`` is the file described. ``sections`` is a full inventory of it
    (providers, agents, tools, hooks, MCP servers, OAuth providers, tool
    bundles, context managers, and any other ``metadata`` keys), each field
    drawn from an explicit allowlist per section — nothing here walks
    unknown structure, and nothing not on a section's list is ever shown.
    Never includes a secret value — only whether one is set, and where it
    comes from.
    """

    path: Path
    sections: list[ConfigSection]


def _provider_items(raw: dict[str, Any]) -> list[ConfigItem]:
    items: list[ConfigItem] = []
    for p in raw.get("llm_providers") or []:
        if not isinstance(p, dict):
            continue
        max_tokens = (p.get("generation_params") or p.get("body") or {}).get("max_tokens")
        fields: list[tuple[str, str]] = [
            ("type", str(p.get("type", "?"))),
            ("model", p.get("model_name") or "(not set)"),
            ("url", _scrub_url(p.get("url")) or "(provider default)"),
            (
                "context window",
                str(window) if (window := p.get("context_window")) else AUTO_CONTEXT_WINDOW,
            ),
            ("max tokens", str(max_tokens) if max_tokens is not None else "(default)"),
            ("api key", _secret_field_state(p.get("api_key"))),
        ]
        if "headers" in p:
            fields.append(("headers", _header_names(p.get("headers"))))
        items.append(ConfigItem(name=str(p.get("name", "?")), fields=fields))
    return items


def _mcp_server_items(entries: Iterable[Any]) -> list[ConfigItem]:
    items: list[ConfigItem] = []
    for m in entries:
        if not isinstance(m, dict):
            continue
        fields = [
            ("url", _scrub_url(m.get("url")) or "(not set)"),
            ("oauth_provider", m.get("oauth_provider") or "(none)"),
            ("tool_autodiscovery", str(m.get("tool_autodiscovery", True))),
            ("headers", _header_names(m.get("headers"))),
        ]
        items.append(ConfigItem(name=str(m.get("name", "?")), fields=fields))
    return items


def _bundle_entries(raw: dict[str, Any]) -> dict[Any, dict[str, Any]]:
    """The top-level ``tool_bundles:`` entries, by name."""
    return {b.get("name"): b for b in raw.get("tool_bundles") or [] if isinstance(b, dict)}


def _mcp_server_names(agent: dict[str, Any]) -> list[str]:
    """The agent's MCP servers by name, in declaration order.

    An entry is either a name referring to the top-level ``mcp_servers:`` or an
    inline server definition; the loader accepts both, and reading only the
    second form left an agent that references its servers by name looking as if
    it had none.
    """
    names: list[str] = []
    for m in agent.get("mcp_servers") or []:
        if isinstance(m, str):
            names.append(m)
        elif isinstance(m, dict):
            names.append(str(m.get("name", "?")))
    return names


def _agent_tool_names(
    agent: dict[str, Any], bundle_entries: dict[Any, dict[str, Any]]
) -> list[str]:
    """The agent's own tools by name, then what its bundles mount, each tagged
    with the built-in bundle it comes from (``read_file(files)``).

    Tolerant like `current_settings_from`: a bundle entry that does not exist,
    has no ``bundles`` key, or names a bundle that is not built in contributes
    nothing, as `bundle()` skips an unknown name rather than failing.
    """
    names = [str(t) for t in agent.get("tools") or []]
    for entry_name in agent.get("tool_bundles") or []:
        entry = bundle_entries.get(entry_name) or {}
        for builtin in entry.get("bundles") or []:
            names.extend(f"{tool}({builtin})" for tool in BUNDLE_TOOL_NAMES.get(builtin, ()))
    return list(dict.fromkeys(names))


def _agent_bundle_names(
    agent: dict[str, Any], bundle_entries: dict[Any, dict[str, Any]]
) -> list[str]:
    """The built-in bundles the agent mounts, each tagged with the
    ``tool_bundles`` entry that groups it (``files(cli_tools)``).

    Same tolerance as `_agent_tool_names` for an entry that is missing or has no
    ``bundles``. A built-in name that does not exist is still shown: it is what
    the file declares, and seeing it is how a typo gets noticed.
    """
    names: list[str] = []
    for entry_name in agent.get("tool_bundles") or []:
        entry = bundle_entries.get(entry_name) or {}
        names.extend(f"{builtin}({entry_name})" for builtin in entry.get("bundles") or [])
    return list(dict.fromkeys(names))


def _agent_items(raw: dict[str, Any]) -> list[ConfigItem]:
    """Rows for ``agents:``.

    An agent's ``mcp_servers:`` — names referring to the top-level list, or inline
    definitions — are named here only — same treatment as
    ``tools``/``hooks``/``tool_bundles`` — with the full definition living in
    the top-level MCP Servers section instead (`describe_configuration_v1`
    feeds it every agent's servers alongside the top-level ones). That
    mirrors how ``llm_provider`` and ``planner_strategy`` already work: a
    name here, the definition in that thing's own section.

    ``tools`` lists the agent's own tools by name, then every tool its bundles
    mount as ``read_file(files)`` — a bare ``tools  (none)`` beside a bundle
    name reads as an agent with no tools. ``tool_bundles`` lists the built-in
    bundles with the entry that groups them, ``files(cli_tools)``; the entry's
    definition is in the Tool Bundles section.

    ``skills`` has no section of its own — it is declared inline on the agent
    — so it is summarized here, as `_skills_summary` describes.
    """
    bundle_entries = _bundle_entries(raw)
    items: list[ConfigItem] = []
    for a in raw.get("agents") or []:
        if not isinstance(a, dict):
            continue
        mcp_names = ", ".join(_mcp_server_names(a))
        tool_names = _agent_tool_names(a, bundle_entries)
        bundle_names = _agent_bundle_names(a, bundle_entries)
        fields = [
            ("description", a.get("description") or "(none)"),
            ("instruction", a.get("instruction") or "(none)"),
            ("llm_provider", a.get("llm_provider") or "(none)"),
            ("planner_strategy", a.get("planner_strategy") or "(none)"),
            ("tools", ", ".join(tool_names) or "(none)"),
            ("tool_bundles", ", ".join(bundle_names) or "(none)"),
            ("hooks", ", ".join(a.get("hooks") or []) or "(none)"),
            ("mcp_servers", mcp_names or "(none)"),
            ("skills", _skills_summary(a.get("skills"), absent="(none)")),
            (
                "standing_instructions",
                _skills_summary(a.get("standing_instructions"), absent="(none)"),
            ),
            ("stream", str(a.get("stream", True))),
        ]
        items.append(ConfigItem(name=str(a.get("name", "?")), fields=fields))
    return items


def _tool_def_items(raw: dict[str, Any]) -> list[ConfigItem]:
    """Items for the top-level ``tools:`` list.

    Not every tool shape identifies itself by ``name``: `MCPTool`
    (`tools/mcp/mcp_tool.py`) and `DelegateTool` (`tools/delegate_tool.py`)
    have no ``name`` field at all — their raw YAML key is ``tool_name``.
    `BaseTool` itself defines neither, so this has to branch on `type`
    rather than assume one shape for all of them.
    """
    items: list[ConfigItem] = []
    for t in raw.get("tools") or []:
        if not isinstance(t, dict):
            continue
        tool_type = str(t.get("type", "function"))
        if tool_type == "mcp":
            name = str(t.get("display_name") or t.get("tool_name", "?"))
            fields: list[tuple[str, str]] = [
                ("type", tool_type),
                ("tool_name", str(t.get("tool_name", "?"))),
                ("server_url", _scrub_url(t.get("server_url")) or "(not set)"),
                ("use_oauth", str(t.get("use_oauth", False))),
            ]
            if t.get("oauth_provider"):
                fields.append(("oauth_provider", str(t.get("oauth_provider"))))
            if "headers" in t:
                fields.append(("headers", _header_names(t.get("headers"))))
            items.append(ConfigItem(name=name, fields=fields))
            continue
        if tool_type == "delegate":
            items.append(
                ConfigItem(
                    name=str(t.get("tool_name", "?")),
                    fields=[("type", tool_type), ("tool_name", str(t.get("tool_name", "?")))],
                )
            )
            continue
        fields = [("type", tool_type)]
        if t.get("path"):
            fields.append(("path", str(t.get("path"))))
        items.append(ConfigItem(name=str(t.get("name", "?")), fields=fields))
    return items


def _hook_items(raw: dict[str, Any]) -> list[ConfigItem]:
    items: list[ConfigItem] = []
    for h in raw.get("hooks") or []:
        if not isinstance(h, dict):
            continue
        fields = [
            ("trigger", str(h.get("trigger", "?"))),
            ("source", str(h.get("path") or h.get("reference") or "?")),
        ]
        items.append(ConfigItem(name=str(h.get("name", "?")), fields=fields))
    return items


def _oauth_provider_items(raw: dict[str, Any]) -> list[ConfigItem]:
    items: list[ConfigItem] = []
    for o in raw.get("oauth_providers") or []:
        if not isinstance(o, dict):
            continue
        fields = [
            ("type", str(o.get("type", "local"))),
            ("client_id", o.get("client_id") or "(none)"),
            ("client_secret", _secret_field_state(o.get("client_secret"))),
            ("scopes", ", ".join(o.get("scopes") or []) or "(none)"),
            ("callback_port", str(o["callback_port"]) if "callback_port" in o else "(default)"),
        ]
        items.append(ConfigItem(name=str(o.get("name", "?")), fields=fields))
    return items


def _browser_fields(options: BrowserOptions) -> list[tuple[str, str]]:
    return [
        ("browser engine", options.engine),
        # Never the URL itself: a CDP endpoint usually carries its token.
        ("browser cdp", "set" if options.cdp_url else "none"),
        ("browser allowed_domains", ", ".join(options.allowed_domains) or "(any public site)"),
        ("browser allow_private_urls", str(options.allow_private_urls)),
        ("browser idle_timeout_s", str(options.idle_timeout_s)),
    ]


def _decision_items(raw: dict[str, Any]) -> list[ConfigItem]:
    """The decision providers, and where each agent lets one matter.

    Worth its own section rather than a line on the agent: a reader asking
    "why did it not ask me" needs the mode and the threshold together, and the
    threshold is the thing that decides, so hiding it would make the section
    decorative. Secrets follow the rule everywhere else in here — present or
    absent, never the value.
    """
    items: list[ConfigItem] = []
    for entry in raw.get("decision_providers") or []:
        if not isinstance(entry, dict):
            continue
        fields = [
            ("type", str(entry.get("type", "none"))),
            ("model", str(entry.get("model", "(none)"))),
            ("url", str(entry.get("base_url", "(none)"))),
        ]
        if entry.get("api_key") or entry.get("api_key_env"):
            fields.append(("api key", "set"))
        if entry.get("max_state_tokens"):
            fields.append(("max state", f"{entry['max_state_tokens']} tokens"))
        items.append(ConfigItem(name=str(entry.get("name", "?")), fields=fields))

    for agent in raw.get("agents") or []:
        if not isinstance(agent, dict):
            continue
        declared = agent.get("decisions")
        if not isinstance(declared, dict):
            continue
        fields = [("provider", str(declared.get("provider", "(none)")))]
        for key, value in declared.items():
            if key == "provider":
                continue
            settings = value if isinstance(value, dict) else {}
            mode = str(settings.get("mode", "off"))
            threshold = settings.get("min_confidence")
            shown = mode if threshold is None else f"{mode} (>= {threshold})"
            if settings.get("on_unknown"):
                shown = f"{shown}, on unknown: {settings['on_unknown']}"
            fields.append((key, shown))
        items.append(ConfigItem(name=f"{agent.get('name', '?')} uses", fields=fields))
    return items


def _tool_bundle_items(raw: dict[str, Any]) -> list[ConfigItem]:
    items: list[ConfigItem] = []
    for b in raw.get("tool_bundles") or []:
        if not isinstance(b, dict):
            continue
        fields = [
            ("bundles", ", ".join(b.get("bundles") or []) or "(none)"),
            ("sandbox", str(b.get("sandbox", "local"))),
            ("workspace", b.get("workspace") or "(cwd)"),
            ("isolate_environment", str(b.get("isolate_environment", True))),
            ("allowed_env", ", ".join(b.get("allowed_env") or []) or "(none)"),
        ]
        if "browser" in (b.get("bundles") or []):
            fields += _browser_fields(_browser_options(b.get("browser")))
        items.append(ConfigItem(name=str(b.get("name", "?")), fields=fields))
    return items


_CONTEXT_MANAGER_TUNABLES = (
    "min_events_for_summary",
    "message_history_length",
    "max_tool_result_chars",
)


def _context_manager_items(raw: dict[str, Any]) -> list[ConfigItem]:
    items: list[ConfigItem] = []
    for c in raw.get("context_manager") or []:
        if not isinstance(c, dict):
            continue
        fields = [("type", str(c.get("type", "?")))]
        for tunable in _CONTEXT_MANAGER_TUNABLES:
            if tunable in c:
                fields.append((tunable, str(c[tunable])))
        items.append(ConfigItem(name=str(c.get("name") or c.get("type", "?")), fields=fields))
    return items


def _planner_strategy_items(raw: dict[str, Any]) -> list[ConfigItem]:
    items: list[ConfigItem] = []
    for ps in raw.get("planner_strategies") or []:
        if not isinstance(ps, dict):
            continue
        fields = [("type", str(ps.get("type", "?")))]
        compactor = ps.get("compactor")
        if isinstance(compactor, dict):
            fields.extend((f"compactor {k}", str(v)) for k, v in compactor.items())
        items.append(ConfigItem(name=str(ps.get("name") or ps.get("type", "?")), fields=fields))
    return items


def _metadata_items(raw: dict[str, Any]) -> list[ConfigItem]:
    """Every key under ``metadata``, by name only.

    ``metadata`` is a genuinely open ``dict[str, Any]`` — only a key's *name*
    is safe to assume isn't a secret, so that's all this ever shows. There's
    no natural per-key element here, so this is the one section with a
    single ``name=""`` item rather than one item per entry.
    """
    metadata = raw.get("metadata") or {}
    if not isinstance(metadata, dict):
        return []
    keys = sorted(str(k) for k in metadata)
    if not keys:
        return []
    return [ConfigItem(name="", fields=[("keys", ", ".join(keys))])]


def describe_configuration_v1(
    raw: dict[str, Any], *, path: Path | None = None
) -> ConfigDescription:
    """Describe the whole config file as a sectioned inventory.
    See `ConfigDescription`.
    """
    all_mcp_servers: list[Any] = list(raw.get("mcp_servers") or [])
    for a in raw.get("agents") or []:
        if isinstance(a, dict):
            all_mcp_servers.extend(a.get("mcp_servers") or [])

    sections = [
        ConfigSection("Providers", _provider_items(raw)),
        ConfigSection("Agents", _agent_items(raw)),
        ConfigSection("Planner Strategies", _planner_strategy_items(raw)),
        ConfigSection("Tools", _tool_def_items(raw)),
        ConfigSection("Hooks", _hook_items(raw)),
        ConfigSection("MCP Servers", _mcp_server_items(all_mcp_servers)),
        ConfigSection("OAuth Providers", _oauth_provider_items(raw)),
        ConfigSection("Tool Bundles", _tool_bundle_items(raw)),
        ConfigSection("Context Managers", _context_manager_items(raw)),
        ConfigSection("Decision Models", _decision_items(raw)),
        ConfigSection("Metadata", _metadata_items(raw)),
    ]
    return ConfigDescription(path=path or config_path(), sections=[s for s in sections if s.items])


def print_config_description(
    console: Console, theme: Theme, description: ConfigDescription
) -> None:
    """Render a `ConfigDescription`: the config file's path, then
    ``Configuration`` broken into named, indented sections.

    Column alignment is computed per element's own fields rather than once across the
    whole output — a long field name on one element shouldn't push every other
    element's short fields out to match.
    """
    style = theme.style("meta")

    def line(text: str, *, indent: int) -> None:
        console.print(f"{'  ' * indent}{text}", style=style, highlight=False)

    def fields(rows: list[tuple[str, str]], *, indent: int) -> None:
        width = max((len(label) for label, _ in rows), default=0)
        for label, value in rows:
            line(f"{label.ljust(width)}   {value}", indent=indent)

    console.print()
    # soft_wrap: a deep path wrapped mid-string is unreadable, and there's
    # nothing gained from breaking it to fit an arbitrary terminal width.
    console.print(f"config   {description.path}", style=style, highlight=False, soft_wrap=True)

    if not description.sections:
        return
    console.print()
    line("Configuration", indent=0)
    for section in description.sections:
        console.print()
        line(section.title, indent=1)
        for item in section.items:
            if item.name:
                line(item.name, indent=2)
                fields(item.fields, indent=3)
            else:
                fields(item.fields, indent=2)


def read_raw_config(path: Path | None = None) -> dict[str, Any] | None:
    """Read the config file as a plain dict, or ``None`` when absent/unreadable.

    A corrupt config is reported and treated as absent rather than crashing the
    CLI: the user can then re-run setup instead of hand-editing YAML to escape.
    """
    target = path or config_path()
    if not target.is_file():
        return None
    try:
        raw = yaml.safe_load(target.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        get_logger().warning(f"ignoring unreadable config at {target}: {type(exc).__name__}")
        get_logger().debug(f"config parse detail: {exc}")
        return None
    return raw if isinstance(raw, dict) else None


def is_configuration_v1_usable(raw: dict[str, Any]) -> bool:
    """Whether the entrypoint agent's provider looks complete enough to run.

    Deliberately checked on the raw dict, before ``${VAR_NAME}`` interpolation
    (``domyn.agents.loader.load_from_file``) or the loader itself ever runs —
    a missing *model name* should re-open the wizard the same friendly way it
    always has, not surface as a stack trace from deep inside config loading.

    Unlike the loader (which reorders ``agents`` by dependency before falling
    back to the last one), this runs on the raw, unordered list — so without
    an explicit ``entrypoint`` the last agent is only a safe stand-in when
    it's the *only* agent. With more than one and no explicit ``entrypoint``,
    which one is the root is genuinely unknown here; reporting the config
    unusable is the safe answer, not guessing.
    """
    agents = raw.get("agents") or []
    if not agents:
        return False
    entrypoint_name = raw.get("entrypoint")
    if not entrypoint_name:
        if len(agents) != 1:
            return False
        entrypoint_name = agents[-1].get("name")
    provider_ref = next(
        (a.get("llm_provider") for a in agents if a.get("name") == entrypoint_name), None
    )
    providers = {p.get("name"): p for p in raw.get("llm_providers", []) if isinstance(p, dict)}
    provider = providers.get(provider_ref)
    if provider is None or not provider.get("model_name"):
        return False
    if provider.get("type") in REQUIRES_URL and not provider.get("url"):
        return False
    # The Domyn platform serves several spaces; the model is the one of a space's configuration.
    # DOMYN_SPACE_ID stands in for a missing one, as it does when the provider is built.
    return not (
        provider.get("type") == "domyn"
        and not (provider.get("space_id") or os.environ.get("DOMYN_SPACE_ID"))
    )


APPROVALS_FILENAME = "approvals.yaml"


def approvals_paths() -> dict[str, Path]:
    """Where each persisted scope keeps its grants, by scope name.

    Never inside the repo: a repo must not ship its own approvals. Inside the active
    ``--project`` folder when there is one, laid out as in ``~/.domyn``: a grant made
    under one configuration does not carry over to another.
    """
    return {
        "user": home_dir() / APPROVALS_FILENAME,
        "project": _project_approvals_path(),
    }


def _project_approvals_path() -> Path:
    """``projects/<name>-<hash of the full path>/``: the hash keeps directories
    with the same name apart; the name is only for humans."""
    cwd = Path.cwd().resolve()
    slug = re.sub(r"[^A-Za-z0-9._-]", "_", cwd.name) or "root"
    digest = hashlib.sha256(str(cwd).encode()).hexdigest()[:12]
    return home_dir() / PROJECTS_DIRNAME / f"{slug}-{digest}" / APPROVALS_FILENAME


def _launched_from_home() -> bool:
    """Whether launched from ``$HOME``, which is not a project."""
    try:
        return Path.cwd().resolve() == Path.home().resolve()
    except (OSError, RuntimeError):
        return False


def build_approval_policy() -> ApprovalPolicy:
    """The CLI's approval policy: this conversation, this project, or always.

    The disk-backed stores are wired *here* and not in the runner. The runner
    defaults to conversation-scoped grants only, so importing the library never
    starts writing permissions under someone's home directory — that is a choice
    the person running an interactive tool makes, not one an import makes for them.

    Neither file is created until something is granted.
    """
    paths = approvals_paths()
    stores: list[Any] = [ConversationApprovalStore()]
    if not _launched_from_home():
        stores.append(FileApprovalStore(paths["project"], ApprovalScope.PROJECT))
    stores.append(FileApprovalStore(paths["user"], ApprovalScope.USER))
    return ApprovalPolicy(stores=stores)


def collect_skill_workspace_roots(raw: dict[str, Any]) -> list[Path]:
    """Every skill directory declared in the config, in agent order.

    As written, with ``${VAR}`` filled in the same way the loader does: the file
    tools and the skill index have to agree on the folder, and the loader builds
    the index from the filled-in paths. Filled in path by path rather than on the
    whole ``raw`` dict, so an unset variable elsewhere in the config (an MCP header,
    say) does not touch this scan.

    Only scans the config — deduplication is `unique_paths`'s job, applied by the
    caller.

    Scans **every** agent's ``skills.source``, not only the entrypoint's:
    ``extra_workspace_roots`` widens every tool bundle alike, and a sub-agent's
    skills must be as openable as the root's. Only ``local_dir`` sources have
    directories; any other type contributes nothing.

    A path that goes through a symlink (`listed_path_is_linked`) is left out, as the
    loader leaves it out.

    Malformed shapes are skipped rather than raised on — reporting a bad config
    is the loader's job, not this scan's.
    """
    local_dir_type = LocalDirSkillSource.model_fields["type"].default
    configured: list[Path] = []

    for agent_dict in raw.get("agents") or []:
        if not isinstance(agent_dict, dict):
            continue
        skills = agent_dict.get("skills")
        source = skills.get("source") if isinstance(skills, dict) else None
        if isinstance(source, dict) and source.get("type") == local_dir_type:
            for path in source.get("paths") or []:
                try:
                    listed = Path(interpolate_env_vars(path, "the configuration"))
                except ValueError:
                    # An unset variable: the loader raises and reports it moments later.
                    continue
                # The loader follows no symlink (`_iter_skill_files`), so neither do the
                # file tools: `Workspace.at` would resolve `.agents -> ~` to the home folder.
                if not listed_path_is_linked(listed.expanduser()):
                    configured.append(listed)

    return configured


def skill_workspace_roots(raw: dict[str, Any]) -> list[Path]:
    """What the file tools may reach for the skills this config lists.

    `collect_skill_workspace_roots` plus, when any listed skill sits inside
    `skills_install_dir()`, that whole directory: the skills `domyn setup` installs
    share files (``resources/``, ``data/*.py``) that live in the folders above
    them, which a skill's own folder does not cover. The install directory holds
    nothing but the chosen skills and what they share, so this is little wider than
    what the listed folders already reach, and writing still needs approval.

    For the file tools only. What gets *loaded* is `collect_skill_workspace_roots`:
    the install directory is never scanned.
    """
    listed = collect_skill_workspace_roots(raw)
    install = skills_install_dir().expanduser().absolute()
    if any(install in path.expanduser().absolute().parents for path in listed):
        return [*listed, install]
    return listed


LOCAL_SKILLS_DIRS: tuple[Path, ...] = (Path(".domyn/skills"), Path(".agents/skills"))
"""The project's own skills folders, relative to where the CLI is started.

Autodiscovery: when `domyn setup` is told to, it writes these paths, as written and in
this order, into the agent's skills paths. The order is the precedence: domyn's own
folder wins a name clash with the one other tools share. Being relative, they are read
every turn from the folder ``domyn run`` starts in, whichever that is, group folders
included (`skills.layout`).

``.domyn`` is `PROJECT_DIR`'s name, spelled out rather than derived from it: the entry
has to stay relative, and tests point `PROJECT_DIR` at absolute folders.
"""


def is_skill_autodiscovery(path: str | Path) -> bool:
    """Whether a listed skills path is one of the entries setup writes for autodiscovery.

    ``Path`` makes ``.agents/skills``, ``.agents/skills/`` and ``./.agents/skills`` one
    value. A folder missing from where the CLI was started is normal for these entries,
    not a stale path.
    """
    return Path(str(path)) in LOCAL_SKILLS_DIRS


def is_instruction_autodiscovery(path: str | Path) -> bool:
    """Whether a listed instructions path is one of the entries setup writes for
    autodiscovery. An exact match, not containment: a file the user's folder holds
    under ``./.domyn`` is the user's entry."""
    return Path(str(path)) in AUTODISCOVERY_DIRS


def skills_install_dir() -> Path:
    """Where ``domyn setup`` installs the builtin skills, and replaces them on every run.

    Owned by setup, so nothing else belongs in it. Read only through the paths the
    config lists, never scanned. A project has its own, inside its folder.
    """
    return home_dir() / BUILTIN_SKILLS_DIRNAME


def builtin_skill_dirs() -> list[Path]:
    """The category directories under ``skills/builtin``, inside the package.

    Not a place skills are loaded from: setup copies the chosen ones out
    (`skills.catalog.install_skills`). Kept for what reads the shipped tree itself.

    One entry per category rather than the root itself: the source walks group
    folders now (`skills.layout`), but the root also holds shared files and the
    categories are what the callers iterate over.
    """
    if not BUILTIN_SKILLS_ROOT.is_dir():
        return []
    return sorted(d for d in BUILTIN_SKILLS_ROOT.iterdir() if d.is_dir())


def resolve_instruction_dirs(declared: list[Path] | None) -> list[Path]:
    """The paths the root agent reads ``SOUL.md`` / ``AGENTS.md`` from: what the agent
    declares, and nothing else.

    An autodiscovery entry that lands in `GLOBAL_DIR` is dropped: started from
    ``$HOME``, ``./.domyn`` *is* ``~/.domyn``, which is not an instructions folder.
    """
    home = GLOBAL_DIR.expanduser().resolve()
    return unique_paths(
        path
        for path in declared or []
        if not (is_instruction_autodiscovery(path) and path.expanduser().resolve() == home)
    )


def unique_paths(paths: Iterable[Path]) -> list[Path]:
    """``paths`` without the ones that resolve to an earlier entry, as written.

    The first is kept, so the order stays the precedence: for skill directories, an
    earlier one wins a name clash in `LocalDirSkillSource`.
    """
    unique: dict[Path, Path] = {}
    for d in paths:
        resolved_path = d.expanduser().resolve()
        if resolved_path not in unique:
            unique[resolved_path] = d
    return list(unique.values())
