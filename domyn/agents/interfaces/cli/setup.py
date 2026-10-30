"""Interactive configuration wizard.

Reachable as ``domyn setup`` and as ``/model`` / ``/config`` inside a session, so
settings can be changed without leaving the conversation — the thing that makes a
CLI feel usable rather than merely functional.

On a re-run each section — model, instructions, tools, skills — first shows what the
config holds and asks whether to keep it (`_keep`); a yes skips its questions, so
changing one section costs an Enter for each of the others. Every prompt shows the
current value and accepts empty input to keep it. The one exception is
the instructions path, where empty removes it (`_choose_instructions`); the question
after it turns ``./.domyn`` / ``./.agents`` instructions autodiscovery on or off
(`_choose_local_instructions`).

On a terminal, Ctrl-B goes back to the previous question (`run_setup`, `_Recall`).
"""

from __future__ import annotations

import copy
import os
import shutil
from collections.abc import Callable, Collection, Iterable
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, TYPE_CHECKING, Any, cast

import yaml

from domyn.agents.agents.agent import MAX_AGENT_NAME_LENGTH
from domyn.agents.instructions.discovery import AUTODISCOVERY_DIRS, find_instruction_files
from domyn.agents.integrations.domyn_platform_sdk import (
    SDK_REQUIREMENT,
    PlatformSDKNotInstalledError,
    PlatformSettings,
    list_models,
    run_sync,
    sdk_installed,
)
from domyn.agents.interfaces.cli.config import (
    AUTO_CONTEXT_WINDOW,
    DECISION_LEVEL_BLURB,
    DECISION_LEVELS,
    DEFAULT_DECISION_KEY_ENV,
    DEFAULT_DECISION_MODEL,
    DEFAULT_DECISION_URL,
    LOCAL_SKILLS_DIRS,
    PROVIDER_HINTS,
    REQUIRES_URL,
    AgentConfig,
    DecisionsChoice,
    ProviderConfig,
    _scrub_url,
    active_project,
    config_path,
    configuration_v1_dict_from,
    credentials_path,
    current_decisions_from,
    current_settings_from,
    describe_configuration_v1,
    describe_project,
    expand_project_dir,
    find_api_key,
    in_project_terms,
    is_configuration_v1_usable,
    is_instruction_autodiscovery,
    is_skill_autodiscovery,
    load_credentials_into_environment,
    print_config_description,
    read_credentials,
    resolve_api_key,
    skills_install_dir,
    vllm_chat_url,
    write_credential,
)
from domyn.agents.interfaces.cli.prompt import GoBack, back_key, interactive_terminal, read_line
from domyn.agents.interfaces.cli.select import multiselect, select
from domyn.agents.skills.catalog import (
    BuiltinSkill,
    SkillInstallError,
    add_skills,
    install_skills,
    list_builtin_skills,
    skill_groups,
    unrecognized_entries,
)
from domyn.agents.skills.local import LocalSkill, discover_local_skills
from domyn.agents.tools.builtin import BUNDLE_DESCRIPTIONS, BUNDLES
from domyn.agents.tools.builtin.browser_install import resolve_binary
from domyn.agents.tools.builtin.browser_sessions import INSTALL_HINT

if TYPE_CHECKING:
    from rich.console import Console

    from domyn.agents.interfaces.cli.config import AgentConfig
    from domyn.agents.interfaces.cli.project_import import ProjectImport
    from domyn.agents.interfaces.cli.theme import Theme


@dataclass
class _Recall:
    """The answers given so far in one wizard run, by step, kind of question and label.

    Going back restores the draft from before the step (`run_setup`), so a question
    asked again would open on the config's old value; recalled, it opens on the
    answer given there, which is what someone going back to fix a typo wants to see.
    Secrets are not recalled.

    Only answers of an earlier run of the step are recalled (`settle`): a question
    asked again within one run, after a refused answer, opens as it did, so an empty
    answer still means what its label says.

    ``asked`` counts the questions, so `run_setup` can tell a step that asked none
    (a key already set): going back passes over it, or Ctrl-B would land on it and
    come straight forward again.
    """

    step: str = ""
    answers: dict[tuple[str, str, str], Any] = field(default_factory=dict)
    given: dict[tuple[str, str, str], Any] = field(default_factory=dict)
    asked: int = 0

    def opening(self, kind: str, label: str, current: Any) -> Any:
        """A question is being asked: what it opens with."""
        self.asked += 1
        return self.answers.get((self.step, kind, label), current)

    def record(self, kind: str, label: str, answer: Any) -> None:
        self.given[(self.step, kind, label)] = answer

    def settle(self) -> None:
        """End a run of the step: what it was given is recalled from now on."""
        self.answers.update(self.given)
        self.given = {}

    def forget(self, steps: Collection[str]) -> None:
        self.answers = {key: value for key, value in self.answers.items() if key[0] not in steps}


# Set by `run_setup` while it asks: a step called on its own, as tests do, recalls nothing.
_recall: ContextVar[_Recall | None] = ContextVar("setup_recall", default=None)


class _Screen:
    """The terminal under the wizard, counting the rows printed on it, so going back can
    erase up to where a question started instead of asking it again below.

    A row is a newline written: rich wraps lines itself, and inside ``back_key`` a
    question erases itself when it ends and an answered one is printed back through the
    console (`prompt.read_line`, `select`), so nothing else is left on screen. A row
    that has scrolled off the top cannot be erased: the cursor stops at the top, and the
    rest stays in the scrollback.
    """

    def __init__(self, file: IO[str]) -> None:
        self.file = file
        self.rows = 0

    def write(self, text: str) -> int:
        self.rows += text.count("\n")
        return self.file.write(text)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.file, name)

    def erase_to(self, mark: int) -> None:
        """Erase every row printed since ``mark``, leaving the cursor where it began."""
        if self.rows > mark:
            # Up that many rows, back to the first column, and clear to the end of the screen.
            self.file.write(f"\x1b[{self.rows - mark}A\r\x1b[J")
            self.file.flush()
        self.rows = mark


# Set by `run_setup` on a terminal; unset, going back asks again below.
_screen: ContextVar[_Screen | None] = ContextVar("setup_screen", default=None)


def _screen_mark() -> int:
    """Where the next question starts on screen, for `_back_to`."""
    screen = _screen.get()
    return screen.rows if screen is not None else 0


def _back_to(mark: int) -> None:
    """Ctrl-B inside a step that asks several questions, going back to its first: erase
    back to ``mark`` (`_screen_mark`) on a terminal, and let the questions asked again
    open on the answers just given (`_Recall.settle`)."""
    screen = _screen.get()
    if screen is not None:
        screen.erase_to(mark)
    recall = _recall.get()
    if recall is not None:
        recall.settle()


@dataclass
class _Draft:
    """What the wizard has been told so far: copied before each step and restored when
    Ctrl-B goes back to it (`run_setup`)."""

    provider: ProviderConfig
    """The config's provider, or a new one of the type picked: what the model questions open with."""
    agent: AgentConfig
    keep_model: bool = False
    keep_instructions: bool = False
    keep_tools: bool = False
    keep_skills: bool = False
    provider_type: str = ""
    model_name: str = ""
    url: str = ""
    space_id: str = ""
    configuration_id: str = ""
    max_tokens: int = 0
    context_window: int | None = None
    decisions: DecisionsChoice | None = None
    discard: list[Path] = field(default_factory=list)
    """Instruction files an import brought and the user did not keep, deleted once saved."""


@dataclass
class _Step:
    """A question of `run_setup`, or a few that belong together, and when it is asked."""

    name: str
    run: Callable[[_Draft], None]
    applies: Callable[[_Draft], bool] = lambda draft: True


def _select(
    console: Console,
    theme: Theme,
    label: str,
    options: list[str],
    *,
    current: str | None = None,
    **kwargs: Any,
) -> str | None:
    """`select`, opening on the answer this step gave last time (`_Recall`)."""
    recall = _recall.get()
    if recall is not None:
        current = recall.opening("select", label, current)
    chosen = select(console, theme, label, options, current=current, **kwargs)
    if recall is not None and chosen is not None:
        recall.record("select", label, chosen)
    return chosen


def _multiselect(
    console: Console,
    theme: Theme,
    label: str,
    options: list[str],
    *,
    selected: Iterable[str] = (),
    **kwargs: Any,
) -> list[str] | None:
    """`multiselect`, opening with the ticks this step gave last time (`_Recall`)."""
    recall = _recall.get()
    if recall is not None:
        selected = recall.opening("multiselect", label, selected)
    chosen = multiselect(console, theme, label, options, selected=selected, **kwargs)
    if recall is not None and chosen is not None:
        recall.record("multiselect", label, chosen)
    return chosen


def _ask(
    console: Console,
    theme: Theme,
    label: str,
    current: str = "",
    *,
    default: str = "",
    password: bool = False,
) -> str:
    """Prompt for a value, starting from the current one.

    The field opens with ``current`` in it: Enter keeps it, and erasing it gives
    ``default``, which the label should name (``empty = …``) — the only way back
    to a default once a value was saved. A pipe cannot pre-fill, so there Enter
    keeps ``current`` (:func:`prompt.read_line`).

    Through the line editor, so ←/→ move the cursor and ``password`` hides what
    is typed; a pipe still gets :func:`input`. Asked again after going back, it
    opens on the answer given last time (`_Recall`), never for a ``password``.
    """
    recall = _recall.get()
    if recall is not None:
        opened = recall.opening("ask", label, current)
        current = current if password else opened
    try:
        answer = read_line(
            console,
            f"  {label}: ",
            style=theme.style("accent"),
            password=password,
            prefill=current,
        )
    except (EOFError, KeyboardInterrupt):
        console.print()
        raise
    if recall is not None and not password:
        recall.record("ask", label, answer)
    return answer.strip() or default


def _decision_endpoint_answers(base_url: str, model: str, api_key: str | None = None) -> str | None:
    """``None`` if the endpoint answered, else one line saying what happened.

    Asked because the common mistakes here are silent ones: an endpoint that
    is not up, and a model name that is not the one the server exposes
    (``clef`` where it serves ``Cloudflare/clef``, which is a 400 nobody sees
    until a turn quietly stops asking). Checked with a real request rather than
    a socket connect, since only a request can tell those two apart.

    Never fatal: the endpoint may come up after setup, exactly as
    ``agent-browser`` may be installed after it.
    """
    import httpx

    from domyn.agents.decisions import noul

    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    try:
        response = httpx.post(
            f"{base_url.rstrip('/')}/v1/systemone",
            headers=headers,
            json={
                "model": model,
                "state": "ready?",
                "questions": {"q": noul("Is this endpoint answering?").to_wire()},
            },
            timeout=5.0,
        )
    except Exception as exc:
        return f"no answer from {base_url} ({type(exc).__name__})"
    if response.status_code >= 400:
        detail = response.text.strip().replace("\n", " ")[:120]
        return f"{base_url} answered {response.status_code}: {detail}"
    return None


def _ask_decision_key(
    console: Console, theme: Theme, current_env: str | None
) -> tuple[str | None, str | None]:
    """The endpoint's key as ``(variable name, key)``, both ``None`` without one.

    Same rule as the LLM key: environment first, then ``~/.domyn/credentials``;
    a pasted key goes to the credentials file, never to ``config.yaml``, which
    only names the variable.
    """
    name = current_env or DEFAULT_DECISION_KEY_ENV
    key = os.environ.get(name) or read_credentials().get(name)
    if key:
        console.print(f"  api key: found in {name}", style=theme.style("meta"), highlight=False)
        return name, key
    secret = _ask(console, theme, "decision api key (empty = none, sent as Bearer)", password=True)
    if not secret:
        return None, None
    saved = write_credential(name, secret)
    os.environ[name] = secret
    console.print(
        f"  saved to {saved} (permissions 0600, never in config.yaml)",
        style=theme.style("meta"),
        highlight=False,
    )
    return name, secret


def _ask_decisions(console: Console, theme: Theme, current: DecisionsChoice) -> DecisionsChoice:
    """Offer a decision model, and say on screen what each level hands over.

    Four levels rather than yes/no because the interesting question is not
    whether a model is configured but where it is allowed to matter, and the
    step that lets it answer at the approval gate is the one worth choosing
    deliberately. See ``DECISION_LEVELS``.

    Ctrl-B on the endpoint, the model name or the key goes back to the level.
    """
    while True:
        mark = _screen_mark()
        chosen = _select(
            console,
            theme,
            "decision model",
            list(DECISION_LEVELS),
            current=current.level,
            descriptions=DECISION_LEVEL_BLURB,
        )
        if chosen is None:
            raise KeyboardInterrupt
        if chosen == "no":
            return DecisionsChoice(level="no", base_url=current.base_url, model=current.model)

        console.print(
            "  a model that answers typed questions with probabilities (Jev, Clef)",
            style=theme.style("meta"),
            highlight=False,
        )
        try:
            base_url = _ask(
                console,
                theme,
                f"decision endpoint (empty = {DEFAULT_DECISION_URL})",
                current.base_url,
                default=DEFAULT_DECISION_URL,
            )
            console.print(
                "  the model name the SERVER exposes, which is not always the short one",
                style=theme.style("meta"),
                highlight=False,
            )
            model = _ask(
                console,
                theme,
                f"decision model name (empty = {DEFAULT_DECISION_MODEL})",
                current.model,
                default=DEFAULT_DECISION_MODEL,
            )
            key_env, api_key = _ask_decision_key(console, theme, current.api_key_env)
        except GoBack:
            _back_to(mark)
            continue
        break

    answer = DecisionsChoice(level=chosen, base_url=base_url, model=model, api_key_env=key_env)
    problem = _decision_endpoint_answers(base_url, model, api_key)
    if problem is None:
        console.print("  endpoint: answering", style=theme.style("meta"), highlight=False)
    else:
        console.print(f"  {problem}", style=theme.style("warning"), highlight=False)
        console.print(
            "  saved anyway — a turn degrades to today's behaviour while it is unreachable",
            style=theme.style("meta"),
            highlight=False,
        )
    return answer


def _ask_max_tokens(console: Console, theme: Theme, current: int) -> int:
    """Prompt for the generation ceiling, re-asking on anything but a positive int."""
    default = str(ProviderConfig().max_tokens)
    while True:
        raw = _ask(
            console,
            theme,
            f"Max tokens per generation (empty = {default})",
            str(current),
            default=default,
        )
        try:
            value = int(raw)
        except ValueError:
            console.print(
                f"  Not a number: {raw!r} — try again",
                style=theme.style("warning"),
                highlight=False,
            )
            continue
        if value <= 0:
            console.print(
                "  Must be a positive number — try again",
                style=theme.style("warning"),
                highlight=False,
            )
            continue
        return value


def _ask_context_window(console: Console, theme: Theme, current: int | None) -> int | None:
    """Prompt for the model's window, a number or ``AUTO_CONTEXT_WINDOW``.

    The field opens with the current value, "auto" when there is no number; an
    erased field, or "auto" typed, goes back to asking the provider.
    """
    display = str(current) if current is not None else AUTO_CONTEXT_WINDOW
    while True:
        raw = _ask(
            console,
            theme,
            f"Context window (tokens, empty = {AUTO_CONTEXT_WINDOW})",
            display,
            default=AUTO_CONTEXT_WINDOW,
        )
        if raw.lower() == AUTO_CONTEXT_WINDOW:
            return None
        try:
            value = int(raw)
        except ValueError:
            console.print(
                f"  Not a number: {raw!r} — try again",
                style=theme.style("warning"),
                highlight=False,
            )
            continue
        if value <= 0:
            console.print(
                "  Must be a positive number — try again",
                style=theme.style("warning"),
                highlight=False,
            )
            continue
        return value


# Shown next to each provider so the choice is informative rather than a bare list.
_PROVIDER_BLURB = {
    "domyn": "the models of your Domyn platform space",
    "openai": "OpenAI API",
    "anthropic": "Anthropic API",
    "open_router": "many models behind one key",
    "cerebras": "Cerebras inference",
    "vllm": "your own vLLM deployment",
    "custom": "any OpenAI-compatible endpoint",
}


def _choose_provider(console: Console, theme: Theme, current: str) -> str:
    """Pick a provider, with arrow keys on a real terminal and numbers otherwise."""
    options = list(PROVIDER_HINTS)
    chosen = _select(
        console,
        theme,
        "Provider",
        options,
        current=current,
        descriptions=_PROVIDER_BLURB,
    )
    if chosen is None:
        raise KeyboardInterrupt
    return chosen


_OTHER_MODEL = "other (type a name)"


def _ask_setting(console: Console, theme: Theme, label: str, saved: str, env_name: str) -> str:
    """A required setting that can come from the environment, the way a key does.

    A value already saved in the config opens the field, as everywhere else, and erasing it falls
    back to the variable. Otherwise the variable is named, never its value: Enter accepts it, and
    typing another value replaces it.
    """
    from_env = os.environ.get(env_name, "")
    if not saved and from_env:
        console.print(f"  {label}: found in {env_name}", style=theme.style("meta"), highlight=False)
    elif not saved:
        console.print(
            f"  {label}: {env_name} is not set or is empty",
            style=theme.style("warning"),
            highlight=False,
        )
    hint = f"(empty = use {env_name})" if from_env else "(required)"
    return _ask(console, theme, f"{label} {hint}", saved, default=from_env)


def _discover_models(console: Console, theme: Theme, settings: PlatformSettings) -> dict[str, str]:
    """The models of the space, by name with a hint about what is behind each; empty when unknown.

    Best effort: with no key, URL or space to ask with, or no SDK, or when the platform does not
    answer, the wizard says why and the model is typed instead. A failed lookup never stops setup.
    """
    needed = {
        "the platform url": settings.base_url,
        "the space id": settings.space_id,
        "an api key": settings.api_key,
    }
    missing = [what for what, value in needed.items() if not value]
    if missing:
        console.print(
            f"  Models: listing them needs {', '.join(missing)}",
            style=theme.style("meta"),
            highlight=False,
        )
        return {}
    if not sdk_installed():
        return {}
    console.print(
        "  Looking up the models of the space...", style=theme.style("meta"), highlight=False
    )
    try:
        found = run_sync(list_models(settings))
    except PlatformSDKNotInstalledError as error:
        # Says which version is installed and how to get the right one: the one error worth showing whole.
        console.print(
            f"  Could not list the models: {error}", style=theme.style("warning"), highlight=False
        )
        return {}
    except Exception as error:
        # An HTTP error's status says what to fix: 404 the route, 401/403 the key or the space.
        status = getattr(getattr(error, "response", None), "status_code", None)
        what = f"{type(error).__name__}, HTTP {status}" if status else type(error).__name__
        console.print(
            f"  Could not list the models ({what}): type the name instead",
            style=theme.style("warning"),
            highlight=False,
        )
        return {}
    models: dict[str, str] = {}
    for model in found:
        if not model.get("id"):
            continue
        behind = " ".join(str(model[k]) for k in ("type", "root") if model.get(k))
        models[str(model["id"])] = behind
    return models


def _choose_domyn_model(
    console: Console, theme: Theme, models: dict[str, str], current: str
) -> str:
    """Pick one of the space's models from the list, or type a name that is not on it.

    Ctrl-B on the typed name goes back to the list."""
    options = [*models, _OTHER_MODEL]
    while True:
        mark = _screen_mark()
        chosen = _select(
            console,
            theme,
            "Model",
            options,
            current=current,
            descriptions={name: hint for name, hint in models.items() if hint},
        )
        if chosen is None:
            raise KeyboardInterrupt
        if chosen != _OTHER_MODEL:
            return chosen
        try:
            return _ask(console, theme, "Model", current)
        except GoBack:
            _back_to(mark)
            continue


def _ensure_api_key(console: Console, theme: Theme, provider: ProviderConfig) -> None:
    """Ask for the key and keep it in the credentials file, or say where it was found.

    A key in a credentials file, saved by an earlier setup or pasted earlier in this one,
    is offered masked: Enter keeps it, a key typed replaces it. One from the environment
    or a ``.env`` is only reported, since the environment wins over the file and setup
    could not replace it.

    A key found in ``~/.domyn/credentials`` is not copied into a project: the project
    reads that file too when it runs (`credential_files`). One typed in a project goes
    to the project's file, and replaces the global one for that project only."""
    key_env = provider.default_key_env()
    found = find_api_key(key_env)
    if found is not None and found.path is None:
        console.print(
            f"  API key: found ({found.source})", style=theme.style("meta"), highlight=False
        )
        return
    if found is None:
        console.print(
            f"  API key: not found in {key_env}", style=theme.style("warning"), highlight=False
        )
        secret = _ask(console, theme, "Paste key (empty = skip)", password=True)
    else:
        console.print(
            f"  API key: saved ({found.source})", style=theme.style("meta"), highlight=False
        )
        secret = _ask(
            console,
            theme,
            "API key",
            found.value,
            default=found.value,
            password=True,
        )
        if secret == found.value:
            return
    if secret:
        saved = write_credential(key_env, secret)
        # Into the environment as a key from the credentials file, so going back to this
        # question finds it there and offers it again rather than reporting it as found.
        load_credentials_into_environment()
        console.print(
            f"  Saved to {saved} (permissions 0600, never in config.yaml)",
            style=theme.style("meta"),
            highlight=False,
        )


def _short(text: str, width: int = 70) -> str:
    """A description cut to one line for the picker, which draws it beside the name."""
    line = " ".join(text.split())
    return line if len(line) <= width else line[: width - 1].rstrip() + "…"


def _inside(path: Path, directory: Path) -> bool:
    return path.expanduser().absolute().is_relative_to(directory.expanduser().absolute())


def _count(n: int, noun: str = "skill") -> str:
    return f"{n} {noun}{'s' if n != 1 else ''}"


def _browser_binary_found(binary: str) -> bool:
    """On PATH, or the copy that ships with the standalone build."""
    resolved = resolve_binary(binary)
    return shutil.which(resolved) is not None or Path(resolved).is_file()


def _choose_tools(
    console: Console, theme: Theme, current: list[str], *, first_setup: bool, browser_binary: str
) -> list[str]:
    """Pick the built-in tool bundles the agent mounts.

    Every bundle starts ticked on a first setup, the browser included, the current
    choice on a re-run; what is unticked is not mounted. A name in ``current`` that is not a built-in
    bundle (written by hand, or a bundle since removed) is not offered and is kept
    after the chosen ones, as the skills step keeps paths it does not own.

    Raises ``KeyboardInterrupt`` when the picker is aborted, like every other question.
    """
    options = list(BUNDLES)
    ticked = options if first_setup else [name for name in options if name in current]
    unknown = [name for name in current if name not in BUNDLES]

    console.print()
    chosen = _multiselect(
        console,
        theme,
        "Built-in tools",
        options,
        selected=ticked,
        descriptions={name: _short(BUNDLE_DESCRIPTIONS.get(name, "")) for name in options},
        max_tries=3,
    )
    if chosen is None:
        raise KeyboardInterrupt

    tools = [*chosen, *unknown]
    console.print(
        f"  Tools: {', '.join(tools)}" if tools else "  Tools: none (conversation only)",
        style=theme.style("meta"),
        highlight=False,
    )
    # Mounted anyway: installing agent-browser is the user's next step, and refusing
    # would only mean running the wizard twice. Checked on this machine, which is
    # where commands run while ``local`` is the only backend.
    if "browser" in chosen and not _browser_binary_found(browser_binary):
        console.print(
            f"  agent-browser was not found on PATH. {INSTALL_HINT}",
            style=theme.style("warning"),
            highlight=False,
        )
    return tools


def _choose_project_skills(
    console: Console, theme: Theme, current: dict[str, Any] | None, project: Path
) -> dict[str, Any] | None:
    """Pick among the skills the imported config lists in the project, ``builtin-skills/``
    and ``skills/`` alike.

    The import's counterpart of `_choose_builtin_skills`: one row per skill, grouped by
    the folder it sits in, all ticked to start with. Every skill is offered as exported,
    whether it began as a built-in or not, and none of them is installed again or
    copied. The project is walked once for both folders, so no two rows share a label.

    The catalog's built-ins the export does not include are offered too, unticked, in
    their category's group beside the exported ones. Those ticked are added to
    `skills_install_dir()` (`add_skills`: nothing already there is replaced, so the
    exported built-ins stay as they came) and listed after the built-ins already listed.

    A listed path whose skills all stay ticked is kept as written, so a folder of
    skills still loads one added to it later; otherwise the ticked ones are listed one
    by one in its place, keeping the order that settles a name clash. A path elsewhere
    is kept and not offered. Raises ``KeyboardInterrupt`` when the picker is aborted.
    """
    source = (current or {}).get("source") or {}
    if source and source.get("type") != "local_dir":
        return current

    discovery = discover_local_skills(project)
    entries: list[tuple[str, list[LocalSkill]]] = []
    offered: set[Path] = set()
    for entry in (str(path) for path in source.get("paths") or []):
        listed = Path(entry).expanduser().absolute()
        members = [
            skill
            for skill in discovery.skills
            if skill.path not in offered and (skill.path == listed or _inside(skill.path, listed))
        ]
        if is_skill_autodiscovery(entry) or not _inside(listed, project):
            members = []
        offered.update(skill.path for skill in members)
        entries.append((entry, members))

    skills = [skill for _, members in entries for skill in members]
    # The built-ins the export left out, as rows of their own, unticked: ticked, they
    # are added to the install directory beside the ones the export brought.
    install_dir = skills_install_dir().expanduser().absolute()
    catalog = list_builtin_skills()
    in_catalog_order = {install_dir / builtin.relative_path: i for i, builtin in enumerate(catalog)}
    exported = {skill.path.expanduser().absolute() for skill in skills}
    taken = {skill.label for skill in skills} | {skill.group for skill in skills if skill.group}
    missing: dict[str, BuiltinSkill] = {}
    missing_rows: list[LocalSkill] = []
    for builtin in catalog:
        path = install_dir / builtin.relative_path
        if path in exported:
            continue
        label = builtin.name if builtin.name not in taken else f"{builtin.name} ({path.as_posix()})"
        taken.add(label)
        category = install_dir / builtin.category
        # The group the exported built-ins of that category sit in, so the two mix.
        group = (
            category.relative_to(project.expanduser().absolute()).as_posix()
            if _inside(category, project)
            else f"built-in/{builtin.category}"
        )
        missing[label] = builtin
        missing_rows.append(
            LocalSkill(
                name=builtin.name,
                label=label,
                description=builtin.description,
                path=path,
                group=group,
            )
        )

    if not skills and not missing:
        console.print(
            "  Skills: the imported project lists none of its own",
            style=theme.style("meta"),
            highlight=False,
        )
        return current

    rows = [*skills, *missing_rows]
    groups: dict[str, list[LocalSkill]] = {}
    for skill in rows:
        if skill.group is not None:
            groups.setdefault(skill.group, []).append(skill)
    for members in groups.values():
        # A category's built-ins in catalog order, exported or not; anything else after.
        members.sort(
            key=lambda skill: in_catalog_order.get(
                skill.path.expanduser().absolute(), len(in_catalog_order)
            )
        )

    console.print()
    if missing:
        console.print(
            f"  The built-in skills the export does not include are listed unticked: "
            f"ticked, they are installed to {install_dir}.",
            style=theme.style("meta"),
            highlight=False,
            markup=False,  # a path may hold `[`
        )
    chosen = _multiselect(
        console,
        theme,
        f"Skills of the imported project, in {project}",
        [skill.label for skill in rows],
        selected=[skill.label for skill in skills],
        descriptions={
            **{skill.label: _short(skill.description) for skill in rows},
            **{group: _count(len(members)) for group, members in groups.items()},
        },
        groups={group: [skill.label for skill in members] for group, members in groups.items()},
        max_tries=3,
    )
    if chosen is None:
        raise KeyboardInterrupt
    ticked = set(chosen)

    paths: list[str] = []
    for entry, members in entries:
        if not members or all(skill.label in ticked for skill in members):
            paths.append(entry)
        else:
            paths.extend(str(skill.path) for skill in members if skill.label in ticked)

    to_add = [missing[label].name for label in chosen if label in missing]
    added: list[Path] = []
    if to_add:
        try:
            added = add_skills(to_add, install_dir)
        except (SkillInstallError, OSError, ValueError) as exc:
            console.print(
                f"  Could not install the built-in skills: {type(exc).__name__}: {exc}",
                style=theme.style("error"),
                highlight=False,
                markup=False,  # `[Errno 13]` is not a rich tag
            )
    if added:
        # After the built-ins already listed, else last: the place a setup gives them.
        after = max(
            (i + 1 for i, path in enumerate(paths) if _inside(Path(path), install_dir)),
            default=len(paths),
        )
        paths[after:after] = [str(path) for path in added]

    kept = sum(1 for skill in skills if skill.label in ticked)
    summary = f"  Skills: {kept} of the imported project's {_count(len(skills))}"
    if added:
        summary += f", and {_count(len(added), 'built-in skill')} installed to {install_dir}"
    console.print(summary, style=theme.style("meta"), highlight=False, markup=False)

    updated = dict(current or {})
    if paths:
        updated["source"] = {**source, "type": "local_dir", "paths": paths}
    else:
        updated.pop("source", None)
    return updated or None


def _choose_builtin_skills(
    console: Console, theme: Theme, current: dict[str, Any] | None
) -> dict[str, Any] | None:
    """Pick the built-in skills, install them, and return the agent's ``skills:`` mapping.

    The skills directory is rewritten from scratch: what was ticked is what it holds
    afterwards. The mapping lists each installed skill's folder by absolute path, so
    the config, not a scan, decides what loads. Paths the user listed that point
    elsewhere are kept; a source that is not a local directory is left alone.

    Returns ``current`` unchanged when nothing was installed (a declined or failed
    replace), so the config never names skills the directory does not hold. Raises
    ``KeyboardInterrupt`` when the picker is aborted, like every other question here.
    """
    install_dir = skills_install_dir()
    source = (current or {}).get("source") or {}
    if source and source.get("type") != "local_dir":
        console.print(
            f"  Skills: kept as they are (source type {source.get('type')!r})",
            style=theme.style("meta"),
            highlight=False,
        )
        return current

    declared = [Path(str(path)) for path in source.get("paths") or []]
    elsewhere = [path for path in declared if not _inside(path, install_dir)]

    catalog = list_builtin_skills()
    names = [skill.name for skill in catalog]
    groups = skill_groups()
    if current is None:
        ticked = names
    else:
        installed = {path.expanduser().absolute() for path in declared}
        ticked = [
            skill.name
            for skill in catalog
            if (install_dir.expanduser().absolute() / skill.relative_path) in installed
        ]

    while True:
        mark = _screen_mark()
        console.print()
        chosen = _multiselect(
            console,
            theme,
            f"Built-in skills, installed to {install_dir}",
            names,
            selected=ticked,
            descriptions={
                **{skill.name: _short(skill.description) for skill in catalog},
                **{group: _count(len(members)) for group, members in groups.items()},
            },
            groups=groups,
            # A stdin that keeps repeating one unusable answer must not hold the wizard forever.
            max_tries=3,
        )
        if chosen is None:
            raise KeyboardInterrupt

        leftovers = unrecognized_entries(install_dir)
        if leftovers:
            shown = ", ".join(str(path) for path in leftovers[:8])
            more = f" and {len(leftovers) - 8} more" if len(leftovers) > 8 else ""
            console.print(
                f"  {install_dir} is rewritten from scratch, and holds files that are not "
                f"built-in skills: {shown}{more}",
                style=theme.style("warning"),
                highlight=False,
                markup=False,  # file names are the user's, and may hold `[`
            )
            try:
                answer = _select(
                    console,
                    theme,
                    "Delete them and continue?",
                    ["no", "yes"],
                    current="no",
                    aliases={"n": "no", "y": "yes"},
                    max_tries=3,
                )
            except GoBack:
                _back_to(mark)
                continue  # Ctrl-B on the question goes back to the picker
            if answer != "yes":
                console.print(
                    "  Skills left as they were.", style=theme.style("meta"), highlight=False
                )
                return current
        break

    try:
        installed_paths = install_skills(chosen, install_dir)
    except (SkillInstallError, OSError) as exc:
        console.print(
            f"  Could not install the skills: {type(exc).__name__}: {exc}",
            style=theme.style("error"),
            highlight=False,
            markup=False,  # `[Errno 13]` is not a rich tag
        )
        return current

    console.print(
        f"  Installed {_count(len(installed_paths))} to {install_dir}",
        style=theme.style("meta"),
        highlight=False,
    )

    paths = [str(path) for path in installed_paths] + [str(path) for path in elsewhere]
    updated = dict(current or {})
    if paths:
        updated["source"] = {**source, "type": "local_dir", "paths": paths}
    else:
        updated.pop("source", None)
    return updated or None


def _local_dir(path: str | Path) -> Path | None:
    """The project skills folder a listed path points into, however it is written, if any."""
    return next((d for d in LOCAL_SKILLS_DIRS if _inside(Path(str(path)), d)), None)


def _is_local(path: str | Path) -> bool:
    """Whether a listed path points into one of the project's skills folders."""
    return _local_dir(path) is not None


def _local_dirs_shown() -> str:
    """The project skills folders as the questions name them: ``./a and ./b``."""
    return " and ".join(f"./{d.as_posix()}" for d in LOCAL_SKILLS_DIRS)


def _resolved(entries: list[str]) -> dict[Path, str]:
    """Listed paths by where they resolve to, relative ones against the working directory
    as the loader reads them, so the same place spelled another way is one key."""
    return {Path(entry).expanduser().resolve(): entry for entry in entries}


def _custom_dir_problem(path: Path, declared: list[str]) -> str | None:
    """Why ``path`` cannot be offered as the user's skills folder, or ``None`` if it can.

    ``path`` is already absolute. A listed entry is compared by where it resolves to,
    so the same folder spelled another way is still caught.
    """
    if path.is_symlink():
        return f"{path} is a link, and links are not followed"
    if not path.exists():
        return f"{path} does not exist"
    if not path.is_dir():
        return f"{path} is not a folder"
    if _inside(path, skills_install_dir()):
        return f"{path} is inside {skills_install_dir()}, which setup rewrites from scratch"
    local = _local_dir(path)
    if local is not None:
        return f"{path} is ./{local.as_posix()}, the next question turns it on"
    entry = _resolved(declared).get(path.resolve())
    if entry is not None:
        return f"{path} is already listed in the config (as {entry})"
    return None


def _choose_custom_skills(
    console: Console, theme: Theme, current: dict[str, Any] | None
) -> dict[str, Any] | None:
    """List every skill in a folder of the user's own, each by its absolute path.

    The folder is walked once, here, as the loader walks a listed directory (group
    folders included, links never followed), and each skill found is listed on its
    own: its folder, or its file when it is a loose ``.md``. Nothing is copied, so a
    skill edited there is read as edited; a skill added there later is not listed
    until setup runs again. The skills go first in the paths list, so the local step
    puts the project folders before them and both win a name clash with a built-in. Entries
    added on an earlier setup are not this step's: like any other path elsewhere,
    both other steps keep them, so an empty answer leaves the config as it is.

    Refused, and asked again up to three times: a folder that is not there, a link
    (the loader does not follow one), one setup owns, one the config already lists,
    and one with no skill the config does not already list. A skill already listed
    is not listed twice.
    """
    source = (current or {}).get("source") or {}
    if source and source.get("type") != "local_dir":
        return current

    declared = [str(path) for path in source.get("paths") or []]
    listed = _resolved(declared)

    console.print()
    for _ in range(3):
        raw = _ask(console, theme, "Add a folder with your own skills (empty = none)")
        if not raw:
            return current
        # Absolute and with `..` collapsed, but a link is not resolved: what is saved
        # is under the folder the user named.
        path = Path(os.path.abspath(Path(raw).expanduser()))
        problem = _custom_dir_problem(path, declared)
        if problem is None:
            found = discover_local_skills(path)
            new = [s for s in found.skills if s.path.resolve() not in listed]
            if not found.skills:
                problem = f"No skills found in {path}"
            elif not new:
                problem = f"Every skill in {path} is already listed in the config"
            else:
                break
        console.print(
            f"  {problem} — try again",
            style=theme.style("warning"),
            highlight=False,
            markup=False,  # the path is the user's, and may hold `[`
        )
    else:
        console.print("  No skills added", style=theme.style("meta"), highlight=False)
        return current

    for skipped, kind in found.skipped:
        console.print(
            f"  Not listed, could not be read ({kind}): {skipped}",
            style=theme.style("warning"),
            highlight=False,
            markup=False,
        )
    for skill in new:
        if skill.missing:
            # Listed all the same: a skill half way through being written must not vanish.
            console.print(
                f"  Skill {skill.name!r} lists a file that is missing: {', '.join(skill.missing)}",
                style=theme.style("warning"),
                highlight=False,
                markup=False,
            )
    already = len(found.skills) - len(new)
    console.print(
        f"  Listed {_count(len(new))} from {path}, read from there, not copied"
        + (f" ({already} already listed)" if already else ""),
        style=theme.style("meta"),
        highlight=False,
        markup=False,
    )

    paths = [str(skill.path) for skill in new]
    updated = dict(current or {})
    updated["source"] = {**source, "type": "local_dir", "paths": [*paths, *declared]}
    return updated


def _choose_instructions(
    console: Console, theme: Theme, current: dict[str, Any] | None
) -> dict[str, Any] | None:
    """Ask where the user's ``SOUL.md`` / ``AGENTS.md`` are, and list what is found there.

    A file is taken whatever its name, as the loader takes one. A folder is walked
    (`find_instruction_files`): its first ``SOUL.md`` and every ``AGENTS.md``, each
    saved by absolute path, so a later edit is read on the next message but a file
    added later needs setup again. Nothing found, or a path that is not there, is a
    warning and saves nothing.

    Unlike the other prompts an empty answer does not keep the current value, it
    removes it, so the current paths are printed first. This step owns every listed
    path but the autodiscovery entries, which `_choose_local_instructions` owns and
    which are kept after the user's. A source setup did not write (not ``local_dir``)
    is its author's and kept without asking.
    """
    source = (current or {}).get("source") or {}
    if source and source.get("type") != "local_dir":
        console.print(
            f"  Instructions: keeping the {source.get('type')} source in the config",
            style=theme.style("meta"),
            highlight=False,
            markup=False,
        )
        return current

    console.print()
    listed = [str(path) for path in source.get("paths") or []]
    autodiscovery = [path for path in listed if is_instruction_autodiscovery(path)]
    declared = [path for path in listed if not is_instruction_autodiscovery(path)]
    if declared:
        console.print(
            f"  Instructions currently: {', '.join(declared)} (empty = remove)",
            style=theme.style("meta"),
            highlight=False,
            markup=False,  # the paths are the user's, and may hold `[`
        )
    found: list[Path] = []
    raw = _ask(console, theme, "Add a folder or file with SOUL.md / AGENTS.md (empty = none)")
    if raw:
        # Absolute and with `..` collapsed, but a link is not resolved: what is saved
        # is under the path the user named.
        path = Path(os.path.abspath(Path(raw).expanduser()))
        problem = None
        if path.is_symlink():
            problem = f"{path} is a link, and links are not followed"
        elif path.is_file():
            found = [path]
        elif path.is_dir():
            found = find_instruction_files(path).paths
            if not found:
                problem = f"No SOUL.md or AGENTS.md found in {path}"
        else:
            problem = f"{path} does not exist"
        if problem is not None:
            console.print(
                f"  {problem} — no instructions saved",
                style=theme.style("warning"),
                highlight=False,
                markup=False,
            )

    for one in found:
        console.print(
            f"  Instructions: {one}", style=theme.style("meta"), highlight=False, markup=False
        )
    paths = [*(str(one) for one in found), *autodiscovery]
    if not paths:
        return None
    return {**(current or {}), "source": {**source, "type": "local_dir", "paths": paths}}


def _choose_local_instructions(
    console: Console,
    theme: Theme,
    current: dict[str, Any] | None,
    *,
    first_setup: bool,
) -> dict[str, Any] | None:
    """Turn autodiscovery of ``./.domyn`` and ``./.agents`` instructions on or off.

    On, the paths list both as written, in `AUTODISCOVERY_DIRS` order and **last**
    (the loader walks them, subfolders included):
    the first ``SOUL.md`` wins, so a project's soul is used only when the user's folder
    has none, and every ``AGENTS.md`` is included with the project's after the user's.
    Relative, they are read every turn from the folder ``domyn run`` starts in, so a file
    added there is loaded without running setup again. The question is always asked,
    since the answer applies to every folder; one answer covers both.

    This step owns the two autodiscovery entries: each setup rewrites them. Every other
    path is kept as it is.
    """
    source = (current or {}).get("source") or {}
    if source and source.get("type") != "local_dir":
        return current

    listed = [str(path) for path in source.get("paths") or []]
    rest = [path for path in listed if not is_instruction_autodiscovery(path)]
    enabled = first_setup or len(rest) < len(listed)
    shown = " and ".join(f"./{d.as_posix()}" for d in AUTODISCOVERY_DIRS)

    console.print()
    answer = _select(
        console,
        theme,
        "Enable instruction autodiscovery? In every folder domyn runs in, "
        f"load SOUL.md and AGENTS.md from its {shown}",
        ["yes", "no"],
        current="yes" if enabled else "no",
        aliases={"y": "yes", "n": "no"},
        max_tries=3,
    )
    if answer is None:
        raise KeyboardInterrupt

    paths = rest
    if answer == "yes":
        paths = [*rest, *(d.as_posix() for d in AUTODISCOVERY_DIRS)]
        console.print(
            f"  {shown} are loaded from the folder domyn runs in, subfolders included; "
            "a SOUL.md there is used only when the instructions above have none",
            style=theme.style("meta"),
            highlight=False,
        )

    updated = dict(current or {})
    if paths:
        updated["source"] = {**source, "type": "local_dir", "paths": paths}
    else:
        updated.pop("source", None)
    return updated or None


def _choose_imported_instructions(
    console: Console,
    theme: Theme,
    current: dict[str, Any] | None,
    *,
    discard: list[Path],
) -> dict[str, Any] | None:
    """The instructions step of an import whose instructions are not kept (`_keep`).

    What the project brought is what its config declares: nothing is read without
    being declared, a ``SOUL.md`` at the project's top included. Not kept, those paths
    are unlisted and added to ``discard``, whose copies inside the project
    `run_setup` deletes once the config is saved (`_discard_imported`), as nothing
    would read them again: deleted here, going back to keep them would keep paths
    whose files are gone. One the saved config lists again is not deleted. Then the
    usual question is asked (`_choose_instructions`), and the path given may bring a
    ``SOUL.md`` of its own. A source setup did not write (not ``local_dir``) is its
    author's and kept.
    """
    source = (current or {}).get("source") or {}
    if source and source.get("type") != "local_dir":
        return _choose_instructions(console, theme, current)
    declared = [str(path) for path in source.get("paths") or []]
    discard.extend(Path(entry) for entry in declared)
    if declared:
        console.print(
            "  Imported instructions unlisted, and deleted once the setup is saved",
            style=theme.style("meta"),
            highlight=False,
        )
    return _choose_instructions(console, theme, None)


def _discard_imported(path: Path, project: Path) -> None:
    """Delete an instruction file or folder the import copied, i.e. one inside the
    project; anything else is only unlisted."""
    if not _inside(path, project) or path.is_symlink():
        return
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)
    elif path.is_file():
        path.unlink()


def _choose_local_skills(
    console: Console,
    theme: Theme,
    current: dict[str, Any] | None,
    *,
    first_setup: bool,
) -> dict[str, Any] | None:
    """Turn autodiscovery of ``./.domyn/skills`` and ``./.agents/skills`` on or off
    (`with_skill_autodiscovery`).

    Relative, the entries are read every turn from the folder ``domyn run`` starts in,
    so they work from any folder and a skill added there is loaded without running
    setup again. Nothing is walked here, and the question is always asked, since the
    answer applies to every folder. One answer covers both folders. ``/skills-autodiscover``
    flips the same switch from inside a session.
    """
    on = skill_autodiscovery(current)
    if on is None:
        return current
    enabled = first_setup or on

    console.print()
    answer = _select(
        console,
        theme,
        "Enable skill autodiscovery? In every folder domyn runs in, "
        f"load skills from its {_local_dirs_shown()}",
        ["yes", "no"],
        current="yes" if enabled else "no",
        aliases={"y": "yes", "n": "no"},
        max_tries=3,
    )
    if answer is None:
        raise KeyboardInterrupt

    if answer == "yes":
        console.print(
            f"  {_local_dirs_shown()} are loaded from the folder domyn runs in",
            style=theme.style("meta"),
            highlight=False,
        )
    return with_skill_autodiscovery(current, answer == "yes")


def skill_autodiscovery(skills: dict[str, Any] | None) -> bool | None:
    """Whether an agent's ``skills`` entry turns on autodiscovery (either of
    `LOCAL_SKILLS_DIRS` listed); ``None`` when its source is not a ``local_dir``, which
    has no such switch."""
    source = (skills or {}).get("source") or {}
    if source and source.get("type") != "local_dir":
        return None
    return any(is_skill_autodiscovery(str(path)) for path in source.get("paths") or [])


def with_skill_autodiscovery(skills: dict[str, Any] | None, enabled: bool) -> dict[str, Any] | None:
    """An agent's ``skills`` entry with autodiscovery of ``./.domyn/skills`` and
    ``./.agents/skills`` on or off.

    On, the paths list both as written, in `LOCAL_SKILLS_DIRS` order and first, so a
    project skill wins a name clash with a built-in and ``.domyn`` wins one with
    ``.agents``. Every listed path that points into either folder is owned by this
    switch and rewritten; paths elsewhere are kept as they are. A source that is not
    a ``local_dir`` is returned unchanged.
    """
    if skill_autodiscovery(skills) is None:
        return skills
    source = (skills or {}).get("source") or {}
    rest = [str(path) for path in source.get("paths") or [] if not _is_local(path)]
    paths = [*(d.as_posix() for d in LOCAL_SKILLS_DIRS), *rest] if enabled else rest

    updated = dict(skills or {})
    if paths:
        updated["source"] = {**source, "type": "local_dir", "paths": paths}
    else:
        updated.pop("source", None)
    return updated or None


def _source_paths(mapping: dict[str, Any] | None) -> list[str]:
    """A ``skills:`` / ``standing_instructions:`` mapping's paths as written, or its
    source type when it is not a local directory."""
    source = (mapping or {}).get("source") or {}
    if source and source.get("type") != "local_dir":
        return [f"{source.get('type')} source"]
    return [str(path) for path in source.get("paths") or []]


def _keep(
    console: Console,
    theme: Theme,
    section: str,
    current: list[str],
    *,
    extra: dict[str, list[str]] | None = None,
) -> bool:
    """Show a section as the config holds it and ask whether to keep it.

    Asked on a re-run only, before the section's own questions, which a yes skips.
    Enter keeps, as on every other prompt. Raises ``KeyboardInterrupt`` when the
    question is aborted, like every other question here. ``extra`` adds blocks of
    the same section under headings of their own, aligned with the first.
    """
    console.print()
    for heading, lines in {f"Current {section}": current, **(extra or {})}.items():
        console.print(f"  {heading}:", style=theme.style("meta"), highlight=False)
        for line in lines or ["none"]:
            console.print(
                f"    {line}",
                style=theme.style("meta"),
                highlight=False,
                markup=False,  # paths are the user's, and may hold `[`
            )
    answer = _select(
        console,
        theme,
        f"Keep the current {section}?",
        ["yes", "no"],
        current="yes",
        aliases={"y": "yes", "n": "no"},
        max_tries=3,
    )
    if answer is None:
        raise KeyboardInterrupt
    return answer == "yes"


def run_setup(
    console: Console,
    theme: Theme,
    existing: dict[str, Any] | None = None,
    *,
    path: Path | None = None,
    provider_only: bool = False,
    imported: ProjectImport | None = None,
) -> dict[str, Any] | None:
    """Walk the user through configuration and persist it as a ConfigurationV1 file.

    Args:
        existing: the current config, as the raw dict ``read_raw_config``
            returns — not the wizard's ``ProviderConfig``/``AgentConfig``; see the
            ConfigurationV1 note in ``interfaces/cli/config.py``.
        provider_only: ask only about the model/provider — what ``/model`` needs.
        imported: the export ``existing`` was imported from (`import_project`),
            named in the title. Each section is offered for keeping, as on a re-run;
            skills not kept are chosen among the project's own instead of the
            built-in ones, and instructions not kept are erased before the usual
            question (`_choose_project_skills`, `_choose_imported_instructions`).

    Returns the saved config (the dict actually written), or ``None`` if the
    user aborted.

    On a re-run each section (model, instructions, tools, skills) first shows what
    the config holds and asks whether to keep it; a yes skips its questions. Not on
    a ``provider_only`` run, whose point is changing the model, and not for a model
    the config leaves incomplete. Keeping the skills does not reinstall
    ``~/.domyn/builtin-skills``, so built-in skills changed by an upgrade arrive only on a
    setup that does not keep them.

    Each question is a step (`_Step`) reading and writing one `_Draft`. On a
    terminal Ctrl-B goes back to the previous step asked, branches included: its draft
    is restored as it was before it, and its prompts open on the answers given there
    last time (`_Recall`). Ctrl-C still aborts, saving nothing.
    """
    project = active_project()
    # Worked on with the project's paths spelled out, so the questions below compare
    # real folders; written back in project terms when saved.
    if existing is not None:
        existing = expand_project_dir(existing, project)
    provider, agent, compaction = current_settings_from(existing)
    # An import asks too, each section showing what the export brought.
    ask_keep = existing is not None and not provider_only
    ask_keep_model = ask_keep and existing is not None and is_configuration_v1_usable(existing)
    first_setup = existing is None
    # Read before the skills steps fill the mapping in: none at all defaults
    # autodiscovery on. An import never does.
    first_skills = agent.skills is None and imported is None
    # An import offers the project's own skills, and erases its instructions when
    # they are not kept (`_choose_project_skills`, `_choose_imported_instructions`).
    import_into = project if imported is not None else None
    domyn_key = ProviderConfig(type="domyn", api_key_env=PROVIDER_HINTS["domyn"].get("env"))
    # The space's models by what they were asked with. Outside the draft, so going
    # back over the lookup does not ask the platform again.
    domyn_models: dict[tuple[str, ...], dict[str, str]] = {}

    console.print()
    title = (
        f"  Configure Domyn Main Agent — Project {describe_project(project)}\n"
        if project
        else "  Configure Domyn Main Agent\n"
    )
    console.print(title, style=theme.style("accent_bold"), highlight=False)
    if imported is not None:
        console.print(
            f"  Importing {imported.source}: each setting below comes from the export.\n",
            style=theme.style("meta"),
            highlight=False,
            markup=False,  # the path is the user's, and may hold `[`
        )
    elif ask_keep:
        console.print(
            f"  Editing {path or config_path()}\n",
            style=theme.style("meta"),
            highlight=False,
            markup=False,  # the path is the user's, and may hold `[`
        )

    def keep_model(d: _Draft) -> None:
        d.keep_model = _keep(
            console,
            theme,
            "model",
            [
                f"provider: {d.provider.type}",
                f"model: {d.provider.model_name}",
                f"URL: {_scrub_url(d.provider.url) or '(provider default)'}",
                *(
                    [
                        f"space id: {d.provider.space_id or '(from DOMYN_SPACE_ID)'}",
                        f"configuration id: {d.provider.configuration_id or '(last released)'}",
                    ]
                    if d.provider.type == "domyn"
                    else []
                ),
                f"max tokens: {d.provider.max_tokens}",
                f"context window: {d.provider.context_window or 'auto'}",
                f"API key: {d.provider.default_key_env()} "
                f"({'set' if resolve_api_key(d.provider) else 'missing'})",
            ],
        )

    def choose_provider(d: _Draft) -> None:
        recall = _recall.get()
        before = recall.answers.get(("provider", "select", "Provider")) if recall else None
        chosen = _choose_provider(console, theme, d.provider.type)
        if recall is not None and before not in (None, chosen):
            # What was typed for another provider is not this one's.
            recall.forget({step.name for step in model_steps})
        # If the chosen provider type has changed, reset the provider configuration.
        if chosen != d.provider.type:
            d.provider = ProviderConfig(type=chosen)
        d.provider_type = chosen
        d.url = d.provider.url or ""
        d.space_id = d.provider.space_id or ""
        d.configuration_id = d.provider.configuration_id or ""

    def model_names_hint(d: _Draft) -> None:
        hint = PROVIDER_HINTS.get(d.provider_type, {}).get("models", "(see your provider)")
        console.print(f"  Model names: {hint}", style=theme.style("meta"), highlight=False)

    def ask_model(d: _Draft) -> None:
        model_names_hint(d)
        d.model_name = _ask(console, theme, "Model", d.provider.model_name)

    def ask_platform_url(d: _Draft) -> None:
        if not sdk_installed():
            console.print(
                f'  Needs {SDK_REQUIREMENT}: uv pip install "{SDK_REQUIREMENT}"',
                style=theme.style("warning"),
                highlight=False,
            )
        d.url = _ask_setting(console, theme, "Platform URL", d.url, "DOMYN_BASE_URL")

    def ask_space_id(d: _Draft) -> None:
        d.space_id = _ask_setting(console, theme, "Space id", d.space_id, "DOMYN_SPACE_ID")

    def ask_configuration_id(d: _Draft) -> None:
        d.configuration_id = _ask(
            console, theme, "Configuration id (empty = last released)", d.configuration_id
        )

    def choose_domyn_model(d: _Draft) -> None:
        settings = PlatformSettings(
            api_key=resolve_api_key(domyn_key) or "",
            space_id=d.space_id,
            base_url=d.url,
            configuration_id=d.configuration_id or None,
        )
        asked_with = (settings.api_key, settings.space_id, settings.base_url, d.configuration_id)
        if asked_with not in domyn_models:
            domyn_models[asked_with] = _discover_models(console, theme, settings)
        models = domyn_models[asked_with]
        if models:
            d.model_name = _choose_domyn_model(console, theme, models, d.provider.model_name)
        else:
            ask_model(d)

    def ask_url(d: _Draft) -> None:
        label = (
            "URL (required)"
            if d.provider_type in REQUIRES_URL
            else "URL (empty = provider default)"
        )
        url = _ask(console, theme, label, d.url)
        if d.provider_type == "vllm" and url and (completed := vllm_chat_url(url)) != url:
            url = completed
            console.print(f"  URL: {_scrub_url(url)}", style=theme.style("meta"), highlight=False)
        d.url = url

    def ask_max_tokens(d: _Draft) -> None:
        d.max_tokens = _ask_max_tokens(console, theme, d.provider.max_tokens)

    def ask_context_window(d: _Draft) -> None:
        d.context_window = _ask_context_window(console, theme, d.provider.context_window)

    def keep_instructions(d: _Draft) -> None:
        d.keep_instructions = _keep(
            console,
            theme,
            "instructions",
            _source_paths(d.agent.standing_instructions),
            extra={
                "Agent name": [d.agent.name],
                "Extra instructions for the agent": (
                    [_short(d.agent.instruction)] if d.agent.instruction else []
                ),
            },
        )

    def choose_instructions(d: _Draft) -> None:
        if import_into is not None:
            d.agent.standing_instructions = _choose_imported_instructions(
                console, theme, d.agent.standing_instructions, discard=d.discard
            )
        else:
            d.agent.standing_instructions = _choose_instructions(
                console, theme, d.agent.standing_instructions
            )

    def choose_local_instructions(d: _Draft) -> None:
        d.agent.standing_instructions = _choose_local_instructions(
            console, theme, d.agent.standing_instructions, first_setup=first_setup
        )

    def ask_agent_name(d: _Draft) -> None:
        # Another agent of the file keeps its name: two agents named alike make the
        # entrypoint ambiguous. A name longer than `Agent` accepts would be saved and
        # then refused by the next `domyn run`.
        other_names = {
            str(a.get("name")) for a in (existing or {}).get("agents") or [] if isinstance(a, dict)
        } - {d.agent.name}
        default_name = AgentConfig().name
        while True:
            name = _ask(
                console,
                theme,
                f"Agent name (empty = {default_name})",
                d.agent.name,
                default=default_name,
            )
            if len(name) > MAX_AGENT_NAME_LENGTH:
                problem = (
                    f"{name} is {len(name)} characters long, at most "
                    f"{MAX_AGENT_NAME_LENGTH} are allowed — try a shorter name"
                )
            elif name in other_names:
                problem = f"Another agent of this config is called {name} — try another name"
            else:
                break
            console.print(
                f"  {problem}",
                style=theme.style("warning"),
                highlight=False,
                markup=False,  # the name is the user's, and may hold `[`
            )
        d.agent.name = name

    def ask_instruction(d: _Draft) -> None:
        d.agent.instruction = _ask(
            console,
            theme,
            "Extra instructions for the agent, be concise and accurate (empty = none)",
            d.agent.instruction,
        )

    def keep_tools(d: _Draft) -> None:
        d.keep_tools = _keep(console, theme, "tools", d.agent.tools)

    def choose_tools(d: _Draft) -> None:
        d.agent.tools = _choose_tools(
            console,
            theme,
            d.agent.tools,
            first_setup=first_setup,
            browser_binary=d.agent.browser.binary,
        )

    def keep_skills(d: _Draft) -> None:
        d.keep_skills = _keep(console, theme, "skills", _source_paths(d.agent.skills))

    def choose_listed_skills(d: _Draft) -> None:
        if import_into is not None:
            d.agent.skills = _choose_project_skills(console, theme, d.agent.skills, import_into)
        else:
            d.agent.skills = _choose_builtin_skills(console, theme, d.agent.skills)

    def choose_custom_skills(d: _Draft) -> None:
        d.agent.skills = _choose_custom_skills(console, theme, d.agent.skills)

    def choose_local_skills(d: _Draft) -> None:
        d.agent.skills = _choose_local_skills(
            console, theme, d.agent.skills, first_setup=first_skills
        )

    def ask_decisions(d: _Draft) -> None:
        d.decisions = _ask_decisions(console, theme, current_decisions_from(existing))

    def asks_model(d: _Draft) -> bool:
        return not d.keep_model

    def asks_domyn(d: _Draft) -> bool:
        return not d.keep_model and d.provider_type == "domyn"

    def asks_other(d: _Draft) -> bool:
        return not d.keep_model and d.provider_type != "domyn"

    model_steps = [
        _Step("keep model", keep_model, lambda d: ask_keep_model),
        _Step("provider", choose_provider, asks_model),
        _Step("model", ask_model, asks_other),
        _Step("platform url", ask_platform_url, asks_domyn),
        _Step("space id", ask_space_id, asks_domyn),
        _Step("configuration id", ask_configuration_id, asks_domyn),
        # The connection comes before the model: it is what the space's models are asked with.
        _Step("domyn api key", lambda d: _ensure_api_key(console, theme, domyn_key), asks_domyn),
        _Step("domyn model", choose_domyn_model, asks_domyn),
        _Step("url", ask_url, asks_other),
        _Step(
            "api key",
            lambda d: _ensure_api_key(
                console,
                theme,
                ProviderConfig(
                    type=d.provider_type,
                    api_key_env=PROVIDER_HINTS.get(d.provider_type, {}).get("env", "LLM_API_KEY"),
                ),
            ),
            asks_other,
        ),
        _Step("max tokens", ask_max_tokens, asks_model),
        _Step("context window", ask_context_window, asks_model),
    ]
    section_steps = [
        _Step("keep instructions", keep_instructions, lambda d: ask_keep),
        _Step("agent name", ask_agent_name, lambda d: not d.keep_instructions),
        _Step("instructions", choose_instructions, lambda d: not d.keep_instructions),
        _Step(
            "instruction autodiscovery",
            choose_local_instructions,
            lambda d: not d.keep_instructions,
        ),
        _Step("extra instructions", ask_instruction, lambda d: not d.keep_instructions),
        _Step("keep tools", keep_tools, lambda d: ask_keep),
        _Step("tools", choose_tools, lambda d: not d.keep_tools),
        _Step("keep skills", keep_skills, lambda d: ask_keep),
        _Step("skills", choose_listed_skills, lambda d: not d.keep_skills),
        _Step("custom skills", choose_custom_skills, lambda d: not d.keep_skills),
        _Step("skill autodiscovery", choose_local_skills, lambda d: not d.keep_skills),
        _Step("decisions", ask_decisions),
    ]
    steps = model_steps if provider_only else [*model_steps, *section_steps]

    draft = _Draft(provider=provider, agent=agent)
    recall = _Recall()
    recall_token = _recall.set(recall)
    # On a terminal every row is counted, so going back erases up to the step gone back
    # to and asks it in its place. The caller's console is left as it was.
    screen = None
    if interactive_terminal() and console.is_terminal is True:
        from rich.console import Console

        screen = _Screen(console.file)
        console = Console(
            file=cast(IO[str], screen),
            force_terminal=True,
            color_system=console.color_system,  # type: ignore[arg-type]
        )
    screen_token = _screen.set(screen)
    # Each step that asked something, with the draft as it was before it. Ctrl-B pops back
    # to the one before and restores its draft, so a step asked again never sees its
    # own earlier answer in the draft (a custom skills folder would be listed twice):
    # only in its prompts, through `_Recall`.
    history: list[tuple[int, _Draft, int]] = []
    index = 0
    try:
        with back_key():
            while index < len(steps):
                step = steps[index]
                if not step.applies(draft):
                    index += 1
                    continue
                history.append((index, copy.deepcopy(draft), _screen_mark()))
                recall.step = step.name
                asked = recall.asked
                try:
                    step.run(draft)
                except GoBack:
                    recall.settle()
                    own = history.pop()
                    # The first question has nothing before it, and is asked again.
                    index, draft, mark = history.pop() if history else own
                    if screen is not None:
                        screen.erase_to(mark)
                    else:
                        arrow = "\u2190" if theme.capabilities.unicode else "<-"
                        console.print(f"  {arrow} back", style=theme.style("meta"), highlight=False)
                    continue
                recall.settle()
                if recall.asked == asked:
                    history.pop()  # nothing to go back to
                index += 1
    except (EOFError, KeyboardInterrupt):
        console.print("  Aborted, nothing saved.", style=theme.style("warning"), highlight=False)
        return None
    finally:
        _recall.reset(recall_token)
        _screen.reset(screen_token)

    agent = draft.agent
    new_provider = provider
    if not draft.keep_model:
        new_provider = ProviderConfig(
            type=draft.provider_type,
            model_name=draft.model_name,
            url=draft.url or None,
            space_id=draft.space_id or None,
            configuration_id=draft.configuration_id or None,
            api_key_env=PROVIDER_HINTS.get(draft.provider_type, {}).get("env", "LLM_API_KEY"),
        ).model_copy(
            update={"max_tokens": draft.max_tokens, "context_window": draft.context_window}
        )
    # `None` unless the wizard actually asked, so a `provider_only` run leaves
    # whatever the file says about decisions exactly where it found it.
    config_dict = configuration_v1_dict_from(
        new_provider, agent, compaction, existing, draft.decisions
    )

    target = path or config_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    # A project names its own files with ${DOMYN_PROJECT_DIR}, so the folder still
    # works moved, renamed or exported. Only when saving into the project itself.
    if project is not None and target.parent.resolve() == project.resolve():
        config_dict = in_project_terms(config_dict, project)
    target.write_text(yaml.safe_dump(config_dict, sort_keys=False), encoding="utf-8")
    if import_into is not None:
        # A file listed again after it was unlisted (the instructions question answered
        # with the project's own folder) is in the config just saved: deleting it would
        # leave the config naming a file that is gone.
        listed = [
            Path(os.path.abspath(Path(entry).expanduser()))
            for entry in _source_paths(agent.standing_instructions)
        ]
        for discarded in draft.discard:
            discarded_path = Path(os.path.abspath(discarded.expanduser()))
            if any(
                one.is_relative_to(discarded_path) or discarded_path.is_relative_to(one)
                for one in listed
            ):
                continue
            _discard_imported(discarded, import_into)

    console.print()
    console.print(f"  Saved {target}", style=theme.style("accent"), highlight=False)
    if not is_configuration_v1_usable(config_dict):
        if not new_provider.model_name:
            missing = "model"
        elif new_provider.type == "domyn" and new_provider.url:
            missing = "space id"
        else:
            missing = "url"
        console.print(
            f"  Incomplete: {missing} is still required before the agent can run",
            style=theme.style("warning"),
            highlight=False,
        )
    console.print()
    return config_dict


def render_config(console: Console, theme: Theme, config: dict[str, Any] | None) -> None:
    """Show the active configuration. Secrets are shown as present/absent only."""
    if config is None:
        console.print(
            f"  No configuration yet. Run `domyn setup` (will be written to {config_path()}).",
            style=theme.style("meta"),
            highlight=False,
        )
        return

    description = describe_configuration_v1(config)
    print_config_description(console, theme, description)
    console.print(
        f"\n  Secrets live in {credentials_path()} · /model to change model\n",
        style=theme.style("meta"),
        highlight=False,
    )
