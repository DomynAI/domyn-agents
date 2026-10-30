from collections.abc import AsyncGenerator, Coroutine, Generator
from typing import Any

import pytest

from domyn.agents.agents import Agent
from domyn.agents.core import BaseEvent, ExecutionEventType, Part, ToolAction
from domyn.agents.llm import BaseLLMProvider, LLMMessage, LLMResponse


class MockLLMProvider(BaseLLMProvider):
    type: str = "mock"
    name: str = "mock-llm"
    model_name: str = "mock-model"
    supported_structured_outputs: None = None

    async def generate(  # type: ignore
        self, messages: list[LLMMessage], **kwargs
    ) -> LLMResponse:
        return LLMResponse(
            content="mock response",
            generated_tokens=0,
            execution_time=0.0,
        )

    async def generate_streaming(  # type: ignore
        self, messages: list[LLMMessage], **kwargs
    ) -> AsyncGenerator[LLMResponse, None]:
        yield LLMResponse(
            content="mock response",
            generated_tokens=0,
            execution_time=0.0,
        )


@pytest.fixture
def llm_provider() -> BaseLLMProvider:
    return MockLLMProvider()


@pytest.fixture
def agent(llm_provider: BaseLLMProvider) -> Agent:
    return Agent(
        name="TestAgent",
        description="A test agent",
        instruction="Help with tests",
        llm_provider=llm_provider,
    )


@pytest.fixture
def child_agent(llm_provider: BaseLLMProvider) -> Agent:
    return Agent(
        name="ChildAgent",
        description="A child sub-agent",
        instruction="Sub-agent instruction",
        llm_provider=llm_provider,
    )


@pytest.fixture
def response_event() -> BaseEvent:
    return BaseEvent(
        event_type=ExecutionEventType.RESPONSE,
        author="TestAgent",
        content=[Part(text="Hello from agent!")],
        is_partial=False,
    )


@pytest.fixture
def tool_start_event() -> BaseEvent:
    return BaseEvent(
        event_type=ExecutionEventType.TOOL_START,
        author="TestAgent",
        action=ToolAction(
            name="my_tool", parameters={"key": "value"}, thought="I will call my_tool"
        ),
    )


@pytest.fixture
def tool_end_event() -> BaseEvent:
    return BaseEvent(
        event_type=ExecutionEventType.TOOL_END,
        author="TestAgent",
        action=ToolAction(name="my_tool", parameters={}, observation="tool result"),
        content=[Part(text="Result: tool result")],
    )


@pytest.fixture(autouse=True)
def _approvals_off_the_real_home(
    tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Keep persisted approvals out of the developer's ``~/.domyn``.

    Autouse and unconditional. Building a ``Repl`` without an explicit runner
    wires the disk-backed stores — that is the product behaviour — so any test
    that constructs one would otherwise read, and on approval *write*, the real
    user's standing permissions. A test suite that can grant itself a permission
    on the machine running it is not a test suite anyone should run twice.
    """
    from domyn.agents.interfaces.cli import config as cli_config

    root = tmp_path_factory.mktemp("approvals")
    # `--project` selects a folder for the rest of the process; a test invoking a
    # command with it must not leave the next test reading that project.
    monkeypatch.setattr(cli_config, "_active_project", None)
    # And the variable it sets for the project's config: put back as it was, by
    # recording it first (a `delenv` of an unset variable records nothing to undo).
    monkeypatch.setenv(cli_config.PROJECT_DIR_VAR, "")
    monkeypatch.delenv(cli_config.PROJECT_DIR_VAR)
    # Likewise what the credentials loader remembers having put in the environment.
    monkeypatch.setattr(cli_config, "_credentials_in_environment", {})
    monkeypatch.setattr(
        cli_config,
        "approvals_paths",
        lambda: {"user": root / "user.yaml", "project": root / "project.yaml"},
    )
    # A Repl built without a runner also wires the persistent conversation store,
    # which is the product behaviour and would otherwise leave every test's
    # transcript in the real history — and offer it back on the next /resume.
    from domyn.agents.interfaces.cli import conversations

    monkeypatch.setattr(conversations, "store_path", lambda: root / "conversations.db")
    # `domyn setup` replaces the whole skills directory from scratch, so a test that
    # reaches the wizard without choosing a location would delete the developer's
    # own `~/.domyn/builtin-skills`. Patched where it is defined and where the wizard
    # imported it.
    from domyn.agents.interfaces.cli import export as cli_export
    from domyn.agents.interfaces.cli import setup as cli_setup

    for module in (cli_config, cli_setup, cli_export):
        monkeypatch.setattr(
            module, "skills_install_dir", lambda: root / cli_config.BUILTIN_SKILLS_DIRNAME
        )
    # Setup also asks whether to load `./.domyn/skills` and `./.agents/skills` from
    # wherever domyn runs, every time: a prompt every scripted wizard answer would then be
    # off by one for. The step is skipped here, keeping the config as it was; the tests of
    # that step put it back. And those folders for a test run are this repository's, so
    # they are pointed at absolute folders that do not exist, for the entries to load
    # nothing.
    from domyn.agents.interfaces.cli import repl as cli_repl

    monkeypatch.setattr(
        cli_setup, "_choose_local_skills", lambda console, theme, current, **_: current
    )
    # The same for the question asking for a folder of the user's own skills, asked
    # before it. `test_custom_skills_setup.py` puts it back.
    monkeypatch.setattr(
        cli_setup, "_choose_custom_skills", lambda console, theme, current, **_: current
    )
    # The same for the agent's name and the extra instructions, asked first and last in
    # the instructions section: the current ones are kept. `test_agent_name_setup.py` and
    # `test_instruction_setup.py` put the real questions back.
    real_ask = cli_setup._ask

    def ask(console: Any, theme: Any, label: str, current: str = "", **kwargs: Any) -> str:
        if label.startswith(("Agent name", "Extra instructions for the agent")):
            return current
        return real_ask(console, theme, label, current, **kwargs)

    monkeypatch.setattr(cli_setup, "_ask", ask)
    # The same for the instructions path, asked after the provider: the step keeps what
    # the config already declares. `test_instructions_setup.py` puts it back.
    monkeypatch.setattr(
        cli_setup, "_choose_instructions", lambda console, theme, current, **_: current
    )
    # The same for the instructions autodiscovery question asked after it, and for the
    # same reason as the skills one its folders are pointed where nothing exists: this
    # repository has a `.domyn` of its own. `test_local_instructions_setup.py` puts
    # both back.
    monkeypatch.setattr(
        cli_setup, "_choose_local_instructions", lambda console, theme, current, **_: current
    )
    for module in (cli_config, cli_setup):
        monkeypatch.setattr(
            module,
            "AUTODISCOVERY_DIRS",
            (root / "no-local-instructions" / ".domyn", root / "no-local-instructions" / ".agents"),
        )
    # The same for the tool bundles, asked after the instructions path: the step keeps
    # what the config already mounts. `test_tool_bundles_setup.py` puts it back.
    monkeypatch.setattr(cli_setup, "_choose_tools", lambda console, theme, current, **_: current)
    # A re-run asks, before each section, whether to keep it: answered no here, so a
    # scripted re-run goes through every section's questions as it did before the
    # question existed. `test_keep_sections_setup.py` puts it back.
    monkeypatch.setattr(cli_setup, "_keep", lambda *_, **__: False)
    nowhere = root / "no-local-skills"
    for module in (cli_config, cli_setup, cli_repl):
        monkeypatch.setattr(
            module,
            "LOCAL_SKILLS_DIRS",
            (nowhere / ".domyn" / "skills", nowhere / ".agents" / "skills"),
        )


@pytest.fixture(autouse=True)
def no_decision_prompt(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    """The wizard's decision-model step keeps its default, so the scripted answers of
    the other setup tests stay about what they test. test_setup_decisions.py covers it."""
    if request.module.__name__.endswith("test_setup_decisions"):
        return
    from domyn.agents.interfaces.cli import setup
    from domyn.agents.interfaces.cli.config import DecisionsChoice

    monkeypatch.setattr(setup, "_ask_decisions", lambda console, theme, current: DecisionsChoice())
