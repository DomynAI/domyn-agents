"""An image attached in the editor, through the real REPL and runner: the image is
copied into the conversation's attachments folder, the message names it, and
``read_image`` can read it for that conversation."""

import io
from pathlib import Path
from typing import Any

import pytest
from PIL import Image
from pydantic import Field
from rich.console import Console

from domyn.agents import Agent
from domyn.agents.interfaces.cli import Theme
from domyn.agents.interfaces.cli import config as cli_config
from domyn.agents.interfaces.cli.attachments import AttachmentError
from domyn.agents.interfaces.cli.prompt import InputSession
from domyn.agents.interfaces.cli.repl import Repl
from domyn.agents.llm import BaseLLMProvider, LLMMessage, LLMResponse
from domyn.agents.planner_strategy import ToolUsePlannerStrategy
from domyn.agents.tools.builtin import bundle
from tests.cli.test_repl_session import ScriptedEditor


class Model(BaseLLMProvider):
    """Answers at once, and records what it was sent."""

    type: str = "scripted"
    model_name: str = "scripted"
    supported_structured_outputs: list[Any] | None = None
    asked: list[list[LLMMessage]] = Field(default_factory=list)

    async def generate(self, messages: list[LLMMessage], **kwargs: Any) -> LLMResponse:
        self.asked.append(messages)
        return LLMResponse(content="A red square.", generated_tokens=1, execution_time=0.0)

    async def generate_streaming(self, messages: list[LLMMessage], **kwargs: Any) -> Any:
        raise NotImplementedError
        yield  # pragma: no cover


class DroppingEditor(ScriptedEditor):
    """Plays back a script in which a ``Path`` is an image dropped into the field: it
    is attached as the editor attaches one, and its chip goes in front of the next
    line. A refusal is kept in ``notices``, where the editor would draw it."""

    def __init__(self, session: InputSession, *script: str | Path) -> None:
        super().__init__()
        self._session = session
        self._script = list(script)
        self.notices: list[str] = []

    async def prompt_async(self, *args: Any, **kwargs: Any) -> str:
        chips = ""
        while self._script and isinstance(self._script[0], Path):
            attachments = self._session.attachments
            assert attachments is not None
            try:
                chips += attachments.add_file(self._script.pop(0)) + " "
            except AttachmentError as e:
                self.notices.append(str(e))
        if not self._script:
            raise EOFError
        return chips + str(self._script.pop(0))


def _repl(tmp_path: Path, *script: str | Path, vision: bool = True) -> tuple[Repl, Model]:
    model = Model(vision=vision)
    work = tmp_path / "work"
    work.mkdir(exist_ok=True)
    built = bundle("files", root=work, llm_provider=model)
    agent = Agent(
        name="Assistant",
        description="test agent",
        instruction="Be brief.",
        llm_provider=model,
        planner=ToolUsePlannerStrategy(),
        tools=list(built),
        builtin_tools=[built],
    )
    repl = Repl(
        agent,
        Console(file=io.StringIO(), width=200, force_terminal=False),
        Theme.build(allow_animation=False),
        enable_enhanced_input=False,
    )
    repl._input._session = DroppingEditor(repl._input, *script)
    return repl, model


def _picture(path: Path) -> Path:
    Image.new("RGB", (8, 8), "red").save(path)
    return path


def _user_text(model: Model, request: int = 0) -> str:
    """The user messages of one request to the model, the first by default."""
    return "\n".join(str(m.content) for m in model.asked[request] if m.role == "user")


@pytest.fixture(autouse=True)
def state_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / "home"
    monkeypatch.setenv("DOMYN_PROJECT_DIR", str(home))
    return home


class TestAnAttachedImage:
    @pytest.mark.asyncio
    async def test_the_message_names_the_attached_copy(
        self, tmp_path: Path, state_dir: Path
    ) -> None:
        shot = _picture(tmp_path / "shot.png")
        repl, model = _repl(tmp_path, shot, "what is this?")
        assert await repl.run() == 0

        folder = state_dir / "conversations" / repl._state.conversation_id / "attachments"
        [copy] = folder.iterdir()
        sent = _user_text(model)
        assert f"[Image #1, attached by the user: {copy}" in sent
        assert sent.endswith("what is this?")

    @pytest.mark.asyncio
    async def test_read_image_can_read_it_in_that_conversation(
        self, tmp_path: Path, state_dir: Path
    ) -> None:
        shot = _picture(tmp_path / "shot.png")
        repl, _ = _repl(tmp_path, shot, "what is this?")
        await repl.run()

        folder = state_dir / "conversations" / repl._state.conversation_id / "attachments"
        [copy] = folder.iterdir()
        tool = repl._agent.get_tool_by_name("read_image")
        assert tool is not None
        result = await tool.execute(
            path=str(copy), query="What is it?", conversation_id=repl._state.conversation_id
        )
        assert result.success, result.error
        assert result.data == "A red square."

    @pytest.mark.asyncio
    async def test_one_in_the_workspace_is_named_in_place(
        self, tmp_path: Path, state_dir: Path
    ) -> None:
        (tmp_path / "work" / "docs").mkdir(parents=True)
        shot = _picture(tmp_path / "work" / "docs" / "shot.png")
        repl, model = _repl(tmp_path, shot, "what is this?")
        assert await repl.run() == 0

        assert not (state_dir / "conversations" / repl._state.conversation_id).exists()
        sent = _user_text(model)
        assert f"[Image #1, attached by the user, in the workspace: {shot.resolve()}." in sent

        tool = repl._agent.get_tool_by_name("read_image")
        assert tool is not None
        result = await tool.execute(
            path=str(shot.resolve()),
            query="What is it?",
            conversation_id=repl._state.conversation_id,
        )
        assert result.success, result.error
        assert result.data == "A red square."

    @pytest.mark.asyncio
    async def test_one_below_the_global_folder_is_copied(
        self, tmp_path: Path, state_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A container whose workspace holds ~/.domyn hides it."""
        global_dir = tmp_path / "work" / ".domyn"
        global_dir.mkdir(parents=True)
        monkeypatch.setattr(cli_config, "GLOBAL_DIR", global_dir)
        shot = _picture(global_dir / "shot.png")
        repl, model = _repl(tmp_path, shot, "what is this?")
        await repl.run()

        folder = state_dir / "conversations" / repl._state.conversation_id / "attachments"
        [copy] = folder.iterdir()
        assert f"[Image #1, attached by the user: {copy}." in _user_text(model)

    @pytest.mark.asyncio
    async def test_refused_for_a_model_that_does_not_read_images(self, tmp_path: Path) -> None:
        shot = _picture(tmp_path / "shot.png")
        repl, model = _repl(tmp_path, shot, "hello", vision=False)
        await repl.run()

        editor = repl._input._session
        assert isinstance(editor, DroppingEditor)
        assert any("cannot look at images" in notice for notice in editor.notices)
        assert "attached by the user" not in _user_text(model)

    @pytest.mark.asyncio
    async def test_clear_forgets_the_attachments(self, tmp_path: Path) -> None:
        """A chip from the conversation left behind names nothing in the new one."""
        shot = _picture(tmp_path / "shot.png")
        repl, model = _repl(tmp_path, shot, "about it", "/clear", "hello [Image #1]")
        await repl.run()
        assert "attached by the user" in _user_text(model, 0)
        assert _user_text(model, -1).endswith("hello [Image #1]")
