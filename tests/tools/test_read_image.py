"""`read_image`: an image answered about in text, by the agent's own model.

The image goes to the provider in a request of its own and never enters the
conversation, so what is tested is what the provider is sent, what comes back
as the tool's result, and that the tool exists only for a model that reads images.
"""

# No `from __future__ import annotations`: FunctionTool reads signatures at construction.

import base64
import io
import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import httpx
import pytest
from PIL import Image
from pydantic import Field

from domyn.agents.llm.base import BaseLLMProvider, LLMMessage, LLMResponse
from domyn.agents.tools.builtin import (
    BUNDLE_TOOL_NAMES,
    VISION_TOOL_NAMES,
    BuiltinBundleConfig,
    Workspace,
    bundle,
    images,
)
from domyn.agents.tools.builtin.backends import CommandBackend, CommandResult, ToolHost
from domyn.agents.tools.builtin.files import serve


class Model(BaseLLMProvider):
    """A provider that records what it is asked and answers what it is told to."""

    model_name: str = "m"
    supported_structured_outputs: list[Any] | None = None
    asked: list[list[LLMMessage]] = Field(default_factory=list)
    answer: str = "A red square."
    finish_reason: str | None = None
    failure: Any = None

    async def generate(self, messages: list[LLMMessage], *args: Any, **kwargs: Any) -> LLMResponse:
        self.asked.append(messages)
        if self.failure is not None:
            raise self.failure
        return LLMResponse(
            content=self.answer,
            generated_tokens=1,
            execution_time=0.0,
            finish_reason=self.finish_reason,
        )

    async def generate_streaming(self, messages: list[LLMMessage], **kwargs: Any) -> Any:
        raise NotImplementedError


def _picture(path: Path, size: tuple[int, int] = (8, 8), mode: str = "RGB", **save: Any) -> Path:
    Image.new(mode, size, "red" if mode == "RGB" else None).save(path, **save)
    return path


@pytest.fixture
def workspace(tmp_path: Path) -> Workspace:
    _picture(tmp_path / "shot.png")
    (tmp_path / "notes.md").write_text("# Notes\n", encoding="utf-8")
    (tmp_path / "folder").mkdir()
    return Workspace.at(tmp_path)


@pytest.fixture
def model() -> Model:
    return Model(vision=True)


async def _read(workspace: Workspace, model: Model, path: str, query: str = "What is it?") -> str:
    tools = {t.name: t for t in bundle("files", workspace=workspace, llm_provider=model)}
    result = await tools["read_image"].execute(path=path, query=query)
    return str(result.data) if result.success else f"ERROR: {result.error}"


class _Elsewhere(CommandBackend):
    """A backend whose tool bodies run "elsewhere": each request goes through `run` and
    `serve` answers it, as the container's tool host does. Like `auto`, it knows where
    the bodies run only once started."""

    name = "elsewhere"

    def __init__(self, *, hosted: bool = True, fail: CommandResult | None = None) -> None:
        self.hosted = hosted
        self.fail = fail
        self.started = False
        self.requests: list[tuple[str, dict[str, Any]]] = []

    @property
    def tool_host(self) -> ToolHost | None:
        return ToolHost("tool-host", roots=(Path("/"),)) if self.started and self.hosted else None

    async def start(self) -> None:
        self.started = True

    async def run(
        self, command: str, *, timeout: float, conversation_id: str = "", stdin: str | None = None
    ) -> CommandResult:
        if self.fail is not None:
            return self.fail
        request = json.loads(stdin or "{}")
        self.requests.append((conversation_id, request))
        return CommandResult(exit_code=0, stdout=json.dumps(serve(request)) + "\n", stderr="")


async def _read_through(backend: _Elsewhere, workspace: Workspace, model: Model, path: str) -> str:
    tools = {
        t.name: t for t in bundle("files", workspace=workspace, backend=backend, llm_provider=model)
    }
    result = await tools["read_image"].execute(path=path, query="What is it?", conversation_id="c1")
    return str(result.data) if result.success else f"ERROR: {result.error}"


def _sent(model: Model) -> tuple[str, str, bytes]:
    """The question, the media type and the image bytes of the one request made."""
    (messages,) = model.asked
    (message,) = messages
    text, picture = message.content
    header, data = picture["image_url"]["url"].split(",", 1)
    return (
        text["text"],
        header.removeprefix("data:").removesuffix(";base64"),
        base64.b64decode(data),
    )


class TestWhoGetsTheTool:
    def test_a_model_that_reads_images(self, workspace: Workspace, model: Model) -> None:
        names = [t.name for t in bundle("files", workspace=workspace, llm_provider=model)]

        assert names == [*BUNDLE_TOOL_NAMES["files"], *VISION_TOOL_NAMES]

    def test_not_a_model_that_does_not(self, workspace: Workspace) -> None:
        names = [t.name for t in bundle("files", workspace=workspace, llm_provider=Model())]

        assert names == list(BUNDLE_TOOL_NAMES["files"])

    def test_not_without_a_provider(self, workspace: Workspace) -> None:
        assert "read_image" not in [t.name for t in bundle("files", workspace=workspace)]

    def test_not_a_mocked_provider(self, workspace: Workspace) -> None:
        """A mock answers `.vision` with something truthy, as it answers everything."""
        mocked = MagicMock(spec=BaseLLMProvider)

        names = [t.name for t in bundle("files", workspace=workspace, llm_provider=mocked)]

        assert "read_image" not in names

    def test_not_without_the_files_bundle(self, workspace: Workspace, model: Model) -> None:
        names = [t.name for t in bundle("shell", workspace=workspace, llm_provider=model)]

        assert "read_image" not in names

    def test_a_bundle_entry_built_for_an_agent(self, tmp_path: Path, model: Model) -> None:
        entry = BuiltinBundleConfig(name="b", bundles=["files"], workspace=str(tmp_path))

        assert "read_image" in [t.name for t in entry.build(llm_provider=model)]
        assert "read_image" not in [t.name for t in entry.build()]

    def test_it_is_not_gated(self, workspace: Workspace, model: Model) -> None:
        tools = {t.name: t for t in bundle("files", workspace=workspace, llm_provider=model)}

        assert tools["read_image"].approval_type is None

    def test_the_model_is_offered_a_path_and_a_query(
        self, workspace: Workspace, model: Model
    ) -> None:
        tools = {t.name: t for t in bundle("files", workspace=workspace, llm_provider=model)}

        parameters = tools["read_image"].tool_definition["function"]["parameters"]

        assert set(parameters["properties"]) == {"path", "query"}


class TestReading:
    @pytest.mark.asyncio
    async def test_the_answer_is_the_result(self, workspace: Workspace, model: Model) -> None:
        assert await _read(workspace, model, "shot.png") == "A red square."

    @pytest.mark.asyncio
    async def test_the_model_gets_the_question_and_the_file_as_it_is(
        self, workspace: Workspace, model: Model, tmp_path: Path
    ) -> None:
        await _read(workspace, model, "shot.png", "What colour is it?")

        question, media_type, sent = _sent(model)
        assert question.startswith("What colour is it?")
        assert media_type == "image/png"
        assert sent == (tmp_path / "shot.png").read_bytes()

    @pytest.mark.asyncio
    async def test_an_empty_query_asks_for_a_description(
        self, workspace: Workspace, model: Model
    ) -> None:
        await _read(workspace, model, "shot.png", "  ")

        assert _sent(model)[0].startswith("Describe the image")

    @pytest.mark.asyncio
    async def test_a_large_image_is_scaled_down(
        self, workspace: Workspace, model: Model, tmp_path: Path
    ) -> None:
        _picture(tmp_path / "wide.png", size=(images.MAX_IMAGE_EDGE * 2, 100))

        await _read(workspace, model, "wide.png")

        _, media_type, sent = _sent(model)
        assert media_type == "image/jpeg"
        assert Image.open(io.BytesIO(sent)).size == (images.MAX_IMAGE_EDGE, 50)

    @pytest.mark.asyncio
    async def test_a_format_not_every_provider_takes_is_converted(
        self, workspace: Workspace, model: Model, tmp_path: Path
    ) -> None:
        _picture(tmp_path / "scan.bmp")

        await _read(workspace, model, "scan.bmp")

        _, media_type, sent = _sent(model)
        assert media_type == "image/jpeg"
        assert Image.open(io.BytesIO(sent)).format == "JPEG"

    @pytest.mark.asyncio
    async def test_a_transparent_area_becomes_white_not_black(
        self, workspace: Workspace, model: Model, tmp_path: Path
    ) -> None:
        """Black lines on a transparent background would otherwise be sent as a black
        rectangle: JPEG has no transparency."""
        _picture(tmp_path / "diagram.tiff", mode="RGBA")

        await _read(workspace, model, "diagram.tiff")

        red, green, blue = Image.open(io.BytesIO(_sent(model)[2])).getpixel((4, 4))
        assert min(red, green, blue) > 240

    @pytest.mark.asyncio
    async def test_a_cut_answer_says_so(self, workspace: Workspace, model: Model) -> None:
        model.finish_reason = "length"

        assert "cut at the model's output limit" in await _read(workspace, model, "shot.png")


class TestRefusals:
    @pytest.mark.asyncio
    async def test_outside_the_workspace(self, workspace: Workspace, model: Model) -> None:
        answer = await _read(workspace, model, "../elsewhere.png")

        assert "outside the workspace" in answer
        assert model.asked == []

    @pytest.mark.asyncio
    async def test_a_file_that_does_not_exist(self, workspace: Workspace, model: Model) -> None:
        assert "does not exist" in await _read(workspace, model, "missing.png")

    @pytest.mark.asyncio
    async def test_a_directory(self, workspace: Workspace, model: Model) -> None:
        assert "is a directory" in await _read(workspace, model, "folder")

    @pytest.mark.asyncio
    async def test_a_file_that_is_not_an_image(self, workspace: Workspace, model: Model) -> None:
        answer = await _read(workspace, model, "notes.md")

        assert "not an image" in answer
        assert "read_file" in answer
        assert model.asked == []

    @pytest.mark.asyncio
    async def test_an_image_cut_short(
        self, workspace: Workspace, model: Model, tmp_path: Path
    ) -> None:
        """Decoded only when it has to be converted, which is when a damaged file shows."""
        whole = _picture(tmp_path / "whole.bmp", size=(64, 64)).read_bytes()
        (tmp_path / "cut.bmp").write_bytes(whole[: len(whole) // 2])

        assert "damaged or cut short" in await _read(workspace, model, "cut.bmp")


class TestWhenTheModelFails:
    @pytest.mark.asyncio
    async def test_a_refusal_says_retrying_will_not_help(
        self, workspace: Workspace, model: Model
    ) -> None:
        request = httpx.Request("POST", "http://llm/v1/chat/completions")
        model.failure = httpx.HTTPStatusError(
            "400", request=request, response=httpx.Response(400, text="secret body")
        )

        answer = await _read(workspace, model, "shot.png")

        assert "HTTP 400" in answer
        assert "Retrying will not help" in answer
        assert "secret body" not in answer

    @pytest.mark.asyncio
    async def test_an_unreachable_model_is_named_by_what_happened(
        self, workspace: Workspace, model: Model
    ) -> None:
        model.failure = httpx.ConnectError("refused")

        answer = await _read(workspace, model, "shot.png")

        assert "ConnectError" in answer
        assert "Retrying will not help" not in answer

    @pytest.mark.asyncio
    async def test_no_answer_is_an_error(self, workspace: Workspace, model: Model) -> None:
        model.answer = "  "

        assert "gave no answer" in await _read(workspace, model, "shot.png")


class TestThroughABackend:
    """Loaded where the backend runs the file tools; the model is asked from here."""

    @pytest.mark.asyncio
    async def test_the_image_is_loaded_where_the_backend_runs_the_tools(
        self, workspace: Workspace, model: Model, tmp_path: Path
    ) -> None:
        backend = _Elsewhere()

        assert await _read_through(backend, workspace, model, "shot.png") == "A red square."

        ((conversation, request),) = backend.requests
        assert conversation == "c1"
        assert request["operation"] == "load_image"
        assert request["arguments"] == {"path": "shot.png"}
        assert request["extra_roots"][-1] == "/"
        assert _sent(model)[1:] == ("image/png", (tmp_path / "shot.png").read_bytes())

    @pytest.mark.asyncio
    async def test_a_backend_that_runs_the_tools_here_loads_here(
        self, workspace: Workspace, model: Model
    ) -> None:
        """`auto` after falling back, or the local backend."""
        backend = _Elsewhere(hosted=False)

        assert await _read_through(backend, workspace, model, "shot.png") == "A red square."
        assert backend.started and backend.requests == []

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("path", "expected"),
        [
            # No "outside the workspace": where the backend runs the tools, `/` is a
            # root, and the boundary is what the backend mounts.
            ("missing.png", "does not exist"),
            ("folder", "is a directory"),
            ("notes.md", "not an image this tool can open"),
        ],
    )
    async def test_a_refusal_comes_back_as_written(
        self, workspace: Workspace, model: Model, path: str, expected: str
    ) -> None:
        answer = await _read_through(_Elsewhere(), workspace, model, path)

        assert expected in answer
        # Not an exception's name, which `serve` puts before anything but a refusal.
        assert "Error:" not in answer
        assert model.asked == []

    @pytest.mark.asyncio
    async def test_a_backend_that_failed_is_an_error_the_model_can_read(
        self, workspace: Workspace, model: Model
    ) -> None:
        failed = CommandResult(exit_code=None, stdout="", stderr="", backend_error="box is gone")

        answer = await _read_through(_Elsewhere(fail=failed), workspace, model, "shot.png")

        assert "box is gone" in answer
        assert model.asked == []
