"""Tests for the built-in tools.

Weighted towards containment and towards the tools' behaviour when the model is
wrong about the world, because those are the two ways they cause damage. A tool
that works on the happy path and silently writes outside the workspace, or
silently edits the wrong occurrence, is worse than no tool.

The shell tests run real commands. Mocking the subprocess would test the mock:
the interesting parts — the working directory, the timeout actually killing
something, the exit code coming back — only exist in the real thing.
"""

# No `from __future__ import annotations`: FunctionTool resolves tool signatures
# at construction and cannot read string annotations.
import asyncio
import io
import re
import zipfile
import zlib
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from domyn.agents.core import (
    BaseEvent,
    ExecutionEventType,
    InteractionContext,
    Part,
    ToolAction,
    documents,
)
from domyn.agents.planner_strategy import ToolUseContextManager
from domyn.agents.tools.builtin import Workspace, WorkspaceError, bundle, is_ignored
from domyn.agents.tools.builtin.files import MAX_READ_CHARS


@pytest.fixture
def workspace(tmp_path: Path) -> Workspace:
    (tmp_path / "app.py").write_text("print('hello')\nprint('world')\n")
    (tmp_path / "notes.md").write_text("# Notes\nsomething to find\n")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "lib.py").write_text("def add(a, b):\n    return a + b\n")
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "junk.py").write_text("something to find\n")
    return Workspace.at(tmp_path)


@pytest.fixture
def tools(workspace: Workspace) -> dict:
    return {t.name: t for t in bundle("files", "shell", workspace=workspace)}


async def _call(tools: dict, name: str, **kwargs: object) -> str:
    result = await tools[name].execute(**kwargs)
    return str(result.data) if result.success else f"ERROR: {result.error}"


async def _as_the_model_sees_it(observation: str) -> str:
    """A tool result as the default tool-use context manager hands it to the model."""
    agent = MagicMock()
    agent.name = "agent"
    agent.description = "An agent reading files"
    agent.instruction = "You are helpful."
    agent.tools = []
    agent.sub_agents = []
    agent.memory_manager = None
    agent.get_tool_by_name = lambda name: None
    agent.stream = False

    def event(event_type: ExecutionEventType, **fields: object) -> BaseEvent:
        return BaseEvent(
            event_type=event_type,
            author=agent.name,
            event_id=str(uuid4()),
            timestamp=datetime.now(),
            **fields,  # type: ignore[arg-type]
        )

    context = MagicMock(spec=InteractionContext)
    context.conversation_id = "conv"
    context.user_id = "user"
    context.interaction_id = "interaction"
    context.conversation_history = [
        event(
            ExecutionEventType.TOOL_END,
            content=[],
            action=ToolAction(
                call_id="call", name="read_file", parameters={}, observation=observation
            ),
        )
    ]
    context.interaction_history = []
    context.handoff_targets_description = []
    context.state = {}
    context.run_config = {}
    context.agent_input = event(ExecutionEventType.USER_INPUT, content=[Part(text="read it")])
    messages = await ToolUseContextManager().get_context(context=context, agent=agent)
    return str(next(m for m in messages if m.role == "tool").content)


class TestContainment:
    """The only structural protection these tools have."""

    @pytest.mark.parametrize(
        "path",
        [
            "../outside.txt",
            "../../etc/passwd",
            "/etc/passwd",
            "src/../../escape.txt",
            "~/.ssh/id_rsa",
        ],
    )
    def test_paths_leaving_the_workspace_are_refused(self, workspace: Workspace, path: str) -> None:
        with pytest.raises(WorkspaceError, match="outside the workspace"):
            workspace.resolve(path)

    def test_a_sibling_with_a_shared_prefix_is_outside(self, tmp_path: Path) -> None:
        """`/a/work` is a string prefix of `/a/work-secrets` and shares no
        directory with it. This is why containment resolves instead of
        comparing strings."""
        (tmp_path / "work").mkdir()
        (tmp_path / "work-secrets").mkdir()
        space = Workspace.at(tmp_path / "work")
        with pytest.raises(WorkspaceError):
            space.resolve(str(tmp_path / "work-secrets" / "keys.txt"))

    def test_a_symlink_pointing_out_is_refused(self, tmp_path: Path) -> None:
        """The case a prefix check cannot see: the path looks internal until it
        is resolved."""
        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / "secret.txt").write_text("secret")
        inside = tmp_path / "work"
        inside.mkdir()
        (inside / "link").symlink_to(outside)

        space = Workspace.at(inside)
        with pytest.raises(WorkspaceError):
            space.resolve("link/secret.txt")

    def test_the_root_itself_is_inside(self, workspace: Workspace) -> None:
        assert workspace.resolve(".") == workspace.root

    def test_a_new_file_is_checked_too(self, workspace: Workspace) -> None:
        """Writing has to be contained even though the path does not exist yet."""
        with pytest.raises(WorkspaceError):
            workspace.resolve("../new.txt")
        assert workspace.resolve("new.txt").parent == workspace.root

    @pytest.mark.asyncio
    async def test_the_tools_report_it_instead_of_crashing(self, tools: dict) -> None:
        """The message reaches the model, so it says what is allowed."""
        out = await _call(tools, "read_file", path="/etc/passwd")
        assert "outside the workspace" in out

    @staticmethod
    def _beside_a_secret(tmp_path: Path) -> dict:
        """A workspace next to a directory holding a secret, and its file tools."""
        (tmp_path / "outside").mkdir()
        (tmp_path / "outside" / "key.txt").write_text("the secret\n")
        inside = tmp_path / "work"
        inside.mkdir()
        (inside / "app.py").write_text("print('hello')\n")
        return {t.name: t for t in bundle("files", workspace=Workspace.at(inside))}

    @pytest.mark.asyncio
    async def test_a_glob_cannot_climb_out(self, tmp_path: Path) -> None:
        """`grep` and `glob_files` take no approval: a glob reaching past the
        root read and listed whatever the process could."""
        tools = self._beside_a_secret(tmp_path)
        for glob in ("../outside/*", "**/../../outside/*", "../**/*.txt"):
            assert "the secret" not in await _call(tools, "grep", pattern="secret", glob=glob)
            assert "key.txt" not in await _call(tools, "glob_files", pattern=glob)

    @pytest.mark.asyncio
    async def test_a_symlink_pointing_out_is_not_searched(self, tmp_path: Path) -> None:
        """As `read_file` refuses it. A cloned repository can carry one."""
        tools = self._beside_a_secret(tmp_path)
        inside = tmp_path / "work"
        (inside / "notes.txt").symlink_to(tmp_path / "outside" / "key.txt")
        (inside / "docs").symlink_to(tmp_path / "outside")
        for glob in ("**/*", "*.txt", "docs/*"):
            assert "the secret" not in await _call(tools, "grep", pattern="secret", glob=glob)
        assert "the secret" not in await _call(tools, "grep", pattern="secret", path="notes.txt")


class TestReadingAndSearching:
    @pytest.mark.asyncio
    async def test_read_file(self, tools: dict) -> None:
        assert "hello" in await _call(tools, "read_file", path="app.py")

    @pytest.mark.asyncio
    async def test_reading_a_directory_says_so(self, tools: dict) -> None:
        assert "list_dir" in await _call(tools, "read_file", path="src")

    @pytest.mark.asyncio
    async def test_a_binary_file_is_refused_clearly(
        self, tools: dict, workspace: Workspace
    ) -> None:
        (workspace.root / "blob.bin").write_bytes(b"\xff\xfe\x00\x01")
        assert "not UTF-8" in await _call(tools, "read_file", path="blob.bin")

    @pytest.mark.asyncio
    async def test_a_huge_single_line_is_cut(self, tools: dict, workspace: Workspace) -> None:
        """One line can be a minified bundle or a base64 blob and weigh
        megabytes. Without cutting it the line budget is no budget at all."""
        (workspace.root / "big.txt").write_text("x" * 200_000)
        out = await _call(tools, "read_file", path="big.txt")
        body, _, note = out.partition("\n\n[")
        assert len(body) == MAX_READ_CHARS
        assert f"start_char={MAX_READ_CHARS}" in note

    @pytest.mark.asyncio
    async def test_grep_reports_file_and_line(self, tools: dict) -> None:
        assert "notes.md:2:" in await _call(tools, "grep", pattern="something to find")

    @pytest.mark.asyncio
    async def test_grep_skips_vendored_directories(self, tools: dict) -> None:
        """Otherwise one search in a repo with a virtualenv spends its whole
        output on third-party code."""
        out = await _call(tools, "grep", pattern="something to find")
        assert "node_modules" not in out

    @pytest.mark.asyncio
    async def test_grep_searches_the_one_file_path_names(self, tools: dict) -> None:
        """A file was searched as a directory with nothing in it: no matches,
        whatever the file said."""
        assert "src/lib.py:2:" in await _call(tools, "grep", pattern="return", path="src/lib.py")

    @pytest.mark.asyncio
    async def test_a_glob_with_a_slash_is_read_from_the_workspace_root(self, tools: dict) -> None:
        """As ripgrep reads it, and as models write it: read from `path`,
        `src/*.py` under `src` named `src/src/*.py`."""
        out = await _call(tools, "grep", pattern="return", path="src", glob="src/*.py")
        assert "src/lib.py:2:" in out

    @pytest.mark.asyncio
    async def test_a_directory_after_a_leading_double_star_may_be_path_itself(
        self, tools: dict
    ) -> None:
        """As ripgrep reads it: `**/src/*.py` under `src` names `src/lib.py`, which a
        glob read from `src` cannot see."""
        out = await _call(tools, "grep", pattern="return", path="src", glob="**/src/*.py")
        assert "src/lib.py:2:" in out

    @pytest.mark.asyncio
    async def test_a_glob_with_a_directory_is_read_from_the_extra_root_path_is_in(
        self, tmp_path: Path
    ) -> None:
        """An extra root, such as a skill directory, lies outside the workspace root:
        a glob read from the workspace root finds nothing in it."""
        scripts = tmp_path / "skills" / "demo" / "scripts"
        scripts.mkdir(parents=True)
        (scripts / "run.py").write_text("def run():\n    return 1\n")
        (tmp_path / "work").mkdir()
        workspace = Workspace.at(tmp_path / "work", also=[tmp_path / "skills"])
        tools = {t.name: t for t in bundle("files", workspace=workspace)}
        out = await _call(
            tools, "grep", pattern="return", path=str(tmp_path / "skills"), glob="**/scripts/*.py"
        )
        assert "run.py:2:" in out

    @pytest.mark.asyncio
    async def test_a_glob_without_a_slash_matches_at_any_depth(self, tools: dict) -> None:
        assert "src/lib.py:2:" in await _call(tools, "grep", pattern="return", glob="*.py")

    @pytest.mark.asyncio
    async def test_a_search_that_reads_no_file_says_so(self, tools: dict) -> None:
        """Not the answer of a search that read files and found nothing: taken for
        it, it reads as "the text is not there" when the arguments were wrong."""
        nothing = await _call(tools, "grep", pattern="zzz", path="src", glob="*.md")
        not_found = await _call(tools, "grep", pattern="zzz", path="src")
        assert nothing != not_found
        assert "*.md" in nothing

    @pytest.mark.asyncio
    async def test_a_bad_regex_is_reported_not_raised(self, tools: dict) -> None:
        assert "not a valid regular expression" in await _call(tools, "grep", pattern="(unclosed")

    @pytest.mark.asyncio
    async def test_glob_finds_by_pattern(self, tools: dict) -> None:
        out = await _call(tools, "glob_files", pattern="**/*.py")
        assert "src/lib.py" in out
        assert "node_modules/junk.py" not in out

    @pytest.mark.asyncio
    async def test_paths_are_shown_relative_to_the_root(self, tools: dict) -> None:
        """Absolute paths are noise in a transcript and push the interesting
        part of an approval line off the screen."""
        out = await _call(tools, "list_dir", path=".")
        assert "app.py" in out
        assert "/private" not in out and "/tmp" not in out


class TestSearchStaysInsideTheWorkspace:
    """`glob_files` and `grep` hand a pattern to `Path.glob`, which knows nothing about
    the workspace: `..` climbs out of it, and a match may be a symlink to anywhere."""

    @pytest.fixture
    def secret(self, workspace: Workspace) -> Path:
        """A file beside the workspace, not in it."""
        outside = workspace.root.parent / f"{workspace.root.name}-secret.txt"
        outside.write_text("TOP-SECRET-CONTENT\n")
        return outside

    @pytest.mark.asyncio
    async def test_an_absolute_pattern_says_what_to_do_instead(
        self, tools: dict, workspace: Workspace
    ) -> None:
        """This crashed with `NotImplementedError: Non-relative patterns are
        unsupported`, which tells the model neither what was wrong nor what to do."""
        out = await _call(tools, "glob_files", pattern=str(workspace.root / "app.py"))

        assert out.startswith("ERROR:")
        assert "NotImplementedError" not in out
        assert "relative" in out and "path" in out

    @pytest.mark.asyncio
    @pytest.mark.parametrize("pattern", ["../*", "src/../../*", "~/x", "~", "..\\*"])
    async def test_a_pattern_that_leaves_the_folder_is_refused(
        self, tools: dict, pattern: str
    ) -> None:
        assert (await _call(tools, "glob_files", pattern=pattern)).startswith("ERROR:")

    @pytest.mark.asyncio
    async def test_an_empty_pattern_is_reported_not_raised(self, tools: dict) -> None:
        assert "empty" in await _call(tools, "glob_files", pattern="  ")

    @pytest.mark.asyncio
    async def test_glob_does_not_list_what_is_beside_the_workspace(
        self, tools: dict, secret: Path
    ) -> None:
        out = await _call(tools, "glob_files", pattern="../*")

        assert secret.name not in out

    @pytest.mark.asyncio
    async def test_grep_does_not_read_what_is_beside_the_workspace(
        self, tools: dict, secret: Path
    ) -> None:
        for pattern in ("../*", "../../*"):
            out = await _call(tools, "grep", pattern="TOP-SECRET", glob=pattern)
            assert "TOP-SECRET" not in out.replace("TOP-SECRET'", "")  # not the echoed pattern
            assert secret.name not in out

    @pytest.mark.asyncio
    async def test_a_symlink_out_of_the_workspace_is_neither_listed_nor_read(
        self, tools: dict, workspace: Workspace, secret: Path
    ) -> None:
        (workspace.root / "link.txt").symlink_to(secret)

        listed = await _call(tools, "glob_files", pattern="*.txt")
        searched = await _call(tools, "grep", pattern="TOP-SECRET")

        assert "link.txt" not in listed
        assert "link.txt" not in searched and "TOP-SECRET-CONTENT" not in searched

    @pytest.mark.asyncio
    async def test_a_symlink_to_a_file_inside_is_still_found(
        self, tools: dict, workspace: Workspace
    ) -> None:
        (workspace.root / "alias.md").symlink_to(workspace.root / "notes.md")

        assert "alias.md" in await _call(tools, "glob_files", pattern="*.md")

    @pytest.mark.asyncio
    async def test_a_folder_given_as_path_is_searched_with_a_relative_pattern(
        self, tools: dict
    ) -> None:
        out = await _call(tools, "glob_files", pattern="*.py", path="src")

        assert "lib.py" in out
        assert "app.py" not in out

    @pytest.mark.asyncio
    async def test_a_path_outside_every_root_is_refused(
        self, tools: dict, workspace: Workspace
    ) -> None:
        out = await _call(tools, "glob_files", pattern="*", path=str(workspace.root.parent))

        assert out.startswith("ERROR:") and "outside the workspace" in out

    @pytest.mark.asyncio
    async def test_a_path_that_is_a_file_is_refused(self, tools: dict) -> None:
        assert "not a directory" in await _call(tools, "glob_files", pattern="*", path="app.py")

    @pytest.mark.asyncio
    async def test_an_extra_root_can_be_searched_by_its_absolute_path(self, tmp_path: Path) -> None:
        """What the model needs to find a skill's shared file: the folder as `path`,
        the pattern relative to it."""
        project = tmp_path / "project"
        project.mkdir()
        skills = tmp_path / "skills"
        (skills / "visualization" / "charts").mkdir(parents=True)
        (skills / "plotly_style.py").write_text("# style\n")
        (skills / "visualization" / "charts" / "SKILL.md").write_text("# charts\n")
        extra = {t.name: t for t in bundle("files", workspace=Workspace.at(project, also=[skills]))}

        out = await _call(extra, "glob_files", pattern="plotly_style.py", path=str(skills))
        deep = await _call(extra, "glob_files", pattern="**/*.md", path=str(skills))

        assert out == str(skills.resolve() / "plotly_style.py")
        assert "SKILL.md" in deep

    @pytest.mark.asyncio
    async def test_a_dependency_folder_above_an_extra_root_is_not_what_hides_its_files(
        self, tmp_path: Path
    ) -> None:
        """`is_ignored` used to see the extra root's absolute path, so a skills
        directory under a `build/` or `dist/` ancestor matched nothing."""
        project = tmp_path / "project"
        project.mkdir()
        skills = tmp_path / "build" / "skills"
        skills.mkdir(parents=True)
        (skills / "plotly_style.py").write_text("# style\n")
        extra = {t.name: t for t in bundle("files", workspace=Workspace.at(project, also=[skills]))}

        assert "plotly_style.py" in await _call(
            extra, "glob_files", pattern="*.py", path=str(skills)
        )


class TestExtraRoots:
    def test_a_root_inside_another_extra_root_is_dropped(self, tmp_path: Path) -> None:
        """An install directory and each skill under it would be listed twice in
        the "outside the workspace" message."""
        project = tmp_path / "project"
        project.mkdir()
        install = tmp_path / "skills"
        (install / "data" / "data-analysis").mkdir(parents=True)

        workspace = Workspace.at(project, also=[install / "data" / "data-analysis", install])

        assert workspace.extra_roots == (install.resolve(),)

    def test_unrelated_roots_are_all_kept_in_order(self, tmp_path: Path) -> None:
        project = tmp_path / "project"
        project.mkdir()
        (tmp_path / "a").mkdir()
        (tmp_path / "b").mkdir()

        workspace = Workspace.at(project, also=[tmp_path / "b", tmp_path / "a"])

        assert workspace.extra_roots == ((tmp_path / "b").resolve(), (tmp_path / "a").resolve())

    def test_contains_resolves_nothing_it_is_not_given(self, tmp_path: Path) -> None:
        workspace = Workspace.at(tmp_path)

        assert workspace.contains(tmp_path.resolve() / "x")
        assert workspace.contains(tmp_path.resolve())
        assert not workspace.contains(tmp_path.resolve().parent)
        assert workspace.anchor(tmp_path.resolve().parent) is None


class TestWriting:
    @pytest.mark.asyncio
    async def test_write_creates_parent_directories(
        self, tools: dict, workspace: Workspace
    ) -> None:
        await _call(tools, "write_file", path="a/b/c.txt", content="hi")
        assert (workspace.root / "a" / "b" / "c.txt").read_text() == "hi"

    @pytest.mark.asyncio
    async def test_write_says_whether_it_overwrote(self, tools: dict) -> None:
        assert "Created" in await _call(tools, "write_file", path="new.txt", content="1")
        assert "Overwrote" in await _call(tools, "write_file", path="new.txt", content="2")

    @pytest.mark.asyncio
    async def test_edit_replaces_the_unique_match(self, tools: dict, workspace: Workspace) -> None:
        await _call(tools, "edit_file", path="src/lib.py", old="a + b", new="a - b")
        assert "a - b" in (workspace.root / "src" / "lib.py").read_text()

    @pytest.mark.asyncio
    async def test_edit_refuses_an_ambiguous_match(self, tools: dict) -> None:
        """Two matches means the model's mental model of the file is not
        precise enough to be trusted with a silent choice of which one."""
        out = await _call(tools, "edit_file", path="app.py", old="print(", new="log(")
        assert "appears 2 times" in out

    @pytest.mark.asyncio
    async def test_edit_refuses_a_missing_match(self, tools: dict, workspace: Workspace) -> None:
        out = await _call(tools, "edit_file", path="app.py", old="not there", new="x")
        assert "does not appear" in out
        assert "print('hello')" in (workspace.root / "app.py").read_text(), "left untouched"


class TestShell:
    @pytest.mark.asyncio
    async def test_it_runs_and_returns_the_exit_code(self, tools: dict) -> None:
        out = await _call(tools, "run_command", command="echo hi")
        assert "exit code: 0" in out
        assert "hi" in out

    @pytest.mark.asyncio
    async def test_a_failure_comes_back_instead_of_raising(self, tools: dict) -> None:
        out = await _call(tools, "run_command", command="exit 3")
        assert "exit code: 3" in out

    @pytest.mark.asyncio
    async def test_stderr_is_kept(self, tools: dict) -> None:
        """A command whose only useful output is its error message is the
        common case, not an edge case."""
        out = await _call(tools, "run_command", command="echo oops >&2")
        assert "oops" in out

    @pytest.mark.asyncio
    async def test_shell_syntax_works(self, tools: dict) -> None:
        """The reason this uses a shell at all: without it an agent cannot
        install a package and check the result in one step."""
        out = await _call(tools, "run_command", command="echo one && echo two | tr a-z A-Z")
        assert "one" in out and "TWO" in out

    @pytest.mark.asyncio
    async def test_it_runs_in_the_workspace(self, tools: dict, workspace: Workspace) -> None:
        out = await _call(tools, "run_command", command="pwd")
        assert str(workspace.root) in out

    @pytest.mark.asyncio
    async def test_a_slow_command_hands_back_a_handle_instead_of_dying(self, tools: dict) -> None:
        """The change at the centre of this feature: kill-on-timeout became
        detach-on-timeout.

        What used to happen is that two minutes of work was destroyed and the
        model was told to guess a bigger number. Now the call gives way and the
        command carries on."""
        started = asyncio.get_running_loop().time()
        out = await _call(tools, "run_command", command="sleep 40", timeout=1)
        elapsed = asyncio.get_running_loop().time() - started
        assert "still running" in out
        assert "check_command" in out, "the model must be told how to look again"
        # Measured against the command's own duration, not a fixed budget: a slow
        # runner needs seconds to start a process, but nothing legitimate takes
        # as long as the sleep it was told to leave alone.
        assert elapsed < 20, f"waited {elapsed:.0f}s before yielding"

    @pytest.mark.asyncio
    async def test_a_command_that_forks_does_not_hold_the_call(self, tools: dict) -> None:
        """The CI failure that shaped `_terminate_tree`, in its new form.

        A forking command leaves a child holding the pipes, and `communicate()`
        waits for the pipes rather than for the child — so the call blocked on
        the very thing it had given up on. Redirecting output to files is what
        removes the child's grip, and it is why this design uses them."""
        started = asyncio.get_running_loop().time()
        out = await _call(tools, "run_command", command="sleep 40 & wait", timeout=1)
        elapsed = asyncio.get_running_loop().time() - started
        assert "still running" in out
        assert elapsed < 20, f"waited {elapsed:.0f}s — the child still held the pipes"

    @pytest.mark.asyncio
    async def test_runaway_output_is_truncated(self, tools: dict) -> None:
        """A build log should not consume the context window."""
        out = await _call(tools, "run_command", command="python -c \"print('x' * 100000)\"")
        assert "omitted" in out
        assert len(out) < 40_000

    @pytest.mark.asyncio
    async def test_a_huge_output_keeps_both_ends(self, tools: dict) -> None:
        """Bigger than head and tail together, so there is a real gap.

        The first version of the stitching returned the head alone here, which
        throws away the half that explains the exit code."""
        script = "python -c \"print('START'); print('x'*200000); print('THE-ERROR')\""
        out = await _call(tools, "run_command", command=script)
        assert "START" in out
        assert "THE-ERROR" in out

    @pytest.mark.asyncio
    async def test_truncation_keeps_the_end(self, tools: dict) -> None:
        """Errors are at the end; a head-only clip hides the explanation."""
        script = "python -c \"print('a'*30000); print('THE-ERROR')\""
        out = await _call(tools, "run_command", command=script)
        assert "THE-ERROR" in out


class TestGating:
    def test_reading_is_not_gated(self, tools: dict) -> None:
        """Gating a harmless tool trains the user to approve without reading."""
        for name in ("read_file", "list_dir", "glob_files", "grep"):
            assert tools[name].approval_type is None

    def test_writing_and_running_are_gated(self, tools: dict) -> None:
        for name in ("write_file", "edit_file", "run_command"):
            assert tools[name].approval_type is not None

    def test_the_shell_subject_is_the_whole_command_line(self, tools: dict) -> None:
        """What the user reads must be what the shell receives. A subject of
        just the program name would show `git` for `git status; rm -rf ~`."""
        subject = tools["run_command"].approval_subject(command="git status; rm -rf ~")
        assert subject == "$ git status; rm -rf ~"

    def test_the_file_subject_is_the_path(self, tools: dict) -> None:
        """So "keep editing this file" is expressible and "write anywhere" is
        not — the content must never be part of the grant."""
        assert tools["write_file"].approval_subject == ["path"]

    def test_a_hardline_command_is_refused_before_the_prompt(self, tools: dict) -> None:
        from domyn.agents.approvals import compute_subject, hardline_match

        subject = compute_subject(tools["run_command"], {"command": "rm -rf /"})
        assert hardline_match(subject) is not None


class TestBundles:
    def test_nothing_is_mounted_without_asking(self) -> None:
        """The framework ships these; it never attaches them. A platform agent
        must not acquire shell access because of a default somewhere."""
        from domyn.agents import Agent
        from domyn.agents.llm import BaseLLMProvider

        class P(BaseLLMProvider):
            type: str = "p"
            model_name: str = "m"
            supported_structured_outputs: None = None

            async def generate(self, messages, **kwargs):  # type: ignore[no-untyped-def]
                raise NotImplementedError

            async def generate_streaming(self, messages, **kwargs):  # type: ignore[no-untyped-def]
                raise NotImplementedError
                yield

        agent = Agent(name="Bare", description="d", instruction="i", llm_provider=P())
        assert agent.tools == []

    def test_an_unknown_bundle_is_skipped_not_fatal(self, tmp_path: Path) -> None:
        """A typo in a config file should not stop an agent from starting."""
        assert bundle("files", "nonsense", root=tmp_path)

    def test_bundles_can_share_one_workspace(self, workspace: Workspace) -> None:
        """Combining bundles yields both sets, against the same root."""
        files = {t.name for t in bundle("files", workspace=workspace)}
        shell = {t.name for t in bundle("shell", workspace=workspace)}
        both = {t.name for t in bundle("files", "shell", workspace=workspace)}
        assert both == files | shell

    def test_the_names_config_lists_are_what_the_bundles_build(self, workspace: Workspace) -> None:
        """`/config` lists tools from `BUNDLE_TOOL_NAMES` without building them;
        a tool added to a bundle but not to the map would go unlisted."""
        from domyn.agents.tools.builtin import BUNDLE_TOOL_NAMES, BUNDLES

        assert set(BUNDLE_TOOL_NAMES) == set(BUNDLES)
        for name in BUNDLES:
            built = {t.name for t in bundle(name, workspace=workspace)}
            assert built == set(BUNDLE_TOOL_NAMES[name])

    def test_the_root_is_not_something_the_model_can_pass(self, tools: dict) -> None:
        """Containment would be advisory if the workspace were an argument."""
        for tool in tools.values():
            assert "root" not in tool.required_param_names()
            assert "workspace" not in tool.required_param_names()


def test_vendored_directories_are_recognised(tmp_path: Path) -> None:
    assert is_ignored(tmp_path / ".venv" / "lib" / "x.py", tmp_path)
    assert is_ignored(tmp_path / "src" / "__pycache__" / "x.pyc", tmp_path)
    assert not is_ignored(tmp_path / "src" / "x.py", tmp_path)


class TestArgumentsArriveInWhateverShapeTheModelSends:
    """JSON tool arguments are generated as text and nothing between the model
    and the tool coerces them to their declared types. A schema saying
    `integer` is a request, not a guarantee.

    Seen in a real session as `tool_error [TypeError]: '>' not supported
    between instances of 'str' and 'int'` — the command never ran because an
    *optional* argument arrived quoted.
    """

    @pytest.mark.parametrize("timeout", [30, "30", "30.5", 0, "", None, "abc", -5, [1]])
    @pytest.mark.asyncio
    async def test_any_timeout_still_runs_the_command(self, tools: dict, timeout: object) -> None:
        assert "exit code: 0" in await _call(
            tools, "run_command", command="echo ok", timeout=timeout
        )

    @pytest.mark.asyncio
    async def test_a_quoted_number_is_honoured_not_ignored(self, tools: dict) -> None:
        """Falling back to the default would be safe and wrong: the model asked
        for a short timeout and would wait the long one."""
        from domyn.agents.tools.builtin.shell import _seconds

        assert _seconds("5", 120.0) == 5.0

    @pytest.mark.asyncio
    async def test_unusable_values_fall_back_rather_than_fail(self) -> None:
        """The caller asked to run a command, not to negotiate a timeout."""
        from domyn.agents.tools.builtin.shell import _seconds

        for value in ("", None, "abc", 0, -1, {}):
            assert _seconds(value, 120.0) == 120.0

    @pytest.mark.asyncio
    async def test_a_quoted_timeout_still_bounds_the_call(self, tools: dict) -> None:
        """The coercion has to reach the real behaviour, not just avoid the
        exception."""
        started = asyncio.get_running_loop().time()
        out = await _call(tools, "run_command", command="sleep 30", timeout="1")
        assert "still running" in out
        assert asyncio.get_running_loop().time() - started < 20


class TestReadingInPieces:
    """A file too large to read at once used to come back truncated mid-line
    with "read a smaller file instead", which is advice and not a way forward.
    Now it comes back as a window that says how to get the next one."""

    @pytest.fixture
    def big(self, workspace: Workspace) -> Path:
        target = workspace.root / "big.txt"
        target.write_text("".join(f"line {i}\n" for i in range(1, 1201)))
        return target

    @pytest.mark.asyncio
    async def test_a_small_file_comes_back_whole_and_unadorned(self, tools: dict) -> None:
        """Most files are small; a note on every read would be noise."""
        out = await _call(tools, "read_file", path="app.py")
        assert out == "print('hello')\nprint('world')\n"

    @pytest.mark.asyncio
    async def test_a_large_file_is_windowed(self, tools: dict, big: Path) -> None:
        out = await _call(tools, "read_file", path="big.txt")
        assert "line 500" in out
        assert "line 501" not in out

    @pytest.mark.asyncio
    async def test_it_says_how_to_get_the_rest(self, tools: dict, big: Path) -> None:
        """The difference between a dead end and a next step."""
        out = await _call(tools, "read_file", path="big.txt")
        assert "start_line=501" in out

    @pytest.mark.asyncio
    async def test_continuing_picks_up_exactly_where_it_stopped(
        self, tools: dict, big: Path
    ) -> None:
        out = await _call(tools, "read_file", path="big.txt", start_line=501)
        assert out.startswith("line 501\n")

    @pytest.mark.asyncio
    async def test_the_last_window_says_so(self, tools: dict, big: Path) -> None:
        out = await _call(tools, "read_file", path="big.txt", start_line=1195)
        assert "end of file" in out
        assert "start_line=" not in out

    @pytest.mark.asyncio
    async def test_max_lines_is_honoured(self, tools: dict, big: Path) -> None:
        out = await _call(tools, "read_file", path="big.txt", start_line=10, max_lines=3)
        assert "line 10" in out and "line 12" in out and "line 13" not in out

    @pytest.mark.asyncio
    async def test_past_the_end_says_how_long_the_file_is(self, tools: dict, big: Path) -> None:
        """An empty result and an empty file look identical to a model; the
        length turns a dead end into a retry."""
        out = await _call(tools, "read_file", path="big.txt", start_line=5000)
        assert "1200 lines" in out

    @pytest.mark.asyncio
    async def test_an_empty_file_is_not_an_error(self, tools: dict, workspace: Workspace) -> None:
        (workspace.root / "empty.txt").write_text("")
        assert "empty" in await _call(tools, "read_file", path="empty.txt")

    @pytest.mark.parametrize(("start", "count"), [("3", "2"), (0, 0), (-1, -1), (None, None)])
    @pytest.mark.asyncio
    async def test_arguments_arrive_in_any_shape(
        self, tools: dict, big: Path, start: object, count: object
    ) -> None:
        """Same lesson as the shell timeout: the schema says integer and a
        model may still send a string."""
        out = await _call(tools, "read_file", path="big.txt", start_line=start, max_lines=count)
        assert "line " in out

    @pytest.mark.asyncio
    async def test_the_whole_file_is_not_pulled_into_memory(
        self, tools: dict, workspace: Workspace, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Slicing a fully-read file would make "read it in pieces" true for
        the model and false for the process doing the reading."""
        target = workspace.root / "huge.txt"
        target.write_text("".join(f"line {i}\n" for i in range(1, 200_001)))

        taken = 0
        original = Path.open

        class Counting:
            """The real handle, counting the lines taken from it."""

            def __init__(self, handle):
                self.handle = handle

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                self.handle.close()

            def __iter__(self):
                return self

            def __next__(self) -> str:
                nonlocal taken
                line = next(self.handle)
                taken += 1
                return line

        def counting_open(self, *args, **kwargs):
            handle = original(self, *args, **kwargs)
            return Counting(handle) if self == target else handle

        monkeypatch.setattr(Path, "open", counting_open)

        out = await _call(tools, "read_file", path="huge.txt", max_lines=5)
        assert "line 5" in out and "line 6" not in out
        # The window, plus the one line that says something follows.
        assert taken <= 6, f"{taken} lines read for a 5-line window"

    @pytest.mark.asyncio
    async def test_a_window_reaches_the_model_with_its_way_forward(
        self, tools: dict, workspace: Workspace
    ) -> None:
        """The context manager keeps only the head of a tool result. A window
        longer than its cap lost its tail, and with it the note saying where to
        continue: the model saw part of the window and had no way on. Checked
        through the real context manager, because what counts is what the model
        is shown, not what the tool returned."""
        row = "'" + "x" * 60 + "'"
        (workspace.root / "wide.py").write_text(
            "".join(f"line_{i:04d} = {row}\n" for i in range(1, 1201))
        )
        seen = await _as_the_model_sees_it(await _call(tools, "read_file", path="wide.py"))

        assert "(truncated)" not in seen
        match = re.search(r"start_line=(\d+)", seen)
        assert match, seen[-200:]
        resume = int(match.group(1))
        # The last line shown is whole, and the next window starts right after it.
        assert f"line_{resume - 1:04d} = {row}" in seen
        following = await _call(tools, "read_file", path="wide.py", start_line=resume)
        assert following.startswith(f"line_{resume:04d} = {row}\n")

    @pytest.mark.asyncio
    async def test_a_single_enormous_line_still_respects_the_budget(
        self, tools: dict, workspace: Workspace
    ) -> None:
        (workspace.root / "one.txt").write_text("x" * 500_000 + "\n")
        out = await _call(tools, "read_file", path="one.txt")
        assert len(out) < 600_000

    @pytest.mark.asyncio
    async def test_a_line_longer_than_a_read_is_continued_inside_it(
        self, tools: dict, workspace: Workspace
    ) -> None:
        """Minified JSON, a base64 blob, a tool result saved on one line: cut
        where a read ends, the rest of the line was unreachable — start_line
        past it skips it whole."""
        line = "".join(f"{i:07d}," for i in range(5000))
        (workspace.root / "one.json").write_text(line + "\nafter\n")

        pieces, start_char, notes = [], 0, []
        while True:
            out = await _call(tools, "read_file", path="one.json", start_char=start_char)
            body, _, note = out.partition("\n\n[")
            pieces.append(body)
            notes.append(note)
            match = re.search(r"start_line=1, start_char=(\d+)", note)
            if not match:
                break
            start_char = int(match.group(1))

        assert "".join(pieces) == line + "\nafter"
        assert len(pieces) == 4, notes
        assert notes[-1] == "lines 1-2, line 1 from character 36000, end of file]"

    @pytest.mark.asyncio
    async def test_a_start_char_past_the_end_of_the_line_says_how_long_it_is(
        self, tools: dict, workspace: Workspace
    ) -> None:
        (workspace.root / "short.txt").write_text("abc\ndef\n")
        out = await _call(tools, "read_file", path="short.txt", start_char=3)
        assert (
            out == "ERROR: 'short.txt' line 1 has 3 characters, so there is nothing at character 3."
        )

    @pytest.mark.asyncio
    async def test_a_line_over_only_by_its_line_break_is_not_cut(
        self, tools: dict, workspace: Workspace
    ) -> None:
        """Cut before its line break, the next read would be offered a
        start_char with nothing after it."""
        (workspace.root / "exact.txt").write_text("x" * MAX_READ_CHARS + "\nnext\n")
        out = await _call(tools, "read_file", path="exact.txt")
        assert "start_char" not in out
        assert out.endswith("more follow — read on with start_line=2]")


def _docx(paragraphs: list[str]) -> bytes:
    """The smallest Word document a converter accepts, built without a writer."""
    body = "".join(f"<w:p><w:r><w:t>{p}</w:t></w:r></w:p>" for p in paragraphs)
    parts = {
        "[Content_Types].xml": (
            '<?xml version="1.0" encoding="UTF-8"?><Types xmlns="http://schemas.openxmlformats'
            '.org/package/2006/content-types"><Default Extension="rels" ContentType="applicati'
            'on/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" Conten'
            'tType="application/xml"/><Override PartName="/word/document.xml" ContentType="app'
            'lication/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
            "</Types>"
        ),
        "_rels/.rels": (
            '<?xml version="1.0" encoding="UTF-8"?><Relationships xmlns="http://schemas.openx'
            'mlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://sc'
            'hemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target'
            '="word/document.xml"/></Relationships>'
        ),
        "word/document.xml": (
            '<?xml version="1.0" encoding="UTF-8"?><w:document xmlns:w="http://schemas.openxml'
            f'formats.org/wordprocessingml/2006/main"><w:body>{body}</w:body></w:document>'
        ),
    }
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, xml in parts.items():
            archive.writestr(name, xml)
    return buffer.getvalue()


_Image = tuple[str, tuple[int, int]]


def _pdf(*pages: str | _Image, size: tuple[int, int] = (200, 200)) -> bytes:
    """A PDF with one page per content stream, cross-reference table and all.
    Every page can draw text in Helvetica as ``/F1``; a page given as its
    content and an image's size in pixels can also draw a flat grey image as
    ``/Im1``. The pages are ``size`` points wide and high, 200 by 200 unless
    said. The header carries the comment of high bytes real PDFs start with,
    so it is not UTF-8 text either."""
    contents = [page if isinstance(page, str) else page[0] for page in pages]
    sizes = [None if isinstance(page, str) else page[1] for page in pages]
    count = len(pages)
    font = 3 + 2 * count
    image_objects = iter(range(font + 1, font + 1 + count))
    images = {i: next(image_objects) for i, pixels in enumerate(sizes) if pixels is not None}
    kids = " ".join(f"{3 + i} 0 R" for i in range(count))
    width, height = size
    objects: list[bytes] = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        f"<< /Type /Pages /Kids [{kids}] /Count {count} >>".encode(),
        *(
            (
                f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {width} {height}] /Resources << "
                f"/Font << /F1 {font} 0 R >> "
                + (f"/XObject << /Im1 {images[i]} 0 R >> " if i in images else "")
                + f">> /Contents {3 + count + i} 0 R >>"
            ).encode()
            for i in range(count)
        ),
        *(f"<< /Length {len(page)} >>\nstream\n{page}\nendstream".encode() for page in contents),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    objects += [_grey_image(*pixels) for pixels in sizes if pixels is not None]
    return _assemble(objects)


def _grey_image(width: int, height: int) -> bytes:
    """An image XObject of ``width`` by ``height`` flat grey pixels."""
    data = zlib.compress(b"\x80" * (width * height))
    header = (
        f"<< /Type /XObject /Subtype /Image /Width {width} /Height {height} "
        "/ColorSpace /DeviceGray /BitsPerComponent 8 /Filter /FlateDecode "
        f"/Length {len(data)} >>\nstream\n"
    )
    return header.encode() + data + b"\nendstream"


def _assemble(objects: list[bytes]) -> bytes:
    """A PDF of ``objects``, numbered from 1 in order, the first its catalog."""
    out, offsets = b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n", []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n".encode() + body + b"\nendobj\n"
    table = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    out += b"".join(f"{offset:010d} 00000 n \n".encode() for offset in offsets)
    trailer = f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{table}\n%%EOF\n"
    return out + trailer.encode()


def _text(words: str) -> str:
    """A page's content: ``words`` written once, in Helvetica."""
    return f"BT /F1 12 Tf 20 100 Td ({words}) Tj ET"


def _lines(*rows: str) -> str:
    """A page's content: each of ``rows`` on its own line, in 9-point Helvetica."""
    return " ".join(
        f"BT /F1 9 Tf 10 {185 - 13 * index} Td ({row}) Tj ET" for index, row in enumerate(rows)
    )


# A page that is one image and nothing else, as a scanner makes one: large
# enough in pixels to be a scan, and covering the whole page.
_SCAN = ("q 200 0 0 200 0 0 cm /Im1 Do Q", (1000, 1000))


class TestReadingDocuments:
    """A PDF or a spreadsheet handed to a model as bytes teaches it nothing, so
    documents are converted to markdown and then read like any other file."""

    @pytest.mark.asyncio
    async def test_a_document_read_in_windows_gives_all_its_text_once(
        self, tools: dict, workspace: Workspace
    ) -> None:
        """Windows over the converted text join up: nothing skipped at a
        boundary, nothing read twice — including a paragraph longer than any
        one window, which a converter makes out of a dense page."""
        paragraphs = [f"clause {i:04d}" for i in range(1, 2001)]
        paragraphs[1000] = " ".join(f"word{i:05d}" for i in range(5000))
        (workspace.root / "contract.docx").write_bytes(_docx(paragraphs))

        bodies: list[str] = []
        start = 1
        while True:
            out = await _call(tools, "read_file", path="contract.docx", start_line=start)
            assert not out.startswith("ERROR"), out
            body, _, note = out.partition("\n\n[lines ")
            bodies.append(body)
            match = re.search(r"start_line=(\d+)", note)
            if not match:
                assert "end of file" in note, note
                break
            start = int(match.group(1))

        tokens = r"clause \d{4}|word\d{5}"
        assert len(bodies) > 1, "large enough to need more than one window"
        assert re.findall(tokens, "".join(bodies)) == re.findall(tokens, "\n".join(paragraphs))

    @pytest.mark.parametrize(
        ("content", "reason"),
        [
            (b"%PDF-1.7\n\xff\xfe\xfd not really\n", "corrupt"),
            (_pdf(_SCAN, _SCAN), "has no text on any page"),
        ],
        ids=["corrupt", "no-text-on-any-page"],
    )
    @pytest.mark.asyncio
    async def test_an_unreadable_document_says_why_and_not_to_retry(
        self, tools: dict, workspace: Workspace, content: bytes, reason: str
    ) -> None:
        """Each of these is a property of the file. A model told only that it
        could not be read tries again, differently, and gets nowhere."""
        (workspace.root / "report.pdf").write_bytes(content)
        out = await _call(tools, "read_file", path="report.pdf")
        assert out.startswith("ERROR"), out
        assert reason in out
        assert "Retrying will not help" in out

    @pytest.mark.asyncio
    async def test_a_pdf_refused_for_some_of_its_pages_is_not_said_to_have_no_text(
        self, tools: dict, workspace: Workspace, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A PDF the page reader cannot open goes to anydoc, which refuses one
        with any page without text. Its other pages have text all the same, so
        the model must not be told it has none."""
        reader = tmp_path / "refusing_reader.py"
        reader.write_text('import sys\nsys.stdout.write(\'{"error": "Invalid PDF structure"}\')\n')
        monkeypatch.setattr(documents, "_READER", reader)
        (workspace.root / "report.pdf").write_bytes(_pdf(_text("Terms"), _SCAN, _text("Penalties")))
        out = await _call(tools, "read_file", path="report.pdf")
        assert out.startswith("ERROR") and "page 2 of its 3 pages is scanned" in out, out
        assert "no text on any page" not in out, out

    @pytest.mark.asyncio
    async def test_a_pdf_page_without_text_is_marked_and_the_rest_is_read(
        self, tools: dict, workspace: Workspace
    ) -> None:
        """One scanned page among pages of text used to make the whole PDF
        unreadable, or vanish from it without a word. Its pages of text are
        read, the scanned one is marked on its page, and the first lines name
        it, since a mark further down may be windows away."""
        (workspace.root / "report.pdf").write_bytes(_pdf(_text("Terms"), _SCAN, _text("Penalties")))
        out = await _call(tools, "read_file", path="report.pdf")
        assert out.startswith("[Not included: page 2 of its 3 pages"), out
        page = r"<!-- Page 2 -->\n\n\[Page not included: [^\]]+\]\n\n<!-- Page 3 -->"
        assert re.search(rf"Terms.*{page}.*Penalties", out, re.S), out

    @pytest.mark.asyncio
    async def test_only_the_first_window_says_what_was_left_out(
        self, tools: dict, workspace: Workspace
    ) -> None:
        (workspace.root / "report.pdf").write_bytes(_pdf(_text("Terms"), _SCAN, _text("Penalties")))
        out = await _call(tools, "read_file", path="report.pdf", start_line=3)
        assert "of its 3 pages" not in out, out
        assert out.startswith("Terms"), out

    @pytest.mark.asyncio
    async def test_the_first_lines_name_twenty_runs_of_pages_and_count_the_rest(
        self, tools: dict, workspace: Workspace
    ) -> None:
        """A scan on every other page of a long PDF must not open its first
        window with a line of hundreds of page numbers."""
        pages = [_SCAN if n % 2 == 0 else _text(f"Clause {n}") for n in range(1, 61)]
        (workspace.root / "report.pdf").write_bytes(_pdf(*pages))
        first = (await _call(tools, "read_file", path="report.pdf")).splitlines()[0]
        assert "pages 2, 4, 6," in first, first
        assert "38, 40 and 10 more of its 60 pages" in first, first

    @pytest.mark.asyncio
    async def test_text_saved_under_a_documents_name_is_read_as_text(
        self, tools: dict, workspace: Workspace
    ) -> None:
        """What "export to Excel" often writes: tab-separated text called .xls.
        It is not the workbook its name claims, and it is readable as it is."""
        (workspace.root / "export.xls").write_text("id\tname\n1\tAda\n")
        assert await _call(tools, "read_file", path="export.xls") == "id\tname\n1\tAda\n"

    @pytest.mark.asyncio
    async def test_a_variant_of_a_known_format_is_read_as_a_document(
        self, tools: dict, workspace: Workspace
    ) -> None:
        """A macro-enabled Word file is a Word file: which names are documents
        is the converter's answer, and it includes the variants."""
        (workspace.root / "memo.docm").write_bytes(_docx(["quarterly figures"]))
        out = await _call(tools, "read_file", path="memo.docm")
        assert "quarterly figures" in out, out

    @pytest.mark.asyncio
    async def test_a_document_under_another_name_is_read_by_its_content(
        self, tools: dict, workspace: Workspace
    ) -> None:
        """A download saved without its extension is still a Word file, and the
        converter knows it by what is inside."""
        (workspace.root / "attachment").write_bytes(_docx(["quarterly figures"]))
        out = await _call(tools, "read_file", path="attachment")
        assert "quarterly figures" in out, out

    @pytest.mark.asyncio
    async def test_a_csv_is_read_as_written(self, tools: dict, workspace: Workspace) -> None:
        """The converter would make it a markdown table, whose lines and text
        are not the file's. Read as written, it is what edit_file matches."""
        (workspace.root / "rows.csv").write_text("id,name\n1,Ada\n")
        assert await _call(tools, "read_file", path="rows.csv") == "id,name\n1,Ada\n"

    @pytest.mark.asyncio
    async def test_a_csv_that_is_not_utf8_is_read_as_a_table(
        self, tools: dict, workspace: Workspace
    ) -> None:
        """An Excel export in Windows-1252 is text no decoding here can read,
        and the converter can: it detects the encoding. Refusing it would leave
        the model a file it cannot see at all."""
        (workspace.root / "export.csv").write_bytes("name;price\ncafé;12 €\n".encode("cp1252"))
        out = await _call(tools, "read_file", path="export.csv")
        assert out.startswith("| name | price |"), out
        assert "café" in out and "12 €" in out, out

    @pytest.mark.asyncio
    async def test_a_csv_that_stops_decoding_late_is_a_table_from_its_first_line(
        self, tools: dict, workspace: Workspace
    ) -> None:
        """How a CSV reads is decided on the whole file. Decided on the first
        window, a CSV that stops decoding further down would come back as the
        file's own lines first and as a table's later, numbered differently."""
        rows = "".join(f"row{i},{i}\n" for i in range(1, 50))
        content = ("id,value\n" + rows).encode() + "café,50\n".encode("cp1252")
        (workspace.root / "late.csv").write_bytes(content)
        first = await _call(tools, "read_file", path="late.csv", max_lines=5)
        assert first.startswith("| id | value |"), first


class TestWritingOverDocuments:
    """What read_file shows of a document is a conversion, and the write tools
    write plain text: writing it back replaces the document, and a later read
    shows that text as if nothing had happened."""

    @pytest.mark.parametrize(
        ("tool", "arguments"),
        [
            ("write_file", {"content": "Delivery within 60 days.\n"}),
            ("append_file", {"content": "Penalty doubled.\n"}),
            ("edit_file", {"old": "30 days", "new": "60 days"}),
        ],
        ids=["write_file", "append_file", "edit_file"],
    )
    @pytest.mark.asyncio
    async def test_no_write_tool_changes_a_document(
        self, tools: dict, workspace: Workspace, tool: str, arguments: dict
    ) -> None:
        """An RTF document is plain ASCII, so no tool is stopped by the file
        failing to decode: each has to recognize it as a document."""
        rtf = b"{\\rtf1\\ansi Delivery within 30 days.\\par}\n"
        (workspace.root / "contract.rtf").write_bytes(rtf)
        out = await _call(tools, tool, path="contract.rtf", **arguments)
        assert out.startswith("ERROR"), out
        assert (workspace.root / "contract.rtf").read_bytes() == rtf

    @pytest.mark.asyncio
    async def test_a_document_under_another_name_is_protected_too(
        self, tools: dict, workspace: Workspace
    ) -> None:
        """read_file converts a document by its content, whatever its name, so
        the name cannot be what protects it."""
        original = _docx(["Delivery within 30 days."])
        (workspace.root / "contract").write_bytes(original)
        out = await _call(tools, "write_file", path="contract", content="Within 60 days.\n")
        assert out.startswith("ERROR"), out
        assert (workspace.root / "contract").read_bytes() == original

    @pytest.mark.asyncio
    async def test_text_under_a_documents_name_stays_editable(
        self, tools: dict, workspace: Workspace
    ) -> None:
        """A TSV saved as .xls is read as the text it is, so it is edited as text."""
        (workspace.root / "export.xls").write_text("id\tname\n1\tAda\n")
        out = await _call(tools, "edit_file", path="export.xls", old="Ada", new="Bob")
        assert not out.startswith("ERROR"), out
        assert (workspace.root / "export.xls").read_text() == "id\tname\n1\tBob\n"

    @pytest.mark.asyncio
    async def test_a_csv_that_reads_as_a_table_is_protected(
        self, tools: dict, workspace: Workspace
    ) -> None:
        """A CSV that is not UTF-8 is shown to the model as a table: writing
        that back would replace the file with the table's text."""
        original = "name;price\ncafé;12 €\n".encode("cp1252")
        (workspace.root / "export.csv").write_bytes(original)
        out = await _call(tools, "write_file", path="export.csv", content="name;price\n")
        assert out.startswith("ERROR"), out
        assert (workspace.root / "export.csv").read_bytes() == original

    @pytest.mark.asyncio
    async def test_a_csv_that_reads_as_written_stays_editable(
        self, tools: dict, workspace: Workspace
    ) -> None:
        (workspace.root / "rows.csv").write_text("id,name\n1,Ada\n")
        out = await _call(tools, "edit_file", path="rows.csv", old="Ada", new="Bob")
        assert not out.startswith("ERROR"), out
        assert (workspace.root / "rows.csv").read_text() == "id,name\n1,Bob\n"

    def test_every_bundle_has_a_description_for_setup(self) -> None:
        """`domyn setup` shows it beside each bundle when asking which to mount."""
        from domyn.agents.tools.builtin import BUNDLE_DESCRIPTIONS, BUNDLES

        assert set(BUNDLE_DESCRIPTIONS) == set(BUNDLES)
        assert all(BUNDLE_DESCRIPTIONS[name] for name in BUNDLES)
