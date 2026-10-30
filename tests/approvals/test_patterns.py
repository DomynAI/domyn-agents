"""Tests for pattern grants and auto-approve.

Exact grants were the safe first cut and they made the CLI tiring: approving
``curl .../Bologna`` did nothing for ``curl .../Milan``, so the user reached the
third prompt and stopped reading it — which is the failure the gate exists to
prevent. Patterns fix that, and in doing so become the one place where a user
can grant more than the line they were shown. Most of these tests are about
keeping that widening deliberate.
"""

# No `from __future__ import annotations`: builds real tools.
import json
from typing import Any

import pytest

from domyn.agents import Agent, Runner
from domyn.agents.approvals import (
    ApprovalGrant,
    ApprovalPolicy,
    ApprovalRequest,
    ConversationApprovalStore,
    InMemoryApprovalStore,
)
from domyn.agents.core import ApprovalType, BaseEvent, ExecutionEventType, Part
from domyn.agents.llm import BaseLLMProvider, LLMResponse, LLMToolCall, ToolCallFunction
from domyn.agents.planner_strategy import PlannerStrategyFactory
from domyn.agents.tools.builtin import bundle


def _request(subject: str, **kwargs: Any) -> ApprovalRequest:
    kwargs.setdefault("tool_name", "run_command")
    kwargs.setdefault("agent_name", "Assistant")
    return ApprovalRequest(subject=subject, **kwargs)


class TestExactGrantsDoNotWiden:
    @pytest.mark.asyncio
    async def test_a_literal_glob_stays_literal(self) -> None:
        """``ls *.py`` is an ordinary command to approve once. Stored as a
        pattern it would silently also permit ``ls anything.py`` — which is why
        the two kinds are separate rather than one glob."""
        store = InMemoryApprovalStore()
        await store.grant(_request("$ ls *.py"))
        assert await store.is_granted(_request("$ ls *.py"))
        assert not await store.is_granted(_request("$ ls secrets.py"))

    @pytest.mark.asyncio
    async def test_a_bracket_in_a_command_is_not_a_character_class(self) -> None:
        store = InMemoryApprovalStore()
        await store.grant(_request("$ grep '[abc]' file"))
        assert not await store.is_granted(_request("$ grep 'a' file"))


class TestPatternGrants:
    @pytest.mark.asyncio
    async def test_they_cover_the_case_that_started_this(self) -> None:
        store = InMemoryApprovalStore()
        await store.grant(_request("$ curl -s https://wttr.in/Bologna"), pattern="$ curl *")
        assert await store.is_granted(_request("$ curl -s https://wttr.in/Milan"))

    @pytest.mark.asyncio
    async def test_they_do_not_cross_tools(self) -> None:
        """ "Commands like this" must not become "anything at all"."""
        store = InMemoryApprovalStore()
        await store.grant(_request("$ curl x"), pattern="$ curl *")
        assert not await store.is_granted(_request("path=/etc/passwd", tool_name="write_file"))

    @pytest.mark.asyncio
    async def test_they_do_not_cross_agents(self) -> None:
        store = InMemoryApprovalStore()
        await store.grant(_request("$ curl x"), pattern="$ curl *")
        assert not await store.is_granted(_request("$ curl y", agent_name="SubAgent"))

    @pytest.mark.asyncio
    async def test_a_narrow_pattern_stays_narrow(self) -> None:
        store = InMemoryApprovalStore()
        await store.grant(_request("$ git status"), pattern="$ git status*")
        assert await store.is_granted(_request("$ git status --short"))
        assert not await store.is_granted(_request("$ git push"))

    @pytest.mark.asyncio
    async def test_they_survive_the_conversation_state_round_trip(self) -> None:
        """Conversation grants are serialised by the conversation manager, so
        they have to come back as grants and not as dicts."""

        class Ctx:
            def __init__(self) -> None:
                self.state: dict = {}

        store = ConversationApprovalStore()
        context = Ctx()
        await store.grant(_request("$ curl x"), context, pattern="$ curl *")
        # what a manager would hand back after a JSON round trip
        context.state["approved_subjects"] = json.loads(
            json.dumps(context.state["approved_subjects"])
        )
        assert await store.is_granted(_request("$ curl y"), context)

    @pytest.mark.asyncio
    async def test_a_corrupt_entry_is_skipped_not_fatal(self) -> None:
        class Ctx:
            def __init__(self) -> None:
                self.state: dict = {"approved_subjects": ["not a grant", {"junk": 1}]}

        assert await ConversationApprovalStore().granted(Ctx()) == []

    @pytest.mark.asyncio
    async def test_hardline_still_beats_a_pattern(self) -> None:
        """The one rule nothing can widen past."""
        store = InMemoryApprovalStore()
        await store.grant(_request("$ rm -rf ./build"), pattern="$ rm *")
        policy = ApprovalPolicy(stores=[store])
        assert (await policy.resolve(_request("$ rm -rf /"))).is_deny

    @pytest.mark.asyncio
    async def test_patterns_cover_a_line_command_by_command(self) -> None:
        """Every command must be covered, by one pattern or several."""
        line = _request("$ git pull && make", parts=["$ git pull", "$ make"])
        store = InMemoryApprovalStore()
        await store.grant(line, pattern="$ git *")
        assert not await store.is_granted(line)
        await store.grant(line, pattern="$ make *")
        assert await store.is_granted(line)

    @pytest.mark.asyncio
    async def test_a_program_pattern_covers_it_without_arguments(self) -> None:
        """``$ make *`` covers a bare ``make``, never ``makeself``."""
        store = InMemoryApprovalStore()
        await store.grant(_request("$ make test"), pattern="$ make *")
        assert await store.is_granted(_request("$ make", parts=["$ make"]))
        assert not await store.is_granted(_request("$ makeself", parts=["$ makeself"]))

    @pytest.mark.asyncio
    async def test_a_line_that_cannot_be_split_takes_only_an_exact_grant(self) -> None:
        line = _request("$ echo $(make)", parts=[])
        store = InMemoryApprovalStore()
        await store.grant(line, pattern="*")
        assert not await store.is_granted(line)
        await store.grant(line)
        assert await store.is_granted(line)


class TestToolsProposeSensiblePatterns:
    def _tools(self, tmp_path: Any) -> dict:
        return {t.name: t for t in bundle("files", "shell", root=tmp_path)}

    @pytest.mark.parametrize(
        ("command", "expected"),
        [
            ("curl -s https://wttr.in/Bologna", "$ curl *"),
            ("git status --short", "$ git *"),
            ("FOO=1 uv add requests", "$ FOO=1 uv *"),
            ("FOO='a b' uv add requests", "$ FOO='a b' uv *"),
            ("pytest", "$ pytest *"),
            # Glue around the program is left out.
            ("cd /tmp && python a.py 2>&1 | tail -5", "$ python *"),
            ('S=/x\nOUT=/y\npython "$S/a.py" --out "$OUT" | tail -n 30', "$ python *"),
            ("git status && git diff", "$ git *"),
            ("cd build && make", "$ make *"),
            # Two programs, or a line that cannot be split.
            ("git status && rm -rf build", None),
            ("echo $(whoami)", None),
            # Programs that run their arguments.
            ("sudo make install", None),
            ("cd /tmp && bash -c 'make'", None),
            ("/bin/bash -c 'make'", None),
            ("/usr/bin/env python a.py", None),
        ],
    )
    def test_the_shell_proposes_the_program(
        self, tmp_path: Any, monkeypatch: pytest.MonkeyPatch, command: str, expected: str | None
    ) -> None:
        """ "Let it run curl" is a permission a person can reason about; "let it
        run this exact URL" is not one they will ever hit twice."""
        for name in ("S", "OUT"):
            monkeypatch.delenv(name, raising=False)
        assert self._tools(tmp_path)["run_command"].approval_pattern(command=command) == expected

    def test_reading_tools_propose_nothing(self, tmp_path: Any) -> None:
        """They are not gated, so a standing permission would be meaningless."""
        assert self._tools(tmp_path)["read_file"].approval_pattern is None


class TestCommandParts:
    """What a pattern must cover: every command, minus glue."""

    @pytest.fixture
    def parts(self, tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> Any:
        for name in ("S", "OUT"):
            monkeypatch.delenv(name, raising=False)
        tool = {t.name: t for t in bundle("shell", root=tmp_path)}["run_command"]
        return lambda command: tool.approval_parts(command=command)

    @pytest.mark.parametrize(
        ("command", "expected"),
        [
            ("git status", ["$ git status"]),
            ("git pull && make", ["$ git pull", "$ make"]),
            (
                "git log; curl -s x -o /tmp/x; sh /tmp/x",
                ["$ git log", "$ curl -s x -o /tmp/x", "$ sh /tmp/x"],
            ),
            ("git status\nrm -rf build", ["$ git status", "$ rm -rf build"]),
            ("sleep 9 & rm -rf build", ["$ sleep 9", "$ rm -rf build"]),
            ("make 2>&1 | sh", ["$ make", "$ sh"]),
            # Inside quotes a separator is text.
            ('git commit -m "a; b && c | d"', ['$ git commit -m "a; b && c | d"']),
            ("python a.py &> log.txt", ["$ python a.py &> log.txt"]),
            # Glue is not asked about...
            ("cd /tmp && make || echo failed", ["$ make"]),
            (
                'S=/x\nOUT=/y\npython "$S/a.py" --sep $\'\\t\' --out "$OUT" 2>&1 | tail -30',
                ['$ python "$S/a.py" --sep $\'\\t\' --out "$OUT"'],
            ),
            ("OUT=/y 2>/dev/null\nmake 2>&1 >/dev/null && echo done >&2", ["$ make"]),
            ("make 2>&1 | grep -iE 'error|warn' | wc -l", ["$ make"]),
            ("cd /tmp && \\\n  make", ["$ make"]),
            # ...unless it can do something...
            ("PATH=/tmp/evil:$PATH; pytest", ["$ PATH=/tmp/evil:$PATH", "$ pytest"]),
            ("cd /tmp > ~/.bashrc && make", ["$ cd /tmp > ~/.bashrc", "$ make"]),
            ("make | tail ~/.ssh/id_rsa", ["$ make", "$ tail ~/.ssh/id_rsa"]),
            ("make | grep -i key ~/.ssh/id_rsa", ["$ make", "$ grep -i key ~/.ssh/id_rsa"]),
            ("make | grep -r key", ["$ make", "$ grep -r key"]),
            ("make | grep $KEY", ["$ make", "$ grep $KEY"]),
            # ...or is all there is.
            ("echo 'a && b'", ["$ echo 'a && b'"]),
            # A quoted heredoc's body is data; what follows it is a command again.
            (
                "python - <<'EOF'\nrm -rf x\nEOF\nrm -rf build",
                ["$ python - <<'EOF'", "$ rm -rf build"],
            ),
            # A line that runs something it does not show is not split at all.
            ("echo $(rm -rf build)", []),
            ("echo `rm -rf build`", []),
            ('echo "$(rm -rf build)"', []),
            ("diff <(ls a) <(ls b)", []),
            ("(cd /tmp && make)", []),
            ("cat <<EOF\n$(rm -rf build)\nEOF", []),
            # Nor one that runs a variable's value as code.
            ("QA=x QX='a[$(touch m)]'; ls ${QA:QX}", []),
            ("QX='a[$(touch m)]'; echo $[QX]; make", []),
            ("echo ${!QX}; make", []),
            ('echo "${QX@P}"; make', []),
            ('make "${OUT}/a"', ['$ make "${OUT}/a"']),
            # Nor one whose program is an expansion: `CMD=rm` before it is glue.
            ("CMD=rm\n$CMD -rf build", []),
            ("${CMD} -rf build", []),
            # A comment does not run, up to the end of its line.
            ("make # && rm -rf build", ["$ make"]),
            ("make # a comment, then\nrm -rf build", ["$ make", "$ rm -rf build"]),
            ("echo 'unterminated", []),
        ],
    )
    def test_a_line_is_split_into_what_it_runs(
        self, parts: Any, command: str, expected: list[str]
    ) -> None:
        assert parts(command) == expected

    def test_a_variable_the_commands_inherit_is_not_glue(
        self, parts: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Assigned, an exported variable changes what the next command sees."""
        monkeypatch.setenv("OUT", "/somewhere")
        assert parts("OUT=/y\nmake") == ["$ OUT=/y", "$ make"]


class TestTheEditedPattern:
    """The proposed pattern can be edited before it becomes a rule."""

    def _answer(
        self, *typed: str, subject: str = "$ python a.py", parts: list[str] | None = None
    ) -> Any:
        from unittest.mock import MagicMock, patch

        from domyn.agents.interfaces.cli import Theme
        from domyn.agents.interfaces.cli.approval import prompt_for_approval

        event = BaseEvent(
            event_type=ExecutionEventType.TOOL_NEEDS_FEEDBACK,
            author="Assistant",
            metadata={
                "approval_subject": subject,
                "approval_parts": parts,
                "approval_scopes": ["conversation", "project", "user"],
                "approval_pattern": "$ python *",
            },
            action=_feedback_action(),
        )
        with patch("builtins.input", side_effect=list(typed)):
            return prompt_for_approval(
                MagicMock(), Theme.build(allow_animation=False), event
            ).action

    def test_a_pattern_that_cannot_match_the_call_is_asked_again(self) -> None:
        """``python *`` misses ``$ python a.py``: asked again."""
        action = self._answer("5", "python *", "")
        assert (action.remember_scope, action.remember_pattern) == ("conversation", "$ python *")

    def test_a_pattern_is_checked_against_the_commands_the_line_runs(self) -> None:
        """Checked per command: ``$ python *`` covers ``cd /tmp && python a.py``."""
        action = self._answer(
            "5",
            "",
            "",
            "",
            subject="$ cd /tmp && python a.py",
            parts=["$ python a.py"],
        )
        assert (action.remember_scope, action.remember_pattern) == ("conversation", "$ python *")

    def test_giving_up_approves_this_call_only(self) -> None:
        action = self._answer("5", "python *", "python *", "python *")
        assert action.approve is True
        assert (action.remember_scope, action.remember_pattern) == (None, None)


class TestThroughTheRunner:
    class _Provider(BaseLLMProvider):
        type: str = "mock"
        name: str = "mock"
        model_name: str = "mock"
        supported_structured_outputs: None = None
        command: str = "curl --version # Bologna"
        _asked: bool = False

        async def generate(self, messages: Any, **kwargs: Any) -> LLMResponse:
            if self._asked:
                return LLMResponse(content="Done.", generated_tokens=1, execution_time=0.0)
            object.__setattr__(self, "_asked", True)
            return LLMResponse(
                content="",
                generated_tokens=1,
                execution_time=0.0,
                tool_calls=[
                    LLMToolCall(
                        id="c1",
                        function=ToolCallFunction(
                            name="run_command", arguments=json.dumps({"command": self.command})
                        ),
                    )
                ],
            )

        async def generate_streaming(self, messages: Any, **kwargs: Any) -> Any:
            yield

    def _agent(self, tmp_path: Any, command: str) -> Agent:
        tools = list(bundle("shell", root=tmp_path))
        return Agent(
            name="Assistant",
            description="d",
            instruction="i",
            llm_provider=self._Provider(command=command),
            planner=PlannerStrategyFactory.from_config(
                {"type": "tool_use"}, save_in_registry=False
            ),
            tools=tools,
        )

    async def _turn(
        self,
        runner: Runner,
        agent: Agent,
        *,
        pattern: str | None = None,
        scope: str | None = None,
        accept_proposal: bool = False,
    ) -> list[BaseEvent]:
        pending: BaseEvent | None = BaseEvent(
            event_type=ExecutionEventType.USER_INPUT, author="user", content=[Part(text="go")]
        )
        seen: list[BaseEvent] = []
        for _ in range(4):
            needs = None
            async for event in runner.run(
                application_name="app",
                user_id="u",
                root=agent,
                user_input=pending,
                conversation_id="c1",
            ):
                seen.append(event)
                if event.event_type is ExecutionEventType.TOOL_NEEDS_FEEDBACK:
                    needs = event
            if needs is None:
                return seen
            action = needs.action
            action.approve = True
            action.remember_scope = scope
            action.remember_pattern = (
                needs.metadata.get("approval_pattern") if accept_proposal else pattern
            )
            pending = BaseEvent(
                event_type=ExecutionEventType.TOOL_FEEDBACK, author="user", action=action
            )
        raise AssertionError("the gate is looping")

    @pytest.mark.asyncio
    async def test_the_proposal_reaches_the_prompt(self, tmp_path: Any) -> None:
        runner = Runner()
        events = await self._turn(runner, self._agent(tmp_path, "curl --version"))
        asked = next(e for e in events if e.event_type is ExecutionEventType.TOOL_NEEDS_FEEDBACK)
        assert asked.metadata["approval_pattern"] == "$ curl *"

    @pytest.mark.asyncio
    async def test_a_command_without_arguments_still_gets_a_proposal(self, tmp_path: Any) -> None:
        events = await self._turn(Runner(), self._agent(tmp_path, "cd . && date"))
        asked = next(e for e in events if e.event_type is ExecutionEventType.TOOL_NEEDS_FEEDBACK)
        assert asked.metadata["approval_pattern"] == "$ date *"

    @pytest.mark.asyncio
    async def test_a_pattern_silences_the_next_similar_call(self, tmp_path: Any) -> None:
        """The exact scenario reported: two cities, one approval."""
        runner = Runner()
        await self._turn(
            runner,
            self._agent(tmp_path, "curl --version # Bologna"),
            scope="conversation",
            pattern="$ curl *",
        )
        events = await self._turn(runner, self._agent(tmp_path, "curl --version # Milan"))
        assert not any(e.event_type is ExecutionEventType.TOOL_NEEDS_FEEDBACK for e in events)

    @pytest.mark.asyncio
    async def test_without_a_pattern_it_still_asks(self, tmp_path: Any) -> None:
        runner = Runner()
        await self._turn(
            runner, self._agent(tmp_path, "curl --version # Bologna"), scope="conversation"
        )
        events = await self._turn(runner, self._agent(tmp_path, "curl --version # Milan"))
        assert any(e.event_type is ExecutionEventType.TOOL_NEEDS_FEEDBACK for e in events)

    @pytest.mark.asyncio
    async def test_a_pattern_does_not_cover_a_different_program(self, tmp_path: Any) -> None:
        runner = Runner()
        await self._turn(
            runner,
            self._agent(tmp_path, "curl --version"),
            scope="conversation",
            pattern="$ curl *",
        )
        events = await self._turn(runner, self._agent(tmp_path, "rm -rf build"))
        assert any(e.event_type is ExecutionEventType.TOOL_NEEDS_FEEDBACK for e in events)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("first", "second"),
        [
            ("echo Bologna", "echo Milan"),
            # Leading assignments are part of the subject and of the proposal.
            ("LANG=C echo Bologna", "LANG=C echo Milan"),
        ],
    )
    async def test_accepting_the_proposal_silences_the_next_similar_call(
        self, tmp_path: Any, first: str, second: str
    ) -> None:
        runner = Runner()
        await self._turn(
            runner, self._agent(tmp_path, first), scope="conversation", accept_proposal=True
        )
        events = await self._turn(runner, self._agent(tmp_path, second))
        assert not any(e.event_type is ExecutionEventType.TOOL_NEEDS_FEEDBACK for e in events)

    @pytest.mark.asyncio
    async def test_a_proposal_that_misses_its_own_call_is_not_offered(self, tmp_path: Any) -> None:
        """``[1]`` is a glob class: ``$ ./run[1].sh *`` misses ``./run[1].sh``."""
        events = await self._turn(Runner(), self._agent(tmp_path, "./run[1].sh"))
        asked = next(e for e in events if e.event_type is ExecutionEventType.TOOL_NEEDS_FEEDBACK)
        assert asked.metadata["approval_pattern"] is None

    @pytest.mark.asyncio
    async def test_a_long_command_still_gets_a_proposal(self, tmp_path: Any) -> None:
        command = f"OUT=/{'x' * 200} echo Bologna"
        events = await self._turn(Runner(), self._agent(tmp_path, command))
        asked = next(e for e in events if e.event_type is ExecutionEventType.TOOL_NEEDS_FEEDBACK)
        assert asked.metadata["approval_pattern"] == f"$ OUT=/{'x' * 200} echo *"

    @pytest.mark.asyncio
    async def test_a_pattern_does_not_cover_what_is_chained_to_it(self, tmp_path: Any) -> None:
        runner = Runner()
        await self._turn(
            runner, self._agent(tmp_path, "echo one"), scope="conversation", accept_proposal=True
        )
        events = await self._turn(runner, self._agent(tmp_path, "echo two && rm -rf build"))
        assert any(e.event_type is ExecutionEventType.TOOL_NEEDS_FEEDBACK for e in events)

    @pytest.mark.asyncio
    async def test_one_pattern_covers_python_behind_glue(
        self, tmp_path: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        for name in ("S", "OUT"):
            monkeypatch.delenv(name, raising=False)
        runner = Runner()
        await self._turn(
            runner, self._agent(tmp_path, "python a.py"), scope="conversation", accept_proposal=True
        )
        script = 'S=/skill\nOUT=/run\npython "$S/profile.py" --out "$OUT" 2>&1 | tail -30'
        events = await self._turn(runner, self._agent(tmp_path, script))
        assert not any(e.event_type is ExecutionEventType.TOOL_NEEDS_FEEDBACK for e in events)


class TestAutoApprove:
    def _repl(self, *script: str):  # type: ignore[no-untyped-def]
        import io

        from rich.console import Console

        from domyn.agents.interfaces.cli import Theme
        from domyn.agents.interfaces.cli.repl import Repl
        from tests.cli.test_repl_session import ScriptedEditor, _agent

        buffer = io.StringIO()
        repl = Repl(
            _agent(),
            Console(file=buffer, width=100),
            Theme.build(allow_animation=False),
            enable_enhanced_input=False,
        )
        repl._input._session = ScriptedEditor(*script)
        return repl, buffer

    @pytest.mark.asyncio
    async def test_it_toggles(self) -> None:
        repl, _ = self._repl("/auto", "/auto")
        await repl.run()
        assert repl._state.auto_approve is False

    @pytest.mark.asyncio
    async def test_turning_it_on_says_what_it_means(self) -> None:
        """It removes the last thing standing between the model and the shell;
        that deserves a sentence, not a checkmark."""
        repl, buffer = self._repl("/auto")
        await repl.run()
        output = buffer.getvalue()
        assert "without asking" in output
        assert "hardline" in output

    @pytest.mark.asyncio
    async def test_it_starts_off(self) -> None:
        repl, _ = self._repl()
        await repl.run()
        assert repl._state.auto_approve is False

    @pytest.mark.asyncio
    async def test_an_auto_approval_is_still_shown(self) -> None:
        """The point is to stop being interrupted, not to stop being told: a
        command that ran unseen is one nobody can notice was wrong."""
        repl, buffer = self._repl()
        repl._state.auto_approve = True
        event = BaseEvent(
            event_type=ExecutionEventType.TOOL_NEEDS_FEEDBACK,
            author="a",
            metadata={"approval_subject": "$ rm -rf build"},
            action=_feedback_action(),
        )
        out = repl._approve_automatically(event)
        assert out.action.approve is True
        assert "auto-approved" in buffer.getvalue()
        assert "$ rm -rf build" in buffer.getvalue()

    @pytest.mark.asyncio
    async def test_it_never_creates_a_standing_grant(self) -> None:
        """It is the front-end saying yes, not a permission being recorded —
        so there is nothing left to revoke when the session ends."""
        repl, _ = self._repl()
        repl._state.auto_approve = True
        event = BaseEvent(
            event_type=ExecutionEventType.TOOL_NEEDS_FEEDBACK,
            author="a",
            metadata={"approval_subject": "$ x"},
            action=_feedback_action(),
        )
        assert repl._approve_automatically(event).action.remember_scope is None


def _feedback_action() -> Any:
    from domyn.agents.core import FeedbackAction

    return FeedbackAction(name="run_command", approval_type=ApprovalType.USER_CONFIRM)


@pytest.mark.asyncio
async def test_a_hardline_command_never_reaches_auto_approve() -> None:
    """Auto-approve answers the prompt; hardline is decided before there is
    one, so the catastrophic cases are refused even with it on."""
    grant = ApprovalGrant(value="A::run_command::$ rm *", kind="pattern")
    policy = ApprovalPolicy()
    assert grant.matches("A::run_command::$ rm -rf /")
    assert (await policy.resolve(_request("$ rm -rf /"))).is_deny


def test_a_pattern_of_none_declines_rather_than_proposing_the_string_none() -> None:
    from domyn.agents.approvals import build_request
    from domyn.agents.runner import _proposed_pattern

    tool = type("T", (), {"name": "t", "approval_pattern": staticmethod(lambda **_: None)})()
    assert _proposed_pattern(tool, build_request(tool, {})) is None
