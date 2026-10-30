"""Tests for ``/approvals``: seeing what you agreed to, and taking it back.

The command exists because approving happens in a hurry and reviewing happens
later. So the properties that matter are not about formatting: nothing may be
hidden, an index must mean one unambiguous thing, and a removal must either
happen or say that it did not. A review screen that quietly under-reports is
worse than no review screen, because it is believed.
"""

# No `from __future__ import annotations`: builds real agents and tools.
import io
from pathlib import Path
from typing import Any

import pytest
from rich.console import Console

from domyn.agents.approvals import (
    ApprovalPolicy,
    ApprovalRequest,
    ApprovalScope,
    ConversationApprovalStore,
    FileApprovalStore,
)
from domyn.agents.interfaces.cli import Theme, grants
from domyn.agents.interfaces.cli.repl import Repl
from tests.cli.test_repl_session import ScriptedEditor, _agent


def _request(subject: str, **kwargs: Any) -> ApprovalRequest:
    kwargs.setdefault("tool_name", "run_command")
    kwargs.setdefault("agent_name", "Assistant")
    return ApprovalRequest(subject=subject, **kwargs)


def _policy(tmp_path: Path) -> ApprovalPolicy:
    return ApprovalPolicy(
        stores=[
            ConversationApprovalStore(),
            FileApprovalStore(tmp_path / "project.yaml", ApprovalScope.PROJECT),
            FileApprovalStore(tmp_path / "user.yaml", ApprovalScope.USER),
        ]
    )


class _Ctx:
    def __init__(self) -> None:
        self.state: dict[str, Any] = {}


class TestCollecting:
    @pytest.mark.asyncio
    async def test_nothing_granted_is_an_empty_list(self, tmp_path: Path) -> None:
        assert await grants.collect(_policy(tmp_path), context=_Ctx()) == []

    @pytest.mark.asyncio
    async def test_grants_from_every_scope_are_listed(self, tmp_path: Path) -> None:
        """The one thing this must never do is hide a standing permission."""
        policy = _policy(tmp_path)
        context = _Ctx()
        await policy.remember(_request("git status"), ApprovalScope.CONVERSATION, context)
        await policy.remember(_request("pytest"), ApprovalScope.PROJECT, context)
        await policy.remember(_request("curl example.com"), ApprovalScope.USER, context)

        found = await grants.collect(policy, context=context)
        assert [e.scope for e in found] == ["conversation", "project", "user"]
        assert [e.label for e in found] == ["git status", "pytest", "curl example.com"]

    @pytest.mark.asyncio
    async def test_indices_run_from_one_without_gaps(self, tmp_path: Path) -> None:
        """``revoke 3`` has to mean the third line the user is looking at."""
        policy = _policy(tmp_path)
        context = _Ctx()
        for scope in (ApprovalScope.CONVERSATION, ApprovalScope.PROJECT, ApprovalScope.USER):
            for subject in ("one", "two"):
                await policy.remember(_request(f"{scope}-{subject}"), scope, context)
        found = await grants.collect(policy, context=context)
        assert [e.index for e in found] == list(range(1, 7))

    @pytest.mark.asyncio
    async def test_narrowest_scope_comes_first(self, tmp_path: Path) -> None:
        """Same order the prompt offers them in, so the list reads the way the
        choice was made."""
        policy = ApprovalPolicy(
            stores=[
                FileApprovalStore(tmp_path / "u.yaml", ApprovalScope.USER),
                ConversationApprovalStore(),
                FileApprovalStore(tmp_path / "p.yaml", ApprovalScope.PROJECT),
            ]
        )
        context = _Ctx()
        for scope in (ApprovalScope.USER, ApprovalScope.CONVERSATION, ApprovalScope.PROJECT):
            await policy.remember(_request(str(scope)), scope, context)
        assert [e.scope for e in await grants.collect(policy, context=context)] == [
            "conversation",
            "project",
            "user",
        ]

    @pytest.mark.asyncio
    async def test_a_broken_store_does_not_hide_the_working_ones(self, tmp_path: Path) -> None:
        class Exploding(ConversationApprovalStore):
            def granted(self, context: Any = None) -> list[Any]:
                raise RuntimeError("store is broken")

        policy = ApprovalPolicy(
            stores=[Exploding(), FileApprovalStore(tmp_path / "u.yaml", ApprovalScope.USER)]
        )
        await policy.remember(_request("pytest"), ApprovalScope.USER)
        found = await grants.collect(policy, context=_Ctx())
        assert [e.label for e in found] == ["pytest"]


class TestRendering:
    def _render(self, entries: list[grants.Entry]) -> str:
        buffer = io.StringIO()
        console = Console(file=buffer, width=100, force_terminal=False)
        grants.render(console, Theme.build(allow_animation=False), entries)
        return buffer.getvalue()

    @pytest.mark.asyncio
    async def test_empty_says_so_rather_than_printing_nothing(self, tmp_path: Path) -> None:
        """A blank response looks like a command that failed."""
        out = self._render(await grants.collect(_policy(tmp_path), context=_Ctx()))
        assert "No standing approvals" in out

    @pytest.mark.asyncio
    async def test_a_pattern_grant_is_called_out(self, tmp_path: Path) -> None:
        """``git status`` and ``git *`` are one character apart in a list and
        worlds apart in what they permit."""
        policy = _policy(tmp_path)
        context = _Ctx()
        await policy.remember(_request("git status"), ApprovalScope.USER, context, pattern="git *")
        out = self._render(await grants.collect(policy, context=context))
        assert "matches anything like it" in out
        assert "match by pattern" in out

    @pytest.mark.asyncio
    async def test_an_exact_grant_is_not_called_out(self, tmp_path: Path) -> None:
        policy = _policy(tmp_path)
        context = _Ctx()
        await policy.remember(_request("git status"), ApprovalScope.USER, context)
        out = self._render(await grants.collect(policy, context=context))
        assert "matches anything like it" not in out
        assert "git status" in out

    @pytest.mark.asyncio
    async def test_it_shows_how_to_remove_one(self, tmp_path: Path) -> None:
        policy = _policy(tmp_path)
        context = _Ctx()
        await policy.remember(_request("pytest"), ApprovalScope.PROJECT, context)
        assert "/approvals revoke" in self._render(await grants.collect(policy, context=context))


class TestParsingTheSelector:
    @pytest.mark.parametrize(
        "argument,expected",
        [("1", [1]), ("2", [2]), ("1 3", [1, 3]), ("1,3", [1, 3]), ("2 2", [2])],
    )
    def test_valid_selectors(self, argument: str, expected: list[int]) -> None:
        assert grants.parse_selector(argument, 3) == (expected, None)

    @pytest.mark.parametrize("argument", ["", "   "])
    def test_no_argument_asks_which_one(self, argument: str) -> None:
        wanted, problem = grants.parse_selector(argument, 3)
        assert wanted == [] and problem and "which one" in problem

    @pytest.mark.parametrize("argument", ["0", "4", "99"])
    def test_out_of_range_is_refused(self, argument: str) -> None:
        wanted, problem = grants.parse_selector(argument, 3)
        assert wanted == [] and problem and "no " in problem

    @pytest.mark.parametrize("argument", ["all", "git status", "-1", "1.5"])
    def test_non_numeric_is_refused(self, argument: str) -> None:
        wanted, problem = grants.parse_selector(argument, 3)
        assert wanted == [] and problem

    def test_a_partly_valid_selector_does_nothing_at_all(self) -> None:
        """ "revoke 2 9" quietly removing only entry 2 would leave the user
        believing a permission is gone when it is not."""
        wanted, problem = grants.parse_selector("2 9", 3)
        assert wanted == [] and problem


class TestThroughTheRepl:
    """End to end via the command, because the wiring is where this breaks: a
    store the REPL cannot reach shows an empty list and looks like good news."""

    def _repl(self, tmp_path: Path, *script: str) -> tuple[Repl, io.StringIO, ApprovalPolicy]:
        from domyn.agents import Runner

        policy = _policy(tmp_path)
        buffer = io.StringIO()
        console = Console(file=buffer, width=100, force_terminal=False)
        repl = Repl(
            _agent(),
            console,
            Theme.build(allow_animation=False),
            runner=Runner(approval_policy=policy),
            enable_enhanced_input=False,
        )
        repl._input._session = ScriptedEditor(*script)
        return repl, buffer, policy

    @pytest.mark.asyncio
    async def test_listing_shows_a_persisted_grant(self, tmp_path: Path) -> None:
        repl, buffer, policy = self._repl(tmp_path, "/approvals")
        await policy.remember(_request("curl example.com"), ApprovalScope.USER)
        await repl.run()
        assert "curl example.com" in buffer.getvalue()

    @pytest.mark.asyncio
    async def test_listing_shows_a_conversation_grant(self, tmp_path: Path) -> None:
        """The one that needs the state to be fetched from the conversation
        manager. Passing no context would have shown an empty list."""
        repl, buffer, policy = self._repl(tmp_path, "/approvals")
        # Granted the way a real session does: on the state the manager holds.
        await repl._runner.conversation_manager.create_conversation(
            application_name="domyn-cli",
            user_id="local-user",
            conversation_id=repl._state.conversation_id,
        )
        holder = await repl._conversation_state()
        await policy.remember(_request("git status"), ApprovalScope.CONVERSATION, holder)
        await repl._persist_conversation_state(holder)

        await repl.run()
        assert "git status" in buffer.getvalue()

    @pytest.mark.asyncio
    async def test_revoking_removes_it_from_disk(self, tmp_path: Path) -> None:
        repl, buffer, policy = self._repl(tmp_path, "/approvals revoke 1")
        await policy.remember(_request("git push"), ApprovalScope.USER)
        await repl.run()
        assert "removed" in buffer.getvalue()
        assert not (await policy.resolve(_request("git push"))).is_allow
        assert await FileApprovalStore(tmp_path / "user.yaml").granted() == []

    @pytest.mark.asyncio
    async def test_revoking_echoes_what_went_rather_than_a_count(self, tmp_path: Path) -> None:
        """Indices shift; "revoked 1" gives no way to notice the wrong line was
        removed."""
        repl, buffer, policy = self._repl(tmp_path, "/approvals revoke 2")
        await policy.remember(_request("git status"), ApprovalScope.USER)
        await policy.remember(_request("git push --force"), ApprovalScope.USER)
        await repl.run()
        output = buffer.getvalue()
        assert "removed" in output and "git push --force" in output
        assert (await policy.resolve(_request("git status"))).is_allow, "the other one must survive"

    @pytest.mark.asyncio
    async def test_clear_removes_everything(self, tmp_path: Path) -> None:
        repl, _, policy = self._repl(tmp_path, "/approvals clear")
        await policy.remember(_request("pytest"), ApprovalScope.PROJECT)
        await policy.remember(_request("git push"), ApprovalScope.USER)
        await repl.run()
        assert await grants.collect(policy, context=_Ctx()) == []

    @pytest.mark.asyncio
    async def test_revoking_with_nothing_stored_says_so(self, tmp_path: Path) -> None:
        repl, buffer, _ = self._repl(tmp_path, "/approvals revoke 1")
        await repl.run()
        assert "no standing approvals" in buffer.getvalue().lower()

    @pytest.mark.asyncio
    async def test_an_unknown_subcommand_is_reported(self, tmp_path: Path) -> None:
        repl, buffer, policy = self._repl(tmp_path, "/approvals delete 1")
        await policy.remember(_request("pytest"), ApprovalScope.USER)
        await repl.run()
        assert "Unknown" in buffer.getvalue()
        assert (await policy.resolve(_request("pytest"))).is_allow, (
            "a typo must not remove anything"
        )

    @pytest.mark.asyncio
    async def test_a_bad_index_leaves_everything_alone(self, tmp_path: Path) -> None:
        repl, buffer, policy = self._repl(tmp_path, "/approvals revoke 7")
        await policy.remember(_request("pytest"), ApprovalScope.USER)
        await repl.run()
        assert (await policy.resolve(_request("pytest"))).is_allow
        assert "7" in buffer.getvalue()

    @pytest.mark.asyncio
    async def test_the_session_survives_all_of_it(self, tmp_path: Path) -> None:
        """A command that ends the session on bad input would be worse than the
        problem it reports."""
        repl, _, policy = self._repl(
            tmp_path, "/approvals", "/approvals revoke x", "/approvals clear", "/approvals"
        )
        await policy.remember(_request("pytest"), ApprovalScope.USER)
        assert await repl.run() == 0


class TestItIsDiscoverable:
    def test_the_command_is_in_help_and_completion(self) -> None:
        """Both come from one list, and a permissions screen nobody can find is a
        permissions screen nobody uses."""
        from domyn.agents.interfaces.cli import commands as slash

        assert "/approvals" in slash.completion_names()
        assert any("/approvals" in invocation for invocation, _ in slash.help_lines())

    @pytest.mark.asyncio
    async def test_it_resolves_with_its_argument(self) -> None:
        from domyn.agents.interfaces.cli import commands as slash

        resolved = slash.resolve("/approvals revoke 2")
        assert resolved is not None
        command, argument = resolved
        assert command.name == "/approvals" and argument == "revoke 2"
