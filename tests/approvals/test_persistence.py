"""Tests for grants that outlive the process, and for taking them back.

Persistence turns an approval into something nobody will look at again unless
something goes wrong, so the properties worth pinning are less about storing and
more about the three ways a permissions file betrays its owner: it survives when
it should not, it cannot be removed, or it answers for the wrong person.

Real files throughout. The point of this store is what happens on disk — the
mode bits, the atomic replace, the corrupt-file path — and a mocked filesystem
would test the mock.
"""

import stat
from pathlib import Path
from typing import Any

import pytest
import yaml

from domyn.agents.approvals import (
    ApprovalGrant,
    ApprovalPolicy,
    ApprovalRequest,
    ApprovalScope,
    ConversationApprovalStore,
    FileApprovalStore,
    InMemoryApprovalStore,
)


def _request(subject: str, **kwargs: Any) -> ApprovalRequest:
    kwargs.setdefault("tool_name", "run_command")
    kwargs.setdefault("agent_name", "Assistant")
    return ApprovalRequest(subject=subject, **kwargs)


@pytest.fixture
def path(tmp_path: Path) -> Path:
    return tmp_path / "approvals.yaml"


class TestItActuallyPersists:
    @pytest.mark.asyncio
    async def test_a_new_store_on_the_same_file_sees_the_grant(self, path: Path) -> None:
        """The entire reason this class exists: a separate process must find it."""
        await FileApprovalStore(path).grant(_request("git status"))
        assert await FileApprovalStore(path).is_granted(_request("git status"))

    @pytest.mark.asyncio
    async def test_nothing_is_written_until_something_is_granted(self, path: Path) -> None:
        """A tool that creates files under a home directory just by starting is a
        tool people stop trusting with their home directory."""
        store = FileApprovalStore(path)
        assert await store.granted() == []
        assert not await store.is_granted(_request("git status"))
        assert not path.exists()

    @pytest.mark.asyncio
    async def test_the_file_is_readable_by_a_human(self, path: Path) -> None:
        """A permissions file that only its writer can read cannot be audited."""
        await FileApprovalStore(path).grant(_request("git status"))
        parsed = yaml.safe_load(path.read_text())
        assert parsed["version"] == 1
        entry = parsed["grants"][0]
        assert entry["subject"] == "git status"
        assert entry["granted_at"], "a standing permission has to say when it was given"

    @pytest.mark.asyncio
    async def test_it_is_owner_only(self, path: Path) -> None:
        await FileApprovalStore(path).grant(_request("git status"))
        assert stat.S_IMODE(path.stat().st_mode) == 0o600

    @pytest.mark.asyncio
    async def test_re_approving_does_not_pile_up_duplicates(self, path: Path) -> None:
        """A list nobody can read is a list nobody revokes."""
        store = FileApprovalStore(path)
        for _ in range(3):
            await store.grant(_request("git status"))
        assert len(await store.granted()) == 1

    @pytest.mark.asyncio
    async def test_no_temporary_files_are_left_behind(self, path: Path) -> None:
        """The write goes via a sibling and a rename; a crash-free run must not
        leave the sibling."""
        store = FileApprovalStore(path)
        await store.grant(_request("git status"))
        await store.grant(_request("git diff"))
        assert sorted(p.name for p in path.parent.iterdir()) == ["approvals.yaml"]

    @pytest.mark.asyncio
    async def test_scopes_do_not_share_a_file(self, tmp_path: Path) -> None:
        project = FileApprovalStore(tmp_path / "p.yaml", ApprovalScope.PROJECT)
        user = FileApprovalStore(tmp_path / "u.yaml", ApprovalScope.USER)
        await project.grant(_request("pytest"))
        assert await project.is_granted(_request("pytest"))
        assert not await user.is_granted(_request("pytest")), "a repo's rules must not go global"


class TestTheCliProjectScope:
    """Project grants live in the user's home, never in the repo."""

    @pytest.fixture(autouse=True)
    def home(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
        from domyn.agents.interfaces.cli import config as cfg

        home = tmp_path / "home"
        home.mkdir()
        monkeypatch.setenv("HOME", str(home))
        monkeypatch.setattr(cfg, "GLOBAL_DIR", home / ".domyn")
        return home

    @staticmethod
    def _policy_in(directory: Path, monkeypatch: pytest.MonkeyPatch) -> ApprovalPolicy:
        from domyn.agents.interfaces.cli import config as cfg

        directory.mkdir(parents=True, exist_ok=True)
        monkeypatch.chdir(directory)
        return cfg.build_approval_policy()

    @pytest.mark.asyncio
    async def test_a_repo_cannot_ship_its_own_grants(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        repo = tmp_path / "cloned"
        (repo / ".domyn").mkdir(parents=True)
        (repo / ".domyn" / "approvals.yaml").write_text(
            "version: 1\ngrants:\n- {value: '*', kind: pattern, user_id: local-user}\n"
        )
        policy = self._policy_in(repo, monkeypatch)
        assert not (
            await policy.resolve(_request("$ rm -rf ~/projects", user_id="local-user"))
        ).is_allow

    @pytest.mark.asyncio
    async def test_two_directories_with_one_name_keep_their_own_grants(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mine = self._policy_in(tmp_path / "work" / "app", monkeypatch)
        assert await mine.remember(_request("$ make", user_id="u"), ApprovalScope.PROJECT)

        again = self._policy_in(tmp_path / "work" / "app", monkeypatch)
        assert (await again.resolve(_request("$ make", user_id="u"))).is_allow, (
            "kept for that directory"
        )
        other = self._policy_in(tmp_path / "downloads" / "app", monkeypatch)
        assert not (await other.resolve(_request("$ make", user_id="u"))).is_allow

    def test_from_home_there_is_no_project_scope(
        self, home: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        policy = self._policy_in(home, monkeypatch)
        assert policy.available_scopes() == [ApprovalScope.CONVERSATION, ApprovalScope.USER]


class TestRevocation:
    """The half that makes the other half acceptable."""

    @pytest.mark.asyncio
    async def test_revoking_stops_it_being_granted(self, path: Path) -> None:
        store = FileApprovalStore(path)
        await store.grant(_request("git push"))
        grant = (await store.granted())[0]
        assert await store.revoke(grant) is True
        assert not await store.is_granted(_request("git push"))

    @pytest.mark.asyncio
    async def test_revoking_survives_a_reload(self, path: Path) -> None:
        """Removed from the file, not only from an in-memory copy of it."""
        store = FileApprovalStore(path)
        await store.grant(_request("git push"))
        await store.revoke((await store.granted())[0])
        assert await FileApprovalStore(path).granted() == []

    @pytest.mark.asyncio
    async def test_revoking_a_pattern_closes_the_door_it_opened(self, path: Path) -> None:
        """The case the whole feature is for. A pattern grant permits calls the
        user has never read, so the ability to take it back is not a
        convenience."""
        store = FileApprovalStore(path)
        await store.grant(_request("git status"), pattern="git *")
        assert await store.is_granted(_request("git rebase -i main"))
        await store.revoke((await store.granted())[0])
        assert not await store.is_granted(_request("git rebase -i main"))

    @pytest.mark.asyncio
    async def test_revoking_leaves_the_others_alone(self, path: Path) -> None:
        store = FileApprovalStore(path)
        for subject in ("git status", "git diff", "pytest"):
            await store.grant(_request(subject))
        await store.revoke(next(g for g in await store.granted() if g.subject == "git diff"))
        assert sorted(g.subject or "" for g in await store.granted()) == ["git status", "pytest"]

    @pytest.mark.asyncio
    async def test_revoking_something_absent_says_so(self, path: Path) -> None:
        """Reported rather than silently succeeding, so a caller can tell the
        user the truth instead of confirming a removal that did not happen."""
        store = FileApprovalStore(path)
        await store.grant(_request("git status"))
        assert await store.revoke(ApprovalGrant(value="nope::nope::nope")) is False
        assert len(await store.granted()) == 1

    @pytest.mark.asyncio
    async def test_clearing_removes_the_file(self, path: Path) -> None:
        store = FileApprovalStore(path)
        await store.grant(_request("git status"))
        await store.revoke_all()
        assert not path.exists(), "an empty grants list invites 'did that fail?'"

    @pytest.mark.asyncio
    async def test_clearing_when_there_is_nothing_is_not_an_error(self, path: Path) -> None:
        await FileApprovalStore(path).revoke_all()

    @pytest.mark.parametrize(
        "store_factory",
        [
            lambda: InMemoryApprovalStore(),
            lambda: ConversationApprovalStore(),
        ],
        ids=["memory", "conversation"],
    )
    @pytest.mark.asyncio
    async def test_every_shipped_store_can_revoke(self, store_factory: Any) -> None:
        """Not only the file one. A store that can create a standing permission
        and not remove one is a one-way door, wherever it keeps things."""
        store = store_factory()

        class Ctx:
            def __init__(self) -> None:
                self.state: dict[str, Any] = {}

        context = Ctx()
        await store.grant(_request("git status"), context)
        grant = (await store.granted(context))[0]
        assert await store.revoke(grant, context) is True
        assert not await store.is_granted(_request("git status"), context)


class TestItFailsSafe:
    @pytest.mark.asyncio
    async def test_a_corrupt_file_grants_nothing(self, path: Path) -> None:
        """Extra prompts, not free passes — and above all not a crash on startup,
        which is what pushes people into deleting the file wholesale."""
        path.write_text("this: is: not: valid: yaml: [[[")
        store = FileApprovalStore(path)
        assert await store.granted() == []
        assert not await store.is_granted(_request("git status"))

    @pytest.mark.asyncio
    async def test_a_truncated_file_grants_nothing(self, path: Path) -> None:
        path.write_text("version: 1\ngrants:\n  - value: 'a::b::c'\n    kind: exa")
        assert await FileApprovalStore(path).granted() == []

    @pytest.mark.asyncio
    async def test_one_bad_record_does_not_discard_the_good_ones(self, path: Path) -> None:
        """The others are permissions the user gave and still expects to hold."""
        path.write_text(
            yaml.safe_dump(
                {
                    "version": 1,
                    "grants": [
                        {"value": "Assistant::run_command::git status", "kind": "exact"},
                        {"kind": "exact"},  # no value at all
                        "not even a mapping",
                    ],
                }
            )
        )
        store = FileApprovalStore(path)
        assert len(await store.granted()) == 1
        assert await store.is_granted(_request("git status"))

    @pytest.mark.asyncio
    async def test_a_directory_where_the_file_should_be_is_survivable(self, tmp_path: Path) -> None:
        """Reads must not raise; the write is reported and dropped."""
        target = tmp_path / "approvals.yaml"
        target.mkdir()
        store = FileApprovalStore(target)
        assert await store.granted() == []
        await store.grant(_request("git status"))  # must not raise
        assert not await store.is_granted(_request("git status"))

    @pytest.mark.asyncio
    async def test_an_unwritable_directory_does_not_crash_the_session(self, tmp_path: Path) -> None:
        locked = tmp_path / "locked"
        locked.mkdir(mode=0o500)
        try:
            store = FileApprovalStore(locked / "approvals.yaml")
            await store.grant(_request("git status"))
            assert not await store.is_granted(_request("git status"))
        finally:
            locked.chmod(0o700)

    @pytest.mark.asyncio
    async def test_a_broken_store_does_not_break_the_gate(self, tmp_path: Path) -> None:
        """A policy holding an unusable store still asks, and still honours the
        stores that do work."""
        broken = tmp_path / "dir-not-file"
        broken.mkdir()
        policy = ApprovalPolicy(
            stores=[FileApprovalStore(broken), FileApprovalStore(tmp_path / "ok.yaml")]
        )
        await policy.remember(_request("git status"), ApprovalScope.USER)
        # The first store owns USER and swallowed it, so nothing is granted — but
        # resolving must still answer rather than raise.
        assert (await policy.resolve(_request("git status"))).decision == "ask"


class TestOneUserDoesNotAnswerForAnother:
    """One file per machine is the normal case. "Normal case" is not a security
    argument, and a store shared by a server would be exactly where this bites."""

    @pytest.mark.asyncio
    async def test_a_grant_is_scoped_to_who_gave_it(self, path: Path) -> None:
        store = FileApprovalStore(path)
        await store.grant(_request("git push", user_id="alice"))
        assert await store.is_granted(_request("git push", user_id="alice"))
        assert not await store.is_granted(_request("git push", user_id="bob"))

    @pytest.mark.asyncio
    async def test_two_users_can_hold_the_same_permission(self, path: Path) -> None:
        """Deduplication is per person: collapsing them would make one user's
        revocation silently remove the other's."""
        store = FileApprovalStore(path)
        await store.grant(_request("git push", user_id="alice"))
        await store.grant(_request("git push", user_id="bob"))
        assert len(await store.granted()) == 2
        await store.revoke(next(g for g in await store.granted() if g.user_id == "alice"))
        assert not await store.is_granted(_request("git push", user_id="alice"))
        assert await store.is_granted(_request("git push", user_id="bob"))

    @pytest.mark.asyncio
    async def test_listing_shows_everyone(self, path: Path) -> None:
        """An audit that hid other people's standing permissions would be worse
        than no audit."""
        store = FileApprovalStore(path)
        await store.grant(_request("git push", user_id="alice"))
        await store.grant(_request("pytest", user_id="bob"))
        assert len(await store.granted()) == 2
        assert len(await store.granted(user_id="alice")) == 1


class TestThroughThePolicy:
    @pytest.mark.asyncio
    async def test_remember_writes_to_the_store_owning_the_scope(self, tmp_path: Path) -> None:
        project = FileApprovalStore(tmp_path / "p.yaml", ApprovalScope.PROJECT)
        user = FileApprovalStore(tmp_path / "u.yaml", ApprovalScope.USER)
        policy = ApprovalPolicy(stores=[project, user])

        assert await policy.remember(_request("pytest"), ApprovalScope.USER) is True
        assert await user.granted() and not await project.granted()

    @pytest.mark.asyncio
    async def test_a_persisted_grant_short_circuits_the_prompt(self, tmp_path: Path) -> None:
        store = FileApprovalStore(tmp_path / "u.yaml")
        policy = ApprovalPolicy(stores=[store])
        assert (await policy.resolve(_request("git status"))).decision == "ask"
        await policy.remember(_request("git status"), ApprovalScope.USER)
        verdict = await policy.resolve(_request("git status"))
        assert verdict.is_allow
        assert verdict.rule == "grant:user"

    @pytest.mark.asyncio
    async def test_hardline_still_wins_over_a_persisted_grant(self, tmp_path: Path) -> None:
        """A stored yes must not be a way to talk the gate out of the rules that
        exist precisely so an accident cannot happen."""
        store = FileApprovalStore(tmp_path / "u.yaml")
        policy = ApprovalPolicy(stores=[store])
        await policy.remember(_request("mkfs.ext4 /dev/disk2"), ApprovalScope.USER)
        assert (await policy.resolve(_request("mkfs.ext4 /dev/disk2"))).is_deny

    @pytest.mark.asyncio
    async def test_a_scope_with_no_store_is_not_offered(self, tmp_path: Path) -> None:
        policy = ApprovalPolicy(stores=[FileApprovalStore(tmp_path / "u.yaml")])
        assert ApprovalScope.PROJECT not in policy.available_scopes()
        assert await policy.remember(_request("pytest"), ApprovalScope.PROJECT) is False
