import importlib.util
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).parents[2] / "scripts" / "release_notes.py"
spec = importlib.util.spec_from_file_location("release_notes", SCRIPT)
assert spec is not None and spec.loader is not None
release_notes = importlib.util.module_from_spec(spec)
# Registered before running it: a dataclass looks its module up by name.
sys.modules["release_notes"] = release_notes
spec.loader.exec_module(release_notes)
Entry = release_notes.Entry

REPO = "acme/widgets"


def git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True
    ).stdout


def history(tmp_path: Path, steps: list[str]) -> Path:
    """A repository with one commit per step, or a tag when a step starts with `v`."""
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "--quiet", "--initial-branch=main")
    git(repo, "config", "user.name", "Test")
    git(repo, "config", "user.email", "test@example.com")
    for step in steps:
        if step.startswith("v"):
            git(repo, "tag", step)
        else:
            git(repo, "commit", "--quiet", "--allow-empty", "-m", step)
    return repo


class TestThePreviousRelease:
    def test_is_the_nearest_stable_tag(self, tmp_path: Path, monkeypatch) -> None:
        repo = history(tmp_path, ["a", "v1.0.0", "b", "v1.1.0", "c", "v1.2.0"])
        monkeypatch.chdir(repo)
        assert release_notes.previous_release("v1.2.0") == "v1.1.0"

    def test_skips_the_candidates_in_between(self, tmp_path: Path, monkeypatch) -> None:
        repo = history(
            tmp_path, ["a", "v1.0.0", "b", "v1.1.0rc1", "c", "v1.1.0-rc2", "d", "v1.1.0"]
        )
        monkeypatch.chdir(repo)
        assert release_notes.previous_release("v1.1.0") == "v1.0.0"
        assert release_notes.previous_release("v1.1.0-rc2") == "v1.0.0"

    def test_is_none_for_the_first_release(self, tmp_path: Path, monkeypatch) -> None:
        repo = history(tmp_path, ["a", "v1.0.0"])
        monkeypatch.chdir(repo)
        assert release_notes.previous_release("v1.0.0") is None

    def test_lists_what_came_since(self, tmp_path: Path, monkeypatch) -> None:
        repo = history(
            tmp_path, ["a (#1)", "v1.0.0", "b (#2)", "Bump version to 1.0.1", "c", "v1.0.1"]
        )
        monkeypatch.chdir(repo)
        entries = release_notes.commits("v1.0.0", "v1.0.1")
        assert [(e.title, e.number) for e in entries] == [("b", 2), ("c", None)]


class TestTheLog:
    def test_oldest_first_with_the_pull_request_number(self) -> None:
        log = "bbb\0fix: two (#12)\naaa\0feat: one (#11)\n"
        assert release_notes.parse_log(log) == [
            Entry("feat: one", 11, "aaa"),
            Entry("fix: two", 12, "bbb"),
        ]

    def test_a_direct_commit_keeps_its_subject(self) -> None:
        assert release_notes.parse_log("abc\0hotfix the thing\n") == [
            Entry("hotfix the thing", None, "abc")
        ]

    def test_version_bumps_are_left_out(self) -> None:
        assert release_notes.parse_log("abc\0Bump version to 1.6.2\n") == []

    def test_only_a_trailing_number_is_the_pull_request(self) -> None:
        [entry] = release_notes.parse_log("abc\0fix: issue (#3) in the parser (#9)\n")
        assert (entry.title, entry.number) == ("fix: issue (#3) in the parser", 9)


class TestNewContributors:
    def test_the_section_of_githubs_notes(self) -> None:
        notes = (
            "## What's Changed\n* fix: x by @a in https://x/pull/1\n\n"
            "## New Contributors\n* @b made their first contribution in https://x/pull/2\n\n"
            "**Full Changelog**: https://x/compare/v1...v2"
        )
        assert release_notes.new_contributors(notes) == [
            "* @b made their first contribution in https://x/pull/2"
        ]

    def test_bots_are_left_out(self) -> None:
        notes = (
            "## New Contributors\n"
            "* @dependabot[bot] made their first contribution in https://x/pull/1\n"
            "* @b made their first contribution in https://x/pull/2\n"
        )
        assert release_notes.new_contributors(notes) == [
            "* @b made their first contribution in https://x/pull/2"
        ]

    def test_none_without_the_section(self) -> None:
        assert release_notes.new_contributors("## What's Changed\n* fix: x by @a\n") == []


ENTRIES = (
    Entry("feat: one", 11, "a" * 40, "alice"),
    Entry("build(deps): bump x", 12, "b" * 40, "dependabot[bot]", is_bot=True),
    Entry("fix: two", 13, "c" * 40, "bob"),
    Entry("fix: three", 14, "d" * 40, "alice"),
    Entry("hotfix", None, "e" * 40, None),
)


class TestTheNotes:
    def render(self, newcomers: list[str] | None = None) -> str:
        return release_notes.render(REPO, "v1.1.0", "v1.0.0", list(ENTRIES), newcomers or [])

    def test_every_change_with_its_author(self) -> None:
        notes = self.render()
        assert (
            "**Full Changelog**: https://github.com/acme/widgets/compare/v1.0.0...v1.1.0" in notes
        )
        assert (
            "- #11 feat: one @alice\n"
            "- #12 build(deps): bump x by dependabot\n"
            "- #13 fix: two @bob\n"
            "- #14 fix: three @alice\n"
            "- eeeeeee hotfix\n"
        ) in notes

    def test_a_bot_is_named_without_a_mention(self) -> None:
        # GitHub draws the avatars of every user the notes mention.
        notes = self.render()
        assert "- #12 build(deps): bump x by dependabot\n" in notes
        assert "@dependabot" not in notes

    def test_no_list_of_contributors(self) -> None:
        assert "Contributors" not in self.render()

    def test_new_contributors_close_the_notes(self) -> None:
        line = "* @bob made their first contribution in https://x/pull/13"
        assert self.render([line]).endswith(f"## New Contributors\n\n{line}\n")

    def test_the_first_release_has_no_comparison(self) -> None:
        notes = release_notes.render(REPO, "v1.0.0", None, list(ENTRIES[:1]), [])
        assert "Full Changelog" not in notes


class TestTheHighlights:
    def highlights(self, *titles: str) -> str:
        entries = [Entry(t, n, f"{n:040d}") for n, t in enumerate(titles, start=1)]
        return "\n".join(release_notes.highlights(entries))

    def test_grouped_by_type_in_a_fixed_order(self) -> None:
        text = self.highlights("ci: build faster", "fix: a crash", "feat: a command")
        assert text.index("## New Features") < text.index("## Bug Fixes") < text.index("## Chores")

    def test_each_line_is_the_description_with_its_number(self) -> None:
        text = self.highlights("feat: a new command")
        assert "## New Features\n\n- A new command (#1)\n" in text

    def test_the_scope_leads_the_line(self) -> None:
        assert "- **cli:** Resume by id (#1)" in self.highlights("feat(cli): resume by id")

    def test_breaking_changes_come_first_whatever_their_type(self) -> None:
        text = self.highlights("feat: a command", "fix!: drop the old flag")
        assert text.startswith("## ⚠️ Breaking Changes\n\n- Drop the old flag (#2)\n")

    def test_the_chore_types_share_one_section(self) -> None:
        text = self.highlights("ci: a", "build(deps): b", "refactor: c", "test: d", "chore: e")
        assert text.count("## ") == 1 and text.count("\n- ") == 5

    def test_any_other_title_is_an_other_change(self) -> None:
        text = self.highlights("update: a skill", "Update the readme")
        assert "## Other Changes\n\n- update: a skill (#1)\n- Update the readme (#2)" in text

    def test_our_own_types(self) -> None:
        text = self.highlights("add: a skill", "change: a default", "remove: the old flag")
        assert "## New Features\n\n- A skill (#1)\n" in text
        assert "## Removed\n\n- The old flag (#3)\n" in text
        assert "## Chores\n\n- A default (#2)\n" in text

    def test_empty_sections_are_left_out(self) -> None:
        assert "Bug Fixes" not in self.highlights("feat: a")

    def test_they_open_the_notes(self) -> None:
        notes = release_notes.render(REPO, "v1.1.0", "v1.0.0", list(ENTRIES), [])
        assert notes.index("## New Features") < notes.index("## Changelog")
