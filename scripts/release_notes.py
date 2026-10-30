#!/usr/bin/env python3
"""Write the release notes of one version tag, as markdown on stdout.

Usage: scripts/release_notes.py TAG --repo OWNER/REPO

Run from a full clone (every tag fetched), with `gh` authenticated. Every section
is computed rather than written, so it is right by construction:

  Highlights    the same pull requests grouped by the type of their Conventional
                Commit title (New Features for `feat:` and `add:`, Bug Fixes for `fix:`, Removed
                for `remove:`...),
                breaking changes first, each as its description with its number
  Changelog     every pull request merged since the previous release, one line
                each (`- #622 fix: ... @author`), oldest first, after the link
                comparing the two tags
  New Contributors  the authors contributing for the first time, when there are

There is no list of contributors: GitHub draws one under every release, with
their avatars, from the users the notes mention. A bot is written without `@`
(`by dependabot`), so the avatars are the people's.

The previous release is the nearest stable tag (vX.Y.Z) behind TAG in history,
so a pre-release lists everything since the last stable one and a stable release
everything since the stable before it, whatever candidates came in between. With
none, the history up to TAG is listed.

One pull request is one commit on main: the repository squash-merges, with the
title of the pull request as the subject and `(#N)` at its end. A commit without
a number (pushed directly) is listed by its short hash; the version bumps of
`make version-*` are left out.
"""

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass

PR_SUFFIX = re.compile(r"^(?P<title>.*?)\s*\(#(?P<number>\d+)\)$")
VERSION_BUMP = re.compile(r"^Bump version to \S+$")
CONVENTIONAL = re.compile(
    r"^(?P<type>[a-z]+)(?:\((?P<scope>[^)]+)\))?(?P<breaking>!)?: (?P<description>.+)$"
)

BREAKING = "⚠️ Breaking Changes"
OTHER = "Other Changes"
# In the order they appear. A type not listed here, or a title that is not a
# Conventional Commit (from before the check on titles), is an Other Change.
SECTIONS = {
    "feat": "New Features",
    "add": "New Features",
    "fix": "Bug Fixes",
    "perf": "Performance",
    "remove": "Removed",
    "docs": "Documentation",
    "change": "Chores",
    "refactor": "Chores",
    "test": "Chores",
    "build": "Chores",
    "ci": "Chores",
    "chore": "Chores",
    "revert": "Chores",
}


@dataclass(frozen=True)
class Entry:
    title: str
    number: int | None
    sha: str
    author: str | None = None
    is_bot: bool = False

    @property
    def handle(self) -> str | None:
        # `dependabot[bot]` is @dependabot on GitHub.
        return self.author.removesuffix("[bot]") if self.author else None


def _run(*args: str) -> str:
    return subprocess.run(args, check=True, capture_output=True, text=True).stdout


def previous_release(tag: str) -> str | None:
    """The nearest stable tag behind `tag`: `v` and nothing but digits and dots."""
    try:
        found = _run(
            "git", "describe", "--tags", "--abbrev=0",
            "--match", "v*", "--exclude", "v*[!0-9.]*", f"{tag}^",
        )  # fmt: skip
    except subprocess.CalledProcessError:
        return None
    return found.strip() or None


def parse_log(log: str) -> list[Entry]:
    """`git log --format=%H%x00%s` output, newest first, into entries oldest first."""
    entries = []
    for line in reversed(log.splitlines()):
        if not line.strip():
            continue
        sha, _, subject = line.partition("\0")
        if VERSION_BUMP.match(subject):
            continue
        match = PR_SUFFIX.match(subject)
        if match:
            entries.append(Entry(match["title"], int(match["number"]), sha))
        else:
            entries.append(Entry(subject, None, sha))
    return entries


def commits(previous: str | None, tag: str) -> list[Entry]:
    span = f"{previous}..{tag}" if previous else tag
    return parse_log(_run("git", "log", "--first-parent", "--format=%H%x00%s", span))


def with_authors(entries: list[Entry], repo: str) -> list[Entry]:
    """The author of each pull request, or of the commit when there is none."""
    found = []
    for entry in entries:
        path = (
            f"repos/{repo}/pulls/{entry.number}"
            if entry.number
            else f"repos/{repo}/commits/{entry.sha}"
        )
        jq = (
            ".user | [.login, .type] | @tsv"
            if entry.number
            else '.author // {} | [.login // "", .type // ""] | @tsv'
        )
        login, _, kind = _run("gh", "api", path, "--jq", jq).strip().partition("\t")
        found.append(
            Entry(entry.title, entry.number, entry.sha, login or None, kind.strip() == "Bot")
        )
    return found


def new_contributors(notes: str) -> list[str]:
    """The "New Contributors" lines of GitHub's own generated notes, which know who
    had never contributed before; empty when there is no such section."""
    lines: list[str] = []
    inside = False
    for line in notes.splitlines():
        if line.startswith("## "):
            inside = line.strip() == "## New Contributors"
            continue
        if line.startswith("**Full Changelog**"):
            inside = False
        # A bot's first pull request (`@dependabot[bot] made their first...`)
        # is no one's first contribution.
        if inside and line.strip() and "[bot]" not in line:
            lines.append(line.strip())
    return lines


def github_new_contributors(repo: str, tag: str, previous: str | None) -> list[str]:
    args = [
        "gh",
        "api",
        "-X",
        "POST",
        f"repos/{repo}/releases/generate-notes",
        "-f",
        f"tag_name={tag}",
    ]
    if previous:
        args += ["-f", f"previous_tag_name={previous}"]
    return new_contributors(json.loads(_run(*args))["body"])


def highlights(entries: list[Entry]) -> list[str]:
    """One section per kind of change, in merge order within each."""
    sections: dict[str, list[str]] = {
        name: [] for name in (BREAKING, *dict.fromkeys(SECTIONS.values()), OTHER)
    }
    for entry in entries:
        ref = f"#{entry.number}" if entry.number else entry.sha[:7]
        match = CONVENTIONAL.match(entry.title)
        # A prefix that is not a known type (`add:`) stays in the line: without
        # it the description may not read as a change at all.
        if match is None or (match["type"] not in SECTIONS and not match["breaking"]):
            sections[OTHER].append(f"- {entry.title} ({ref})")
            continue
        description = match["description"]
        scope = f"**{match['scope']}:** " if match["scope"] else ""
        line = f"- {scope}{description[0].upper()}{description[1:]} ({ref})"
        section = BREAKING if match["breaking"] else SECTIONS[match["type"]]
        sections[section].append(line)

    out: list[str] = []
    for name, lines in sections.items():
        if lines:
            out += [f"## {name}", "", *lines, ""]
    return out


def render(
    repo: str, tag: str, previous: str | None, entries: list[Entry], newcomers: list[str]
) -> str:
    out = [*highlights(entries), "## Changelog", ""]
    if previous:
        out += [f"**Full Changelog**: https://github.com/{repo}/compare/{previous}...{tag}", ""]
    for entry in entries:
        ref = f"#{entry.number}" if entry.number else entry.sha[:7]
        by = ""
        if entry.handle:
            by = f" by {entry.handle}" if entry.is_bot else f" @{entry.handle}"
        out.append(f"- {ref} {entry.title}{by}")
    if not entries:
        out.append("No changes.")

    if newcomers:
        out += ["", "## New Contributors", "", *newcomers]
    return "\n".join(out) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("tag")
    parser.add_argument("--repo", required=True, help="OWNER/REPO")
    args = parser.parse_args(argv)

    previous = previous_release(args.tag)
    entries = with_authors(commits(previous, args.tag), args.repo)
    newcomers = github_new_contributors(args.repo, args.tag, previous)
    sys.stdout.write(render(args.repo, args.tag, previous, entries, newcomers))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
