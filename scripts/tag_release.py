#!/usr/bin/env python3
"""Tag a release on the tip of origin/main, and push the tag.

Usage: scripts/tag_release.py patch|minor|major|alpha|VERSION [--yes]

A release is a tag and nothing else: the version is read from the tag when the
package is built (hatch-vcs), so no commit bumps it, and pushing the tag starts
.github/workflows/release.yml, which tests and only then publishes.

`patch`, `minor` and `major` count from the latest stable tag (v1.6.2 gives
v1.6.3, v1.7.0, v2.0.0). VERSION names one, for a pre-release: `1.7.0rc1`
(also `a1`, `b1`), in the PEP 440 spelling, so the tag is the package's version
as written. The tag is annotated and goes on origin/main as fetched now, whatever
is checked out here; it is pushed after a confirmation, unless `--yes`.

`alpha` is the nightly build (.github/workflows/nightly.yml): the next patch
after the latest stable tag, numbered after the alphas it already has (v1.6.2
gives v1.6.3a1, then v1.6.3a2). It tags nothing when origin/main carries a tag
already, so a night without merges, or right after a release, builds nothing.
`--github-output` writes the tag pushed, if any, to $GITHUB_OUTPUT as `tag=`.
"""

import argparse
import os
import re
import subprocess
import sys

STABLE = re.compile(r"^v(\d+)\.(\d+)\.(\d+)$")
VERSION = re.compile(r"^(\d+)\.(\d+)\.(\d+)(?:(a|b|rc)(\d+))?$")


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], check=True, capture_output=True, text=True).stdout


def latest_stable(tags: list[str]) -> tuple[int, int, int] | None:
    found = [(int(m[1]), int(m[2]), int(m[3])) for t in tags if (m := STABLE.match(t)) is not None]
    return max(found) if found else None


def next_version(spec: str, latest: tuple[int, int, int] | None) -> str:
    """The version to tag, from a bump or a version given in full."""
    major, minor, patch = latest or (0, 0, 0)
    if spec == "patch":
        return f"{major}.{minor}.{patch + 1}"
    if spec == "minor":
        return f"{major}.{minor + 1}.0"
    if spec == "major":
        return f"{major + 1}.0.0"
    match = VERSION.match(spec)
    if match is None:
        raise ValueError(f"{spec!r} is not patch, minor, major or a version like 1.7.0 or 1.7.0rc1")
    release = (int(match[1]), int(match[2]), int(match[3]))
    if latest and release <= latest:
        raise ValueError(f"{spec} is not after the latest release, v{'.'.join(map(str, latest))}")
    return spec


def next_alpha(tags: list[str], latest: tuple[int, int, int] | None) -> str:
    """The next patch after the latest stable release, as its next alpha."""
    major, minor, patch = latest or (0, 0, 0)
    base = f"{major}.{minor}.{patch + 1}"
    numbers = [int(m[1]) for t in tags if (m := re.fullmatch(rf"v{re.escape(base)}a(\d+)", t))]
    return f"{base}a{max(numbers, default=0) + 1}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("spec", help="patch, minor, major, alpha, or a version such as 1.7.0rc1")
    parser.add_argument("--yes", action="store_true", help="push without asking")
    parser.add_argument(
        "--github-output", action="store_true", help="write tag=<tag pushed> to $GITHUB_OUTPUT"
    )
    args = parser.parse_args(argv)

    _git("fetch", "--quiet", "--tags", "origin", "main")
    tags = _git("tag", "--list", "v*").split()
    latest = latest_stable(tags)
    commit = _git("rev-parse", "origin/main").strip()
    if args.spec == "alpha":
        already = _git("tag", "--list", "v*", "--points-at", commit).split()
        if already:
            print(f"Nothing new on origin/main since {already[0]}: no alpha.")
            return 0
    try:
        version = (
            next_alpha(tags, latest) if args.spec == "alpha" else next_version(args.spec, latest)
        )
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    tag = f"v{version}"
    if tag in tags:
        print(f"error: {tag} exists already", file=sys.stderr)
        return 1

    subject = _git("log", "-1", "--format=%h %s", commit).strip()
    since = f"v{'.'.join(map(str, latest))}" if latest else None
    merged = _git("rev-list", "--count", f"{since}..{commit}" if since else commit).strip()
    print(f"{tag} on origin/main: {subject}")
    print(f"{merged} commits since {since or 'the beginning'}")
    if not args.yes and input(f"Tag and push {tag}? [y/N] ").strip().lower() != "y":
        print("Nothing tagged.")
        return 1

    _git("tag", "--annotate", tag, "--message", f"Release {version}", commit)
    _git("push", "--quiet", "origin", tag)
    print(f"Pushed {tag}: the Release workflow tests it, then publishes it.")
    if args.github_output:
        with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as out:
            out.write(f"tag={tag}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
