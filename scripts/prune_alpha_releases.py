#!/usr/bin/env python3
"""Delete the GitHub Releases of nightly alphas that are no longer worth a page.

Usage: scripts/prune_alpha_releases.py TAG --repo OWNER/REPO [--keep 5]

Run after TAG is released. After an alpha, only the newest KEEP alpha releases
stay; after a stable release, the alphas before it go (it contains them). Any
other pre-release leaves them alone. Only the Release pages are deleted: the tags
and the published packages stay, so an installed alpha can still be traced back
to its commit.
"""

import argparse
import re
import subprocess

ALPHA = re.compile(r"^v(\d+)\.(\d+)\.(\d+)a(\d+)$")
STABLE = re.compile(r"^v(\d+)\.(\d+)\.(\d+)$")


def _alpha(tag: str) -> tuple[int, int, int, int] | None:
    """`(major, minor, patch, n)` of an alpha tag, None for any other tag."""
    match = ALPHA.match(tag)
    if match is None:
        return None
    return (int(match[1]), int(match[2]), int(match[3]), int(match[4]))


def to_delete(released: list[str], tag: str, keep: int = 5) -> list[str]:
    """The alpha releases to delete once `tag` is out, oldest first."""
    alphas = sorted((key, t) for t in released if (key := _alpha(t)) is not None)
    if _alpha(tag) is not None:
        return [t for _, t in (alphas[:-keep] if keep else alphas)]
    stable = STABLE.match(tag)
    if stable:
        version = (int(stable[1]), int(stable[2]), int(stable[3]))
        return [t for key, t in alphas if key[:3] <= version]
    return []


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("tag")
    parser.add_argument("--repo", required=True, help="OWNER/REPO")
    parser.add_argument("--keep", type=int, default=5)
    args = parser.parse_args(argv)

    released = subprocess.run(
        ["gh", "api", "--paginate", f"repos/{args.repo}/releases", "--jq", ".[].tag_name"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.split()
    for old in to_delete(released, args.tag, args.keep):
        subprocess.run(["gh", "release", "delete", old, "--repo", args.repo, "--yes"], check=True)
        print(f"deleted the release of {old} (its tag stays)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
