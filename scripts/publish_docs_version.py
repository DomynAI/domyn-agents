#!/usr/bin/env python3
"""Add one version of the documentation site to the tree served by GitHub Pages.

Usage: scripts/publish_docs_version.py SITE_DIR BUILD_DIR VERSION [--base /repo/]

SITE_DIR is the checkout of the Pages branch, BUILD_DIR the Docusaurus build for
VERSION (built with its baseUrl set to <base><VERSION>/). The build lands in
SITE_DIR/<VERSION>/, replacing any earlier one, and the files that make the
latest release the default are rewritten:

  versions.json   {"latest": ..., "versions": [...]} newest first; read by the
                  version dropdown of every version
  index.html      sends the root to the latest version
  404.html        sends a path that exists in no version (a link from before
                  versions) to the same path in the latest one
  .nojekyll       Pages must not process the files

The latest version is the highest stable release; a pre-release (v1.6.0-rc1)
is published but never becomes the default unless nothing stable exists.
"""

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

VERSION_RE = re.compile(r"^v(\d+)\.(\d+)\.(\d+)([0-9A-Za-z.+-]*)$")

REDIRECT_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Domyn Agents documentation</title>
<meta http-equiv="refresh" content="0; url={target}">
<link rel="canonical" href="{target}">
<script>location.replace({target_js} + location.hash);</script>
</head>
<body><a href="{target}">Domyn Agents documentation, {latest}</a></body>
</html>
"""

NOT_FOUND_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Page not found</title>
<script>
(function () {{
  var base = {base_js};
  var path = location.pathname;
  if (path.indexOf(base) !== 0) return;
  var rest = path.slice(base.length);
  // Inside a version the page really is missing; anything else is an old link.
  if (/^v\\d+\\.\\d+\\.\\d+/.test(rest)) return;
  location.replace(base + {latest_js} + "/" + rest + location.search + location.hash);
}})();
</script>
</head>
<body>
<h1>Page not found</h1>
<p><a href="{base}{latest}/">Domyn Agents documentation, {latest}</a></p>
</body>
</html>
"""


def parse_version(version: str) -> tuple[int, int, int, str]:
    match = VERSION_RE.match(version)
    if match is None:
        raise ValueError(f"not a release tag like v1.2.3: {version!r}")
    major, minor, patch, suffix = match.groups()
    return int(major), int(minor), int(patch), suffix


def sort_key(version: str) -> tuple[int, int, int, int, str]:
    major, minor, patch, suffix = parse_version(version)
    # A release outranks its own pre-releases.
    return (major, minor, patch, 0 if suffix else 1, suffix)


def pick_latest(versions: list[str]) -> str:
    stable = [v for v in versions if not parse_version(v)[3]]
    return max(stable or versions, key=sort_key)


def read_versions(site: Path) -> list[str]:
    path = site / "versions.json"
    if not path.exists():
        return []
    found = json.loads(path.read_text(encoding="utf-8")).get("versions", [])
    return [v for v in found if VERSION_RE.match(v)]


def publish(site: Path, build: Path, version: str, base: str = "/") -> list[str]:
    """Publish `build` as `version` in `site` and return the versions now listed."""
    parse_version(version)
    if not (build / "index.html").is_file():
        raise ValueError(f"{build} does not look like a Docusaurus build")
    if not base.startswith("/") or not base.endswith("/"):
        raise ValueError(f"--base must start and end with '/': {base!r}")

    site.mkdir(parents=True, exist_ok=True)
    target = site / version
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(build, target)

    versions = sorted({*read_versions(site), version}, key=sort_key, reverse=True)
    latest = pick_latest(versions)

    (site / "versions.json").write_text(
        json.dumps({"latest": latest, "versions": versions}, indent=2) + "\n", encoding="utf-8"
    )
    destination = f"{base}{latest}/"
    (site / "index.html").write_text(
        REDIRECT_PAGE.format(target=destination, target_js=json.dumps(destination), latest=latest),
        encoding="utf-8",
    )
    (site / "404.html").write_text(
        NOT_FOUND_PAGE.format(
            base=base, base_js=json.dumps(base), latest=latest, latest_js=json.dumps(latest)
        ),
        encoding="utf-8",
    )
    (site / ".nojekyll").touch()
    return versions


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("site", type=Path)
    parser.add_argument("build", type=Path)
    parser.add_argument("version")
    parser.add_argument("--base", default="/", help="path the site is served under, e.g. /repo/")
    args = parser.parse_args(argv)
    try:
        versions = publish(args.site, args.build, args.version, args.base)
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(f"published {args.version}; latest is {pick_latest(versions)} of {', '.join(versions)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
