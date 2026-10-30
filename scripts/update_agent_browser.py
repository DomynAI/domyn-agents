#!/usr/bin/env python3
"""Pin another version of agent-browser in scripts/agent_browser.lock.

Usage: scripts/update_agent_browser.py VERSION

Reads the sha256 GitHub publishes for each binary of the release v<VERSION> of
vercel-labs/agent-browser. Commit the lock like any change, after running the
browser tests against the new binary.
"""

import json
import subprocess
import sys
from pathlib import Path

LOCK = Path(__file__).with_name("agent_browser.lock")
ASSETS = {
    "aarch64_apple_darwin": "agent-browser-darwin-arm64",
    "x86_64_unknown_linux_gnu": "agent-browser-linux-x64",
    "aarch64_unknown_linux_gnu": "agent-browser-linux-arm64",
}


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    version = argv[1].removeprefix("v")
    out = subprocess.run(
        ["gh", "release", "view", f"v{version}", "-R", "vercel-labs/agent-browser", "--json", "assets"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    digests = {a["name"]: a.get("digest", "") for a in json.loads(out)["assets"]}
    lines = [
        "# The agent-browser binary that goes into the standalone build, and its checksums.",
        "# Read by scripts/build_standalone.sh; refresh with scripts/update_agent_browser.py.",
        "# The binary is downloaded from the release of the same version on GitHub",
        "# (vercel-labs/agent-browser) and refused when its sha256 differs.",
        f"AGENT_BROWSER_VERSION={version}",
    ]
    for target, asset in ASSETS.items():
        digest = digests.get(asset, "")
        if not digest.startswith("sha256:"):
            print(f"error: no sha256 published for {asset} in v{version}", file=sys.stderr)
            return 1
        lines.append(f"AGENT_BROWSER_SHA256_{target}={digest.removeprefix('sha256:')}")
    LOCK.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"pinned agent-browser {version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
