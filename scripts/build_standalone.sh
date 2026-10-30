#!/usr/bin/env bash
# Build a self-contained domyn: a relocatable Python with domyn installed in it.
# Runs on the target platform (compiled wheels rule out cross-building).
#
# Usage: scripts/build_standalone.sh [OUTDIR]
#
# Env:
#   PYTHON_VERSION   CPython series to embed (default 3.14)
#   PBS_RELEASE      python-build-standalone release tag (default: latest)
#   GITHUB_TOKEN     optional, authenticates the release lookup
#   SEED=1           embed ~/.domyn/config.yaml and credentials, for a build
#                    shared privately; off by default, so a build carries no
#                    configuration and no credential
#   DOMYN_SEED_CONFIG, DOMYN_SEED_CREDENTIALS   other files to embed with SEED=1
#   INSTALL_BASE_URL folder the files will be uploaded to (e.g. a self-hosted mirror);
#                    written into the copy of install.sh, which otherwise
#                    downloads from the GitHub releases

set -euo pipefail

OUTDIR="${1:-dist/standalone}"
PYTHON_VERSION="${PYTHON_VERSION:-3.14}"
PBS_REPO="astral-sh/python-build-standalone"

info() { printf '\033[36m[build]\033[0m %s\n' "$1" >&2; }
fail() { printf '\033[31m[fail]\033[0m %s\n' "$1" >&2; exit 1; }
sha256_of() {
  if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | cut -d' ' -f1
  else shasum -a 256 "$1" | cut -d' ' -f1; fi
}

case "${INSTALL_BASE_URL:-}" in *\'*) fail "INSTALL_BASE_URL cannot contain a single quote" ;; esac
SEED_CONFIG="${DOMYN_SEED_CONFIG:-$HOME/.domyn/config.yaml}"
SEED_CREDENTIALS="${DOMYN_SEED_CREDENTIALS:-$HOME/.domyn/credentials}"
if [ -n "${SEED:-}" ] && [ ! -f "$SEED_CONFIG" ]; then
  fail "SEED=1 but there is no $SEED_CONFIG to embed — run 'domyn setup' first"
fi

# No Intel Mac: cryptography has no x86_64 macOS wheel. Under Rosetta an Apple
# Silicon Mac reports x86_64.
ARCH="$(uname -m)"
if [ "$(uname -s)-$ARCH" = "Darwin-x86_64" ] && [ "$(/usr/sbin/sysctl -n sysctl.proc_translated 2>/dev/null)" = "1" ]; then
  ARCH="arm64"
fi
case "$(uname -s)-$ARCH" in
  Darwin-arm64)  TARGET="aarch64-apple-darwin" ;;
  Darwin-x86_64) fail "Intel Macs are not supported: build on Apple Silicon" ;;
  Linux-aarch64) TARGET="aarch64-unknown-linux-gnu" ;;
  Linux-x86_64)  TARGET="x86_64-unknown-linux-gnu" ;;
  *) fail "unsupported platform: $(uname -s)-$(uname -m)" ;;
esac
info "target: $TARGET"

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

info "building the wheel"
uv build --wheel -o "$WORK/wheel" >/dev/null
WHEEL="$(ls "$WORK"/wheel/*.whl)"
# The version is the git tag's, read by hatch-vcs while building: the wheel's
# name is where it can be read back (domyn-<version>-py3-none-any.whl).
VERSION="$(basename "$WHEEL" | cut -d- -f2)"
info "version: $VERSION"

info "resolving python-build-standalone ($PYTHON_VERSION, $TARGET)"
if [ -n "${PBS_RELEASE:-}" ]; then
  API="https://api.github.com/repos/$PBS_REPO/releases/tags/$PBS_RELEASE"
else
  API="https://api.github.com/repos/$PBS_REPO/releases/latest"
fi
AUTH=()
[ -n "${GITHUB_TOKEN:-}" ] && AUTH=(-H "Authorization: Bearer $GITHUB_TOKEN")
# `${AUTH[@]+…}`: macOS bash 3.2 treats an empty array as unbound under -u.
ASSET="$(curl -fsSL ${AUTH[@]+"${AUTH[@]}"} "$API" \
  | grep -oE "https://[^\"]*cpython-${PYTHON_VERSION}\.[0-9]+%2B[0-9]+-${TARGET}-install_only_stripped\.tar\.gz" \
  | head -1)"
[ -n "$ASSET" ] || fail "no python-build-standalone asset for $PYTHON_VERSION / $TARGET"

RUNTIME_FILE="$(basename "$ASSET" | sed 's/%2B/+/g')"
info "downloading $RUNTIME_FILE"
curl -fsSL -o "$WORK/runtime.tar.gz" "$ASSET"
# Verified against the SHA256SUMS published with the same release.
curl -fsSL -o "$WORK/SHA256SUMS" "${ASSET%/*}/SHA256SUMS" || fail "could not download SHA256SUMS for $RUNTIME_FILE"
EXPECTED="$(awk -v f="$RUNTIME_FILE" '$2 == f { print $1 }' "$WORK/SHA256SUMS")"
[ -n "$EXPECTED" ] || fail "$RUNTIME_FILE is not listed in its release's SHA256SUMS"
[ "$(sha256_of "$WORK/runtime.tar.gz")" = "$EXPECTED" ] || fail "checksum mismatch for $RUNTIME_FILE"
info "runtime checksum verified"
mkdir -p "$WORK/stage"
tar -xzf "$WORK/runtime.tar.gz" -C "$WORK/stage"
[ -x "$WORK/stage/python/bin/python3" ] || fail "unexpected archive layout"

# Dependencies pinned and hash-checked from uv.lock, then domyn itself.
# --no-compile: no .pyc in the archive (a third smaller); Python writes them on
# first import, and install.sh's pre-swap check already imports the main ones.
info "installing domyn[cli,audio] into the runtime, pinned to uv.lock"
uv export --frozen --no-dev --extra cli --extra audio --no-emit-project -o "$WORK/requirements.txt" >/dev/null
"$WORK/stage/python/bin/python3" -m pip install --quiet --no-warn-script-location --no-compile \
  --require-hashes -r "$WORK/requirements.txt"
"$WORK/stage/python/bin/python3" -m pip install --quiet --no-warn-script-location --no-compile \
  --no-deps "$WHEEL"

# Its shebang points at this build directory; install.sh writes its own launcher.
rm -f "$WORK/stage/python/bin/domyn"

# The browser tools' program, beside the runtime's own: domyn looks for it in
# <runtime>/bin when it is not on PATH. Chrome is not here; the first browser tool
# call downloads it. Pinned and checked against scripts/agent_browser.lock.
# shellcheck source=scripts/agent_browser.lock
. "$(dirname "$0")/agent_browser.lock"
case "$TARGET" in
  aarch64-apple-darwin)      AB_ASSET="agent-browser-darwin-arm64"; AB_SHA="$AGENT_BROWSER_SHA256_aarch64_apple_darwin" ;;
  x86_64-unknown-linux-gnu)  AB_ASSET="agent-browser-linux-x64";    AB_SHA="$AGENT_BROWSER_SHA256_x86_64_unknown_linux_gnu" ;;
  aarch64-unknown-linux-gnu) AB_ASSET="agent-browser-linux-arm64";  AB_SHA="$AGENT_BROWSER_SHA256_aarch64_unknown_linux_gnu" ;;
  *) fail "no agent-browser binary for $TARGET" ;;
esac
info "downloading agent-browser $AGENT_BROWSER_VERSION ($AB_ASSET)"
curl -fsSL -o "$WORK/stage/python/bin/agent-browser" \
  "https://github.com/vercel-labs/agent-browser/releases/download/v${AGENT_BROWSER_VERSION}/${AB_ASSET}"
[ "$(sha256_of "$WORK/stage/python/bin/agent-browser")" = "$AB_SHA" ] || fail "checksum mismatch for $AB_ASSET"
chmod +x "$WORK/stage/python/bin/agent-browser"
info "agent-browser checksum verified"

if [ -n "${SEED:-}" ]; then
  mkdir -p "$WORK/stage/seed"
  cp "$SEED_CONFIG" "$WORK/stage/seed/config.yaml"
  chmod 600 "$WORK/stage/seed/config.yaml"
  SEEDED="config"
  if [ -f "$SEED_CREDENTIALS" ]; then
    cp "$SEED_CREDENTIALS" "$WORK/stage/seed/credentials"
    chmod 600 "$WORK/stage/seed/credentials"
    # Key names only, never values; read the way read_credentials() does.
    KEYS="$(sed -n 's/^[[:space:]]*\([A-Za-z_][A-Za-z0-9_]*\)[[:space:]]*=.*/\1/p' "$SEED_CREDENTIALS" | tr '\n' ' ')"
    KEYS="${KEYS% }"
    SEEDED="$SEEDED + ${KEYS:-credentials (no keys)}"
  fi
  info "embedding: $SEEDED"
fi

mkdir -p "$OUTDIR"
NAME="domyn-${TARGET}.tar.gz"
info "packing $NAME"
CONTENTS="python"
[ -d "$WORK/stage/seed" ] && CONTENTS="python seed"
# shellcheck disable=SC2086  # deliberate word splitting: one or two members
tar -czf "$OUTDIR/$NAME" -C "$WORK/stage" $CONTENTS

echo "$(sha256_of "$OUTDIR/$NAME")  $NAME" > "$OUTDIR/$NAME.sha256"

if [ -n "${INSTALL_BASE_URL:-}" ]; then
  awk -v url="$INSTALL_BASE_URL" '!done && /^BASE_URL=""$/ { print "BASE_URL='\''" url "'\''"; done = 1; next } { print }' \
    scripts/install.sh > "$OUTDIR/install.sh"
  grep -qF "BASE_URL='$INSTALL_BASE_URL'" "$OUTDIR/install.sh" || fail "could not write INSTALL_BASE_URL into install.sh"
  SHARE_URL="${INSTALL_BASE_URL%/}/install.sh"
else
  cp scripts/install.sh "$OUTDIR/install.sh"
  SHARE_URL="https://github.com/domynai/domyn-agents/releases/latest/download/install.sh"
fi
chmod +x "$OUTDIR/install.sh"

info "built domyn $VERSION — $(du -h "$OUTDIR/$NAME" | cut -f1)"
cat >&2 <<TXT

  $OUTDIR/
    $NAME
    $NAME.sha256
    install.sh

  Share remotely:  curl -fsSL $SHARE_URL | sh
  Install locally: sh $OUTDIR/install.sh $OUTDIR

TXT
if [ -n "${SEEDED:-}" ]; then
  printf '\033[33m  ! this archive contains a working credential (%s).\n    Anyone who can reach the URL can read it — keep the location private.\033[0m\n\n' \
    "$SEEDED" >&2
fi
echo "$OUTDIR/$NAME"
