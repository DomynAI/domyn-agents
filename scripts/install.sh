#!/bin/sh
# domyn installer, for macOS (Apple Silicon) and Linux (glibc 2.33 or newer).
#
#   curl -fsSL https://github.com/domynai/domyn-agents/releases/latest/download/install.sh | sh
#   sh install.sh <dir>        install from a local build directory instead
#
# Downloads from the GitHub releases unless a base URL is given.
#
# Env:
#   DOMYN_VERSION       a release to install, e.g. 1.4.26 (default: the latest)
#   DOMYN_BASE_URL      a folder holding the archives (e.g. a self-hosted mirror);
#                       used as is, so it picks the version itself
#   DOMYN_RELEASES_URL  another repository's releases (a fork)
#   DOMYN_HOME          runtime location (default ~/.local/share/domyn)
#   DOMYN_BIN_DIR       where the launcher goes (default ~/.local/bin)
#   DOMYN_LOCAL_DIR     local build directory, same as the first argument

RELEASES_URL="https://github.com/domynai/domyn-agents/releases"
# Set by `INSTALL_BASE_URL=… scripts/build_standalone.sh` in the copy it writes.
BASE_URL=""

set -eu

if [ -t 1 ]; then
  BOLD="\033[1m"; CYAN="\033[36m"; GREEN="\033[32m"; RED="\033[31m"
  YELLOW="\033[33m"; DIM="\033[2m"; RESET="\033[0m"
else
  BOLD=""; CYAN=""; GREEN=""; RED=""; YELLOW=""; DIM=""; RESET=""
fi

info() { printf "${CYAN}==>${RESET} %s\n" "$1"; }
ok()   { printf "${GREEN} ok${RESET} %s\n" "$1"; }
warn() { printf "${YELLOW}  ! ${RESET}%s\n" "$1"; }
fail() { printf "${RED}fail${RESET} %s\n" "$1" >&2; exit 1; }

detect_target() {
  os="$(uname -s)"
  arch="$(uname -m)"
  # Under Rosetta an Apple Silicon Mac reports x86_64.
  if [ "$os-$arch" = "Darwin-x86_64" ] \
    && [ "$(/usr/sbin/sysctl -n sysctl.proc_translated 2>/dev/null || true)" = "1" ]; then
    arch="arm64"
  fi
  case "$os-$arch" in
    Darwin-arm64) TARGET="aarch64-apple-darwin" ;;
    Darwin-x86_64) fail "Intel Macs are not supported: domyn is built for Apple Silicon only." ;;
    Linux-aarch64 | Linux-arm64) TARGET="aarch64-unknown-linux-gnu" ;;
    Linux-x86_64) TARGET="x86_64-unknown-linux-gnu" ;;
    *) fail "unsupported platform: $os $arch" ;;
  esac
  if [ "$os" = "Linux" ] && ls /lib/ld-musl-* >/dev/null 2>&1; then
    fail "this looks like a musl system (Alpine?): the Linux builds need glibc."
  fi
}

sha256_of() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | cut -d' ' -f1
  elif command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$1" | cut -d' ' -f1
  else
    fail "neither sha256sum nor shasum is available to verify the download"
  fi
}

shell_profile() {
  case "${SHELL:-}" in
    */zsh) echo "${ZDOTDIR:-$HOME}/.zshrc" ;;
    */bash)
      if [ "$(uname -s)" = "Darwin" ]; then echo "$HOME/.bash_profile"; else echo "$HOME/.bashrc"; fi
      ;;
    */fish) echo "" ;;
    *) echo "$HOME/.profile" ;;
  esac
}

main() {
  LOCAL_DIR="${1:-${DOMYN_LOCAL_DIR:-}}"
  BASE_URL="${DOMYN_BASE_URL:-$BASE_URL}"
  RELEASES_URL="${DOMYN_RELEASES_URL:-$RELEASES_URL}"
  VERSION="${DOMYN_VERSION:-}"
  VERSION="${VERSION#v}"
  HOME_DIR="${DOMYN_HOME:-$HOME/.local/share/domyn}"
  BIN_DIR="${DOMYN_BIN_DIR:-$HOME/.local/bin}"
  INCOMING="$HOME_DIR.incoming"
  MARKER=".domyn-runtime"

  printf "\n${BOLD}${CYAN}  domyn${RESET}${DIM}  installer${RESET}\n\n"

  if [ -n "$LOCAL_DIR" ]; then
    [ -d "$LOCAL_DIR" ] || fail "no such directory: $LOCAL_DIR"
    info "installing from local build: $LOCAL_DIR"
  else
    if [ -n "$BASE_URL" ] && [ -n "$VERSION" ]; then
      fail "DOMYN_VERSION applies to the GitHub releases only; with a base URL, point it at that version's folder instead"
    fi
    command -v curl >/dev/null 2>&1 || fail "curl is required"
  fi
  command -v tar >/dev/null 2>&1 || fail "tar is required"

  # The runtime is replaced wholesale: never replace a directory we did not make.
  # Ours carries the marker, so a damaged install can still be repaired;
  # python/bin/python3 covers installs made before the marker existed.
  if [ -d "$HOME_DIR" ] && [ -n "$(ls -A "$HOME_DIR" 2>/dev/null)" ] \
    && [ ! -e "$HOME_DIR/$MARKER" ] && [ ! -e "$HOME_DIR/python/bin/python3" ]; then
    fail "$HOME_DIR exists and is not a domyn runtime; refusing to replace it (set DOMYN_HOME elsewhere)"
  fi

  detect_target
  ARCHIVE="domyn-${TARGET}.tar.gz"
  if [ -n "$BASE_URL" ]; then
    FROM="${BASE_URL%/}"
  elif [ -n "$VERSION" ]; then
    FROM="${RELEASES_URL%/}/download/v$VERSION"
  else
    FROM="${RELEASES_URL%/}/latest/download"
  fi
  URL="$FROM/$ARCHIVE"

  WORK="$(mktemp -d)"
  trap 'rm -rf "$WORK" "$INCOMING"' EXIT
  trap 'exit 130' INT TERM

  if [ -n "$LOCAL_DIR" ]; then
    [ -f "$LOCAL_DIR/$ARCHIVE" ] || fail "not found: $LOCAL_DIR/$ARCHIVE (built for a different platform?)"
    info "copying domyn for $TARGET"
    cp "$LOCAL_DIR/$ARCHIVE" "$WORK/$ARCHIVE"
    if [ -f "$LOCAL_DIR/$ARCHIVE.sha256" ]; then
      cp "$LOCAL_DIR/$ARCHIVE.sha256" "$WORK/$ARCHIVE.sha256"
    fi
  else
    info "downloading domyn${VERSION:+ $VERSION} for $TARGET from $FROM"
    curl -fSL --retry 3 --progress-bar -o "$WORK/$ARCHIVE" "$URL" \
      || fail "download failed: $URL${VERSION:+ (is there a release v$VERSION?)}"
    curl -fsSL --retry 3 -o "$WORK/$ARCHIVE.sha256" "$URL.sha256" \
      || fail "could not download the checksum: $URL.sha256"
  fi

  if [ -f "$WORK/$ARCHIVE.sha256" ]; then
    info "verifying checksum"
    EXPECTED="$(cut -d' ' -f1 < "$WORK/$ARCHIVE.sha256")"
    ACTUAL="$(sha256_of "$WORK/$ARCHIVE")"
    [ "$EXPECTED" = "$ACTUAL" ] || fail "checksum mismatch — the download is not what was published"
    ok "checksum matches"
  else
    warn "no checksum next to the local build; installing unverified"
  fi

  # Unpacked next to the destination, not in /tmp: /tmp can be noexec.
  info "unpacking"
  mkdir -p "$(dirname "$HOME_DIR")"
  rm -rf "$INCOMING"
  mkdir -p "$INCOMING"
  tar -xzf "$WORK/$ARCHIVE" -C "$INCOMING"
  [ -x "$INCOMING/python/bin/python3" ] || fail "the archive does not contain a runtime"

  # Checked before the swap, so a build that cannot run here (too old a glibc)
  # leaves the previous install untouched.
  if ! VERSION_OUT="$("$INCOMING/python/bin/python3" -P -c '
import importlib, importlib.util
import domyn.agents.cli.main
from domyn.agents.cli.utils import get_version
for name in ("pydantic_core", "yaml", "cryptography.hazmat.bindings._rust", "rpds", "anydoc", "watchfiles"):
    if importlib.util.find_spec(name.split(".")[0]) is not None:
        importlib.import_module(name)
print(get_version())
' 2>&1)"; then
    printf '%s\n' "$VERSION_OUT" | tail -5 >&2
    fail "the new runtime does not start on this machine; nothing was changed"
  fi
  INSTALLED="$(printf '%s\n' "$VERSION_OUT" | tail -1)"
  echo "domyn $INSTALLED, installed by install.sh" > "$INCOMING/$MARKER"

  rm -rf "$HOME_DIR.previous"
  if [ -d "$HOME_DIR" ]; then mv "$HOME_DIR" "$HOME_DIR.previous"; fi
  mv "$INCOMING" "$HOME_DIR"
  rm -rf "$HOME_DIR.previous"
  # An archive saved from a browser carries macOS's quarantine mark, and Gatekeeper
  # kills a binary that is not notarized (the runtime's, agent-browser's) when it has it.
  if [ "$(uname -s)" = "Darwin" ]; then xattr -dr com.apple.quarantine "$HOME_DIR" 2>/dev/null || true; fi
  ok "runtime installed in $HOME_DIR"

  # A config embedded in the build is copied only where the user has none.
  SEED="$HOME_DIR/seed"
  PLANTED=""
  if [ -d "$SEED" ]; then
    mkdir -p "$HOME/.domyn"
    chmod 700 "$HOME/.domyn" 2>/dev/null || true
    for f in config.yaml credentials; do
      if [ ! -f "$SEED/$f" ]; then
        continue
      elif [ -e "$HOME/.domyn/$f" ]; then
        info "keeping your existing $f"
      else
        cp "$SEED/$f" "$HOME/.domyn/$f"
        chmod 600 "$HOME/.domyn/$f"
        PLANTED="$PLANTED $f"
      fi
    done
    if [ -n "$PLANTED" ]; then
      ok "configuration installed (${PLANTED# })"
    fi
  fi

  # -P keeps the current directory off sys.path, so a local yaml.py or typer.py
  # cannot shadow domyn's imports.
  mkdir -p "$BIN_DIR"
  cat > "$BIN_DIR/domyn" <<SHIM
#!/bin/sh
# Generated by the domyn installer. Runtime: $HOME_DIR
RUNTIME="\${DOMYN_RUNTIME:-$HOME_DIR/python}"
[ -x "\$RUNTIME/bin/python3" ] || {
  echo "domyn: runtime missing at \$RUNTIME — reinstall with the installer." >&2
  exit 1
}
exec "\$RUNTIME/bin/python3" -P -c 'import sys; sys.argv[0] = "domyn"; from domyn.agents.cli.main import cli; cli()' "\$@"
SHIM
  chmod +x "$BIN_DIR/domyn"
  ok "launcher at $BIN_DIR/domyn"

  NEEDS_RELOAD=""
  case ":$PATH:" in
    *":$BIN_DIR:"*) ok "$BIN_DIR is already on your PATH" ;;
    *)
      RC="$(shell_profile)"
      LINE="export PATH=\"$BIN_DIR:\$PATH\""
      if [ -z "$RC" ]; then
        warn "add $BIN_DIR to your PATH (fish: fish_add_path $BIN_DIR)"
      elif grep -qF "$BIN_DIR" "$RC" 2>/dev/null; then
        ok "$BIN_DIR is already added to PATH in $RC"
        NEEDS_RELOAD="$RC"
      else
        printf '\n# domyn\n%s\n' "$LINE" >> "$RC"
        ok "added $BIN_DIR to PATH in $RC"
        NEEDS_RELOAD="$RC"
      fi
      ;;
  esac

  if ! "$BIN_DIR/domyn" --version >/dev/null 2>&1; then
    fail "the installed launcher does not run: try $BIN_DIR/domyn --version"
  fi
  ok "domyn $INSTALLED runs"

  printf "\n${BOLD}Ready.${RESET}\n"
  if [ -n "$NEEDS_RELOAD" ]; then
    printf "  Open a new terminal, or: ${BOLD}. %s${RESET}\n" "$NEEDS_RELOAD"
  fi
  case "$PLANTED" in
    *config.yaml*) ;;
    *) printf "  Then: ${BOLD}domyn setup${RESET}   ${DIM}(set up your agent)${RESET}\n      " ;;
  esac
  printf "  ${BOLD}domyn run${RESET}     ${DIM}(start talking)${RESET}\n\n"
  printf "${DIM}  runtime  %s\n  config   %s\n  remove   rm -rf %s %s/domyn${RESET}\n\n" \
    "$HOME_DIR" "$HOME/.domyn" "$HOME_DIR" "$BIN_DIR"
}

# Called last, so a download cut short by `curl | sh` runs nothing.
main "$@"
