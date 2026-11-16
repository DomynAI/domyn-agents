# syntax=docker/dockerfile:1
#
# The image `backend: docker` runs a conversation's tools in. Built by CI on every
# release (.github/workflows/docker-backend-image.yml) and pulled at run time, never built
# on a user's machine.
#
# What is in it, and why:
#   * Python, Node, git, curl, procps: what an agent working on a project reaches
#     for. procps because the background-command wrapper needs `ps` and `pkill`.
#   * `domyn`, in its own virtualenv at /opt/domyn (the agent loop, the approvals and the
#     model stay on the host): the file and browser tools run in the container as
#     operations of this same release, and a command the agent starts can run `domyn`
#     itself, which is on the PATH. A separate environment, so the agent's own
#     `pip install --user` never touches it.
#   * agent-browser (pinned, checked against scripts/agent_browser.lock) and Debian's
#     chromium-headless-shell: what the browser tools drive. It is the distro's
#     BSD-3-Clause Chromium built from source, not Google's Chrome for Testing, which is
#     branded Chrome and bundles the non-redistributable Widevine CDM. The headless
#     shell is the headless-only build, so there is no `chromium` command either.
#
# Build from the repository root, with the wheel to install in ./dist:
#   uv build --wheel && docker build -f docker/docker-backend.Dockerfile -t domyn-docker-backend .

ARG PYTHON_IMAGE=python:3.12-slim
ARG NODE_IMAGE=node:22-slim

FROM ${NODE_IMAGE} AS node

# Everything but domyn itself: the part that changes least, so it caches best.
FROM ${PYTHON_IMAGE} AS base
# The fonts are spelled out: they are only Recommends of the browser package, which
# --no-install-recommends drops, and every screenshot would render tofu boxes.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      procps git curl ca-certificates \
      chromium-headless-shell fonts-liberation fonts-noto-color-emoji \
 && rm -rf /var/lib/apt/lists/* \
 && test -x /usr/lib/chromium/chromium-headless-shell
COPY --from=node /usr/local/bin/node /usr/local/bin/node
COPY --from=node /usr/local/lib/node_modules /usr/local/lib/node_modules
RUN ln -s ../lib/node_modules/npm/bin/npm-cli.js /usr/local/bin/npm \
 && ln -s ../lib/node_modules/npm/bin/npx-cli.js /usr/local/bin/npx

FROM base AS agent-browser
ARG TARGETARCH
COPY scripts/agent_browser.lock /tmp/agent_browser.lock
RUN set -eu; . /tmp/agent_browser.lock; \
    case "$TARGETARCH" in \
      amd64) asset=agent-browser-linux-x64;   sha="$AGENT_BROWSER_SHA256_x86_64_unknown_linux_gnu" ;; \
      arm64) asset=agent-browser-linux-arm64; sha="$AGENT_BROWSER_SHA256_aarch64_unknown_linux_gnu" ;; \
      *) echo "unsupported architecture: $TARGETARCH" >&2; exit 1 ;; \
    esac; \
    curl -fsSL -o /usr/local/bin/agent-browser \
      "https://github.com/vercel-labs/agent-browser/releases/download/v${AGENT_BROWSER_VERSION}/${asset}"; \
    echo "${sha}  /usr/local/bin/agent-browser" | sha256sum -c -; \
    chmod 0755 /usr/local/bin/agent-browser

FROM base AS final
COPY --from=agent-browser /usr/local/bin/agent-browser /usr/local/bin/agent-browser
COPY dist/*.whl /tmp/dist/
RUN python -m venv /opt/domyn \
 && /opt/domyn/bin/pip install --no-cache-dir /tmp/dist/*.whl \
 && rm -rf /tmp/dist
RUN chmod -R a+rX /opt/domyn \
 && ln -s /opt/domyn/bin/domyn /usr/local/bin/domyn
# What the file tools run for each call: a script, found by this name because the path
# inside site-packages depends on the Python version (see backends/docker/tool_host.py).
RUN ln -s "$(/opt/domyn/bin/python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')/domyn/agents/tools/builtin/backends/docker/tool_host.py" /opt/domyn/tool_host.py
# The real binary, not the /usr/bin wrapper (a shell script that runs it as a child, so the
# PID a launcher holds would be the shell's). agent-browser adds --no-sandbox itself; Docker's
# /dev/shm is 64 MB, which renderers outgrow on anything non-trivial.
ENV AGENT_BROWSER_EXECUTABLE_PATH=/usr/lib/chromium/chromium-headless-shell \
    AGENT_BROWSER_ARGS=--no-sandbox,--disable-dev-shm-usage
