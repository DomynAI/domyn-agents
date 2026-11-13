# syntax=docker/dockerfile:1
#
# The image `backend: docker` runs a conversation's tools in. Built by CI on every
# release (.github/workflows/docker-backend-image.yml) and pulled at run time, never built
# on a user's machine.
#
# What is in it, and why:
#   * Python, Node, git, curl, procps: what an agent working on a project reaches
#     for. procps because the background-command wrapper needs `ps` and `pkill`.
#   * `domyn`, in its own virtualenv at /opt/domyn: the file and browser tools run
#     in the container as operations of this same release (the agent loop, the
#     approvals and the model stay on the host). A separate environment, so the
#     agent's own `pip install --user` never touches it.
#   * agent-browser and Chrome for Testing, pinned: what the browser tools drive. The
#     agent-browser binary is checked against scripts/agent_browser.lock.
#
# Build from the repository root, with the wheel to install in ./dist:
#   uv build --wheel && docker build -f docker/docker-backend.Dockerfile -t domyn-docker-backend .

ARG PYTHON_IMAGE=python:3.12-slim
ARG NODE_IMAGE=node:22-slim

FROM ${NODE_IMAGE} AS node

# Everything but domyn itself: the part that changes least, so it caches best.
FROM ${PYTHON_IMAGE} AS base
# The libraries are Chrome's own runtime dependencies on Debian.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      procps git curl ca-certificates \
      fonts-liberation fonts-noto-color-emoji \
      libasound2 libatk-bridge2.0-0 libatk1.0-0 libcairo2 libcups2 libdbus-1-3 libdrm2 \
      libgbm1 libglib2.0-0 libnspr4 libnss3 libpango-1.0-0 libx11-6 libxcb1 \
      libxcomposite1 libxdamage1 libxext6 libxfixes3 libxkbcommon0 libxrandr2 \
 && rm -rf /var/lib/apt/lists/*
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
# Chrome for Testing, through the same code the host uses when it downloads one.
RUN mkdir -p /opt/browsers && /opt/domyn/bin/python -c "\
from pathlib import Path; \
from domyn.agents.tools.builtin.browser_install import download_chrome; \
Path('/opt/browsers/chrome').symlink_to(download_chrome(Path('/opt/browsers')))" \
 && chmod -R a+rX /opt/browsers /opt/domyn
# What the file tools run for each call: a script, found by this name because the path
# inside site-packages depends on the Python version (see backends/docker/tool_host.py).
RUN ln -s "$(/opt/domyn/bin/python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')/domyn/agents/tools/builtin/backends/docker/tool_host.py" /opt/domyn/tool_host.py
# The stable name agent-browser is told, whatever Chrome's version.
ENV AGENT_BROWSER_EXECUTABLE_PATH=/opt/browsers/chrome
