"""The image commands run in: pulled when named, built on first use otherwise."""

import time
from collections.abc import Callable

from domyn.agents.tools.builtin.docker_sandbox.cli import (
    BUILD_SECONDS,
    INSPECT_SECONDS,
    PULL_SECONDS,
    DockerCli,
    TransientDockerError,
)

BASE_IMAGE = "python:3.12-slim"
"""What the default image is made from, and what the network proxy runs."""

NODE_IMAGE = "node:22-slim"
"""Where the default image takes Node and npm from (the same Debian family as ``BASE_IMAGE``)."""

DERIVED_IMAGE = "domyn-sandbox:py312-2"
"""``BASE_IMAGE`` plus what an agent working on a project reaches for, built once on
this machine: ``procps`` (the background command wrapper needs ``ps`` and ``pkill``,
which the slim image does not have), ``git`` and ``curl``, and Node with npm."""

# A string and not a file next to this module: a frozen build would have to be told
# to ship it, and the image is built from stdin either way.
SANDBOX_DOCKERFILE = f"""\
FROM {NODE_IMAGE} AS node
FROM {BASE_IMAGE}
RUN apt-get update \\
 && apt-get install -y --no-install-recommends procps git curl ca-certificates \\
 && rm -rf /var/lib/apt/lists/*
COPY --from=node /usr/local/bin/node /usr/local/bin/node
COPY --from=node /usr/local/lib/node_modules /usr/local/lib/node_modules
RUN ln -s ../lib/node_modules/npm/bin/npm-cli.js /usr/local/bin/npm \\
 && ln -s ../lib/node_modules/npm/bin/npx-cli.js /usr/local/bin/npx
"""

Notify = Callable[[str], None]


async def ensure_image(cli: DockerCli, wanted: str, notify: Notify) -> str:
    """The image to run commands in, pulled or built first if this machine lacks it.

    ``wanted`` is ``auto`` for the image built here, or the name of one to pull. A
    pull or a build that fails raises ``TransientDockerError``: a daemon that has just
    woken answers with errors a moment later it does not.
    """
    image = DERIVED_IMAGE if wanted == "auto" else wanted
    if (await cli("image", "inspect", image, timeout=INSPECT_SECONDS)).ok:
        return image
    if wanted == "auto":
        await _build_default(cli, notify)
    else:
        await _pull_named(cli, image, notify)
    return image


async def _pull_named(cli: DockerCli, image: str, notify: Notify) -> None:
    notify(f"Pulling sandbox image {image} (first run only)…")
    started = time.time()
    pulled = await cli("pull", image, timeout=PULL_SECONDS)
    if not pulled.ok:
        raise TransientDockerError(f"could not pull the sandbox image {image}: {pulled.error(200)}")
    notify(f"Pulled {image} in {time.time() - started:.0f}s.")


async def _build_default(cli: DockerCli, notify: Notify) -> None:
    notify("Preparing the sandbox image (first run only; Python, Node, git and curl)…")
    started = time.time()
    pulled = await cli("pull", BASE_IMAGE, timeout=PULL_SECONDS)
    if not pulled.ok:
        raise TransientDockerError(
            f"could not pull {BASE_IMAGE}: {pulled.error(200)}. Check the network, or set "
            "`docker: image:` to an image you already have."
        )
    built = await cli(
        "build", "-t", DERIVED_IMAGE, "-", stdin=SANDBOX_DOCKERFILE, timeout=BUILD_SECONDS
    )
    if not built.ok:
        raise TransientDockerError(f"could not build the sandbox image: {built.error()}")
    notify(f"Sandbox image ready ({time.time() - started:.0f}s).")
