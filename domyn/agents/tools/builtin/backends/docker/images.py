"""The image tools run in: published by CI, pulled here, never built on this machine."""

import os
import time
from collections.abc import Callable
from importlib import metadata

from domyn.agents.tools.builtin.backends.base import BackendUnavailableError
from domyn.agents.tools.builtin.backends.docker.cli import (
    INSPECT_SECONDS,
    PULL_SECONDS,
    DockerCli,
    TransientDockerError,
)

IMAGE_REPOSITORY = "ghcr.io/domynai/domyn-docker-backend"
"""Where CI publishes the image (``docker/docker-backend.Dockerfile``). Public, so a pull
needs no login. Defined here only: CI reads it from this file (``scripts/backend_image.py``)."""

LATEST_TAG = "latest"
"""The newest stable release, which CI moves with each one. What a build with no
published image of its own pulls: an editable install or a commit between releases has a
version like ``0.1.1a6.dev12+ged86876``, which no image has, and the release is what pip and
the curl installer give. Defined here only, like ``IMAGE_REPOSITORY``."""

IMAGE_TAG_ENVIRONMENT_VARIABLE = "DOMYN_BACKEND_IMAGE_TAG"
"""Names the tag of ``IMAGE_REPOSITORY`` that ``image: auto`` stands for, in place of the
one of this release: how a maintainer tries an image a manual CI run pushed, without
editing the configuration. Any other image is ``docker.image``."""


def default_image(version: str | None = None) -> str:
    """The image that matches this release of ``domyn``.

    The image carries the same ``domyn`` as the CLI that starts it, because the file
    and browser tools run in it. A release or an alpha (``0.1.1``, ``0.1.2a1``) has an
    image tagged with its version; anything else pulls the rolling tag.
    """
    if version is None:
        try:
            version = metadata.version("domyn")
        except metadata.PackageNotFoundError:
            version = ""
    published = bool(version) and "dev" not in version and "+" not in version
    return f"{IMAGE_REPOSITORY}:{version if published else LATEST_TAG}"


def image_from_environment() -> str | None:
    """The image of the tag ``DOMYN_BACKEND_IMAGE_TAG`` names, when it is set."""
    tag = os.environ.get(IMAGE_TAG_ENVIRONMENT_VARIABLE, "").strip()
    if not tag:
        return None
    if "/" in tag or ":" in tag:
        raise BackendUnavailableError(
            f"{IMAGE_TAG_ENVIRONMENT_VARIABLE} is a tag (`my-test`), not an image: {tag!r}.",
            fix="To use another image, set `docker: image:` in the configuration.",
        )
    return f"{IMAGE_REPOSITORY}:{tag}"


Notify = Callable[[str], None]


async def ensure_image(cli: DockerCli, wanted: str, notify: Notify) -> str:
    """The image to run in, pulled first when this machine lacks it.

    ``wanted`` is ``auto`` for the image of this release (or the one
    ``DOMYN_BACKEND_IMAGE_TAG`` names), or the name of one to use. An image that moves
    (``latest``, or one named by the environment, which is being tried) is pulled every
    time, and a copy already here is used when that fails; any other is pulled only when
    missing. A failed pull raises ``TransientDockerError``: a daemon that has just woken
    answers with errors a moment later it does not.
    """
    named = image_from_environment() if wanted == "auto" else None
    image = named or (default_image() if wanted == "auto" else wanted)
    present = (await cli("image", "inspect", image, timeout=INSPECT_SECONDS)).ok
    moves = named is not None or image.endswith(f":{LATEST_TAG}")
    if present and not moves:
        return image
    notify(f"Pulling image {image}" + ("" if present else " (first run only)") + "…")
    started = time.time()
    pulled = await cli("pull", image, timeout=PULL_SECONDS)
    if pulled.ok:
        notify(f"Pulled {image} in {time.time() - started:.0f}s.")
    elif not present:
        raise TransientDockerError(
            f"could not pull the image {image}: {pulled.error(200).rstrip('.')}.",
            fix="Check the network, or set `docker: image:` (or DOMYN_BACKEND_IMAGE_TAG) to an image "
            "you already have (`make docker-backend-image` builds `domyn-docker-backend:dev` from a checkout).",
        )
    return image
