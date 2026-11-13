"""The Docker backend for ``run_command``: ``backend: docker`` or ``auto``.

``backend`` holds the backend, which coordinates the rest: ``cli`` runs ``docker``,
``images`` prepares the image, ``network`` is the proxy and the internal network,
``mounts`` and ``state`` decide what the container sees, ``containers`` starts it,
``installs`` keeps what the agent installed, and ``scope`` names and labels it all.
"""

from domyn.agents.tools.builtin.backends.docker.backend import DockerBackend
from domyn.agents.tools.builtin.backends.docker.images import IMAGE_REPOSITORY, default_image
from domyn.agents.tools.builtin.backends.docker.options import (
    REGISTRY_HOSTS,
    DockerBackendOptions,
)

__all__ = [
    "IMAGE_REPOSITORY",
    "REGISTRY_HOSTS",
    "DockerBackend",
    "DockerBackendOptions",
    "default_image",
]
