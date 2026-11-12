"""The Docker sandbox for ``run_command``: ``sandbox: docker`` or ``auto``.

``backend`` holds the backend, which coordinates the rest: ``cli`` runs ``docker``,
``images`` prepares the image, ``network`` is the proxy and the internal network,
``mounts`` and ``state`` decide what the container sees, ``containers`` starts it,
``installs`` keeps what the agent installed, and ``scope`` names and labels it all.
"""

from domyn.agents.tools.builtin.docker_sandbox.backend import DockerBackend
from domyn.agents.tools.builtin.docker_sandbox.images import BASE_IMAGE, DERIVED_IMAGE
from domyn.agents.tools.builtin.docker_sandbox.options import REGISTRY_HOSTS, DockerBackendOptions

__all__ = [
    "BASE_IMAGE",
    "DERIVED_IMAGE",
    "REGISTRY_HOSTS",
    "DockerBackend",
    "DockerBackendOptions",
]
