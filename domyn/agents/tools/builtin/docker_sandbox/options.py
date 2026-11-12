"""The ``docker:`` block of a shell bundle's configuration."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, field_validator

REGISTRY_HOSTS: tuple[str, ...] = ("pypi.org", "files.pythonhosted.org", "registry.npmjs.org")
"""The package registries. Always reachable, also when the network is limited to a
list of hosts: a sandbox that cannot ``pip install`` would be a worse agent, not a
safer one."""

_NOT_A_HOST_CHARACTERS = "/: \t"


class DockerBackendOptions(BaseModel):
    """How the Docker sandbox runs. Read when ``sandbox`` is ``docker`` or ``auto``."""

    model_config = ConfigDict(extra="forbid")

    image: str = "auto"
    """``auto`` builds ``domyn-sandbox`` from ``python:3.12-slim`` on first use. Any
    other value is used as it is and must provide ``bash``, ``timeout`` and ``ps``."""

    allowed_network: Literal["public", "none"] | list[str] = "public"
    """What a command can reach, in one setting:

    * ``public`` (default): any public host over HTTPS (port 443). Private,
      loopback, link-local and metadata addresses are refused, so the container
      cannot reach your network or your machine.
    * a list of hosts (``[github.com, "*.example.com"]``): only those, plus the
      package registries. Setting a list replaces ``public``.
    * ``none``: no network at all.
    """

    memory: str = "2g"
    cpus: float = 2.0
    pids_limit: int = 512

    docker_binary: str = "docker"
    """The CLI to run: ``docker``, or ``podman``, or a path."""

    @field_validator("allowed_network")
    @classmethod
    def _clean_hosts(cls, value: Any) -> Any:
        if not isinstance(value, list):
            return value
        hosts = [_clean_host(entry) for entry in value]
        return list(dict.fromkeys(hosts))

    @property
    def has_network(self) -> bool:
        return self.allowed_network != "none"

    def allowed_hosts(self) -> list[str]:
        """What the proxy is started with: ``*`` for ``public``, else the registries
        and the listed hosts. Empty for ``none``, which has no proxy."""
        if self.allowed_network == "public":
            return ["*"]
        if isinstance(self.allowed_network, list):
            return list(dict.fromkeys([*REGISTRY_HOSTS, *self.allowed_network]))
        return []

    def describe_network(self) -> str:
        """One phrase for the notice that says the sandbox is ready."""
        if self.allowed_network == "public":
            return "network: public HTTPS only"
        if isinstance(self.allowed_network, list):
            return f"network: {len(self.allowed_network)} allowed host(s) + registries"
        return "no network"


def _clean_host(entry: object) -> str:
    host = str(entry).strip().lower()
    if not host or host == "*" or any(c in host for c in _NOT_A_HOST_CHARACTERS):
        raise ValueError(
            f"'{entry}' is not a host name: write github.com or *.example.com "
            "(HTTPS, port 443), or use `allowed_network: public` for any public host"
        )
    return host
