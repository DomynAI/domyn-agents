"""Resources: places files live, declared once in a configuration and named by id.

See :mod:`domyn.agents.resources.base` for the interface and
:mod:`domyn.agents.resources.config` for what a configuration's ``resources:``
must satisfy when it loads. One type ships here, ``local_dir``; an application
registers a type of its own by subclassing :class:`Resource`.
"""

from domyn.agents.resources.base import (
    MountSpec,
    Resource,
    ResourceEntry,
    ResourceError,
    relative_path,
    resolve_resource,
)
from domyn.agents.resources.local import LocalDirResource

__all__ = [
    "LocalDirResource",
    "MountSpec",
    "Resource",
    "ResourceEntry",
    "ResourceError",
    "relative_path",
    "resolve_resource",
]
