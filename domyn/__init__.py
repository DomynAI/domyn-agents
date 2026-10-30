"""Domyn.

The top-level package of the Domyn distribution. Each product is a subpackage:
``domyn.agents`` is the agent framework. Importing ``domyn`` imports none of them.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("domyn")
except PackageNotFoundError:
    __version__ = "unknown"

__all__ = ["__version__"]
