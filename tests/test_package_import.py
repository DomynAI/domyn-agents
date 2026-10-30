"""What importing the package costs, and that its package-level names still resolve.

Each check runs in a fresh interpreter: this one has imported everything already.
"""

import subprocess
import sys

import pytest


def _run(code: str) -> str:
    return subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    ).stdout.strip()


def test_importing_the_package_imports_none_of_its_subpackages() -> None:
    """A process that needs one light module must not pay for the agent machinery."""
    loaded = _run(
        "import sys, domyn.agents\n"
        "print(sorted(m for m in sys.modules if m.split('.')[0] == 'domyn'))"
    )
    assert loaded == "['domyn', 'domyn.agents']"


@pytest.mark.parametrize(
    "reach",
    [
        "import domyn.agents.decorators as found",
        "import domyn.agents\nfound = domyn.agents.decorators",
    ],
    ids=["as-a-module", "as-an-attribute"],
)
def test_decorators_is_the_core_module_under_either_name(reach: str) -> None:
    out = _run(f"{reach}\nfrom domyn.agents.core import decorators\nprint(found is decorators)")
    assert out == "True"


def test_register_transport_resolves_on_first_use() -> None:
    out = _run(
        "from domyn.agents import register_transport\n"
        "from domyn.agents.runner.transport import register_transport as defined\n"
        "print(register_transport is defined)"
    )
    assert out == "True"
