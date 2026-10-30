"""``domyn.agents.core.decorators`` under its shorter name, ``domyn.agents.decorators``.

The module replaces itself with that one on import, so the two names are the same module.
"""

import sys

from domyn.agents.core import decorators as _decorators

sys.modules[__name__] = _decorators
