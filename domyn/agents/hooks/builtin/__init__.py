"""Hooks the framework ships and attaches for a feature an agent declares.

Like :mod:`domyn.agents.tools.builtin`, nothing here is attached by itself: each
module builds the hooks of one feature, and the agent mounts them when its
configuration asks for that feature (an ``AUTOMATIC`` memory manager, say).
"""
