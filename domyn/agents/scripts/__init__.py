"""Programs this package runs in a process of their own.

Each is a file run with ``sys.executable -P``. It imports only the standard
library and what it does its work with, never the rest of this package, so
starting one costs about what starting Python does. A process that cannot
start one, a frozen or compiled build, imports it and calls it instead.
"""
