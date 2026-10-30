"""Tests for the types in the schema handed to the model.

A tool schema without types tells the model a parameter's name and description
and nothing else. It then guesses, and a model guessing at `timeout` sends
`"30"` — which is how a real session produced
`tool_error [TypeError]: '>' not supported between instances of 'str' and 'int'`
before the command ever ran.

The types were always there, in the signature. Google-style docstrings make the
type optional, and reading the parameters from the docstring alone threw the
annotation away.
"""

# No `from __future__ import annotations`: declarations are built from real
# annotations at construction time.
from domyn.agents.tools import FunctionTool
from domyn.agents.tools.base import FunctionDeclaration


def _declare(func: object) -> dict:
    tool = FunctionTool(func=func)
    return tool.tool_definition["function"]["parameters"]["properties"]


class TestTypesReachTheModel:
    def test_the_signature_fills_a_docstring_that_omits_the_type(self) -> None:
        def search(query: str, limit: int = 10, exact: bool = False) -> str:
            """Search.

            Args:
                query: What to look for.
                limit: How many results.
                exact: Whether to match exactly.
            """
            return ""

        props = _declare(search)
        assert props["query"]["type"] == "string"
        assert props["limit"]["type"] == "integer"
        assert props["exact"]["type"] == "boolean"

    def test_a_docstring_that_states_a_type_still_wins(self) -> None:
        """Nothing that used to work changes: this is only filling a gap."""

        def f(value: str) -> str:
            """Do.

            Args:
                value (int): Deliberately disagreeing with the annotation.
            """
            return ""

        assert _declare(f)["value"]["type"] == "integer"

    def test_an_unannotated_parameter_stays_untyped(self) -> None:
        """There is nothing to fill it from, and inventing one would be worse
        than saying nothing."""

        def f(anything) -> str:  # type: ignore[no-untyped-def]
            """Do.

            Args:
                anything: No annotation anywhere.
            """
            return ""

        assert "type" not in _declare(f)["anything"]

    def test_descriptions_are_untouched(self) -> None:
        def f(path: str) -> str:
            """Do.

            Args:
                path: Where the thing lives.
            """
            return ""

        assert _declare(f)["path"]["description"] == "Where the thing lives."

    def test_required_is_untouched(self) -> None:
        def f(a: str, b: str = "x") -> str:
            """Do.

            Args:
                a: Required.
                b: Optional.
            """
            return ""

        params = FunctionTool(func=f).tool_definition["function"]["parameters"]
        assert params["required"] == ["a"]

    def test_a_tool_without_a_docstring_section_is_unaffected(self) -> None:
        """That path already read the signature; it must keep doing so."""

        def f(n: int) -> str:
            """Just a description, no Args section."""
            return ""

        assert FunctionDeclaration.from_callable(f, []).parameters[0].type == "int"


class TestTheBuiltinToolsAreTyped:
    def test_the_shell_timeout_is_declared_an_integer(self, tmp_path: object) -> None:
        """The parameter that caused the original failure."""
        import tempfile

        from domyn.agents.tools.builtin import bundle

        tools = {t.name: t for t in bundle("shell", root=tempfile.mkdtemp())}
        props = tools["run_command"].tool_definition["function"]["parameters"]["properties"]
        assert props["timeout"]["type"] == "integer"
        assert props["command"]["type"] == "string"

    def test_no_builtin_ships_an_untyped_parameter(self) -> None:
        import tempfile

        from domyn.agents.tools.builtin import bundle

        for tool in bundle("files", "shell", "browser", root=tempfile.mkdtemp()):
            props = tool.tool_definition["function"]["parameters"]["properties"]
            missing = [name for name, spec in props.items() if "type" not in spec]
            assert not missing, f"{tool.name} declares {missing} without a type"
