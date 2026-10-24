"""
Tests for the ``filter_scope`` constructor parameter on ``FunctionManager``.

A ``filter_scope`` is a SQL ``WHERE`` clause that is automatically ANDed
onto every read query (``list_functions``, ``filter_functions``,
``search_functions``, ``get_precondition``).  Write paths are unaffected.
"""

from __future__ import annotations

from tests.helpers import _handle_project
from unify.function_manager.function_manager import FunctionManager


def _FM(**kwargs) -> FunctionManager:
    """Create a FunctionManager with primitives disabled (tests focus on compositional functions)."""
    kwargs.setdefault("include_primitives", False)
    return FunctionManager(**kwargs)


# --------------------------------------------------------------------------- #
#  Shared source snippets                                                      #
# --------------------------------------------------------------------------- #

_PY_ALPHA = 'def alpha(x):\n    """double x"""\n    return x * 2\n'
_PY_BETA = 'def beta(y):\n    """square y"""\n    return y ** 2\n'

_PY_HELLO = (
    'def hello_world():\n    """Prints hello world"""\n    print("Hello, World!")\n'
)


# --------------------------------------------------------------------------- #
#  list_functions                                                              #
# --------------------------------------------------------------------------- #


@_handle_project
def test_filter_scope_filters_list_functions():
    """A scoped instance's list_functions only returns matching rows."""
    fm_all = _FM()
    fm_all.add_functions(implementations=[_PY_ALPHA, _PY_BETA])
    fm_all.add_functions(implementations=_PY_HELLO)

    # Unscoped – should see all three
    assert set(fm_all.list_functions().keys()) == {"alpha", "beta", "hello_world"}

    # Scoped away from hello_world – should exclude it
    fm_py = _FM(filter_scope="name <> 'hello_world'")
    listing = fm_py.list_functions()
    assert "alpha" in listing
    assert "beta" in listing
    assert "hello_world" not in listing


@_handle_project
def test_entrypoint_id_catalogue_ignores_runtime_discovery_scope():
    """Stored references resolve against storage, not actor visibility.

    A runtime can hide functions because an execution environment is
    unavailable. That must affect discovery only: a stored id must still
    resolve for a function that executes in another environment.
    """
    fm_all = _FM()
    fm_all.add_functions(implementations=[_PY_ALPHA, _PY_BETA])
    fm_all.add_functions(implementations=_PY_HELLO)
    all_ids = fm_all.list_function_name_to_ids()

    fm_scoped = _FM(
        filter_scope="name <> 'hello_world'",
        exclude_compositional_ids={all_ids["alpha"], all_ids["hello_world"]},
    )

    assert set(fm_scoped.list_functions()) == {"beta"}
    assert fm_scoped.list_function_name_to_ids() == all_ids


# --------------------------------------------------------------------------- #
#  filter_functions                                                            #
# --------------------------------------------------------------------------- #


@_handle_project
def test_filter_scope_filters_filter_functions_no_caller_filter():
    """filter_functions with no explicit filter still applies the scope."""
    fm = _FM()
    fm.add_functions(implementations=[_PY_ALPHA, _PY_BETA])
    fm.add_functions(implementations=_PY_HELLO)

    fm_py = _FM(filter_scope="name <> 'hello_world'")
    hits = fm_py.filter_functions()
    names = {h["name"] for h in hits}
    assert "alpha" in names
    assert "beta" in names
    assert "hello_world" not in names


@_handle_project
def test_filter_scope_composes_with_caller_filter():
    """When the caller also supplies a filter, both are ANDed together."""
    fm = _FM()
    fm.add_functions(implementations=[_PY_ALPHA, _PY_BETA])
    fm.add_functions(implementations=_PY_HELLO)

    fm_py = _FM(filter_scope="name <> 'hello_world'")
    hits = fm_py.filter_functions(filter="docstring LIKE '%double%'")
    names = {h["name"] for h in hits}
    # Only alpha has 'double' in its docstring AND is in scope
    assert names == {"alpha"}


# --------------------------------------------------------------------------- #
#  search_functions                                                            #
# --------------------------------------------------------------------------- #


@_handle_project
def test_filter_scope_filters_search_functions():
    """Search on a scoped instance never returns out-of-scope rows."""
    fm = _FM()
    fm.add_functions(implementations=[_PY_ALPHA, _PY_BETA])
    fm.add_functions(implementations=_PY_HELLO)

    fm_py = _FM(filter_scope="name <> 'hello_world'")
    hits = fm_py.search_functions(query="hello world", n=10)
    for h in hits:
        assert (
            h["name"] != "hello_world"
        ), f"search_functions returned out-of-scope row: {h['name']}"


# --------------------------------------------------------------------------- #
#  get_precondition                                                            #
# --------------------------------------------------------------------------- #


@_handle_project
def test_filter_scope_filters_get_precondition():
    """A scoped instance can't see a function outside its scope via get_precondition."""
    fm = _FM()
    fm.add_functions(
        implementations=_PY_HELLO,
        preconditions={"hello_world": {"needs_auth": True}},
    )

    # Unscoped can retrieve it
    assert fm.get_precondition(function_name="hello_world") is not None

    # Scoped away from hello_world – it is invisible
    fm_py = _FM(filter_scope="name <> 'hello_world'")
    assert fm_py.get_precondition(function_name="hello_world") is None


# --------------------------------------------------------------------------- #
#  Default (None) is a no-op                                                   #
# --------------------------------------------------------------------------- #


@_handle_project
def test_filter_scope_none_is_unscoped():
    """filter_scope=None (default) behaves identically to no scope."""
    fm = _FM()
    fm.add_functions(implementations=[_PY_ALPHA])
    fm.add_functions(implementations=_PY_HELLO)

    fm_none = _FM(filter_scope=None)
    assert set(fm_none.list_functions().keys()) == {"alpha", "hello_world"}


# --------------------------------------------------------------------------- #
#  Writes are unaffected                                                       #
# --------------------------------------------------------------------------- #


@_handle_project
def test_filter_scope_does_not_affect_writes():
    """A scoped instance can still add functions outside its own scope."""
    fm_py = _FM(filter_scope="name <> 'hello_world'")
    # Add an out-of-scope function through the scoped instance
    result = fm_py.add_functions(implementations=_PY_HELLO)
    assert result == {"hello_world": "added"}

    # The scoped instance can't see it (correct – out of scope)
    assert "hello_world" not in fm_py.list_functions()

    # An unscoped instance can see it (proves the write succeeded)
    fm_all = _FM()
    assert "hello_world" in fm_all.list_functions()


@_handle_project
def test_scoped_add_records_dependency_on_hidden_function():
    """A call to a stored function outside the scope is still a dependency."""
    _FM().add_functions(implementations=_PY_HELLO)
    fm_scoped = _FM(filter_scope="name <> 'hello_world'")

    fm_scoped.add_functions(
        implementations="def greet():\n    return hello_world()\n",
    )

    greet = _FM()._get_function_data_by_name(name="greet")
    assert greet is not None
    assert greet["depends_on"] == ["hello_world"]


@_handle_project
def test_scoped_overwrite_updates_hidden_function():
    """overwrite=True updates a same-named function the scope hides."""
    fm_all = _FM()
    fm_all.add_functions(implementations=_PY_HELLO)
    hello_id = fm_all.list_function_name_to_ids()["hello_world"]
    fm_scoped = _FM(exclude_compositional_ids={hello_id})

    result = fm_scoped.add_functions(
        implementations='def hello_world():\n    """Greets"""\n    return "hi"\n',
        overwrite=True,
    )

    assert result == {"hello_world": "updated"}
    assert fm_all.list_functions()["hello_world"]["docstring"] == "Greets"


# --------------------------------------------------------------------------- #
#  Two differently-scoped instances see disjoint subsets                        #
# --------------------------------------------------------------------------- #


@_handle_project
def test_two_scoped_instances_see_different_subsets():
    """Non-overlapping scopes produce disjoint views of the same data."""
    fm = _FM()
    fm.add_functions(implementations=[_PY_ALPHA, _PY_BETA])
    fm.add_functions(implementations=_PY_HELLO)

    fm_math = _FM(filter_scope="name <> 'hello_world'")
    fm_hello = _FM(filter_scope="name = 'hello_world'")

    math_names = set(fm_math.list_functions().keys())
    hello_names = set(fm_hello.list_functions().keys())

    assert math_names == {"alpha", "beta"}
    assert hello_names == {"hello_world"}
    assert math_names & hello_names == set()
