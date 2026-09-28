"""
Comprehensive unit-tests for `FunctionManager`

Coverage
========
✓ add_functions                           (existing happy-path + validation)
✓ list_functions                          (with / without implementations)
✓ delete_function                         (single, cascading, non-cascading)
✓ filter_functions                        (flexible Python-expr filtering)

The tests introduce a *minimal* stub of the `unify` API so that they remain
fully hermetic.  Nothing outside this file is required.
"""

from __future__ import annotations

import pytest
from tests.helpers import _handle_project
from unify.function_manager.function_manager import FunctionManager


def _FM(**kwargs) -> FunctionManager:
    """Create a FunctionManager with primitives disabled (tests focus on compositional functions)."""
    kwargs.setdefault("include_primitives", False)
    return FunctionManager(**kwargs)


# --------------------------------------------------------------------------- #
#  4.  Existing add_functions tests                                           #
# --------------------------------------------------------------------------- #


@_handle_project
def test_add_single_success():
    src = (
        "def double(x):\n"
        "    y = 0\n"
        "    for _ in range(2):\n"
        "        y = y + x\n"
        "    return y\n"
    )
    fm = _FM()
    result = fm.add_functions(implementations=src)
    assert result == {"double": "added"}


@_handle_project
def test_add_multiple_with_dependency():
    add_src = "def add(a, b):\n    return a + b\n"
    # Update to avoid user-defined function calls (now disallowed)
    twice_src = "def twice(x):\n    return x + x\n"
    fm = _FM()
    result = fm.add_functions(implementations=[add_src, twice_src])
    assert result == {"add": "added", "twice": "added"}


@_handle_project
def test_batch_add_multiple():
    """Test that adding multiple functions works efficiently (uses batch creation)."""
    # Create several independent functions
    sources = [
        "def func_a(x):\n    return x * 2\n",
        "def func_b(y):\n    return y + 10\n",
        "def func_c(z):\n    return z - 5\n",
        "def func_d(w):\n    return w / 2\n",
        "def func_e(v):\n    return v ** 2\n",
    ]

    fm = _FM()
    result = fm.add_functions(implementations=sources)

    # All functions should be added successfully
    assert result == {
        "func_a": "added",
        "func_b": "added",
        "func_c": "added",
        "func_d": "added",
        "func_e": "added",
    }

    # Verify all functions are listed
    listing = fm.list_functions()
    assert set(listing.keys()) == {"func_a", "func_b", "func_c", "func_d", "func_e"}

    # Verify each function has correct metadata
    for func_name in ["func_a", "func_b", "func_c", "func_d", "func_e"]:
        assert "argspec" in listing[func_name]
        assert "function_id" in listing[func_name]


@_handle_project
def test_add_duplicate_skips_by_default():
    """Test that adding duplicate functions skips them by default."""
    fm = _FM()

    # Add initial function
    result1 = fm.add_functions(implementations="def alpha(x):\n    return x * 2\n")
    assert result1 == {"alpha": "added"}

    # Try to add the same function again
    result2 = fm.add_functions(implementations="def alpha(x):\n    return x * 3\n")
    assert result2 == {"alpha": "skipped: already exists"}

    # Verify only one function exists with original implementation
    listing = fm.list_functions()
    assert len(listing) == 1
    assert "alpha" in listing

    # Verify the original implementation is still there
    full = fm.list_functions(include_implementations=True)
    assert "x * 2" in full["alpha"]["implementation"]
    assert "x * 3" not in full["alpha"]["implementation"]


@_handle_project
def test_add_duplicate_with_overwrite():
    """Test that adding duplicate functions with overwrite=True updates them in-place."""
    fm = _FM()

    # Add initial function
    result1 = fm.add_functions(implementations="def alpha(x):\n    return x * 2\n")
    assert result1 == {"alpha": "added"}

    original_id = fm.list_functions()["alpha"]["function_id"]

    # Add the same function with overwrite=True
    result2 = fm.add_functions(
        implementations="def alpha(x):\n    return x * 3\n",
        overwrite=True,
    )
    assert result2 == {"alpha": "updated"}

    # Verify still only one function exists
    listing = fm.list_functions()
    assert len(listing) == 1
    assert "alpha" in listing

    # Function ID should remain the same (in-place update, not delete+create)
    new_id = listing["alpha"]["function_id"]
    assert new_id == original_id, "Function ID should be preserved with in-place update"

    # Verify the new implementation replaced the old one
    full = fm.list_functions(include_implementations=True)
    assert "x * 3" in full["alpha"]["implementation"]
    assert "x * 2" not in full["alpha"]["implementation"]


@_handle_project
def test_add_mixed_new_and_duplicate():
    """Test adding a batch with both new and duplicate functions."""
    fm = _FM()

    # Add initial functions
    result1 = fm.add_functions(
        implementations=[
            "def alpha():\n    return 1\n",
            "def beta():\n    return 2\n",
        ],
    )
    assert result1 == {"alpha": "added", "beta": "added"}

    # Add a mix of new and duplicate functions
    result2 = fm.add_functions(
        implementations=[
            "def alpha():\n    return 10\n",  # Duplicate
            "def gamma():\n    return 3\n",  # New
            "def beta():\n    return 20\n",  # Duplicate
        ],
    )

    assert result2 == {
        "alpha": "skipped: already exists",
        "gamma": "added",
        "beta": "skipped: already exists",
    }

    # Verify correct functions exist
    listing = fm.list_functions()
    assert set(listing.keys()) == {"alpha", "beta", "gamma"}

    # Verify original implementations are preserved
    full = fm.list_functions(include_implementations=True)
    assert "return 1" in full["alpha"]["implementation"]
    assert "return 2" in full["beta"]["implementation"]
    assert "return 3" in full["gamma"]["implementation"]


@_handle_project
@pytest.mark.parametrize(
    "source,exp_msg",
    [
        ("def bad(x)\n    return x", "Syntax error"),  # syntax error
        ("    def indented(x):\n        return x", "must start at column 0"),  # indent
        # Dangerous built-ins are disallowed
        ("def uses_eval(x):\n    return eval(str(x))", "Dangerous built-in 'eval'"),
        ("def uses_open():\n    return open('f.txt','w')", "Dangerous built-in 'open'"),
        # Self-recursive calls (user-defined) are disallowed under current policy
        # Note: User-defined function calls are now allowed (tracked as dependencies)
        # (
        #     "def recurse(x):\n    return recurse(x-1)",
        #     "cannot call user-defined function 'recurse'",
        # ),
    ],
)
def test_validation_errors(source: str, exp_msg: str):
    fm = _FM()
    results = fm.add_functions(implementations=source, raise_on_error=False)
    assert any(
        "error" in str(v) and exp_msg in str(v) for v in results.values()
    ), f"Expected error containing '{exp_msg}' in results: {results}"


# --------------------------------------------------------------------------- #
#  5.  list_functions                                                         #
# --------------------------------------------------------------------------- #


@_handle_project
def test_list_with_and_without_implementations():
    add_src = (
        "def add(a: int, b: int) -> int:\n"
        '    """Add two numbers"""\n'
        "    return a + b\n"
    )
    fm = _FM()
    fm.add_functions(implementations=add_src)

    # (a) default summary
    listing = fm.list_functions()
    assert listing.keys() == {"add"}
    assert "implementation" not in listing["add"]
    # The argspec includes type hints and return annotation
    assert "(a: int, b: int) -> int" in listing["add"]["argspec"]
    assert listing["add"]["docstring"] == "Add two numbers"

    # (b) include full source
    full = fm.list_functions(include_implementations=True)
    assert add_src.strip() == full["add"]["implementation"].strip()


# --------------------------------------------------------------------------- #
#  6.  delete_function                                                        #
# --------------------------------------------------------------------------- #


@_handle_project
def test_delete_single():
    fm = _FM()
    fm.add_functions(implementations="def alpha():\n    return 1\n")
    listing = fm.list_functions()
    assert len(listing) == 1

    fm.delete_function(function_id=listing["alpha"]["function_id"])
    assert fm.list_functions() == {}


@_handle_project
def test_delete_with_dependants_cascades():
    add_src = "def add(a, b):\n    return a + b\n"
    # No user-defined calls allowed; keep independent
    twin_src = "def twin(x):\n    return x + x\n"
    fm = _FM()
    fm.add_functions(implementations=[add_src, twin_src])

    # delete `add`; since no dependants are allowed, only `add` is removed
    add_id = fm.list_functions()["add"]["function_id"]
    fm.delete_function(function_id=add_id, delete_dependents=True)
    remaining = fm.list_functions()
    assert remaining.keys() == {"twin"}


@_handle_project
def test_delete_without_cascading_leaves_dependants():
    add_src = "def add(a, b):\n    return a + b\n"
    twin_src = "def twin(x):\n    return x + x\n"
    fm = _FM()
    fm.add_functions(implementations=[add_src, twin_src])

    add_id = fm.list_functions()["add"]["function_id"]
    fm.delete_function(function_id=add_id, delete_dependents=False)
    remaining = fm.list_functions()
    assert remaining.keys() == {"twin"}


@_handle_project
def test_delete_already_deleted():
    """Test that deleting an already-deleted function doesn't raise an error."""
    fm = _FM()
    fm.add_functions(implementations="def alpha():\n    return 1\n")
    listing = fm.list_functions()
    function_id = listing["alpha"]["function_id"]

    # Delete the function
    result = fm.delete_function(function_id=function_id)
    assert result == {"alpha": "deleted"}
    assert fm.list_functions() == {}

    # Try to delete it again - should return success, not raise error
    result = fm.delete_function(function_id=function_id)
    assert "already_deleted" in result[f"function_{function_id}"]


@_handle_project
def test_delete_multiple_handles_duplicates():
    """Test that deleting multiple functions handles cases where some are already deleted."""
    fm = _FM()
    sources = [
        "def func_a():\n    return 1\n",
        "def func_b():\n    return 2\n",
        "def func_c():\n    return 3\n",
    ]
    fm.add_functions(implementations=sources)

    listing = fm.list_functions()
    ids = [listing[name]["function_id"] for name in ["func_a", "func_b", "func_c"]]

    # Delete all functions in a loop (simulates user script pattern)
    for function_id in ids:
        result = fm.delete_function(function_id=function_id)
        # Should either delete or report already deleted
        assert (
            "deleted" in list(result.values())[0]
            or "already_deleted" in list(result.values())[0]
        )

    # All functions should be gone
    assert fm.list_functions() == {}


@_handle_project
def test_batch_delete():
    """Test batch deletion of multiple functions at once."""
    fm = _FM()
    sources = [
        "def func_a():\n    return 1\n",
        "def func_b():\n    return 2\n",
        "def func_c():\n    return 3\n",
        "def func_d():\n    return 4\n",
        "def func_e():\n    return 5\n",
    ]
    fm.add_functions(implementations=sources)

    listing = fm.list_functions()
    assert len(listing) == 5

    # Get IDs for batch deletion
    ids = [listing[name]["function_id"] for name in ["func_a", "func_b", "func_c"]]

    # Batch delete 3 functions at once using delete_function with list
    result = fm.delete_function(function_id=ids)
    assert set(result.keys()) == {"func_a", "func_b", "func_c"}
    assert all(v == "deleted" for v in result.values())

    # Check remaining functions
    remaining = fm.list_functions()
    assert set(remaining.keys()) == {"func_d", "func_e"}


@_handle_project
def test_batch_delete_all():
    """Test deleting all functions at once by passing all IDs."""
    fm = _FM()
    sources = [
        "def func_a():\n    return 1\n",
        "def func_b():\n    return 2\n",
        "def func_c():\n    return 3\n",
    ]
    fm.add_functions(implementations=sources)

    listing = fm.list_functions()
    assert len(listing) == 3

    # Get all IDs and delete all functions at once
    all_ids = [data["function_id"] for data in listing.values()]
    result = fm.delete_function(function_id=all_ids, delete_dependents=False)
    assert set(result.keys()) == {"func_a", "func_b", "func_c"}
    assert all(v == "deleted" for v in result.values())

    # No functions should remain
    assert fm.list_functions() == {}


@_handle_project
def test_batch_delete_with_cascade():
    """Test batch deletion with cascading deletes of dependents."""
    fm = _FM()
    sources = [
        "def base_a():\n    return 1\n",
        "def base_b():\n    return 2\n",
        "def caller_ab():\n    return base_a\n",  # Depends on base_a
        "def caller_b():\n    return base_b\n",  # Depends on base_b
        "def independent():\n    return 99\n",
    ]
    fm.add_functions(implementations=sources)

    listing = fm.list_functions()
    assert len(listing) == 5

    # Get IDs for base functions
    base_ids = [listing["base_a"]["function_id"], listing["base_b"]["function_id"]]

    # Batch delete with cascade should also delete caller_ab and caller_b
    result = fm.delete_function(function_id=base_ids, delete_dependents=True)

    # Should delete base_a, base_b, caller_ab, caller_b (4 functions)
    assert len(result) == 4
    assert set(result.keys()) == {"base_a", "base_b", "caller_ab", "caller_b"}
    assert all(v == "deleted" for v in result.values())

    # Only independent should remain
    remaining = fm.list_functions()
    assert set(remaining.keys()) == {"independent"}


@_handle_project
def test_batch_delete_empty_list():
    """Test that batch deleting an empty list is a no-op."""
    fm = _FM()
    fm.add_functions(implementations="def alpha():\n    return 1\n")

    result = fm.delete_function(function_id=[])
    assert result == {}

    # Function should still exist
    assert "alpha" in fm.list_functions()


@_handle_project
def test_batch_delete_nonexistent():
    """Test that batch deleting non-existent functions doesn't error."""
    fm = _FM()
    fm.add_functions(implementations="def alpha():\n    return 1\n")

    # Try to delete functions that don't exist
    result = fm.delete_function(function_id=[9999, 8888])

    # Each unknown id is reported as already gone; nothing is raised
    assert result == {
        "function_9999": "already_deleted",
        "function_8888": "already_deleted",
    }

    # Original function should still exist
    assert "alpha" in fm.list_functions()


# --------------------------------------------------------------------------- #
#  7.  filter_functions                                                       #
# --------------------------------------------------------------------------- #


@_handle_project
def test_search_filtering_across_columns():
    price_src = (
        "def price_total(p, tax):\n"
        '    """Return total price including tax"""\n'
        "    return p + tax\n"
    )
    square_src = "def square(x):\n    return x * x\n"
    use_src = "def apply_price(x):\n    return x\n"
    fm = _FM()
    fm.add_functions(implementations=[price_src, square_src, use_src])

    # filter on docstring contents
    hits = fm.filter_functions(filter="docstring LIKE '%price%'")
    names = {h["name"] for h in hits}
    assert names == {"price_total"}

    # filter by a SQL expression over the `name` column
    hits = fm.filter_functions(filter="substr(name, 1, 2) = 'sq'")
    assert {h["name"] for h in hits} == {"square"}


@_handle_project
def test_filter_functions_include_implementations():
    """filter_functions respects include_implementations parameter."""
    fm = _FM()
    fm.add_functions(implementations="def foo(x):\n    return x * 2\n")

    # Default (True): includes implementation
    hits = fm.filter_functions(filter="name = 'foo'")
    assert len(hits) == 1
    assert "implementation" in hits[0]

    # Explicit False: excludes implementation
    hits = fm.filter_functions(filter="name = 'foo'", include_implementations=False)
    assert len(hits) == 1
    assert "implementation" not in hits[0]
    assert "name" in hits[0]  # Other fields still present


@_handle_project
def test_search_functions_include_implementations():
    """search_functions respects include_implementations parameter."""
    fm = _FM()
    fm.add_functions(implementations="def double_value(x):\n    return x * 2\n")

    # Default (True): includes implementation
    hits = fm.search_functions(
        query="double a number",
        n=5,
    )
    user_funcs = [h for h in hits if h.get("name") == "double_value"]
    assert len(user_funcs) == 1
    assert "implementation" in user_funcs[0]

    # Explicit False: excludes implementation
    hits = fm.search_functions(
        query="double a number",
        n=5,
        include_implementations=False,
    )
    user_funcs = [h for h in hits if h.get("name") == "double_value"]
    assert len(user_funcs) == 1
    assert "implementation" not in user_funcs[0]
    assert "name" in user_funcs[0]  # Other fields still present


@_handle_project
def test_search_functions_matches_meaning_not_words():
    """A query that shares no words with a function's name or docstring finds it."""
    fm = _FM()
    fm.add_functions(
        implementations=[
            "def celsius_to_fahrenheit(celsius: float) -> float:\n"
            '    """Convert a temperature from Celsius to Fahrenheit."""\n'
            "    return celsius * 9 / 5 + 32\n",
            "def send_invoice_email(customer_email: str, invoice_path: str) -> None:\n"
            '    """Email an invoice PDF to a customer."""\n'
            "    return None\n",
            "def resize_image(path: str, width: int, height: int) -> str:\n"
            '    """Scale an image file to the given dimensions and save it."""\n'
            "    return path\n",
        ],
    )

    hits = fm.search_functions(query="mail a bill to a client", n=3)

    assert [h["name"] for h in hits][0] == "send_invoice_email"
    assert hits[0]["_similarity"] > hits[1]["_similarity"] >= hits[2]["_similarity"]


# --------------------------------------------------------------------------- #
#  8.  clear                                                                  #
# --------------------------------------------------------------------------- #


@_handle_project
def test_clear():
    fm = _FM()

    # Seed a couple of functions
    fm.add_functions(implementations="def alpha():\n    return 1\n")
    fm.add_functions(implementations="def beta():\n    return 2\n")

    listing = fm.list_functions()
    assert set(listing.keys()) == {"alpha", "beta"}
    first_id = listing["alpha"]["function_id"]
    assert isinstance(first_id, int)
    assert isinstance(listing["beta"]["function_id"], int)

    # Execute clear
    fm.clear()

    # After clear: no functions should remain
    assert fm.list_functions() == {}

    # New additions work against a clean slate: the manager is in the state
    # of a brand-new instance, so ids restart where the first one started.
    fm.add_functions(implementations="def gamma():\n    return 3\n")
    post = fm.list_functions()
    assert set(post.keys()) == {"gamma"}
    assert post["gamma"]["function_id"] == first_id

    fm.add_functions(implementations="def square(x):\n    return x * x\n")
    hits = fm.filter_functions(filter="implementation LIKE '%return x * x%'")
    assert {h["name"] for h in hits} == {"square"}


# --------------------------------------------------------------------------- #
#  9. _inject_dependencies — primitives.actor.act round-trip                    #
# --------------------------------------------------------------------------- #


@_handle_project
def test_inject_dependencies_resolves_actor_act():
    """_inject_dependencies injects a Primitives instance for 'primitives.actor.act' deps.

    When a stored function declares depends_on=["primitives.actor.act"],
    _inject_dependencies should call construct_sandbox_root("primitives")
    and place the resulting Primitives instance in the namespace
    under the key "primitives".
    """
    from unify.function_manager.primitives.runtime import Primitives

    fm = _FM()

    func_data = {
        "name": "delegate_contact_research",
        "depends_on": ["primitives.actor.act"],
    }
    namespace = {}
    visited = set()

    fm._inject_dependencies(func_data, namespace=namespace, visited=visited)

    assert "primitives" in namespace
    assert isinstance(namespace["primitives"], Primitives)
    assert hasattr(namespace["primitives"].actor, "act")
    assert callable(namespace["primitives"].actor.act)


@_handle_project
def test_inject_dependencies_actor_idempotent():
    """Injecting "primitives.actor.act" twice doesn't replace or duplicate the namespace entry."""
    fm = _FM()

    func_data_a = {"name": "fn_a", "depends_on": ["primitives.actor.act"]}
    func_data_b = {"name": "fn_b", "depends_on": ["primitives.actor.act"]}
    namespace = {}
    visited = set()

    fm._inject_dependencies(func_data_a, namespace=namespace, visited=visited)
    first_primitives = namespace["primitives"]

    fm._inject_dependencies(func_data_b, namespace=namespace, visited=visited)
    assert namespace["primitives"] is first_primitives


# --------------------------------------------------------------------------- #
#  10. add_functions records primitives.actor.act in depends_on                 #
# --------------------------------------------------------------------------- #


@_handle_project
def test_add_function_with_actor_act_records_dependency():
    """Storing a function that calls primitives.actor.act(...) records 'primitives.actor.act' in depends_on.

    This verifies the storage-time half: DependencyVisitor detects the
    primitives.actor.act call and FunctionManager.add_functions persists it in the
    depends_on field of the stored function record.
    """
    fm = _FM()

    source = (
        "async def research_contact(request: str):\n"
        '    """Delegate contact research to a scoped sub-agent."""\n'
        "    handle = await primitives.actor.act(\n"
        "        request=request,\n"
        '        guidelines="Check all fields.",\n'
        '        prompt_functions=["summarise_report"],\n'
        "    )\n"
        "    return await handle.result()\n"
    )

    result = fm.add_functions(implementations=source)
    assert result == {"research_contact": "added"}

    func_data = fm._get_function_data_by_name(name="research_contact")
    assert func_data is not None
    depends_on = func_data.get("depends_on", [])
    assert "primitives.actor.act" in depends_on


@_handle_project
def test_add_function_with_env_and_compositional_deps_records_all():
    """A function using both primitives.actor.act and a stored function records both deps."""
    fm = _FM()
    fm.add_functions(
        implementations="def summarise(text: str) -> str:\n    return text[:10]\n",
    )

    source = (
        "async def orchestrate(query: str):\n"
        '    """Use the environment namespace and a stored function."""\n'
        "    short = summarise(query)\n"
        "    handle = await primitives.actor.act(request=short)\n"
        "    return await handle.result()\n"
    )

    result = fm.add_functions(implementations=source)
    assert result == {"orchestrate": "added"}

    func_data = fm._get_function_data_by_name(name="orchestrate")
    depends_on = set(func_data.get("depends_on", []))
    assert depends_on == {"primitives.actor.act", "summarise"}


# --------------------------------------------------------------------------- #
#  11. _inject_callables_for_functions with primitives.actor.act dependency    #
# --------------------------------------------------------------------------- #


@_handle_project
def test_inject_callables_for_stored_actor_function():
    """A stored function with depends_on=["primitives.actor.act"] can be prepared for execution.

    This exercises the runtime half of the pipeline:
    1. Store a function that calls primitives.actor.act(...)
    2. Retrieve it and pass through _inject_callables_for_functions
    3. Verify the namespace contains a live Primitives instance
    4. Verify the returned callable is valid
    """
    from unify.function_manager.primitives.runtime import Primitives

    fm = _FM()

    source = (
        "async def delegate_task(request: str):\n"
        '    """Delegate a task to a sub-agent."""\n'
        "    handle = await primitives.actor.act(request=request, timeout=30)\n"
        "    return await handle.result()\n"
    )

    fm.add_functions(implementations=source)

    func_data = fm._get_function_data_by_name(name="delegate_task")
    assert func_data is not None
    assert "primitives.actor.act" in func_data.get("depends_on", [])

    namespace = {}
    callables = fm._inject_callables_for_functions(
        [func_data],
        namespace=namespace,
    )

    assert len(callables) == 1
    assert "primitives" in namespace
    assert isinstance(namespace["primitives"], Primitives)
    assert "delegate_task" in namespace
    assert callable(namespace["delegate_task"])
