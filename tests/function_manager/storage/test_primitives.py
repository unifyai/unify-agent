"""
Tests for action primitives in FunctionManager.

Tests the primitives registry, the seeded ``primitives`` table, and semantic
search that includes both user-defined functions and action primitives.

Primitives are seeded into the ``primitives`` table from the registry when a
FunctionManager is constructed, with stable hash-based function_id values;
user-defined functions live in the ``functions`` table with auto-incrementing
ids. The two are read together through the ``all_functions`` view.
"""

import pytest

from unify import db
from unify.function_manager.function_manager import FunctionManager
from unify.function_manager.primitives import (
    Primitives,
    PrimitiveScope,
    get_primitive_callable,
    get_registry,
)
from tests.helpers import _handle_project

_ACTOR_ACT = "primitives.actor.act"
_ACTOR_CLASS_PATH = "unify.actor.environments.actor._ActorRunner"

# ────────────────────────────────────────────────────────────────────────────
# Fixtures
# ────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def function_manager_factory():
    """Factory fixture that creates FunctionManager instances."""
    managers = []

    def _create():
        fm = FunctionManager()
        managers.append(fm)
        return fm

    yield _create

    # Cleanup all created managers
    for fm in managers:
        try:
            fm.clear()
        except Exception:
            pass


# ────────────────────────────────────────────────────────────────────────────
# 1. Primitives collection tests
# ────────────────────────────────────────────────────────────────────────────


def test_collect_primitives_returns_expected_methods():
    """Registry should return metadata for all auto-discovered methods."""
    from unify.function_manager.primitives.registry import get_primitive_sources

    registry = get_registry()
    scope = PrimitiveScope.all_managers()
    primitives = registry.collect_primitives(scope)

    assert set(primitives) == {_ACTOR_ACT}
    assert primitives[_ACTOR_ACT]["primitive_class"] == _ACTOR_CLASS_PATH

    # Verify primitives match what get_primitive_sources returns
    # (i.e., the auto-discovery is working correctly), keyed by
    # (class name, method) since names are ``primitives.{alias}.{method}``.
    method_to_name = {
        (row["primitive_class"].rsplit(".", 1)[-1], row["primitive_method"]): name
        for name, row in primitives.items()
    }
    for cls, method_names in get_primitive_sources():
        class_name = cls.__name__
        for method_name in method_names:
            assert (
                class_name,
                method_name,
            ) in method_to_name, f"Expected auto-discovered primitive for {class_name}.{method_name} not found"


def test_collect_primitives_has_required_fields():
    """Each primitive should have the required metadata fields including function_id."""
    registry = get_registry()
    scope = PrimitiveScope.all_managers()
    primitives = registry.collect_primitives(scope)

    for data in primitives.values():
        assert "name" in data
        assert "argspec" in data
        assert "docstring" in data
        assert data.get("is_primitive") is True
        assert "primitive_class" in data
        assert "primitive_method" in data
        # Primitives have explicit integer function_ids
        assert "function_id" in data
        assert isinstance(data["function_id"], int)


def test_collect_primitives_has_stable_ids():
    """Primitive function_ids should be stable hash-based IDs."""
    registry = get_registry()
    scope = PrimitiveScope.all_managers()
    primitives = registry.collect_primitives(scope)

    # Verify IDs are deterministic (calling twice gives same IDs)
    primitives2 = registry.collect_primitives(scope)
    for name, data in primitives.items():
        assert (
            primitives2[name]["function_id"] == data["function_id"]
        ), f"ID for '{name}' should be stable across calls"

    # Verify IDs are non-negative integers within the signed 32-bit range
    for data in primitives.values():
        assert isinstance(data["function_id"], int)
        assert 0 <= data["function_id"] <= 0x7FFFFFFF


def test_collect_primitives_has_docstrings():
    """Every primitive carries a non-empty docstring."""
    registry = get_registry()
    scope = PrimitiveScope.all_managers()
    primitives = registry.collect_primitives(scope)

    assert primitives
    for name, p in primitives.items():
        assert p["docstring"].strip(), f"{name} should have a docstring"


def test_compute_primitives_hash_is_stable():
    """Hash should be deterministic for the same primitives."""
    registry = get_registry()
    scope = PrimitiveScope.all_managers()

    hash1 = registry.compute_primitives_hash(primitive_scope=scope)
    hash2 = registry.compute_primitives_hash(primitive_scope=scope)

    assert hash1 == hash2
    assert len(hash1) == 16  # 16 hex chars


def test_compute_primitives_hash_changes_on_modification():
    """Hash should change when primitives are modified."""
    registry = get_registry()
    scope = PrimitiveScope.single("actor")
    primitives = registry.collect_primitives(scope)

    # Compute original hash
    original_hash = registry.compute_primitives_hash(primitives=primitives)

    # Modify a primitive's docstring
    first_name = next(iter(primitives.keys()))
    modified_primitives = dict(primitives)
    modified_primitives[first_name] = dict(modified_primitives[first_name])
    modified_primitives[first_name]["docstring"] = "MODIFIED DOCSTRING FOR TESTING"

    # Hash should change
    modified_hash = registry.compute_primitives_hash(primitives=modified_primitives)
    assert (
        original_hash != modified_hash
    ), "Hash should change when primitives are modified"


# ────────────────────────────────────────────────────────────────────────────
# 2. Seeded primitives table
# ────────────────────────────────────────────────────────────────────────────


@_handle_project
def test_constructing_a_manager_seeds_primitives_table(function_manager_factory):
    """FunctionManager() makes the ``primitives`` table equal to the registry."""
    function_manager_factory()

    rows = db.query(
        "SELECT function_id, name, primitive_class, primitive_method FROM primitives",
    )
    expected = get_registry().collect_primitives(PrimitiveScope.all_managers())
    assert {row["name"] for row in rows} == set(expected)
    for row in rows:
        registry_row = expected[row["name"]]
        assert row["function_id"] == registry_row["function_id"]
        assert row["primitive_class"] == registry_row["primitive_class"]
        assert row["primitive_method"] == registry_row["primitive_method"]


@_handle_project
def test_constructing_two_managers_does_not_duplicate_rows(function_manager_factory):
    """The seed converges: a second manager leaves exactly one row per primitive."""
    function_manager_factory()
    function_manager_factory()

    rows = db.query("SELECT name, COUNT(*) AS n FROM primitives GROUP BY name")
    assert rows
    assert all(row["n"] == 1 for row in rows)
    assert {row["name"] for row in rows} == set(
        get_registry().collect_primitives(PrimitiveScope.all_managers()),
    )


@_handle_project
def test_list_primitives_reads_primitives_table(function_manager_factory):
    """list_primitives() reads the seeded rows."""
    function_manager = function_manager_factory()

    primitives = function_manager.list_primitives()
    stored = {
        row["name"]: row["function_id"]
        for row in db.query("SELECT name, function_id FROM primitives")
    }
    assert primitives
    assert {name: data["function_id"] for name, data in primitives.items()} == stored


@_handle_project
def test_list_primitives_returns_primitive_metadata(function_manager_factory):
    """list_primitives() should return primitive metadata with integer function_ids."""
    function_manager = function_manager_factory()

    primitives = function_manager.list_primitives()

    for name, data in primitives.items():
        assert data.get("is_primitive") is True
        assert "argspec" in data
        assert "docstring" in data
        # Verify function_id is an integer (not None)
        assert "function_id" in data
        assert isinstance(data["function_id"], int)


@_handle_project
def test_primitives_have_stable_ids_in_table(function_manager_factory):
    """Seeded rows expose the same stable ids as the registry."""
    function_manager = function_manager_factory()

    registry = get_registry()
    expected = {
        name: row["function_id"]
        for name, row in registry.collect_primitives(
            PrimitiveScope.all_managers(),
        ).items()
    }
    stored = function_manager.list_primitives()
    assert stored

    for name, data in stored.items():
        assert name in expected, f"Unexpected seeded primitive {name}"
        assert data["function_id"] == expected[name], (
            f"Primitive {name} ID {data['function_id']} does not match "
            f"registry ID {expected[name]}"
        )


@_handle_project
def test_seeded_rows_resolve_to_runtime_callables(function_manager_factory):
    """Stored primitive metadata resolves back to the live runtime method."""
    function_manager = function_manager_factory()
    row = function_manager.list_primitives()[_ACTOR_ACT]

    primitives = Primitives()
    resolved = get_primitive_callable(row, primitives=primitives)

    assert resolved is not None
    assert resolved.__func__ is type(primitives.actor).act
    assert resolved.__self__ is primitives.actor


# ────────────────────────────────────────────────────────────────────────────
# 3. Semantic search with primitives tests
# ────────────────────────────────────────────────────────────────────────────


@_handle_project
def test_search_includes_primitives_by_default(function_manager_factory):
    """search_functions should include primitives by default."""
    function_manager = function_manager_factory()

    # Search for something that should match a primitive
    results = function_manager.search_functions(
        query="spawn a sub-actor for a focused sub-task",
        n=5,
    )

    # Should have results (primitives are seeded on construction)
    assert len(results) > 0

    # At least one result should be a primitive
    has_primitive = any(r.get("is_primitive") for r in results)
    assert has_primitive, "Expected at least one primitive in search results"


@_handle_project
def test_search_ranks_functions_and_primitives_together(function_manager_factory):
    """Search should return both user functions and primitives, ranked together."""
    function_manager = function_manager_factory()

    # Add a user function related to delegation
    implementation = '''
def delegate_subtask(request: str) -> str:
    """Hand a focused sub-task to a helper and return its summary."""
    return f"Delegated: {request}"
'''
    function_manager.add_functions(implementations=[implementation])

    results = function_manager.search_functions(
        query="delegate a focused sub-task to a helper",
        n=10,
    )

    # Should have both user functions and primitives
    user_funcs = [r for r in results if not r.get("is_primitive")]
    primitives = [r for r in results if r.get("is_primitive")]

    assert {r["name"] for r in user_funcs} == {"delegate_subtask"}
    assert {r["name"] for r in primitives} == {_ACTOR_ACT}


# ────────────────────────────────────────────────────────────────────────────
# 4. Clear behaviour
# ────────────────────────────────────────────────────────────────────────────


@_handle_project
def test_clear_preserves_primitives(function_manager_factory):
    """clear() drops stored functions but never the seeded primitives."""
    function_manager = function_manager_factory()
    function_manager.add_functions(implementations="def own():\n    return 1\n")

    count_before = len(function_manager.list_primitives())
    assert count_before > 0

    function_manager.clear()

    assert db.query("SELECT 1 FROM functions") == []
    assert len(function_manager.list_primitives()) == count_before
    assert db.query_one("SELECT COUNT(*) AS n FROM primitives")["n"] == count_before


# ────────────────────────────────────────────────────────────────────────────
# 5. Registry configuration
# ────────────────────────────────────────────────────────────────────────────


def test_manager_spec_has_excluded_methods():
    """ManagerSpec entries should have excluded_methods."""
    registry = get_registry()

    for spec in registry.MANAGERS:
        # excluded_methods should be a frozenset
        assert isinstance(
            spec.excluded_methods,
            frozenset,
        ), f"{spec.manager_alias} excluded_methods should be frozenset"


def test_common_excluded_methods():
    """Common excluded methods should include lifecycle and internal helpers."""
    from unify.function_manager.primitives.registry import _COMMON_EXCLUDED_METHODS

    assert "clear" in _COMMON_EXCLUDED_METHODS
    assert "add_tools" in _COMMON_EXCLUDED_METHODS
    assert "get_tools" in _COMMON_EXCLUDED_METHODS


def test_primitive_callable_unknown_class_is_none():
    """get_primitive_callable() returns None for metadata outside the registry."""
    assert (
        get_primitive_callable(
            {"primitive_class": "unify.nowhere.Nothing", "primitive_method": "act"},
        )
        is None
    )
    assert get_primitive_callable({"primitive_class": _ACTOR_CLASS_PATH}) is None
