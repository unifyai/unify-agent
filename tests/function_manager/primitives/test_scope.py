"""Tests for PrimitiveScope."""

import pytest

from unify.function_manager.primitives.scope import (
    PrimitiveScope,
    VALID_MANAGER_ALIASES,
    default_runtime_scope,
)

# ────────────────────────────────────────────────────────────────────────────
# PrimitiveScope validation tests
# ────────────────────────────────────────────────────────────────────────────


def test_valid_single_manager():
    """Can create scope with a single valid alias."""
    scope = PrimitiveScope(scoped_managers=frozenset({"actor"}))
    assert scope.scoped_managers == frozenset({"actor"})
    assert scope.includes("actor")
    assert not scope.includes("files")


def test_invalid_manager_raises():
    """Invalid alias raises ValueError."""
    with pytest.raises(ValueError, match="Invalid manager aliases"):
        PrimitiveScope(scoped_managers=frozenset({"invalid_manager"}))


def test_mixed_valid_invalid_raises():
    """A valid alias does not mask an invalid one in the same scope."""
    with pytest.raises(ValueError, match="Invalid manager aliases"):
        PrimitiveScope(scoped_managers=frozenset({"actor", "not_a_manager"}))


def test_empty_scope_exposes_nothing():
    """An empty scope is valid: a runtime granted no primitives."""
    scope = PrimitiveScope.none()
    assert scope == PrimitiveScope(scoped_managers=frozenset())
    assert scope.scoped_managers == frozenset()
    assert not scope.includes("actor")
    assert scope.scope_key == ""


def test_out_of_scope_namespace_is_refused_with_its_reason():
    """Reaching a namespace outside the scope names what the runtime lacks."""
    from unify.function_manager.primitives import Primitives

    primitives = Primitives(primitive_scope=PrimitiveScope.none())
    with pytest.raises(AttributeError) as refused:
        primitives.actor
    assert str(refused.value) == (
        "primitives.actor is not available to this actor: it was started "
        "without Actor Delegation (Spawn focused sub-actors for isolated "
        "multi-step sub-tasks). Complete the task with the tools this actor has."
    )


def test_scope_key_single_manager():
    """scope_key is the alias itself for a single-alias scope."""
    scope = PrimitiveScope.single("actor")
    assert scope.scope_key == "actor"


# ────────────────────────────────────────────────────────────────────────────
# Factory method tests
# ────────────────────────────────────────────────────────────────────────────


def test_all_managers_factory():
    """all_managers() creates scope with every valid alias."""
    scope = PrimitiveScope.all_managers()
    assert scope.scoped_managers == VALID_MANAGER_ALIASES


def test_single_factory():
    """single() creates scope with one alias."""
    scope = PrimitiveScope.single("actor")
    assert scope.scoped_managers == frozenset({"actor"})


def test_single_factory_invalid_raises():
    """single() with invalid alias raises ValueError."""
    with pytest.raises(ValueError, match="Invalid manager aliases"):
        PrimitiveScope.single("not_a_manager")


def test_single_factory_for_each_manager():
    """single() works for every valid alias."""
    for alias in VALID_MANAGER_ALIASES:
        scope = PrimitiveScope.single(alias)
        assert scope.scoped_managers == frozenset({alias})
        assert scope.includes(alias)


def test_default_runtime_scope_exposes_every_manager():
    """default_runtime_scope() exposes the full alias set."""
    assert default_runtime_scope().scoped_managers == VALID_MANAGER_ALIASES
    assert default_runtime_scope() is default_runtime_scope()


# ────────────────────────────────────────────────────────────────────────────
# Immutability and equality tests
# ────────────────────────────────────────────────────────────────────────────


def test_frozen_immutable():
    """PrimitiveScope is frozen (immutable)."""
    scope = PrimitiveScope(scoped_managers=frozenset({"actor"}))
    with pytest.raises(AttributeError):
        scope.scoped_managers = frozenset()  # type: ignore


def test_scope_equality():
    """Two separately constructed scopes with the same aliases are equal."""
    scope1 = PrimitiveScope(scoped_managers=frozenset({"actor"}))
    scope2 = PrimitiveScope.single("actor")
    assert scope1 is not scope2
    assert scope1 == scope2


def test_scope_hashable():
    """PrimitiveScope can be used as dict key, keyed by value."""
    scope1 = PrimitiveScope(scoped_managers=frozenset({"actor"}))
    scope2 = PrimitiveScope.single("actor")

    d = {scope1: "value1"}
    assert d[scope2] == "value1"


# ────────────────────────────────────────────────────────────────────────────
# VALID_MANAGER_ALIASES tests
# ────────────────────────────────────────────────────────────────────────────


def test_valid_manager_aliases_contains_expected():
    """VALID_MANAGER_ALIASES is exactly the actor alias."""
    assert VALID_MANAGER_ALIASES == {"actor"}


def test_valid_manager_aliases_is_frozenset():
    """VALID_MANAGER_ALIASES is immutable."""
    assert isinstance(VALID_MANAGER_ALIASES, frozenset)
