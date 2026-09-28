"""
Integration tests for PrimitiveScope across all system layers.

These tests verify that the scoping mechanism works consistently at ALL levels:
- Tool list (tool_names, ActorEnvironment.get_tools)
- Catalogue reads and search (FunctionManager)
- Primitives syncing (collect_primitives, per-manager hash tracking)
- Sandbox runtime vars (Primitives, ActorEnvironment)

This ensures the single source of truth (PrimitiveScope) controls what the model sees.
"""

import pytest

from unify.function_manager.primitives import (
    PrimitiveScope,
    Primitives,
    get_registry,
)
from unify.function_manager.primitives.registry import get_primitive_sources
from unify.function_manager.function_manager import FunctionManager
from unify.actor.environments import ActorEnvironment
from tests.helpers import _handle_project

_ACTOR_ACT = "primitives.actor.act"
_ACTOR_CLASS_PATH = "unify.actor.environments.actor._ActorRunner"

# ────────────────────────────────────────────────────────────────────────────
# Fixtures
# ────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def scoped_function_manager_factory():
    """Factory that creates FunctionManager with specific scope."""
    managers = []

    def _create(scope: PrimitiveScope):
        fm = FunctionManager(primitive_scope=scope)
        managers.append(fm)
        return fm

    yield _create

    for fm in managers:
        try:
            fm.clear()
        except Exception:
            pass


# ────────────────────────────────────────────────────────────────────────────
# 1. Tool List Scoping
# ────────────────────────────────────────────────────────────────────────────


def test_actor_env_get_tools_exposes_actor_primitives():
    """ActorEnvironment.get_tools() exposes exactly the actor primitives."""
    env = ActorEnvironment()
    assert set(env.get_tools()) == {_ACTOR_ACT}
    assert env.get_instance().primitive_scope == PrimitiveScope.single("actor")


def test_actor_env_allowed_methods_filters_tools_and_prompt():
    """allowed_methods narrows both get_tools() and get_prompt_context()."""
    kept = ActorEnvironment(allowed_methods={_ACTOR_ACT})
    assert kept.allowed_methods == frozenset({_ACTOR_ACT})
    assert set(kept.get_tools()) == {_ACTOR_ACT}
    kept_context = kept.get_prompt_context()
    assert "#### `primitives.actor`" in kept_context
    assert "**`async def primitives.actor.act(" in kept_context

    dropped = ActorEnvironment(allowed_methods={"primitives.actor.nonexistent"})
    assert dropped.get_tools() == {}
    assert "primitives.actor.act(" not in dropped.get_prompt_context()

    unfiltered = ActorEnvironment()
    assert unfiltered.allowed_methods is None
    assert "**`async def primitives.actor.act(" in unfiltered.get_prompt_context()


# ────────────────────────────────────────────────────────────────────────────
# 2. Catalogue Read and Search Scoping
# ────────────────────────────────────────────────────────────────────────────


@_handle_project
def test_catalog_reads_only_scoped_managers(scoped_function_manager_factory):
    """Catalogue reads are filtered to the scoped primitive classes."""
    fm = scoped_function_manager_factory(PrimitiveScope.single("actor"))

    primitives = fm.list_primitives()

    assert set(primitives) == {_ACTOR_ACT}
    assert primitives[_ACTOR_ACT]["primitive_class"] == _ACTOR_CLASS_PATH


@_handle_project
def test_search_functions_respects_scope(scoped_function_manager_factory):
    """search_functions() only returns primitives for scoped classes."""
    fm = scoped_function_manager_factory(PrimitiveScope.single("actor"))

    results = fm.search_functions(query="delegate work to a sub-actor", n=20)

    primitive_hits = [r for r in results if r.get("is_primitive")]
    assert primitive_hits, "the scoped primitive should be searchable"
    for r in primitive_hits:
        assert r["primitive_class"] == _ACTOR_CLASS_PATH


# ────────────────────────────────────────────────────────────────────────────
# 3. Sandbox Runtime Scoping
# ────────────────────────────────────────────────────────────────────────────


def test_primitives_instance_respects_scope():
    """Primitives instance only exposes aliases in its scope."""
    scope = PrimitiveScope.single("actor")
    primitives = Primitives(primitive_scope=scope)

    assert primitives.primitive_scope.scoped_managers == frozenset({"actor"})
    assert primitives.actor.__class__.__name__ == "_ActorRunner"
    assert primitives.actor is primitives.actor, "namespace objects are cached"

    with pytest.raises(AttributeError):
        _ = primitives.files


def test_primitives_default_scope_is_runtime_scope():
    """Primitives() without an explicit scope uses the default runtime scope."""
    from unify.function_manager.primitives import default_runtime_scope

    assert Primitives().primitive_scope is default_runtime_scope()


# ────────────────────────────────────────────────────────────────────────────
# 4. Cross-Layer Consistency
# ────────────────────────────────────────────────────────────────────────────


@_handle_project
def test_scope_consistency_across_layers(scoped_function_manager_factory):
    """Every layer exposes the same tool names for a given scope."""
    registry = get_registry()
    scope = PrimitiveScope.single("actor")

    registry_names = set(registry.tool_names(scope))
    collected_names = set(registry.collect_primitives(scope))
    env_names = set(ActorEnvironment().get_tools())
    catalog_names = set(scoped_function_manager_factory(scope).list_primitives())

    assert registry_names == {_ACTOR_ACT}
    assert collected_names == registry_names
    assert env_names == registry_names
    assert catalog_names == registry_names


def test_primitive_discovery_complete():
    """All auto-discovered primitives should be indexed correctly."""
    registry = get_registry()
    scope = PrimitiveScope.all_managers()
    collected = registry.collect_primitives(scope)

    # Verify against get_primitive_sources, keyed by (class name, method)
    # since names are in ``primitives.{alias}.{method}`` format.
    method_to_name = {
        (row["primitive_class"].rsplit(".", 1)[-1], row["primitive_method"]): name
        for name, row in collected.items()
    }
    for cls, method_names in get_primitive_sources():
        class_name = cls.__name__
        for method_name in method_names:
            assert (
                class_name,
                method_name,
            ) in method_to_name, (
                f"Primitive for {class_name}.{method_name} should be collected"
            )
            # Verify it has all required fields
            data = collected[method_to_name[(class_name, method_name)]]
            assert data["is_primitive"] is True
            assert data["primitive_class"] is not None
            assert data["primitive_method"] == method_name
