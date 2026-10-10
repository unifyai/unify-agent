"""
Integration tests for FunctionManager environment exclusion.

Verifies that when exclude_primitive_ids is set on a FunctionManager instance,
the excluded primitives do NOT appear in search_functions, list_functions,
filter_functions, or list_primitives results -- using real store queries.
"""

import pytest

from unify.function_manager.function_manager import FunctionManager
from unify.function_manager.primitives import PrimitiveScope, get_registry
from unify.actor.environments import ActorEnvironment
from tests.helpers import _handle_project

# ────────────────────────────────────────────────────────────────────────────
# Fixtures
# ────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def fm_factory():
    """Factory that creates FunctionManager instances with context cleanup."""
    managers = []

    def _create(**kwargs):
        fm = FunctionManager(**kwargs)
        managers.append(fm)
        return fm

    yield _create

    for fm in managers:
        try:
            fm.clear()
        except Exception:
            pass


# ────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────

_ACTOR_SCOPE = PrimitiveScope.single("actor")
_ACTOR_ACT = "primitives.actor.act"


def _get_actor_act_id() -> int:
    """Get the stable function_id for primitives.actor.act."""
    return get_registry().get_function_id("actor", "act")


# ────────────────────────────────────────────────────────────────────────────
# 1. list_primitives exclusion
# ────────────────────────────────────────────────────────────────────────────


@_handle_project
def test_list_primitives_excludes_tagged_ids(fm_factory):
    """list_primitives() should not return primitives whose IDs are excluded."""
    actor_act_id = _get_actor_act_id()

    # Baseline: unexcluded FM sees actor.act
    fm_all = fm_factory(primitive_scope=_ACTOR_SCOPE)
    prims_all = fm_all.list_primitives()
    assert _ACTOR_ACT in prims_all, "Baseline: actor.act should be visible"

    # Excluded FM should NOT see actor.act
    fm_excl = fm_factory(
        primitive_scope=_ACTOR_SCOPE,
        exclude_primitive_ids=frozenset({actor_act_id}),
    )
    prims_excl = fm_excl.list_primitives()
    assert _ACTOR_ACT not in prims_excl, "actor.act should be excluded"


# ────────────────────────────────────────────────────────────────────────────
# 2. list_functions exclusion
# ────────────────────────────────────────────────────────────────────────────


@_handle_project
def test_list_functions_excludes_tagged_primitive_ids(fm_factory):
    """list_functions() should not return primitives whose IDs are excluded."""
    actor_act_id = _get_actor_act_id()

    # Baseline
    fm_all = fm_factory(primitive_scope=_ACTOR_SCOPE)
    listing_all = fm_all.list_functions()
    assert _ACTOR_ACT in listing_all

    # Excluded
    fm_excl = fm_factory(
        primitive_scope=_ACTOR_SCOPE,
        exclude_primitive_ids=frozenset({actor_act_id}),
    )
    listing_excl = fm_excl.list_functions()
    assert _ACTOR_ACT not in listing_excl


# ────────────────────────────────────────────────────────────────────────────
# 3. search_functions exclusion
# ────────────────────────────────────────────────────────────────────────────


@pytest.mark.requires_provider_key
@_handle_project
def test_search_functions_excludes_tagged_primitive_ids(fm_factory):
    """search_functions() should not return primitives whose IDs are excluded."""
    actor_act_id = _get_actor_act_id()
    query = "spawn a sub-actor for a focused sub-task"

    # Baseline: search should find actor.act
    fm_all = fm_factory(primitive_scope=_ACTOR_SCOPE)
    hits_all = fm_all.search_functions(query=query, n=20)
    names_all = {h["name"] for h in hits_all}
    assert _ACTOR_ACT in names_all, "Baseline: search should find actor.act"

    # Excluded: search should NOT find actor.act
    fm_excl = fm_factory(
        primitive_scope=_ACTOR_SCOPE,
        exclude_primitive_ids=frozenset({actor_act_id}),
    )
    hits_excl = fm_excl.search_functions(query=query, n=20)
    names_excl = {h["name"] for h in hits_excl}
    assert _ACTOR_ACT not in names_excl, "actor.act should be excluded from search"


# ────────────────────────────────────────────────────────────────────────────
# 4. filter_functions exclusion
# ────────────────────────────────────────────────────────────────────────────


@_handle_project
def test_filter_functions_excludes_tagged_primitive_ids(fm_factory):
    """filter_functions() should not return primitives whose IDs are excluded."""
    actor_act_id = _get_actor_act_id()

    # Baseline
    fm_all = fm_factory(primitive_scope=_ACTOR_SCOPE)
    hits_all = fm_all.filter_functions(filter="is_primitive = 1")
    names_all = {h["name"] for h in hits_all}
    assert _ACTOR_ACT in names_all

    # Excluded
    fm_excl = fm_factory(
        primitive_scope=_ACTOR_SCOPE,
        exclude_primitive_ids=frozenset({actor_act_id}),
    )
    hits_excl = fm_excl.filter_functions(filter="is_primitive = 1")
    names_excl = {h["name"] for h in hits_excl}
    assert _ACTOR_ACT not in names_excl


@_handle_project
def test_filter_functions_handles_production_sized_primitive_exclusions(fm_factory):
    """Large primitive exclusion sets should not trigger store recursion,
    and IDs that match nothing leave the real primitive visible."""
    production_like_ids = frozenset(range(1000, 1120))
    assert _get_actor_act_id() not in production_like_ids

    fm = fm_factory(
        primitive_scope=PrimitiveScope.all_managers(),
        exclude_primitive_ids=production_like_ids,
    )

    hits = fm.filter_functions(
        filter="is_primitive = 1",
        limit=5,
        include_implementations=False,
    )

    assert isinstance(hits, list)
    assert _ACTOR_ACT in {h["name"] for h in hits}


# ────────────────────────────────────────────────────────────────────────────
# 5. Environment-driven exclusion (CodeActActor path)
# ────────────────────────────────────────────────────────────────────────────


def test_environment_function_ids_match_exclusion_targets():
    """The function_ids tagged on ActorEnvironment tools should correspond
    to actual primitives in the FunctionManager, ensuring the exclusion targets
    the right rows."""
    registry = get_registry()

    # Get IDs from environment
    env = ActorEnvironment()
    env_ids = {
        meta.function_id
        for meta in env.get_tools().values()
        if meta.function_id is not None
    }

    # Get IDs from collect_primitives (the source the primitives table is seeded from)
    collected = registry.collect_primitives(_ACTOR_SCOPE)
    collected_ids = {row["function_id"] for row in collected.values()}

    # They should be identical
    assert (
        env_ids == collected_ids
    ), f"Environment IDs {env_ids} should match collected primitive IDs {collected_ids}"
