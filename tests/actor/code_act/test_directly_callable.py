"""
Tests for the directly-callable environment feature.

Covers:
1. matches_segment — dotted-path segment matching
2. resolve_directly_callable — pattern expansion + error handling
3. ActorEnvironment per-method filtering (allowed_methods)
4. _build_environments_from_db — environment construction from DB patterns
"""

import pytest

from unify.actor.environments.actor import ActorEnvironment
from unify.actor.environments.base import (
    matches_segment,
    resolve_directly_callable,
)
from unify.actor.environments.function_store import FunctionStoreEnvironment
from unify.function_manager.primitives.registry import get_registry

# ────────────────────────────────────────────────────────────────────────────
# 1. matches_segment
# ────────────────────────────────────────────────────────────────────────────


def test_matches_segment_exact():
    """Exact match returns True."""
    assert matches_segment("primitives.actor.act", "primitives.actor.act")


def test_matches_segment_ancestor_one_level():
    """One-level ancestor matches."""
    assert matches_segment("primitives.actor", "primitives.actor.act")


def test_matches_segment_ancestor_two_levels():
    """Two-level ancestor matches."""
    assert matches_segment("primitives", "primitives.actor.act")


def test_matches_segment_partial_segment_no_match():
    """Partial segment (not on a dot boundary) does NOT match."""
    assert not matches_segment("primitives.act", "primitives.actor.act")


def test_matches_segment_different_root():
    """Different root does not match."""
    assert not matches_segment("functions", "primitives.actor.act")


def test_matches_segment_longer_pattern():
    """Pattern longer than name does not match."""
    assert not matches_segment(
        "primitives.actor.act.extra",
        "primitives.actor.act",
    )


def test_matches_segment_single_segment():
    """Single-segment names work for both pattern and name."""
    assert matches_segment("alpha", "alpha")
    assert not matches_segment("alpha", "beta")


def test_matches_segment_namespace_matches_children():
    """Namespace-only pattern matches all children."""
    assert matches_segment("functions", "functions.alpha")
    assert matches_segment("functions", "functions.beta")
    assert matches_segment("my_service", "my_service.do_something")


# ────────────────────────────────────────────────────────────────────────────
# 2. resolve_directly_callable
# ────────────────────────────────────────────────────────────────────────────


_ALL_TOOLS = {
    "primitives.actor.act",
    "functions.alpha",
    "functions.beta",
    "my_service.do_something",
    "my_service.other_method",
}


def test_resolve_exact_match():
    """Exact name resolves to itself."""
    result = resolve_directly_callable(["functions.alpha"], _ALL_TOOLS)
    assert result == {"functions.alpha"}


def test_resolve_namespace_expands():
    """Namespace pattern expands to all children."""
    result = resolve_directly_callable(["functions"], _ALL_TOOLS)
    assert result == {"functions.alpha", "functions.beta"}


def test_resolve_top_level_namespace():
    """Top-level namespace expands to everything under it."""
    result = resolve_directly_callable(["primitives"], _ALL_TOOLS)
    assert result == {"primitives.actor.act"}


def test_resolve_multiple_patterns():
    """Multiple patterns are unioned."""
    result = resolve_directly_callable(
        ["primitives.actor.act", "functions.alpha"],
        _ALL_TOOLS,
    )
    assert result == {"primitives.actor.act", "functions.alpha"}


def test_resolve_mixed_granularity():
    """Mix of exact and namespace patterns works."""
    result = resolve_directly_callable(
        ["primitives.actor", "functions.alpha"],
        _ALL_TOOLS,
    )
    assert result == {"primitives.actor.act", "functions.alpha"}


def test_resolve_unknown_pattern_raises():
    """Pattern matching zero tools raises ValueError."""
    with pytest.raises(ValueError, match="did not match"):
        resolve_directly_callable(["nonexistent"], _ALL_TOOLS)


def test_resolve_partial_segment_raises():
    """Partial segment pattern raises (not a dotted ancestor)."""
    with pytest.raises(ValueError, match="did not match"):
        resolve_directly_callable(["primitives.act"], _ALL_TOOLS)


def test_resolve_custom_env_namespace():
    """Custom environment namespace resolves to its children."""
    result = resolve_directly_callable(["my_service"], _ALL_TOOLS)
    assert result == {"my_service.do_something", "my_service.other_method"}


# ────────────────────────────────────────────────────────────────────────────
# 3. ActorEnvironment per-method filtering
# ────────────────────────────────────────────────────────────────────────────


def test_actor_env_allowed_methods_filters_get_tools():
    """get_tools() only returns methods in the allowed set."""
    env = ActorEnvironment(allowed_methods={"primitives.actor.act"})
    tools = env.get_tools()

    assert set(tools) == {"primitives.actor.act"}


def test_actor_env_allowed_methods_excludes_unlisted():
    """An allowed set that names no actor method exposes no tools."""
    env = ActorEnvironment(allowed_methods={"primitives.actor.plan"})

    assert env.get_tools() == {}
    assert env.allowed_methods == frozenset({"primitives.actor.plan"})


def test_actor_env_allowed_methods_preserves_function_id():
    """Filtered tools still have correct function_id and function_context."""
    registry = get_registry()
    env = ActorEnvironment(allowed_methods={"primitives.actor.act"})
    meta = env.get_tools()["primitives.actor.act"]

    assert meta.function_id == registry.get_function_id("actor", "act")
    assert meta.function_context == "primitive"
    assert meta.is_steerable is True


def test_actor_env_allowed_methods_none_returns_all():
    """allowed_methods=None (default) returns all methods."""
    env_all = ActorEnvironment()
    env_filtered = ActorEnvironment(allowed_methods=None)

    assert env_all.allowed_methods is None
    assert env_all.get_tools().keys() == env_filtered.get_tools().keys()
    assert set(env_all.get_tools()) == {"primitives.actor.act"}


def test_actor_env_allowed_methods_filters_prompt_context():
    """get_prompt_context() renders only the allowed methods as a method reference."""
    env = ActorEnvironment(allowed_methods={"primitives.actor.act"})
    context = env.get_prompt_context()

    assert "### Method Reference" in context
    assert "`primitives.actor`" in context
    assert ".act(" in context
    assert "### `primitives.actor` — Actor Delegation" not in context


def test_actor_env_allowed_methods_empty_prompt_when_nothing_matches():
    """An allowed set outside the actor namespace renders no method docs."""
    env = ActorEnvironment(allowed_methods={"functions.alpha"})

    assert env.get_prompt_context() == ""


def test_actor_env_unfiltered_prompt_includes_delegation_header():
    """Without a filter the full delegation guidance is rendered."""
    env = ActorEnvironment()
    context = env.get_prompt_context()

    assert "### `primitives.actor` — Actor Delegation" in context
    assert "primitives.actor.act(" in context


# ────────────────────────────────────────────────────────────────────────────
# 4. _build_environments_from_db
# ────────────────────────────────────────────────────────────────────────────


def _make_mock_fm(known_names: dict):
    """Create a mock FunctionManager whose list_functions returns *known_names*."""
    from unittest.mock import MagicMock

    fm = MagicMock()
    fm.list_functions.return_value = known_names
    return fm


def test_build_envs_from_db_primitives_only():
    """The actor primitive pattern produces a filtered ActorEnvironment."""
    from unify.actor.environments.actor import _build_environments_from_db

    fm = _make_mock_fm({"primitives.actor.act": {}})
    envs = _build_environments_from_db(["primitives.actor.act"], fm)

    assert len(envs) == 1
    env = envs[0]
    assert isinstance(env, ActorEnvironment)
    assert env.allowed_methods == frozenset({"primitives.actor.act"})
    assert set(env.get_tools()) == {"primitives.actor.act"}


def test_build_envs_from_db_empty_prompt_functions():
    """Empty prompt_functions produces no environments."""
    from unify.actor.environments.actor import _build_environments_from_db

    envs = _build_environments_from_db([], None)
    assert envs == []


def test_build_envs_from_db_none_prompt_functions():
    """None prompt_functions produces no environments."""
    from unify.actor.environments.actor import _build_environments_from_db

    envs = _build_environments_from_db(None, None)
    assert envs == []


def test_build_envs_from_db_unknown_pattern_raises():
    """Unknown pattern raises ValueError when no names match."""
    from unify.actor.environments.actor import _build_environments_from_db

    fm = _make_mock_fm({"primitives.actor.act": {}})
    with pytest.raises(ValueError, match="did not match"):
        _build_environments_from_db(["nonexistent"], fm)


def test_build_envs_from_db_namespace_expansion():
    """A namespace pattern expands to the actor methods stored under it."""
    from unify.actor.environments.actor import _build_environments_from_db

    fm = _make_mock_fm({"primitives.actor.act": {}, "alpha": {}})
    envs = _build_environments_from_db(["primitives"], fm)

    assert len(envs) == 1
    assert isinstance(envs[0], ActorEnvironment)
    assert set(envs[0].get_tools()) == {"primitives.actor.act"}


def test_build_envs_from_db_bare_names_use_function_store():
    """Bare compositional names resolve to a FunctionStoreEnvironment."""
    from unify.actor.environments.actor import _build_environments_from_db

    fm = _make_mock_fm({"alpha": {}, "beta": {}})
    fm._get_function_data_by_name.return_value = {
        "function_id": 1,
        "name": "alpha",
        "docstring": "Alpha.",
    }
    envs = _build_environments_from_db(["alpha"], fm)

    assert len(envs) == 1
    assert isinstance(envs[0], FunctionStoreEnvironment)
    fm._get_function_data_by_name.assert_called_once_with(name="alpha")
    assert set(envs[0].get_tools()) == {"functions.alpha"}


def test_build_envs_from_db_skips_foreign_dotted_names():
    """Dotted names outside ``primitives.actor`` are skipped, not built."""
    from unify.actor.environments.actor import _build_environments_from_db

    fm = _make_mock_fm({"other.thing": {}, "primitives.actor.act": {}})
    envs = _build_environments_from_db(["other.thing"], fm)

    assert envs == []
