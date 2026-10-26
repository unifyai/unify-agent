"""Tests for GuidanceManager environment exclusion (guidance_id masking).

Mirrors tests/function_manager/test_fm_environment_exclusion.py for the
guidance side.  Verifies:
1. The SQL clause helpers produce correct exclusion clauses
2. GuidanceManager._scope composes caller filter / filter_scope / exclude_ids
3. _resolve_prompt_guidance returns (text, resolved_ids)
4. The wiring in ActorEnvironment.act() sets exclude_ids on the inner GM
"""

from __future__ import annotations

from typing import Optional

from unify.common.sql_filters import and_clauses, not_in
from unify.guidance_manager.guidance_manager import GuidanceManager
from tests.helpers import _handle_project

# ────────────────────────────────────────────────────────────────────────────
# Exclusion clause
# ────────────────────────────────────────────────────────────────────────────


def test_not_in_none_when_empty():
    assert not_in("guidance_id", None) is None
    assert not_in("guidance_id", frozenset()) is None


def test_not_in_single_id():
    assert not_in("guidance_id", frozenset({7})) == "guidance_id NOT IN (7)"


def test_not_in_multiple_ids_sorted():
    result = not_in("guidance_id", frozenset({30, 10, 20}))
    assert result == "guidance_id NOT IN (10, 20, 30)"


# ────────────────────────────────────────────────────────────────────────────
# _scope (composition of caller filter / filter_scope / exclude_ids)
# ────────────────────────────────────────────────────────────────────────────


def _make_gm(
    *,
    filter_scope: Optional[str] = None,
    exclude_ids: Optional[frozenset[int]] = None,
) -> GuidanceManager:
    return GuidanceManager(filter_scope=filter_scope, exclude_ids=exclude_ids)


def test_scope_includes_exclusion():
    gm = _make_gm(filter_scope="is_builtin = 0", exclude_ids={5})
    result = gm._scope("title = 'Deploy'")
    assert result == and_clauses(
        "title = 'Deploy'",
        "is_builtin = 0",
        "guidance_id NOT IN (5)",
    )
    assert "title = 'Deploy'" in result
    assert "is_builtin = 0" in result
    assert "guidance_id NOT IN (5)" in result


def test_scope_exclusion_only():
    gm = _make_gm(exclude_ids={99})
    assert gm._scope(None) == "guidance_id NOT IN (99)"


def test_scope_filter_scope_only():
    gm = _make_gm(filter_scope="is_builtin = 0")
    assert gm._scope(None) == "is_builtin = 0"


def test_scope_all_none_returns_none():
    gm = _make_gm()
    assert gm._scope(None) is None


# ────────────────────────────────────────────────────────────────────────────
# _resolve_prompt_guidance — empty / None fast path (no store rows needed)
# ────────────────────────────────────────────────────────────────────────────


def test_resolve_prompt_guidance_none_input():
    from unify.actor.environments.actor import _resolve_prompt_guidance

    text, ids = _resolve_prompt_guidance(None)
    assert text is None
    assert ids == frozenset()


def test_resolve_prompt_guidance_empty_list():
    from unify.actor.environments.actor import _resolve_prompt_guidance

    text, ids = _resolve_prompt_guidance([])
    assert text is None
    assert ids == frozenset()


# ────────────────────────────────────────────────────────────────────────────
# _resolve_prompt_guidance — with real guidance entries
# ────────────────────────────────────────────────────────────────────────────


def _seed(gm: GuidanceManager) -> dict[str, int]:
    """Create guidance entries and return {title: guidance_id}."""
    ids = {}
    for title, content in [
        ("Deploy Guide", "Step-by-step deployment procedure"),
        ("Review Checklist", "Code review checklist for PRs"),
    ]:
        out = gm.add_guidance(title=title, content=content)
        ids[title] = out["details"]["guidance_id"]
    return ids


@_handle_project
def test_resolve_prompt_guidance_by_title():
    from unify.actor.environments.actor import _resolve_prompt_guidance

    gm = GuidanceManager()
    ids = _seed(gm)

    text, resolved_ids = _resolve_prompt_guidance(["Deploy Guide"])
    assert text is not None
    assert "Deploy Guide" in text
    assert "Step-by-step deployment procedure" in text
    assert f"guidance_id: {ids['Deploy Guide']}" in text
    assert resolved_ids == frozenset({ids["Deploy Guide"]})


@_handle_project
def test_resolve_prompt_guidance_by_id():
    from unify.actor.environments.actor import _resolve_prompt_guidance

    gm = GuidanceManager()
    ids = _seed(gm)

    text, resolved_ids = _resolve_prompt_guidance([ids["Review Checklist"]])
    assert text is not None
    assert "Review Checklist" in text
    assert f"guidance_id: {ids['Review Checklist']}" in text
    assert resolved_ids == frozenset({ids["Review Checklist"]})


@_handle_project
def test_resolve_prompt_guidance_title_with_apostrophe():
    """A title containing a single quote resolves like any other title."""
    from unify.actor.environments.actor import _resolve_prompt_guidance

    gm = GuidanceManager()
    out = gm.add_guidance(
        title="Client's Refund Policy",
        content="Refund within 30 days",
    )
    guidance_id = out["details"]["guidance_id"]

    text, resolved_ids = _resolve_prompt_guidance(["Client's Refund Policy"])
    assert text is not None
    assert "Refund within 30 days" in text
    assert resolved_ids == frozenset({guidance_id})


@_handle_project
def test_resolve_prompt_guidance_mixed():
    from unify.actor.environments.actor import _resolve_prompt_guidance

    gm = GuidanceManager()
    ids = _seed(gm)

    text, resolved_ids = _resolve_prompt_guidance(
        ["Deploy Guide", ids["Review Checklist"]],
    )
    assert text is not None
    assert "Deploy Guide" in text
    assert "Review Checklist" in text
    assert resolved_ids == frozenset({ids["Deploy Guide"], ids["Review Checklist"]})


@_handle_project
def test_resolve_prompt_guidance_renders_function_ids():
    """function_ids cross-references appear in the rendered guidance text."""
    from unify.actor.environments.actor import _resolve_prompt_guidance

    gm = GuidanceManager()
    gm.add_guidance(
        title="Linked Guide",
        content="Guide with linked functions",
        function_ids=[10, 20],
    )

    text, _ = _resolve_prompt_guidance(["Linked Guide"])
    assert text is not None
    assert "Related functions:" in text
    assert "10" in text
    assert "20" in text


@_handle_project
def test_resolve_prompt_guidance_unmatched_returns_empty():
    """Identifiers that don't match any guidance produce no text and no IDs."""
    from unify.actor.environments.actor import _resolve_prompt_guidance

    GuidanceManager()

    text, resolved_ids = _resolve_prompt_guidance(["Nonexistent Guide"])
    assert text is None
    assert resolved_ids == frozenset()


# ────────────────────────────────────────────────────────────────────────────
# Wiring: _build_scoped_gm receives exclude_ids from resolved prompt_guidance
# ────────────────────────────────────────────────────────────────────────────


@_handle_project
def test_build_scoped_gm_receives_exclude_ids():
    """Simulates the wiring in ActorEnvironment.act(): resolved guidance IDs
    are set on the inner GuidanceManager via exclude_ids, masking those
    entries from subsequent discovery queries."""
    from unify.actor.environments.actor import (
        _build_scoped_gm,
        _resolve_prompt_guidance,
    )

    gm = GuidanceManager()
    ids = _seed(gm)

    _, resolved_ids = _resolve_prompt_guidance(["Deploy Guide"])
    assert resolved_ids

    inner_gm = _build_scoped_gm(None)
    assert inner_gm.exclude_ids is None

    inner_gm.exclude_ids = resolved_ids
    assert inner_gm.exclude_ids == frozenset({ids["Deploy Guide"]})

    rows = inner_gm.filter()
    returned_ids = {r.guidance_id for r in rows}
    assert ids["Deploy Guide"] not in returned_ids
    assert ids["Review Checklist"] in returned_ids


@_handle_project
def test_build_scoped_gm_applies_guidance_scope():
    """A guidance_scope becomes the inner manager's SQL filter_scope."""
    from unify.actor.environments.actor import _build_scoped_gm

    gm = GuidanceManager()
    ids = _seed(gm)

    inner_gm = _build_scoped_gm("title LIKE '%Checklist%'")
    assert inner_gm.filter_scope == "title LIKE '%Checklist%'"
    assert {r.guidance_id for r in inner_gm.filter()} == {ids["Review Checklist"]}
