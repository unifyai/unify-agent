from __future__ import annotations

import pytest

from unify.guidance_manager.guidance_manager import GuidanceManager
from tests.helpers import _handle_project

# Reads span the assistant's own rows plus the read-only builtins library, so
# whole-view assertions scope to the assistant's own rows explicitly.
OWN_ONLY = "is_builtin = 0"


def _seed(gm: GuidanceManager) -> dict[str, int]:
    """Create three guidance entries and return their IDs keyed by title."""
    ids = {}
    for title, content in [
        ("Alpha", "Guide for alpha procedures"),
        ("Beta", "Guide for beta procedures"),
        ("Gamma", "Guide for gamma procedures"),
    ]:
        out = gm.add_guidance(title=title, content=content)
        ids[title] = out["details"]["guidance_id"]
    return ids


# -- filter_scope -----------------------------------------------------------


@_handle_project
def test_filter_scope_restricts_filter():
    gm = GuidanceManager()
    ids = _seed(gm)

    gm.filter_scope = f"guidance_id = {ids['Beta']}"
    rows = gm.filter()
    assert len(rows) == 1
    assert rows[0].guidance_id == ids["Beta"]

    gm.filter_scope = None


@_handle_project
def test_filter_scope_composes_with_caller_filter():
    gm = GuidanceManager()
    ids = _seed(gm)

    gm.filter_scope = f"guidance_id = {ids['Alpha']} OR guidance_id = {ids['Beta']}"

    rows = gm.filter(filter="title = 'Alpha'")
    assert len(rows) == 1
    assert rows[0].title == "Alpha"

    rows = gm.filter(filter="title = 'Gamma'")
    assert len(rows) == 0

    gm.filter_scope = None


@_handle_project
def test_rejected_filter_returns_error_payload():
    gm = GuidanceManager()
    _seed(gm)

    payload = gm.filter(filter="no_such_column = 1")
    assert isinstance(payload, dict)
    assert payload["error_kind"] == "invalid_filter"
    assert payload["details"]["filter"] == "no_such_column = 1"
    assert "guidance_id" in payload["details"]["columns"]

    # A clause that tries to write is refused rather than run.
    payload = gm.filter(filter="1 = 1; DELETE FROM guidance")
    assert isinstance(payload, dict)
    assert payload["error_kind"] == "invalid_filter"
    assert len(gm.filter(filter=OWN_ONLY)) == 3


# -- exclude_ids ------------------------------------------------------------


@_handle_project
def test_exclude_ids_restricts_filter():
    gm = GuidanceManager()
    ids = _seed(gm)

    gm.exclude_ids = frozenset({ids["Alpha"]})

    rows = gm.filter()
    returned_ids = {r.guidance_id for r in rows}
    assert ids["Alpha"] not in returned_ids
    assert ids["Beta"] in returned_ids
    assert ids["Gamma"] in returned_ids

    gm.exclude_ids = None


@_handle_project
def test_exclude_ids_multiple():
    gm = GuidanceManager()
    ids = _seed(gm)

    gm.exclude_ids = frozenset({ids["Alpha"], ids["Gamma"]})

    rows = gm.filter(filter=OWN_ONLY)
    assert len(rows) == 1
    assert rows[0].guidance_id == ids["Beta"]

    gm.exclude_ids = None


# -- combined filter_scope + exclude_ids ------------------------------------


@_handle_project
def test_scope_and_exclusion_combined():
    gm = GuidanceManager()
    ids = _seed(gm)

    gm.filter_scope = f"guidance_id = {ids['Alpha']} OR guidance_id = {ids['Beta']}"
    gm.exclude_ids = frozenset({ids["Alpha"]})

    rows = gm.filter()
    assert len(rows) == 1
    assert rows[0].guidance_id == ids["Beta"]

    gm.filter_scope = None
    gm.exclude_ids = None


# -- search respects scope/exclusion --------------------------------------


@pytest.mark.requires_provider_key
@_handle_project
def test_search_respects_filter_scope():
    gm = GuidanceManager()
    ids = _seed(gm)

    gm.filter_scope = f"guidance_id = {ids['Alpha']}"

    results = gm.search(references={"title": "procedures"}, k=10)
    returned_ids = {r.guidance_id for r in results}
    assert returned_ids == {ids["Alpha"]}

    gm.filter_scope = None


@pytest.mark.requires_provider_key
@_handle_project
def test_search_respects_exclude_ids():
    gm = GuidanceManager()
    ids = _seed(gm)

    gm.exclude_ids = frozenset({ids["Beta"]})

    # k spans the whole view (own rows + builtins library) so the assertion
    # checks exclusion rather than ranking position.
    results = gm.search(references={"title": "procedures"}, k=100)
    returned_ids = {r.guidance_id for r in results}
    assert ids["Beta"] not in returned_ids
    assert ids["Alpha"] in returned_ids
    assert ids["Gamma"] in returned_ids

    gm.exclude_ids = None


# -- _num_items respects scope/exclusion ------------------------------------


@_handle_project
def test_num_items_respects_filter_scope():
    gm = GuidanceManager()
    builtin_count = gm._num_items()
    ids = _seed(gm)
    assert gm._num_items() == builtin_count + 3

    # The scope applies to the whole view, builtins included.
    gm.filter_scope = f"guidance_id = {ids['Alpha']}"
    assert gm._num_items() == 1

    gm.filter_scope = None


@_handle_project
def test_num_items_respects_exclude_ids():
    gm = GuidanceManager()
    builtin_count = gm._num_items()
    ids = _seed(gm)

    gm.exclude_ids = frozenset({ids["Alpha"], ids["Gamma"]})
    assert gm._num_items() == builtin_count + 1

    gm.exclude_ids = None


# -- clearing scope restores full view --------------------------------------


@_handle_project
def test_clearing_scope_restores_full_view():
    gm = GuidanceManager()
    ids = _seed(gm)

    gm.filter_scope = f"guidance_id = {ids['Gamma']}"
    assert len(gm.filter()) == 1

    gm.filter_scope = None
    assert len(gm.filter(filter=OWN_ONLY)) == 3

    gm.exclude_ids = frozenset({ids["Alpha"], ids["Beta"]})
    assert len(gm.filter(filter=OWN_ONLY)) == 1

    gm.exclude_ids = None
    assert len(gm.filter(filter=OWN_ONLY)) == 3


# -- limit correctness with scoping ----------------------------------------


@_handle_project
def test_limit_with_scope():
    gm = GuidanceManager()
    ids = _seed(gm)

    gm.filter_scope = f"guidance_id = {ids['Alpha']} OR guidance_id = {ids['Beta']}"

    rows = gm.filter(limit=1)
    assert len(rows) == 1
    assert rows[0].guidance_id in {ids["Alpha"], ids["Beta"]}

    rows = gm.filter(limit=10)
    assert len(rows) == 2

    gm.filter_scope = None
