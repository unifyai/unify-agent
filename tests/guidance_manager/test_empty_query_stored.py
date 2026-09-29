"""Symbolic: ``UNIFY_GUIDANCE_EMPTY_QUERY=stored``: a query-less guidance search reads the stored entries.

Without reference text every row is unscored and returned newest first by id.
Built-in ids are hashes (up to 2**31) and stored ids count up from 1, so, as
shipped, the built-in catalogue fills every slot ahead of anything stored: on
28 September two of twelve first-turn guidance searches (``"references": null``)
showed only built-in skills. The embedder is replaced by a deterministic one,
so nothing leaves the process.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from unify.common import semantic_search
from unify.guidance_manager.guidance_manager import GuidanceManager
from unify.settings import ProductionSettings, SETTINGS
from tests.helpers import _handle_project


def _fake_embed(texts):
    vectors = []
    for text in texts:
        seed = int.from_bytes(hashlib.sha256(text.encode()).digest()[:4], "big")
        v = np.random.default_rng(seed).normal(size=16)
        vectors.append(v / np.linalg.norm(v))
    return np.array(vectors)


@pytest.fixture(autouse=True)
def _offline_embed(monkeypatch):
    monkeypatch.setattr(semantic_search, "embed", _fake_embed)


def _store(gm: GuidanceManager) -> list[int]:
    ids = []
    for title in (
        "Pay a contact on Venmo",
        "Rename downloads by date",
        "Download liked songs",
    ):
        out = gm.add_guidance(title=title, content=f"Procedure: {title.lower()}.")
        ids.append(out["details"]["guidance_id"])
    return ids


@_handle_project
def test_off_a_query_less_search_returns_the_built_in_catalogue(monkeypatch):
    monkeypatch.setattr(SETTINGS, "UNIFY_GUIDANCE_EMPTY_QUERY", "")
    gm = GuidanceManager()
    _store(gm)
    rows = gm.search(references=None, k=10)
    assert len(rows) == 10
    assert all(row.is_builtin for row in rows)  # the defect, as shipped


@_handle_project
@pytest.mark.parametrize(
    "references",
    [None, {}, {"title": ""}, {"title": "  ", "content": ""}],
)
def test_on_a_query_less_search_returns_the_stored_entries_newest_first(
    monkeypatch,
    references,
):
    monkeypatch.setattr(SETTINGS, "UNIFY_GUIDANCE_EMPTY_QUERY", "stored")
    gm = GuidanceManager()
    ids = _store(gm)
    rows = gm.search(references=references, k=10)
    assert [row.guidance_id for row in rows] == list(reversed(ids))
    assert not any(row.is_builtin for row in rows)
    assert rows[0].title == "Download liked songs"
    assert rows[0].content == "Procedure: download liked songs."
    # Same shape as any search result.
    assert type(rows[0]) is type(gm.search(references={"title": "venmo"}, k=1)[0])


@_handle_project
def test_on_k_is_respected_and_an_empty_library_returns_nothing(monkeypatch):
    monkeypatch.setattr(SETTINGS, "UNIFY_GUIDANCE_EMPTY_QUERY", "stored")
    gm = GuidanceManager()
    assert gm.search(references=None, k=5) == []
    ids = _store(gm)
    assert [row.guidance_id for row in gm.search(references=None, k=2)] == [
        ids[2],
        ids[1],
    ]


@_handle_project
def test_on_a_search_with_reference_text_is_unchanged(monkeypatch):
    gm = GuidanceManager()
    _store(gm)
    query = {"title": "make a slide deck"}
    monkeypatch.setattr(SETTINGS, "UNIFY_GUIDANCE_EMPTY_QUERY", "")
    off = [
        (row.guidance_id, row.is_builtin) for row in gm.search(references=query, k=10)
    ]
    monkeypatch.setattr(SETTINGS, "UNIFY_GUIDANCE_EMPTY_QUERY", "stored")
    on = [
        (row.guidance_id, row.is_builtin) for row in gm.search(references=query, k=10)
    ]
    assert on == off
    assert any(is_builtin for _, is_builtin in on)  # built-ins still reachable by query


def test_the_setting_accepts_only_empty_or_stored():
    assert (
        ProductionSettings(
            UNIFY_GUIDANCE_EMPTY_QUERY=" Stored ",
        ).UNIFY_GUIDANCE_EMPTY_QUERY
        == "stored"
    )
    assert (
        ProductionSettings(UNIFY_GUIDANCE_EMPTY_QUERY="").UNIFY_GUIDANCE_EMPTY_QUERY
        == ""
    )
    with pytest.raises(ValueError):
        ProductionSettings(UNIFY_GUIDANCE_EMPTY_QUERY="builtin")
