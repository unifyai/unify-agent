"""Ranking rows by meaning, and the word-match fallback, with no network."""

from __future__ import annotations

import pytest

from unify.common.semantic_search import embed_ahead, rank_by_meaning, rank_rows

pytestmark = pytest.mark.no_unify_context

ROWS = [
    {"id": 1, "title": "Company directory", "content": "Look up a company record."},
    {"id": 2, "title": "Utility bills", "content": "Pay each utility bill on time."},
    {"id": 3, "title": "Sourdough", "content": "Bake a sourdough loaf."},
]
QUERY = "settle what I owe the electricity company"


def _terms(rows, references):
    return [
        (text, [str(row.get(field) or "") for row in rows])
        for field, text in references.items()
    ]


def _copies():
    return [dict(row) for row in ROWS]


def test_rows_rank_by_meaning_not_by_shared_words(fake_embeddings):
    rows = _copies()

    ranked = rank_by_meaning(rows, _terms(rows, {"content": QUERY}), limit=3)

    # Only "Company directory" shares a word with the query; the bill entry
    # shares its meaning.
    assert [row["title"] for row in ranked] == [
        "Utility bills",
        "Company directory",
        "Sourdough",
    ]
    similarities = [row["_similarity"] for row in ranked]
    assert similarities == sorted(similarities, reverse=True)
    assert all(0.0 <= value <= 1.0 for value in similarities)


def test_several_terms_score_their_mean(fake_embeddings):
    rows = _copies()

    ranked = rank_by_meaning(
        rows,
        _terms(rows, {"title": "bread", "content": "bake a loaf"}),
        limit=1,
    )

    assert [row["title"] for row in ranked] == ["Sourdough"]


def test_rows_without_text_are_backfilled_newest_first(fake_embeddings):
    rows = _copies() + [{"id": 4, "title": "Blank", "content": ""}]

    ranked = rank_by_meaning(
        rows,
        _terms(rows, {"content": QUERY}),
        limit=4,
        id_field="id",
        backfill=True,
    )

    assert ranked[-1]["title"] == "Blank"
    assert ranked[-1]["_similarity"] == 0.0


def test_no_query_backfills_without_embedding_anything(fake_embeddings):
    rows = _copies()

    ranked = rank_by_meaning(rows, [], limit=2, id_field="id", backfill=True)

    assert [row["id"] for row in ranked] == [3, 2]
    assert fake_embeddings.calls == []


def test_rank_rows_falls_back_to_word_match(
    no_embedding_key,
    semantic_search_warnings,
):
    for _ in range(2):
        rows = _copies()
        ranked = rank_rows(
            rows,
            _terms(rows, {"content": QUERY}),
            word_references={"content": QUERY},
            limit=3,
            id_field="id",
            backfill=True,
        )

    # Word match puts the entry that shares "company" first.
    assert ranked[0]["title"] == "Company directory"
    # One warning per reason, not one per search.
    warnings = semantic_search_warnings()
    assert len(warnings) == 1
    assert "ranked by shared words" in warnings[0]
    assert "OPENROUTER_API_KEY" in warnings[0]


def test_embed_ahead_leaves_unembeddable_texts_to_search(no_embedding_key):
    embed_ahead(["Pay each utility bill on time.", "   "])
