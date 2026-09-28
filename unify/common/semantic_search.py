"""Ranking rows from the store by how close their text is in meaning to a query.

The skill libraries hold tens to hundreds of entries, so a search fetches
every candidate row with SQL and ranks it here. Each text is embedded as a
unit vector, almost always from the cache, so a dot product is its cosine
similarity to the query.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from .embeddings import embed

SIMILARITY_FIELD = "_similarity"


def rank_by_similarity(
    rows: Sequence[dict[str, Any]],
    references: Mapping[str, str] | None,
    *,
    limit: int,
    id_field: str,
) -> list[dict[str, Any]]:
    """Order ``rows`` by semantic similarity to the reference texts.

    ``references`` maps a field name to the text that field is compared
    with. A row's similarity is the cosine similarity of each referenced
    field's text to its reference, averaged over the fields where the row has
    text and clipped to 0–1; every returned row carries it as
    ``_similarity``. Rows with no text to compare follow the scored rows,
    newest first by ``id_field``, so a vague query still returns a sample of
    the library.
    """
    references = {
        field: text for field, text in (references or {}).items() if text.strip()
    }
    compared = [
        (index, field, str(row[field]))
        for index, row in enumerate(rows)
        for field in references
        if str(row.get(field) or "").strip()
    ]
    scores: dict[int, list[float]] = {}
    if compared:
        texts = list(
            dict.fromkeys([*references.values(), *(text for *_, text in compared)]),
        )
        vectors = dict(zip(texts, embed(texts)))
        for index, field, text in compared:
            scores.setdefault(index, []).append(
                float(vectors[text] @ vectors[references[field]]),
            )
    similarity = {
        index: max(0.0, sum(values) / len(values)) for index, values in scores.items()
    }
    unscored = [index for index in range(len(rows)) if index not in similarity]
    order = sorted(similarity, key=lambda index: (-similarity[index], index))
    order += sorted(unscored, key=lambda index: -rows[index][id_field])

    ranked: list[dict[str, Any]] = []
    for index in order[:limit]:
        rows[index][SIMILARITY_FIELD] = similarity.get(index, 0.0)
        ranked.append(rows[index])
    return ranked


__all__ = ["SIMILARITY_FIELD", "rank_by_similarity"]
