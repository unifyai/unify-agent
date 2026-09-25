"""Ranking rows fetched from the store by the meaning of a query.

A skill search fetches its candidate rows with SQL and ranks them here. Each
term pairs a query text with the text of every row it is compared against;
both are embedded (:mod:`unify.common.embeddings`, cached by model and text)
and a row scores the mean cosine similarity over the terms it has text for,
so a query finds the entry that means the same thing whatever words either
side uses. Writes embed their rows ahead of time; a row written while
embeddings were unavailable is embedded by the first search that meets it.

When texts cannot be embedded (no OpenRouter key, or the endpoint fails),
the same rows are ranked by shared words instead
(:mod:`unify.common.text_search`) and a warning names the reason, once per
reason per process.
"""

from __future__ import annotations

import logging
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .embeddings import EmbeddingsUnavailable, embed_many
from .text_search import SIMILARITY_FIELD, newest_first, rank_by_text

logger = logging.getLogger(__name__)

# A query text and, in the order of the rows, the text each row is compared on.
Term = tuple[str, Sequence[str]]

_REPORTED: set[str] = set()


def _report_unavailable(exc: EmbeddingsUnavailable) -> None:
    reason = str(exc)
    if reason in _REPORTED:
        logger.debug("Embeddings unavailable: %s", reason)
        return
    _REPORTED.add(reason)
    logger.warning(
        "Semantic skill search is unavailable (%s); skills are ranked by "
        "shared words until their texts can be embedded.",
        reason,
    )


def rank_by_meaning(
    rows: Sequence[dict[str, Any]],
    terms: Sequence[Term],
    *,
    limit: int,
    id_field: str | None = None,
    backfill: bool = False,
) -> list[dict[str, Any]]:
    """Order ``rows`` by mean cosine similarity to the query of each term.

    A row is not compared on a term whose text is empty for it; a row with
    no text for any term cannot be ranked and, with ``backfill``, joins the
    tail newest first by ``id_field``. Every returned row carries
    ``_similarity``: its mean cosine similarity clamped to ``[0, 1]``
    (``0.0`` for backfilled rows).

    Raises :class:`EmbeddingsUnavailable` when a text cannot be embedded.
    """
    if limit <= 0:
        return []
    terms = [(query, texts) for query, texts in terms if query.strip()]
    if not terms and not backfill:
        return []
    wanted = list(
        dict.fromkeys(
            text for query, texts in terms for text in (query, *texts) if text.strip()
        ),
    )
    vectors: dict[str, np.ndarray] = {}
    if wanted:
        for text, vector in zip(wanted, embed_many(wanted), strict=True):
            norm = float(np.linalg.norm(vector))
            vectors[text] = vector / norm if norm else vector

    scored: list[tuple[float, int, dict[str, Any]]] = []
    unranked: list[dict[str, Any]] = []
    for order, row in enumerate(rows):
        similarities = [
            float(vectors[query] @ vectors[texts[order]])
            for query, texts in terms
            if texts[order].strip()
        ]
        if similarities:
            scored.append((sum(similarities) / len(similarities), order, row))
        else:
            unranked.append(row)
    scored.sort(key=lambda item: (-item[0], item[1]))

    ranked: list[dict[str, Any]] = []
    for similarity, _order, row in scored:
        row[SIMILARITY_FIELD] = min(1.0, max(0.0, similarity))
        ranked.append(row)
    if backfill and len(ranked) < limit:
        for row in newest_first(unranked, id_field):
            row[SIMILARITY_FIELD] = 0.0
            ranked.append(row)
    return ranked[:limit]


def rank_rows(
    rows: Sequence[dict[str, Any]],
    terms: Sequence[Term],
    *,
    word_references: Mapping[str, str] | None,
    limit: int,
    id_field: str | None = None,
    backfill: bool = False,
) -> list[dict[str, Any]]:
    """Rank ``rows`` by meaning, or by the words of ``word_references`` when
    embeddings are unavailable (see :func:`rank_by_meaning` and
    :func:`~unify.common.text_search.rank_by_text`)."""
    try:
        return rank_by_meaning(
            rows,
            terms,
            limit=limit,
            id_field=id_field,
            backfill=backfill,
        )
    except EmbeddingsUnavailable as exc:
        _report_unavailable(exc)
        return rank_by_text(
            rows,
            word_references,
            limit=limit,
            id_field=id_field,
            backfill=backfill,
        )


def embed_ahead(texts: Iterable[str]) -> None:
    """Embed the texts of rows being written, so later searches are lookups.

    A failure is reported and left to the first search that meets the rows.
    """
    texts = [text for text in texts if text.strip()]
    if not texts:
        return
    try:
        embed_many(texts)
    except EmbeddingsUnavailable as exc:
        _report_unavailable(exc)


__all__ = ["Term", "embed_ahead", "rank_by_meaning", "rank_rows"]
