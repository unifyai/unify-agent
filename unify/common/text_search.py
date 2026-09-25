"""Plain word matching over rows fetched from the store.

Skill search ranks by meaning (:mod:`unify.common.semantic_search`); this is
the ranking it falls back to when texts cannot be embedded, ordering the
candidate rows by how many query tokens each contains.
"""

from __future__ import annotations

import re
from typing import Any, Mapping, Sequence

SIMILARITY_FIELD = "_similarity"

_TOKEN = re.compile(r"[a-z0-9]+")


def query_tokens(text: str) -> list[str]:
    """Distinct lower-case alphanumeric tokens of ``text``, in first-seen order."""
    return list(dict.fromkeys(_TOKEN.findall(str(text).lower())))


def text_match(
    row: Mapping[str, Any],
    references: Mapping[str, str],
) -> tuple[int, int]:
    """Count how many query tokens ``row`` contains.

    ``references`` maps a field name to the text to look for in it. A token
    hits a field when some word of the field's value starts with it, so
    ``slide`` finds ``slides`` and ``deploy`` finds ``deploying``; a token
    of one or two characters must match a whole word, so ``a`` does not
    hit ``and``. Returns ``(matched, hits)``: the number of distinct tokens
    found in at least one field, and the total number of (token, field)
    hits, which ranks a row whose name and docstring both carry a token
    above one where only the docstring does.
    """
    matched: set[str] = set()
    hits = 0
    for field, text in references.items():
        value = row.get(field)
        if value is None:
            continue
        words = set(_TOKEN.findall(str(value).lower()))
        for token in query_tokens(text):
            if len(token) < 3:
                found = token in words
            else:
                found = any(word.startswith(token) for word in words)
            if found:
                matched.add(token)
                hits += 1
    return len(matched), hits


def rank_by_text(
    rows: Sequence[dict[str, Any]],
    references: Mapping[str, str] | None,
    *,
    limit: int,
    id_field: str | None = None,
    backfill: bool = False,
) -> list[dict[str, Any]]:
    """Order ``rows`` by how many query tokens each contains.

    Rows containing at least one token come first, by distinct tokens
    matched, then total hits, then input order. With ``backfill`` the window
    is topped up with the remaining rows, newest first by ``id_field``, so
    a vague query still returns a sample of the library. Every returned row
    carries ``_similarity``: the fraction of distinct query tokens it matched
    (``0.0`` for backfilled rows).
    """
    if limit <= 0:
        return []
    references = {
        field: text for field, text in (references or {}).items() if str(text).strip()
    }
    if not references and not backfill:
        return []
    total_tokens = len(
        {token for text in references.values() for token in query_tokens(text)},
    )

    scored: list[tuple[int, int, int, dict[str, Any]]] = []
    rest: list[dict[str, Any]] = []
    for order, row in enumerate(rows):
        matched, hits = text_match(row, references) if references else (0, 0)
        if matched:
            scored.append((matched, hits, order, row))
        else:
            rest.append(row)
    scored.sort(key=lambda item: (-item[0], -item[1], item[2]))

    ranked: list[dict[str, Any]] = []
    for matched, _hits, _order, row in scored:
        row[SIMILARITY_FIELD] = matched / total_tokens
        ranked.append(row)
    if backfill and len(ranked) < limit:
        for row in newest_first(rest, id_field):
            row[SIMILARITY_FIELD] = 0.0
            ranked.append(row)
    return ranked[:limit]


def newest_first(
    rows: Sequence[dict[str, Any]],
    id_field: str | None,
) -> list[dict[str, Any]]:
    """``rows`` by descending integer ``id_field``, rows without one last,
    ties in input order: the tail a search backfills its window with."""

    def key(item: tuple[int, dict[str, Any]]) -> tuple[bool, int, int]:
        index, row = item
        value = row.get(id_field) if id_field else None
        return (value is None, -value if isinstance(value, int) else 0, index)

    return [row for _index, row in sorted(enumerate(rows), key=key)]


__all__ = [
    "SIMILARITY_FIELD",
    "newest_first",
    "query_tokens",
    "rank_by_text",
    "text_match",
]
