"""``UNIFY_LIBRARY_SHORTLIST``: the library entries closest to a task, listed once in its first message.

At the start of each ``act()`` (a sub-agent's task included) the harness ranks
the stored functions and guidance entries in scope against the request text
by embedding similarity (the ranking the library searches use; no model call)
and lists the closest :data:`K` of them, functions and guidance together, one
line each: a
function's name, signature and the first line of its docstring, a guidance
entry's id, title and the first line of its content, and ``similar_request``
where ``UNIFY_TASK_ORIGIN`` or ``UNIFY_TRY_FIRST`` marks it. The list opens
the first user message with the session's other first-message context, so
every later request shares it as a prefix; it is never repeated or updated
during the task. It
asks nothing of the model: reading an entry, calling it, or searching the
libraries is the model's choice.

Nothing here counts as a search hit: the shortlist is the harness's ranking,
not the model's retrieval, so it leaves the functions' standing unchanged.
Primitives are left out (they are platform surface, documented elsewhere), as
are entries with nothing to compare and functions the activation ranking
drops as lapsed. A ranking that fails (no embeddings) gives no list.

``UNIFY_SHORTLIST_GATE=similar_request:<t>`` lists by request instead: only
stored functions recorded under a request (``UNIFY_TASK_ORIGIN``) whose
``similar_request`` to the current one is at least *t*, ranked by that score,
then by how often each was called, then newest first (:func:`gate_rows`).
Nothing is embedded, the activation ranking and its hiding of lapsed
functions do not apply, and guidance, which records no request, is not
listed. Each line shows the score and the call count as evidence; a task
whose request resembles none of the recorded ones gets no list.

``UNIFY_ORIGIN_PROVENANCE`` ends a marked function's line, in either list,
with why it is marked, in parentheses: the identifiers its origin request
shares with this one, or that it was this same request, and whether the
checker accepted that session's answer when that was recorded
(:meth:`~unify.function_manager.task_origin.Marker.provenance`).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

K = 5
"""At most this many entries, functions and guidance together."""

_SUMMARY_CHARS = 140
_HEADER = (
    "Library entries closest to this request, ranked by similarity "
    "(read or call any of them if useful):"
)
# UNIFY_SHORTLIST_GATE
_GATED_HEADER = (
    "Stored functions saved while handling requests similar to this one "
    "(similar_request: overlap of the two requests' words, 1 is the same "
    "request; used: times called). Read or call any of them if useful:"
)


def request_text(request: Any) -> str:
    """The text of *request* the entries are ranked against."""
    if isinstance(request, str):
        return request.strip()
    if isinstance(request, dict):
        content = request.get("content")
        return content.strip() if isinstance(content, str) else ""
    if isinstance(request, list):
        return "\n\n".join(
            part for part in (request_text(item) for item in request) if part
        )
    return ""


def _first_line(text: Any) -> str:
    for line in str(text or "").splitlines():
        line = " ".join(line.split())
        if line:
            return (
                line
                if len(line) <= _SUMMARY_CHARS
                else line[: _SUMMARY_CHARS - 1].rstrip() + "…"
            )
    return ""


def _function_line(row: Dict[str, Any]) -> str:
    argspec = str(row.get("argspec") or "").strip()
    signature = f"{row.get('name')}{argspec if argspec.startswith('(') else '(' + argspec + ')'}"
    line = f"- function `{signature}`"
    summary = _first_line(row.get("docstring"))
    if summary:
        line += f": {summary}"
    if row.get("similar_request") is not None:
        line += f" [similar_request {row['similar_request']}]"
    return line + _origin_suffix(row)


def _origin_suffix(row: Dict[str, Any]) -> str:
    """``UNIFY_ORIGIN_PROVENANCE``: `` (<why>)`` when the row says why it is marked."""
    why = row.get("origin")
    return f" ({why})" if isinstance(why, str) and why else ""


def _gated_function_line(row: Dict[str, Any]) -> str:
    argspec = str(row.get("argspec") or "").strip()
    signature = f"{row.get('name')}{argspec if argspec.startswith('(') else '(' + argspec + ')'}"
    line = f"- function `{signature}`"
    summary = _first_line(row.get("docstring"))
    if summary:
        line += f": {summary}"
    score = float(row.get("similar_request") or 0.0)
    calls = int(row.get("usage_calls") or 0)
    return (
        line + f" [similar_request {score:.2f} · used {calls}×]" + _origin_suffix(row)
    )


def gate_rows(
    rows: Sequence[Dict[str, Any]],
    marker: Any,
    threshold: float,
    *,
    k: int = K,
) -> List[Dict[str, Any]]:
    """``UNIFY_SHORTLIST_GATE``: the at most *k* rows *marker* scores at least *threshold*.

    *marker* is a :class:`~unify.function_manager.task_origin.Marker`; a row
    it cannot score (no recorded request, or no current one) is never kept.
    Ranked by score, then call count (``usage_calls``), then newest (the
    larger id: the store's insertion order, which no benchmark clock moves).
    Each kept row is a copy carrying its ``similar_request`` score.
    """
    kept: List[tuple[float, int, int, Dict[str, Any]]] = []
    for row in rows:
        score = marker.score(row)
        if score is None or score < threshold:
            continue
        newest = int(row.get("function_id") or row.get("guidance_id") or 0)
        kept.append((score, int(row.get("usage_calls") or 0), newest, row))
    kept.sort(key=lambda item: (-item[0], -item[1], -item[2]))
    return [
        {**row, "similar_request": round(score, 2)}
        for score, _, _, row in kept[: max(k, 0)]
    ]


def require_gate_prerequisites() -> None:
    """Refuse ``UNIFY_SHORTLIST_GATE`` without the switches it reads.

    Raises :class:`ValueError` naming the missing switch: without the
    shortlist there is nothing to gate, and without request records nothing
    could ever pass the gate.
    """
    from unify.function_manager import task_origin
    from unify.settings import SETTINGS

    if SETTINGS.shortlist_gate_threshold() is None:
        return
    if not SETTINGS.UNIFY_LIBRARY_SHORTLIST:
        raise ValueError(
            "UNIFY_SHORTLIST_GATE needs UNIFY_LIBRARY_SHORTLIST=1: it decides "
            "which entries the shortlist lists.",
        )
    if not task_origin.enabled():
        raise ValueError(
            "UNIFY_SHORTLIST_GATE needs UNIFY_TASK_ORIGIN=1 (or UNIFY_TRY_FIRST=1): "
            "without the requests stored functions were recorded under, no "
            "function could pass the gate.",
        )


def _guidance_line(row: Dict[str, Any]) -> str:
    line = f"- guidance {row.get('guidance_id')} `{_first_line(row.get('title'))}`"
    summary = _first_line(row.get("content"))
    if summary:
        line += f": {summary}"
    return line


def shortlist_rows(
    function_manager: Any,
    guidance_manager: Any,
    text: str,
    *,
    k: int = K,
    functions: bool = True,
    guidance: bool = True,
) -> List[tuple[str, Dict[str, Any]]]:
    """``[(kind, row), ...]``: the closest *k* entries, most similar first."""
    if not text:
        return []
    found: List[tuple[float, int, str, Dict[str, Any]]] = []
    if functions and function_manager is not None:
        ranked = getattr(function_manager, "_shortlist_rows", None)
        if callable(ranked):
            for row in ranked(text, k):
                found.append(
                    (float(row.get("_similarity") or 0.0), len(found), "function", row),
                )
    if guidance and guidance_manager is not None:
        ranked = getattr(guidance_manager, "_shortlist_rows", None)
        if callable(ranked):
            for row in ranked(text, k):
                found.append(
                    (float(row.get("_similarity") or 0.0), len(found), "guidance", row),
                )
    found = [f for f in found if f[0] > 0]
    found.sort(key=lambda f: (-f[0], f[1]))
    return [(kind, row) for _, _, kind, row in found[:k]]


def shortlist_block(
    function_manager: Any,
    guidance_manager: Any,
    request: Any,
    *,
    functions: bool = True,
    guidance: bool = True,
    gate: Optional[float] = None,
) -> Optional[str]:
    """The shortlist as first-message text, or ``None`` (nothing to list, or no ranking).

    With *gate* (``UNIFY_SHORTLIST_GATE``'s threshold) only the stored
    functions whose ``similar_request`` passes it, and no embedding.
    """
    if gate is not None:
        return _gated_block(function_manager, gate, functions=functions)
    try:
        rows = shortlist_rows(
            function_manager,
            guidance_manager,
            request_text(request),
            functions=functions,
            guidance=guidance,
        )
    except Exception as exc:
        logger.debug(f"library shortlist unavailable: {type(exc).__name__}: {exc}")
        return None
    if not rows:
        return None
    lines = [
        _function_line(row) if kind == "function" else _guidance_line(row)
        for kind, row in rows
    ]
    return "\n".join([_HEADER, *lines])


def _gated_block(
    function_manager: Any,
    gate: float,
    *,
    functions: bool,
) -> Optional[str]:
    ranked = getattr(function_manager, "_gated_shortlist_rows", None)
    if not functions or not callable(ranked):
        return None
    try:
        rows = ranked(gate, K)
    except Exception as exc:
        logger.debug(f"gated shortlist unavailable: {type(exc).__name__}: {exc}")
        return None
    if not rows:
        return None
    return "\n".join([_GATED_HEADER, *(_gated_function_line(row) for row in rows)])


def shortlisted_names(block: Optional[str]) -> Dict[str, List[str]]:
    """The function names and guidance ids a block lists (for analysis and tests)."""
    out: Dict[str, List[str]] = {"functions": [], "guidance": []}
    for line in (block or "").splitlines():
        if line.startswith("- function `"):
            out["functions"].append(line[len("- function `") :].split("(", 1)[0])
        elif line.startswith("- guidance "):
            out["guidance"].append(line[len("- guidance ") :].split(" ", 1)[0])
    return out


__all__ = [
    "K",
    "gate_rows",
    "request_text",
    "require_gate_prerequisites",
    "shortlist_block",
    "shortlist_rows",
    "shortlisted_names",
]
