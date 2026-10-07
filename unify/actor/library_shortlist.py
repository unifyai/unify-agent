"""``UNIFY_LIBRARY_SHORTLIST``: the library entries closest to a task, listed once in its first message.

At the start of each ``act()`` (a sub-agent's task included) the harness ranks
the stored functions and guidance entries in scope against the request text
by embedding similarity (the ranking the library searches use; no model call)
and lists the closest :data:`K` of them, functions and guidance together, one
line each: a
function's name, signature and the first line of its docstring, a guidance
entry's id, title and the first line of its content. The list opens
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

``UNIFY_CORE_BIND_LISTED`` (core tool surface): the caller passes *bind*,
which binds the listed functions in the sandbox as a read would and says
which are ``async def``; either header then says how to call a listed
function (:data:`CALL_FORM`), and an async one's line says ``(async)``.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

K = 5
"""At most this many entries, functions and guidance together."""

_SUMMARY_CHARS = 140
_HEADER = (
    "Library entries closest to this request, ranked by similarity "
    "(read or call any of them if useful):"
)
# UNIFY_CORE_BIND_LISTED: how to call a listed function, which is bound.
CALL_FORM = (
    "listed functions are loaded, so call one directly: `name(...)`, or "
    '`await functions.run("name", arg=...)`'
)
_HEADER_CALL = (
    "Library entries closest to this request, ranked by similarity "
    f"(read or call any of them if useful; {CALL_FORM}):"
)
_ASYNC_MARK = " (async)"

#: ``bind(names) -> {name: is_async}`` for the names it bound.
Binder = Callable[[List[str]], Dict[str, bool]]


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


def _function_line(row: Dict[str, Any], is_async: bool = False) -> str:
    argspec = str(row.get("argspec") or "").strip()
    signature = f"{row.get('name')}{argspec if argspec.startswith('(') else '(' + argspec + ')'}"
    line = f"- function `{signature}`" + (_ASYNC_MARK if is_async else "")
    summary = _first_line(row.get("docstring"))
    if summary:
        line += f": {summary}"
    return line


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
    bind: Optional[Binder] = None,
) -> Optional[str]:
    """The shortlist as first-message text, or ``None`` (nothing to list, or no ranking).

    With *bind* (``UNIFY_CORE_BIND_LISTED``) the listed functions are bound
    by it before the text is written, and the header says how to call one.
    """
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
    bound = _bind(bind, [row for kind, row in rows if kind == "function"])
    lines = [
        (
            _function_line(row, bound.get(str(row.get("name")), False))
            if kind == "function"
            else _guidance_line(row)
        )
        for kind, row in rows
    ]
    header = _HEADER_CALL if bind is not None and bound else _HEADER
    return "\n".join([header, *lines])


def _bind(bind: Optional[Binder], rows: Sequence[Dict[str, Any]]) -> Dict[str, bool]:
    """``UNIFY_CORE_BIND_LISTED``: ``{name: is_async}`` for the listed functions *bind* bound."""
    names = [str(row.get("name")) for row in rows if row.get("name")]
    if bind is None or not names:
        return {}
    try:
        return dict(bind(names) or {})
    except Exception as exc:  # noqa: BLE001 - the list stands without it
        logger.warning(
            "could not load the shortlisted functions: %s: %s",
            type(exc).__name__,
            exc,
        )
        return {}


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
    "CALL_FORM",
    "K",
    "request_text",
    "shortlist_block",
    "shortlist_rows",
    "shortlisted_names",
]
