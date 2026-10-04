"""``UNIFY_LIBRARY_SHORTLIST``: the library entries closest to a task, listed once in its first message.

At the start of each ``act()`` (a sub-agent's task included) the harness ranks
the stored functions and guidance entries in scope against the request text
by embedding similarity (the ranking the library searches use; no model call)
and lists the closest :data:`K` of them, functions and guidance together, one
line each: a
function's name, signature and the first line of its docstring, a guidance
entry's id, title and the first line of its content, and ``similar_request``
where ``UNIFY_TRY_FIRST`` marks it. The list opens the first user message
with the session's other first-message context, so every later request
shares it as a prefix; it is never repeated or updated during the task. It
asks nothing of the model: reading an entry, calling it, or searching the
libraries is the model's choice.

Nothing here counts as a search hit: the shortlist is the harness's ranking,
not the model's retrieval, so it leaves the functions' standing unchanged.
Primitives are left out (they are platform surface, documented elsewhere), as
are entries with nothing to compare and functions the activation ranking
drops as lapsed. A ranking that fails (no embeddings) gives no list.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

K = 5
"""At most this many entries, functions and guidance together."""

_SUMMARY_CHARS = 140
_HEADER = (
    "Library entries closest to this request, ranked by similarity "
    "(read or call any of them if useful):"
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
) -> Optional[str]:
    """The shortlist as first-message text, or ``None`` (nothing to list, or no ranking)."""
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
    "request_text",
    "shortlist_block",
    "shortlist_rows",
    "shortlisted_names",
]
