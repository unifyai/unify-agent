"""``UNIFY_REVIEW_SHORTLIST``: show the storage review the library entries closest to its session, in full.

Offline (research artifact memory-a-v1/review-shape-v1: 656 reviews, 4,211
calls), the storage review spent 47% of its cost looking the library up --
searching (21%) and reading entries (25%) -- before it wrote; on ARC 52%.
Most of those lookups land on a few entries: the ones the session used and
the ones closest to its request. This puts them in the review's prompt, in
full: the stored functions the session called, then the functions and
guidance entries whose cards are closest to the session's request (the
evidence list's card embedding, at most :data:`K_FUNCTIONS` and
:data:`K_NOTES`), each with its signature, description, the request it was
stored for and its whole source or content (clipped).

The review keeps every tool it had: the section says the library may hold
other entries. Whether it then searches less is what a screen measures; the
section only informs. An embedding failure leaves only the used entries; any
other failure leaves no section. Off: no section, no embedding call.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Sequence, Tuple

logger = logging.getLogger(__name__)

K_FUNCTIONS = 5
K_NOTES = 5
MAX_SOURCE = 3000
MAX_NOTE = 1500
MAX_ORIGIN = 300

HEADER = (
    "## Library entries close to this session\n\n"
    "The stored entries this session used, then those whose cards are closest "
    "to its request, shown in full so they need not be looked up again. The "
    "library may hold other entries.\n\n"
)


def enabled() -> bool:
    from unify.settings import SETTINGS

    return bool(getattr(SETTINGS, "UNIFY_REVIEW_SHORTLIST", False))


def _clip(text: Any, limit: int) -> str:
    text = str(text or "").strip()
    return text if len(text) <= limit else text[:limit] + "\n… [cut]"


def _request(trajectory: Sequence[Dict[str, Any]]) -> str:
    from unify.function_manager import task_origin

    current = task_origin.current_request()
    if current:
        return current
    for message in trajectory:
        if message.get("role") == "user":
            content = message.get("content")
            return (
                content
                if isinstance(content, str)
                else json.dumps(content, default=str)
            )
    return ""


def _used(names: Sequence[str], trajectory: Sequence[Dict[str, Any]]) -> List[str]:
    """The stored functions the session called: named as a call anywhere in its trajectory."""
    text = json.dumps(list(trajectory), default=str)
    return [
        n for n in names if re.search(r"(?<![\w.])" + re.escape(n) + r"\s*\(", text)
    ]


def _function_block(row: Dict[str, Any]) -> str:
    from unify.function_manager import task_origin

    name = row.get("name")
    doc = str(row.get("docstring") or "").strip().splitlines()
    origins = task_origin._origins(row)[1] if hasattr(task_origin, "_origins") else []
    lines = [f"### function `{name}{row.get('argspec') or '()'}`"]
    if doc:
        lines.append(doc[0])
    if origins:
        lines.append("Stored for: " + _clip(origins[-1], MAX_ORIGIN))
    lines.append("```python\n" + _clip(row.get("implementation"), MAX_SOURCE) + "\n```")
    return "\n".join(lines)


def _note_block(row: Dict[str, Any]) -> str:
    return (
        f"### guidance #{row.get('guidance_id')}: {row.get('title') or ''}\n"
        + _clip(row.get("content"), MAX_NOTE)
    )


def section(
    function_manager: Any,
    guidance_manager: Any,
    trajectory: Sequence[Dict[str, Any]],
    *,
    embed: Any = None,
) -> str:
    """The section for this review, or "" (off, an empty library, or any failure)."""
    if not enabled() or function_manager is None:
        return ""
    try:
        from unify.actor import evidence_list
        from unify.common import embeddings

        fn_rows = getattr(function_manager, "_evidence_rows", None)
        note_rows = getattr(guidance_manager, "_evidence_rows", None)
        functions = [
            r
            for r in (fn_rows() if callable(fn_rows) else [])
            if not r.get("is_primitive")
        ]
        notes = list(note_rows() if callable(note_rows) else [])
        if not functions and not notes:
            return ""
        by_name = {str(r.get("name")): r for r in functions}
        used = _used(list(by_name), trajectory)[:K_FUNCTIONS]
        picked_f: List[str] = list(used)
        picked_g: List[Any] = []
        scores: Dict[Tuple[str, str], float] = {}
        request = _request(trajectory)
        if request:
            try:
                lib = evidence_list.build_library(functions, notes, [])
                pool = list(lib.entries)
                scores = evidence_list.JudgeMatcher().related_scores(
                    lib,
                    pool,
                    request,
                    embed=embed or embeddings.embed,
                )
            except Exception as exc:  # only the used entries, then
                logger.warning(
                    f"review shortlist not ranked: {type(exc).__name__}: {exc}",
                )
        for (kind, ident), _ in sorted(scores.items(), key=lambda item: -item[1]):
            if (
                kind == "function"
                and ident not in picked_f
                and len(picked_f) < K_FUNCTIONS
            ):
                picked_f.append(ident)
            elif kind == "guidance" and len(picked_g) < K_NOTES:
                picked_g.append(ident)
        notes_by_id = {str(r.get("guidance_id")): r for r in notes}
        blocks = [_function_block(by_name[n]) for n in picked_f if n in by_name]
        blocks += [
            _note_block(notes_by_id[str(g)]) for g in picked_g if str(g) in notes_by_id
        ]
        if not blocks:
            return ""
        return HEADER + "\n\n".join(blocks) + "\n\n"
    except Exception as exc:  # an aid; never blocks the review
        logger.warning(f"review shortlist not built: {type(exc).__name__}: {exc}")
        return ""


__all__ = ["HEADER", "K_FUNCTIONS", "K_NOTES", "enabled", "section"]
