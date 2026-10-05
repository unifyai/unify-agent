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
functions do not apply, and guidance, which records no request (unless
``UNIFY_GUIDANCE_ORIGIN``, below), is not listed. Each line shows the score and the call count as evidence; a task
whose request resembles none of the recorded ones gets no list.

``UNIFY_GUIDANCE_ORIGIN`` lets guidance entries that recorded the request
they were written for join the gated list: scored the same way, sharing the
same places, each line labelled ``guidance <id>`` with its title, first
content line and score.

``UNIFY_ORIGIN_PROVENANCE`` ends a marked function's line, in either list,
with why it is marked, in parentheses: the identifiers its origin request
shares with this one, or that it was this same request, and whether the
checker accepted that session's answer when that was recorded
(:meth:`~unify.function_manager.task_origin.Marker.provenance`).

``UNIFY_SHORTLIST_LIFT`` ranks the embedding list by lift (similarity to this
request less the mean similarity to recent requests,
:mod:`unify.actor.shortlist_lift`) once the stream has enough history, under
its own header.

``UNIFY_LISTING_PROVENANCE`` follows every listed entry, in either list, with
an ``origin:`` line (:meth:`~unify.function_manager.task_origin.Marker.origin_line`);
``UNIFY_LESSON_STATUS`` lists a guidance entry from a session whose answer was
not accepted, or whose outcome is unknown, as unverified and without its
first content line; ``UNIFY_LISTING_USAGE`` adds a function's call count and
how the sessions of its last calls ended
(:func:`~unify.function_manager.task_origin.listing_notes`).

``UNIFY_SHORTLIST_RELATED`` follows the gated list with at most two
"possibly related" entries ranked by the meaning of their "use this when"
statements, under a header of their own, never bound
(:mod:`unify.actor.related_shortlist`).

``UNIFY_EVIDENCE_LIST`` replaces both lists with the evidence list
(:mod:`unify.actor.evidence_list`).

``UNIFY_CORE_BIND_LISTED`` (core tool surface): the caller passes *bind*,
which binds the listed functions in the sandbox as a read would and says
which are ``async def``; either header then says how to call a listed
function (:data:`CALL_FORM`), and an async one's line says ``(async)``.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Sequence

from unify.function_manager import task_origin

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
# UNIFY_GUIDANCE_ORIGIN
_GATED_HEADER_WITH_GUIDANCE = (
    "Stored functions and guidance saved while handling requests similar to "
    "this one (similar_request: overlap of the two requests' words, 1 is the "
    "same request; used: times called). Read or call any of them if useful:"
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
_GATED_HEADER_CALL = (
    "Stored functions saved while handling requests similar to this one "
    "(similar_request: overlap of the two requests' words, 1 is the same "
    "request; used: times called). Read or call any of them if useful; "
    f"{CALL_FORM}:"
)
# UNIFY_GUIDANCE_ORIGIN with UNIFY_CORE_BIND_LISTED: guidance listed too.
_GATED_HEADER_WITH_GUIDANCE_CALL = (
    "Stored functions and guidance saved while handling requests similar to "
    "this one (similar_request: overlap of the two requests' words, 1 is the "
    "same request; used: times called). Read or call any of them if useful; "
    f"{CALL_FORM}:"
)
# UNIFY_SHORTLIST_LIFT: the list ranked by lift.
_LIFT_HEADER = (
    "Library entries that match this request more closely than recent "
    "requests, closest first (read or call any of them if useful):"
)
_LIFT_HEADER_CALL = (
    "Library entries that match this request more closely than recent "
    f"requests, closest first (read or call any of them if useful; {CALL_FORM}):"
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
    if row.get("similar_request") is not None:
        line += f" [similar_request {row['similar_request']}]"
    # UNIFY_LISTING_USAGE
    usage = row.get(task_origin.USAGE)
    if usage:
        line += f" [{usage}]"
    return line + _origin_suffix(row) + _origin_line(row)


def _origin_suffix(row: Dict[str, Any]) -> str:
    """``UNIFY_ORIGIN_PROVENANCE``: `` (<why>)`` when the row says why it is marked.

    Left out when the row has an ``origin:`` line (``UNIFY_LISTING_PROVENANCE``),
    which says the same and more.
    """
    why = row.get("origin")
    if task_origin.ORIGIN_LINE in row:
        return ""
    return f" ({why})" if isinstance(why, str) and why else ""


def _origin_line(row: Dict[str, Any]) -> str:
    """``UNIFY_LISTING_PROVENANCE``: the entry's ``origin:`` line, indented under it."""
    text = row.get(task_origin.ORIGIN_LINE)
    return f"\n  origin: {text}" if text else ""


def _gated_function_line(row: Dict[str, Any], is_async: bool = False) -> str:
    argspec = str(row.get("argspec") or "").strip()
    signature = f"{row.get('name')}{argspec if argspec.startswith('(') else '(' + argspec + ')'}"
    line = f"- function `{signature}`" + (_ASYNC_MARK if is_async else "")
    summary = _first_line(row.get("docstring"))
    if summary:
        line += f": {summary}"
    score = float(row.get("similar_request") or 0.0)
    calls = int(row.get("usage_calls") or 0)
    # UNIFY_LISTING_USAGE: the call count with how its sessions ended.
    usage = row.get(task_origin.USAGE) or f"used {calls}×"
    return (
        line
        + f" [similar_request {score:.2f} · {usage}]"
        + _origin_suffix(row)
        + _origin_line(row)
    )


def _gated_guidance_line(row: Dict[str, Any]) -> str:
    """``UNIFY_GUIDANCE_ORIGIN``: a guidance entry in the gated list."""
    score = float(row.get("similar_request") or 0.0)
    line = _guidance_line(row)
    head, sep, tail = line.partition("\n")
    return head + f" [similar_request {score:.2f}]" + sep + tail


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
    # UNIFY_LESSON_STATUS: an unverified lesson's first line is not shown.
    status = row.get(task_origin.LESSON)
    if status:
        return line + f" ({status})" + _origin_line(row)
    summary = _first_line(row.get("content"))
    if summary:
        line += f": {summary}"
    return line + _origin_line(row)


def _lift_rows(
    function_manager: Any,
    guidance_manager: Any,
    text: str,
    chosen: Any,
    *,
    k: int,
    functions: bool,
    guidance: bool,
) -> Optional[List[tuple[str, Dict[str, Any]]]]:
    """``UNIFY_SHORTLIST_LIFT``: every candidate ranked by lift; ``None`` to rank as shipped."""
    from unify.actor import shortlist_lift

    pool: List[tuple[str, Dict[str, Any]]] = []
    if functions and function_manager is not None:
        ranked = getattr(function_manager, "_shortlist_rows", None)
        if callable(ranked):
            pool += [("function", row) for row in ranked(text, k, pool=True)]
    if guidance and guidance_manager is not None:
        ranked = getattr(guidance_manager, "_shortlist_rows", None)
        if callable(ranked):
            pool += [("guidance", row) for row in ranked(text, 1 << 30)]
    return shortlist_lift.rank(pool, text, chosen, k_list=k)


def _with_notes(
    rows: List[tuple[str, Dict[str, Any]]],
    function_manager: Any,
    guidance_manager: Any,
) -> List[tuple[str, Dict[str, Any]]]:
    """The listing switches' notes on each row (``UNIFY_LISTING_PROVENANCE`` and its siblings).

    Scored as the gated list scores: over the stored functions and the
    guidance entries with a recorded request.
    """
    library = getattr(function_manager, "_library_rows", None)
    origins = getattr(guidance_manager, "_origin_rows", None)
    functions = list(library()) if callable(library) else []
    guidance = list(origins()) if callable(origins) else []
    marker = task_origin.Marker([*functions, *guidance])
    by_function = {row.get("function_id"): row for row in functions}
    by_guidance = {row.get("guidance_id"): row for row in guidance}
    out = []
    for kind, row in rows:
        if kind == "function":
            stored = by_function.get(row.get("function_id")) or {}
            source = {**stored, "name": row.get("name")}
        else:
            stored = by_guidance.get(row.get("guidance_id")) or {}
            source = {**stored, "is_builtin": row.get("is_builtin")}
        out.append((kind, {**row, **task_origin.listing_notes(marker, kind, source)}))
    return out


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
    bind: Optional[Binder] = None,
) -> Optional[str]:
    """The shortlist as first-message text, or ``None`` (nothing to list, or no ranking).

    With *gate* (``UNIFY_SHORTLIST_GATE``'s threshold) only the stored
    functions whose ``similar_request`` passes it, and no embedding.

    With *bind* (``UNIFY_CORE_BIND_LISTED``) the listed functions are bound
    by it before the text is written, and the header says how to call one.

    ``UNIFY_EVIDENCE_LIST``: the evidence list instead
    (:mod:`unify.actor.evidence_list`).
    """
    from unify.actor import evidence_list

    if evidence_list.enabled():
        return evidence_list.block(
            function_manager,
            guidance_manager,
            request_text(request),
            functions=functions,
            guidance=guidance,
            bind=bind,
            call_form=CALL_FORM,
        )
    if gate is not None:
        block = _gated_block(
            function_manager,
            gate,
            functions=functions,
            guidance_manager=guidance_manager if guidance else None,
            bind=bind,
        )
        return _with_related(
            block,
            function_manager,
            guidance_manager,
            request,
            functions=functions,
            guidance=guidance,
        )
    from unify.actor import shortlist_lift

    text = request_text(request)
    lifted = None
    try:
        # UNIFY_SHORTLIST_LIFT: ranked by lift once the stream has history.
        chosen = shortlist_lift.spec()
        if chosen is not None and text:
            lifted = _lift_rows(
                function_manager,
                guidance_manager,
                text,
                chosen,
                k=K,
                functions=functions,
                guidance=guidance,
            )
        rows = (
            lifted
            if lifted is not None
            else shortlist_rows(
                function_manager,
                guidance_manager,
                text,
                functions=functions,
                guidance=guidance,
            )
        )
        if rows and task_origin.listing_notes_enabled():
            rows = _with_notes(rows, function_manager, guidance_manager)
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
    call = bind is not None and bool(bound)
    if lifted is not None:
        header = _LIFT_HEADER_CALL if call else _LIFT_HEADER
    else:
        header = _HEADER_CALL if call else _HEADER
    return "\n".join([header, *lines])


def _with_related(
    block: Optional[str],
    function_manager: Any,
    guidance_manager: Any,
    request: Any,
    *,
    functions: bool,
    guidance: bool,
) -> Optional[str]:
    """``UNIFY_SHORTLIST_RELATED``: *block*, then the possibly related entries it did not list.

    *block* unchanged while the switch is off. The tier's functions are
    never bound (:mod:`unify.actor.related_shortlist`).
    """
    from unify.actor import related_shortlist

    if not related_shortlist.enabled():
        return block
    related = related_shortlist.related_block(
        function_manager,
        guidance_manager,
        request_text(request),
        listed=shortlisted_names(block),
        functions=functions,
        guidance=guidance,
    )
    parts = [part for part in (block, related) if part]
    return "\n\n".join(parts) if parts else None


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


def _gated_block(
    function_manager: Any,
    gate: float,
    *,
    functions: bool,
    guidance_manager: Any = None,
    bind: Optional[Binder] = None,
) -> Optional[str]:
    from unify.function_manager import task_origin

    ranked = getattr(function_manager, "_gated_shortlist_rows", None)
    # UNIFY_GUIDANCE_ORIGIN: guidance with recorded origins joins the list.
    origin_rows = getattr(guidance_manager, "_origin_rows", None)
    with_guidance = task_origin.guidance_enabled() and callable(origin_rows)
    if not callable(ranked) or not (functions or with_guidance):
        return None
    try:
        if with_guidance:
            rows = ranked(gate, K, origin_rows(), functions=functions)
        else:
            rows = ranked(gate, K)
    except Exception as exc:
        logger.debug(f"gated shortlist unavailable: {type(exc).__name__}: {exc}")
        return None
    if not rows:
        return None
    bound = _bind(bind, [row for row in rows if row.get("kind") != "guidance"])
    lines = [
        (
            _gated_guidance_line(row)
            if row.get("kind") == "guidance"
            else _gated_function_line(row, bound.get(str(row.get("name")), False))
        )
        for row in rows
    ]
    call = bind is not None and bool(bound)
    if with_guidance:
        header = (
            _GATED_HEADER_WITH_GUIDANCE_CALL if call else _GATED_HEADER_WITH_GUIDANCE
        )
    else:
        header = _GATED_HEADER_CALL if call else _GATED_HEADER
    return "\n".join([header, *lines])


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
    "gate_rows",
    "request_text",
    "require_gate_prerequisites",
    "shortlist_block",
    "shortlist_rows",
    "shortlisted_names",
]
