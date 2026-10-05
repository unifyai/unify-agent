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

``UNIFY_CORE_BIND_LISTED`` (core tool surface): the caller passes *bind*,
which binds the listed functions in the sandbox as a read would and says
which are ``async def``; either header then says how to call a listed
function (:data:`CALL_FORM`), and an async one's line says ``(async)``.
"""

from __future__ import annotations

import ast
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
_ASYNC_MARK = " (async)"

#: ``bind(names) -> {name: is_async}`` for the names it bound.
Binder = Callable[[List[str]], Dict[str, bool]]

# UNIFY_SHORTLIST_CALLABLE_FIRST
_HEADER_CALLABLE_FIRST = (
    "Library entries closest to this request, ranked by similarity, functions "
    "first (use any of them if useful):"
)
_LOADED = "already loaded; call directly"


def callable_first() -> bool:
    """Whether ``UNIFY_SHORTLIST_CALLABLE_FIRST`` is on."""
    from unify.settings import SETTINGS

    return bool(getattr(SETTINGS, "UNIFY_SHORTLIST_CALLABLE_FIRST", False))


def call_arguments(argspec: Any) -> Optional[str]:
    """The arguments of a call of a function with *argspec*, by name.

    ``(grid, color=0, *rest, scale: int, **extra)`` gives ``grid=...,
    color=0, *rest, scale=...``: a parameter without a default by name, one
    with a default as its default, ``*args`` as written, ``**kwargs`` left
    out. ``None`` when *argspec* does not parse.
    """
    text = str(argspec or "").strip()
    if not text.startswith("("):
        text = f"({text})"
    try:
        tree = ast.parse(f"def _f{text}: pass")
        args = tree.body[0].args  # type: ignore[attr-defined]
    except (SyntaxError, AttributeError, IndexError):
        return None
    out: List[str] = []
    positional = [*args.posonlyargs, *args.args]
    defaults = [None] * (len(positional) - len(args.defaults)) + list(args.defaults)
    for arg, default in zip(positional, defaults):
        if arg.arg in ("self", "cls"):
            continue
        value = "..." if default is None else ast.unparse(default)
        out.append(
            value if arg in args.posonlyargs else f"{arg.arg}={value}",
        )
    if args.vararg is not None:
        out.append(f"*{args.vararg.arg}")
    for arg, default in zip(args.kwonlyargs, args.kw_defaults):
        out.append(f"{arg.arg}={'...' if default is None else ast.unparse(default)}")
    return ", ".join(out)


def _call_form(row: Dict[str, Any], is_async: bool, surface: str, loaded: bool) -> str:
    """How to call the stored function *row*: by name where it is loaded."""
    name = str(row.get("name"))
    arguments = call_arguments(row.get("argspec"))
    if arguments is None:
        arguments = "..."
    if loaded:
        return ("await " if is_async else "") + f"{name}({arguments})"
    if surface == "json":
        mapping = ", ".join(
            f'"{part.split("=", 1)[0]}": {part.split("=", 1)[1]}'
            for part in arguments.split(", ")
            if "=" in part
        )
        return f'execute_function("{name}", {{{mapping}}})'
    sep = ", " if arguments else ""
    return f'await functions.run("{name}"{sep}{arguments})'


def _callable_function_line(
    row: Dict[str, Any],
    is_async: bool,
    *,
    surface: str,
    loaded: bool,
    gated: bool = False,
) -> str:
    """``UNIFY_SHORTLIST_CALLABLE_FIRST``: a function's line, led by its call."""
    line = f"- function `{_call_form(row, is_async, surface, loaded)}`"
    if loaded:
        line += f" ({_LOADED})"
    summary = _first_line(row.get("docstring"))
    if summary:
        line += f": {summary}"
    if gated:
        score = float(row.get("similar_request") or 0.0)
        calls = int(row.get("usage_calls") or 0)
        line += f" [similar_request {score:.2f} · used {calls}×]"
    elif row.get("similar_request") is not None:
        line += f" [similar_request {row['similar_request']}]"
    return line + _origin_suffix(row)


def _linked_guidance_line(row: Dict[str, Any], linked: Sequence[str]) -> str:
    """A guidance entry's line naming the stored functions it links."""
    line = f"- guidance {row.get('guidance_id')} `{_first_line(row.get('title'))}`"
    if linked:
        line += " (for " + ", ".join(f"`{n}`" for n in linked) + ")"
    summary = _first_line(row.get("content"))
    if summary:
        line += f": {summary}"
    return line


def render_callable_first(
    rows: Sequence[tuple],
    bound: Dict[str, bool],
    *,
    surface: str,
    links: Optional[Dict[int, List[str]]] = None,
    gated: bool = False,
    header: Optional[str] = None,
) -> str:
    """``UNIFY_SHORTLIST_CALLABLE_FIRST``: *rows* (``[(kind, row), ...]``,
    ranked) as a list that leads with the functions.

    *bound* is ``{name: is_async}`` for the listed functions the harness
    loaded into the session; *surface* is ``core`` or ``json`` (how to call
    a function that is not loaded); *links* maps a guidance id to the names
    of the functions it links (a guidance row may carry them as ``linked``).
    """
    links = links or {}
    functions = [row for kind, row in rows if kind == "function"]
    guidance = [row for kind, row in rows if kind != "function"]
    lines = [header or _HEADER_CALLABLE_FIRST]
    for row in functions:
        name = str(row.get("name"))
        lines.append(
            _callable_function_line(
                row,
                bound.get(name, False),
                surface=surface,
                loaded=name in bound,
                gated=gated,
            ),
        )
    for row in guidance:
        gid = row.get("guidance_id")
        linked = row.get("linked")
        if linked is None:
            linked = links.get(int(gid), []) if gid is not None else []
        line = _linked_guidance_line(row, linked)
        if gated and row.get("similar_request") is not None:
            line += f" [similar_request {float(row['similar_request']):.2f}]"
        lines.append(line)
    return "\n".join(lines)


def _guidance_links(
    guidance_manager: Any,
    rows: Sequence[Dict[str, Any]],
) -> Dict[int, List[str]]:
    lookup = getattr(guidance_manager, "_linked_function_names", None)
    ids = [int(r["guidance_id"]) for r in rows if r.get("guidance_id") is not None]
    if not callable(lookup) or not ids:
        return {}
    try:
        return dict(lookup(ids) or {})
    except Exception as exc:  # noqa: BLE001 - the list stands without the links
        logger.debug(f"guidance links unavailable: {type(exc).__name__}: {exc}")
        return {}


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
    return line + _origin_suffix(row)


def _origin_suffix(row: Dict[str, Any]) -> str:
    """``UNIFY_ORIGIN_PROVENANCE``: `` (<why>)`` when the row says why it is marked."""
    why = row.get("origin")
    return f" ({why})" if isinstance(why, str) and why else ""


def _gated_function_line(row: Dict[str, Any], is_async: bool = False) -> str:
    argspec = str(row.get("argspec") or "").strip()
    signature = f"{row.get('name')}{argspec if argspec.startswith('(') else '(' + argspec + ')'}"
    line = f"- function `{signature}`" + (_ASYNC_MARK if is_async else "")
    summary = _first_line(row.get("docstring"))
    if summary:
        line += f": {summary}"
    score = float(row.get("similar_request") or 0.0)
    calls = int(row.get("usage_calls") or 0)
    return (
        line + f" [similar_request {score:.2f} · used {calls}×]" + _origin_suffix(row)
    )


def _gated_guidance_line(row: Dict[str, Any]) -> str:
    """``UNIFY_GUIDANCE_ORIGIN``: a guidance entry in the gated list."""
    score = float(row.get("similar_request") or 0.0)
    return _guidance_line(row) + f" [similar_request {score:.2f}]"


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
    bind: Optional[Binder] = None,
    surface: str = "json",
) -> Optional[str]:
    """The shortlist as first-message text, or ``None`` (nothing to list, or no ranking).

    With *gate* (``UNIFY_SHORTLIST_GATE``'s threshold) only the stored
    functions whose ``similar_request`` passes it, and no embedding.

    With *bind* (``UNIFY_CORE_BIND_LISTED``) the listed functions are bound
    by it before the text is written, and the header says how to call one.
    """
    if gate is not None:
        return _gated_block(
            function_manager,
            gate,
            functions=functions,
            guidance_manager=guidance_manager if guidance else None,
            bind=bind,
            surface=surface,
        )
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
    if callable_first():
        return render_callable_first(
            rows,
            bound,
            surface=surface,
            links=_guidance_links(
                guidance_manager,
                [row for kind, row in rows if kind != "function"],
            ),
        )
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


def _gated_block(
    function_manager: Any,
    gate: float,
    *,
    functions: bool,
    guidance_manager: Any = None,
    bind: Optional[Binder] = None,
    surface: str = "json",
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
    if callable_first():
        header = (
            _GATED_HEADER_WITH_GUIDANCE if with_guidance else _GATED_HEADER
        ).replace(" Read or call any of them if useful:", " Functions first:")
        kinds = [
            ("guidance" if row.get("kind") == "guidance" else "function", row)
            for row in rows
        ]
        return render_callable_first(
            kinds,
            bound,
            surface=surface,
            links=_guidance_links(
                guidance_manager,
                [row for kind, row in kinds if kind == "guidance"],
            ),
            gated=True,
            header=header,
        )
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
            call = line[len("- function `") :].removeprefix("await ")
            call = call.removeprefix('execute_function("').removeprefix(
                'functions.run("',
            )
            out["functions"].append(call.split("(", 1)[0].split('"', 1)[0])
        elif line.startswith("- guidance "):
            out["guidance"].append(line[len("- guidance ") :].split(" ", 1)[0])
    return out


__all__ = [
    "CALL_FORM",
    "K",
    "call_arguments",
    "callable_first",
    "gate_rows",
    "render_callable_first",
    "request_text",
    "require_gate_prerequisites",
    "shortlist_block",
    "shortlist_rows",
    "shortlisted_names",
]
