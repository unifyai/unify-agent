"""``UNIFY_SHORTLIST_RELATED=statement:<k>``: entries possibly related to a task, listed apart from the gated ones.

The request-gated shortlist (``UNIFY_SHORTLIST_GATE``) lists an entry only
when the new request shares the rare words of a request the entry was
stored for. The retrieval study SEMANTIC-V2 (5 Oct) found that this is
what makes it reliable on benchmark streams, which repeat their own
templates, and also what makes it miss the same task asked again in other
words: with the stored entry and stream unchanged and only the task sentence
reworded, it found the entry for 10-13% of requests (88-96% as logged), and
for 3% of day-to-day requests in other words. No similarity threshold,
lexical or semantic, told a reworded repeat from a near-miss (similar words,
another intent), so this module asserts no match: it lists, under its own
header and after the gated list, at most *k* (1 or 2) further entries whose
"use this when" statement is closest in meaning to the request, each with
its statement and the request it was first stored for, and leaves the
judgement of intent to the model.

* **Statements.** The storage review that adds or updates an entry writes a
  one-sentence statement of the kind of request it serves
  (:data:`REVIEW_SECTION`, :func:`record_statements`). The harness keeps a
  statement only for an entry stored or updated under the current request
  and only when it names nothing that identifies this task instance (the
  store's instance lint, :mod:`unify.function_manager.instance_lint`). A
  function keeps it in its ``metadata`` (``use_when``), a guidance entry
  with a recorded origin (``UNIFY_GUIDANCE_ORIGIN``) in that origin. An
  entry without one (a library that predates the switch, an entry stored
  outside a review) gets a template from its name and the first line of its
  docstring or title, labelled as such in its line.
* **At task start.** The request is reduced to its *distinct* lines: a line
  that at least half of the earlier logged requests contain is dropped (the
  stream's standing instructions), as in the retrieval audit; with fewer than
  two earlier requests the request is used whole. The log keeps a hash of
  each line of every request for this. One call of
  :func:`unify.common.embeddings.embed` embeds it together with every
  candidate's statement; vectors are cached by text hash, so a statement is
  embedded once (the review's store embeds it then) and only the request is
  new. Candidates are the stored functions in scope, and guidance entries
  with a recorded origin, that the gated list did not list. They are ranked
  by cosine; those under the embedder's floor (a minimal one, there only to
  keep entries resembling nothing out) are dropped. No score is shown.
* **What it does not do.** It never lists anything in a sub-agent, never
  binds a listed function under ``UNIFY_TOOL_SURFACE=core`` (unlike the
  gated list's under ``UNIFY_CORE_BIND_LISTED``: the model reads one before
  using it), never counts a search hit and asks nothing of the model. A
  ranking that fails (no embedder) gives no list.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Callable, Collection, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

STATEMENT = "use_when"
"""The metadata (or guidance origin) key a written statement is kept under."""

MAX_K = 2
MAX_STATEMENT_CHARS = 300
_ORIGIN_CHARS = 200
_TEMPLATE_DOC_CHARS = 160
# A line at least this share of the earlier requests contain is the stream's.
_SHARED_SHARE = 0.5
_MIN_EARLIER = 2

# The floor a candidate's cosine must reach, per embedder (its cache label),
# calibrated offline on the SEMANTIC-V2 task starts with template
# statements (research artifact overhaul-lanes/tier2-related-v1). Local
# bge-small: the smallest floor listing anything for at most 5% of requests
# that need no stored entry. text-embedding-3-small, whose cosines sit lower
# and closer together: the floor that best trades recall on every kind of
# repeat against lists where nothing fits, chosen leave-one-cluster-out
# over all the data (at the 5% rule's 0.385 it listed something on 51% of
# task starts and the original entry for 63% of near-misses; at 0.45, 29%
# and 14%, keeping 62% of reworded AppWorld repeats against 10% from the
# gate alone).
DEFAULT_FLOORS = {
    "BAAI/bge-small-en-v1.5": 0.68,
    "openai/text-embedding-3-small": 0.45,
}

HEADER = (
    "Possibly related (judge whether the intent matches; these were stored "
    "for different wording, and are not loaded until read):"
)
TEMPLATE_LABEL = "(statement from its name and docstring)"

REVIEW_SECTION = (
    "## Use-when statements\n\n"
    "For each function or guidance entry you add or update in this review, "
    "also write one line in your final reply: `use_when <function name>: "
    "Use this when ...` or `use_when guidance <id>: Use this when ...`. One "
    "sentence on the kind of request the entry serves, in general terms, "
    "without names, ids, numbers or quoted values from this request. A later "
    "session reads it beside the entry to judge whether the entry fits its "
    "own request.\n\n"
)

_LINE = re.compile(
    r"^[ \t>*\-]*`?use_when\s+(?:guidance\s+#?(\d+)|`?([A-Za-z_][A-Za-z0-9_]*)`?)"
    r"\s*:\s*(.+?)\s*$",
    re.IGNORECASE | re.MULTILINE,
)
_SENTENCE_END = re.compile(r"[.!?](?=\s|$)")
_WS = re.compile(r"\s+")
_WORDS = re.compile(r"[_\W]+")

#: ``embed(texts) -> unit vectors``, one row per text.
Embed = Callable[[Sequence[str]], Any]


# ── the switch ───────────────────────────────────────────────────────────


def setting() -> Optional[Tuple[int, Optional[float]]]:
    """``(k, floor)`` of ``UNIFY_SHORTLIST_RELATED``; ``None`` when off."""
    from unify.settings import SETTINGS

    return SETTINGS.shortlist_related()


def enabled() -> bool:
    return setting() is not None


def statements_enabled() -> bool:
    """Whether reviews write, and the harness keeps, "use this when" statements.

    ``UNIFY_SHORTLIST_RELATED``, or ``UNIFY_EVIDENCE_LIST``, whose possibly
    related tier ranks by them.
    """
    from unify.settings import SETTINGS

    return enabled() or SETTINGS.evidence_list() is not None


def require_prerequisites() -> None:
    """Refuse ``UNIFY_SHORTLIST_RELATED`` without the gated shortlist it extends."""
    from unify.settings import SETTINGS

    if setting() is None:
        return
    if SETTINGS.shortlist_gate_threshold() is None:
        raise ValueError(
            "UNIFY_SHORTLIST_RELATED needs UNIFY_SHORTLIST_GATE (with "
            "UNIFY_LIBRARY_SHORTLIST and UNIFY_TASK_ORIGIN): it lists what the "
            "gated shortlist did not.",
        )


def floor_for(model: str, configured: Optional[float]) -> float:
    """The cosine floor: *configured*, else the default of the embedder labelled *model*."""
    if configured is not None:
        return configured
    for prefix, floor in DEFAULT_FLOORS.items():
        if model.startswith(prefix):
            return floor
    return max(DEFAULT_FLOORS.values())


# ── texts ────────────────────────────────────────────────────────────────


def _norm(line: str) -> str:
    return _WS.sub(" ", line).strip()


def distinct_text(
    request: str,
    earlier: Sequence[Collection[str]],
) -> Tuple[str, List[str]]:
    """``(distinct, shared)``: *request* without the lines the stream shares, and those lines.

    *earlier* holds, per earlier logged request, the hashes of its lines
    (:func:`~unify.function_manager.task_origin.line_keys`). A line is shared
    when at least half of them contain it. With fewer than two earlier
    requests, or when every line is shared, the whole request.
    """
    from unify.function_manager import task_origin

    lines = [line.strip() for line in str(request or "").splitlines()]
    lines = [line for line in lines if line]
    if len(earlier) < _MIN_EARLIER:
        return str(request or "").strip(), []
    keep: List[str] = []
    shared: List[str] = []
    for line in lines:
        (key,) = task_origin.line_keys(line)
        df = sum(1 for keys in earlier if key in keys)
        (shared if df >= _SHARED_SHARE * len(earlier) else keep).append(line)
    if not keep:
        return str(request or "").strip(), []
    return "\n".join(keep), shared


def _bounded(text: str, limit: int) -> str:
    text = _norm(text)
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _first_line(text: Any) -> str:
    for line in str(text or "").splitlines():
        if line.strip():
            return _norm(line)
    return ""


def template_statement(kind: str, row: Dict[str, Any]) -> str:
    """The statement of an entry nobody wrote one for: its name (or title) and first line."""
    if kind == "guidance":
        title = _bounded(_first_line(row.get("title")), _TEMPLATE_DOC_CHARS)
        first = _bounded(_first_line(row.get("content")), _TEMPLATE_DOC_CHARS)
        text = f"Use this when: {title.rstrip('.')}."
    else:
        words = _WORDS.sub(" ", str(row.get("name") or "")).strip()
        first = _bounded(_first_line(row.get("docstring")), _TEMPLATE_DOC_CHARS)
        text = f"Use this when you need to {words}."
    return f"{text} {first}".strip() if first else text


def _written(row: Dict[str, Any]) -> Optional[str]:
    metadata = row.get("metadata")
    if not isinstance(metadata, dict):
        return None
    value = metadata.get(STATEMENT)
    return value.strip() if isinstance(value, str) and value.strip() else None


def statement_of(kind: str, row: Dict[str, Any]) -> Tuple[str, bool]:
    """``(statement, written)``: the review's statement, else the template (``written`` False)."""
    written = _written(row)
    if written:
        return written, True
    return template_statement(kind, row), False


def _origin_excerpt(row: Dict[str, Any], shared: Sequence[str]) -> str:
    """The latest request the entry was stored for, without the stream's shared lines, bounded."""
    from unify.function_manager import task_origin

    metadata = row.get("metadata")
    texts = (
        metadata.get(task_origin.REQUESTS_FIELD) if isinstance(metadata, dict) else None
    )
    texts = [t for t in (texts or []) if isinstance(t, str) and t.strip()]
    if not texts:
        return ""
    origin = _norm(texts[-1])
    stripped = origin
    for line in sorted({_norm(s) for s in shared}, key=len, reverse=True):
        if line:
            stripped = stripped.replace(line, " ")
    stripped = _norm(stripped)
    return _bounded(stripped or origin, _ORIGIN_CHARS)


# ── ranking ──────────────────────────────────────────────────────────────


def _key(kind: str, row: Dict[str, Any]) -> Tuple[str, str]:
    if kind == "guidance":
        return kind, str(row.get("guidance_id"))
    return kind, str(row.get("name"))


def related_rows(
    candidates: Sequence[Tuple[str, Dict[str, Any]]],
    request: str,
    earlier: Sequence[Collection[str]],
    *,
    k: int,
    floor: float,
    exclude: Sequence[Tuple[str, str]] = (),
    embed: Optional[Embed] = None,
) -> List[Dict[str, Any]]:
    """At most *k* of *candidates* closest to *request* by statement, none under *floor*.

    *candidates* are ``(kind, row)`` pairs, kind ``"function"`` or
    ``"guidance"``; *exclude* holds ``(kind, name or id)`` keys already
    listed. *earlier* are the earlier requests of the stream (for the
    distinct lines). One call of *embed* (default
    :func:`unify.common.embeddings.embed`) embeds the distinct request and
    every statement. Each kept row is a copy with ``kind``, ``statement``,
    ``written``, ``origin_excerpt`` and ``_cosine``, best first (ties:
    newest first).
    """
    import numpy as np

    if k <= 0 or not request:
        return []
    skip = set(exclude)
    pool = [(kind, row) for kind, row in candidates if _key(kind, row) not in skip]
    if not pool:
        return []
    query, shared = distinct_text(request, earlier)
    statements = [statement_of(kind, row) for kind, row in pool]
    texts = list(dict.fromkeys([query, *(text for text, _ in statements)]))
    if embed is None:
        from unify.common import embeddings

        embed = embeddings.embed
    vectors = np.asarray(embed(texts), dtype=np.float32)
    index = {text: i for i, text in enumerate(texts)}
    q = vectors[index[query]]
    scored = []
    for (kind, row), (text, written) in zip(pool, statements):
        cosine = float(np.dot(q, vectors[index[text]]))
        if cosine < floor:
            continue
        newest = int(row.get("function_id") or row.get("guidance_id") or 0)
        scored.append((cosine, newest, kind, row, text, written))
    scored.sort(key=lambda item: (-item[0], -item[1]))
    return [
        {
            **row,
            "kind": kind,
            "statement": text,
            "written": written,
            "origin_excerpt": _origin_excerpt(row, shared),
            "_cosine": round(cosine, 4),
        }
        for cosine, _, kind, row, text, written in scored[:k]
    ]


def _line(row: Dict[str, Any]) -> str:
    statement = _bounded(row["statement"], MAX_STATEMENT_CHARS)
    if not row.get("written"):
        statement += f" {TEMPLATE_LABEL}"
    if row["kind"] == "guidance":
        title = _bounded(_first_line(row.get("title")), 140)
        line = f"- guidance {row.get('guidance_id')} `{title}`: {statement}"
    else:
        argspec = str(row.get("argspec") or "").strip()
        if not argspec.startswith("("):
            argspec = f"({argspec})"
        line = f"- function `{row.get('name')}{argspec}`: {statement}"
    origin = row.get("origin_excerpt")
    if origin:
        line += f' · stored for: "{origin}"'
    return line


def render(rows: Sequence[Dict[str, Any]]) -> Optional[str]:
    """The tier's text: :data:`HEADER` and one line per row, or ``None`` with no rows."""
    if not rows:
        return None
    return "\n".join([HEADER, *(_line(row) for row in rows)])


def related_names(block: Optional[str]) -> Dict[str, List[str]]:
    """The function names and guidance ids the tier lists in *block* (for analysis and tests)."""
    from unify.actor.library_shortlist import shortlisted_names

    text = block or ""
    if HEADER not in text:
        return {"functions": [], "guidance": []}
    tier = text[text.index(HEADER) :].split("\n\n", 1)[0]
    return shortlisted_names(tier)


def related_block(
    function_manager: Any,
    guidance_manager: Any,
    request_text: str,
    *,
    listed: Dict[str, List[str]],
    functions: bool = True,
    guidance: bool = True,
    embed: Optional[Embed] = None,
) -> Optional[str]:
    """The tier for the current top-level request, or ``None`` (off, nothing close, or no ranking).

    *listed* is what the gated list showed
    (:func:`~unify.actor.library_shortlist.shortlisted_names`).
    """
    from unify.common import embeddings
    from unify.function_manager import task_origin

    chosen = setting()
    if chosen is None or not request_text:
        return None
    k, configured = chosen
    try:
        candidates: List[Tuple[str, Dict[str, Any]]] = []
        rows = getattr(function_manager, "_related_candidates", None)
        if functions and callable(rows):
            candidates += [("function", row) for row in rows()]
        origin_rows = getattr(guidance_manager, "_origin_rows", None)
        if guidance and task_origin.guidance_enabled() and callable(origin_rows):
            candidates += [("guidance", row) for row in origin_rows()]
        if not candidates:
            return None
        earlier = task_origin.logged_request_lines()
        exclude = [("function", name) for name in listed.get("functions", [])]
        exclude += [("guidance", gid) for gid in listed.get("guidance", [])]
        kept = related_rows(
            candidates,
            request_text,
            earlier,
            k=k,
            floor=floor_for(embeddings.embedder().model, configured),
            exclude=exclude,
            embed=embed,
        )
    except Exception as exc:  # an aid; the task starts without it
        logger.debug(f"related shortlist unavailable: {type(exc).__name__}: {exc}")
        return None
    return render(kept)


# ── statements written by the storage review ─────────────────────────────


def clean_statement(text: Any, request: Any = None) -> Optional[str]:
    """*text* as a kept statement, or ``None``.

    Whitespace collapsed, quotes and backticks around it dropped, cut at the
    first sentence end within :data:`MAX_STATEMENT_CHARS` (else bounded).
    ``None`` when empty or when it names an instance token of *request*
    (default: the current request) or a task-alias/UUID shape.
    """
    from unify.function_manager import instance_lint, task_origin

    value = _norm(str(text or "")).strip("`\"'“”‘’ ")
    if not value:
        return None
    end = _SENTENCE_END.search(value)
    if end is not None and end.end() <= MAX_STATEMENT_CHARS:
        value = value[: end.end()]
    value = _bounded(value, MAX_STATEMENT_CHARS)
    request = task_origin.current_request() if request is None else request
    tokens = instance_lint.tokens_of(request or "")
    if instance_lint._found_in_text(value, tokens) is not None:
        return None
    return value


def parse_statements(text: Any) -> Dict[Tuple[str, str], str]:
    """``{(kind, name or id): statement}`` from the ``use_when`` lines of *text* (the last per entry)."""
    found: Dict[Tuple[str, str], str] = {}
    for match in _LINE.finditer(str(text or "")):
        gid, name, statement = match.groups()
        key = ("guidance", gid) if gid else ("function", name)
        found[key] = statement.strip().rstrip("`").strip()
    return found


def record_statements(
    text: Any,
    function_manager: Any,
    guidance_manager: Any,
    *,
    embed: Optional[Embed] = None,
) -> Dict[Tuple[str, str], str]:
    """Keep the statements the review's reply *text* states; ``{key: outcome}``.

    An outcome is ``"kept"``, ``"instance"`` (it names this task instance),
    ``"empty"`` or ``"not stored under this request"`` (no such entry, or
    the entry was not added or updated while handling the current request).
    The kept statements are embedded once, so the next task start finds
    them in the cache. Nothing happens while the switch is off.
    """
    if not statements_enabled():
        return {}
    outcomes: Dict[Tuple[str, str], str] = {}
    kept: List[str] = []
    for key, raw in parse_statements(text).items():
        kind, ident = key
        statement = clean_statement(raw)
        if statement is None:
            outcomes[key] = "instance" if _norm(raw) else "empty"
            continue
        owner = function_manager if kind == "function" else guidance_manager
        setter = getattr(owner, "_set_use_when", None)
        try:
            stored = callable(setter) and bool(setter(ident, statement))
        except Exception as exc:  # an aid; never fails the review
            logger.warning(f"statement not kept: {type(exc).__name__}: {exc}")
            stored = False
        outcomes[key] = "kept" if stored else "not stored under this request"
        if stored:
            kept.append(statement)
    if kept:
        try:
            if embed is None:
                from unify.common import embeddings

                embed = embeddings.embed
            embed(list(dict.fromkeys(kept)))
        except Exception as exc:  # the next task start embeds it instead
            logger.debug(f"statement not embedded: {type(exc).__name__}: {exc}")
    if outcomes:
        logger.info(f"use-when statements: {outcomes}")
    return outcomes


__all__ = [
    "DEFAULT_FLOORS",
    "HEADER",
    "MAX_K",
    "REVIEW_SECTION",
    "STATEMENT",
    "TEMPLATE_LABEL",
    "clean_statement",
    "distinct_text",
    "enabled",
    "floor_for",
    "parse_statements",
    "record_statements",
    "related_block",
    "related_names",
    "related_rows",
    "render",
    "require_prerequisites",
    "setting",
    "statement_of",
    "statements_enabled",
    "template_statement",
]
