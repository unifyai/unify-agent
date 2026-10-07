"""``UNIFY_NOTE_INDEX``: the notes written for the closest earlier requests, with their functions attached.

Offline (research artifact memory-a-v1/note-index-v1; AppWorld, 278 returns
of three Overhauled cells), ranking the guidance notes by how close the new
request is to the request each note was written for, and attaching the
functions linked to the top three, made the job's stored function available
on 79.8% of returns, against 34.9% for the shipped card ranking (+44.9
points, job-clustered interval [33.3, 58.0]), with about three functions
attached. Ranked by the notes' own text the gain was inconclusive (+9.7
[-0.4, 20.1]), and it came from functions with no note of their own.

With the switch on, for a top-level task:

* **Index.** Each guidance note in scope is keyed by its origin requests:
  the "Written for" copies :mod:`unify.function_manager.task_origin` keeps
  (``origin_requests``). A note with none is not ranked. A stored function
  no note links gets an entry of its own, keyed by the function's own
  origin requests and shown by the first line of its docstring. That
  stand-in is built at read time from the function's metadata; nothing is
  written to the store.
* **Ranking.** One embedding call
  (:func:`unify.common.embeddings.embed`, cached on this machine) over the
  request and every origin request; an entry scores the best cosine over
  its origin requests. The top :data:`K_NOTES` are shown; a tie goes to the
  newest entry (the higher store id). Nothing is ranked, filtered or bound
  by shared words or identifiers, and an embedding failure falls back to
  nothing.
* **Showing.** One section in the first message: each note's title and
  content (clipped), and the functions it links with their signatures and
  first docstring lines, each with the evidence the listing switches
  already keep (its record under ``UNIFY_ENTRY_RECORD``, else the
  ``UNIFY_LISTING_*`` notes; its recorded cases under
  ``UNIFY_FUNCTION_CASES``). Nothing new is computed for it.
* **Attachment.** The linked functions are bound in the sandbox as a
  library read binds one (:meth:`unify.actor.core_surface.FunctionLibrary._bind_names`,
  the mechanism of ``UNIFY_CORE_BIND_LISTED``). The harness never calls
  them; the section says they may be called and forces nothing.

An embedding failure or an empty index leaves no section and binds nothing,
with a warning; so does any other failure, which never reaches the task.
Entries are keyed by the requests they were written for, never by a task.

Off: no embedding call, no section, nothing bound.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

K_NOTES = 3
"""Notes shown, at most."""
MAX_FUNCTIONS = 10
"""Linked functions shown and bound per note, at most."""
CONTENT_CHARS = 600
"""A note's content is clipped to this many characters."""
_LINE_CHARS = 140

HEADER = "## Notes From Earlier Requests\n\n"
INTRO = (
    "Stored notes whose requests (what each was written for) are the closest "
    "to this one, with the functions each links. Close requests are no proof "
    "that the same job is asked, and nothing here was run for this request: "
    "use what fits."
)
CALL_FORM = (
    " The functions marked (loaded) are defined in this session: call one "
    "directly, `name(...)`, if it fits (await one marked async)."
)
STAND_IN = "a stored function's own note (no note was written for it)"

#: ``embed(texts) -> unit vectors``, one row per text.
Embed = Callable[[Sequence[str]], Any]
#: ``bind(names) -> {name: is_async}`` for the names it bound.
Binder = Callable[[List[str]], Dict[str, bool]]


def enabled() -> bool:
    from unify.settings import SETTINGS

    return bool(getattr(SETTINGS, "UNIFY_NOTE_INDEX", False))


def require_prerequisites() -> None:
    """Refuse ``UNIFY_NOTE_INDEX`` without the requests it ranks notes by."""
    from unify.function_manager import task_origin

    if not enabled():
        return
    if not task_origin.enabled():
        raise ValueError(
            "UNIFY_NOTE_INDEX needs UNIFY_TASK_ORIGIN=1 (or UNIFY_TRY_FIRST=1): "
            "it ranks notes by the requests they were written for.",
        )
    if not task_origin.guidance_recorded():
        raise ValueError(
            "UNIFY_NOTE_INDEX needs guidance origins recorded "
            "(UNIFY_GUIDANCE_ORIGIN=1 or UNIFY_ENTRY_RECORD=1): without them no "
            "note could be ranked.",
        )


# ── the index ────────────────────────────────────────────────────────────


@dataclass
class Note:
    """One index entry: a guidance note, or a stand-in for a function no note links."""

    title: str
    content: str
    origins: List[str]
    functions: List[str] = field(default_factory=list)
    row: Optional[Dict[str, Any]] = None  # the guidance row; None for a stand-in
    newest: int = 0

    @property
    def stand_in(self) -> bool:
        return self.row is None


def _first_line(text: Any, limit: int = _LINE_CHARS) -> str:
    for line in str(text or "").splitlines():
        line = " ".join(line.split())
        if line:
            return line if len(line) <= limit else line[: limit - 1].rstrip() + "…"
    return ""


def _clip(text: Any, limit: int = CONTENT_CHARS) -> str:
    text = str(text or "").strip()
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _origins(row: Dict[str, Any]) -> List[str]:
    from unify.function_manager import task_origin

    return list(dict.fromkeys(t for t in task_origin._origins(row)[1] if t.strip()))


def build_index(
    functions: Sequence[Dict[str, Any]],
    notes: Sequence[Dict[str, Any]],
) -> List[Note]:
    """Every rankable entry: each note with an origin request, then a stand-in per unlinked function with one.

    A note's linked functions are its ``function_ids`` that name a function
    in *functions*, in id order.
    """
    by_id = {
        int(row["function_id"]): str(row.get("name"))
        for row in functions
        if row.get("function_id") is not None and row.get("name")
    }
    index: List[Note] = []
    linked: set = set()
    for row in sorted(notes, key=lambda r: int(r.get("guidance_id") or 0)):
        ids = sorted({int(i) for i in (row.get("function_ids") or [])})
        names = [by_id[i] for i in ids if i in by_id]
        linked.update(names)
        origins = _origins(row)
        if not origins:
            continue
        index.append(
            Note(
                title=_first_line(row.get("title")),
                content=_clip(row.get("content")),
                origins=origins,
                functions=names,
                row=row,
                newest=int(row.get("guidance_id") or 0),
            ),
        )
    for row in sorted(functions, key=lambda r: int(r.get("function_id") or 0)):
        name = str(row.get("name") or "")
        origins = _origins(row)
        if not name or name in linked or not origins:
            continue
        index.append(
            Note(
                title=_first_line(row.get("docstring")),
                content="",
                origins=origins,
                functions=[name],
                newest=int(row.get("function_id") or 0),
            ),
        )
    return index


def rank(
    index: Sequence[Note],
    request: str,
    *,
    embed: Embed,
    k: int = K_NOTES,
) -> List[Tuple[Note, float]]:
    """The *k* entries whose closest origin request is closest to *request*, best first (one embedding call).

    The score is the embedding cosine alone; ties go to the newest entry.
    """
    import numpy as np

    if not index or k <= 0:
        return []
    texts = list(dict.fromkeys([request, *(t for n in index for t in n.origins)]))
    vectors = np.asarray(embed(texts), dtype=np.float32)
    at = {text: i for i, text in enumerate(texts)}
    q = vectors[at[request]]
    scored = [
        (max(float(np.dot(q, vectors[at[t]])) for t in note.origins), note)
        for note in index
    ]
    scored.sort(key=lambda item: (-item[0], -item[1].newest))
    return [(note, score) for score, note in scored[:k]]


# ── the evidence already kept ────────────────────────────────────────────


def _evidence(
    shown: Sequence[Dict[str, Any]],
    notes: Sequence[Dict[str, Any]],
    library: Sequence[Dict[str, Any]],
    guidance: Sequence[Dict[str, Any]],
) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """``(lines by function name, lines by guidance id)`` from the switches that keep them; empty when all are off."""
    from unify.function_manager import entry_record, store_cases, task_origin

    by_fn: Dict[str, List[str]] = {str(r.get("name")): [] for r in shown}
    by_note: Dict[str, List[str]] = {str(r.get("guidance_id")): [] for r in notes}
    recording = entry_record.enabled()
    listing = task_origin.listing_notes_enabled()
    if recording or listing:
        marker = task_origin.Marker([*library, *guidance])
        uses = (
            entry_record.uses_of(
                [(entry_record.FUNCTION, name) for name in by_fn]
                + [(entry_record.GUIDANCE, gid) for gid in by_note],
            )
            if recording
            else {}
        )
        for kind, rows, out in (
            (entry_record.FUNCTION, shown, by_fn),
            (entry_record.GUIDANCE, notes, by_note),
        ):
            for row in rows:
                ident = entry_record.ident_of(kind, row)
                if recording:
                    text = entry_record.record_text(
                        marker,
                        kind,
                        row,
                        uses.get((kind, ident)),
                    )
                    out[ident].append(f"record: {text}")
                else:
                    out[ident] += [
                        note
                        for note in task_origin.listing_notes(
                            marker,
                            kind,
                            row,
                        ).values()
                        if note
                    ]
    if store_cases.enabled():
        rows = store_cases.with_summaries([dict(row) for row in shown])
        for row in rows:
            if row.get("cases"):
                by_fn[str(row.get("name"))].append(f"cases: {row['cases']}")
    return by_fn, by_note


# ── text ─────────────────────────────────────────────────────────────────


def _signature(row: Dict[str, Any]) -> str:
    argspec = str(row.get("argspec") or "").strip()
    return f"{row.get('name')}{argspec if argspec.startswith('(') else '(' + argspec + ')'}"


def render(
    selected: Sequence[Tuple[Note, float]],
    functions: Dict[str, Dict[str, Any]],
    *,
    bound: Optional[Dict[str, bool]] = None,
    fn_lines: Optional[Dict[str, List[str]]] = None,
    note_lines: Optional[Dict[str, List[str]]] = None,
) -> str:
    """The section's text; ``""`` when nothing is selected."""
    if not selected:
        return ""
    bound = bound or {}
    fn_lines = fn_lines or {}
    note_lines = note_lines or {}
    blocks = [INTRO + (CALL_FORM if bound else "")]
    shown: set = set()
    for note, _score in selected:
        if note.stand_in:
            lines = [f"### {STAND_IN}: {note.title or note.functions[0]}"]
        else:
            gid = str(note.row.get("guidance_id"))
            lines = [f"### Note {gid}: {note.title}"]
            if note.content:
                lines.append(note.content)
            lines += note_lines.get(gid, [])
        names = note.functions[:MAX_FUNCTIONS]
        if names and not note.stand_in:
            lines.append("Functions it links:")
        for name in names:
            row = functions[name]
            mark = ""
            if name in bound:
                mark = " (loaded, async)" if bound[name] else " (loaded)"
            line = f"- `{_signature(row)}`{mark}"
            if name in shown:
                lines.append(line + ": shown above")
                continue
            shown.add(name)
            summary = _first_line(row.get("docstring"))
            if summary and not note.stand_in:
                line += f": {summary}"
            lines.append(line)
            lines += [f"  {text}" for text in fn_lines.get(name, [])]
        more = len(note.functions) - len(names)
        if more > 0:
            lines.append(
                f"- and {more} more linked function{'s' if more != 1 else ''}",
            )
        blocks.append("\n".join(lines))
    return HEADER + "\n\n".join(blocks) + "\n"


# ── the section for a task start ─────────────────────────────────────────


def section(
    function_manager: Any,
    guidance_manager: Any,
    request: Optional[str],
    *,
    functions: bool = True,
    guidance: bool = True,
    bind: Optional[Binder] = None,
    embed: Optional[Embed] = None,
) -> str:
    """The section for the current top-level task's first message, binding its functions with *bind*; ``""`` when off or nothing qualifies.

    *request* is the request as its origin is recorded
    (:func:`unify.function_manager.task_origin.current_request`).
    """
    if not enabled() or not request:
        return ""
    try:
        return _section(
            function_manager,
            guidance_manager,
            request,
            functions=functions,
            guidance=guidance,
            bind=bind,
            embed=embed,
        )
    except Exception as exc:  # noqa: BLE001 - an aid; the task starts without it
        logger.warning(f"note index left out: {type(exc).__name__}: {exc}")
        return ""


def _rows(manager: Any, wanted: bool) -> List[Dict[str, Any]]:
    rows = getattr(manager, "_evidence_rows", None) if wanted else None
    return list(rows()) if callable(rows) else []


def _section(
    function_manager: Any,
    guidance_manager: Any,
    request: str,
    *,
    functions: bool,
    guidance: bool,
    bind: Optional[Binder],
    embed: Optional[Embed],
) -> str:
    library = _rows(function_manager, functions)
    notes_in = _rows(guidance_manager, guidance)
    index = build_index(library, notes_in)
    if not index:
        logger.warning("note index left out: no note or function has a request")
        return ""
    if embed is None:
        from unify.common import embeddings

        embed = embeddings.embed
    try:
        selected = rank(index, request, embed=embed)
    except Exception as exc:  # noqa: BLE001 - no ranking, no section
        logger.warning(
            f"note index left out: embedding failed: {type(exc).__name__}: {exc}",
        )
        return ""
    by_name = {str(row.get("name")): row for row in library}
    names = list(
        dict.fromkeys(
            name for note, _ in selected for name in note.functions[:MAX_FUNCTIONS]
        ),
    )
    fn_lines, note_lines = _evidence(
        [by_name[name] for name in names],
        [note.row for note, _ in selected if note.row is not None],
        library,
        notes_in,
    )
    bound: Dict[str, bool] = {}
    if bind is not None and names:
        try:
            bound = dict(bind(names) or {})
        except Exception as exc:  # noqa: BLE001 - the section stands without it
            logger.warning(
                f"note index: could not load the linked functions: "
                f"{type(exc).__name__}: {exc}",
            )
    return render(
        selected,
        by_name,
        bound=bound,
        fn_lines=fn_lines,
        note_lines=note_lines,
    )


def binder(actor: Any, sandbox: Any, core_session: Any = None) -> Optional[Binder]:
    """What binds the attached functions in *sandbox*, as a library read binds one; ``None`` without a function library.

    Under the core surface, the session's ``functions`` object; otherwise
    the same read on the actor's function manager, as the JSON library
    tools bind what they return.
    """
    from unify.actor import core_surface

    if core_session is not None:
        library = core_session.objects.get(core_surface.FUNCTIONS)
    elif getattr(actor, "function_manager", None) is not None:
        library = core_surface.FunctionLibrary(actor, core_surface.WritePolicy())
    else:
        library = None
    if library is None:
        return None
    return lambda names: library._bind_names(names, sandbox=sandbox)


__all__ = [
    "CALL_FORM",
    "HEADER",
    "INTRO",
    "K_NOTES",
    "MAX_FUNCTIONS",
    "Note",
    "STAND_IN",
    "binder",
    "build_index",
    "enabled",
    "rank",
    "render",
    "require_prerequisites",
    "section",
]
