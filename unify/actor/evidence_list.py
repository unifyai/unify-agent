"""``UNIFY_EVIDENCE_LIST``: the library shortlist as a list of evidence, not a ranking.

The retrieval design study of 5 Oct (research artifact retrieval-design-v1)
found the shipped shortlist asks one similarity score to answer two
questions: "have I done this exact job before?", a question of identity, and
"is anything in my library about this?", a question of topic. Ranked by the
embedding of the whole request, which on a stream is mostly its standing
instructions, the list measured how much an entry sounded like those
instructions: one generic note, written after a failed visit, headed every
list of the Continual-ARC paper-protocol run, entries from the same task
scored no higher than others, the list never stayed silent, and it never
said why an entry was there. This module separates the two questions, shows
the evidence behind each entry, and stays silent when nothing qualifies.

* **Cards.** Functions are procedures and guidance entries are guidelines
  for using them, linked many to many in the link table
  (:mod:`unify.function_manager.entry_links`); a note may stand alone. A
  listed entry is shown as a card with what it links: a function with its
  notes, or a note with the functions it guides. The entry that qualifies
  best heads the card; an entry an earlier card already shows is not listed
  again.
* **Seen before** (may claim a match; no embedding). Cards whose recorded
  requests -- the requests a member was stored or written for, and the
  sessions that called, read or relied on one -- include this same request,
  share a rare whole identifier with it (the rule of
  ``UNIFY_SIMILAR_REQUEST_IDENTIFIERS``), or reach ``similar_request`` of the
  gate threshold (``UNIFY_SHORTLIST_GATE``, else :data:`DEFAULT_THRESHOLD`)
  with the stream's logged requests as the corpus. Ranked by that order,
  then score, then accepted uses, then a function before a note (a
  procedure is shown with its guidelines), then newest; at most :data:`K`
  cards.
* **Possibly related** (claims nothing). At most *k* cards not seen before,
  by the cosine of the request's distinct lines (those fewer than half of
  the earlier logged requests contain) with each card's "use this when"
  statement (:mod:`unify.actor.related_shortlist`), at or above the floor;
  one embedding call per task start.
* **Standing cards, one line.** A card the floor also passes for most of the
  last :data:`STANDING_WINDOW` logged requests of other jobs (requests not
  seen-before matches of this one) matches whatever comes and says little
  about this request. It is not listed as possibly related; its id and
  title are named once, on one line. Computed from the vectors those
  requests' own task starts cached: no further embedding.
* **Evidence, never hiding.** Each card says why it is listed, how that
  session ended, its status (verified or not, and why) and its use, from
  :mod:`unify.function_manager.entry_record`. A note's first line is shown
  with its status.
* **Silence.** Nothing seen before, nothing above the floor and no standing
  card: no list.

It runs for a top-level task only (a sub-agent's request is its caller's).
Under ``UNIFY_CORE_BIND_LISTED`` only the seen-before functions are bound.
"""

from __future__ import annotations

import logging
import sqlite3
from contextlib import closing
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

K = 5
DEFAULT_THRESHOLD = 0.175
STANDING_WINDOW = 8
STANDING_MIN_EARLIER = 4
STANDING_SHARE = 0.5
MAX_STANDING_NAMED = 4
_LINE_CHARS = 140
_STANDING_TITLE_CHARS = 48
TABLE = "evidence_queries"

SEEN_HEADER = (
    "Seen before: entries recorded while handling a request like this one "
    "(read or call any of them if useful):"
)
RELATED_HEADER = (
    "Possibly related (no match is claimed; judge whether the intent is the same):"
)
STANDING_HEAD = (
    "Standing entries (general entries that match most requests, so they say "
    "little about this one; read any if useful): "
)

#: ``embed(texts) -> unit vectors``, one row per text.
Embed = Callable[[Sequence[str]], Any]


def setting() -> Optional[Tuple[int, Optional[float]]]:
    """``(k, floor)`` of ``UNIFY_EVIDENCE_LIST``; ``None`` when off."""
    from unify.settings import SETTINGS

    return SETTINGS.evidence_list()


def enabled() -> bool:
    return setting() is not None


def require_prerequisites() -> None:
    """Refuse ``UNIFY_EVIDENCE_LIST`` without what it reads, or with a ranking it replaces."""
    from unify.function_manager import entry_record, task_origin
    from unify.settings import SETTINGS

    if setting() is None:
        return
    if not SETTINGS.UNIFY_LIBRARY_SHORTLIST:
        raise ValueError(
            "UNIFY_EVIDENCE_LIST needs UNIFY_LIBRARY_SHORTLIST=1: it is the "
            "shortlist's form.",
        )
    if not task_origin.enabled():
        raise ValueError(
            "UNIFY_EVIDENCE_LIST needs UNIFY_TASK_ORIGIN=1 (or UNIFY_TRY_FIRST=1): "
            "it lists entries by the requests they were recorded under.",
        )
    if not entry_record.enabled():
        raise ValueError(
            "UNIFY_EVIDENCE_LIST needs UNIFY_ENTRY_RECORD=1: every card shows "
            "the entry's record.",
        )
    if SETTINGS.shortlist_lift() is not None:
        raise ValueError(
            "UNIFY_EVIDENCE_LIST replaces the ranked shortlist; turn "
            "UNIFY_SHORTLIST_LIFT off.",
        )


def threshold() -> float:
    from unify.settings import SETTINGS

    gate = SETTINGS.shortlist_gate_threshold()
    return gate if gate is not None else DEFAULT_THRESHOLD


# ── cards ────────────────────────────────────────────────────────────────


Key = Tuple[str, str]


@dataclass
class Library:
    """The entries in scope and their links (many to many, from the link table)."""

    entries: Dict[Key, Dict[str, Any]] = field(default_factory=dict)
    linked: Dict[Key, List[Key]] = field(default_factory=dict)

    def kind_rows(self) -> List[Tuple[str, Dict[str, Any]]]:
        return [(key[0], row) for key, row in self.entries.items()]


def build_library(
    functions: Sequence[Dict[str, Any]],
    notes: Sequence[Dict[str, Any]],
    links: Sequence[Tuple[int, int]],
) -> Library:
    """The entries, keyed ``("function", name)`` / ``("guidance", id)``, with each one's linked entries.

    *links* are ``(function_id, guidance_id)`` pairs; a pair naming an entry
    not in scope is left out. A note with no link stands alone.
    """
    lib = Library()
    by_fid: Dict[int, Key] = {}
    by_gid: Dict[int, Key] = {}
    for row in sorted(functions, key=lambda r: int(r.get("function_id") or 0)):
        key = ("function", str(row.get("name")))
        lib.entries[key] = row
        if row.get("function_id") is not None:
            by_fid[int(row["function_id"])] = key
    for row in sorted(notes, key=lambda r: int(r.get("guidance_id") or 0)):
        key = ("guidance", str(row.get("guidance_id")))
        lib.entries[key] = row
        by_gid[int(row["guidance_id"])] = key
    for fid, gid in sorted(set((int(f), int(g)) for f, g in links)):
        fk, gk = by_fid.get(fid), by_gid.get(gid)
        if fk is None or gk is None:
            continue
        lib.linked.setdefault(fk, []).append(gk)
        lib.linked.setdefault(gk, []).append(fk)
    return lib


@dataclass
class Card:
    """One listed entry (the head) with the entries linked to it: a function with its
    guidance notes, or a note with the functions it guides."""

    head: Tuple[str, Dict[str, Any]]
    linked: List[Tuple[str, Dict[str, Any]]] = field(default_factory=list)

    @property
    def function(self) -> Optional[Dict[str, Any]]:
        return self.head[1] if self.head[0] == "function" else None

    @property
    def notes(self) -> List[Dict[str, Any]]:
        if self.head[0] == "function":
            return [row for kind, row in self.linked if kind == "guidance"]
        return [self.head[1]]

    def members(self) -> List[Tuple[str, Dict[str, Any]]]:
        return [self.head, *self.linked]

    def primary(self) -> Tuple[str, Dict[str, Any]]:
        return self.head

    def key(self) -> Key:
        kind, row = self.head
        return kind, _ident(kind, row)

    def keys(self) -> List[Key]:
        return [(kind, _ident(kind, row)) for kind, row in self.members()]

    def newest(self) -> int:
        kind, row = self.head
        return int(row.get("function_id") or row.get("guidance_id") or 0)


def _ident(kind: str, row: Dict[str, Any]) -> str:
    return str(row.get("guidance_id") if kind == "guidance" else row.get("name"))


def card_for(lib: Library, key: Key) -> Card:
    """The card headed by entry *key*: it, then every entry linked to it."""
    return Card(
        head=(key[0], lib.entries[key]),
        linked=[(k[0], lib.entries[k]) for k in lib.linked.get(key, [])],
    )


def build_cards(
    functions: Sequence[Dict[str, Any]],
    notes: Sequence[Dict[str, Any]],
    links: Sequence[Tuple[int, int]] = (),
) -> List[Card]:
    """One card per entry (each headed by it, with its links), functions first."""
    lib = build_library(functions, notes, links)
    return [card_for(lib, key) for key in lib.entries]


def choose_cards(
    lib: Library,
    ranked: Sequence[Key],
    k: int,
    shown: Optional[set] = None,
) -> List[Card]:
    """At most *k* cards headed by *ranked* entries in order, skipping an entry an earlier card already shows."""
    shown = set() if shown is None else shown
    out: List[Card] = []
    for key in ranked:
        if len(out) >= k:
            break
        if key in shown or key not in lib.entries:
            continue
        card = card_for(lib, key)
        out.append(card)
        shown.update(card.keys())
    return out


# ── seen before ──────────────────────────────────────────────────────────


@dataclass
class Match:
    """Why an entry is seen before: rank 3 same request, 2 shared identifiers, 1 similar wording."""

    rank: int
    score: float
    kind: str
    how: str  # "origin", or the use: "call", "read", "relied"
    text: str
    shared: List[str] = field(default_factory=list)
    recurred: int = 0


def _origins(row: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    from unify.function_manager import task_origin

    return task_origin._origins(row)


def entry_matches(
    lib: Library,
    current_text: str,
    current_key: Optional[str],
    logged: Sequence[str],
    uses: Dict[Key, Any],
    *,
    threshold: float,
) -> List[Tuple[Key, Match, int]]:
    """``[(entry, match, accepted uses)]`` for every entry recorded under a request like *current_text*, best first."""
    from unify.function_manager import task_origin

    text_by_key = {task_origin.text_key(text): text for text in logged}
    current_hash = task_origin.text_key(current_text)
    origin_texts = [t for _, row in lib.kind_rows() for t in _origins(row)[1]]
    known = list(dict.fromkeys([*logged, *origin_texts]))
    weights = task_origin.token_weights([*known, current_text])
    found: List[Tuple[int, float, int, int, Key, Match]] = []
    for key, row in lib.entries.items():
        kind = key[0]
        keys, texts = _origins(row)
        candidates: List[Tuple[str, str, bool]] = [("origin", t, False) for t in texts]
        if (current_key is not None and current_key in keys) or current_text in texts:
            candidates.append(("origin", current_text, True))
        record = uses.get(key)
        accepted = 0
        if record is not None:
            accepted = sum(1 for v in record.outcomes.values() if v is True)
            for how, hashes in record.by_how.items():
                for h in hashes:
                    if h == current_hash:
                        candidates.append((how, current_text, True))
                    elif h in text_by_key:
                        candidates.append((how, text_by_key[h], False))
        best: Optional[Match] = None
        for how, text, same in candidates:
            if same or text == current_text:
                match = Match(3, 1.0, kind, how, current_text)
            else:
                shared = task_origin.shared_identifiers(current_text, text, known)
                score = task_origin.similarity(current_text, text, weights)
                if shared:
                    match = Match(2, score, kind, how, text, shared)
                elif score >= threshold:
                    match = Match(1, score, kind, how, text)
                else:
                    continue
            if best is None or (match.rank, match.score) > (best.rank, best.score):
                best = match
        if best is not None:
            newest = int(row.get("function_id") or row.get("guidance_id") or 0)
            procedure = 1 if kind == "function" else 0
            found.append(
                (best.rank, best.score, accepted, procedure, newest, key, best),
            )
    # Ties: a function (a procedure, shown with its notes) before a note.
    found.sort(key=lambda item: (-item[0], -item[1], -item[2], -item[3], -item[4]))
    return [(key, match, accepted) for _, _, accepted, _, _, key, match in found]


def recurrence(
    match: Match,
    current_text: str,
    logged: Sequence[str],
    *,
    threshold: float,
) -> int:
    """How many earlier logged requests (not this one) name the shared identifiers, or resemble the matched request."""
    from unify.function_manager import task_origin

    earlier = [t for t in logged if t != current_text]
    if match.rank == 2 and match.shared:
        wanted = {w.lower() for w in match.shared}
        return sum(1 for t in earlier if wanted <= set(task_origin.identifiers(t)))
    if match.rank == 1:
        weights = task_origin.token_weights([*earlier, current_text, match.text])
        return sum(
            1
            for t in earlier
            if t != match.text
            and task_origin.similarity(match.text, t, weights) >= threshold
        )
    return sum(1 for t in earlier if t == current_text)


def seen_before(
    lib: Library,
    current_text: str,
    current_key: Optional[str],
    logged: Sequence[str],
    uses: Dict[Key, Any],
    *,
    threshold: float,
    k: int = K,
    shown: Optional[set] = None,
) -> List[Tuple[Card, Match]]:
    """The at most *k* cards headed by the entries recorded under a request like *current_text*, best first."""
    matches = entry_matches(
        lib,
        current_text,
        current_key,
        logged,
        uses,
        threshold=threshold,
    )
    by_key = {key: match for key, match, _ in matches}
    cards = choose_cards(lib, [key for key, _, _ in matches], k, shown)
    out = []
    for card in cards:
        match = by_key[card.key()]
        match.recurred = recurrence(match, current_text, logged, threshold=threshold)
        out.append((card, match))
    return out


# ── possibly related and standing ────────────────────────────────────────


def _connect(path) -> sqlite3.Connection:
    from unify.function_manager import task_origin

    conn = task_origin._connect_log(path)
    conn.execute(
        f"CREATE TABLE IF NOT EXISTS {TABLE} (key TEXT PRIMARY KEY, text_hash TEXT NOT NULL)",
    )
    return conn


def log_query(query: str) -> None:
    """Keep the hash of the current request's distinct text (its vector is cached), for later breadth."""
    from unify.common import embeddings
    from unify.function_manager import task_origin

    key = task_origin.current()
    if key is None or not query:
        return
    try:
        path = task_origin.request_log_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with closing(_connect(path)) as conn, conn:
            conn.execute(
                f"INSERT OR REPLACE INTO {TABLE} (key, text_hash) VALUES (?, ?)",
                (key, embeddings.text_hash(query)),
            )
            conn.execute(
                f"DELETE FROM {TABLE} WHERE key NOT IN (SELECT key FROM requests)",
            )
    except (OSError, sqlite3.Error) as exc:
        logger.warning(f"evidence query not logged: {type(exc).__name__}: {exc}")


def earlier_queries() -> List[Tuple[str, str]]:
    """``[(request text, query hash)]`` of the logged requests but the current one, oldest first."""
    from unify.function_manager import task_origin

    path = task_origin.request_log_path()
    if not path.exists():
        return []
    current = task_origin.current()
    try:
        with closing(_connect(path)) as conn:
            rows = conn.execute(
                f"SELECT r.key, r.text, q.text_hash FROM requests r JOIN {TABLE} q"
                " ON q.key = r.key ORDER BY r.seq",
            ).fetchall()
    except sqlite3.Error as exc:
        logger.warning(f"evidence queries not read: {type(exc).__name__}: {exc}")
        return []
    return [(text, h) for key, text, h in rows if key != current]


def other_jobs(
    earlier: Sequence[Tuple[str, Any]],
    current_text: str,
    known: Sequence[str],
    *,
    threshold: float,
    window: int = STANDING_WINDOW,
) -> List[Any]:
    """The payloads of the latest *window* earlier requests that are not seen-before matches of this one."""
    from unify.function_manager import task_origin

    weights = task_origin.token_weights(
        [*known, *(t for t, _ in earlier), current_text],
    )
    out = []
    for text, payload in earlier:
        if text == current_text:
            continue
        if task_origin.similarity(current_text, text, weights) >= threshold:
            continue
        if task_origin.shared_identifiers(current_text, text, known):
            continue
        out.append(payload)
    return out[-window:]


def standing_breadth(
    statement_vector: Any,
    earlier_vectors: Sequence[Any],
    floor: float,
) -> Optional[float]:
    """The share of *earlier_vectors* the floor passes for this statement; ``None`` with too few."""
    import numpy as np

    if len(earlier_vectors) < STANDING_MIN_EARLIER:
        return None
    sims = np.asarray(earlier_vectors, dtype=np.float32) @ np.asarray(
        statement_vector,
        dtype=np.float32,
    )
    return float((sims >= floor).mean())


def statement(kind: str, row: Dict[str, Any]) -> Tuple[str, bool]:
    """``(text, written)``: an entry's "use this when" statement, else its template."""
    from unify.actor import related_shortlist

    return related_shortlist.statement_of(kind, row)


# ── text ─────────────────────────────────────────────────────────────────

MAX_LINKED = 3


def _first_line(text: Any, limit: int = _LINE_CHARS) -> str:
    for line in str(text or "").splitlines():
        line = " ".join(line.split())
        if line:
            return line if len(line) <= limit else line[: limit - 1].rstrip() + "…"
    return ""


def _signature(row: Dict[str, Any]) -> str:
    argspec = str(row.get("argspec") or "").strip()
    return f"{row.get('name')}{argspec if argspec.startswith('(') else '(' + argspec + ')'}"


def _why(match: Match) -> str:
    from unify.function_manager import task_origin

    verb = {
        "origin": "stored" if match.kind == "function" else "written",
        "call": "called",
        "read": "read",
        "relied": "relied on",
    }[match.how]
    if match.rank == 3:
        where = f"{verb} while handling this same request"
        if match.recurred:
            where += f" (asked {match.recurred} time{'s' if match.recurred != 1 else ''} before)"
    elif match.rank == 2:
        named = " and ".join(f"`{w}`" for w in match.shared)
        where = f"{verb} while handling a request that also named {named}"
        if match.recurred:
            where += f" ({match.recurred} earlier request{'s' if match.recurred != 1 else ''} named it)"
    else:
        where = (
            f"{verb} while handling a request with similar wording "
            f"(similar_request {match.score:.2f}"
            + (f"; {match.recurred} earlier requests like it" if match.recurred else "")
            + ")"
        )
    kept = task_origin.origin_outcome(match.text)
    outcome = (
        task_origin._OUTCOME_TEXT[kept]
        if kept is not None
        else "that session's outcome is unknown"
    )
    return f"{where}; {outcome}"


def _record(kind: str, row: Dict[str, Any], uses: Dict[Key, Any]) -> str:
    from unify.function_manager import entry_record

    record = uses.get((kind, _ident(kind, row)))
    return "; ".join(
        [
            entry_record.status(kind, row, record),
            entry_record.use_phrase(kind, record or entry_record.Uses()),
        ],
    )


def _status(kind: str, row: Dict[str, Any], uses: Dict[Key, Any]) -> str:
    from unify.function_manager import entry_record

    return entry_record.status(kind, row, uses.get((kind, _ident(kind, row))))


def _head_line(
    card: Card,
    *,
    bound: Dict[str, bool],
    first: Optional[str] = None,
) -> str:
    kind, row = card.head
    if kind == "function":
        line = f"- function `{_signature(row)}`" + (
            " (async)" if bound.get(str(row.get("name"))) else ""
        )
        summary = first if first is not None else _first_line(row.get("docstring"))
    else:
        line = f"- guidance {row.get('guidance_id')} `{_first_line(row.get('title'))}`"
        summary = first if first is not None else _first_line(row.get("content"))
    return line + (f": {summary}" if summary else "")


def _linked_lines(
    card: Card,
    uses: Dict[Key, Any],
    *,
    bound: Dict[str, bool],
    brief: bool,
) -> List[str]:
    """A function's notes (``with guidance ...``) or a note's functions (``guides function ...``)."""
    lines = []
    for kind, row in card.linked[:MAX_LINKED]:
        if kind == "guidance":
            line = f"  with guidance {row.get('guidance_id')} `{_first_line(row.get('title'))}`"
            if not brief:
                line += f" ({_status(kind, row, uses)})"
                summary = _first_line(row.get("content"))
                line += f": {summary}" if summary else ""
        else:
            line = f"  guides function `{_signature(row)}`" + (
                " (async)" if bound.get(str(row.get("name"))) else ""
            )
            if not brief:
                summary = _first_line(row.get("docstring"))
                line += f": {summary}" if summary else ""
        lines.append(line)
    more = len(card.linked) - MAX_LINKED
    if more > 0:
        lines.append(f"  and {more} more linked entr{'ies' if more != 1 else 'y'}")
    return lines


def render(
    seen: Sequence[Tuple[Card, Match]],
    related: Sequence[Tuple[Card, str, bool, str]],
    standing: Sequence[Card],
    uses: Dict[Key, Any],
    *,
    bound: Optional[Dict[str, bool]] = None,
    call_form: Optional[str] = None,
) -> Optional[str]:
    """The list's text, or ``None`` when every tier is empty.

    *related* holds ``(card, statement, written, origin excerpt)``.
    """
    from unify.actor import related_shortlist

    bound = bound or {}
    blocks: List[str] = []
    if seen:
        header = SEEN_HEADER
        if call_form:
            header = header[: -len("):")] + f"; {call_form}):"
        lines = [header]
        for card, match in seen:
            lines.append(_head_line(card, bound=bound))
            lines += _linked_lines(card, uses, bound=bound, brief=False)
            kind, row = card.head
            lines.append(f"  why listed: {_why(match)}")
            lines.append(f"  record: {_record(kind, row, uses)}")
        blocks.append("\n".join(lines))
    if related:
        lines = [RELATED_HEADER]
        for card, text, written, origin in related:
            stated = related_shortlist._bounded(
                text,
                related_shortlist.MAX_STATEMENT_CHARS,
            )
            if not written:
                stated += f" {related_shortlist.TEMPLATE_LABEL}"
            if origin:
                stated += f' · stored for: "{origin}"'
            lines.append(_head_line(card, bound={}, first=stated))
            lines += _linked_lines(card, uses, bound={}, brief=True)
            kind, row = card.head
            lines.append(f"  record: {_record(kind, row, uses)}")
        blocks.append("\n".join(lines))
    if standing:
        named = []
        for card in standing[:MAX_STANDING_NAMED]:
            kind, row = card.head
            if kind == "guidance":
                label = _first_line(row.get("title"), _STANDING_TITLE_CHARS)
                named.append(f"guidance {row.get('guidance_id')} `{label}`")
            else:
                named.append(f"function `{row.get('name')}`")
        more = len(standing) - len(named)
        text = (
            STANDING_HEAD
            + ", ".join(named)
            + (f", and {more} more" if more > 0 else "")
            + "."
        )
        blocks.append(text)
    return "\n\n".join(blocks) if blocks else None


# ── the list for a task start ────────────────────────────────────────────


@dataclass
class Listing:
    """What the list holds, for analysis and tests."""

    seen: List[Tuple[Card, Match]] = field(default_factory=list)
    related: List[Tuple[Card, str, bool, str]] = field(default_factory=list)
    standing: List[Card] = field(default_factory=list)
    cosines: Dict[Key, float] = field(default_factory=dict)
    breadth: Dict[Key, Optional[float]] = field(default_factory=dict)
    query: str = ""


def select(
    lib: Library,
    request: str,
    current_text: str,
    current_key: Optional[str],
    logged: Sequence[str],
    uses: Dict[Key, Any],
    *,
    earlier_lines: Sequence[Any],
    earlier: Sequence[Tuple[str, str]],
    k: int,
    floor: float,
    threshold: float,
    embed: Embed,
    cached: Callable[[Sequence[str]], Dict[str, Any]],
) -> Listing:
    """Choose each tier's cards for one task start (no text, no writes)."""
    import numpy as np

    from unify.actor import related_shortlist

    out = Listing()
    shown: set = set()
    out.seen = seen_before(
        lib,
        current_text,
        current_key,
        logged,
        uses,
        threshold=threshold,
        shown=shown,
    )
    pool = [key for key in lib.entries if key not in shown]
    if not pool or k <= 0:
        return out
    query, shared = related_shortlist.distinct_text(request, earlier_lines)
    out.query = query
    statements = {key: statement(key[0], lib.entries[key]) for key in pool}
    texts = list(dict.fromkeys([query, *(text for text, _ in statements.values())]))
    vectors = np.asarray(embed(texts), dtype=np.float32)
    index = {text: i for i, text in enumerate(texts)}
    q = vectors[index[query]]
    # Standing: the floor passes this entry for most recent other jobs.
    origin_texts = [t for _, row in lib.kind_rows() for t in _origins(row)[1]]
    others = other_jobs(
        earlier,
        current_text,
        list(dict.fromkeys([*logged, *origin_texts])),
        threshold=threshold,
    )
    found = cached(others)
    earlier_vectors = [found[h] for h in others if h in found]
    ranked: List[Tuple[float, int, Key]] = []
    standing: List[Tuple[float, Key]] = []
    for key in pool:
        text, _ = statements[key]
        v = vectors[index[text]]
        cosine = float(np.dot(q, v))
        breadth = standing_breadth(v, earlier_vectors, floor)
        out.cosines[key] = cosine
        out.breadth[key] = breadth
        if cosine < floor:
            continue
        if breadth is not None and breadth > STANDING_SHARE:
            standing.append((cosine, key))
            continue
        row = lib.entries[key]
        ranked.append(
            (cosine, int(row.get("function_id") or row.get("guidance_id") or 0), key),
        )
    ranked.sort(key=lambda item: (-item[0], -item[1]))
    for card in choose_cards(lib, [key for _, _, key in ranked], k, shown):
        text, written = statements[card.key()]
        out.related.append(
            (
                card,
                text,
                written,
                related_shortlist._origin_excerpt(card.head[1], shared),
            ),
        )
    standing.sort(key=lambda item: -item[0])
    out.standing = [card_for(lib, key) for _, key in standing if key not in shown]
    return out


def block(
    function_manager: Any,
    guidance_manager: Any,
    request_text: str,
    *,
    functions: bool = True,
    guidance: bool = True,
    bind: Optional[Callable[[List[str]], Dict[str, bool]]] = None,
    call_form: Optional[str] = None,
    embed: Optional[Embed] = None,
) -> Optional[str]:
    """The evidence list for the current top-level task, or ``None`` (off, not top-level, or nothing qualifies)."""
    from unify.actor import related_shortlist
    from unify.common import embeddings
    from unify.function_manager import entry_links, entry_record, task_origin

    chosen = setting()
    current_text = task_origin.current_request()
    if chosen is None or not request_text or current_text is None:
        return None
    k, configured = chosen
    try:
        fn_rows = getattr(function_manager, "_evidence_rows", None)
        note_rows = getattr(guidance_manager, "_evidence_rows", None)
        functions_in = list(fn_rows()) if functions and callable(fn_rows) else []
        notes_in = list(note_rows()) if guidance and callable(note_rows) else []
        if not functions_in and not notes_in:
            return None
        lib = build_library(functions_in, notes_in, sorted(entry_links.links()))
        uses = entry_record.uses_of(list(lib.entries))
        listing = select(
            lib,
            request_text,
            current_text,
            task_origin.current(),
            task_origin.logged_requests(),
            uses,
            earlier_lines=task_origin.logged_request_lines(),
            earlier=earlier_queries(),
            k=k,
            floor=related_shortlist.floor_for(embeddings.embedder().model, configured),
            threshold=threshold(),
            embed=embed or embeddings.embed,
            cached=embeddings.cached_vectors,
        )
        if listing.query:
            log_query(listing.query)
    except Exception as exc:  # an aid; the task starts without it
        logger.debug(f"evidence list unavailable: {type(exc).__name__}: {exc}")
        return None
    bound: Dict[str, bool] = {}
    names = [
        str(row.get("name"))
        for card, _ in listing.seen
        for kind, row in card.members()
        if kind == "function" and row.get("name")
    ]
    if bind is not None and names:
        try:
            bound = dict(bind(list(dict.fromkeys(names))) or {})
        except Exception as exc:  # noqa: BLE001 - the list stands without it
            logger.warning(
                f"could not load the listed functions: {type(exc).__name__}: {exc}",
            )
    return render(
        listing.seen,
        listing.related,
        listing.standing,
        uses,
        bound=bound,
        call_form=call_form if bound else None,
    )


def listed_names(block_text: Optional[str]) -> Dict[str, Dict[str, List[str]]]:
    """Per tier, the function names and guidance ids a list names as card heads (for analysis and tests)."""
    from unify.actor.library_shortlist import shortlisted_names

    out: Dict[str, Dict[str, List[str]]] = {}
    for part in (block_text or "").split("\n\n"):
        if part.startswith(SEEN_HEADER[:20]):
            out["seen"] = shortlisted_names(part)
        elif part.startswith(RELATED_HEADER):
            out["related"] = shortlisted_names(part)
        elif part.startswith(STANDING_HEAD):
            out["standing"] = {"text": [part]}
    return out


__all__ = [
    "Card",
    "Library",
    "build_library",
    "card_for",
    "choose_cards",
    "entry_matches",
    "DEFAULT_THRESHOLD",
    "K",
    "Listing",
    "Match",
    "RELATED_HEADER",
    "SEEN_HEADER",
    "STANDING_HEAD",
    "block",
    "build_cards",
    "enabled",
    "listed_names",
    "render",
    "require_prerequisites",
    "seen_before",
    "select",
    "setting",
    "threshold",
]
