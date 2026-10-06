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
* **Judged** (``UNIFY_EVIDENCE_LIST_MATCHER=judge``). Seen before is the
  same request only; the entries recorded under a request sharing a rare
  identifier with this one and the closest cards by embedding go to one
  small model call that picks the one
  doing this request's job, or none (:mod:`unify.actor.evidence_judge`). Its
  pick is listed as seen before and says a model judged it; no cosine floor
  lists anything.
* **Possibly related** (claims nothing). At most *k* cards not seen before,
  by the cosine of the request with each entry's "use this when" statement
  (:mod:`unify.actor.related_shortlist`; written by the review, else a
  template from its name and docstring or title), at or above the floor;
  one embedding call per task start. The request is compared whole: nothing
  is masked and nothing is scored against recent requests.
* **One matcher interface.** Both tiers go through a :class:`Matcher`
  (:data:`MATCHERS`), so the rule that decides "seen before" and the score
  that ranks "possibly related" can be replaced without touching cards,
  records or text. :class:`KeysAndStatements` is the default.
* **Evidence, never hiding.** Each card says why it is listed, how that
  session ended, its status (verified or not, and why) and its use, from
  :mod:`unify.function_manager.entry_record`. A note's first line is shown
  with its status.
* **Silence.** Nothing seen before and nothing above the floor: no list.

It runs for a top-level task only (a sub-agent's request is its caller's).
Under ``UNIFY_CORE_BIND_LISTED`` only the seen-before functions are bound.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

K = 5
DEFAULT_THRESHOLD = 0.175
_LINE_CHARS = 140

SEEN_HEADER = (
    "Seen before: entries recorded while handling a request like this one "
    "(read or call any of them if useful):"
)
RELATED_HEADER = (
    "Possibly related (no match is claimed; judge whether the intent is the same):"
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
        if getattr(SETTINGS, "UNIFY_EVIDENCE_LIST_MATCHER", ""):
            raise ValueError(
                "UNIFY_EVIDENCE_LIST_MATCHER needs UNIFY_EVIDENCE_LIST: it "
                "chooses how the list matches.",
            )
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
    how: str  # "origin", the use: "call", "read", "relied", or "judged"
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


# ── the matcher ──────────────────────────────────────────────────────────


class Matcher:
    """What decides "seen before" and ranks "possibly related"; replaceable as a whole.

    :meth:`seen` returns ``[(entry key, Match, accepted uses)]`` best first;
    :meth:`related_scores` returns one score per entry key (higher is
    closer) from at most one embedding call. Cards, records and text do not
    depend on which matcher chose the entries.
    """

    name = "base"

    def seen(
        self,
        lib: Library,
        current_text: str,
        current_key: Optional[str],
        logged: Sequence[str],
        uses: Dict[Key, Any],
        *,
        threshold: float,
    ) -> List[Tuple[Key, Match, int]]:
        raise NotImplementedError

    def related_scores(
        self,
        lib: Library,
        keys: Sequence[Key],
        request: str,
        *,
        embed: Embed,
    ) -> Dict[Key, float]:
        raise NotImplementedError


class KeysAndStatements(Matcher):
    """Seen before by keys (same request, rare shared identifier, ``similar_request``); related by statement cosine."""

    name = "keys"

    def seen(self, lib, current_text, current_key, logged, uses, *, threshold):
        return entry_matches(
            lib,
            current_text,
            current_key,
            logged,
            uses,
            threshold=threshold,
        )

    def related_scores(self, lib, keys, request, *, embed):
        import numpy as np

        if not keys:
            return {}
        statements = {key: statement(key[0], lib.entries[key])[0] for key in keys}
        texts = list(dict.fromkeys([request, *statements.values()]))
        vectors = np.asarray(embed(texts), dtype=np.float32)
        index = {text: i for i, text in enumerate(texts)}
        q = vectors[index[request]]
        return {
            key: float(np.dot(q, vectors[index[text]]))
            for key, text in statements.items()
        }


class JudgeMatcher(KeysAndStatements):
    """Seen before by the same request only; a model judges the other candidates (:func:`aselect`).

    :meth:`related_scores` ranks by the embedded card (name, signature,
    description and the request the entry was stored for) and feeds the
    judge's candidates, never a listing of its own.
    """

    name = "judge"

    def seen(self, lib, current_text, current_key, logged, uses, *, threshold):
        return [
            item
            for item in entry_matches(
                lib,
                current_text,
                current_key,
                logged,
                uses,
                threshold=threshold,
            )
            if item[1].rank == 3
        ]

    def related_scores(self, lib, keys, request, *, embed):
        import numpy as np

        from unify.actor import evidence_judge

        if not keys:
            return {}
        cards = {
            key: evidence_judge.card_text(key[0], lib.entries[key]) for key in keys
        }
        texts = list(dict.fromkeys([request, *cards.values()]))
        vectors = np.asarray(embed(texts), dtype=np.float32)
        index = {text: i for i, text in enumerate(texts)}
        q = vectors[index[request]]
        return {
            key: float(np.dot(q, vectors[index[text]])) for key, text in cards.items()
        }


MATCHERS: Dict[str, Callable[[], Matcher]] = {
    "keys": KeysAndStatements,
    "judge": JudgeMatcher,
}
"""The matchers by name; the default is ``keys``."""


def matcher_name() -> str:
    """``UNIFY_EVIDENCE_LIST_MATCHER``, ``keys`` when unset."""
    from unify.settings import SETTINGS

    return getattr(SETTINGS, "UNIFY_EVIDENCE_LIST_MATCHER", "") or "keys"


def judged() -> bool:
    """The evidence list is on and a model judges its candidates."""
    return enabled() and matcher_name() == "judge"


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
    }.get(match.how, "")
    if match.how == "judged":
        where = (
            "a model judged that it does the same job as this request "
            "(its inputs may differ)"
        )
    elif match.rank == 3:
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
    return "\n\n".join(blocks) if blocks else None


# ── the list for a task start ────────────────────────────────────────────


@dataclass
class Listing:
    """What the list holds, for analysis and tests."""

    seen: List[Tuple[Card, Match]] = field(default_factory=list)
    related: List[Tuple[Card, str, bool, str]] = field(default_factory=list)
    scores: Dict[Key, float] = field(default_factory=dict)
    verdict: Optional[Any] = None


def select(
    lib: Library,
    request: str,
    current_text: str,
    current_key: Optional[str],
    logged: Sequence[str],
    uses: Dict[Key, Any],
    *,
    k: int,
    floor: float,
    threshold: float,
    embed: Embed,
    matcher: Optional[Matcher] = None,
) -> Listing:
    """Choose each tier's cards for one task start (no text, no writes)."""
    from unify.actor import related_shortlist

    matcher = matcher or KeysAndStatements()
    out = Listing()
    shown: set = set()
    matches = matcher.seen(
        lib,
        current_text,
        current_key,
        logged,
        uses,
        threshold=threshold,
    )
    by_key = {key: match for key, match, _ in matches}
    for card in choose_cards(lib, [key for key, _, _ in matches], K, shown):
        match = by_key[card.key()]
        match.recurred = recurrence(match, current_text, logged, threshold=threshold)
        out.seen.append((card, match))
    pool = [key for key in lib.entries if key not in shown]
    if not pool or k <= 0:
        return out
    out.scores = matcher.related_scores(lib, pool, request, embed=embed)
    ranked = sorted(
        (
            (
                score,
                int(
                    lib.entries[key].get("function_id")
                    or lib.entries[key].get("guidance_id")
                    or 0,
                ),
                key,
            )
            for key, score in out.scores.items()
            if score >= floor
        ),
        key=lambda item: (-item[0], -item[1]),
    )
    for card in choose_cards(lib, [key for _, _, key in ranked], k, shown):
        text, written = statement(*card.head)
        out.related.append(
            (card, text, written, related_shortlist._origin_excerpt(card.head[1], [])),
        )
    return out


async def aselect(
    lib: Library,
    request: str,
    current_text: str,
    current_key: Optional[str],
    logged: Sequence[str],
    uses: Dict[Key, Any],
    *,
    threshold: float,
    embed: Embed,
    generate: Any,
) -> Listing:
    """:func:`select` with a model as the judge (``UNIFY_EVIDENCE_LIST_MATCHER=judge``).

    The same request is seen before, as with the keys, and then decides
    alone. Otherwise every entry recorded under a request that shares a rare
    whole identifier with this one, and the closest cards by embedding, are
    candidates, at most
    :data:`unify.actor.evidence_judge.K`; the judge picks the one doing this
    request's job, or none. Its pick heads a seen-before card that says a
    model judged it. Nothing is listed as possibly related.
    """
    from unify.actor import evidence_judge

    matcher = JudgeMatcher()
    out = Listing()
    shown: set = set()
    matches = entry_matches(
        lib,
        current_text,
        current_key,
        logged,
        uses,
        threshold=threshold,
    )
    exact = {key: match for key, match, _ in matches if match.rank == 3}
    # Identifier evidence only: similar wording (rank 1) on a stream of
    # templated requests is mostly the shared instructions.
    keyed = [(key, match) for key, match, _ in matches if match.rank == 2]
    for card in choose_cards(lib, list(exact), K, shown):
        match = exact[card.key()]
        match.recurred = recurrence(match, current_text, logged, threshold=threshold)
        out.seen.append((card, match))
    pool = [key for key in lib.entries if key not in shown]
    # The same request decides alone (in the bake-off that tier was right on
    # all 66 of its firings): no embedding and no judge.
    if out.seen or not pool:
        return out
    out.scores = matcher.related_scores(lib, pool, request, embed=embed)
    candidates = evidence_judge.pool(
        [key for key, _ in keyed],
        out.scores,
        exclude=sorted(shown),
        include_keyed=evidence_judge.POOL_KEYED,
    )
    shared = (
        {key: list(match.shared) for key, match in keyed if match.shared}
        if evidence_judge.NAME_SHARED
        else {}
    )
    out.verdict = await evidence_judge.decide(
        request,
        lib.entries,
        candidates,
        generate=generate,
        shared=shared,
    )
    choice = out.verdict.choice
    if choice is not None:
        for card in choose_cards(lib, [choice], 1, shown):
            out.seen.append(
                (
                    card,
                    Match(
                        1,
                        float(out.verdict.confidence or 0.0),
                        choice[0],
                        "judged",
                        evidence_judge.origin_text(lib.entries[choice]),
                    ),
                ),
            )
    return out


def _inputs(
    function_manager: Any,
    guidance_manager: Any,
    *,
    functions: bool,
    guidance: bool,
) -> Optional[Tuple[Library, Dict[Key, Any]]]:
    """The library in scope and its use records, or ``None`` when it is empty."""
    from unify.function_manager import entry_links, entry_record

    fn_rows = getattr(function_manager, "_evidence_rows", None)
    note_rows = getattr(guidance_manager, "_evidence_rows", None)
    functions_in = list(fn_rows()) if functions and callable(fn_rows) else []
    notes_in = list(note_rows()) if guidance and callable(note_rows) else []
    if not functions_in and not notes_in:
        return None
    lib = build_library(functions_in, notes_in, sorted(entry_links.links()))
    return lib, entry_record.uses_of(list(lib.entries))


def _text(
    listing: Listing,
    uses: Dict[Key, Any],
    *,
    bind: Optional[Callable[[List[str]], Dict[str, bool]]],
    call_form: Optional[str],
) -> Optional[str]:
    """The list's text, with the seen-before functions bound first when *bind* is given."""
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
        uses,
        bound=bound,
        call_form=call_form if bound else None,
    )


async def ablock(
    function_manager: Any,
    guidance_manager: Any,
    request_text: str,
    *,
    functions: bool = True,
    guidance: bool = True,
    bind: Optional[Callable[[List[str]], Dict[str, bool]]] = None,
    call_form: Optional[str] = None,
    embed: Optional[Embed] = None,
    generate: Any = None,
    judge_model: Optional[str] = None,
) -> Optional[str]:
    """:func:`block` with a model as the judge; ``generate`` defaults to a client of *judge_model*."""
    from unify.actor import evidence_judge
    from unify.common import embeddings
    from unify.function_manager import task_origin

    current_text = task_origin.current_request()
    if setting() is None or not request_text or current_text is None:
        return None
    try:
        inputs = _inputs(
            function_manager,
            guidance_manager,
            functions=functions,
            guidance=guidance,
        )
        if inputs is None:
            return None
        lib, uses = inputs
        listing = await aselect(
            lib,
            request_text,
            current_text,
            task_origin.current(),
            task_origin.logged_requests(),
            uses,
            threshold=threshold(),
            embed=embed or embeddings.embed,
            generate=generate or evidence_judge.client_generate(judge_model),
        )
    except Exception as exc:  # an aid; the task starts without it
        logger.debug(f"evidence list unavailable: {type(exc).__name__}: {exc}")
        return None
    return _text(listing, uses, bind=bind, call_form=call_form)


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
    from unify.function_manager import task_origin

    chosen = setting()
    current_text = task_origin.current_request()
    if chosen is None or not request_text or current_text is None:
        return None
    k, configured = chosen
    try:
        inputs = _inputs(
            function_manager,
            guidance_manager,
            functions=functions,
            guidance=guidance,
        )
        if inputs is None:
            return None
        lib, uses = inputs
        listing = select(
            lib,
            request_text,
            current_text,
            task_origin.current(),
            task_origin.logged_requests(),
            uses,
            k=k,
            floor=related_shortlist.floor_for(embeddings.embedder().model, configured),
            threshold=threshold(),
            embed=embed or embeddings.embed,
        )
    except Exception as exc:  # an aid; the task starts without it
        logger.debug(f"evidence list unavailable: {type(exc).__name__}: {exc}")
        return None
    return _text(listing, uses, bind=bind, call_form=call_form)


def listed_names(block_text: Optional[str]) -> Dict[str, Dict[str, List[str]]]:
    """Per tier, the function names and guidance ids a list names as card heads (for analysis and tests)."""
    from unify.actor.library_shortlist import shortlisted_names

    out: Dict[str, Dict[str, List[str]]] = {}
    for part in (block_text or "").split("\n\n"):
        if part.startswith(SEEN_HEADER[:20]):
            out["seen"] = shortlisted_names(part)
        elif part.startswith(RELATED_HEADER):
            out["related"] = shortlisted_names(part)
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
    "JudgeMatcher",
    "KeysAndStatements",
    "MATCHERS",
    "Matcher",
    "ablock",
    "aselect",
    "block",
    "build_cards",
    "enabled",
    "judged",
    "listed_names",
    "matcher_name",
    "render",
    "require_prerequisites",
    "seen_before",
    "select",
    "setting",
    "threshold",
]
