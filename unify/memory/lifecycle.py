"""The item lifecycle of a memory library (statuses, taint, failure-only provenance). Deterministic; no model
calls; run only at consolidation, so the prompt prefix changes only when a commit lands.

Use records come from each request's ``memory_use.json`` (:func:`.analysis.use.request_use`), keyed by the
library commit the request was pinned to (``memory_main`` in its ``meta.json``). The evidence store indexes both
when the episode is recorded (``episodes`` and ``item_use``), and :func:`facts_from_evidence` reads that index;
:func:`facts_from_use` reads one record directly, and a test keeps the two equal.

What counts:
- an item is *used* in an episode when a call to it returned. The evidence is the runtime record of the calls
  the actor's cells made (the generated library package raises one audit event per call, the worker's audit
  hook counts them, and the use record carries ``runtime`` = {ok, refused, error} per item): used means at
  least one ``ok`` call. Only a record without that evidence (an older record, or a run without the hook)
  falls back to the static estimate: more call sites of it than exceptions that left it. Each item's facts
  name the basis they rest on (``runtime`` or ``static``). An import, or a call that only raised, is
  exposure (*seen*), not use; a lookup through ``memory.get`` or ``memory.items`` is neither;
- an *error* is any exception other than a refusal that left it (on the runtime basis, the calls that
  ended in one);
- a *refusal* (``MemoryInputError`` left it: it declined an input it was not built for) is the input contract
  working. Refusals are counted and reported in the use record and the status reason, and are never a failure;
- a *positive* or *negative* signal is an episode-level signal the episode's regime can observe
  (``signals.REGIME_SOURCES``): checker ``pass``/``fail``, only when the bed declares the verdict
  agent-visible; it counts for an item only in an episode that used it;
- in the regime with no signal (``none``) nothing but errors counts.

Environment text is never read here.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
from typing import Any

from .evidence import EvidenceStore, _unknown_items, runtime_counts
from .item_states import (
    ADOPTED,
    CURATE_RULE,
    ON_TRIAL,
    RETIRED,
    level_of_record,
    status_from_stored,
)
from .signals import _CONTRARY, _SUPPORT, REGIME_SOURCES, Signal

#: The status rule of an item the writer retired or superseded with a decision (``attention.item_decisions``).
WRITER_RULE = "writer"

K_STABLE = 3
MIN_EPISODES = 2  # one episode never decides a status
NO_SIGNAL_REGIME = "none"
USE_KEYS = (
    "episodes",
    "uses",
    "seen",
    "errors",
    "refused",
    "negative_signals",
    "positive_signals",
    "unknown",
)
RUNTIME = "runtime"
STATIC = "static"


@dataclass(frozen=True)
class ItemUse:
    used: bool  # a call to it returned (see :func:`returned`)
    uses: int  # call sites (a static count, as analysis.use records it)
    refused: bool  # it refused an input (MemoryInputError left it): reported, never a failure
    errored: bool  # another exception left it
    unknown: bool  # a cell with an unknown outcome could reach it: its counts are lower bounds
    seen: bool = False  # imported or called without a call that returned: exposure only
    basis: str = (
        STATIC  # what used/refused/errored rest on: observed calls (runtime) or call sites (static)
    )
    calls: tuple[int, int, int] | None = (
        None  # observed calls (ok, refused, error) on the runtime basis
    )
    # per-call outcome rows (:class:`.trust.Use`: answered, verdict, end outcome, conditions, correction) when the
    # use record attributes its calls (``items[<id>].outcomes``); empty: the facts above, attributed as always
    outcomes: tuple = ()
    # what the use record says of a returned call: whether its result was handed back (None: not recorded), the
    # visible verdict after it, the episode's end outcome, its stated conditions (held, failed or none) and whether
    # a correction followed (:mod:`.trust`)
    answered: bool | None = None
    verdict: str | None = None
    end_outcome: str | None = None
    conditions: str | None = None
    corrected: bool = False


def returned(called: int, refused: int, errored: int) -> bool:
    """Whether a call to an item returned in an episode, from its use record's static counts: more call sites
    than exceptions that left it. Each exception that left it ended one call; a call site in a loop that raised
    on a later iteration counts as not returned, so this errs towards "not used". Only the fallback for a
    record without runtime call counts (:func:`_item_use`)."""
    return called > refused + errored


def _item_use(
    imported: int,
    called: int,
    refused: int,
    errored: int,
    unknown: bool,
    runtime: tuple[int, int, int] | None = None,
) -> ItemUse | None:
    """One item's facts in one episode, or None when the episode neither imported, called nor had an
    exception leave it (a lookup or a bare reference).

    With *runtime* (the observed calls ``(ok, refused, error)``), they rest on it alone: used when a call
    ended ``ok``, refused and errored when calls ended so; the static counts then only mark exposure (an
    import or a call site that never ran is *seen*). Without it, on the static counts (:func:`returned`).
    """
    if runtime is not None:
        ok, r_refused, r_errored = runtime
        if not (
            imported or called or refused or errored or ok or r_refused or r_errored
        ):
            return None
        used = ok > 0
        return ItemUse(
            used,
            called,
            r_refused > 0,
            r_errored > 0,
            unknown,
            seen=not used and bool(imported or called or r_refused or r_errored),
            basis=RUNTIME,
            calls=(ok, r_refused, r_errored),
        )
    if not (imported or called or refused or errored):
        return None
    used = returned(called, refused, errored)
    return ItemUse(
        used,
        called,
        refused > 0,
        errored > 0,
        unknown,
        seen=not used and bool(imported or called),
    )


@dataclass(frozen=True)
class EpisodeUse:
    episode_id: str
    memory_main: str
    regime: str
    positive: bool
    negative: bool
    items: dict = field(default_factory=dict)  # item id -> ItemUse


def counted_signals(
    signals: Iterable[Signal],
    regime: str
) -> tuple[bool, bool]:
    """``(positive, negative)`` for one episode, from the signals the lifecycle may use (module docstring).

    A ``checker`` verdict counts only when the signal itself carries ``visible_to_actor``: a grader the actor never
    sees is never used. A source the regime cannot observe never counts, and in the no-signal regime
    nothing does."""
    if regime == NO_SIGNAL_REGIME:
        return False, False
    allowed = REGIME_SOURCES.get(regime, frozenset())
    positive = negative = False
    for s in signals:
        if s.source not in allowed:
            continue
        if s.source == "checker" and not (
            (s.visible_to_actor is True)
        ):
            continue
        positive |= (s.source, s.label) in _SUPPORT
        negative |= (s.source, s.label) in _CONTRARY
    return positive, negative


def _n(value: Any) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def facts_from_use(
    eid: str,
    memory_main: str,
    regime: str,
    memory_use: Any,
    signals: Iterable[Signal]
) -> EpisodeUse:
    """One episode's facts from its use record (``memory_use.json``) and its signals. An item with neither a
    use nor an error (a lookup or a bare reference) is left out."""
    use = memory_use if isinstance(memory_use, dict) else {}
    rows = use.get("items") if isinstance(use.get("items"), dict) else {}
    unknown = _unknown_items(use, {k for k in rows if isinstance(k, str)})
    items: dict[str, ItemUse] = {}
    for item, r in rows.items():
        if not isinstance(item, str) or not isinstance(r, dict):
            continue
        u = _item_use(
            _n(r.get("imported")),
            _n(r.get("called")),
            _n(r.get("refused")),
            _n(r.get("errored")),
            item in unknown,
            runtime_counts(use, item),
        )
        attributed = _outcome_rows(eid, regime, r.get("outcomes"))
        if u is not None and attributed:
            u = replace(u, outcomes=attributed)
        if u is not None:
            items[item] = u
    positive, negative = counted_signals(
        signals,
        regime,
    )
    return EpisodeUse(
        eid,
        memory_main,
        regime,
        positive,
        negative,
        dict(sorted(items.items())),
    )


def _outcome_rows(eid: str, regime: str, rows: Any) -> tuple:
    """A use record row's per-call ``outcomes`` (:func:`.analysis.use.attribute_outcomes`) as :class:`.trust.Use`
    values of episode *eid* in *regime*; () when it has none."""
    from . import trust

    if not isinstance(rows, list):
        return ()
    reg = regime if regime in trust.REGIMES else NO_SIGNAL_REGIME
    out = []
    for row in rows:
        if isinstance(row, dict):
            u = trust.use_from_row({**row, "episode": eid, "regime": reg})
            if u is not None:
                out.append(u)
    return tuple(out)


def facts_from_evidence(
    ev: EvidenceStore,
    *,
    eids: Iterable[str] | None = None,
) -> list[EpisodeUse]:
    """Every indexed episode's facts (only *eids* when given), in recording order, from the evidence store's
    index of each ``memory_use.json`` (``item_use``) and ``meta.json`` (``episodes.memory_main``, ``regime``).
    """
    want = set(eids) if eids is not None else None
    by_episode: dict[str, dict[str, ItemUse]] = {}
    for (
        item,
        eid,
        imported,
        called,
        refused,
        errored,
        unknown,
        basis,
        r_ok,
        r_refused,
        r_errored,
    ) in ev.db.execute(
        "SELECT item, episode_id, imported, called, refused, errored, outcome_unknown, runtime_basis, "
        "runtime_ok, runtime_refused, runtime_errored FROM item_use ORDER BY episode_id, item",
    ):
        u = _item_use(
            int(imported or 0),
            int(called or 0),
            int(refused or 0),
            int(errored or 0),
            bool(unknown),
            (
                (int(r_ok or 0), int(r_refused or 0), int(r_errored or 0))
                if basis
                else None
            ),
        )
        if u is not None:
            by_episode.setdefault(eid, {})[item] = u
    out: list[EpisodeUse] = []
    for eid, main, regime in ev.db.execute(
        "SELECT episode_id, memory_main, regime FROM episodes ORDER BY seq",
    ):
        if want is not None and eid not in want:
            continue
        positive, negative = counted_signals(
            ev.signals_for(eid),
            regime or "",
        )
        items = by_episode.get(eid, {})
        out.append(
            EpisodeUse(
                eid,
                main or "",
                regime or "",
                positive,
                negative,
                dict(sorted(items.items())),
            ),
        )
    return out


def aggregate_use(facts: Iterable[EpisodeUse]) -> dict[str, dict[str, dict]]:
    """The use record of every item: library commit -> counts. ``episodes`` used it (a call returned),
    ``uses`` are its call sites, ``seen`` the episodes that imported or called it without a call that returned,
    ``errors`` the episodes in which an exception other than a refusal left it, ``refused`` those in which it
    refused an input (reported, never a failure), ``negative_signals`` and ``positive_signals`` the episodes
    that used it and carry such a signal (positive only without a negative one), ``unknown`` the episodes whose
    outcome for it is unknown. ``runtime_basis`` (present only when non-zero) counts the episodes whose facts
    rest on observed calls rather than static call sites.
    """
    out: dict[str, dict[str, dict]] = {}
    for f in facts:
        for item, u in f.items.items():
            row = out.setdefault(item, {}).setdefault(
                f.memory_main,
                dict.fromkeys(USE_KEYS, 0),
            )
            row["uses"] += u.uses
            row["episodes"] += int(u.used)
            row["seen"] += int(u.seen)
            row["errors"] += int(u.errored)
            row["refused"] += int(u.refused)
            row["unknown"] += int(u.unknown)
            row["negative_signals"] += int(u.used and f.negative)
            row["positive_signals"] += int(u.used and f.positive and not f.negative)
            if u.basis == RUNTIME:
                row["runtime_basis"] = row.get("runtime_basis", 0) + 1
    return {i: dict(sorted(c.items())) for i, c in sorted(out.items())}


# --- statuses -----------------------------------------------------------------------------------------------

def _negative(f: EpisodeUse) -> bool:
    return f.negative and f.regime != NO_SIGNAL_REGIME


def trust_uses(f: EpisodeUse, u: ItemUse) -> list:
    """*u*'s uses in episode *f* as :class:`.trust.Use` values: its attributed per-call outcomes when the use record
    has them, else one use per kind of call it saw (returned, error, refused), not attributed: a contrary signal in
    its episode counts against a returned call, as the lifecycle always counted it."""
    from . import trust

    if u.outcomes:
        return list(u.outcomes)
    regime = f.regime if f.regime in trust.REGIMES else NO_SIGNAL_REGIME
    out = []
    if u.used and u.answered is not None:
        out.append(
            trust.Use(
                f.episode_id,
                regime,
                "returned",
                answered=u.answered,
                verdict=u.verdict if u.verdict in trust.VERDICTS else None,
                end_outcome=u.end_outcome if u.end_outcome in trust.END_OUTCOMES else None,
                holds={"held": True, "failed": False}.get(u.conditions or ""),
                corrected=u.corrected,
                unknown=u.unknown,
            ),
        )
    elif u.used:
        out.append(
            trust.Use(
                f.episode_id,
                regime,
                "returned",
                answered=None,
                episode_negative=_negative(f),
                unknown=u.unknown,
            ),
        )
    if u.errored:
        out.append(trust.Use(f.episode_id, regime, "error", answered=None))
    if u.refused:
        out.append(trust.Use(f.episode_id, regime, "refused", answered=None))
    return out


def item_outcome(
    item: str,
    prior: str | None,
    facts: Iterable[EpisodeUse],
    level: str | None = None,
) -> tuple[Any, dict]:
    """``(trust outcome, track record)`` of *item* from the episodes that ran its current version: the one decider
    (:func:`.trust.apply_outcome`) and the record per kind of signal (:func:`.trust.track`). *prior* is in the
    lifecycle's names; *level* is the evidence level the gate set (None: none)."""
    from . import trust

    uses = [x for f in facts if item in f.items for x in trust_uses(f, f.items[item])]
    # the prior status in either set of names (the lifecycle's, or the trust statuses once renamed)
    prior_trust = trust.FROM_LIFECYCLE.get(prior, prior if prior in trust.STATUSES else None) if prior else None
    return trust.apply_outcome(level, prior_trust, uses), trust.track(uses)


def served(f: EpisodeUse, item: str, versions: set[str], pins: Mapping[tuple[str, str], list] | None) -> bool:
    """Whether episode *f* used the version of *item* whose commits are *versions*: by the pins its calls were served
    under when the episode recorded them (``UNIFY_MEMORY=in_episode``, :func:`use_pins`), else by the episode's first
    pin."""
    got = (pins or {}).get((f.episode_id, item))
    if got:
        return any(row[0] in versions for row in got)
    return f.memory_main in versions


def version_use(f: EpisodeUse, item: str, versions: set[str], pins: Mapping[tuple[str, str], list] | None) -> EpisodeUse:
    """*f* with *item*'s use restricted to the calls served under the version whose commits are *versions*, when its
    calls were served under more than one pin and some of them are another version's (memory changed during the
    episode, ``UNIFY_MEMORY=in_episode``): its observed calls (ok, refused, error) are those under that version's pins,
    and per-call outcomes the use record attributed are left out (they are not kept per pin), so the episode's other
    version's calls never count for this one. Otherwise *f* unchanged."""
    rows = (pins or {}).get((f.episode_id, item))
    u = f.items.get(item)
    if not rows or u is None or all(r[0] in versions for r in rows):
        return f
    ok = sum(r[1] for r in rows if r[0] in versions)
    refused = sum(r[2] for r in rows if r[0] in versions)
    error = sum(r[3] for r in rows if r[0] in versions)
    mine = replace(
        u,
        used=ok > 0,
        refused=refused > 0,
        errored=error > 0,
        seen=u.seen or bool(ok + refused + error),
        basis=RUNTIME,
        calls=(ok, refused, error),
        outcomes=(),
    )
    return replace(f, items={**f.items, item: mine})


def _labels(
    records: dict[str, dict],
    items: Mapping[str, str],
    facts: list[EpisodeUse],
    since: Mapping[str, set[str]],
    inputs: Mapping[str, Any],
    verification: Mapping[str, dict],
    pins: Mapping[tuple[str, str], list] | None = None,
) -> None:
    """Each record's evidence label (:func:`.item_label.build`): its current version's stated tests and uses (a use
    counted under the version served when it was called, *pins*), and its failure record (every failed use, each
    with that version)."""
    from . import item_label

    tests = inputs.get("tests") or {}
    order = inputs.get("pass_index") or {}
    failures = inputs.get("failures") or {}
    basis = inputs.get("basis") or {}
    for item, rec in records.items():
        v = rec.get("verification") or verification.get(item) or {}
        landed_at = order.get(v.get("pass")) if isinstance(v, Mapping) else None
        mine_tests = [
            t for t in tests.get(item, []) if landed_at is None or order.get(t.get("pass_id"), -1) >= landed_at
        ]
        versions = since.get(item, set())
        mine = [version_use(f, item, versions, pins) for f in facts if item in f.items and served(f, item, versions, pins)]
        uses = [x for f in mine for x in trust_uses(f, f.items[item])]
        rec["label"] = item_label.build(
            kind=items[item],
            record=rec,
            verification=v if isinstance(v, Mapping) else {},
            tests=mine_tests,
            uses=uses,
            failures=failures.get(item, []),
            basis=basis.get(item),
        )


def evidence_basis(item: str, facts: Iterable[EpisodeUse]) -> tuple[str, dict]:
    """What *item*'s status rests on (:data:`.trust.STATUS_BASES`) and how its uses were judged (``{"clean",
    "failed", "neutral"}``), over the episodes of *facts* that used it: its uses' own signals
    (:func:`.trust.status_basis`), and an episode-level signal its regime observes on an episode that used it (a
    verdict in ``dense``, an end outcome in ``sparse``, a correction in ``implicit``), positive or negative."""
    from . import trust

    facts = [f for f in facts if item in f.items]
    uses = [x for f in facts for x in trust_uses(f, f.items[item])]
    basis = trust.status_basis(uses)
    signalled = {f.regime for f in facts if (f.positive or f.negative) and f.items[item].used}
    order = (trust.BASIS_VERDICTS, trust.BASIS_END_OUTCOMES, trust.BASIS_IMPLICIT, trust.BASIS_NO_OUTCOME_SIGNALS)
    by_regime = {"dense": trust.BASIS_VERDICTS, "sparse": trust.BASIS_END_OUTCOMES, "implicit": trust.BASIS_IMPLICIT}
    for regime in signalled:
        other = by_regime.get(regime)
        if other is not None and order.index(other) < order.index(basis):
            basis = other
    return basis, trust.use_counts(uses)


def status_basis(item: str, facts: Iterable[EpisodeUse]) -> str | None:
    """What *item*'s adoption rests on (``verdicts`` or ``implicit``), or None when it is not adopted."""
    from . import trust

    out, _ = item_outcome(item, None, facts)
    return out.basis if out.status == trust.ADOPTED else None


def failed_uses(facts: Iterable[EpisodeUse]) -> list[dict]:
    """Every failed use in *facts*, one row per item, episode and failing signal: ``{item, episode, regime,
    signal}``, the signal as :func:`.trust.judge` names it (``rejected``, ``contrary_end``, ``holds_false``,
    ``corrected``, ``disagreed``, ``error``, ``negative``). A refusal is never one."""
    from . import trust

    out: list[dict] = []
    seen: set[tuple] = set()
    for f in facts:
        for item, u in sorted(f.items.items()):
            for x in trust_uses(f, u):
                kind, why, _ = trust.judge(x)
                key = (item, f.episode_id, why)
                if kind == trust.FAILED and key not in seen:
                    seen.add(key)
                    out.append({"item": item, "episode": f.episode_id, "regime": x.regime, "signal": why})
    return out


def decide_status(
    item: str,
    prior: str | None,
    facts: Iterable[EpisodeUse],
    prior_rule: str | None = None,
) -> tuple[str, str | None, str | None, list[str]]:
    """``(status, rule, reason, evidence episodes)`` of *item* from the episodes that ran its current version, in the
    lifecycle's names. It delegates to the one decider, :func:`.trust.apply_outcome` (statuses ``on-trial``,
    ``adopted``, ``retired`` are ``experimental``, ``stable``, ``suspect`` here):

    *prior* is its status before (old names are mapped), and *prior_rule* the rule that gave it.

    - ``retired`` by the curator (rule ``curate``) is kept as it is, with that rule.
    - ``retired`` (rule ``errors``): an exception other than a refusal left it in at least
      :data:`MIN_EPISODES` episodes, in any regime.
    - ``retired`` (rule ``negative_signals``): a negative signal followed its use in at least
      :data:`MIN_EPISODES` episodes, in regimes that have signals (a use the record does not attribute).
    - ``retired`` by an attributed failing signal (``rejected``, ``contrary_end``, ``corrected``, ``holds_false``,
      ``disagreed``) in :data:`MIN_EPISODES` episodes, or by failed uses of any kind together (``failed_uses``).
      A retirement by outcomes is decided again from the facts each time.
    - ``adopted``: used (a call returned) in at least :data:`K_STABLE` episodes, each with a known outcome, no
      failed use, and no episode with a failed use; where attributed uses are in ``dense`` or ``sparse``
      episodes, at least one carries an accepted verdict or a positive end. An ``adopted`` item stays
      ``adopted`` until a retirement rule fires: one episode never decides a status.
    - otherwise ``on-trial``. An import alone never promotes an item, and a refusal never demotes one: an
      item that has only ever refused its inputs stays ``on-trial``, and its reason says how often it
      refused.
    """
    from .item_records import OLD_CURATOR_STATUS

    if prior == OLD_CURATOR_STATUS:  # an old name: always the curator's
        prior, prior_rule = RETIRED, CURATE_RULE
    prior = status_from_stored(prior, default=None) if prior is not None else None
    if prior == RETIRED and prior_rule == CURATE_RULE:
        return RETIRED, CURATE_RULE, None, []
    facts = list(facts)
    out, _track = item_outcome(item, prior, facts)
    if out.status == RETIRED:
        return RETIRED, out.rule, out.reason, list(out.episodes)
    if out.status == ADOPTED:
        return ADOPTED, None, out.reason, list(out.episodes)
    rows = [f.items[item] for f in facts if item in f.items]
    refused = sorted({f.episode_id for f in facts if item in f.items and f.items[item].refused})
    if refused and not any(u.used for u in rows):
        return (
            ON_TRIAL,
            None,
            f"it refused inputs it was not built for in {len(refused)} episodes and no call to it has "
            "returned yet; a refusal is not a failure",
            [],
        )
    return ON_TRIAL, None, out.reason, []


def taint(
    status: Mapping[str, str],
    functions: Mapping[str, str],
    graph: Mapping[str, set[str]],
    note_uses: Mapping[str, list[str]],
    rules: Mapping[str, str | None] | None = None,
    readers: Mapping[str, Iterable[str]] | None = None,
) -> dict[str, tuple[str, list[str]]]:
    """Items that depend on a function retired by its outcomes, with the reason and the items behind it.

    A retirement is one by outcomes unless its rule (*rules*: item -> status rule) is the curator's
    (``curate``): an alias forwards to its target and a curator's retirement is a decision, not a failure, so
    neither taints. The edges:

    - every function of a module that imports the retired function's module, directly or through others (the
      library's import graph, whose package imports resolve to the defining module). Functions of the retired
      function's own module are not tainted (the graph is per module);
    - every function whose tests load parsed fixtures a retired reader produced (*readers*: item -> the ids of
      the readers its record lists), and, in turn, what depends on such a function;
    - every note whose ``uses:`` names a retired or tainted function.

    Items already ``retired`` are left out. Old status names are mapped.
    """
    from .layout import dependants

    status = {i: status_from_stored(st) for i, st in status.items()}
    rules = rules or {}
    consumers: dict[str, list[str]] = {}
    for f, rs in sorted((readers or {}).items()):
        for r in sorted(set(rs or ())):
            consumers.setdefault(r, []).append(f)
    out: dict[str, tuple[str, list[str]]] = {}

    def mark(f: str, reason: str, s: str, nxt: list[tuple[str, bool]], importers: bool) -> None:
        if status.get(f) != RETIRED and f not in out:
            out[f] = (reason, [s])
            nxt.append((f, importers))

    # (function, whether its importers are still to be tainted): a function tainted through the import graph
    # has its importers in the same closure already, so only its own readers' consumers are followed
    frontier = sorted(
        (i, True)
        for i, st in status.items()
        if st == RETIRED and rules.get(i) != CURATE_RULE and i in functions
    )
    while frontier:
        nxt: list[tuple[str, bool]] = []
        for s, importers in frontier:
            if importers:
                mod = functions[s]
                for m in sorted(dependants(dict(graph), {mod}) - {mod}):
                    for f in sorted(i for i, fm in functions.items() if fm == m):
                        mark(f, f"depends on retired {s}", s, nxt, False)
            for f in consumers.get(s, []):
                if f in functions:
                    mark(f, f"its tests load fixtures read by retired {s}", s, nxt, True)
        frontier = sorted(nxt)
    failed = {
        i
        for i, st in status.items()
        if st == RETIRED and rules.get(i) != CURATE_RULE
    } | set(out)
    for note, uses in sorted(note_uses.items()):
        hit = sorted(u for u in uses if u in failed)
        if hit and status.get(note) != RETIRED and note not in out:
            out[note] = (f"uses retired {hit[0]}", hit)
    return out


# --- poisoning ----------------------------------------------------------------------------------------------


def typed_cover_episodes(ev: EvidenceStore, item: str) -> list[str]:
    """The episodes of *item*'s procedure covers (the ``typed_covers`` table, when the store has it)."""
    if (
        ev.db.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='typed_covers'",
        ).fetchone()
        is None
    ):
        return []
    return sorted(
        {
            r[0]
            for r in ev.db.execute(
                "SELECT episode_id FROM typed_covers WHERE item=?",
                (item,),
            )
        },
    )


def failure_only(
    ev: EvidenceStore,
    functions: Iterable[str],
    facts_by_eid: Mapping[str, EpisodeUse],
) -> dict[str, list[str]]:
    """Functions whose provenance is only episodes with negative signals (no positive one), with those episodes.

    They are found through each negative episode's citing items (``signals.tainted_items``). A function is
    exempt when one of its recorded inputs (its covers, typed covers included) comes from an episode with a
    positive signal: its tests then check a value from a successful episode.
    """
    from .signals import tainted_items

    wanted = set(functions)

    def bad(eid: str) -> bool:
        f = facts_by_eid.get(eid)
        return f is not None and f.negative and not f.positive

    candidates = sorted(
        {
            i
            for e in sorted(facts_by_eid)
            if bad(e)
            for i in tainted_items(e, ev)
            if i in wanted
        },
    )
    out: dict[str, list[str]] = {}
    for item in candidates:
        provenance = ev.item_episodes(item)
        if not provenance or not all(bad(e) for e in provenance):
            continue
        covered = {e for e, _ in ev.covers_of(item)} | set(
            typed_cover_episodes(ev, item),
        )
        if any(
            facts_by_eid.get(e) is not None and facts_by_eid[e].positive
            for e in covered
        ):
            continue
        out[item] = sorted(provenance)
    return out


# --- the next map ------------------------------------------------------------------------------------------


def _as_item(f: EpisodeUse, item: str, group: list[str]) -> EpisodeUse:
    """*f* with the use of any id of *item*'s identity (*group*) counted as *item*'s own: its own use when it
    has one, else the first other id's."""
    if item in f.items or len(group) == 1:
        return f
    for n in group:
        if n in f.items:
            return replace(f, items={**f.items, item: f.items[n]})
    return f


def _status_in(rec: Mapping | None) -> str:
    s = rec.get("status") if isinstance(rec, Mapping) else None
    return status_from_stored(s)


def _combined_use(rows: Iterable[Mapping[str, dict]]) -> dict[str, dict]:
    """Per-commit use rows of several ids of one item, summed key by key."""
    out: dict[str, dict] = {}
    for by_commit in rows:
        for commit, row in by_commit.items():
            acc = out.setdefault(commit, {})
            for key, value in row.items():
                acc[key] = acc.get(key, 0) + _n(value)
    return dict(sorted(out.items()))


def identities(
    items: Iterable[str],
    prev: Mapping[str, dict],
    changed_at: Mapping[str, str | None],
    renames: Iterable[tuple[str, str]] = (),
    aliases: Mapping[str, dict] | None = None,
) -> dict[str, tuple[str, list[str]]]:
    """Each item's persistent identity: ``(uid, renamed_from)``.

    An item with a record keeps that record's uid and lineage. An item without one that a landed pass renamed
    from an older id (*renames*: ``(old, new)`` in landing order, followed back through several renames) takes
    the nearest recorded ancestor's uid, with the ids it passed through appended to the lineage. A new function
    that live CURATE aliases (*aliases*: old id -> ``{"target"}``) forward to takes the uid of the first such
    alias (by id) that has one, the alias's id joining its lineage; the alias keeps its uid as well, since calls
    to it reach the target. Anything else gets a new uid (:func:`.item_records.new_uid`).
    """
    from .item_records import new_uid

    renamed_to: dict[str, str] = {}
    for old, new in renames:
        renamed_to[new] = old

    def recorded(name: str) -> tuple[str, list[str]] | None:
        rec = prev.get(name)
        if not isinstance(rec, Mapping):
            return None
        uid = rec.get("uid")
        if not isinstance(uid, str) or not uid:
            # a record written before uids existed: derive it the way a first record does
            uid = new_uid(name, rec.get("changed_at"))
        lineage = rec.get("renamed_from")
        lineage = (
            [x for x in lineage if isinstance(x, str)]
            if isinstance(lineage, list)
            else []
        )
        return uid, lineage

    by_target: dict[str, list[str]] = {}
    for alias, row in sorted((aliases or {}).items()):
        target = row.get("target") if isinstance(row, Mapping) else None
        if isinstance(target, str):
            by_target.setdefault(target, []).append(alias)
    out: dict[str, tuple[str, list[str]]] = {}
    for item in sorted(items):
        found = recorded(item)
        if found is None:
            name, lineage, seen = item, [], {item}
            while name in renamed_to and renamed_to[name] not in seen:
                name = renamed_to[name]
                seen.add(name)
                lineage.insert(0, name)
                ancestor = recorded(name)
                if ancestor is not None:
                    found = (ancestor[0], ancestor[1] + lineage)
                    break
            if found is None and lineage:
                # renamed from ids no map recorded: a new identity that still names where it came from
                found = (new_uid(lineage[0], changed_at.get(item)), lineage)
        if found is None:
            for alias in by_target.get(item, []):
                ancestor = recorded(alias)
                if ancestor is not None:
                    found = (ancestor[0], ancestor[1] + [alias])
                    break
        out[item] = found or (new_uid(item, changed_at.get(item)), [])
    return out


def next_records(
    *,
    items: Mapping[str, str],
    prev: Mapping[str, dict],
    facts: list[EpisodeUse],
    since: Mapping[str, set[str]],
    changed_at: Mapping[str, str | None],
    functions: Mapping[str, str],
    graph: Mapping[str, set[str]],
    note_uses: Mapping[str, list[str]],
    verification: Mapping[str, dict] | None = None,
    provenance: Mapping[str, dict] | None = None,
    failure: Mapping[str, list[str]] | None = None,
    channels: Mapping[str, list[str]] | None = None,
    aliases: Mapping[str, dict] | None = None,
    curations: Iterable[Mapping] | None = None,
    unknown_version: Iterable[str] = (),
    landed: Mapping[str, Mapping] | None = None,
    label_inputs: Mapping[str, Any] | None = None,
    pins: Mapping[tuple[str, str], list] | None = None,
    writer_decisions: Mapping[str, Mapping[str, Any]] | None = None,
) -> tuple[dict[str, dict], dict[str, str]]:
    """The map of every item of the tree (*items*: id -> kind) and the status changes against *prev*.

    With *pins* (``UNIFY_MEMORY=in_episode``: (episode, item) -> the pins its calls were served under, with their
    counts, :func:`use_pins`) an item's use counts for its current version when it was served under one of that
    version's commits, wherever the episode started (:func:`served`), and only its calls under those commits
    (:func:`version_use`); without them, by the episode's first pin, as built.

    With *writer_decisions* (item -> what the writer's latest decision naming it
    says, :func:`.attention.item_decisions`) an item the writer retired is ``retired`` (rule ``writer``) and one an
    entry about it was merged into another item is ``retired`` with ``superseded_by`` that item; a later ``new_item``,
    ``changed_item`` or ``note`` naming it undoes that. The decisions are read again each time; None: as built.

    With *label_inputs* (:func:`label_inputs`: ``tests`` and ``failures`` by item,
    ``pass_index`` by pass id) every record carries its evidence label (:mod:`.item_label`), built last, from its
    final status; None: no label, the map as built.

    An item's status counts only the episodes pinned to the library commits that ran its current version
    (*since*: from *changed_at*, the commit that last changed it, onward). An item in *unknown_version* (that
    commit could not be found) is never promoted to ``adopted``: its counted use may be an older version's. A new
    version therefore starts as ``on-trial``, and its predecessors' use stays in its ``use`` record. ``bisect`` and
    ``rollback`` carry over while the version is unchanged; ``alias_of`` and a curator's retirement (``retired``
    with rule ``curate``) are CURATE's and carry over. A function with failure-only provenance in at least
    :data:`MIN_EPISODES` episodes is ``retired`` (rule ``failure_only``: it belongs in a note); with one such
    episode it is only flagged. Taint is applied last. Old status names in *prev* are mapped
    (:func:`.item_records.record_from_stored`).

    CURATE's decisions (``EvidenceStore.aliases()`` and ``curations()``), when given: a live alias is
    ``retired`` (rule ``curate``) with ``alias_of`` its target, and an item a landed CURATE pass retired is
    ``retired`` (rule ``curate``) with its reason. Without them ``alias_of`` and a curator's retirement carry over.

    The evidence fields of a function come from its verification record (the gate's): ``level``
    (:func:`.item_states.level_of_record`: a verification without one, or none, fails closed), ``level_reasons``,
    ``reads_for`` (a reader's parsed fixtures; ``role`` is then ``reader``) and ``reader`` (the readers its tests
    load fixtures of). A note's level stays None. Nothing here raises a level or sets ``confirmed-by-outcomes``.

    *landed* (:func:`.captures.landed_items`: item -> its capture, reader, signature, applies and recognition) gives
    a procedure landed from a capture its ``landed_from``, ``signature``, ``applies`` and ``recognition``; a function
    it does not name keeps its previous record's.

    Identity (:func:`identities`): every record has a ``uid`` and its ``renamed_from`` lineage, from the
    ``rename`` rows of *curations* and the live aliases. Use, the status rules' episodes and source episodes are
    joined through the uid: the ids an item had before (and an alias sharing its uid) count as the item, so its
    counts survive a rename.
    """
    from .item_records import empty_record, record_from_stored
    from .library_index import EFFECTS

    verification, provenance = verification or {}, provenance or {}
    failure, channels = failure or {}, channels or {}
    live = dict(aliases) if aliases is not None else None
    retired = {
        r["item"]: r
        for r in (curations or ())
        if isinstance(r, Mapping)
        and r.get("action") == "retire"
        and isinstance(r.get("item"), str)
    }
    use = aggregate_use(facts)
    unknown = set(unknown_version)
    renames = [
        (r["item"], r["target"])
        for r in (curations or ())
        if isinstance(r, Mapping)
        and r.get("action") == "rename"
        and isinstance(r.get("item"), str)
        and isinstance(r.get("target"), str)
    ]
    ident = identities(
        items,
        prev,
        changed_at,
        renames,
        live if live is not None else {},
    )
    names: dict[str, set[str]] = {}  # uid -> every id that stands for it
    for item, (uid, lineage) in ident.items():
        names.setdefault(uid, set()).update([item, *lineage])
    records: dict[str, dict] = {}
    for item in sorted(items):
        old = record_from_stored(dict(prev.get(item) or {}))
        rec = empty_record(item, items[item])
        uid, lineage = ident[item]
        rec["uid"], rec["renamed_from"] = uid, list(lineage)
        group = sorted(names[uid])
        rec["changed_at"] = changed_at.get(item)
        same = old.get("changed_at") == rec["changed_at"]
        alias = live.get(item) if live is not None else None
        rec["alias_of"] = (
            (alias.get("target") if isinstance(alias, Mapping) else None)
            if live is not None
            else old.get("alias_of")
        )
        if same:
            rec["bisect"], rec["rollback"] = old.get("bisect"), old.get("rollback")
        rec["use"] = (
            use.get(item, {})
            if group == [item]
            else _combined_use(use.get(n, {}) for n in group)
        )
        v = verification.get(item) or (old.get("verification") if same else None)
        rec["verification"] = v
        # the example call a gate before rev 2.1 picked from the item's tests, kept in its record; None: none
        examples = v.get("examples") if isinstance(v, dict) else None
        example = examples.get("example") if isinstance(examples, dict) else None
        rec["example"] = example if isinstance(example, dict) else None
        # the effect the gate derived from the item's covers (library_checks.derive_effect); None: not derived
        effect = v.get("effect") if isinstance(v, dict) else None
        rec["effect"] = effect if effect in EFFECTS else None
        if items[item] == "function":
            # the gate's evidence level; a verification without one fails closed (item_states.level_of_record)
            rec["level"] = level_of_record(v)
        if items[item] == "function" and isinstance(v, dict):
            rec["level_reasons"] = [str(r) for r in (v.get("level_reasons") or [])]
            rec["reads_for"] = sorted(str(p) for p in (v.get("reads_for") or []))
            rec["role"] = "reader" if rec["reads_for"] else None
            rec["reader"] = sorted(str(r) for r in (v.get("readers") or []))
        if items[item] == "function":
            got = (landed or {}).get(item)
            if got is not None:
                rec["landed_from"] = {"capture": got.get("capture"), "reader": got.get("reader")}
                rec["signature"], rec["applies"] = got.get("signature"), got.get("applies")
                rec["recognition"] = got.get("recognition")
            else:
                rec["landed_from"], rec["signature"], rec["applies"], rec["recognition"] = (
                    old.get("landed_from"),
                    old.get("signature"),
                    old.get("applies"),
                    old.get("recognition"),
                )
        rec["source_channels"] = sorted(channels.get(item, []))
        flagged = sorted(failure.get(item, []))
        p = provenance.get(item) or {}
        episodes = list(p.get("episodes", []))
        for n in group:
            if n != item:
                episodes += [
                    e
                    for e in (provenance.get(n) or {}).get("episodes", [])
                    if e not in episodes
                ]
        rec["provenance"] = {
            "episodes": episodes,
            "pass": p.get("pass"),
            "failure_only": bool(flagged),
        }
        # a curator's retirement carries across versions; any other status only while the version is unchanged
        kept = old.get("status") == RETIRED and old.get("status_rule") == CURATE_RULE
        prior = old.get("status") if (same or kept) else None
        prior_rule = old.get("status_rule") if (same or kept) else None
        wd = (writer_decisions or {}).get(item) if writer_decisions is not None else None
        if writer_decisions is not None and prior_rule == WRITER_RULE:
            # the writer's decisions are read again each time: a status they gave is never carried over
            prior, prior_rule = None, None
        mine = [
            version_use(_as_item(f, item, group), item, since.get(item, set()), pins)
            for f in facts
            if served(f, item, since.get(item, set()), pins)
        ]
        status, rule, reason, evidence = decide_status(item, prior, mine, prior_rule)
        # the record per kind of signal and what an adoption rests on (trust.py); the level moves only when the
        # gate set one (outcomes raise it to confirmed-by-outcomes or lower it from there)
        outcome, rec["track"] = item_outcome(item, prior, mine, rec.get("level"))
        rec["status_basis"] = outcome.basis if status == ADOPTED else None
        if rec.get("level") is not None:
            rec["level"] = outcome.level
        if item in unknown and status == ADOPTED and prior != ADOPTED:
            # its current version is not known: the use counted may be an older version's
            status, rule, evidence = ON_TRIAL, None, []
            reason = "not promoted to adopted: the commit that last changed it is beyond the scanned history"
        if rec["alias_of"]:  # CURATE: the name forwards to its target
            status, rule, evidence = RETIRED, CURATE_RULE, []
            reason = f"an alias of {rec['alias_of']}"
        elif item in retired:
            status, rule, evidence = RETIRED, CURATE_RULE, []
            reason = f"retired: {retired[item].get('reason') or ''}".strip()
        elif isinstance(wd, Mapping) and wd.get("status") in ("retired", "superseded"):
            status, rule, evidence = RETIRED, WRITER_RULE, []
            by = wd.get("by") if wd.get("status") == "superseded" and isinstance(wd.get("by"), str) else None
            reason = (f"superseded by {by}" if by else "retired") + f" by the writer ({wd.get('ref') or 'a decision'})"
            if by:
                rec["superseded_by"] = by
        if (
            status in (ON_TRIAL, ADOPTED)
            and items[item] == "function"
            and len(flagged) >= MIN_EPISODES
        ):
            status, rule, evidence = RETIRED, "failure_only", flagged
            reason = (
                f"failure-only provenance: its {len(flagged)} source episodes all ended with negative signals; "
                "it belongs in a note"
            )
        rec.update(
            status=status,
            status_rule=rule,
            status_reason=reason,
            status_evidence=evidence,
        )
        records[item] = rec
    current = {i: r["status"] for i, r in records.items()}
    rules = {i: r["status_rule"] for i, r in records.items()}
    # the reader edge of taint (taint's *readers*) is not passed here: status changes beyond the renamed rules
    # belong to outcome tracking, which wires it
    for item, (reason, evidence) in taint(current, functions, graph, note_uses, rules).items():
        records[item].update(
            status=RETIRED,
            status_rule="taint",
            status_reason=reason,
            status_evidence=evidence,
        )
    if label_inputs is not None:
        _labels(records, items, facts, since, label_inputs, verification or {}, pins)
    changes = {
        i: r["status"]
        for i, r in records.items()
        if r["status"] != _status_in(prev.get(i))
    }
    return records, changes


# --- at consolidation ---------------------------------------------------------------------------------------

VERIFICATION_LINE = "library_mode-verification "


def _json_list(raw: Any) -> list:
    import json

    try:
        value = json.loads(raw or "[]")
    except ValueError:
        return []
    return value if isinstance(value, list) else []


def _writer_decisions(stores: Any) -> dict[str, dict]:
    from . import attention

    return attention.item_decisions(stores.paths.home, state_dir=stores.paths.state_dir)


def use_pins(ev: EvidenceStore) -> dict[tuple[str, str], list[tuple[str, int, int, int]]]:
    """(episode, item) -> each pin its calls were served under with their (ok, refused, error) counts, from
    ``item_use_pins`` (``UNIFY_MEMORY=in_episode``); {} when the store has none."""
    out: dict[tuple[str, str], list[tuple[str, int, int, int]]] = {}
    if not ev.db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='item_use_pins'").fetchone():
        return out
    for item, eid, pin, ok, refused, error in ev.db.execute(
        "SELECT item, episode_id, pin, ok, refused, error FROM item_use_pins ORDER BY rowid",
    ):
        out.setdefault((eid, item), []).append((pin, int(ok or 0), int(refused or 0), int(error or 0)))
    return out


def label_inputs(stores: Any) -> dict[str, Any]:
    """What the evidence labels are built from (:func:`_labels`): the stated tests by item, the passes' order, and
    each item's failure record and status basis as serving gives them (:func:`.serving.item_evidence`: one row per
    failed use with the version served when it was called, why, a late contrary signal told apart, and the actors'
    reports on the run record), so a label and a card never disagree on them."""
    from pathlib import Path

    from . import item_label, serving

    ev = stores.evidence
    order = {pid: i for i, (pid,) in enumerate(ev.db.execute("SELECT pass_id FROM passes ORDER BY rowid"))}
    try:
        found = serving.item_evidence(stores, home=stores.paths.home)
    except Exception:  # noqa: BLE001 - the labels then carry no failure record, never the rest
        found = {}
    return {
        "tests": item_label.tests_by_item(Path(stores.paths.state_dir)),
        "pass_index": order,
        "failures": {item: list(e.get("failure_record") or []) for item, e in found.items()},
        "basis": {item: e["status_basis"] for item, e in found.items() if isinstance(e.get("status_basis"), str)},
    }


def verification_from_passes(ev: EvidenceStore) -> dict[str, dict]:
    """The newest verification record (the gate's ``library_mode-verification`` pass-row line) of each item a pass landed,
    with the pass that landed it."""
    import json

    out: dict[str, dict] = {}
    for pass_id, reasons, merged in ev.db.execute(
        "SELECT pass_id, reasons, items_merged FROM passes ORDER BY rowid",
    ):
        landed = {i for i in _json_list(merged) if isinstance(i, str)}
        for line in _json_list(reasons):
            if not (isinstance(line, str) and line.startswith(VERIFICATION_LINE)):
                continue
            try:
                rec = json.loads(line[len(VERIFICATION_LINE) :])
            except ValueError:
                continue
            for item, v in (rec.items() if isinstance(rec, dict) else ()):
                if item in landed and isinstance(v, dict):
                    out[item] = {**v, "pass": pass_id}
    return out


def provenance_from_evidence(
    ev: EvidenceStore,
    items: Iterable[str],
) -> dict[str, dict]:
    """Each item's source episodes (``item_evidence``) and the last pass that landed it."""
    last: dict[str, str] = {}
    for pass_id, merged in ev.db.execute(
        "SELECT pass_id, items_merged FROM passes ORDER BY rowid",
    ):
        for name in _json_list(merged):
            if isinstance(name, str):
                last[name] = pass_id
    return {
        i: {"episodes": ev.item_episodes(i), "pass": last.get(i)} for i in sorted(items)
    }


def source_channels(
    ev: EvidenceStore,
    items: Iterable[str],
    action_lookup: Any,
) -> dict[str, list[str]]:
    """The channels of the recorded actions each item covers (the source channel lives in the record)."""
    out: dict[str, list[str]] = {}
    for item in sorted(items):
        seen: set[str] = set()
        for eid, idx in ev.covers_of(item):
            action = action_lookup(eid, idx) if action_lookup is not None else None
            if action is not None and isinstance(getattr(action, "channel", None), str):
                seen.add(action.channel)
        out[item] = sorted(seen)
    return out


def consolidate_records(
    stores: Any,
    *,
    action_lookup: Any = None,
    captures: Any = None,
) -> dict:
    """The item records of ``main``'s head, after a consolidation's passes: every record carries its evidence label
    (:func:`label_inputs`), and the writer's retire and merge decisions set the records' statuses (:func:`next_records`).
    No retired item is bisected: a bisect would replay recorded task data.

    - The map is computed from the tree at the head, every indexed episode's use and signals, the verification
      and provenance of the passes, and the map carried to the head.
    - When the head has no map yet (a pass landed it), the map is written on it: the status changes ride the
      pass's commit.
    - When the head has a map and a status changed, one empty status commit lands on ``main`` and carries the
      new map (:meth:`.memory_repo.MemoryRepo.status_commit`). Otherwise nothing is written, and the use counts
      of this consolidation appear with the next commit's map.
    - An item whose last change is beyond :func:`.library_export.item_history`'s scan gets it from its own
      file's history (:func:`.item_versions.versions_of`). If that fails too, its version is unknown, and it is
      never promoted to ``adopted`` (:func:`next_records`).

    Returns ``{"head", "noted", "changes"}``: ``noted`` is the commit whose map was written, or None.
    """

    from .captures import landed_items
    from .gitio import GitError
    from .item_versions import versions_of
    from .item_records import records_at
    from .layout import discover, import_graph
    from .library_export import item_history

    mem, ev = stores.memory, stores.evidence
    head = mem.head()
    with mem.temp_checkout(head) as wt:
        lib = discover(wt)
        graph = import_graph(wt)
    items = {f.item_id: "function" for f in lib.functions} | {
        n.item_id: "note" for n in lib.notes
    }
    functions = {f.item_id: f.module for f in lib.functions}
    note_uses = {n.item_id: list(n.uses) for n in lib.notes}
    prev, _ = records_at(mem, head)
    order = mem.log_shas()
    pos = {c: i for i, c in enumerate(order)}
    full = {c[:12]: c for c in order}
    history, _complete = item_history(mem, head)
    changed_at = {
        i: (full.get(history[i][0].split(" ", 1)[0]) if history.get(i) else None)
        for i in items
    }
    unknown_version: set[str] = set()
    for i in sorted(items):
        if (
            changed_at[i] is None
        ):  # beyond item_history's scan: read its own file's history
            try:
                own = versions_of(mem, i, head)
            except GitError:
                own = []
            changed_at[i] = own[-1] if own else None
            if changed_at[i] is None:
                unknown_version.add(i)
    since = {
        i: set(order[pos[c] :]) if c in pos else set(order)
        for i, c in changed_at.items()
    }
    facts = facts_from_evidence(ev)
    records, changes = next_records(
        items=items,
        prev=prev,
        facts=facts,
        since=since,
        changed_at=changed_at,
        functions=functions,
        graph=graph,
        note_uses=note_uses,
        verification=verification_from_passes(ev),
        # an item's source episodes include those recorded under the ids it was renamed from
        provenance=provenance_from_evidence(
            ev,
            set(items)
            | {r["item"] for r in ev.curations() if r.get("action") == "rename"},
        ),
        failure=failure_only(ev, functions, {f.episode_id: f for f in facts}),
        channels=source_channels(ev, items, action_lookup),
        # CURATE's decisions: live aliases and landed retirements
        aliases=ev.aliases(),
        curations=ev.curations(),
        unknown_version=unknown_version,
        # a procedure landed from a capture: its capture, reader, signature, applies and recognition (the store's)
        landed=landed_items(captures),
        # each record's evidence label
        label_inputs=label_inputs(stores),
        # UNIFY_MEMORY=in_episode: the pin each use was served under (none recorded: as built)
        **({"pins": pins} if (pins := use_pins(ev)) else {}),
        # what the writer's decisions retired or superseded
        writer_decisions=_writer_decisions(stores),
    )
    noted = _publish(mem, head, records, changes)
    return {"head": head, "noted": noted, "changes": changes}


def _publish(
    mem: Any,
    head: str,
    records: dict[str, dict],
    changes: dict[str, str],
) -> str | None:
    """Write the map: on *head* when it has none (a pass landed it), else on one empty status commit when a
    status changed; None when nothing was written."""
    from .item_records import read_records, write_records
    from .memory_repo import MemoryRepo

    noted = None
    if read_records(mem, head) is None:
        write_records(mem, head, records)
        noted = head
    elif changes:
        evidence = sorted(
            {
                e
                for i in changes
                if records[i]["status_rule"] != "taint"
                for e in records[i]["status_evidence"]
            },
        )
        from .memory_writer import ensure_served, main_lock

        with main_lock(
            mem,
        ):  # the one writer of main; served exists before main moves
            ensure_served(mem)
            noted = MemoryRepo(mem).status_commit(changes, evidence)
        write_records(mem, noted, records)
    return noted
