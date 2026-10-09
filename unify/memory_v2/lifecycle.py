"""The item lifecycle of memory v2.1 (spec §4.4, §10.1, §11; D38, D40). Deterministic; no model calls; run only
at consolidation, so the prompt prefix changes only when a commit lands.

Use records come from each request's ``memory_use.json`` (:func:`.analysis.use.request_use`), keyed by the
library commit the request was pinned to (``memory_main`` in its ``meta.json``). The evidence store indexes both
when the episode is recorded (``episodes`` and ``item_use``), and :func:`facts_from_evidence` reads that index;
:func:`facts_from_use` reads one record directly, and a test keeps the two equal.

What counts (spec §5, §10.1):
- an item is *used* in an episode when the episode imported or called it; a lookup through ``memory.show``,
  ``index`` or ``find`` is not a use;
- an *error* is a refusal (``MemoryInputError`` left it) or any other exception that left it;
- a *positive* or *negative* signal is an episode-level signal the episode's regime can observe
  (``signals.REGIME_SOURCES``): checker ``pass``/``fail`` (only when the bed declares the verdict
  agent-visible), provenance ``support``/``correct``, reader ``correct``/``re_ask``, recurrence ``re_ask``;
- in the regime with no signal (``none``) nothing but errors counts.

Environment text is never read here.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from .evidence import EvidenceStore, _unknown_items
from .signals import _CONTRARY, _SUPPORT, REGIME_SOURCES, Signal

K_STABLE = 3
MIN_EPISODES = 2  # one episode never decides a status (spec §10.1, §11 rule 4)
NO_SIGNAL_REGIME = "none"
USE_KEYS = (
    "episodes",
    "uses",
    "errors",
    "refused",
    "negative_signals",
    "positive_signals",
    "unknown",
)


@dataclass(frozen=True)
class ItemUse:
    used: bool  # imported or called
    uses: int  # call sites (a static count, as analysis.use records it)
    refused: bool  # its input check failed (MemoryInputError left it)
    errored: bool  # another exception left it
    unknown: bool  # a cell with an unknown outcome could reach it: its counts are lower bounds


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
    regime: str,
    *,
    checker_visible: bool,
) -> tuple[bool, bool]:
    """``(positive, negative)`` for one episode, from the signals the lifecycle may use (module docstring).

    A ``checker`` verdict counts only when the signal itself carries ``visible_to_actor`` (P9) and the run allows
    such signals at all (*checker_visible*, the bed's ``checker_visible_to_actor``): a grader the actor never sees
    is never used (P5 Amendment A). A source the regime cannot observe never counts, and in the no-signal regime
    nothing does."""
    if regime == NO_SIGNAL_REGIME:
        return False, False
    allowed = REGIME_SOURCES.get(regime, frozenset())
    positive = negative = False
    for s in signals:
        if s.source not in allowed:
            continue
        if s.source == "checker" and not (
            checker_visible and s.visible_to_actor is True
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
    signals: Iterable[Signal],
    *,
    checker_visible: bool,
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
        imported, called = _n(r.get("imported")), _n(r.get("called"))
        refused, errored = _n(r.get("refused")), _n(r.get("errored"))
        if not (imported or called or refused or errored):
            continue
        items[item] = ItemUse(
            bool(imported or called),
            called,
            refused > 0,
            errored > 0,
            item in unknown,
        )
    positive, negative = counted_signals(
        signals,
        regime,
        checker_visible=checker_visible,
    )
    return EpisodeUse(
        eid,
        memory_main,
        regime,
        positive,
        negative,
        dict(sorted(items.items())),
    )


def facts_from_evidence(
    ev: EvidenceStore,
    *,
    checker_visible: bool,
    eids: Iterable[str] | None = None,
) -> list[EpisodeUse]:
    """Every indexed episode's facts (only *eids* when given), in recording order, from the evidence store's
    index of each ``memory_use.json`` (``item_use``) and ``meta.json`` (``episodes.memory_main``, ``regime``).
    """
    want = set(eids) if eids is not None else None
    by_episode: dict[str, dict[str, ItemUse]] = {}
    for item, eid, imported, called, refused, errored, unknown in ev.db.execute(
        "SELECT item, episode_id, imported, called, refused, errored, outcome_unknown FROM item_use "
        "ORDER BY episode_id, item",
    ):
        if imported or called or refused or errored:
            by_episode.setdefault(eid, {})[item] = ItemUse(
                bool(imported or called),
                int(called or 0),
                (refused or 0) > 0,
                (errored or 0) > 0,
                bool(unknown),
            )
    out: list[EpisodeUse] = []
    for eid, main, regime in ev.db.execute(
        "SELECT episode_id, memory_main, regime FROM episodes ORDER BY seq",
    ):
        if want is not None and eid not in want:
            continue
        positive, negative = counted_signals(
            ev.signals_for(eid),
            regime or "",
            checker_visible=checker_visible,
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
    """The use record of every item (spec §4.4 ``use``): library commit -> counts. ``episodes`` used it,
    ``uses`` are its call sites, ``errors`` the episodes in which it refused or raised (``refused`` those that
    refused), ``negative_signals`` and ``positive_signals`` the episodes that used it and carry such a signal
    (positive only without a negative one), ``unknown`` the episodes whose outcome for it is unknown.
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
            row["errors"] += int(u.refused or u.errored)
            row["refused"] += int(u.refused)
            row["unknown"] += int(u.unknown)
            row["negative_signals"] += int(u.used and f.negative)
            row["positive_signals"] += int(u.used and f.positive and not f.negative)
    return {i: dict(sorted(c.items())) for i, c in sorted(out.items())}


# --- statuses (spec §10.1) ----------------------------------------------------------------------------------

_HIDDEN_BY_RULE = ("suspect", "deprecated")


def _negative(f: EpisodeUse) -> bool:
    return f.negative and f.regime != NO_SIGNAL_REGIME


def decide_status(
    item: str,
    prior: str | None,
    facts: Iterable[EpisodeUse],
) -> tuple[str, str | None, str | None, list[str]]:
    """``(status, rule, reason, evidence episodes)`` of *item* from the episodes that ran its current version.

    - ``deprecated`` is CURATE's (P6) and is kept as it is.
    - ``suspect`` (rule ``errors``): its input check failed or it raised in at least :data:`MIN_EPISODES`
      episodes, in any regime.
    - ``suspect`` (rule ``negative_signals``): a negative signal followed its use in at least
      :data:`MIN_EPISODES` episodes, in regimes that have signals.
    - ``stable``: used in at least :data:`K_STABLE` episodes, each with a known outcome, no error and no
      negative signal, and no episode with an error or a negative signal. A ``stable`` item stays ``stable``
      until a ``suspect`` rule fires: one episode never decides a status.
    - otherwise ``experimental``.
    """
    if prior == "deprecated":
        return "deprecated", None, None, []
    rows = [(f, f.items[item]) for f in facts if item in f.items]
    failed = sorted({f.episode_id for f, u in rows if u.refused or u.errored})
    if len(failed) >= MIN_EPISODES:
        return (
            "suspect",
            "errors",
            f"its input check failed or it raised in {len(failed)} episodes",
            failed,
        )
    negative = sorted({f.episode_id for f, u in rows if u.used and _negative(f)})
    if len(negative) >= MIN_EPISODES:
        return (
            "suspect",
            "negative_signals",
            f"negative signals followed its use in {len(negative)} episodes",
            negative,
        )
    clean = sorted(
        {
            f.episode_id
            for f, u in rows
            if u.used
            and not u.unknown
            and not (u.refused or u.errored)
            and not _negative(f)
        },
    )
    if prior == "stable" or (not failed and not negative and len(clean) >= K_STABLE):
        return "stable", None, None, clean
    return "experimental", None, None, []


def taint(
    status: Mapping[str, str],
    functions: Mapping[str, str],
    graph: Mapping[str, set[str]],
    note_uses: Mapping[str, list[str]],
) -> dict[str, tuple[str, list[str]]]:
    """Items that depend on a ``suspect`` function (spec §10.1(c), §11 rule 3), with the reason and the items
    behind it: every function of a module that imports the suspect function's module, directly or through
    others (P3's import graph, whose package imports resolve to the defining module), and every note whose
    ``uses:`` names a suspect or tainted function. Functions of the suspect function's own module are not
    tainted (the graph is per module). Items already ``suspect`` or ``deprecated`` are left out.
    """
    from .layout import dependants

    out: dict[str, tuple[str, list[str]]] = {}
    for s in sorted(
        i for i, st in status.items() if st == "suspect" and i in functions
    ):
        mod = functions[s]
        for m in sorted(dependants(dict(graph), {mod}) - {mod}):
            for f in sorted(i for i, fm in functions.items() if fm == m):
                if status.get(f) not in _HIDDEN_BY_RULE and f not in out:
                    out[f] = (f"depends on suspect {s}", [s])
    suspect = {i for i, st in status.items() if st == "suspect"} | set(out)
    for note, uses in sorted(note_uses.items()):
        hit = sorted(u for u in uses if u in suspect)
        if hit and status.get(note) not in _HIDDEN_BY_RULE and note not in out:
            out[note] = (f"uses suspect {hit[0]}", hit)
    return out


# --- poisoning (spec §8.2 rule 3, §11 rule 1) ---------------------------------------------------------------


def typed_cover_episodes(ev: EvidenceStore, item: str) -> list[str]:
    """The episodes of *item*'s procedure covers (P4's ``typed_covers`` table, when the store has it)."""
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
    positive signal: its tests then check a value from a successful episode (§8.2 rule 3's exception).
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


def _status_in(rec: Mapping | None) -> str:
    from .library_index import STATUSES

    s = rec.get("status") if isinstance(rec, Mapping) else None
    return s if s in STATUSES else "experimental"


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
) -> tuple[dict[str, dict], dict[str, str]]:
    """The map of every item of the tree (*items*: id -> kind) and the status changes against *prev*.

    An item's status counts only the episodes pinned to the library commits that ran its current version
    (*since*: from *changed_at*, the commit that last changed it, onward). A new version therefore starts as
    ``experimental``, and its predecessors' use stays in its ``use`` record. ``bisect`` and ``rollback`` carry
    over while the version is unchanged; ``alias_of`` and ``deprecated`` are CURATE's and carry over.
    A function with failure-only provenance in at least :data:`MIN_EPISODES` episodes is ``suspect`` (rule
    ``failure_only``: it belongs in a note); with one such episode it is only flagged. Taint is applied last.

    CURATE's decisions (P6, ``EvidenceStore.aliases()`` and ``curations()``), when given: a live alias is
    ``deprecated`` with ``alias_of`` its target, and an item a landed CURATE pass retired is ``deprecated`` with
    its reason (rule ``curate``). Without them ``alias_of`` and ``deprecated`` carry over.
    """
    from .item_records import empty_record

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
    records: dict[str, dict] = {}
    for item in sorted(items):
        old = prev.get(item) or {}
        rec = empty_record(item, items[item])
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
        rec["use"] = use.get(item, {})
        v = verification.get(item) or (old.get("verification") if same else None)
        rec["verification"] = v
        rec["input"] = v.get("input") if isinstance(v, dict) else None
        rec["source_channels"] = sorted(channels.get(item, []))
        flagged = sorted(failure.get(item, []))
        p = provenance.get(item) or {}
        rec["provenance"] = {
            "episodes": list(p.get("episodes", [])),
            "pass": p.get("pass"),
            "failure_only": bool(flagged),
        }
        prior = (
            old.get("status") if (same or old.get("status") == "deprecated") else None
        )
        mine = [f for f in facts if f.memory_main in since.get(item, set())]
        status, rule, reason, evidence = decide_status(item, prior, mine)
        if rec["alias_of"]:  # CURATE (P6): the name forwards to its target
            status, rule, evidence = "deprecated", "curate", []
            reason = f"an alias of {rec['alias_of']}"
        elif item in retired:
            status, rule, evidence = "deprecated", "curate", []
            reason = f"retired: {retired[item].get('reason') or ''}".strip()
        if (
            status in ("experimental", "stable")
            and items[item] == "function"
            and len(flagged) >= MIN_EPISODES
        ):
            status, rule, evidence = "suspect", "failure_only", flagged
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
    for item, (reason, evidence) in taint(current, functions, graph, note_uses).items():
        records[item].update(
            status="suspect",
            status_rule="taint",
            status_reason=reason,
            status_evidence=evidence,
        )
    changes = {
        i: r["status"]
        for i, r in records.items()
        if r["status"] != _status_in(prev.get(i))
    }
    return records, changes


# --- at consolidation (spec §4.4, §10; D38, D40) ------------------------------------------------------------

VERIFICATION_LINE = "v21-verification "


def _json_list(raw: Any) -> list:
    import json

    try:
        value = json.loads(raw or "[]")
    except ValueError:
        return []
    return value if isinstance(value, list) else []


def verification_from_passes(ev: EvidenceStore) -> dict[str, dict]:
    """The newest verification record (P4's ``v21-verification`` pass-row line) of each item a pass landed,
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
    """The channels of the recorded actions each item covers (D42: the source channel lives in the record)."""
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
    checker_visible: bool,
    work: Any,
    probe: Any = None,
    refused: Any = None,
    action_lookup: Any = None,
) -> dict:
    """The item records of ``main``'s head, after a consolidation's passes (spec §4.4, §10).

    - The map is computed from the tree at the head, every indexed episode's use and signals, the verification
      and provenance of the passes, and the map carried to the head.
    - When the head has no map yet (a pass landed it), the map is written on it: the status changes ride the
      pass's commit.
    - When the head has a map and a status changed, one empty status commit lands on ``main`` and carries the
      new map (:meth:`.memory_repo.MemoryRepo.status_commit`). Otherwise nothing is written, and the use counts
      of this consolidation appear with the next commit's map.
    - Each function that turned ``suspect`` by its own use (rules ``errors`` and ``negative_signals``) is
      bisected first (:func:`.item_bisect.bisect_item`), and a rollback target is proposed.

    Returns ``{"head", "noted", "changes", "bisected"}``; ``noted`` is the commit whose map was written, or None.
    """
    from pathlib import Path

    from .gitio import GitError
    from .item_bisect import bisect_item, rollback_target
    from .item_records import read_records, records_at, write_records
    from .layout import discover, import_graph
    from .library_export import item_history
    from .memory_repo import MemoryRepo

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
    since = {
        i: set(order[pos[c] :]) if c in pos else set(order)
        for i, c in changed_at.items()
    }
    facts = facts_from_evidence(ev, checker_visible=checker_visible)
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
        provenance=provenance_from_evidence(ev, items),
        failure=failure_only(ev, functions, {f.episode_id: f for f in facts}),
        channels=source_channels(ev, items, action_lookup),
        # CURATE's decisions (P6); getattr until P6 is integrated (a1 drops it)
        aliases=ev.aliases(),
        curations=ev.curations(),
    )
    bisected: list[str] = []
    for item in sorted(changes):
        rec = records[item]
        if rec["status"] != "suspect" or rec["status_rule"] not in (
            "errors",
            "negative_signals",
        ):
            continue
        tests = (rec.get("verification") or {}).get("tests") or []
        try:
            result = bisect_item(
                mem,
                ev,
                item,
                head,
                list(tests),
                work=Path(work) / item.replace(":", "__"),
                probe=probe,
                refused=refused,
            )
        except (GitError, OSError, ValueError) as exc:
            result = {"error": type(exc).__name__}
        rec["bisect"] = result
        rec["rollback"] = rollback_target(
            rec["use"],
            result.get("versions", []),
            result.get("introduced"),
            order,
        )
        bisected.append(item)
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
        noted = MemoryRepo(mem).status_commit(changes, evidence)
        write_records(mem, noted, records)
    return {"head": head, "noted": noted, "changes": changes, "bisected": bisected}
