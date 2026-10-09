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

from collections.abc import Iterable
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
            checker_visible and getattr(s, "visible_to_actor", False) is True
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
