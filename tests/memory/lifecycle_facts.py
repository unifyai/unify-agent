"""One episode's lifecycle facts read straight from its use record, as the evidence store's index gives them
(:func:`unify.memory.lifecycle.facts_from_evidence`): the tests compare the two and build facts from records."""

from __future__ import annotations

from typing import Any, Iterable

from unify.memory import lifecycle as lc
from unify.memory.evidence import _unknown_items, runtime_counts


def _n(value: Any) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def facts_from_use(eid: str, memory_main: str, regime: str, memory_use: Any, signals: Iterable[Any]) -> lc.EpisodeUse:
    """One episode's facts from its use record (``memory_use.json``) and its signals. An item with neither a use nor
    an error (a lookup or a bare reference) is left out."""
    use = memory_use if isinstance(memory_use, dict) else {}
    rows = use.get("items") if isinstance(use.get("items"), dict) else {}
    unknown = _unknown_items(use, {k for k in rows if isinstance(k, str)})
    items: dict[str, lc.ItemUse] = {}
    for item, r in rows.items():
        if not isinstance(item, str) or not isinstance(r, dict):
            continue
        u = lc._item_use(
            _n(r.get("imported")),
            _n(r.get("called")),
            _n(r.get("refused")),
            _n(r.get("errored")),
            item in unknown,
            runtime_counts(use, item),
        )
        if u is not None:
            items[item] = u
    positive, negative = lc.counted_signals(signals, regime)
    return lc.EpisodeUse(eid, memory_main, regime, positive, negative, dict(sorted(items.items())))
