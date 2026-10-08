"""Implicit use signals over the evidence store's ``item_use`` rows (memory v2.1, stage 1).

The rows come from :func:`.analysis.use.request_use` at each request's finish: harness numbers from cell
code and tracebacks, never a checker's verdict or text (ruling R10). Two readers:

* :func:`item_signals`: the flags the hygiene and promotion stages will read (this module only reports
  them; nothing here promotes, retires or hides an item);
* :func:`usage_table`: the compact, deterministic table each consolidation pass's first message carries.
"""

from __future__ import annotations

from typing import Iterable

from .evidence import EvidenceStore

__all__ = ["MAX_TABLE_ROWS", "USAGE_HEADING", "item_signals", "usage_table"]

MAX_TABLE_ROWS = 60
USAGE_HEADING = "Library use since the previous consolidation"


def item_signals(item: str, evidence: EvidenceStore) -> dict:
    """The implicit use signals of *item* over every indexed request whose pin held it.

    * ``used``: some request has a call site of it;
    * ``never_used``: requests saw it at their pin, and none imported, called or referenced it;
    * ``refusing_accepted_inputs``: some refusal of it was followed, later in the same request, by an
      action on its channel the environment recorded as ``ok`` (a channel-level proxy);
    * ``uncertain``: dynamic calls on its channel (``getattr`` and the like) could have reached it, so
      the counts are lower bounds.

    The counts behind them are returned too, with ``last_call_seq`` (None when never called).
    """
    use = evidence.item_use().get(item) or {}

    def n(key: str) -> int:
        return int(use.get(key, 0) or 0)

    touched = n("imported") + n("called") + n("referenced")
    return {
        "item": item,
        "requests": n("requests"),
        "used_requests": n("used_requests"),
        "imported": n("imported"),
        "calls": n("called"),
        "referenced": n("referenced"),
        "guarded_calls": n("guarded"),
        "refusals": n("refused"),
        "errors": n("errored"),
        "refused_accepted": n("refused_accepted"),
        "unknown_calls": n("unknown_calls"),
        "last_call_seq": evidence.last_call_seq(item),
        "used": n("called") > 0,
        "never_used": n("requests") > 0 and touched == 0,
        "refusing_accepted_inputs": n("refused_accepted") > 0,
        "uncertain": n("unknown_calls") > 0,
    }


def usage_table(
    evidence: EvidenceStore,
    eids: Iterable[str],
    items: Iterable[str],
) -> str:
    """The use of each item in *items* over the requests *eids* (a pass's batch), one line per item.

    Deterministic: items in id order, at most :data:`MAX_TABLE_ROWS` lines (the rest are counted), and
    only harness counts. ``last call`` is how many requests ago the latest call site was recorded, over
    every indexed request (``never`` when none).
    """
    eids = sorted({e for e in eids if isinstance(e, str)})
    ids = sorted({i for i in items if isinstance(i, str)})
    use = evidence.item_use(eids) if eids else {}
    latest = evidence.latest_seq()
    lines = [
        f"{USAGE_HEADING} ({len(eids)} requests; harness counts from cell code and tracebacks: "
        "static call sites, refusals are MemoryInputError raised out of the item, "
        "'then accepted' means a later action on its channel succeeded):",
        "item | requests seeing it | requests calling | calls | refusals | then accepted | "
        "other errors | dynamic calls in channel | last call",
    ]
    for item in ids[:MAX_TABLE_ROWS]:
        u = use.get(item) or {}
        seq = evidence.last_call_seq(item)
        last = "never" if seq is None else f"{latest - seq} requests ago"
        lines.append(
            f"{item} | {u.get('requests', 0)} | {u.get('used_requests', 0)} | "
            f"{u.get('called', 0)} | {u.get('refused', 0)} | {u.get('refused_accepted', 0)} | "
            f"{u.get('errored', 0)} | {u.get('unknown_calls', 0)} | {last}",
        )
    if len(ids) > MAX_TABLE_ROWS:
        lines.append(f"(+{len(ids) - MAX_TABLE_ROWS} more items not shown)")
    if not ids:
        lines.append("(the library has no functions yet)")
    return "\n".join(lines) + "\n"
