"""Implicit use signals over the evidence store's ``item_use`` rows (memory v2.1, stage 1).

The rows come from :func:`.analysis.use.request_use` at each request's finish: harness numbers from cell
code and tracebacks, never a checker's verdict or text (ruling R10). Two readers:

* :func:`item_signals`: the flags the hygiene and promotion stages will read (this module only reports
  them; nothing here promotes, retires or hides an item);
* :func:`usage_table`: the compact, deterministic table a consolidation pass's first message carries when
  ``UNIFY_MEMORY_V2_SOL_USAGE=on`` (off by default).
"""

from __future__ import annotations

from typing import Iterable

from .evidence import EvidenceStore

__all__ = ["MAX_TABLE_ROWS", "USAGE_HEADING", "item_signals", "usage_table"]

MAX_TABLE_ROWS = 60
USAGE_HEADING = "Library use since the previous consolidation"


def item_signals(item: str, evidence: EvidenceStore) -> dict:
    """The implicit use signals of *item* over every indexed request.

    Exposure is what the request's prompt showed, not what its pin held: ``requests_shown`` counts the
    requests whose memory section carried the item's own line, ``requests_channel_shown`` those that
    showed its channel (a catalogue of channels shows no item lines). ``exposure_sources`` counts the
    requests at its pin by where their shown lists came from: ``record`` (the harness recorded what it
    rendered), ``legacy_text`` (a recording without that record, read from the prompt's text by the v2
    index's wording) or ``unknown`` (neither; such a request counts as showing nothing).

    * ``used``: some request has a call site of it;
    * ``never_used``: requests were shown it, or its channel when ``never_used_basis`` is ``"channel"``,
      and none imported, called or referenced it, and it never refused or failed;
    * ``refusing_accepted_inputs``: some refusal of it was followed, later in the same request, by an
      action on its channel the environment recorded as ``ok`` (a channel-level proxy);
    * ``uncertain``: dynamic calls on its channel (``getattr`` and the like) could have reached it, so
      the counts are lower bounds.

    Refusals and errors from requests that edited the item's channel in their scratch copy are not the
    stored item's and are left out of every count and flag here. When a request did not record every
    cell's outcome (``requests_outcome_unknown``), ``refusals`` and ``errors`` are None (unknown, never 0),
    ``refusals_at_least`` and ``errors_at_least`` hold the counts that were recorded, and ``never_used``
    is false.
    """
    use = evidence.item_use(item=item).get(item) or {}

    def n(key: str) -> int:
        return int(use.get(key, 0) or 0)

    touched = n("imported") + n("called") + n("referenced")
    failed = n("refused") + n("errored")
    known = n("outcome_unknown") == 0
    if n("shown") > 0:
        basis: str | None = "item"
    elif n("channel_shown") > 0:
        basis = "channel"
    else:
        basis = None
    return {
        "item": item,
        "requests_at_pin": n("requests"),
        "requests_shown": n("shown"),
        "requests_channel_shown": n("channel_shown"),
        "exposure_sources": {
            "record": n("exposure_record"),
            "legacy_text": n("exposure_legacy_text"),
            "unknown": n("exposure_unknown"),
        },
        "used_requests": n("used_requests"),
        "imported": n("imported"),
        "calls": n("called"),
        "referenced": n("referenced"),
        "guarded_calls": n("guarded"),
        "refusals": n("refused") if known else None,
        "errors": n("errored") if known else None,
        "refusals_at_least": n("refused"),
        "errors_at_least": n("errored"),
        "requests_outcome_unknown": n("outcome_unknown"),
        "outcomes_known": known,
        "refused_accepted": n("refused_accepted"),
        "unknown_calls": n("unknown_calls"),
        "last_call_seq": evidence.last_call_seq(item),
        "used": n("called") > 0,
        "never_used": basis is not None and touched == 0 and failed == 0 and known,
        "never_used_basis": basis,
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
    only harness counts. ``shown it`` counts requests whose prompt carried the item's own line, ``shown
    its channel`` those that showed its channel. Refusals and errors exclude requests that edited the
    item's channel in their scratch copy. ``last call`` is how many requests ago the latest call site
    was recorded, over every indexed request (``never`` when none).
    """
    eids = sorted({e for e in eids if isinstance(e, str)})
    ids = sorted({i for i in items if isinstance(i, str)})
    use = evidence.item_use(eids) if eids else {}
    latest = evidence.latest_seq()
    lines = [
        f"{USAGE_HEADING} ({len(eids)} requests; harness counts from cell code and tracebacks: "
        "static call sites, refusals are MemoryInputError raised out of the item, "
        "'then accepted' means a later action on its channel succeeded; requests that edited "
        "the item's channel are left out of refusals and errors):",
        "item | requests at pin | shown it | shown its channel | requests calling | calls | "
        "refusals | then accepted | other errors | dynamic calls in channel | last call",
    ]
    for item in ids[:MAX_TABLE_ROWS]:
        u = use.get(item) or {}
        seq = evidence.last_call_seq(item)
        last = "never" if seq is None else f"{latest - seq} requests ago"
        # "+?": some request at its pin did not record every cell's outcome; at least this many
        more = "+?" if u.get("outcome_unknown", 0) else ""
        lines.append(
            f"{item} | {u.get('requests', 0)} | {u.get('shown', 0)} | "
            f"{u.get('channel_shown', 0)} | {u.get('used_requests', 0)} | "
            f"{u.get('called', 0)} | {u.get('refused', 0)}{more} | "
            f"{u.get('refused_accepted', 0)}{more} | "
            f"{u.get('errored', 0)}{more} | {u.get('unknown_calls', 0)} | {last}",
        )
    if len(ids) > MAX_TABLE_ROWS:
        lines.append(f"(+{len(ids) - MAX_TABLE_ROWS} more items not shown)")
    if not ids:
        lines.append("(the library has no functions yet)")
    flags = evidence.request_flags(eids) if eids else {}
    if flags.get("outcome_unknown", 0):
        lines.append(
            f"(+?: {flags['outcome_unknown']} of these requests did not record every cell's outcome, "
            "so refusals, then accepted and other errors are at least the number shown)",
        )
    legacy, unknown = flags.get("exposure_legacy_text", 0), flags.get(
        "exposure_unknown",
        0,
    )
    if legacy or unknown:
        lines.append(
            f"(shown counts: {flags.get('exposure_record', 0)} requests from the harness's record of "
            f"what it showed, {legacy} read from the prompt's text (legacy), {unknown} unknown and "
            "counted as showing nothing)",
        )
    return "\n".join(lines) + "\n"
