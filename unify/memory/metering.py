"""Metering and attribution: what each writer-service pass, each attention entry and each
library item cost, and the whole run's spend for reports.

**Per pass.** The service writes each pass's row to ``pass-results.jsonl`` (:func:`write_pass_row`): the entries it was
handed, took, read (verified), closed and returned; the items it landed, changed, retired and superseded; its known USD
(the writer lane's journal window when it was read, else the pass's own turns), its unpriced calls with their booking at
the worst case of one writer call, its killed turns (each once); its tokens, turns, and the record messages its agents
sent and were given. Every call of the pass carries the pass and entry ids as request metadata (ids only:
:func:`unify.common.llm_client.attribution`), and the fork's calls its episode (and segment).

**Per entry** (``metering/entries.jsonl``): a pass's known USD is split equally over the entries it *read*, exactly in
Decimal (the remainder of the division goes to the last share, so the shares sum to the pass's USD); an entry handed
but not read gets ``0`` and the basis ``handed_not_read``, counted apart as waste, and a pass that read none keeps its
whole USD on one ``handed_not_read`` row with no entry, so the view always sums to the passes' known USD.

**Per item** (``metering/items.jsonl``): the shares of the entries that closed on it (an entry closed on several items
splits its share equally over them, exactly), the requests after the item's pass that called it and those of them
with no error or refusal, and solves where a scorer supplies them (never in the build: ``null``), with USD per solve
(``"no solve"`` at zero). What the items do not take of the passes' USD is ``no_yield`` (``metering/summary.json``).

**Reports** (:func:`run_report`, :func:`render_report`): the headline is the whole run's all-lane USD (actor, writer,
fork, helper, embedding and the user simulator where a world has one), summed from the key proxy's lane journals,
divided by instances; an unpriced call is booked at its worst case and the figure is then a lower bound (``≥``),
never 0. Writer spend after the last instance is in that USD, labelled ``post_stream``, and never counted toward
stream-position curves. Per-visit rows never take writer cost: they keep the actor's calls and the librarian's (booked
in the actor's lane, labelled ``librarian``). A mesh turn is booked to the lane of the agent that makes it: a writer
turn answering an actor is part of the pass that handled the message.

Money is a decimal string or ``"unknown"``; nothing here is a float.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import ROUND_DOWN, Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

UNKNOWN = "unknown"
METERING_DIR = "metering"
ENTRIES_FILE = "entries.jsonl"
ITEMS_FILE = "items.jsonl"
SUMMARY_FILE = "summary.json"
#: The basis of an entry's share.
BASES = ("read_equal_split", "handed_not_read")
#: Shares are cut to this quantum before the last one takes the remainder (the split stays exact).
QUANTUM = Decimal("0.000000000001")
#: What reports say of the writer's place, as the published harnesses' background memory runs.
PAPER_NOTE = "writer runs outside the instance session, as the published harnesses' background memory does"
#: The lanes a report may sum (the user simulator only where a world has one).
LANES = ("actor", "writer", "fork", "helper", "embed", "user_simulator")


def _dec(value: Any) -> Optional[Decimal]:
    if value is None or isinstance(value, bool):
        return None
    try:
        d = Decimal(str(value).strip())
    except (InvalidOperation, ValueError):
        return None
    return d if d.is_finite() and d >= 0 else None


def usd(d: Decimal) -> str:
    """A non-negative Decimal as a plain decimal string (no exponent, never ``-0``)."""
    d = abs(d) if d.is_zero() else d
    text = format(d, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".") or "0"
    return text


def split_equal(total: Decimal, n: int) -> list[Decimal]:
    """*total* in *n* shares that sum to it exactly: equal shares cut to :data:`QUANTUM`, the last taking the rest."""
    if n <= 0:
        return []
    base = (total / n).quantize(QUANTUM, rounding=ROUND_DOWN)
    return [base] * (n - 1) + [total - base * (n - 1)]


def _rows(path: Path) -> list[dict]:
    try:
        data = Path(path).read_bytes()
    except OSError:
        return []
    out = []
    for line in data[: data.rfind(b"\n") + 1].split(b"\n"):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict):
            out.append(row)
    return out


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def state_dir(home: Path | str) -> Path:
    return Path(home) / "memory-state"


def metering_dir(home: Path | str) -> Path:
    return state_dir(home) / METERING_DIR


def results_path(home: Path | str) -> Path:
    return state_dir(home) / "pass-results.jsonl"


def ledger_rows(home: Path | str) -> list[dict]:
    """The attention ledger's rows (``memory-attention/ledger.jsonl``), read without the ledger's lock."""
    return _rows(state_dir(home) / "memory-attention" / "ledger.jsonl")


# --- the per-pass row ---------------------------------------------------------------------------------------------


def pass_entries(
    rows: Iterable[Mapping[str, Any]],
    pass_id: str,
    *,
    read: Iterable[str] = (),
    returned: Iterable[str] = (),
) -> dict:
    """The pass's entries by what happened to them, from the ledger's rows: ``handed`` (by the service), ``taken``
    (by the writer), ``read`` (verified), ``closed`` and ``returned`` (back, or made unreadable).
    """
    handed: list[str] = []
    taken: list[str] = []
    read_: list[str] = []
    closed: list[str] = []
    back: list[str] = []
    for r in rows:
        if r.get("pass_id") != pass_id or not isinstance(r.get("entry"), str):
            continue
        e, op = r["entry"], r.get("op")
        if op == "hand":
            (taken if r.get("by") == "take" else handed).append(e)
        elif op == "read":
            read_.append(e)
        elif op == "close":
            closed.append(e)
        elif op in ("return", "unreadable"):
            back.append(e)
    read_ += list(read)
    back += list(returned)
    uniq = lambda xs: list(dict.fromkeys(xs))  # noqa: E731
    return {
        "handed": uniq(handed),
        "taken": uniq(taken),
        "read": uniq(read_),
        "closed": uniq(closed),
        "returned": uniq(back),
    }


def pass_items(rows: Iterable[Mapping[str, Any]], pass_id: str, outcome: Any) -> dict:
    """The items the pass landed, changed, retired and superseded: what the gate merged, split by the decision that
    named it (``new_item`` or none: landed; ``changed_item`` or ``merged_into``: changed), the items a ``retire``
    decision named, and the pool entries the pass closed as superseded."""
    decided: dict[str, str] = {}
    for r in rows:
        if r.get("pass_id") == pass_id and r.get("op") == "close":
            for i in r.get("items") or ():
                if isinstance(i, str):
                    decided.setdefault(i, str(r.get("decision")))
    merged = [
        i for i in (getattr(outcome, "items_merged", None) or []) if isinstance(i, str)
    ]
    landed = [i for i in merged if decided.get(i) in (None, "new_item")]
    changed = [i for i in merged if decided.get(i) in ("changed_item", "merged_into")]
    retired = [i for i, d in decided.items() if d == "retire"]
    superseded: list[str] = []
    pool = getattr(outcome, "pool", None)
    for c in (pool or {}).get("changed") or () if isinstance(pool, dict) else ():
        if (
            isinstance(c, dict)
            and c.get("status") == "superseded"
            and isinstance(c.get("id"), str)
        ):
            superseded.append(c["id"])
    return {
        "landed": sorted(set(landed)),
        "changed": sorted(set(changed)),
        "retired": sorted(set(retired)),
        "superseded": sorted(set(superseded)),
    }


def pass_messages(home: Path | str, pass_id: str) -> dict:
    """The run-record messages the pass's agents (the writer and its helpers) posted in it, and the entries they were
    given (blocks, waits and message reads)."""
    from unify.agents.run_record import read_deliveries, read_entries

    sent = sum(
        1
        for e in read_entries(home)
        if e.get("pass_id") == pass_id
        and e.get("kind") in ("post", "reply")
        and e.get("role") in ("writer", "helper")
    )
    received = sum(
        len(d.get("seqs") or ())
        for d in read_deliveries(home)
        if d.get("pass_id") == pass_id
    )
    return {"sent": sent, "received": received}


def _tokens(outcome: Any) -> dict:
    """The writer's tokens: prompt, cached (read from the provider's cache) and completion, each as reported (None:
    no call reported it)."""
    return {
        "prompt": getattr(outcome, "input_tokens", None),
        "cached": getattr(outcome, "cache_read_tokens", None),
        "completion": getattr(outcome, "output_tokens", None),
    }


def _episode_seq_now(home: Path | str) -> Optional[int]:
    """The evidence store's last episode number now (later uses are the episodes after it), or None."""
    try:
        import sqlite3

        db = sqlite3.connect(
            f"file:{state_dir(home).parent / 'memory-evidence.sqlite'}?mode=ro",
            uri=True,
            timeout=5,
        )
        try:
            row = db.execute("SELECT MAX(seq) FROM episodes").fetchone()
        finally:
            db.close()
        return int(row[0]) if row and row[0] is not None else 0
    except Exception:  # noqa: BLE001 - unknown stays unknown
        return None


def pass_row(home: Path | str, report: Any) -> dict:
    """The pass's ``pass-results.jsonl`` row: how it ended, its commit, its entries, items, USD, tokens, turns and
    messages."""
    outcome = getattr(report, "outcome", None)
    row = {
        "pass_id": report.pass_id,
        "kind": "attention",
        "ended": report.ended,
        "commit": getattr(outcome, "commit", None),
        "drift_cleared": [],
        "suspect_cleared": [],
        "seconds": round(float(getattr(report, "seconds", 0.0) or 0.0), 3),
        "reconciled": getattr(report, "reconciled", None),
        "ended_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    if report.ended == "error":
        # rev 2: a failed pass always says why (a class name and where; never a message); a path that recorded
        # neither is named as such, so a missing cause is visible, never silent
        row["error_class"] = getattr(report, "error_class", None) or "unrecorded"
        row["error_where"] = getattr(report, "error_where", None) or "unrecorded"
    rows = ledger_rows(home)
    row["entries"] = pass_entries(
        rows,
        report.pass_id,
        read=getattr(report, "read", ()) or (),
        returned=getattr(report, "returned", ()) or (),
    )
    row["items"] = pass_items(rows, report.pass_id, outcome)
    row["usd"] = {
        "known": report.usd_known,
        "unknown_calls": int(report.unknown_calls),
        "worst_case_booked": report.worst_case_booked,
        "killed_turns": len(report.killed_turns or ()),
        # the helpers' lane's part of the above (known and unpriced count): its calls are not in the writer journal
        "helper_known": getattr(report, "helper_usd_known", "0"),
        "helper_unknown_calls": int(getattr(report, "helper_unknown_calls", 0) or 0),
    }
    row["tokens"] = _tokens(outcome)
    row["turns"] = int(getattr(report, "turns", 0) or 0)
    row["messages"] = pass_messages(home, report.pass_id)
    row["episodes_seq_at_end"] = _episode_seq_now(home)
    return row


def write_pass_row(home: Path | str, report: Any) -> dict:
    """Append the pass's row to ``pass-results.jsonl`` (the next request reports it as landed), and rebuild the
    per-entry and per-item views."""
    row = pass_row(home, report)
    path = results_path(home)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, sort_keys=True, default=str) + "\n")
        fh.flush()
        os.fsync(fh.fileno())
    rebuild_views(home)
    return row


# --- the per-entry and per-item views -----------------------------------------------------------------------------


def metered_passes(home: Path | str) -> list[dict]:
    """The pass rows that carry metering (``entries`` and ``usd``), in order."""
    return [
        r
        for r in _rows(results_path(home))
        if isinstance(r.get("entries"), dict) and isinstance(r.get("usd"), dict)
    ]


def entries_view(passes: Iterable[Mapping[str, Any]]) -> list[dict]:
    """One row per entry each pass read (its equal share of the pass's known USD, exactly) or was handed and did not
    read (0, ``handed_not_read``); a pass that read none keeps its USD on one ``handed_not_read`` row with no entry.
    A pass whose known USD is unknown gives ``unknown`` shares."""
    out: list[dict] = []
    for p in passes:
        pid = p.get("pass_id")
        ent = p.get("entries") or {}
        read = [e for e in ent.get("read") or () if isinstance(e, str)]
        owed = list(
            dict.fromkeys([*(ent.get("handed") or ()), *(ent.get("taken") or ())])
        )
        unread = [e for e in owed if e not in read]
        total = _dec((p.get("usd") or {}).get("known"))
        if total is None:
            shares: list[Any] = [UNKNOWN] * len(read)
        else:
            shares = [usd(s) for s in split_equal(total, len(read))]
        for e, share in zip(read, shares):
            out.append(
                {
                    "entry": e,
                    "pass_id": pid,
                    "share_usd": share,
                    "basis": "read_equal_split",
                }
            )
        for e in unread:
            out.append(
                {
                    "entry": e,
                    "pass_id": pid,
                    "share_usd": "0",
                    "basis": "handed_not_read",
                }
            )
        if not read:
            out.append(
                {
                    "entry": None,
                    "pass_id": pid,
                    "share_usd": usd(total) if total is not None else UNKNOWN,
                    "basis": "handed_not_read",
                },
            )
    return out


def _closures(home: Path | str) -> dict[str, list[str]]:
    """entry -> the items its (latest) close named."""
    out: dict[str, list[str]] = {}
    for r in ledger_rows(home):
        if r.get("op") == "close" and isinstance(r.get("entry"), str):
            out[r["entry"]] = [i for i in r.get("items") or () if isinstance(i, str)]
    return out


def _later_uses(
    home: Path | str, item: str, after_seq: Optional[int]
) -> tuple[Optional[int], Optional[int]]:
    """(requests after *after_seq* that called *item*, of those the ones with no error or refusal), or unknown."""
    if after_seq is None:
        return None, None
    try:
        import sqlite3

        db = sqlite3.connect(
            f"file:{state_dir(home).parent / 'memory-evidence.sqlite'}?mode=ro",
            uri=True,
            timeout=5,
        )
        try:
            row = db.execute(
                "SELECT COUNT(*), SUM(CASE WHEN u.errored = 0 AND u.refused = 0 AND u.outcome_unknown = 0 THEN 1 "
                "ELSE 0 END) FROM item_use u JOIN episodes e ON e.episode_id = u.episode_id "
                "WHERE u.item = ? AND e.seq > ? AND u.called > 0",
                (item, int(after_seq)),
            ).fetchone()
        finally:
            db.close()
    except Exception:  # noqa: BLE001 - no store or no table: unknown
        return None, None
    return int(row[0] or 0), int(row[1] or 0)


def items_view(
    home: Path | str,
    passes: list[Mapping[str, Any]],
    entries: list[Mapping[str, Any]],
    *,
    solves: Optional[Mapping[str, int]] = None,
) -> list[dict]:
    """One row per item an entry closed on: the shares of those entries (split equally over the items one closed on,
    exactly), its later calls and clean calls, and solves when *solves* (a scorer's, by item) gives them.
    """
    closes = _closures(home)
    shares: dict[str, Decimal] = {}
    unknown_items: set[str] = set()
    for row in entries:
        e = row.get("entry")
        if not isinstance(e, str) or row.get("basis") != "read_equal_split":
            continue
        items = closes.get(e) or []
        if not items:
            continue
        share = _dec(row.get("share_usd"))
        if share is None:
            unknown_items.update(items)
            continue
        for item, part in zip(items, split_equal(share, len(items))):
            shares[item] = shares.get(item, Decimal(0)) + part
    first_seq: dict[str, Optional[int]] = {}
    for p in passes:
        for e in (p.get("entries") or {}).get("closed") or ():
            for item in closes.get(e, []):
                first_seq.setdefault(item, p.get("episodes_seq_at_end"))
    out = []
    for item in sorted(set(shares) | unknown_items):
        calls, clean = _later_uses(home, item, first_seq.get(item))
        solved = None if solves is None else int(solves.get(item, 0))
        spent = None if item in unknown_items else shares.get(item, Decimal(0))
        if solved is None:
            per = None
        elif solved == 0:
            per = "no solve"
        else:
            per = usd(spent / solved) if spent is not None else UNKNOWN
        out.append(
            {
                "item": item,
                "from_entries_usd": usd(spent) if spent is not None else UNKNOWN,
                "later_calls": calls,
                "later_clean": clean,
                "later_solves": solved,
                "usd_per_solve": per,
            },
        )
    return out


def summary(
    passes: list[Mapping[str, Any]],
    entries: list[Mapping[str, Any]],
    items: list[Mapping[str, Any]],
) -> dict:
    """The views' totals: the passes' known USD, what the entries and the items take of it, the remainder the items
    do not take (``no_yield``), the waste (entries handed and not read), unpriced calls with their booking, and
    killed turns."""
    known = [_dec((p.get("usd") or {}).get("known")) for p in passes]
    total = sum((k for k in known if k is not None), Decimal(0))
    unknown_passes = sum(1 for k in known if k is None)
    ent = sum(
        (
            _dec(r.get("share_usd")) or Decimal(0)
            for r in entries
            if r.get("share_usd") != UNKNOWN
        ),
        Decimal(0),
    )
    it = sum(
        (
            _dec(r.get("from_entries_usd")) or Decimal(0)
            for r in items
            if r.get("from_entries_usd") != UNKNOWN
        ),
        Decimal(0),
    )
    booked = sum(
        (
            _dec((p.get("usd") or {}).get("worst_case_booked")) or Decimal(0)
            for p in passes
        ),
        Decimal(0),
    )
    return {
        "passes": len(passes),
        "passes_with_unknown_usd": unknown_passes,
        "known_usd": usd(total),
        "entries_usd": usd(ent),
        "items_usd": usd(it),
        "no_yield_usd": usd(max(Decimal(0), total - it)),
        "handed_not_read": sum(
            1 for r in entries if r.get("basis") == "handed_not_read" and r.get("entry")
        ),
        "handed_not_read_usd": usd(
            sum(
                (
                    _dec(r.get("share_usd")) or Decimal(0)
                    for r in entries
                    if r.get("basis") == "handed_not_read"
                ),
                Decimal(0),
            ),
        ),
        "unknown_calls": sum(
            int((p.get("usd") or {}).get("unknown_calls") or 0) for p in passes
        ),
        "worst_case_booked_usd": usd(booked),
        "killed_turns": sum(
            int((p.get("usd") or {}).get("killed_turns") or 0) for p in passes
        ),
    }


def rebuild_views(
    home: Path | str, *, solves: Optional[Mapping[str, int]] = None
) -> dict:
    """Write ``metering/entries.jsonl``, ``metering/items.jsonl`` and ``metering/summary.json`` from the metered pass
    rows (each file replaced whole, atomically). Returns the summary."""
    passes = metered_passes(home)
    entries = entries_view(passes)
    items = items_view(home, passes, entries, solves=solves)
    totals = summary(passes, entries, items)
    folder = metering_dir(home)
    _atomic_text(
        folder / ENTRIES_FILE,
        "".join(json.dumps(r, sort_keys=True) + "\n" for r in entries),
    )
    _atomic_text(
        folder / ITEMS_FILE,
        "".join(json.dumps(r, sort_keys=True) + "\n" for r in items),
    )
    _atomic_text(
        folder / SUMMARY_FILE, json.dumps(totals, sort_keys=True, indent=1) + "\n"
    )
    return totals


# --- the whole run, from the proxy's lane journals --------------------------------------------------------------------


@dataclass(frozen=True)
class LaneTotal:
    lane: str
    requests: int
    known_usd: Decimal
    unknown: int
    worst_case_usd: Decimal

    @property
    def booked_usd(self) -> Decimal:
        return self.known_usd + self.unknown * self.worst_case_usd


def lane_total(lane: str, journal: Path | str, worst_case_usd: Decimal) -> LaneTotal:
    """One lane's spend from its journal: each request (by its attempt id) at its last charge, or unknown (booked at
    *worst_case_usd*, never 0)."""
    last: dict[str, Optional[Decimal]] = {}
    for row in _rows(Path(journal)):
        rid = row.get("request_attempt_id") or row.get("call_id")
        if not isinstance(rid, str) or not rid:
            continue
        charge = _dec(row.get("account_charge"))
        if rid not in last:
            last[rid] = None
        if charge is not None:
            last[rid] = charge
    known = sum((v for v in last.values() if v is not None), Decimal(0))
    unknown = sum(1 for v in last.values() if v is None)
    return LaneTotal(lane, len(last), known, unknown, worst_case_usd)


def fork_by_episode(journal: Path | str, worst_case_usd: Decimal) -> dict[str, dict]:
    """The fork lane's spend per episode (its segment forks included), from the journal's request metadata: the
    episode a fork call names (``episode``), else its session (``fork.<episode>``, segments ``<episode>.s<n>``).
    Each episode's known USD, its unpriced calls, and its booking at the worst case; calls that name none are under
    ``unknown``."""
    last: dict[str, Optional[Decimal]] = {}
    owner: dict[str, str] = {}
    for row in _rows(Path(journal)):
        rid = row.get("request_attempt_id") or row.get("call_id")
        if not isinstance(rid, str) or not rid:
            continue
        meta = row.get("unify_meta") if isinstance(row.get("unify_meta"), dict) else {}
        episode = meta.get("episode")
        if not isinstance(episode, str) or not episode:
            session = meta.get("session")
            episode = session[len("fork.") :] if isinstance(session, str) and session.startswith("fork.") else None
        if isinstance(episode, str) and ".s" in episode:
            base, _, seg = episode.rpartition(".s")
            episode = base if seg.isdigit() else episode
        if episode:
            owner.setdefault(rid, episode)
        charge = _dec(row.get("account_charge"))
        if rid not in last:
            last[rid] = None
        if charge is not None:
            last[rid] = charge
    out: dict[str, dict] = {}
    for rid, charge in last.items():
        ep = owner.get(rid, "unknown")
        cur = out.setdefault(ep, {"requests": 0, "known": Decimal(0), "unknown": 0})
        cur["requests"] += 1
        if charge is None:
            cur["unknown"] += 1
        else:
            cur["known"] += charge
    return {
        ep: {
            "requests": v["requests"],
            "known_usd": usd(v["known"]),
            "unknown": v["unknown"],
            "booked_usd": usd(v["known"] + v["unknown"] * worst_case_usd),
        }
        for ep, v in sorted(out.items())
    }


def run_report(
    home: Path | str,
    journals: Mapping[str, Path | str],
    instances: int,
    *,
    worst_case_usd: Mapping[str, Decimal],
    last_instance_ended_at: Optional[str] = None,
    solves: Optional[Mapping[str, int]] = None,
    paper: bool = False,
) -> dict:
    """The three views of one run: the headline (whole-run all-lane USD from *journals*, lane -> path, over
    *instances*), the per-entry and the per-item views (:func:`rebuild_views`). Writer passes that started after
    *last_instance_ended_at* (UTC ISO-8601) are ``post_stream``: in the USD, never in stream-position curves.
    *worst_case_usd* gives, for every lane in *journals*, one request's worst case for that lane's model (the pinned
    catalogue's bound, :func:`.call_bounds.call_worst_case_usd`): an unpriced call is booked at it. There is no
    default: a lane without one is refused.
    """
    if isinstance(instances, bool) or not isinstance(instances, int) or instances <= 0:
        raise ValueError(f"instances is a positive whole number, not {instances!r}")
    missing = sorted(set(journals) - set(worst_case_usd or {}))
    if missing:
        raise ValueError(f"no worst case per request for lane(s) {', '.join(missing)}; give each lane its model's bound")
    lanes = []
    for lane, path in journals.items():
        if lane not in LANES:
            raise ValueError(f"unknown lane {lane!r}; one of {', '.join(LANES)}")
        lanes.append(lane_total(lane, path, Decimal(worst_case_usd[lane])))
    known = sum((t.known_usd for t in lanes), Decimal(0))
    booked = sum((t.booked_usd for t in lanes), Decimal(0))
    unknown = sum(t.unknown for t in lanes)
    passes = metered_passes(home)
    post = Decimal(0)
    post_passes = 0
    if last_instance_ended_at:
        for p in _rows(results_path(home)):
            if p.get("kind") != "attention":
                continue
            started = _pass_started(home, p.get("pass_id"))
            if started is not None and started > last_instance_ended_at:
                post += _dec((p.get("usd") or {}).get("known")) or Decimal(0)
                post_passes += 1
    totals = rebuild_views(home, solves=solves)
    headline = {
        "usd": usd(booked),
        "bound": "≥" if unknown else "=",
        "known_usd": usd(known),
        "unknown_calls": unknown,
        "instances": instances,
        "usd_per_instance": usd(booked / instances),
        "by_lane": {
            t.lane: {
                "requests": t.requests,
                "known_usd": usd(t.known_usd),
                "unknown": t.unknown,
                "booked_usd": usd(t.booked_usd),
            }
            for t in lanes
        },
        "writer_post_stream_usd": usd(post),
        "writer_post_stream_passes": post_passes,
        # segment forks multiply the fork's calls on long episodes: the fork lane's spend per episode
        "fork_by_episode": (
            fork_by_episode(journals["fork"], Decimal(worst_case_usd["fork"]))
            if "fork" in journals
            else {}
        ),
        "labels": {
            "post_stream": "writer spend after the last instance: in the USD, not in stream-position curves",
            "librarian": "booked in the actor's lane",
            **({"paper": PAPER_NOTE} if paper else {}),
        },
    }
    return {
        "headline": headline,
        "entries": entries_view(passes),
        "items": _rows(metering_dir(home) / ITEMS_FILE),
        "summary": totals,
    }


def _pass_started(home: Path | str, pass_id: Any) -> Optional[str]:
    if not isinstance(pass_id, str):
        return None
    try:
        doc = json.loads(
            (
                state_dir(home) / "memory-service" / "turns" / pass_id / "pass.json"
            ).read_text()
        )
    except (OSError, ValueError):
        return None
    started = doc.get("started_at") if isinstance(doc, dict) else None
    return started if isinstance(started, str) else None


def render_report(report: Mapping[str, Any]) -> str:
    """The three views as text: the headline (with its bound and labels), the per-entry rows, the per-item rows."""
    h = report["headline"]
    s = report["summary"]
    lines = [
        f"Whole run, all lanes: {h['bound']} {h['usd']} USD over {h['instances']} instances "
        f"({h['bound']} {h['usd_per_instance']} USD per instance); known {h['known_usd']} USD, "
        f"{h['unknown_calls']} unpriced calls booked at their worst case.",
        f"Writer after the last instance (post_stream): {h['writer_post_stream_usd']} USD in "
        f"{h['writer_post_stream_passes']} passes, in the USD above, not in stream-position curves.",
        "By lane: "
        + "; ".join(
            f"{lane} {v['booked_usd']} USD ({v['requests']} requests)"
            for lane, v in h["by_lane"].items()
        ),
        "The librarian's calls are in the actor's lane.",
    ]
    if "paper" in h["labels"]:
        lines.append(f"Paper rows: {h['labels']['paper']}.")
    lines += [
        "",
        f"Writer passes: {s['passes']}, known {s['known_usd']} USD; entries {s['entries_usd']} USD; items "
        f"{s['items_usd']} USD; no_yield {s['no_yield_usd']} USD; handed and not read {s['handed_not_read']} "
        f"entries ({s['handed_not_read_usd']} USD); killed turns {s['killed_turns']}.",
        "",
        "Per entry:",
        *[
            f"  {r['entry'] or '(no entry read)'} pass {r['pass_id']}: {r['share_usd']} USD ({r['basis']})"
            for r in report["entries"]
        ],
        "",
        "Per item:",
        *[
            f"  {r['item']}: {r['from_entries_usd']} USD from entries; later calls {r['later_calls']}, clean "
            f"{r['later_clean']}, solves {r['later_solves']}, USD per solve {r['usd_per_solve']}"
            for r in report["items"]
        ],
    ]
    return "\n".join(lines)


__all__ = [
    "BASES",
    "LaneTotal",
    "PAPER_NOTE",
    "entries_view",
    "fork_by_episode",
    "items_view",
    "lane_total",
    "metered_passes",
    "pass_entries",
    "pass_items",
    "pass_messages",
    "pass_row",
    "rebuild_views",
    "render_report",
    "run_report",
    "split_equal",
    "summary",
    "usd",
    "write_pass_row",
]
