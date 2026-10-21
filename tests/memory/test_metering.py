"""Metering and attribution (UNIFY_MEMORY_METERING): exact per-entry shares, per-item shares with the remainder as
no_yield, unpriced calls booked at their worst case (a lower bound, never 0), a killed turn counted once, the headline
summed from the proxy's lane journals with post-stream writer spend labelled, ids-only call attribution, and per-visit
rows that never take writer cost (the redesign's A3 tests 1-5 and the failure modes they rule out).
"""

from __future__ import annotations

import json
import os
import signal
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import pytest

from unify.memory import metering as M

SPLIT = [
    # three reads: 0.01 does not divide by 3 in decimal
    {"pass_id": "p1", "kind": "attention", "ended": "landed",
     "entries": {"handed": ["e1", "e2", "e3", "e4"], "taken": ["e5"], "read": ["e1", "e2", "e5"], "closed": ["e1", "e5"],
                 "returned": ["e3", "e4"]},
     "usd": {"known": "0.01", "unknown_calls": 0, "worst_case_booked": "0", "killed_turns": 0}},
    # read nothing: its USD is waste, on one handed_not_read row
    {"pass_id": "p2", "kind": "attention", "ended": "landed",
     "entries": {"handed": ["e6"], "taken": [], "read": [], "closed": [], "returned": ["e6"]},
     "usd": {"known": "0.0213", "unknown_calls": 1, "worst_case_booked": "13.125", "killed_turns": 1}},
    {"pass_id": "p3", "kind": "attention", "ended": "landed",
     "entries": {"handed": ["e7"], "taken": [], "read": ["e7"], "closed": ["e7"], "returned": []},
     "usd": {"known": "0.000000000000000007", "unknown_calls": 0, "worst_case_booked": "0", "killed_turns": 0}},
]  # fmt: skip
CLOSES = [
    {
        "op": "close",
        "entry": "e1",
        "pass_id": "p1",
        "decision": "new_item",
        "items": ["fn:a", "fn:b"],
    },
    {
        "op": "close",
        "entry": "e5",
        "pass_id": "p1",
        "decision": "changed_item",
        "items": ["fn:a"],
    },
    {
        "op": "close",
        "entry": "e7",
        "pass_id": "p3",
        "decision": "nothing_to_keep",
        "items": [],
    },
]


def _home(tmp_path: Path, passes=SPLIT, ledger=CLOSES) -> Path:
    home = tmp_path / "home"
    state = home / "memory-state"
    (state / "memory-attention").mkdir(parents=True)
    (state / "pass-results.jsonl").write_text(
        "".join(json.dumps(p) + "\n" for p in passes)
    )
    (state / "memory-attention" / "ledger.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in ledger)
    )
    return home


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_split_equal_is_exact_for_any_total_and_count():
    for total in ("0.01", "1", "0.000000000000001", "13.125", "0"):
        for n in (1, 2, 3, 7, 13):
            parts = M.split_equal(Decimal(total), n)
            assert (
                len(parts) == n
                and sum(parts) == Decimal(total)
                and all(p >= 0 for p in parts)
            )


def test_the_items_take_the_closing_entries_shares_and_the_rest_is_no_yield(tmp_path):
    """A3 test 2: sum(items.from_entries_usd) <= the passes' totals, the remainder labelled no_yield."""
    home = _home(tmp_path)
    totals = M.rebuild_views(home)
    items = {r["item"]: r for r in _rows(M.metering_dir(home) / M.ITEMS_FILE)}
    entries = {
        r["entry"]: Decimal(r["share_usd"])
        for r in _rows(M.metering_dir(home) / M.ENTRIES_FILE)
        if r["entry"]
    }
    # e1 closed on two items: its share split equally over them, exactly; e5 on fn:a alone
    a, b = M.split_equal(entries["e1"], 2)
    assert Decimal(items["fn:a"]["from_entries_usd"]) == a + entries["e5"]
    assert Decimal(items["fn:b"]["from_entries_usd"]) == b
    spent = sum(Decimal(r["from_entries_usd"]) for r in items.values())
    known = sum(Decimal(p["usd"]["known"]) for p in SPLIT)
    assert spent <= known and Decimal(totals["no_yield_usd"]) == known - spent
    # no scorer gave solves: unknown, never 0
    assert (
        items["fn:a"]["later_solves"] is None and items["fn:a"]["usd_per_solve"] is None
    )
    solved = {
        r["item"]: r
        for r in M.items_view(
            home,
            M.metered_passes(home),
            _rows(M.metering_dir(home) / M.ENTRIES_FILE),
            solves={"fn:a": 2},
        )
    }
    assert solved["fn:b"]["usd_per_solve"] == "no solve"
    assert (
        Decimal(solved["fn:a"]["usd_per_solve"])
        == Decimal(items["fn:a"]["from_entries_usd"]) / 2
    )


def _journal(path: Path, rows: list[dict]) -> Path:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


def test_an_unpriced_call_is_booked_at_its_worst_case_and_the_figure_is_a_lower_bound(
    tmp_path,
):
    """A3 test 3."""
    home = _home(tmp_path)
    writer = _journal(
        tmp_path / "writer.jsonl",
        [
            {"request_attempt_id": "w1", "origin": "request_started"},
            {
                "request_attempt_id": "w1",
                "status": "completed",
                "account_charge": "0.02",
            },
            {"request_attempt_id": "w2", "origin": "request_started"},
            {"request_attempt_id": "w2", "status": "cancelled"},  # never priced
        ],
    )
    report = M.run_report(
        home, {"writer": writer}, 2, worst_case_usd={"writer": Decimal("13.125")}
    )
    h = report["headline"]
    assert h["bound"] == "≥" and h["unknown_calls"] == 1 and h["known_usd"] == "0.02"
    assert Decimal(h["usd"]) == Decimal("13.145") and Decimal(
        h["usd_per_instance"]
    ) == Decimal("6.5725")
    assert "≥ 13.145 USD" in M.render_report(report)


def test_the_headline_is_the_sum_over_the_lane_journals_with_post_stream_writer_spend_labelled(
    tmp_path,
):
    """A3 test 5, and rules 1-2 and 6: whole-run all-lane USD over instances; writer passes after the last instance in
    it, labelled post_stream; paper rows carry the background-memory note."""
    home = _home(tmp_path)
    lanes = {}
    want = Decimal(0)
    for lane, charges in {
        "actor": [
            "0.10",
            "0.20",
        ],  # the librarian's calls are in the actor's lane: never counted apart
        "writer": ["0.05"],
        "fork": ["0.01", "0.01"],
        "helper": ["0.003"],
        "embed": ["0.0001"],
    }.items():
        rows = []
        for i, c in enumerate(charges):
            rows += [
                {"request_attempt_id": f"{lane}{i}", "origin": "request_started"},
                {
                    "request_attempt_id": f"{lane}{i}",
                    "status": "completed",
                    "account_charge": "0",
                },
                {
                    "request_attempt_id": f"{lane}{i}",
                    "status": "completed",
                    "account_charge": c,
                },  # the last charge
            ]
            want += Decimal(c)
        lanes[lane] = _journal(tmp_path / f"{lane}.jsonl", rows)
    turns = home / "memory-state" / "memory-service" / "turns"
    for pid, started in (
        ("p1", "2026-10-20T20:00:00.000Z"),
        ("p2", "2026-10-20T23:00:00.000Z"),
    ):
        (turns / pid).mkdir(parents=True)
        (turns / pid / "pass.json").write_text(json.dumps({"started_at": started}))
    # the fork lane's calls name their episode (the proxy journals the metadata): one episode, its segment fork too
    fork_rows = _rows(lanes["fork"])
    fork_rows[0]["unify_meta"] = {"session": "fork.ep-1"}
    fork_rows[3]["unify_meta"] = {"session": "fork.ep-1.s2", "episode": "ep-1"}
    _journal(lanes["fork"], fork_rows)
    # each lane's unpriced calls at its model's catalogue bound (none unpriced here)
    worst = {lane: Decimal("0.65625") for lane in lanes}
    report = M.run_report(
        home, lanes, 4, worst_case_usd=worst, last_instance_ended_at="2026-10-20T22:00:00.000Z", paper=True
    )
    assert report["headline"]["fork_by_episode"] == {
        "ep-1": {"requests": 2, "known_usd": "0.02", "unknown": 0, "booked_usd": "0.02"},
    }
    h = report["headline"]
    assert Decimal(h["usd"]) == want and h["bound"] == "="
    assert Decimal(h["usd_per_instance"]) == want / 4
    assert sum(Decimal(v["booked_usd"]) for v in h["by_lane"].values()) == want
    assert set(h["by_lane"]) == {"actor", "writer", "fork", "helper", "embed"}
    assert (
        h["writer_post_stream_usd"] == "0.0213" and h["writer_post_stream_passes"] == 1
    )
    assert h["labels"]["paper"] == M.PAPER_NOTE and "post_stream" in h["labels"]
    text = M.render_report(report)
    assert (
        "post_stream" in text
        and "Per entry:" in text
        and "Per item:" in text
        and M.PAPER_NOTE in text
    )
    with pytest.raises(ValueError):
        M.run_report(home, {"librarian": lanes["actor"]}, 1, worst_case_usd={"librarian": Decimal(1)})  # no lane
    # a lane with no worst case is refused: there is no default to book an unpriced call at
    with pytest.raises(ValueError, match="no worst case per request for lane"):
        M.run_report(home, lanes, 4, worst_case_usd={"writer": Decimal("13.125")})


def test_calls_in_a_pass_carry_its_ids_only_and_calls_outside_it_carry_none(
    monkeypatch,
):
    """Rule 7: a call is booked to the lane of the agent that makes it; only the writer service's pass (and the tasks
    it starts) sets the pass's attribution, so an actor's answer to the writer is never the pass's.
    """
    from unify.common import llm_client as L

    state = L._RequestMetadata()
    client = SimpleNamespace(origin="memory_writer")
    assert L.PASS_HEADER not in L._request_metadata_headers(client, state, [])
    entries = ["a" * 16, "b" * 16]
    with L.attribution("c" * 16, entries) as current:
        L.add_attributed_entry("d" * 16)
        got = L._request_metadata_headers(client, state, [])
        assert got[L.PASS_HEADER] == "c" * 16 and got[L.ENTRIES_HEADER] == ".".join(
            ["a" * 16, "b" * 16, "d" * 16]
        )
        assert current.entries[-1] == "d" * 16
    assert L.PASS_HEADER not in L._request_metadata_headers(client, state, [])
    # ids only: anything that is not a 16-hex id is never sent
    with L.attribution("not an id", ["x y", "e" * 16]):
        got = L._request_metadata_headers(client, state, [])
        assert L.PASS_HEADER not in got and got[L.ENTRIES_HEADER] == "e" * 16


def test_per_visit_rows_never_take_writer_cost_and_label_the_librarian(monkeypatch):
    """Rule 3, and the failure modes: writer or helper cost in a per-visit row; the librarian counted twice."""
    from unify.memory.integration import cost

    monkeypatch.setattr(cost, "_metering", lambda: True)
    listener = cost.CostListener()
    listener.activate()
    for origin in ("CodeActActor.act", "memory_librarian", "memory_writer", "memory_writer.x", "memory_helper",
                   "unify_embed"):  # fmt: skip
        listener(
            SimpleNamespace(
                origin=origin, request={"model": "m"}, response={}, provider_cost=0.001
            )
        )
    assert [r.purpose for r in listener.rows] == ["actor", "librarian", "embedding"]
    monkeypatch.setattr(cost, "_metering", lambda: False)
    listener.activate()
    listener(
        SimpleNamespace(
            origin="memory_librarian",
            request={"model": "m"},
            response={},
            provider_cost=0.001,
        )
    )
    assert [r.purpose for r in listener.rows] == [
        "actor"
    ]  # as built, with metering off


def test_a_pass_row_names_its_entries_items_usd_and_a_killed_turn_once(tmp_path):
    """The row's side: a killed turn is one count on its pass's row; the per-pass row carries the metering
    fields."""
    home = _home(tmp_path, passes=[], ledger=[
        {"op": "hand", "entry": "e1", "pass_id": "p9"},
        {"op": "hand", "entry": "e2", "pass_id": "p9", "by": "take"},
        {"op": "read", "entry": "e1", "pass_id": "p9"},
        {"op": "close", "entry": "e1", "pass_id": "p9", "decision": "new_item", "items": ["fn:new"]},
        {"op": "return", "entry": "e2", "pass_id": "p9", "why": "handed_not_read"},
    ])  # fmt: skip
    outcome = SimpleNamespace(items_merged=["fn:new", "fn:old"], commit="abc", pool=None, input_tokens=100,
                              cache_read_tokens=40, output_tokens=7, reasons=[])  # fmt: skip
    # the attributes the service's pass report (consolidate.AttentionPassReport) carries
    report = SimpleNamespace(pass_id="p9", ended="landed", outcome=outcome, usd_known="0.02", unknown_calls=1,
                             worst_case_booked="13.125", killed_turns=["p9/3"], turns=6, seconds=1.5,
                             reconciled=None, read=[], returned=[], helper_usd_known="0.005",
                             helper_unknown_calls=0)  # fmt: skip
    row = M.pass_row(home, report)
    assert row["entries"] == {
        "handed": ["e1"],
        "taken": ["e2"],
        "read": ["e1"],
        "closed": ["e1"],
        "returned": ["e2"],
    }
    assert row["items"] == {
        "landed": ["fn:new", "fn:old"],
        "changed": [],
        "retired": [],
        "superseded": [],
    }
    assert row["usd"] == {
        "known": "0.02",
        "unknown_calls": 1,
        "worst_case_booked": "13.125",
        "killed_turns": 1,
        # the helpers' lane's part of the known USD and the unpriced calls
        "helper_known": "0.005",
        "helper_unknown_calls": 0,
    }
    assert (
        row["tokens"] == {"prompt": 100, "cached": 40, "completion": 7}
        and row["turns"] == 6
    )
    assert set(row["messages"]) == {"sent", "received"}


@pytest.mark.timeout(300)
def test_through_the_service_a_killed_turn_is_counted_once_and_the_shares_sum_to_the_pass(
    tmp_path, monkeypatch
):
    """A3 tests 1 and 4 through the writer service: a pass killed mid-turn and resumed has one killed turn on its
    row and one killed-turn event, and the views sum to the pass's known USD."""
    from tests.memory import test_writer_service as T
    from unify.memory import writer_service as ws

    for k, v in {
        **T.SWITCHES,
        "UNIFY_MEMORY_METERING": "on",
        "UNIFY_REQUEST_METADATA_HEADERS": "on",
    }.items():
        monkeypatch.setenv(k, v)
    home, out = tmp_path / "home", tmp_path / "out"
    monkeypatch.setenv("DRIVER_OUT", str(out))
    monkeypatch.setenv("DRIVER_BLOCK_AT", "2")
    T._open_entry(home, 2)
    over = {"UNIFY_MEMORY_METERING": "on", "UNIFY_REQUEST_METADATA_HEADERS": "on"}
    assert T._ensure(home, over=over).status == "started"
    T._until(lambda: (out / "blocked").exists())
    doc = ws.read_state(home)
    os.killpg(doc["pgid"], signal.SIGKILL)
    T._until(lambda: not ws.owns(doc))
    assert T._ensure(home, over=over).status == "started"
    row = T._until(lambda: next((r for r in M.metered_passes(home)), None))
    try:
        assert (
            row["usd"]["killed_turns"] == 1 and len(T._events(home, "killed-turn")) == 1
        )
        entries = _rows(M.metering_dir(home) / M.ENTRIES_FILE)
        assert sum(Decimal(r["share_usd"]) for r in entries) == Decimal(
            row["usd"]["known"]
        )
    finally:
        ws.stop_service(home, "operator", "done", grace_s=0.5, term_s=2)
