"""Spec v2.1 §6 (P7 Task 6): a pass runs under its wall-clock bound; cancellation goes through the abort path and
keeps the work as a draft; the calls are reconciled from the Sol lane's journal window (Amendment D) before the
pass is settled."""

from __future__ import annotations

import asyncio
import json
import time
from decimal import Decimal

import pytest

from unify.memory_v2.integration import async_pass as ap
from unify.memory_v2.sol_pass import _Spend
from unify.memory_v2.trigger import PassRequest
from tests.memory_v2.test_sol_pass import _call, _sol, needs_bwrap
from tests.memory_v2.test_sol_v21_tools import EPS


class _Out:
    def __init__(self, passed):
        self.passed, self.commit = passed, "c" * 40 if passed else None


@pytest.mark.asyncio
async def test_wall_bound_cancels_through_the_abort_path():
    seen = []

    async def slow():
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            seen.append("abort path ran")
            raise

    sup = ap.Supervisor(0.2, asyncio.Event())
    started = time.monotonic()
    out = await sup(slow)
    assert (
        out.ended == "deadline" and out.outcome is None and seen == ["abort path ran"]
    )
    assert time.monotonic() - started < 5 and sup.last is out


@pytest.mark.asyncio
async def test_sigterm_cancels_and_a_pass_that_ends_is_returned():
    stop = asyncio.Event()
    sup = ap.Supervisor(60, stop)
    asyncio.get_running_loop().call_later(0.1, stop.set)
    assert (await sup(lambda: asyncio.sleep(3600))).ended == "cancelled"

    async def done(passed):
        return _Out(passed)

    assert (
        await ap.Supervisor(60, asyncio.Event())(lambda: done(True))
    ).ended == "landed"
    assert (
        await ap.Supervisor(60, asyncio.Event())(lambda: done(False))
    ).ended == "refused"

    async def broken():
        raise RuntimeError("x")

    assert (await ap.Supervisor(60, asyncio.Event())(broken)).ended == "error"
    assert (
        ap.Supervisor(2700, asyncio.Event()).pass_deadline_s
        == 2700 - ap.MERGE_RESERVE_S
    )


@pytest.mark.asyncio
async def test_the_window_is_the_journal_range_the_pass_ran_in(tmp_path):
    j = tmp_path / "costs.jsonl"
    j.write_text('{"origin": "request_started"}\n')
    before = j.stat().st_size

    async def writes():
        with open(j, "a") as fh:
            fh.write('{"origin": "response"}\n')
        return _Out(True)

    sup = ap.Supervisor(60, asyncio.Event(), journal=str(j), models=("m",))
    await sup(writes)
    assert sup.window == (before, j.stat().st_size)
    assert (await sup.reconcile(budget_s=0))[
        "calls"
    ] == 0  # no Sol-lane request started


def test_state_view_reports_what_a_pass_cleared(tmp_path):
    from unify.memory_v2.integration.state import State

    State(tmp_path / "state.json", drift={"a", "b"}, suspect={"a", "c"}).save()
    view = ap.StateView.load(tmp_path / "state.json")
    view.drift.difference_update({"a"})
    view.suspect.difference_update({"a"})
    assert view.cleared() == (["a"], ["a"])
    assert State.load(tmp_path / "state.json").drift == {
        "a",
        "b",
    }  # the worker never saves state


class _Blocking:
    """A model turn that writes one function and a manifest, then never answers again."""

    def __init__(self):
        self.n = 0

    async def __call__(self, messages, tools):
        self.n += 1
        if self.n > 1:
            await asyncio.Event().wait()
        code = (
            "import json, os\n"
            "os.makedirs('/memory/text', exist_ok=True)\n"
            "open('/memory/text/extra.py', 'w').write('def more(x):\\n    \"\"\"More.\"\"\"\\n    return x\\n')\n"
            "os.makedirs('/memory/.pass', exist_ok=True)\n"
            "json.dump({'items': [{'item': 'memory.text.extra:more', 'kind': 'function'}]},"
            " open('/memory/.pass/manifest.json', 'w'))\n"
        )
        return _call("c1", "execute_code", {"code": code}), "0.001"


async def _cancelled_pass(tmp_path):
    _, ev, sol = _sol(tmp_path, _Blocking(), v21=True)
    sol.load = EPS.__getitem__
    # the brief is not under test here (it reads P2's and P4's constants; test_sol_v21_brief.py pins it)
    sol._brief_v21 = lambda: "brief"
    req = PassRequest("incremental", "svc", ["e1", "e2"], False)
    task = asyncio.ensure_future(sol.run(req, "p9"))
    for _ in range(600):
        live = getattr(sol, "_live", None)
        if live is not None and (live[0] / "text/extra.py").exists():
            break
        await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    return ev, sol


@needs_bwrap
@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_a_cancelled_pass_keeps_its_work_as_a_draft(tmp_path):
    ev, sol = await _cancelled_pass(tmp_path)
    patch_blob, refused, passed = ev.db.execute(
        "SELECT patch_blob, items_refused, passed FROM passes WHERE pass_id='p9'",
    ).fetchone()
    assert patch_blob and b"text/extra.py" in sol.gate.blobs.get(patch_blob)
    assert json.loads(refused) == {"memory.text.extra:more": ["cancelled"]}
    assert passed == 0
    assert sol.mem.head() == sol.mem.head(
        "main",
    )  # nothing landed: main is where it was


@needs_bwrap
@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_a_cancelled_pass_round_row_holds_the_cancelled_result(tmp_path):
    """Needs P2's pass_rounds (k) and P5's drafts: runs once they are integrated."""
    drafts = pytest.importorskip("unify.memory_v2.drafts")
    ev, sol = await _cancelled_pass(tmp_path)
    rows = ev.pass_rounds("write")
    assert rows[-1]["pass_id"] == "p9" and rows[-1]["rounds"] is None
    assert b"cancelled before the gate ran" in sol.gate.blobs.get(rows[-1]["gate_blob"])
    assert drafts.draft_states(rows)["p9"] == "open"


def test_v2_records_a_cancelled_pass_as_before(tmp_path):
    """Off the v2.1 switch, the failed row is v2's byte for byte: no patch and no refused items."""
    _, ev, sol = _sol(tmp_path, _Blocking())
    sol._cancel_patch = ("x" * 64, ["memory.text.extra:more"])  # ignored: v21 is off
    sol._fail(
        PassRequest("incremental", "svc", [], False),
        "p0",
        "0" * 40,
        _Spend(),
        ["r"],
    )
    row = ev.db.execute(
        "SELECT patch_blob, items_refused FROM passes WHERE pass_id='p0'",
    ).fetchone()
    assert tuple(row) == (None, "{}")


def _supervised_driver(tmp_path, monkeypatch, journal=None):
    from tests.memory_v2.integration import test_consolidate as tc
    from unify.memory_v2.integration import consolidate

    class _SlowSol:
        def __init__(self, *a, **k):
            pass

        async def run(self, req, pass_id):
            await asyncio.sleep(3600)

        def transcript(self, pass_id):
            return []

    monkeypatch.setattr(consolidate, "SolPass", _SlowSol)
    monkeypatch.setattr(consolidate, "unillm_turn", lambda *a, **k: None)
    stores = tc._stores(tmp_path)
    sha, _ = tc._record(stores, "e1")
    settings = tc._settings(e=1)
    settings.UNIFY_MEMORY_V21, settings.UNIFY_MEMORY_V21_E = "on", 1
    sup = ap.Supervisor(
        0.3,
        asyncio.Event(),
        journal=journal,
        models=("openai/gpt-6-sol",),
    )
    return consolidate, stores, sha, settings, sup


async def _drive(consolidate, stores, sha, settings, sup):
    return await consolidate.run_due_passes(
        stores,
        "e1",
        sha,
        ap.StateView.load(stores.paths.state),
        effort="low",
        settings=settings,
        emit=None,
        supervise=sup,
    )


def _end(consolidate, stores):
    lines = consolidate.events_path(stores.paths).read_text().splitlines()
    return [json.loads(x) for x in lines][-1]


def _ledger(consolidate, stores):
    lines = consolidate.ledger_path(stores.paths).read_text().splitlines()
    return [json.loads(x) for x in lines]


@pytest.mark.asyncio
async def test_run_due_passes_under_the_supervisor_ends_at_the_bound_and_books_the_worst_case(
    monkeypatch,
    tmp_path,
):
    j = tmp_path / "costs.jsonl"
    j.write_text("")
    consolidate, stores, sha, settings, sup = _supervised_driver(
        tmp_path,
        monkeypatch,
        journal=str(j),
    )
    loop = asyncio.get_running_loop()

    def sol_calls():  # the proxy journals two Sol requests while the pass runs; one is never priced
        rows = [
            {
                "origin": "request_started",
                "request_attempt_id": "a",
                "requested_model": "openai/gpt-6-sol",
            },
            {
                "origin": "response",
                "request_attempt_id": "a",
                "requested_model": "openai/gpt-6-sol",
                "status": "completed",
                "account_charge": "0.40",
            },
            {
                "origin": "request_started",
                "request_attempt_id": "b",
                "requested_model": "openai/gpt-6-sol",
            },
            {
                "origin": "transport_error",
                "request_attempt_id": "b",
                "requested_model": "openai/gpt-6-sol",
                "status": "cancelled",
            },
        ]
        with open(j, "a") as fh:
            fh.writelines(json.dumps(r) + "\n" for r in rows)

    loop.call_later(0.1, sol_calls)
    out = await _drive(consolidate, stores, sha, settings, sup)
    assert out == [] and sup.last.ended == "deadline"
    end = _end(consolidate, stores)
    assert end["phase"] == "end" and end["reason_codes"] == ["deadline"]
    assert end["ended"] == "deadline"
    assert end["usd"] == "0.40" and end["unknown_cost_calls"] == 1
    assert end["reconciled"]["journal"] == "read" and end["reconciled"]["calls"] == 2
    assert end["reconciled"]["booked_usd"] == "13.525"
    settle = _ledger(consolidate, stores)[-1]
    assert settle["phase"] == "settle" and settle["usd"] == "0.40"
    assert settle["unknown_cost_calls"] == 1 and settle["unknown_usd_each"] == "13.125"
    # the run guard books the unpriced call at the worst case of one Sol call, not at the per-call reserve
    assert consolidate.committed_sol_usd(stores) == Decimal("13.525")


@pytest.mark.asyncio
async def test_with_no_journal_the_pass_is_not_settled_and_its_whole_cap_stays_committed(
    monkeypatch,
    tmp_path,
):
    consolidate, stores, sha, settings, sup = _supervised_driver(tmp_path, monkeypatch)
    assert await _drive(consolidate, stores, sha, settings, sup) == []
    end = _end(consolidate, stores)
    assert end["ended"] == "deadline" and end["reconciled"]["journal"] == "unavailable"
    assert end["usd"] == "unknown" and end["unknown_cost_calls"] is None
    ledger = _ledger(consolidate, stores)
    assert [r["phase"] for r in ledger] == ["reserve"]
    assert consolidate.committed_sol_usd(stores) == Decimal(ledger[0]["cap_usd"])


@pytest.mark.asyncio
async def test_a_due_curate_runs_after_write_in_the_same_worker_and_served_is_published_last(
    monkeypatch,
    tmp_path,
):
    """MAIN's ruling (9 Oct; spec §7): WRITE, its item records, the CURATE it made due in the same supervised
    slot, CURATE's records, then ``served`` published last; each pass keeps its own supervised run.
    """
    from types import SimpleNamespace

    from tests.memory_v2.integration import test_consolidate as tc
    from unify.memory_v2 import memory_writer
    from unify.memory_v2.integration import consolidate
    from unify.memory_v2.sol_pass import PassOutcome

    order = []

    class _QuickSol:
        def __init__(self, *a, **k):
            pass

        async def run(self, req, pass_id):
            order.append((req.kind, pass_id))
            return PassOutcome(pass_id, False, None, "0", 0, ["nothing to merge"])

        def transcript(self, pass_id):
            return []

    due = iter([SimpleNamespace(suspects={}, fired={"index": "index over its view"})])
    monkeypatch.setattr(consolidate, "SolPass", _QuickSol)
    monkeypatch.setattr(consolidate, "unillm_turn", lambda *a, **k: None)
    monkeypatch.setattr(
        consolidate,
        "_curate_state",
        lambda stores, lookup: next(due, None),
    )
    monkeypatch.setattr(
        consolidate,
        "after_passes",
        lambda stores, settings, recorded, emit: order.append(("records",)),
    )
    real = memory_writer.publish
    monkeypatch.setattr(
        memory_writer,
        "publish",
        lambda mem, *a, **k: order.append(("publish",)) or real(mem, *a, **k),
    )
    stores = tc._stores(tmp_path)
    sha, _ = tc._record(stores, "e1")
    settings = tc._settings(e=1)
    settings.UNIFY_MEMORY_V21, settings.UNIFY_MEMORY_V21_E = "on", 1
    sup = ap.Supervisor(30.0, asyncio.Event(), models=("openai/gpt-6-sol",))
    out = await _drive(consolidate, stores, sha, settings, sup)
    write = next(o for o in order if len(o) == 2)
    assert write[1] == "e1.p0" and write[0] != "curate"
    assert order[order.index(write) + 1 :] == [
        ("records",),
        ("curate", "e1.p1"),
        ("records",),
        ("publish",),
    ]
    assert (
        len(out) == 2 and len(sup.runs) == 2
    )  # each pass its own supervised run and reconciliation


@pytest.mark.asyncio
async def test_a_stopped_worker_starts_no_curate(monkeypatch, tmp_path):
    from types import SimpleNamespace

    from tests.memory_v2.integration import test_consolidate as tc
    from unify.memory_v2.integration import consolidate
    from unify.memory_v2.sol_pass import PassOutcome

    stop = asyncio.Event()
    kinds = []

    class _StoppingSol:
        def __init__(self, *a, **k):
            pass

        async def run(self, req, pass_id):
            kinds.append(req.kind)
            stop.set()  # SIGTERM arrives during WRITE, after its last call
            return PassOutcome(pass_id, False, None, "0", 0, [])

        def transcript(self, pass_id):
            return []

    monkeypatch.setattr(consolidate, "SolPass", _StoppingSol)
    monkeypatch.setattr(consolidate, "unillm_turn", lambda *a, **k: None)
    monkeypatch.setattr(
        consolidate,
        "_curate_state",
        lambda stores, lookup: SimpleNamespace(suspects={}, fired={"index": "x"}),
    )
    stores = tc._stores(tmp_path)
    sha, _ = tc._record(stores, "e1")
    settings = tc._settings(e=1)
    settings.UNIFY_MEMORY_V21, settings.UNIFY_MEMORY_V21_E = "on", 1
    sup = ap.Supervisor(30.0, stop, models=("openai/gpt-6-sol",))
    await _drive(consolidate, stores, sha, settings, sup)
    assert "curate" not in kinds and len(sup.runs) == 1
