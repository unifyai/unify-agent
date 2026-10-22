"""The writer service's spend: a service pass is bounded by the service's room (its cap less what its other
passes committed), for the writer's calls and every helper's, so a pass never takes the service past its cap; and the
helpers' lane, which the writer journal does not hold, is settled toward the service's committed total.

Keyless: every model is scripted.
"""

from __future__ import annotations

import asyncio
import json
import os
from decimal import Decimal
from types import SimpleNamespace

from tests.memory import test_helpers as th
from tests.memory.test_helpers import helper_boxes  # noqa: F401 - the helpers' scripted box
from tests.memory.test_writer_pass import Turns, _call, _run, _writer, needs_bwrap
from unify.memory import attention, writer_pass
from unify.memory import writer_service as ws
from unify.memory.call_bounds import call_worst_case_usd
from unify.memory.writer_pass import STEP_GUARD


#: The default writer model's bound in the pinned catalogue
W = call_worst_case_usd("openai/gpt-6-sol")


def _cfg(limit, worst=W):
    return SimpleNamespace(usd_limit=None if limit is None else str(limit), usd_call_worst=str(worst))


def test_the_room_counts_known_spend_and_each_unpriced_in_flight_and_new_call_at_the_worst_case():
    spend = SimpleNamespace(usd=Decimal("0.70"), unknown=0, in_flight=0)
    assert writer_pass.usd_room(_cfg(None), spend)
    assert writer_pass.usd_room(_cfg(Decimal("0.70") + W), spend)  # the new call at the worst case fits
    assert not writer_pass.usd_room(_cfg(Decimal("0.69") + W), spend)
    spend.unknown = 1  # an unpriced call books a whole writer call
    assert not writer_pass.usd_room(_cfg(Decimal("0.70") + W), spend)
    assert writer_pass.usd_room(_cfg(Decimal("0.70") + 2 * W), spend)
    spend.in_flight = 1  # so does a call still running (a helper's, say)
    assert not writer_pass.usd_room(_cfg(Decimal("0.70") + 2 * W), spend)
    assert writer_pass.usd_room(_cfg(Decimal("0.70") + 3 * W), spend)
    with writer_pass.call_in_flight(spend):
        assert spend.in_flight == 2
    assert spend.in_flight == 1
    # a limit with no worst case to count with lets no call start (fail closed)
    assert not writer_pass.usd_room(SimpleNamespace(usd_limit="1000", usd_call_worst=None), spend)
    # with a cheaper model's worst case (0.65625) the same room holds far more
    spend.unknown, spend.in_flight = 0, 0
    assert writer_pass.usd_room(_cfg(Decimal("0.70") + Decimal("0.65625"), Decimal("0.65625")), spend)


@needs_bwrap
def test_a_service_pass_stops_before_a_call_that_could_take_it_past_its_room(tmp_path):
    """Each call costs 0.30 and the room is 14.00: a call starts only while the spend plus one call at the worst case
    (13.125) fits, so three calls (0.90), then no fourth (0.90 + 13.125 > 14.00). The pass's spend never exceeds its
    room, and the pass says why it ended."""
    model = Turns(
        [_call(f"c{i}", "execute_code", {"code": f"print({i})"}) for i in range(20)],
        usd="0.30",
    )
    _mem, _ev, writer = _writer(
        tmp_path,
        model,
        deadline_s=300.0,
        usd_limit="14.00",
        usd_call_worst=str(W),
    )
    out = _run(writer)
    assert Decimal(out.usd) == Decimal("0.90") <= Decimal("14.00")
    assert any("room of 14.00 USD" in r for r in out.reasons), out.reasons


@needs_bwrap
def test_without_a_room_an_uncapped_pass_is_as_built(tmp_path):
    model = Turns(
        [_call(f"c{i}", "execute_code", {"code": f"print({i})"}) for i in range(5)],
        usd="0.30",
    )
    _mem, _ev, writer = _writer(tmp_path, model, max_calls=STEP_GUARD, deadline_s=300.0)
    out = _run(writer)
    assert Decimal(out.usd) > Decimal("1.00")  # every scripted cell ran: no USD stop
    assert not any("room of" in r for r in out.reasons)


def test_a_helper_stops_before_a_call_past_the_passes_room(tmp_path, helper_boxes):
    """Helpers share the pass's spend: a helper starts no call once the pass's room is used (0.001 a call, room one
    worst-case call plus 0.0015: two calls, then it stops)."""
    script = th.Script(th._drafting(), usd="0.001")
    spend = SimpleNamespace(usd=Decimal("0"), unknown=0, in_flight=0)
    cfg = _cfg(W + Decimal("0.0015"))
    ctx = th._ctx(tmp_path, script, spend=spend, usd_room=lambda: writer_pass.usd_room(cfg, spend))
    th._run({"task": "draft area", "episodes": ["e1"]}, ctx)
    assert spend.usd == Decimal("0.002") and spend.in_flight == 0
    (row,) = ctx.records
    assert "room for this pass reached" in row["status"], row["status"]


def test_the_room_is_the_cap_less_the_other_passes_commitments(tmp_path):
    home = tmp_path / "home"
    settings = SimpleNamespace(UNIFY_MEMORY_CAP_USD="10")
    for row in (
        {"op": "reserve", "pass_id": "a", "usd": "4"},
        {"op": "settle", "pass_id": "a", "usd": "1.5"},
        {"op": "reserve", "pass_id": "b", "usd": "3"},  # still running or killed: its whole reservation
        {"op": "reserve", "pass_id": "me", "usd": "4"},  # this pass's own reservation is its room's, not a debt
    ):
        ws._append(ws.service_dir(home) / ws.SPEND_FILE, {"v": 1, **row})
    assert ws.pass_room(home, "me", settings) == Decimal("5.5")
    ws._append(ws.service_dir(home) / ws.SPEND_FILE, {"v": 1, "op": "reserve", "pass_id": "c", "usd": "9"})
    assert ws.pass_room(home, "me", settings) == Decimal(0)  # never below 0
    # rev 2: memory's cap always applies; unset, it is the default (switch.MEMORY_CAP_DEFAULT, 20.00): 20 - 13.5 committed
    from unify.memory.integration import switch

    assert ws.pass_room(home, "me", SimpleNamespace()) == Decimal(switch.MEMORY_CAP_DEFAULT) - Decimal("13.5")


def test_the_helpers_spend_and_their_lost_turns_count(tmp_path):
    from unify.memory.integration import consolidate

    writer = SimpleNamespace(
        _helper_rows=[
            {"usd_known": "0.25", "unknown_cost_calls": 0},
            {"usd_known": "0.10", "unknown_cost_calls": 2},
        ],
    )
    # a lost helper call is counted once, from its pending turn file (turns_cost at the resume), never here
    lost = [SimpleNamespace(killed=["h1/3"]), SimpleNamespace(killed=[])]
    assert consolidate._helper_spend(writer, lost) == (Decimal("0.35"), 2)


def test_a_service_pass_is_bounded_by_the_room_and_settles_the_helpers_lane(tmp_path, monkeypatch):
    """Through consolidate.run_attention_pass: the pass's config carries the service's room (cap 30 less another
    pass's 7 committed) as its USD stop; its settle is the writer journal's booking plus the helpers' (priced 0.35 and
    one unpriced call at the worst case), so helper spend counts toward the service's committed total."""
    from tests.memory.integration import test_consolidate as tc
    from unify.memory.integration import consolidate
    from unify.memory.integration.async_pass import Supervisor

    stores = tc._stores(tmp_path)
    tc._record(stores, "e1")
    home = stores.paths.home
    ws._append(ws.service_dir(home) / ws.SPEND_FILE, {"v": 1, "op": "reserve", "pass_id": "other", "usd": "7"})
    ledger = attention.Ledger(home)
    ledger.create("message", attention.Refs(record_seq=1), 1)
    handed = ledger.hand("ab" * 8, 4)
    seen = []

    class Spy(consolidate.WriterPass):
        def __init__(self, *a, **k):
            seen.append(a[5])
            super().__init__(*a, **k)

    monkeypatch.setattr(consolidate, "WriterPass", Spy)

    async def recon(self, **kw):
        return {"journal": "read", "usd": "0.02", "unknown": 0, "worst_case_usd": "13.125", "booked_usd": "0.02"}

    monkeypatch.setattr(Supervisor, "reconcile", recon)
    monkeypatch.setattr(consolidate, "_helper_spend", lambda writer, turns: (Decimal("0.35"), 1))
    settings = tc._settings()
    settings.UNIFY_MEMORY_CAP_USD = "30"
    report = asyncio.run(
        consolidate.run_attention_pass(
            home,
            "ab" * 8,
            handed,
            resume=False,
            folder=ws.service_dir(home) / ws.TURNS_DIR / ("ab" * 8),
            journal_start=None,
            effort="low",
            settings=settings,
            turn_factory=lambda: tc.FakeWriter()("m", "low"),
        ),
    )
    (cfg,) = seen
    assert Decimal(cfg.usd_limit) == Decimal(23) and not hasattr(cfg, "usd_call_reserve")
    # rev 2.1: no per-pass USD or call cap, with no writer lane as with one: the run's spend cap (its room, above), the
    # wall clock and the operational step guard bound the pass
    assert not hasattr(cfg, "max_usd") and cfg.max_calls == STEP_GUARD
    assert Decimal(report.settle_usd) == Decimal("0.02") + Decimal("0.35") + W
    assert report.usd_known == "0.37" and report.unknown_calls == 1
    assert report.helper_usd_known == "0.35" and report.helper_unknown_calls == 1


def _attention_pass(tmp_path, monkeypatch, *, recon, resume=False, pending=0, helpers=None, done=None, cap="30",
                    model=None, journal_start=None):  # fmt: skip
    """One run_attention_pass with a fake writer that finishes at once (0.01 USD), the service's cap 30 with another
    pass's 7 committed; returns (the pass's config, its report)."""
    from tests.memory.integration import test_consolidate as tc
    from unify.memory.integration import consolidate
    from unify.memory.integration.async_pass import Supervisor

    stores = tc._stores(tmp_path)
    tc._record(stores, "e1")
    home = stores.paths.home
    ws._append(ws.service_dir(home) / ws.SPEND_FILE, {"v": 1, "op": "reserve", "pass_id": "other", "usd": "7"})
    ledger = attention.Ledger(home)
    ledger.create("message", attention.Refs(record_seq=1), 1)
    handed = ledger.hand("cd" * 8, 4)
    folder = ws.service_dir(home) / ws.TURNS_DIR / ("cd" * 8)
    folder.mkdir(parents=True, exist_ok=True)
    for n in range(pending):  # calls a killed run left pending (in another conversation than this one's)
        (folder / f"{50 + n}.json").write_text('{"state": "pending", "turn": %d, "digest": "x"}' % (50 + n))
    for name, doc in (done or {}).items():  # a killed run's done calls
        (folder / name).parent.mkdir(parents=True, exist_ok=True)
        (folder / name).write_text(json.dumps(doc))
    seen = []

    class Spy(consolidate.WriterPass):
        def __init__(self, *a, **k):
            seen.append(a[5])
            super().__init__(*a, **k)

    monkeypatch.setattr(consolidate, "WriterPass", Spy)

    asked = []

    async def fake_recon(self, **kw):
        asked.append(kw)
        return recon(kw) if callable(recon) else recon

    monkeypatch.setattr(Supervisor, "reconcile", fake_recon)
    if helpers is not None:
        monkeypatch.setattr(consolidate, "_helper_spend", lambda writer, turns: helpers)
    settings = tc._settings(**({"model": model} if model else {}))
    settings.UNIFY_MEMORY_CAP_USD = cap
    report = asyncio.run(
        consolidate.run_attention_pass(
            home, "cd" * 8, handed, resume=resume, folder=folder, journal_start=journal_start, effort="low",
            settings=settings, turn_factory=lambda: tc.FakeWriter()("m", "low"),
        ),
    )  # fmt: skip
    (cfg,) = seen
    report.asked = asked
    return cfg, report


def test_a_resumed_pass_has_its_lost_calls_taken_from_its_room(tmp_path, monkeypatch):
    """On resume, what the killed run already spent comes off the room: its done calls replay with their
    USD (so they count in the pass's spend again), and each call it lost is taken off at the worst case."""
    read = {"journal": "read", "usd": "0", "unknown": 0, "worst_case_usd": "13.125", "booked_usd": "0"}
    cfg, _ = _attention_pass(tmp_path, monkeypatch, recon=read, resume=True, pending=1)
    assert Decimal(cfg.usd_limit) == Decimal(23) - W


def test_with_no_journal_a_pass_settles_its_own_figures_at_the_worst_case(tmp_path, monkeypatch):
    """With no readable journal the pass is not left unsettled (its whole room booked for good): it
    settles its own priced calls (the writer's and its helpers', 0.01 here) and each unpriced or lost call at the
    worst case."""
    from unify.memory.reconcile import unavailable

    _cfg_, report = _attention_pass(tmp_path, monkeypatch, recon=unavailable(W))
    assert Decimal(report.settle_usd) == Decimal("0.01") and report.unknown_calls == 0


def _step(home, runner, monkeypatch, **over):
    from tests.memory import test_writer_service as T
    from unify.memory.integration import request

    monkeypatch.setattr(request, "actor_effort", lambda: "low")
    return ws.step(home, settings=T._settings(**over), runner=runner)


def test_the_service_books_the_whole_room_until_the_pass_settles(tmp_path, monkeypatch):
    """The pass may spend the whole room, so the whole room is booked: cap 100 with 40 settled by an
    earlier pass books 60 for the new pass, and its settle then counts instead."""
    from tests.memory import test_writer_service as T

    home = tmp_path / "home"
    for row in ({"op": "reserve", "pass_id": "a", "usd": "60"}, {"op": "settle", "pass_id": "a", "usd": "40"}):
        ws._append(ws.service_dir(home) / ws.SPEND_FILE, {"v": 1, **row})
    T._open_entry(home)
    runner = T.Recording(decide=True)
    got = _step(home, runner, monkeypatch)
    rows = ws._rows(ws.service_dir(home) / ws.SPEND_FILE)
    mine = [r for r in rows if r.get("pass_id") == got["pass_id"]]
    assert [(r["op"], r["usd"]) for r in mine] == [("reserve", "60"), ("settle", "0.001")]
    assert ws.spend_totals(home)[0] == Decimal("40.001")


def test_a_cap_whose_room_holds_no_worst_case_call_starts_no_pass(tmp_path, monkeypatch):
    from tests.memory import test_writer_service as T

    home = tmp_path / "home"
    ws._append(ws.service_dir(home) / ws.SPEND_FILE, {"v": 1, "op": "settle", "pass_id": "a", "usd": "90"})
    T._open_entry(home)
    got = _step(home, T._no_pass, monkeypatch)
    assert got["why_idle"] == "spend_cap" and got["pass_id"] is None


def test_a_pass_that_raises_settles_from_its_turn_files(tmp_path, monkeypatch):
    """A pass that ended with no report is settled from its own turn files (each done call's USD, each
    pending or unpriced call at the worst case), never left at a reservation a later room would ignore."""
    from tests.memory import test_writer_service as T

    home = tmp_path / "home"
    T._open_entry(home)

    async def broken(home_, pass_id, handed, *, resume, folder, journal_start, effort, settings, stop=None):
        at = ws.stamp(ws._now())
        (folder / "0.json").write_text(json.dumps({"state": "done", "usd": "0.5", "turn": 0, "at": at}))
        (folder / "h0").mkdir()
        (folder / "h0" / "0.json").write_text(json.dumps({"state": "done", "usd": "unknown", "turn": 0, "at": at}))
        (folder / "1.json").write_text(json.dumps({"state": "pending", "turn": 1, "at": at}))
        raise RuntimeError("the pass broke")

    got = _step(home, broken, monkeypatch)
    settle = [r for r in ws._rows(ws.service_dir(home) / ws.SPEND_FILE) if r["op"] == "settle"]
    assert got["ended"] == "error" and [r["usd"] for r in settle] == [str(Decimal("0.5") + 2 * W)]
    folder = ws.service_dir(home) / ws.TURNS_DIR / got["pass_id"]
    assert ws.turns_spend(folder, W) == Decimal("0.5") + 2 * W
    assert ws.turns_cost(folder) == {"writer": (Decimal("0.5"), 1), "helpers": (Decimal(0), 1)}


def test_a_step_that_raises_leaves_the_service_stopped_and_the_home_free(tmp_path, monkeypatch):
    """Review: a step whose turn raises leaves service.json stopped (never running with its caller's group) and the
    lock free."""
    from tests.memory import test_writer_service as T

    home = tmp_path / "home"
    T._open_entry(home)

    def boom(self):
        raise RuntimeError("the ledger broke")

    monkeypatch.setattr(attention.Ledger, "sync", boom)
    try:
        _step(home, T._no_pass, monkeypatch)
    except RuntimeError:
        pass
    else:  # pragma: no cover
        raise AssertionError("the step did not raise")
    doc = ws.read_state(home)
    assert doc["state"] == "stopped" and doc["members"] == []
    fd = ws.try_lock(ws.service_dir(home) / ws.LOCK_FILE)
    assert fd is not None
    os.close(fd)


# --- one call's worst case by model (the pinned catalogue cbb212a9) ---------------------------------------------------


def test_each_models_worst_case_is_the_pinned_catalogues_bound_and_an_unknown_model_is_refused():
    from unify.memory import call_bounds

    assert call_bounds.call_worst_case_usd("openai/gpt-6-luna") == Decimal("0.65625")
    assert call_bounds.call_worst_case_usd("openai/gpt-6-sol") == Decimal("13.125")
    assert call_bounds.call_worst_case_usd("openai/gpt-6-luna@openrouter") == Decimal("0.65625")
    assert call_bounds.worst_of(["openai/gpt-6-luna", "openai/gpt-6-sol"]) == Decimal("13.125")
    for model in ("vendor/no-such-model", "", "openai/gpt-6-luna-but-not"):
        try:
            call_bounds.call_worst_case_usd(model)
        except call_bounds.NoBound:
            continue
        raise AssertionError(f"{model!r} was bounded")
    # the table is the one computed from the catalogue the guards pin
    import json

    assert json.loads(call_bounds.TABLE.read_text())["catalogue_sha256"] == call_bounds.CATALOGUE_SHA256


def test_a_service_pass_counts_calls_at_its_lanes_models_worst_case(monkeypatch):
    """The writer's model's bound, which its helpers run on too (with or without the helper lane)."""
    from pydantic import SecretStr

    from tests.memory import test_writer_service as T
    from unify.memory.integration import switch

    cheap = T._settings(UNIFY_MEMORY_WRITER_MODEL="openai/gpt-6-luna", UNIFY_MODEL="openai/gpt-6-sol")
    monkeypatch.setattr(switch, "helper_lane_route", lambda: (None, "no helper lane"))
    assert ws.pass_call_worst_case(cheap) == Decimal("0.65625")
    monkeypatch.setattr(
        switch, "helper_lane_route", lambda: (("http://127.0.0.1:9/helper/v1", SecretStr("t-for-a-test")), None)
    )
    # rev 2: the helpers run on the writer's model (switch.writer_model), so the lane changes nothing
    assert ws.pass_call_worst_case(cheap) == Decimal("0.65625")
    unknown = T._settings(UNIFY_MEMORY_WRITER_MODEL="vendor/no-such-model")
    monkeypatch.setattr(switch, "helper_lane_route", lambda: (None, "no helper lane"))
    try:
        ws.pass_call_worst_case(unknown)
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("an unbounded model was given a worst case")


def test_a_cheaper_writer_runs_passes_under_a_cap_below_the_default_models_worst_case(tmp_path, monkeypatch):
    """With the cheaper writer model a cap of 5 USD holds a pass (0.65625 a call), where the default's 13.125 would hold
    none."""
    from tests.memory import test_writer_service as T
    from unify.memory.integration import switch

    monkeypatch.setattr(switch, "helper_lane_route", lambda: (None, "no helper lane"))
    home = tmp_path / "home"
    T._open_entry(home)
    runner = T.Recording(decide=True)
    got = _step(home, runner, monkeypatch, UNIFY_MEMORY_WRITER_MODEL="openai/gpt-6-luna",
                UNIFY_MEMORY_CAP_USD="5")  # fmt: skip
    assert got["pass_id"] is not None and len(runner.passes) == 1
    dear = tmp_path / "default-model"
    T._open_entry(dear)
    got = _step(dear, T._no_pass, monkeypatch, UNIFY_MEMORY_CAP_USD="5")
    assert got["why_idle"] == "spend_cap"


def test_a_writer_model_with_no_catalogue_bound_runs_no_pass(tmp_path, monkeypatch):
    from tests.memory import test_writer_service as T
    from unify.memory.integration import switch

    monkeypatch.setattr(switch, "helper_lane_route", lambda: (None, "no helper lane"))
    home = tmp_path / "home"
    T._open_entry(home)
    got = _step(home, T._no_pass, monkeypatch, UNIFY_MEMORY_WRITER_MODEL="vendor/no-such-model")
    assert got["why_idle"] == "refused" and got["pass_id"] is None
    (reason,) = [e["reason"] for e in ws.events(home) if e.get("event") == "refused"]
    assert "UNIFY_MEMORY_WRITER_MODEL" in reason and "is not one the pinned price catalogue bounds" in reason
    assert "no-such-model" not in reason


def test_a_resumed_pass_that_diverges_keeps_its_killed_runs_spend(tmp_path, monkeypatch):
    """Review: a resumed run whose conversation differs from the killed run's overwrites that run's turn files, so
    their USD would drop out. The killed run's cost is read at the resume, before any file is written: it comes off
    the room (2.50 priced by the writer, 0.25 by a helper, one lost call at the worst case) and is settled with this
    run's figures; a done call replayed from its file costs nothing again."""
    from unify.memory.reconcile import unavailable

    reply = {"role": "assistant", "content": "earlier"}
    done = {
        "0.json": {"state": "done", "turn": 0, "digest": "not-this-conversation", "reply": reply, "usd": "2.50"},
        "h0/0.json": {"state": "done", "turn": 0, "digest": "x", "reply": reply, "usd": "0.25"},
    }
    cfg, report = _attention_pass(tmp_path, monkeypatch, recon=unavailable(W), resume=True, pending=1, done=done,
                                  cap="60")  # fmt: skip
    prior = Decimal("2.50") + Decimal("0.25") + W
    assert Decimal(cfg.usd_limit) == Decimal(53) - prior
    # this run's own call (0.01, the writer diverged and made it again) plus the killed run's
    assert Decimal(report.settle_usd) == Decimal("0.01") + prior
    assert Decimal(report.usd_known) == Decimal("2.76") and report.unknown_calls == 1
    assert report.helper_usd_known == "0.25"


def test_a_replayed_done_call_costs_nothing_again(tmp_path):
    made = []

    async def model(messages, tools):
        made.append(1)
        return {"role": "assistant", "content": "r"}, "0.40"

    folder = tmp_path / "turns" / "p1"
    first = ws.ResumableTurn(model, folder)
    msgs = [{"role": "user", "content": "go"}]
    assert asyncio.run(first(msgs, []))[1] == "0.40"
    again = ws.ResumableTurn(model, folder)
    assert asyncio.run(again(msgs, []))[1] == "0" and made == [1] and again.replayed == 1
    assert ws.turns_cost(folder)["writer"] == (Decimal("0.40"), 0)


def test_a_resumed_pass_whose_killed_run_used_its_room_makes_no_call(tmp_path, monkeypatch):
    """The same killed run under cap 30: its 15.875 leaves 7.125, less than one call at the worst case, so this run
    makes no call and settles the killed run's spend alone."""
    from unify.memory.reconcile import unavailable

    reply = {"role": "assistant", "content": "earlier"}
    done = {"0.json": {"state": "done", "turn": 0, "digest": "x", "reply": reply, "usd": "2.50"},
            "h0/0.json": {"state": "done", "turn": 0, "digest": "x", "reply": reply, "usd": "0.25"}}  # fmt: skip
    cfg, report = _attention_pass(tmp_path, monkeypatch, recon=unavailable(W), resume=True, pending=1, done=done)
    prior = Decimal("2.50") + Decimal("0.25") + W
    assert Decimal(cfg.usd_limit) == Decimal(23) - prior < W
    assert Decimal(report.settle_usd) == prior
    assert any("room of" in n for n in report.notes), report.notes


# --- the journal books at the writer's model's bound (review: no 13.125 default) ---------------------------------------


def _read(kw):
    """A journal window with one unpriced writer call, booked at the worst case the service passed."""
    from unify.memory.reconcile import money

    worst = kw["worst_case_usd"]
    return {"journal": "read", "usd": "0", "unknown": 1, "worst_case_usd": money(worst), "booked_usd": money(worst)}


def test_an_unpriced_writer_call_of_a_model_bounded_above_the_default_settles_at_its_own_bound(tmp_path, monkeypatch):
    pro = call_worst_case_usd("openai/gpt-5.5-pro")
    assert pro == Decimal("220.5") > W
    cfg, report = _attention_pass(tmp_path, monkeypatch, recon=_read, cap="1000", model="openai/gpt-5.5-pro")
    assert [kw["worst_case_usd"] for kw in report.asked] == [pro]
    assert Decimal(report.settle_usd) == pro and Decimal(cfg.usd_call_worst) == pro


def _ledger_rows(stores):
    from unify.memory.integration.consolidate import ledger_path

    path = ledger_path(stores.paths)
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


# --- a refusal for now is tried again; a refusal the route made before sending the call on costs nothing -------------


def _busy() -> writer_pass.WriterCallError:
    return writer_pass.WriterCallError("writer call failed: APIStatusError (http_403_busy)", writer_pass.REFUSED_BUSY)


class RefusedFirst:
    """A model whose first *n* calls the route refuses with *error*, then *model*'s answers."""

    def __init__(self, model, n=1, error=_busy):
        self.model, self.n, self.error, self.refused, self.answered = model, n, error, 0, 0

    async def __call__(self, messages, tools):
        if self.refused < self.n:
            self.refused += 1
            raise self.error()
        self.answered += 1
        return await self.model(messages, tools)


@needs_bwrap
def test_a_writer_call_refused_while_another_is_in_flight_is_tried_again_and_the_pass_goes_on(tmp_path, monkeypatch):
    """The key proxy's one-call-at-a-time refusal ends nothing: the call is made again after a short wait and the
    pass runs to its end; the refused call cost nothing (no unpriced call booked at the worst case)."""
    monkeypatch.setattr(writer_pass, "TRANSIENT_BACKOFF_S", 0.01)
    scripted = Turns([_call(f"c{i}", "execute_code", {"code": f"print({i})"}) for i in range(3)], usd="0.30")
    model = RefusedFirst(scripted)
    _mem, _ev, writer = _writer(
        tmp_path, model, max_calls=STEP_GUARD, deadline_s=300.0, usd_limit="100.00", usd_call_worst=str(W)
    )
    out = _run(writer)
    assert model.refused == 1 and model.answered >= 4  # the three scripted calls and the finish, after the refusal
    assert not any("model call failed" in r for r in out.reasons), out.reasons
    assert any("refused for now (http_403_busy)" in r for r in out.reasons), out.reasons
    assert out.unknown_cost_calls == 0, out.unknown_cost_calls
    assert Decimal(out.usd) == Decimal("0.30") * model.answered, (out.usd, model.answered)


def test_transient_failures_are_tried_again_a_bounded_number_of_times_each_booked_by_what_it_may_have_cost():
    async def scenario(errors, room=None, remaining=1000.0):
        booked, waits, notes = [], [], []
        queue = list(errors)

        async def call():
            if queue:
                raise queue.pop(0)()
            return "answer"

        async def sleep(s):
            waits.append(s)

        try:
            got = await writer_pass.call_retrying(
                call,
                book_unpriced=lambda: booked.append(1),
                remaining=lambda: remaining,
                room=room,
                note=notes.append,
                sleep=sleep,
            )
        except writer_pass.WriterCallError as exc:
            got = exc
        return got, len(booked), waits, notes

    def server_error():
        return writer_pass.WriterCallError("writer call failed: X (http_503)", "http_503")

    def cap():
        return writer_pass.WriterCallError("writer call failed: X (http_403_cap)", writer_pass.REFUSED_CAP)

    # busy twice, then answered: nothing booked (the route refused it unsent), two waits that double
    got, booked, waits, notes = asyncio.run(scenario([_busy, _busy]))
    assert (got, booked, waits, len(notes)) == ("answer", 0, [2.0, 4.0], 2)
    # a server error may have cost money: each retried attempt is booked as one unpriced call
    got, booked, waits, _ = asyncio.run(scenario([server_error]))
    assert (got, booked, waits) == ("answer", 1, [2.0])
    # never more than TRANSIENT_RETRIES retries in a row: the last error is raised, not yet booked
    got, booked, waits, _ = asyncio.run(scenario([server_error] * 9))
    assert isinstance(got, writer_pass.WriterCallError) and not got.counted
    assert booked == writer_pass.TRANSIENT_RETRIES and waits == [2.0, 4.0, 8.0, 16.0]
    # the cap is not retried
    got, booked, waits, _ = asyncio.run(scenario([cap]))
    assert got.category == writer_pass.REFUSED_CAP and (booked, waits) == (0, [])
    # no wait past the pass's deadline
    got, booked, waits, _ = asyncio.run(scenario([_busy], remaining=1.0))
    assert isinstance(got, writer_pass.WriterCallError) and waits == []
    # with the failed attempt booked, the room holds no more call: raised, marked as booked already
    got, booked, waits, _ = asyncio.run(scenario([server_error], room=lambda: False))
    assert got.counted and booked == 1 and waits == []
    assert not writer_pass.unbooked_cost(got) and not writer_pass.unbooked_cost(_busy())
    assert writer_pass.unbooked_cost(server_error())


