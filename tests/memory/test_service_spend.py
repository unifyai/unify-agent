"""The writer service's spend (review C1): a service pass is bounded by the service's room (its cap less what its other
passes committed), for the writer's calls and every helper's, so a pass never takes the service past its cap; and the
helpers' lane, which the writer journal does not hold, is settled toward the service's committed total.

Keyless: every model is scripted.
"""

from __future__ import annotations

import asyncio
import os
from decimal import Decimal
from types import SimpleNamespace

from tests.memory import test_helpers as th
from tests.memory.test_helpers import fake_box  # noqa: F401 - the helpers' scripted box
from tests.memory.test_writer_pass import Turns, _call, _run, _writer, needs_bwrap
from unify.memory import attention, writer_pass
from unify.memory import writer_service as ws
from unify.memory.reconcile import WRITER_CALL_WORST_CASE_USD


W = WRITER_CALL_WORST_CASE_USD


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
    # with Luna's worst case (0.65625) the same room holds far more
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
        library_mode=True,
        max_usd=None,
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
    _mem, _ev, writer = _writer(tmp_path, model, library_mode=True, max_usd=None, deadline_s=300.0)
    out = _run(writer)
    assert Decimal(out.usd) > Decimal("1.00")  # every scripted cell ran: no USD stop
    assert not any("room of" in r for r in out.reasons)


def test_a_helper_stops_before_a_call_past_the_passes_room(tmp_path, fake_box):
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
    settings = SimpleNamespace(UNIFY_MEMORY_WRITER_SERVICE_CAP_USD="10")
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
    assert ws.pass_room(home, "me", SimpleNamespace()) is None


def test_the_helpers_spend_and_their_lost_turns_count(tmp_path):
    from unify.memory.integration import consolidate

    writer = SimpleNamespace(
        _helper_rows=[
            {"usd_known": "0.25", "unknown_cost_calls": 0},
            {"usd_known": "0.10", "unknown_cost_calls": 2},
        ],
    )
    lost = [SimpleNamespace(killed=["h1/3"]), SimpleNamespace(killed=[])]
    assert consolidate._helper_spend(writer, lost) == (Decimal("0.35"), 3)


def test_a_service_pass_is_bounded_by_the_room_and_settles_the_helpers_lane(tmp_path, monkeypatch):
    """Through consolidate.run_attention_pass: the pass's config carries the service's room (cap 30 less another
    pass's 7 committed) as its USD stop; its settle is the writer journal's booking plus the helpers' (priced 0.35 and
    one unpriced call at the worst case), so helper spend counts toward the service's committed total."""
    from tests.memory.integration import test_consolidate as tc
    from unify.memory.integration import consolidate
    from unify.memory.integration.async_pass import Supervisor

    tc._no_records(monkeypatch)
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
    settings = tc._settings_library()
    settings.UNIFY_MEMORY_WRITER_SERVICE_CAP_USD = "30"
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
    assert cfg.usd_limit == "23" and not hasattr(cfg, "usd_call_reserve")
    assert report.settle_usd == str(Decimal("0.02") + Decimal("0.35") + WRITER_CALL_WORST_CASE_USD)
    assert report.usd_known == "0.37" and report.unknown_calls == 1
    assert report.helper_usd_known == "0.35" and report.helper_unknown_calls == 1


def _attention_pass(tmp_path, monkeypatch, *, recon, resume=False, pending=0, helpers=None):
    """One run_attention_pass with a fake writer that finishes at once (0.01 USD), the service's cap 30 with another
    pass's 7 committed; returns (the pass's config, its report)."""
    from tests.memory.integration import test_consolidate as tc
    from unify.memory.integration import consolidate
    from unify.memory.integration.async_pass import Supervisor

    tc._no_records(monkeypatch)
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
    seen = []

    class Spy(consolidate.WriterPass):
        def __init__(self, *a, **k):
            seen.append(a[5])
            super().__init__(*a, **k)

    monkeypatch.setattr(consolidate, "WriterPass", Spy)

    async def fake_recon(self, **kw):
        return recon

    monkeypatch.setattr(Supervisor, "reconcile", fake_recon)
    if helpers is not None:
        monkeypatch.setattr(consolidate, "_helper_spend", lambda writer, turns: helpers)
    settings = tc._settings_library()
    settings.UNIFY_MEMORY_WRITER_SERVICE_CAP_USD = "30"
    report = asyncio.run(
        consolidate.run_attention_pass(
            home, "cd" * 8, handed, resume=resume, folder=folder, journal_start=None, effort="low",
            settings=settings, turn_factory=lambda: tc.FakeWriter()("m", "low"),
        ),
    )  # fmt: skip
    (cfg,) = seen
    return cfg, report


def test_a_resumed_pass_has_its_lost_calls_taken_from_its_room(tmp_path, monkeypatch):
    """Review C1: on resume, what the killed run already spent comes off the room: its done calls replay with their
    USD (so they count in the pass's spend again), and each call it lost is taken off at the worst case."""
    read = {"journal": "read", "usd": "0", "unknown": 0, "worst_case_usd": "13.125", "booked_usd": "0"}
    cfg, _ = _attention_pass(tmp_path, monkeypatch, recon=read, resume=True, pending=1)
    assert Decimal(cfg.usd_limit) == Decimal(23) - W


def test_with_no_journal_a_pass_settles_its_own_figures_at_the_worst_case(tmp_path, monkeypatch):
    """Review C1: with no readable journal the pass is not left unsettled (its whole room booked for good): it
    settles its own priced calls (the writer's and its helpers', 0.01 here) and each unpriced or lost call at the
    worst case."""
    from unify.memory.reconcile import unavailable

    _cfg_, report = _attention_pass(tmp_path, monkeypatch, recon=unavailable())
    assert Decimal(report.settle_usd) == Decimal("0.01") and report.unknown_calls == 0


def _step(home, runner, monkeypatch, **over):
    from tests.memory import test_writer_service as T
    from unify.memory.integration import request

    monkeypatch.setattr(request, "actor_effort", lambda: "low")
    return ws.step(home, settings=T._settings(**over), runner=runner)


def test_the_service_books_the_whole_room_until_the_pass_settles(tmp_path, monkeypatch):
    """Review C1: the pass may spend the whole room, so the whole room is booked: cap 100 with 40 settled by an
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
    """Review C1: a pass that ended with no report is settled from its own turn files (each done call's USD, each
    pending or unpriced call at the worst case), never left at a reservation a later room would ignore."""
    from tests.memory import test_writer_service as T

    home = tmp_path / "home"
    T._open_entry(home)

    async def broken(home_, pass_id, handed, *, resume, folder, journal_start, effort, settings, stop=None):
        (folder / "0.json").write_text('{"state": "done", "usd": "0.5", "turn": 0}')
        (folder / "h0").mkdir()
        (folder / "h0" / "0.json").write_text('{"state": "done", "usd": "unknown", "turn": 0}')
        (folder / "1.json").write_text('{"state": "pending", "turn": 1}')
        raise RuntimeError("the pass broke")

    got = _step(home, broken, monkeypatch)
    settle = [r for r in ws._rows(ws.service_dir(home) / ws.SPEND_FILE) if r["op"] == "settle"]
    assert got["ended"] == "error" and [r["usd"] for r in settle] == [str(Decimal("0.5") + 2 * W)]
    assert ws.turns_spend(ws.service_dir(home) / ws.TURNS_DIR / got["pass_id"]) == Decimal("0.5") + 2 * W


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
    """The writer's model's bound, and the helpers' when the helper lane is set (the larger of the two)."""
    from pydantic import SecretStr

    from tests.memory import test_writer_service as T
    from unify.memory.integration import switch

    luna = T._settings(UNIFY_MEMORY_WRITER_MODEL="openai/gpt-6-luna", UNIFY_MODEL="openai/gpt-6-sol")
    monkeypatch.setattr(switch, "helper_lane_route", lambda: (None, "no helper lane"))
    assert ws.pass_call_worst_case(luna) == Decimal("0.65625")
    monkeypatch.setattr(
        switch, "helper_lane_route", lambda: (("http://127.0.0.1:9/helper/v1", SecretStr("t-for-a-test")), None)
    )
    assert ws.pass_call_worst_case(luna) == Decimal("13.125")  # the helpers run on the actor's Sol
    unknown = T._settings(UNIFY_MEMORY_WRITER_MODEL="vendor/no-such-model")
    monkeypatch.setattr(switch, "helper_lane_route", lambda: (None, "no helper lane"))
    try:
        ws.pass_call_worst_case(unknown)
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("an unbounded model was given a worst case")


def test_a_luna_cell_runs_passes_under_a_cap_below_sols_worst_case(tmp_path, monkeypatch):
    """With a Luna writer a cap of 5 USD holds a pass (0.65625 a call), where Sol's 13.125 would hold none."""
    from tests.memory import test_writer_service as T
    from unify.memory.integration import switch

    monkeypatch.setattr(switch, "helper_lane_route", lambda: (None, "no helper lane"))
    home = tmp_path / "home"
    T._open_entry(home)
    runner = T.Recording(decide=True)
    got = _step(home, runner, monkeypatch, UNIFY_MEMORY_WRITER_MODEL="openai/gpt-6-luna",
                UNIFY_MEMORY_WRITER_SERVICE_CAP_USD="5")  # fmt: skip
    assert got["pass_id"] is not None and len(runner.passes) == 1
    sol = tmp_path / "sol"
    T._open_entry(sol)
    got = _step(sol, T._no_pass, monkeypatch, UNIFY_MEMORY_WRITER_SERVICE_CAP_USD="5")
    assert got["why_idle"] == "spend_cap"


def test_a_writer_model_with_no_catalogue_bound_runs_no_pass(tmp_path, monkeypatch):
    from tests.memory import test_writer_service as T
    from unify.memory.integration import switch

    monkeypatch.setattr(switch, "helper_lane_route", lambda: (None, "no helper lane"))
    home = tmp_path / "home"
    T._open_entry(home)
    got = _step(home, T._no_pass, monkeypatch, UNIFY_MEMORY_WRITER_MODEL="vendor/no-such-model")
    assert got["why_idle"] == "refused" and got["pass_id"] is None
    assert [e["reason"] for e in ws.events(home) if e.get("event") == "refused"] == ["NoBound"]
