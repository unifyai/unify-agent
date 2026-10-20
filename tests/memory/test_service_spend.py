"""The writer service's spend (review C1): a service pass is bounded by the service's room (its cap less what its other
passes committed), for the writer's calls and every helper's, so a pass never takes the service past its cap; and the
helpers' lane, which the writer journal does not hold, is settled toward the service's committed total.

Keyless: every model is scripted.
"""

from __future__ import annotations

import asyncio
from decimal import Decimal
from types import SimpleNamespace

from tests.memory import test_helpers as th
from tests.memory.test_helpers import fake_box  # noqa: F401 - the helpers' scripted box
from tests.memory.test_writer_pass import Turns, _call, _run, _writer, needs_bwrap
from unify.memory import attention, writer_pass
from unify.memory import writer_service as ws
from unify.memory.reconcile import WRITER_CALL_WORST_CASE_USD


def _cfg(limit, reserve="0"):
    return SimpleNamespace(usd_limit=limit, usd_call_reserve=reserve)


def test_the_room_counts_known_spend_unpriced_calls_at_the_worst_case_and_the_next_call():
    spend = SimpleNamespace(usd=Decimal("0.70"), unknown=0)
    assert writer_pass.usd_room(_cfg(None), spend)
    assert writer_pass.usd_room(_cfg("1.00", "0.30"), spend)  # 0.70 + 0.30 <= 1.00
    assert not writer_pass.usd_room(_cfg("1.00", "0.31"), spend)
    spend.unknown = 1  # one unpriced call books a whole writer call
    assert not writer_pass.usd_room(_cfg(str(WRITER_CALL_WORST_CASE_USD), "0"), spend)


@needs_bwrap
def test_a_service_pass_stops_before_a_call_that_could_take_it_past_its_room(tmp_path):
    """Each call costs 0.30, the room is 1.00 and a call reserves 0.25: three calls (0.90), then no fourth (0.90 +
    0.25 > 1.00). The pass's known spend never exceeds its room, and the pass says why it ended."""
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
        usd_limit="1.00",
        usd_call_reserve="0.25",
    )
    out = _run(writer)
    assert Decimal(out.usd) == Decimal("0.90") <= Decimal("1.00")
    assert any("room of 1.00 USD" in r for r in out.reasons), out.reasons


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
    """Helpers share the pass's spend: a helper starts no call once the pass's room is used (0.001 a call, room
    0.0025, 0.001 reserved a call: two calls, then it stops)."""
    script = th.Script(th._drafting(), usd="0.001")
    spend = SimpleNamespace(usd=Decimal("0"), unknown=0)
    cfg = _cfg("0.0025", "0.001")
    ctx = th._ctx(tmp_path, script, spend=spend, usd_room=lambda: writer_pass.usd_room(cfg, spend))
    th._run({"task": "draft area", "episodes": ["e1"]}, ctx)
    assert spend.usd == Decimal("0.002") <= Decimal("0.0025")
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
    """Through consolidate.run_attention_pass: the pass's config carries the service's room (cap 10 less another
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
    settings.UNIFY_MEMORY_WRITER_SERVICE_CAP_USD = "10"
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
    assert cfg.usd_limit == "3" and Decimal(cfg.usd_call_reserve) > 0
    assert report.settle_usd == str(Decimal("0.02") + Decimal("0.35") + WRITER_CALL_WORST_CASE_USD)
    assert report.usd_known == "0.37" and report.unknown_calls == 1
    assert report.helper_usd_known == "0.35" and report.helper_unknown_calls == 1
