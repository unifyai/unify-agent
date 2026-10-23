"""S10: memory's spend is booked under its cap from the costs the scripted models declare: the kinship judge's call
and the writer's passes in ``spend.jsonl`` and ``pass-results.jsonl``, the request-start librarian call in the
request's own cost rows (the actor's lane), never measured from a provider; and with a cap too small for one call,
no memory model call is made at all while the captures stay served.

Design: H1 (one knob plus the cap; the judge's spend under memory's cap), R2.1 (the judge's prompt names no id, alias
or "task"), R2.3/R2.4 (bounded and metered: reserve the worst case, then settle), "the request-start librarian call
is booked under the cell's cap" (post-audit), change h; AGENTS.md: money as decimal strings, unknown stays unknown.
"""

from __future__ import annotations

import json
import re
from decimal import Decimal

import pytest

from tests.actor.code_act.sandbox_world import needs_bwrap
from tests.memory.conformance import fake_lanes as FL
from tests.memory.conformance import harness as H
from tests.scripted_model import Always, ScriptedModel, reply, cell
from tests.system import sysworld as SW
from tests.system.drive import MEMORY_MODES

DECIMAL = re.compile(r"^[0-9]+(\.[0-9]+)?$")
LIB_TOKENS = 4321


def _rows_for(rows: list[dict], pid: str) -> list[dict]:
    return [r for r in rows if r.get("pass_id") == pid]


@needs_bwrap
@pytest.mark.asyncio
@pytest.mark.timeout(1800)
@pytest.mark.parametrize("mode", MEMORY_MODES)
async def test_the_judge_the_passes_and_the_start_block_are_booked_from_declared_costs_under_the_cap(system, mode):
    s = system
    s.writer.plan = "keep"
    r1 = await s.accepted_words(["stone"], "reverse")
    r2 = await s.accepted_tally()  # unrelated by the harness's relations: the judge is asked
    assert s.writer.judge_bodies, "the kinship judge was never asked"
    # ── the judge's prompt names no id, alias or "task" (R2.1) ──
    for body in s.writer.judge_bodies:
        text = json.dumps(body.get("messages"))
        assert not body.get("tools"), "the judge's call carries tools"
        assert not re.search(r"\btasks?\b", text, re.I), text[:2000]
        for eid in (r1.episode_id, r2.episode_id):
            assert eid not in text, eid
        assert not [c for c in s.captures() if c in text], "a capture id reached the judge"
    # ── its spend: a worst-case reserve, then a settle at the declared cost, both decimal strings ──
    spend = s.spend_rows()
    assert all(DECIMAL.match(str(r.get("usd"))) for r in spend if r.get("op") in ("reserve", "settle")), spend
    kin = [r for r in s.pass_results() if r.get("kind") == "kin"]
    assert kin, s.pass_results()
    kid = kin[0].get("pass_id") or kin[0].get("kin_id")
    rows = _rows_for(spend, kid)
    assert [r["op"] for r in rows] == ["reserve", "settle"], rows
    assert Decimal(rows[1]["usd"]) == Decimal(FL.CALL_USD), rows
    assert Decimal(str((kin[0].get("usd") or {}).get("known"))) == Decimal(FL.CALL_USD), kin[0]

    # ── a pass: its known USD is what the writer lane's calls for it declared ──
    librarian = Always(lambda c: reply(calls=[cell("tools.finish({'fits': [], 'nearest': []})\n")],
                                       prompt_tokens=LIB_TOKENS))
    r3 = await s.accepted_words(["glass"], "upper", librarian=librarian)
    att = [r for r in s.pass_results() if r.get("kind", "attention") == "attention" and r.get("pass_id")]
    assert att, s.pass_results()
    for p in att:
        calls = [c for c in s.lanes.of("writer") if c.meta.get("x-unify-pass-id") == p["pass_id"]]
        declared = Decimal(FL.CALL_USD) * len(calls)
        assert calls and Decimal(str(p["usd"]["known"])) == declared, (p["usd"], len(calls))
        pr = _rows_for(spend := s.spend_rows(), p["pass_id"])
        assert pr and pr[0]["op"] == "reserve" and any(r["op"] == "settle" for r in pr), pr
        assert Decimal([r for r in pr if r["op"] == "settle"][-1]["usd"]) >= declared, pr
    # ── never above the cap: what every pass and the judge reserved stays within it ──
    cap = Decimal(H.SERVICE_CAP_USD)
    assert all(Decimal(r["usd"]) <= cap for r in s.spend_rows() if r.get("op") == "reserve")
    # ── the start-block call: in the request's own cost rows, labelled librarian, as the model declared ──
    costs = [json.loads(x) for x in s.episode_file(r3.episode_id, "cost.jsonl").splitlines() if x.strip()]
    lib = [c for c in costs if c.get("purpose") == "librarian"]
    assert lib and lib[0].get("prompt_tokens") == LIB_TOKENS, costs
    assert all(c.get("usd") == "unknown" or DECIMAL.match(str(c.get("usd"))) for c in costs), costs
    assert not [r for r in s.spend_rows() if "librarian" in json.dumps(r)], "the actor-lane call is in memory's ledger"


@needs_bwrap
@pytest.mark.asyncio
@pytest.mark.timeout(1800)
@pytest.mark.parametrize("mode", ["on"])
async def test_a_cap_smaller_than_one_call_makes_no_memory_model_call_and_captures_stay_served(system, mode):
    s = system
    H.apply_settings(s.monkeypatch, {"UNIFY_MEMORY_CAP_USD": "0.000001"})
    a = s.capture_of((await s.accepted_words(["stone"], "reverse")).episode_id)
    await s.accepted_tally()
    b = s.capture_of((await s.accepted_words(["glass"], "upper")).episode_id)
    assert not s.lanes.of("writer"), [c.meta for c in s.lanes.of("writer")][:3]
    assert not [r for r in s.spend_rows() if r.get("op") in ("reserve", "settle")], s.spend_rows()
    sink: dict = {}
    await s.request(SW.words_request(["oak"]), ScriptedModel(
        actor=SW.probe_turns(SW.probe_json("_try('items', lambda: [i.id for i in memory.items()])\n"), sink),
        librarian=Always(lambda c: SW.librarian_answer([], nearest=[]))))
    assert {f"capture:{a}", f"capture:{b}"} <= set(sink["out"]["items"]), sink["out"]
