"""The consolidation driver (integration Task 24, v1): checker pass/fail, size trigger, Sol, gate, events.

Sol is a fake turn factory monkeypatched over ``consolidate.unillm_turn``: no network and no key. The
fake ends at once (``finish``) without a manifest, so no sandboxed cell runs; the tests marked
``needs_bwrap`` run the real gate.
"""

import asyncio
import json
import re
import shutil
from dataclasses import fields
from decimal import Decimal
from types import SimpleNamespace

import pytest

from unify.memory_v2.admission import cover_problem
from unify.memory_v2.episodes import Action, CostRow, EpisodeWriter
from unify.memory_v2.experience import experience_tokens
from unify.memory_v2.integration import consolidate
from unify.memory_v2.integration.consolidate import (
    EpisodeLookup,
    episode_costs,
    events_path,
    open_stores,
    post_checker,
    run_due_passes,
    sol_settings,
)
from unify.memory_v2.integration.paths import Paths
from unify.memory_v2.integration.state import State
from unify.memory_v2.qa import QAConfig
from unify.memory_v2.sol_pass import SOL_SYSTEM
from unify.memory_v2.redact import Redactor
from unify.memory_v2.signals import Signal
from unify.memory_v2.trigger import BATCHED_CURSOR
from tests.memory_v2.test_episodes import _ep

needs_bwrap = pytest.mark.skipif(
    shutil.which("bwrap") is None,
    reason="bubblewrap required",
)

A_TOK = "0.00000073"
MONEY = re.compile(r"^[0-9]+(\.[0-9]+)?\Z")
CSV = b"vendor_id,invoice_no,amount,due_date,status\nV-17,INV-0042,1250.50,2026-10-14,open\n"


def _settings(e=1, guard="", model="openai/gpt-6-sol", usage="", scale="", calls=""):
    return SimpleNamespace(
        UNIFY_MEMORY_V2="on",
        UNIFY_MEMORY_V2_E=e,
        UNIFY_MEMORY_V2_SOL_MODEL=model,
        UNIFY_MEMORY_V2_SOL_ALLOWANCE_USD_PER_TOKENS=A_TOK,
        UNIFY_MEMORY_V2_SOL_RUN_GUARD_USD=guard,
        UNIFY_MEMORY_V2_SOL_USAGE=usage,
        UNIFY_MEMORY_V2_SOL_EFFORT_SCALE=scale,
        UNIFY_MEMORY_V2_SOL_MAX_CALLS=calls,
    )


def _finish():
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "f",
                "type": "function",
                "function": {"name": "finish", "arguments": '{"summary": "nothing"}'},
            },
        ],
    }


class FakeSol:
    """A turn factory: each turn ends the pass at once (no manifest), or follows ``script``."""

    def __init__(self, usd="0.01", script=None):
        self.usd, self.script = usd, script
        self.made: list[tuple[str, str]] = []
        self.calls = 0
        self.seen: list[list[dict]] = []

    def __call__(self, model, effort, **kw):
        self.made.append((model, effort))

        async def turn(messages, tools):
            self.calls += 1
            self.seen.append([dict(m) for m in messages])
            if self.script is not None:
                step = self.script[min(self.calls, len(self.script)) - 1]
                if isinstance(step, BaseException):
                    raise step
                if step == "sleep":
                    await asyncio.sleep(30)
                return step
            return _finish(), self.usd

        return turn


def _stores(tmp_path):
    return open_stores(Paths.under(tmp_path / "home"))


def _record(stores, eid, actions=None, request="Pay my Venmo friends back", minute=0):
    ep = _ep(
        episode_id=eid,
        started_at=f"2026-10-08T01:{minute:02d}:00Z",
        request=[request],
        actions=(
            actions
            if actions is not None
            else [
                Action(0, "venmo", "me", [], {}, {"user_id": "u-1"}, "ok", "read"),
            ]
        ),
        costs=[CostRow("actor", "openai/gpt-6-luna", 10, 2, "0.000123")],
    )
    sha = EpisodeWriter(stores.episodes, stores.blobs, Redactor()).write(ep)
    stores.evidence.index_episode(ep, sha)
    return sha, ep


def _run(stores, eid, sha, state=None, *, effort="low", emit=None, **kw):
    # low: scale 1 and 40 calls (the per-effort defaults), so a pass's cap is E x a_tok
    state = state if state is not None else State(stores.paths.state)
    return asyncio.run(
        run_due_passes(
            stores,
            eid,
            sha,
            state,
            effort=effort,
            settings=_settings(**kw),
            emit=emit,
        ),
    )


def _events(stores):
    p = events_path(stores.paths)
    return [json.loads(ln) for ln in p.read_text().splitlines()] if p.exists() else []


# --- (a) the checker -------------------------------------------------------------------------------------


def test_checker_posts_pass_or_fail_only(tmp_path):
    stores = _stores(tmp_path)
    sha1, _ = _record(stores, "e1")
    sha2, _ = _record(stores, "e2", minute=1)
    assert post_checker(stores, "e1", sha1, None, "2026-10-08T01:02:00Z") is False
    assert stores.episodes.notes(sha1) == []
    assert post_checker(stores, "e1", sha1, False, "2026-10-08T01:02:00Z") is True
    assert post_checker(stores, "e2", sha2, True, "2026-10-08T01:03:00Z") is True
    (line,) = stores.episodes.notes(sha1)
    note = json.loads(line)
    assert set(note) == {f.name for f in fields(Signal)}
    assert (note["source"], note["label"], note["refers_to"]) == (
        "checker",
        "fail",
        "e1",
    )
    assert note["regime"] == "dense" and note["signal_id"] == "e1.checker"
    assert json.loads(stores.episodes.notes(sha2)[0])["label"] == "pass"
    assert [s.label for s in stores.evidence.signals_for("e1")] == ["fail"]
    assert (
        post_checker(stores, "e1", sha1, "yes", "t") is False
    )  # only a bool is a verdict


def test_checker_posts_nothing_where_the_regime_masks_it(tmp_path):
    stores = _stores(tmp_path)
    ep = _ep(episode_id="e9", regime="implicit")
    sha = EpisodeWriter(stores.episodes, stores.blobs, Redactor()).write(ep)
    stores.evidence.index_episode(ep, sha)
    assert post_checker(stores, "e9", sha, True, "t") is False
    assert stores.episodes.notes(sha) == []


# --- the size trigger ------------------------------------------------------------------------------------


def test_size_trigger_fires_only_at_e(tmp_path, monkeypatch):
    fake = FakeSol()
    monkeypatch.setattr(consolidate, "unillm_turn", fake)
    stores = _stores(tmp_path)
    sha1, ep1 = _record(stores, "e1")
    t1 = experience_tokens(ep1)[0]
    ep2_tokens = experience_tokens(
        _ep(episode_id="e2", request=["Split the dinner bill"], actions=ep1.actions),
    )[0]
    e = t1 + ep2_tokens
    assert _run(stores, "e1", sha1, e=e) == []  # t1 < E
    assert fake.calls == 0 and _events(stores) == []
    sha2, ep2 = _record(stores, "e2", request="Split the dinner bill", minute=1)
    assert experience_tokens(ep2)[0] == ep2_tokens
    (out,) = _run(stores, "e2", sha2, e=e)  # t1 + t2 == E
    assert fake.calls == 1
    start = _events(stores)[0]
    assert start["trigger_tokens"] == e and start["episodes"] == ["e1", "e2"]
    assert stores.evidence.cursor(BATCHED_CURSOR) == stores.evidence.seq_of("e2")


def test_one_token_short_of_e_runs_nothing(tmp_path, monkeypatch):
    fake = FakeSol()
    monkeypatch.setattr(consolidate, "unillm_turn", fake)
    stores = _stores(tmp_path)
    sha, ep = _record(stores, "e1")
    assert _run(stores, "e1", sha, e=experience_tokens(ep)[0] + 1) == []
    assert fake.calls == 0


# --- (b) a pass without a manifest -----------------------------------------------------------------------


def test_pass_without_manifest_is_recorded_failed_and_the_cursor_advances(
    tmp_path,
    monkeypatch,
):
    fake = FakeSol()
    monkeypatch.setattr(consolidate, "unillm_turn", fake)
    stores = _stores(tmp_path)
    sha, _ = _record(stores, "e1")
    (out,) = _run(stores, "e1", sha)
    assert out.pass_id == "e1.p0" and not out.passed and "no manifest" in out.reasons
    row = stores.evidence.db.execute(
        "SELECT kind, channel, passed, reasons FROM passes WHERE pass_id='e1.p0'",
    ).fetchone()
    assert row[:3] == ("batched", None, 0) and "no manifest" in json.loads(row[3])
    assert stores.evidence.cursor(BATCHED_CURSOR) == stores.evidence.seq_of("e1")
    assert _run(stores, "e1", sha) == []  # nothing new since the pass
    assert fake.calls == 1


# --- (c) no channel filter -------------------------------------------------------------------------------


def test_kind_qualified_channels_take_part_in_the_pass(tmp_path, monkeypatch):
    fake = FakeSol()
    monkeypatch.setattr(consolidate, "unillm_turn", fake)
    stores = _stores(tmp_path)
    blob = stores.blobs.put(CSV)
    acts = [
        Action(
            0,
            "worktree:workspace",
            "read",
            ["ap/invoices.csv"],
            {},
            {
                "blob_before": blob,
                "blob_after": blob,
                "size": len(CSV),
                "shape": {"format": "csv"},
            },
            "ok",
            "read",
            kind="worktree",
        ),
        Action(
            0,
            "shell:git",
            "run",
            ["git status"],
            {},
            {"exit_code": 0, "tail": "clean"},
            "ok",
            kind="shell",
        ),
    ]
    sha, _ = _record(stores, "o1", actions=acts)
    assert stores.evidence.channels_of("o1") == ["shell:git", "worktree:workspace"]
    (out,) = _run(stores, "o1", sha)
    assert fake.calls == 1
    user = fake.seen[0][1]["content"]
    assert '"channel": null' in user and '"o1"' in user


# --- (d) and the run guard -------------------------------------------------------------------------------


def test_run_guard_below_one_cap_runs_nothing_and_leaves_the_pass_due(
    tmp_path,
    monkeypatch,
):
    fake = FakeSol()
    monkeypatch.setattr(consolidate, "unillm_turn", fake)
    stores = _stores(tmp_path)
    sha, _ = _record(stores, "e1")
    assert _run(stores, "e1", sha, guard="0.0000005") == []  # cap is 0.00000073
    assert fake.calls == 0
    (held,) = _events(stores)  # no pass started: one end event, no start event
    assert held["phase"] == "end" and held["reason_codes"] == ["run_guard"]
    assert (held["calls"], held["usd"], held["gate_passed"]) == (0, "0", False)
    assert stores.evidence.cursor(BATCHED_CURSOR) == 0
    (out,) = _run(stores, "e1", sha)  # without the guard, the pass still due runs
    assert fake.calls == 1


def test_run_guard_stops_the_second_pass(tmp_path, monkeypatch):
    cap = Decimal(A_TOK)  # E = 1
    fake = FakeSol(usd=format(cap * Decimal("0.6"), "f"))
    monkeypatch.setattr(consolidate, "unillm_turn", fake)
    stores = _stores(tmp_path)
    guard = format(cap * Decimal("1.5"), "f")
    sha1, _ = _record(stores, "e1")
    (first,) = _run(stores, "e1", sha1, guard=guard)
    assert first.usd == format(cap * Decimal("0.6"), "f")
    sha2, _ = _record(stores, "e2", minute=1)
    assert (
        _run(stores, "e2", sha2, guard=guard) == []
    )  # 0.6 cap committed + 1 cap > 1.5 cap
    assert fake.calls == 1
    events = _events(stores)
    assert [e["phase"] for e in events] == ["start", "end", "end"]
    assert events[1]["reason_codes"] == ["no_manifest"]
    assert events[2]["pass_id"] == "e2.p0" and events[2]["reason_codes"] == [
        "run_guard",
    ]
    assert stores.evidence.cursor(BATCHED_CURSOR) == stores.evidence.seq_of("e1")


def test_unpriced_calls_count_at_the_reserve_for_the_guard(tmp_path, monkeypatch):
    fake = FakeSol(usd="unknown")
    monkeypatch.setattr(consolidate, "unillm_turn", fake)
    stores = _stores(tmp_path)
    sha, _ = _record(stores, "e1")
    _run(stores, "e1", sha)
    reserve = Decimal(A_TOK) / consolidate.MAX_CALLS
    assert consolidate.committed_sol_usd(stores) == reserve


def test_the_guard_never_reads_reason_text(tmp_path, monkeypatch):
    monkeypatch.setattr(consolidate, "unillm_turn", FakeSol(usd="0.0000001"))
    stores = _stores(tmp_path)
    sha, _ = _record(stores, "e1")
    _run(stores, "e1", sha)
    stores.evidence.record_pass(
        {
            "pass_id": "forged",
            "passed": 0,
            "usd": "0",
            "reasons": json.dumps(["note: 999 unpriced calls"]),
        },
    )
    assert consolidate.committed_sol_usd(stores) == Decimal("0.0000001")


def test_a_pass_that_starts_but_never_settles_counts_its_whole_cap(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(consolidate, "unillm_turn", FakeSol(usd="0.0000001"))
    monkeypatch.setattr(
        consolidate,
        "_settle",
        lambda *a, **k: None,
    )  # the process died mid-pass
    stores = _stores(tmp_path)
    sha, _ = _record(stores, "e1")
    _run(stores, "e1", sha)
    assert consolidate.committed_sol_usd(stores) == Decimal(A_TOK)
    sha2, _ = _record(stores, "e2", minute=1)
    guard = format(Decimal(A_TOK) * Decimal("1.5"), "f")
    assert (
        _run(stores, "e2", sha2, guard=guard) == []
    )  # the reserved cap holds the guard


def test_an_empty_effort_runs_nothing_and_is_logged(tmp_path, monkeypatch):
    fake = FakeSol()
    monkeypatch.setattr(consolidate, "unillm_turn", fake)
    stores = _stores(tmp_path)
    sha, _ = _record(stores, "e1")
    assert _run(stores, "e1", sha, effort="  ") == []
    assert fake.calls == 0 and _events(stores) == []
    assert "effort" in stores.paths.errors.read_text()
    assert stores.evidence.cursor(BATCHED_CURSOR) == 0  # still due


# --- events ----------------------------------------------------------------------------------------------


def test_events_start_and_end_with_decimal_money_and_the_inherited_effort(
    tmp_path,
    monkeypatch,
):
    fake = FakeSol(usd="1E-7")
    monkeypatch.setattr(consolidate, "unillm_turn", fake)
    stores = _stores(tmp_path)
    sha, _ = _record(stores, "e1")
    got: list[dict] = []
    ticks = iter([100.0, 102.5])
    out = asyncio.run(
        run_due_passes(
            stores,
            "e1",
            sha,
            State(stores.paths.state),
            effort="medium",
            settings=_settings(e=1),
            emit=got.append,
            clock=lambda: next(ticks),
        ),
    )
    assert len(out) == 1 and fake.made == [("openai/gpt-6-sol", "medium")]
    start, end = got
    assert start == {
        "type": "consolidation",
        "phase": "start",
        "pass_id": "e1.p0",
        "trigger_tokens": start["trigger_tokens"],
        "episodes": ["e1"],
        "sol_model": "openai/gpt-6-sol",
        "sol_effort": "medium",
        "cap_usd": "0.00000146",  # E x a_tok x 2, the medium scale
        "max_calls": 80,
        "effort_scale": "2",
    }
    assert isinstance(start["trigger_tokens"], int) and start["trigger_tokens"] >= 1
    assert set(end) == {
        "type",
        "phase",
        "pass_id",
        "usd",
        "unknown_cost_calls",
        "calls",
        "checks",
        "seconds",
        "gate_passed",
        "items",
        "index_tokens",
        "reason_codes",
        "items_merged",
        "items_refused",
    }
    assert end["phase"] == "end" and end["pass_id"] == "e1.p0"
    assert end["usd"] == "0.0000001" and MONEY.match(end["usd"])
    assert (end["calls"], end["unknown_cost_calls"], end["seconds"]) == (1, 0, 2.5)
    assert end["checks"] == 0
    assert end["gate_passed"] is False and end["reason_codes"] == ["no_manifest"]
    assert end["items"] == 0 and isinstance(end["index_tokens"], int)
    assert end["items_merged"] == [] and end["items_refused"] == {}
    assert _events(stores) == got  # the same rows, always appended to the events file
    assert events_path(stores.paths) == stores.paths.state_dir / "events.jsonl"
    for line in events_path(stores.paths).read_text().splitlines():
        assert not re.search(r"[0-9][eE][-+]?[0-9]", line)


def test_cap_is_e_times_the_allowance_as_a_plain_decimal():
    cfg = sol_settings(_settings(e=150000))
    assert format(cfg.cap_usd, "f") == "0.10950000"
    assert cfg.run_guard_usd is None and cfg.model == "openai/gpt-6-sol"
    defaults = sol_settings(SimpleNamespace())
    assert (defaults.experience_budget, format(defaults.usd_per_token, "f")) == (
        150000,
        A_TOK,
    )
    with pytest.raises(ValueError):
        sol_settings(
            SimpleNamespace(UNIFY_MEMORY_V2_SOL_ALLOWANCE_USD_PER_TOKENS="7.3E-7"),
        )
    with pytest.raises(ValueError):
        sol_settings(SimpleNamespace(UNIFY_MEMORY_V2_SOL_RUN_GUARD_USD="1e1"))
    with pytest.raises(ValueError):
        sol_settings(SimpleNamespace(UNIFY_MEMORY_V2_E="0"))
    for zero in ("0", "0.0", "0.00000000"):
        with pytest.raises(ValueError):
            sol_settings(
                SimpleNamespace(UNIFY_MEMORY_V2_SOL_ALLOWANCE_USD_PER_TOKENS=zero),
            )
    assert sol_settings(_settings(guard="2.50")).run_guard_usd == Decimal("2.50")
    assert defaults.show_usage is False and cfg.show_usage is False
    assert sol_settings(_settings(usage="on")).show_usage is True
    assert sol_settings(_settings(usage="off")).show_usage is False
    with pytest.raises(ValueError):
        sol_settings(SimpleNamespace(UNIFY_MEMORY_V2_SOL_USAGE="yes"))


# --- per-effort pass limits (UNIFY_MEMORY_V2_SOL_EFFORT_SCALE, UNIFY_MEMORY_V2_SOL_MAX_CALLS) ------------------


def test_per_effort_limits_default_to_one_rule_for_every_bed():
    cfg = sol_settings(SimpleNamespace())
    assert cfg.effort_scale == {
        "low": Decimal("1"),
        "medium": Decimal("2"),
        "high": Decimal("5"),
    }
    assert cfg.max_calls == {"low": 40, "medium": 80, "high": 80}
    assert (
        sol_settings(_settings()).effort_scale == cfg.effort_scale
    )  # empty: the defaults
    assert format(sol_settings(_settings(e=150000)).cap_usd * 5, "f") == "0.54750000"
    custom = sol_settings(
        _settings(scale="high:4, Low:1.5,medium:3", calls="low:10,medium:20,high:30"),
    )
    assert custom.effort_scale == {
        "low": Decimal("1.5"),
        "medium": Decimal("3"),
        "high": Decimal("4"),
    }
    assert custom.max_calls == {"low": 10, "medium": 20, "high": 30}


@pytest.mark.parametrize(
    "scale",
    [
        "low:1,medium:2",  # an effort missing
        "low:1,medium:2,high:5,xhigh:9",  # an unknown effort
        "low:1,low:1,medium:2,high:5",  # an effort twice
        "low:0,medium:2,high:5",  # not positive
        "low:1,medium:-2,high:5",
        "low:1,medium:2,high:5e0",  # an exponent
        "low:1,medium:2,high:.5",
        "low=1,medium=2,high=5",
        "low:1;medium:2;high:5",
        "low:1,medium:2,high:",
        "5",
    ],
)
def test_a_bad_effort_scale_map_is_refused(scale):
    with pytest.raises(ValueError, match="UNIFY_MEMORY_V2_SOL_EFFORT_SCALE"):
        sol_settings(_settings(scale=scale))


@pytest.mark.parametrize(
    "calls",
    [
        "low:40,medium:80",
        "low:40,medium:80,high:80,max:9",
        "low:0,medium:80,high:80",
        "low:40,medium:80,high:1.5",
        "low:40,medium:80,high:-1",
        "low:40,medium:80,high:1e2",
        "low:40,medium:80,high:many",
        "80",
    ],
)
def test_a_bad_max_calls_map_is_refused(calls):
    with pytest.raises(ValueError, match="UNIFY_MEMORY_V2_SOL_MAX_CALLS"):
        sol_settings(_settings(calls=calls))


def _spy_configs(monkeypatch) -> list:
    seen: list = []
    real = consolidate.SolPass

    def spy(*args, **kwargs):
        seen.append(args[5])  # the PassConfig
        return real(*args, **kwargs)

    monkeypatch.setattr(consolidate, "SolPass", spy)
    return seen


def _ledger_rows(stores):
    return [
        json.loads(ln)
        for ln in consolidate.ledger_path(stores.paths).read_text().splitlines()
    ]


@pytest.mark.parametrize(
    "actor, scale, calls",
    [("low", "1", 40), ("medium", "2", 80), ("high", "5", 80)],
)
def test_an_actor_effort_sets_the_pass_cap_and_calls(
    tmp_path,
    monkeypatch,
    actor,
    scale,
    calls,
):
    from unify.memory_v2.integration import request as request_mod
    from unify.settings import SETTINGS

    monkeypatch.setattr(SETTINGS, "UNIFY_MEMORY_V2_SOL_EFFORT", "actor")
    effort = request_mod.sol_effort(
        actor,
    )  # Sol follows the actor's effort (the default)
    assert effort == actor
    fake = FakeSol()
    monkeypatch.setattr(consolidate, "unillm_turn", fake)
    configs = _spy_configs(monkeypatch)
    stores = _stores(tmp_path)
    sha, _ = _record(stores, "e1")
    got: list[dict] = []
    (out,) = _run(stores, "e1", sha, effort=effort, emit=got.append)
    cap = Decimal(A_TOK) * Decimal(scale)  # E = 1
    (cfg,) = configs
    assert (cfg.effort, cfg.max_calls, cfg.max_usd) == (actor, calls, cap)
    start = got[0]
    assert start["phase"] == "start" and start["sol_effort"] == actor
    assert (start["cap_usd"], start["max_calls"], start["effort_scale"]) == (
        format(cap, "f"),
        calls,
        scale,
    )
    reserve, settle = _ledger_rows(stores)
    assert reserve["phase"] == "reserve" and settle["phase"] == "settle"
    assert (reserve["cap_usd"], reserve["max_calls"], reserve["effort_scale"]) == (
        format(cap, "f"),
        calls,
        scale,
    )
    assert Decimal(reserve["per_call_usd"]) == cap / calls


def test_a_high_pass_is_capped_at_five_allowances_and_80_calls_at_e_150k(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(consolidate, "unillm_turn", FakeSol())
    configs = _spy_configs(monkeypatch)
    monkeypatch.setattr(
        consolidate.Trigger,
        "pass_budget_usd",
        lambda self: Decimal(150000) * Decimal(A_TOK),
    )
    stores = _stores(tmp_path)
    sha, _ = _record(stores, "e1")
    got: list[dict] = []
    _run(stores, "e1", sha, effort="high", emit=got.append)
    (cfg,) = configs
    assert format(cfg.max_usd, "f") == "0.54750000" and cfg.max_calls == 80
    assert got[0]["cap_usd"] == "0.54750000" and got[0]["effort_scale"] == "5"


def test_the_effort_maps_reach_the_pass(tmp_path, monkeypatch):
    monkeypatch.setattr(consolidate, "unillm_turn", FakeSol())
    configs = _spy_configs(monkeypatch)
    stores = _stores(tmp_path)
    sha, _ = _record(stores, "e1")
    got: list[dict] = []
    _run(
        stores,
        "e1",
        sha,
        effort="High",
        emit=got.append,
        scale="low:1,medium:2,high:3.5",
        calls="low:40,medium:80,high:60",
    )
    (cfg,) = configs
    assert cfg.max_calls == 60 and cfg.max_usd == Decimal(A_TOK) * Decimal("3.5")
    assert (got[0]["max_calls"], got[0]["effort_scale"]) == (60, "3.5")


def test_the_run_guard_compares_against_the_scaled_cap(tmp_path, monkeypatch):
    fake = FakeSol()
    monkeypatch.setattr(consolidate, "unillm_turn", fake)
    stores = _stores(tmp_path)
    sha, _ = _record(stores, "e1")
    guard = format(Decimal(A_TOK) * 3, "f")  # above one allowance, below five
    assert _run(stores, "e1", sha, effort="high", guard=guard) == []
    assert fake.calls == 0 and _events(stores)[0]["reason_codes"] == ["run_guard"]
    (out,) = _run(stores, "e1", sha, effort="low", guard=guard)
    assert fake.calls == 1


def test_an_effort_without_limits_runs_nothing_and_is_logged(tmp_path, monkeypatch):
    fake = FakeSol()
    monkeypatch.setattr(consolidate, "unillm_turn", fake)
    stores = _stores(tmp_path)
    sha, _ = _record(stores, "e1")
    assert _run(stores, "e1", sha, effort="xhigh") == []
    assert fake.calls == 0 and _events(stores) == []
    assert "pass limits" in stores.paths.errors.read_text()
    assert stores.evidence.cursor(BATCHED_CURSOR) == 0  # still due


@pytest.mark.parametrize("usage, shown", [("", False), ("off", False), ("on", True)])
def test_the_sol_usage_switch_reaches_the_passes_first_message(
    tmp_path,
    monkeypatch,
    usage,
    shown,
):
    from unify.memory_v2.usage import USAGE_HEADING

    fake = FakeSol()
    monkeypatch.setattr(consolidate, "unillm_turn", fake)
    stores = _stores(tmp_path)
    sha, _ = _record(stores, "e1")
    _run(stores, "e1", sha, usage=usage)
    first = fake.seen[0][1]["content"]
    assert (USAGE_HEADING in first) is shown


def test_emit_failures_never_stop_the_pass(tmp_path, monkeypatch):
    monkeypatch.setattr(consolidate, "unillm_turn", FakeSol())
    stores = _stores(tmp_path)
    sha, _ = _record(stores, "e1")

    def broken(row):
        raise OSError("stdout closed")

    (out,) = _run(stores, "e1", sha, emit=broken)
    assert out.pass_id == "e1.p0" and len(_events(stores)) == 2
    assert "stdout closed" in stores.paths.errors.read_text()


# --- (e) Sol's cost rows as notes ------------------------------------------------------------------------


def test_sol_cost_rows_are_notes_on_the_episode_commit(tmp_path, monkeypatch):
    msg = {"role": "assistant", "content": "thinking"}
    fake = FakeSol(script=[(msg, "0.0000001"), RuntimeError("provider down")])
    monkeypatch.setattr(consolidate, "unillm_turn", fake)
    stores = _stores(tmp_path)
    sha, _ = _record(stores, "e1")
    (out,) = _run(stores, "e1", sha)
    notes = [json.loads(n) for n in stores.episodes.notes(sha, ref="costs")]
    assert [(n["purpose"], n["usd"], n["episode"], n["pass_id"]) for n in notes] == [
        ("sol", "0.0000001", "e1", "e1.p0"),
        ("sol", "unknown", "e1", "e1.p0"),
    ]
    assert {n["sol_effort"] for n in notes} == {
        "low",
    }  # the inherited effort, on the pass's notes
    assert stores.episodes.notes(sha) == []  # nothing on the signals ref
    costs = episode_costs(stores, "e1")
    assert [(c.purpose, c.usd) for c in costs] == [
        ("actor", "0.000123"),
        ("sol", "0.0000001"),
        ("sol", "unknown"),
    ]
    assert out.usd == "0.0000001" and out.unknown_cost_calls == 1


def test_sol_transcript_is_kept_redacted_on_the_episode_commit(tmp_path, monkeypatch):
    secret = "planted-fake-token-6c1d2e"  # pragma: allowlist secret
    monkeypatch.setenv("UNIFY_TEST_FAKE_TOKEN", secret)
    msg = {"role": "assistant", "content": f"the token is {secret}"}
    fake = FakeSol(script=[(msg, "0.0000001"), (_finish(), "0.0000001")])
    monkeypatch.setattr(consolidate, "unillm_turn", fake)
    stores = _stores(tmp_path)
    sha, _ = _record(stores, "e1")
    (out,) = _run(stores, "e1", sha)
    assert secret in json.dumps(fake.seen)  # Sol itself saw it; the record must not
    rows = [json.loads(n) for n in stores.episodes.notes(sha, ref="sol-transcripts")]
    assert {r["pass_id"] for r in rows} == {"e1.p0"}
    assert [r["message"]["role"] for r in rows] == [
        "system",
        "user",
        "assistant",
        "user",
        "assistant",
        "tool",
    ]
    assert (
        rows[2]["message"]["content"] == "the token is <secret:UNIFY_TEST_FAKE_TOKEN>"
    )
    assert secret not in stores.episodes.run("log", "-p", "--all")
    assert secret not in json.dumps(rows)


# --- drift and suspect -----------------------------------------------------------------------------------


def test_drift_rides_with_the_pass_and_is_cleared_once_recorded(tmp_path, monkeypatch):
    monkeypatch.setattr(consolidate, "unillm_turn", FakeSol())
    stores = _stores(tmp_path)
    sha, _ = _record(stores, "e1")
    state = State(stores.paths.state, drift={"venmo"}, suspect={"venmo", "slack"})
    _run(stores, "e1", sha, state)
    assert state.drift == set()
    assert state.suspect == {"venmo", "slack"}  # a failed pass clears no suspect flag


# --- R10 -------------------------------------------------------------------------------------------------


SENTINEL = "SENTINEL-7f3a"


def test_checker_text_reaches_no_event_note_evidence_row_or_sol_input(
    tmp_path,
    monkeypatch,
):
    from unify.memory_v2 import sol_pass

    fake = FakeSol()
    monkeypatch.setattr(consolidate, "unillm_turn", fake)
    staged = tmp_path / "staged"
    stage = sol_pass.SolPass._stage_inputs

    def keep(self, req, inputs, *tree):
        stage(self, req, inputs, *tree)
        shutil.copytree(inputs, staged)  # what Sol's box would see at /inputs

    monkeypatch.setattr(sol_pass.SolPass, "_stage_inputs", keep)
    stores = _stores(tmp_path)
    sha, _ = _record(stores, "e1")
    outcome = {
        "solved": False,
        "checks": [{"name": "c", "passed": False, "reason": SENTINEL}],
        "summary": SENTINEL,
    }
    assert post_checker(stores, "e1", sha, outcome["solved"], "2026-10-08T01:02:00Z")
    (out,) = _run(stores, "e1", sha)
    dump = [
        json.dumps(fake.seen),
        events_path(stores.paths).read_text(),
        stores.episodes.run("log", "-p", "--all"),
        stores.memory.run("log", "-p", "--all"),
        "\n".join(stores.evidence.db.iterdump()),
    ]
    dump += [p.read_text(errors="replace") for p in staged.rglob("*.json")]
    dump += [
        p.read_text(errors="replace")
        for p in stores.paths.state_dir.rglob("*")
        if p.is_file()
    ]
    assert staged.exists() and (staged / "episodes" / "e1.json").is_file()
    assert all(SENTINEL not in text for text in dump)
    assert "e1.checker" not in json.dumps(fake.seen)
    assert '"label": "fail"' in dump[2]  # the verdict itself is on the signals ref


# --- reason codes: structural, never read from reason text ----------------------------------------------


@pytest.mark.parametrize(
    "script, codes",
    [
        (
            [
                ({"role": "assistant", "content": "x"}, "0.0000001"),
                RuntimeError("down"),
            ],
            ["sol_error", "no_manifest"],
        ),
        (
            [({"role": "assistant", "content": "x"}, "0.00000073")],
            ["pass_cap", "no_manifest"],
        ),
        (["sleep"], ["deadline", "no_manifest"]),
    ],
    ids=["sol-error", "usd-cap", "deadline"],
)
def test_end_event_reason_codes_come_from_the_outcome(
    tmp_path,
    monkeypatch,
    script,
    codes,
):
    monkeypatch.setattr(consolidate, "unillm_turn", FakeSol(script=script))
    monkeypatch.setattr(consolidate, "DEADLINE_S", 0.3)
    stores = _stores(tmp_path)
    sha, _ = _record(stores, "e1")
    (out,) = _run(stores, "e1", sha)
    assert out.codes == codes
    assert _events(stores)[-1]["reason_codes"] == codes


def test_reason_codes_are_known_codes_deduplicated_and_capped():
    from unify.memory_v2.sol_pass import PassOutcome

    def oc(passed, codes):
        return PassOutcome("p", passed, None, "0", 1, ["free text G1"], "", 0, codes)

    assert consolidate.reason_codes(oc(True, ["ok"]), None) == ["ok"]
    assert consolidate.reason_codes(oc(False, ["G2", "G2", "x", "G5"]), None) == [
        "G2",
        "G5",
    ]
    assert consolidate.reason_codes(oc(False, []), None) == ["sol_error"]
    assert consolidate.reason_codes(None, "deadline") == ["deadline"]
    assert consolidate.reason_codes(None, None) == ["sol_error"]
    many = [f"G{n}" for n in range(1, 7)] + [
        "no_manifest",
        "over_quota",
        "deadline",
        "pass_cap",
        "sol_error",
    ]
    assert len(consolidate.reason_codes(oc(False, many), None)) == 10


def test_the_gate_names_the_checks_that_refused_structurally(tmp_path):
    from unify.memory_v2.gate import Gate

    stores = _stores(tmp_path)
    head = stores.memory.head()
    res = Gate(stores.memory, stores.evidence, stores.blobs).check(
        head,
        head,
        {"items": "not a list"},
    )
    assert not res.passed and res.refused == ["G1"] and res.manifest_invalid
    assert not res.checks[
        "G4"
    ]  # unevaluated checks are False in checks, but not refused


# --- (f) the episode lookup ------------------------------------------------------------------------------


def test_lookup_returns_recorded_actions_of_any_kind(tmp_path):
    stores = _stores(tmp_path)
    tool = Action(0, "venmo", "me", [], {}, {"user_id": "u-1"}, "ok", "read")
    talk = Action(
        -1,
        "dialogue:user",
        "reply",
        ["do"],
        {},
        "ok then",
        "ok",
        kind="dialogue",
    )
    _record(stores, "e1", actions=[tool, talk])
    look = EpisodeLookup(stores)
    assert look.action("e1", 0) == tool
    assert look.action("e1", 1) == talk
    for eid, idx in [
        ("e1", 2),
        ("e1", -1),
        ("e1", True),
        ("e1", "0"),
        ("nope", 0),
        ("../e1", 0),
        (None, 0),
    ]:
        assert look.action(eid, idx) is None
    look.action("e1", 0).response[
        "user_id"
    ] = "changed"  # a copy: the cache is untouched
    assert look.action("e1", 0) == tool


def test_a_worktree_cover_is_admitted_by_the_g2_rule_through_the_lookup(tmp_path):
    stores = _stores(tmp_path)
    blob = stores.blobs.put(CSV)
    wt = Action(
        0,
        "worktree:workspace",
        "read",
        ["finance/ap/invoices-2026-10.csv"],
        {},
        {
            "blob_before": blob,
            "blob_after": blob,
            "size": len(CSV),
            "shape": {"format": "csv"},
        },
        "ok",
        "read",
        kind="worktree",
    )
    _record(
        stores,
        "o1",
        actions=[wt, Action(0, "venmo", "me", [], {}, {"u": 1}, "ok")],
    )
    look = EpisodeLookup(stores)
    has = stores.blobs.has
    assert cover_problem(look.action("o1", 0), "worktree_workspace", has) is None
    assert cover_problem(look.action("o1", 1), "worktree_workspace", has) is not None
    assert (
        cover_problem(look.action("o1", 9), "worktree_workspace", has)
        == "not a recorded action"
    )


@needs_bwrap
def test_the_gate_admits_a_worktree_cover_through_the_lookup(tmp_path):
    """Remote (bwrap): the real gate's G2, held-out values included, with the lookup over a written episode."""
    from tests.memory_v2.test_kinds_gate import (
        INVENTORY,
        INVOICES,
        WT_FILES,
        WT_MAN,
        _commit,
        _wt,
    )
    from unify.memory_v2.gate import Gate

    stores = _stores(tmp_path)
    inv, tsv = stores.blobs.put(INVOICES), stores.blobs.put(INVENTORY)
    _record(
        stores,
        "o1",
        actions=[
            _wt("finance/ap/invoices-2026-10.csv", inv),
            _wt("ops/inventory.csv", tsv),
        ],
    )
    gate = Gate(
        stores.memory,
        stores.evidence,
        stores.blobs,
        action_lookup=EpisodeLookup(stores).action,
    )
    parent = stores.memory.head()
    cand = _commit(stores.memory, WT_FILES)
    res = gate.check(parent, cand, WT_MAN)
    assert res.checks["G2"], res.reasons
    blind = Gate(
        stores.memory,
        stores.evidence,
        stores.blobs,
    )  # no lookup: fails closed
    assert not blind.check(parent, cand, WT_MAN).checks["G2"]


def test_open_stores_is_idempotent(tmp_path):
    a = _stores(tmp_path)
    head = (a.memory.head(), a.episodes.head())
    b = _stores(tmp_path)
    assert (b.memory.head(), b.episodes.head()) == head


def test_every_setting_the_driver_reads_is_a_build_setting():
    """The driver reads its switches by name; a renamed setting must never be read under its old name."""
    import inspect
    import re

    from unify.memory_v2.integration import consolidate
    from unify.settings import ProductionSettings

    read = set(
        re.findall(r'"(UNIFY_MEMORY_V2_[A-Z0-9_]+)"', inspect.getsource(consolidate)),
    )
    assert read, "the driver reads no v2 setting by name"
    assert read <= set(ProductionSettings.model_fields), sorted(
        read - set(ProductionSettings.model_fields),
    )


# --- stage-5 test checks (memory v2.1): the switches reach the gate and Sol's brief ------------------------


@pytest.mark.parametrize("on", [False, True])
def test_the_qa_switches_reach_the_gate_and_sols_brief(tmp_path, monkeypatch, on):
    fake = FakeSol()
    monkeypatch.setattr(consolidate, "unillm_turn", fake)
    made = []
    real = consolidate.Gate

    def gate(*args, **kwargs):
        made.append(real(*args, **kwargs))
        return made[-1]

    monkeypatch.setattr(consolidate, "Gate", gate)
    stores = _stores(tmp_path)
    sha, _ = _record(stores, "e1")
    settings = _settings(e=1)
    if on:
        settings.UNIFY_MEMORY_V2_QA_FIXTURES = "strict"
        settings.UNIFY_MEMORY_V2_QA_MUTATION = "on"
    asyncio.run(
        run_due_passes(
            stores,
            "e1",
            sha,
            State(stores.paths.state),
            effort="high",
            settings=settings,
            emit=None,
        ),
    )
    (g,) = made
    system = fake.seen[0][0]["content"]
    if on:
        assert g.qa == QAConfig(fixtures="strict", mutation=True)
        assert system.startswith(SOL_SYSTEM.split("Run tests as the gate does")[0])
        assert (
            "Mutants:" in system
            and "Drawn inputs:" in system
            and "Replay:" not in system
        )
    else:
        assert g.qa == QAConfig() and not g.qa.on
        assert system == SOL_SYSTEM


@pytest.mark.parametrize("step", [RuntimeError("upstream failed"), "sleep"])
def test_no_further_pass_starts_after_a_call_that_may_still_be_in_flight(
    tmp_path,
    monkeypatch,
    step,
):
    # Sol's proxy serves one Sol call at a time: after a call ended by an error or the deadline (it may still be
    # running upstream), the session starts no further pass; the request stays due
    fake = FakeSol(script=[step])
    monkeypatch.setattr(consolidate, "unillm_turn", fake)
    monkeypatch.setattr(consolidate, "DEADLINE_S", 0.2)
    original = consolidate.Trigger.after_episode

    def twice(self, eid):
        due = original(self, eid)
        return list(due) * 2  # two passes due in one session

    monkeypatch.setattr(consolidate.Trigger, "after_episode", twice)
    stores = _stores(tmp_path)
    sha, _ = _record(stores, "e1")
    outcomes = _run(stores, "e1", sha)
    assert fake.calls == 1
    assert [e["phase"] for e in _events(stores)] == ["start", "end"]
    assert len(outcomes) <= 1


# --- memory v2.1 P1 (UNIFY_MEMORY_V21) ------------------------------------------------------------------


def _v21_settings(raw):
    s = _settings()
    s.UNIFY_MEMORY_V21 = raw
    s.UNIFY_MEMORY_V21_E = (
        1  # v2.1's own E (D43, 100k by default): one small episode makes a pass due
    )
    return s


def test_sol_settings_reads_the_v21_switch():
    assert sol_settings(_settings()).v21 is False
    assert sol_settings(_v21_settings("on")).v21 is True
    assert sol_settings(_v21_settings("off")).v21 is False
    with pytest.raises(ValueError):
        sol_settings(_v21_settings("maybe"))


def test_v21_reaches_the_pass_and_its_end_event_records_coverage(tmp_path, monkeypatch):
    fake = FakeSol()  # finishes every turn: refused under v21 until e1 is covered
    monkeypatch.setattr(consolidate, "unillm_turn", fake)
    stores = _stores(tmp_path)
    sha, _ = _record(stores, "e1")
    (out,) = asyncio.run(
        run_due_passes(
            stores,
            "e1",
            sha,
            State(stores.paths.state),
            effort="low",
            settings=_v21_settings("on"),
            emit=None,
        ),
    )
    assert not out.passed and out.coverage["missing"] == ["e1"]
    end = [e for e in _events(stores) if e.get("phase") == "end"][-1]
    assert end["coverage"] == out.coverage
    assert end["reads"] == 0 and end["exported_bytes"] == 0


def test_v21_off_leaves_the_end_event_unchanged(tmp_path, monkeypatch):
    monkeypatch.setattr(consolidate, "unillm_turn", FakeSol())
    stores = _stores(tmp_path)
    sha, _ = _record(stores, "e1")
    (out,) = _run(stores, "e1", sha)
    assert out.coverage is None
    end = [e for e in _events(stores) if e.get("phase") == "end"][-1]
    assert not {"coverage", "reads", "exported_bytes"} & set(end)


@pytest.mark.parametrize("on", [False, True])
def test_v21_gates_write_passes_with_the_v21_checks(tmp_path, monkeypatch, on):
    monkeypatch.setattr(consolidate, "unillm_turn", FakeSol())
    made = []

    class Recording(consolidate.Gate):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            made.append(self)

    monkeypatch.setattr(consolidate, "Gate", Recording)
    stores = _stores(tmp_path)
    sha, _ = _record(stores, "e1")
    asyncio.run(
        run_due_passes(
            stores,
            "e1",
            sha,
            State(stores.paths.state),
            effort="low",
            settings=_v21_settings("on" if on else "off"),
            emit=None,
        ),
    )
    (g,) = made
    if on:
        assert g.v21 is not None and g.v21.checks and g.v21.role == "write" and g.qa.v21
    else:
        assert g.v21 is None and not g.qa.v21


# --- memory v2.1: CURATE after WRITE, triggered by library state (P6) ------------------------------------------

from unify.memory_v2.curate import curate_system
from tests.memory_v2.test_gate import _merged
from tests.memory_v2.test_layout import LIB
from tests.memory_v2.test_overlap import SPLIT, SPLIT_ID, SPLIT_PATH, TOKENS

#: an earlier pass left two functions doing one job
DUPLICATE = {
    "memory/text/__init__.py": '"""Text helpers."""\n',
    "memory/text/parse.py": LIB["memory/text/parse.py"],
    SPLIT_PATH: SPLIT,
}


def _settings21(on=True, **kw):
    s = _settings(**kw)
    s.UNIFY_MEMORY_V21 = "on" if on else "off"
    return s


def _run21(stores, eid, sha, *, on=True, **kw):
    return asyncio.run(
        run_due_passes(
            stores,
            eid,
            sha,
            State(stores.paths.state),
            effort="low",
            settings=_settings21(on, **kw),
            emit=None,
        ),
    )


def _no_records(monkeypatch):
    # P5's records are not under test here: no suspect item, every item experimental
    monkeypatch.setattr(consolidate, "_lifecycle", lambda stores, sha: (None, {}, {}))


def _kinds(stores):
    return [
        r[0]
        for r in stores.evidence.db.execute("SELECT kind FROM passes ORDER BY pass_id")
    ]


def test_curate_runs_after_write_when_the_library_state_warrants_it(
    tmp_path,
    monkeypatch,
):
    fake = FakeSol()
    monkeypatch.setattr(consolidate, "unillm_turn", fake)
    _no_records(monkeypatch)
    stores = _stores(tmp_path)
    _merged(stores.memory, DUPLICATE)
    sha, _ = _record(stores, "e1")
    outs = _run21(stores, "e1", sha)
    assert [o.pass_id for o in outs] == ["e1.p0", "e1.p1"] and _kinds(stores) == [
        "batched",
        "curate",
    ]
    starts = [e for e in _events(stores) if e["phase"] == "start"]
    assert "curate" not in starts[0]
    assert f"overlap (antiunify): {TOKENS}, {SPLIT_ID}" in starts[1]["curate"]
    assert fake.seen[-1][0] == {"role": "system", "content": curate_system()}
    # the library is unchanged and CURATE was shown these reasons: the next WRITE pass runs alone
    sha2, _ = _record(stores, "e2", minute=1)
    assert [o.pass_id for o in _run21(stores, "e2", sha2)] == ["e2.p0"]
    assert _kinds(stores) == ["batched", "curate", "batched"]


def test_no_curate_on_a_library_with_nothing_to_curate(tmp_path, monkeypatch):
    monkeypatch.setattr(consolidate, "unillm_turn", FakeSol())
    _no_records(monkeypatch)
    stores = _stores(tmp_path)
    sha, _ = _record(stores, "e1")
    assert [o.pass_id for o in _run21(stores, "e1", sha)] == ["e1.p0"]


def test_curate_never_runs_with_v21_off(tmp_path, monkeypatch):
    monkeypatch.setattr(consolidate, "unillm_turn", FakeSol())
    monkeypatch.setattr(
        consolidate,
        "_lifecycle",
        lambda *a: pytest.fail("v2 never reads CURATE's state"),
    )
    stores = _stores(tmp_path)
    _merged(stores.memory, DUPLICATE)
    sha, _ = _record(stores, "e1")
    assert [o.pass_id for o in _run21(stores, "e1", sha, on=False)] == ["e1.p0"]
    found = stores.evidence.db.execute(
        "SELECT 1 FROM sqlite_master WHERE name IN ('curate_seen', 'curations')",
    ).fetchone()
    assert found is None


def test_curate_waits_for_the_run_guard(tmp_path, monkeypatch):
    monkeypatch.setattr(
        consolidate,
        "unillm_turn",
        FakeSol(),
    )  # 0.01 USD a call: the WRITE pass commits 0.01
    _no_records(monkeypatch)
    stores = _stores(tmp_path)
    _merged(stores.memory, DUPLICATE)
    sha, _ = _record(stores, "e1")
    assert [o.pass_id for o in _run21(stores, "e1", sha, guard="0.01")] == ["e1.p0"]
    held = _events(stores)[-1]
    assert held["pass_id"] == "e1.p1" and held["reason_codes"] == ["run_guard"]
    assert (
        stores.evidence.curate_seen() == set()
    )  # never shown: it fires again once the guard allows


def test_no_curate_after_a_write_whose_call_may_still_be_in_flight(
    tmp_path,
    monkeypatch,
):
    fake = FakeSol(script=[RuntimeError("upstream failed")])
    monkeypatch.setattr(consolidate, "unillm_turn", fake)
    _no_records(monkeypatch)
    stores = _stores(tmp_path)
    _merged(stores.memory, DUPLICATE)
    sha, _ = _record(stores, "e1")
    _run21(stores, "e1", sha)
    assert fake.calls == 1 and _kinds(stores) == ["batched"]
