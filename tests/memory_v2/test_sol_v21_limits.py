"""Memory v2.1 r5 (S6): an uncapped pass (PassConfig.max_usd None) has no USD cap and no count limits; the
operational step guard and the deadline end a runaway. A capped pass (v2, or v2.1 with a cap) is unchanged.
"""

from decimal import Decimal

from unify.memory_v2 import prompts_v21, repair, sol_pass
from unify.memory_v2.sol_pass import CODE_PASS_CAP, CODE_STEP_GUARD
from tests.memory_v2.test_sol_pass import _call
from tests.memory_v2.test_sol_v21_tools import _pass


def _reads(n):
    return [_call(f"r{i}", "read", {"path": "/inputs"}) for i in range(n)]


def test_an_uncapped_pass_has_no_usd_stop_or_read_limit(tmp_path, monkeypatch):
    # the fake model costs USD 0.001 a call: a USD 0.0025 cap would end it after 3 calls; max_reads 2 would refuse
    # the third read. With no manifest written, the pass ends at the (lowered) operational step guard.
    monkeypatch.setattr(sol_pass, "STEP_GUARD", 30)
    turns = _reads(12) + [
        _call("d", "dismiss", {"episode": "e2", "reason": "nothing reusable"}),
    ]
    out, model = _pass(tmp_path, turns, eids=("e2",), max_usd=None, max_reads=2)
    assert CODE_PASS_CAP not in out.codes and out.reads >= 12
    assert not any(
        str(v).startswith("not run: reader budget") for v in model.outputs.values()
    )


def test_the_step_guard_ends_a_runaway_with_its_operational_code(tmp_path, monkeypatch):
    monkeypatch.setattr(sol_pass, "STEP_GUARD", 4)
    out, _ = _pass(tmp_path, _reads(20), eids=("e2",), max_usd=None)
    assert (
        CODE_STEP_GUARD in out.codes
        and CODE_PASS_CAP not in out.codes
        and out.calls == 4
    )


def test_a_capped_v21_pass_keeps_its_cap(tmp_path):
    out, _ = _pass(
        tmp_path,
        _reads(20),
        eids=("e2",),
        max_usd=Decimal("0.0025"),
        max_calls=40,
    )
    assert CODE_PASS_CAP in out.codes


def test_an_uncapped_finish_keeps_refusing_an_unparseable_manifest(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(sol_pass, "STEP_GUARD", 8)
    turns = [_call("d", "dismiss", {"episode": "e2", "reason": "nothing reusable"})]
    out, model = _pass(tmp_path, turns, eids=("e2",), max_usd=None)
    refused = [
        m
        for m in model.seen
        if m.get("role") == "tool" and "not finished: G1" in str(m.get("content"))
    ]
    assert len(refused) > sol_pass.FINISH_G1_REFUSALS and CODE_STEP_GUARD in out.codes


def test_repair_without_a_usd_figure_has_no_round_count():
    assert repair.may_repair(5, None, 600.0, 10, [Decimal("1")], [1.0]) is None
    assert "round time reserve" in repair.may_repair(
        5,
        None,
        0.5,
        10,
        [Decimal("1")],
        [1.0],
    )
    assert "at most" in repair.may_repair(
        repair.REPAIR_ROUNDS,
        Decimal("9"),
        600.0,
        10,
        [Decimal("1")],
        [1.0],
    )


def test_the_v21_briefs_state_no_count_limits_but_numbers_still_render_when_given():
    text = prompts_v21.write_brief_now()
    assert "times per pass" not in text and "while the pass budget lasts" not in text
    assert "When the pass's time runs out" in text
    numbered = prompts_v21.write_brief(
        view_bytes=8000,
        max_checks=7,
        repair_rounds=3,
        mutation_min=Decimal("0.7"),
        sample_k=11,
        lint_min=6,
        pytest_env={},
        semantic_types="",
    )
    assert "at most 7 times per pass" in numbered and "up to 3 of them" in numbered
    assert "at most" not in sol_pass._v21_check_description()
