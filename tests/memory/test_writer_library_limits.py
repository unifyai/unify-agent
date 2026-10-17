"""In library mode an uncapped pass (PassConfig.max_usd None) has no USD cap and no count limits; the
operational step guard and the deadline end a runaway. A capped pass (v2, or library mode with a cap) is unchanged.
"""

from decimal import Decimal

from unify.common._async_tool import loop_stop
from unify.memory import library_prompts, repair, writer_pass
from unify.memory.writer_pass import CODE_PASS_CAP, CODE_STEP_GUARD
from tests.memory.test_writer_pass import _call
from tests.memory.test_writer_library_tools import _pass


def _reads(n):
    return [_call(f"r{i}", "read", {"path": "/inputs"}) for i in range(n)]


def test_an_uncapped_pass_has_no_usd_stop_or_read_limit(tmp_path, monkeypatch):
    # the fake model costs USD 0.001 a call: a USD 0.0025 cap would end it after 3 calls; max_reads 2 would refuse
    # the third read. With no manifest written, the pass ends at the (lowered) operational step guard.
    monkeypatch.setattr(writer_pass, "STEP_GUARD", 30)
    # twelve identical reads make no progress: the no-progress rule, not under test here, would end them first
    monkeypatch.setattr(loop_stop, "MEMORY_K", 100)
    turns = _reads(12)
    out, model = _pass(tmp_path, turns, eids=("e2",), max_usd=None, max_reads=2)
    assert CODE_PASS_CAP not in out.codes and out.reads >= 12
    assert not any("reader-call limit" in str(v) for v in model.outputs.values())


def test_the_step_guard_ends_a_runaway_with_its_operational_code(tmp_path, monkeypatch):
    monkeypatch.setattr(writer_pass, "STEP_GUARD", 4)
    out, _ = _pass(tmp_path, _reads(20), eids=("e2",), max_usd=None)
    assert (
        CODE_STEP_GUARD in out.codes
        and CODE_PASS_CAP not in out.codes
        and out.calls == 4
    )


def test_a_capped_library_pass_keeps_its_cap(tmp_path):
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
    monkeypatch.setattr(writer_pass, "STEP_GUARD", 8)
    out, model = _pass(tmp_path, [], eids=("e2",), max_usd=None)
    refused = [
        m
        for m in model.seen
        if m.get("role") == "tool" and "not finished: G1" in str(m.get("content"))
    ]
    assert (
        len(refused) > writer_pass.FINISH_G1_REFUSALS and CODE_STEP_GUARD in out.codes
    )


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


def test_the_library_briefs_state_no_count_limits_but_numbers_still_render_when_given():
    text = library_prompts.write_brief_now()
    assert "times per pass" not in text and "while the pass budget lasts" not in text
    # no sentence about time running out (it fed the writer's economy worry); the limits are stated as facts
    assert (
        "time runs out" not in text
        and "The items that pass land, and the rest is kept as a draft" in text
    )
    numbered = library_prompts.write_brief(
        limits=library_prompts.limits_now(cell_s=45.0),
        pytest_env={},
    )
    # the real limits render as information: the cell time, the step guard, no count of checks or rounds
    assert (
        "at most 45 s" in numbered
        and "There is no set number of repair rounds" in numbered
    )
    assert "times per pass" not in numbered
    check = next(
        t
        for t in writer_pass.writer_tools(library_mode=True)
        if t["function"]["name"] == "check"
    )
    assert "at most" not in check["function"]["description"]
