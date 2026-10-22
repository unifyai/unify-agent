"""A pass has no USD cap of its own and no count limits; the operational step guard (``max_calls``, STEP_GUARD in
production) and the deadline end a runaway, and the run's room (``usd_limit``) is its one USD stop.
"""

from unify.common._async_tool import loop_stop
from unify.memory import repair
from unify.memory.writer_pass import CODE_PASS_CAP, CODE_STEP_GUARD, STEP_GUARD, PassConfig
from tests.memory.test_writer_pass import _call
from tests.memory.test_writer_library_tools import _pass


def _reads(n):
    return [_call(f"r{i}", "read", {"path": "/inputs"}) for i in range(n)]


def test_a_pass_has_no_usd_stop_or_read_limit_of_its_own(tmp_path, monkeypatch):
    # the fake model costs USD 0.001 a call; reads are counted, never limited. With no manifest written, the pass ends
    # at its (lowered) operational step guard.
    # twelve identical reads make no progress: the no-progress rule, not under test here, would end them first
    monkeypatch.setattr(loop_stop, "MEMORY_K", 100)
    turns = _reads(12)
    out, model = _pass(tmp_path, turns, eids=("e2",), max_calls=30)
    assert CODE_PASS_CAP not in out.codes and out.reads >= 12
    assert not any("reader-call limit" in str(v) for v in model.outputs.values())
    assert not hasattr(PassConfig, "max_usd") and not hasattr(PassConfig, "max_reads")


def test_the_step_guard_ends_a_runaway_with_its_operational_code(tmp_path):
    out, _ = _pass(tmp_path, _reads(20), eids=("e2",), max_calls=4)
    assert (
        CODE_STEP_GUARD in out.codes
        and CODE_PASS_CAP not in out.codes
        and out.calls == 4
    )
    assert PassConfig(run_home=str(tmp_path)).max_calls == STEP_GUARD


def test_finish_keeps_refusing_an_unparseable_manifest(tmp_path):
    out, model = _pass(tmp_path, [], eids=("e2",), max_calls=8)
    refused = [
        m
        for m in model.seen
        if m.get("role") == "tool" and "not finished: G1" in str(m.get("content"))
    ]
    # no finish cap: every finish is judged until the step guard ends the pass
    assert len(refused) > 2 and CODE_STEP_GUARD in out.codes


def test_repair_has_no_round_count():
    assert repair.may_repair(5, 600.0, 10, [1.0]) is None
    assert "round time reserve" in repair.may_repair(5, 0.5, 10, [1.0])
    assert not hasattr(repair, "REPAIR_ROUNDS")
