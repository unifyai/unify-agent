"""A writer pass whose round the memory agents' no-progress stop ended (``loop_stop.AgentNoProgress``: 2 * k
no-progress model calls in a row) has a reason code of its own. A refused pass lists it in the consolidation end
event's ``reason_codes`` beside the gate checks (``refused_by`` stays the gate checks alone); a passed pass's codes
stay exactly ``["ok"]``, and the end event's ``no_progress`` report says the stop ended its round. A pass that only
got the notice, or never got one, does not carry it.

Keyless: the model is scripted; the end-to-end case runs the writer's cells in the real box.
"""

from __future__ import annotations


from tests.memory.test_agent_no_progress import COUNTER, NOOP, _admit
from tests.memory.test_writer_library_repair import FILES_CODE
from tests.memory.test_writer_pass import Turns, _call, _run, _writer, needs_bwrap
from unify.common._async_tool.loop_stop import MEMORY_K
from unify.memory.gate import GateResult
from unify.memory.writer_pass import CODE_OK, PassOutcome
from unify.memory.writer_pass import STEP_GUARD

# ── ASSUMED NAMES (the integrator adapts them here, in one place) ────────────────────────────────────────────────
#: The reason code of a pass the no-progress stop ended.
NO_PROGRESS_CODE = "no_progress"
# ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────

ENDED = {"k": MEMORY_K, "notices": 1, "ended": True, "at_call": 2 * MEMORY_K + 2}
NOTICE_ONLY = {"k": MEMORY_K, "notices": 1, "ended": False, "at_call": None}


def _stores(tmp_path):
    from tests.memory.integration.test_consolidate import _stores as stores

    return stores(tmp_path / "s")


def _end(tmp_path, outcome):
    from unify.memory.integration import consolidate

    return consolidate._end_event(_stores(tmp_path), "px", outcome, [], 1.0, None)


def _outcome(*, codes, no_progress, passed=True) -> PassOutcome:
    """A library-mode outcome (``coverage`` set, so the end event carries ``refused_by``)."""
    return PassOutcome(
        pass_id="px",
        passed=passed,
        commit=None,
        usd="0.01",
        calls=2 * MEMORY_K + 2,
        codes=list(codes),
        coverage={},
        no_progress=no_progress,
    )


def test_the_code_is_a_known_reason_code_of_its_own():
    from unify.memory.integration import consolidate
    from unify.memory import writer_pass

    assert NO_PROGRESS_CODE in consolidate.REASON_CODES
    others = {
        writer_pass.CODE_OK,
        writer_pass.CODE_WRITER_ERROR,
        writer_pass.CODE_STEP_GUARD,
        writer_pass.CODE_DEADLINE,
        writer_pass.CODE_PASS_CAP,
        writer_pass.CODE_NO_MANIFEST,
    }
    assert NO_PROGRESS_CODE not in others


def test_a_refused_pass_the_stop_ended_lists_the_code_beside_the_gate_checks(tmp_path):
    # refused_by stays the gate checks that refused; the stop is a reason code beside them
    out = _outcome(codes=["G2", NO_PROGRESS_CODE], no_progress=ENDED, passed=False)
    end = _end(tmp_path, out)
    assert end["reason_codes"] == ["G2", NO_PROGRESS_CODE], end["reason_codes"]
    assert end["refused_by"] == ["G2"], end["refused_by"]
    assert end["no_progress"]["ended"] is True
    # codes only, never text
    assert all(isinstance(c, str) and " " not in c for c in end["reason_codes"] + end["refused_by"])


def test_a_passed_pass_the_stop_ended_is_ok_and_its_report_says_so(tmp_path):
    # a passed pass's codes are exactly ["ok"]; the stop's own report on the end event tells it apart
    end = _end(tmp_path, _outcome(codes=[CODE_OK], no_progress=ENDED, passed=True))
    assert end["reason_codes"] == [CODE_OK]
    assert end["refused_by"] == []
    assert end["no_progress"]["ended"] is True


def test_a_notice_alone_or_no_notice_carries_no_code(tmp_path):
    for no_progress in (NOTICE_ONLY, None):
        end = _end(tmp_path, _outcome(codes=[CODE_OK], no_progress=no_progress))
        assert NO_PROGRESS_CODE not in end["reason_codes"]
        assert NO_PROGRESS_CODE not in end["refused_by"]


@needs_bwrap
def test_a_writer_pass_the_stop_ended_reports_the_code(tmp_path):
    loops = 4 * MEMORY_K
    model = Turns(
        [_call("w0", "execute_code", {"code": FILES_CODE})]
        + [_call(f"n{i}", "execute_code", {"code": NOOP}) for i in range(loops)],
    )
    mem, ev, writer = _writer(tmp_path, model, max_calls=STEP_GUARD, deadline_s=300.0)
    merged: list = []
    writer.gate.check = lambda *a, **k: GateResult(True, {}, [], [])
    writer.gate.merge = _admit(writer, merged)
    out = _run(writer)
    assert out.no_progress and out.no_progress["ended"] is True, out.reasons
    # the final merge landed the items, so the pass passed: exactly ["ok"], with the stop in its report
    assert out.passed and out.codes == [CODE_OK], out.codes
    end = _end(tmp_path, out)
    assert end["no_progress"]["ended"] is True and end["reason_codes"] == [CODE_OK], end
    # not mistaken for the operational step guard or a writer error
    assert "operational_step_guard" not in end["reason_codes"]
    assert "writer_error" not in end["reason_codes"]


@needs_bwrap
def test_a_writer_never_stopped_carries_no_code(tmp_path):
    loops = 4 * MEMORY_K
    model = Turns(
        [_call("w0", "execute_code", {"code": FILES_CODE})]
        + [_call(f"n{i}", "execute_code", {"code": COUNTER}) for i in range(loops)],
    )
    mem, ev, writer = _writer(tmp_path, model, max_calls=STEP_GUARD, deadline_s=300.0)
    writer.gate.check = lambda *a, **k: GateResult(True, {}, [], [])
    writer.gate.merge = _admit(writer, [])
    out = _run(writer)
    assert out.no_progress is None
    assert NO_PROGRESS_CODE not in out.codes
    end = _end(tmp_path, out)
    assert NO_PROGRESS_CODE not in end["reason_codes"] and NO_PROGRESS_CODE not in end["refused_by"]
