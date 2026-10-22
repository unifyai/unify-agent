"""The shared no-progress rule for the memory agents (``loop_stop.AgentNoProgress``): the actor's definition of a
model call that makes no progress, fed from the writer's, curator's and helpers' own conversations. At
``loop_stop.MEMORY_K`` such calls in a row the agent gets one notice; at twice as many its round ends. A writer's
pass then goes to its final merge; a helper ends with what it staged. Keyless: every model is scripted.
"""

import json

from unify.common._async_tool import loop_stop
from unify.common._async_tool.loop_stop import MEMORY_K, AgentNoProgress, terminal_tool
from unify.memory.drafts import draft_states
from unify.memory.gate import GateResult
from tests.memory.test_helpers import helper_boxes  # noqa: F401 - the helpers' scripted box
from tests.memory import writer_library_items as wl
from tests.memory.test_writer_library_repair import FILES_CODE
from tests.memory.test_writer_pass import Turns, _call, _run, _writer, needs_bwrap

ME, OTHER = wl.ITEM_ID, "memory.acct.users:other"
NOTICE = f"You have repeated this call {MEMORY_K} times with the same result"

# the recorded loop: it replaces a line of a test file with the same line, writes the file back unchanged and prints
# a constant
NOOP = (
    "import pathlib\n"
    f"p = pathlib.Path('/memory/{wl.TEST_FILE}')\n"
    "t = p.read_text()\n"
    "p.write_text(t.replace('u-1', 'u-1'))\n"
    "print('none')\n"
)
# the same code each time, with a result that changes each time
COUNTER = (
    "import pathlib\n"
    "p = pathlib.Path('/memory/.pass/count.txt')\n"
    "n = int(p.read_text()) + 1 if p.exists() else 1\n"
    "p.write_text(str(n))\n"
    "print(n)\n"
)


# --- the rule on a conversation ------------------------------------------------------------------------------


def _turn(messages, i, code, result):
    messages.append(
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": f"c{i}",
                    "type": "function",
                    "function": {"name": "execute_code", "arguments": json.dumps({"code": code})},
                },
            ],
        },
    )
    messages.append({"role": "tool", "tool_call_id": f"c{i}", "content": result})


def _start():
    return [{"role": "system", "content": "brief"}, {"role": "user", "content": "pass p1"}]


def test_each_memory_agent_is_told_its_own_terminal_tool():
    assert terminal_tool("writer") == "finish"
    assert terminal_tool("helper") == "done"
    assert "call finish and keep the item as a draft" in AgentNoProgress("finish").notice
    assert "call tools.done and keep the item as a draft" in AgentNoProgress("done", in_code=True).notice
    assert NOTICE in AgentNoProgress("finish").notice


def test_a_repeated_call_gets_one_notice_at_k_and_ends_at_twice_k():
    watch, messages = AgentNoProgress("finish"), _start()
    acted = []
    for i in range(3 * MEMORY_K):
        _turn(messages, i, "x = open('/memory/a').read()\nprint(len(x))", "12")
        acted.append(watch.after_call(messages))
        if acted[-1] == "end":
            break
    # the first call is new; each repeat after it is a no-progress call
    assert acted.index("notice") == MEMORY_K and acted.count("notice") == 1
    assert acted[-1] == "end" and len(acted) == 2 * MEMORY_K + 1
    notes = [m for m in messages if m.get("role") == "user" and NOTICE in str(m.get("content"))]
    assert len(notes) == 1 and "_loop_authored" not in notes[0]  # the request carries plain messages only
    assert watch.report() == {"k": MEMORY_K, "notices": 1, "ended": True, "at_call": 2 * MEMORY_K + 1}
    assert "no progress: " in watch.note("end") and "round ended at model call" in watch.note("end")


def test_a_repeated_call_whose_result_changes_is_never_stopped():
    watch, messages = AgentNoProgress("finish"), _start()
    for i in range(6 * MEMORY_K):
        _turn(messages, i, COUNTER, f"{i + 1}\n")
        assert watch.after_call(messages) is None
    assert watch.report() is None and len(messages) == 2 + 2 * 6 * MEMORY_K


def test_another_user_message_starts_the_count_again_and_a_helper_running_is_not_counted():
    watch, messages = AgentNoProgress("finish"), _start()
    # a cell that does nothing makes no progress from its first run
    for i in range(MEMORY_K - 1):
        _turn(messages, i, "print('none')", "none")
        assert watch.after_call(messages) is None
    assert watch.tracker.count == MEMORY_K - 1
    messages.append({"role": "user", "content": "This is repair round 1."})  # a new round
    for i in range(MEMORY_K, 2 * MEMORY_K - 1):
        _turn(messages, i, "print('none')", "none")
        assert watch.after_call(messages) is None
    assert watch.tracker.count == MEMORY_K - 1
    # calls made while other work of the agent runs are not counted
    for i in range(2 * MEMORY_K, 5 * MEMORY_K):
        _turn(messages, i, "print('none')", "none")
        assert watch.after_call(messages, in_flight=True) is None
    assert watch.tracker.count == 0 and watch.report() is None


def test_a_round_that_ends_anyway_gets_no_notice():
    watch, messages = AgentNoProgress("finish"), _start()
    for i in range(3 * MEMORY_K):
        _turn(messages, i, "print('none')", "none")
        assert watch.after_call(messages, act=False) is None
    assert watch.report() is None and watch.calls == 3 * MEMORY_K
    assert not any(m.get("role") == "user" and NOTICE in str(m.get("content")) for m in messages)


def test_k_is_the_module_constant_when_not_given(monkeypatch):
    monkeypatch.setattr(loop_stop, "MEMORY_K", 2)
    assert AgentNoProgress("finish").k == 2 and AgentNoProgress("finish", k=7).k == 7


# --- the writer pass -----------------------------------------------------------------------------------------


def _admit(writer, merged):
    """The final merge's per-item admission: ME lands, OTHER is refused and kept as a draft."""

    def merge(parent, candidate, manifest, pass_id, kind, channel, usd, **kwargs):
        merged.append(candidate)
        res = GateResult(
            True,
            {},
            [f"item refused: G2: {OTHER} covers no recorded action"],
            [],
            merged=parent,
            items_merged=[ME],
            items_refused={OTHER: ["G2"]},
        )
        writer.gate._record(parent, candidate, pass_id, kind, channel, usd, res)
        return res

    return merge


@needs_bwrap
def test_a_writer_repeating_a_no_op_cell_is_told_then_its_round_ends_and_the_merge_lands_what_passes(tmp_path):
    from tests.memory.integration.test_consolidate import _stores
    from unify.memory.integration import consolidate

    loops = 4 * MEMORY_K
    model = Turns(
        [_call("w0", "execute_code", {"code": FILES_CODE})]
        + [_call(f"n{i}", "execute_code", {"code": NOOP}) for i in range(loops)],
    )
    mem, ev, writer = _writer(tmp_path, model, max_usd=None, deadline_s=300.0)
    checked, merged = [], []
    writer.gate.check = lambda *a, **k: checked.append(a) or GateResult(True, {}, [], [])
    writer.gate.merge = _admit(writer, merged)
    out = _run(writer)
    # call 1 writes the files; call 2 is the first no-op cell; calls 3.. repeat it: the notice comes with the
    # K-th repeat and the round ends at the 2K-th, without a finish and without a repair round's check
    ended_at = 2 * MEMORY_K + 2
    assert len(model.queue) == loops + 1 - ended_at, out.reasons
    assert out.no_progress == {"k": MEMORY_K, "notices": 1, "ended": True, "at_call": ended_at}
    assert out.finished is False and checked == []
    notices = [m for m in model.seen if m.get("role") == "user" and NOTICE in str(m.get("content"))]
    assert len(notices) == 1 and "call tools.finish and keep the item as a draft" in notices[0]["content"]
    assert [r for r in out.reasons if r.startswith("no progress: ")] == [
        f"no progress: {MEMORY_K} model calls in a row made no progress; a notice was given at model call "
        f"{MEMORY_K + 2}",
        f"no progress: {2 * MEMORY_K} model calls in a row made no progress (each ran a cell that does nothing, "
        "or repeated one of the two calls before it and got the same result), so the round ended at model call "
        f"{ended_at}",
    ]
    # the final merge ran once: what passed landed and the rest is kept as a draft
    assert len(merged) == 1 and out.passed and out.items_merged == [ME] and out.items_refused == {OTHER: ["G2"]}
    assert draft_states(ev.pass_rounds()) == {"px": "open"}
    # the end event carries the field at its top level, beside repair_stuck; a passed pass's codes are ["ok"]
    end = consolidate._end_event(_stores(tmp_path / "s"), "px", out, [], 1.0, None)
    assert end["no_progress"] == out.no_progress and end["repair_stuck"] is None
    assert out.codes == ["ok"] and end["reason_codes"] == ["ok"] and end["refused_by"] == []


def _refuse(writer, merged):
    """The final merge refuses the whole pass under G2."""

    def merge(parent, candidate, manifest, pass_id, kind, channel, usd, **kwargs):
        merged.append(candidate)
        res = GateResult(False, {"G2": False}, [f"G2: {ME} covers no recorded action"], ["G2"])
        writer.gate._record(parent, candidate, pass_id, kind, channel, usd, res)
        return res

    return merge


@needs_bwrap
def test_a_round_ended_by_no_progress_has_its_own_reason_code_beside_the_gate_checks(tmp_path):
    from tests.memory.integration.test_consolidate import _stores
    from unify.memory import writer_pass
    from unify.memory.integration import consolidate

    model = Turns(
        [_call("w0", "execute_code", {"code": FILES_CODE})]
        + [_call(f"n{i}", "execute_code", {"code": NOOP}) for i in range(4 * MEMORY_K)],
    )
    mem, ev, writer = _writer(tmp_path, model, max_usd=None, deadline_s=300.0)
    merged: list = []
    writer.gate.check = lambda *a, **k: GateResult(True, {}, [], [])
    writer.gate.merge = _refuse(writer, merged)
    out = _run(writer)
    assert out.no_progress is not None and out.no_progress["ended"] is True, out.reasons
    assert len(merged) == 1 and not out.passed
    # the stop's own code first (where it arose), then the gate check that refused the final merge
    assert out.codes == [writer_pass.CODE_NO_PROGRESS, "G2"] == ["no_progress", "G2"]
    end = consolidate._end_event(_stores(tmp_path / "s"), "px", out, [], 1.0, None)
    assert end["reason_codes"] == ["no_progress", "G2"]
    # refused_by holds the gate's checks only: the stop is told apart from the gate's refusal
    assert end["refused_by"] == ["G2"]
    assert "no_progress" in consolidate.REASON_CODES


@needs_bwrap
def test_a_round_not_ended_by_no_progress_has_no_such_code(tmp_path):
    from unify.memory import writer_pass

    model = Turns([_call("w0", "execute_code", {"code": FILES_CODE})])
    mem, ev, writer = _writer(tmp_path, model, max_usd=None, deadline_s=300.0)
    merged: list = []
    writer.gate.check = lambda *a, **k: GateResult(True, {}, [], [])
    writer.gate.merge = _refuse(writer, merged)
    out = _run(writer)
    assert out.no_progress is None and not out.passed
    assert writer_pass.CODE_NO_PROGRESS not in out.codes and "G2" in out.codes


@needs_bwrap
def test_a_writer_whose_repeated_cell_changes_its_result_is_never_stopped(tmp_path):
    from tests.memory.integration.test_consolidate import _stores
    from unify.memory.integration import consolidate

    loops = 4 * MEMORY_K
    model = Turns(
        [_call("w0", "execute_code", {"code": FILES_CODE})]
        + [_call(f"n{i}", "execute_code", {"code": COUNTER}) for i in range(loops)],
    )
    mem, ev, writer = _writer(tmp_path, model, max_usd=None, deadline_s=300.0)
    merged: list = []
    writer.gate.check = lambda *a, **k: GateResult(True, {}, [], [])
    writer.gate.merge = _admit(writer, merged)
    out = _run(writer)
    # every scripted call ran, then the writer finished (the scripted model's last turn)
    assert model.queue == [] and out.finished is True, out.reasons
    assert [model.outputs[f"n{i}"].strip() for i in (0, loops - 1)] == ["1", str(loops)]
    assert out.no_progress is None and not any(r.startswith("no progress") for r in out.reasons)
    assert not any(m.get("role") == "user" and NOTICE in str(m.get("content")) for m in model.seen)
    end = consolidate._end_event(_stores(tmp_path / "s"), "px", out, [], 1.0, None)
    assert "no_progress" in end and end["no_progress"] is None


# --- a helper ------------------------------------------------------------------------------------------------


def test_a_helper_repeating_a_no_op_is_told_then_ends_with_what_it_staged(tmp_path, helper_boxes):
    from tests.memory.test_helpers import CANDIDATE, Script, _ctx, _msg, _run as _delegate, _tc

    loops = 3 * MEMORY_K
    script = Script(
        {
            "t": [_msg(_tc("s", "stage", {"path": "candidates/area.py", "text": CANDIDATE}))]
            + [_msg(_tc(f"x{i}", "execute_code", {"code": "print('none')"})) for i in range(loops)],
        },
    )
    ctx = _ctx(tmp_path, script)
    reply = _delegate({"task": "t", "episodes": ["e1"]}, ctx)
    # call 1 stages; every cell after it does nothing: the notice at the K-th, the end at the 2K-th
    ended_at = 2 * MEMORY_K + 1
    assert reply.startswith("helper call-1: no progress: "), reply
    assert "candidates/area.py" in reply
    status = json.loads((ctx.out_root / "call-1" / "helper.json").read_text())
    assert status["stopped_by"] == "no_progress" and status["turns"] == ended_at
    assert status["no_progress"] == {"k": MEMORY_K, "notices": 1, "ended": True, "at_call": ended_at}
    assert [f["path"] for f in status["files"]] == ["candidates/area.py"]
    assert ctx.records[0]["no_progress"] == status["no_progress"]
    messages, _ = ctx.transcripts["call-1"]
    notices = [m for m in messages if m.get("role") == "user" and NOTICE in str(m.get("content"))]
    assert len(notices) == 1 and "call tools.done and keep the item as a draft" in notices[0]["content"]


def test_a_helper_that_finishes_records_no_notice(tmp_path, helper_boxes):
    from tests.memory.test_helpers import Script, _ctx, _drafting, _run as _delegate

    ctx = _ctx(tmp_path, Script(_drafting()))
    reply = _delegate({"task": "draft area", "episodes": ["e1"]}, ctx)
    assert reply.startswith("helper call-1: ok")
    status = json.loads((ctx.out_root / "call-1" / "helper.json").read_text())
    assert status["no_progress"] is None and status["stopped_by"] == "done"
