"""The capture contract: a capture records what its episode did.

A procedure takes one of three forms (the request's value, the episode's state, or the state and the episode's tools).
The capture step runs nothing, so its own check is recorded as not run (``capture_trace.NOT_RUN``). The value is
recorded only when one was handed back; the outcome comes from signals only; no recurrence row is written.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from unify.memory import capture_trace as ct
from unify.memory import captures as cp
from unify.memory.episodes import Action, Episode
from unify.memory.sandbox_run import SandboxResult
from unify.memory.signals import Signal

needs_bwrap = pytest.mark.skipif(shutil.which("bwrap") is None, reason="bubblewrap required")

VALUE_PROC = '''"""Total each named row of the request's table. Accepted: not shown."""


def read_input(request):
    """The request's rows, by name."""
    return request.data[0]["rows"]


def run(value):
    """Each row's total, in name order."""
    return [sum(value[k]) for k in sorted(value)]
'''

STATE_PROC = '''"""Book the date the user gave in their last message. Accepted: not shown."""


def run(state):
    """The date from the latest message's JSON, with the name from the first."""
    return {"name": state.data[0]["name"], "date": state.data[-1]["date"]}
'''

ENV_PROC = '''"""Write the summary file, then read it back. Accepted: not shown."""


def run(state, env):
    """Write the report and check it."""
    env.files.write("out/report.txt", "total 6")
    return None
'''

MISS_PROC = '''"""Write the summary file under a different name. Accepted: not shown."""


def run(state, env):
    """Write the report."""
    env.files.write("out/other.txt", "total 6")
'''

FORMS = {"value": VALUE_PROC, "state": STATE_PROC, "state_env": ENV_PROC}


def _ep(eid="e1", *, request=None, actions=(), regime="none", replies=(), worktree=(None, None)):
    return Episode(
        episode_id=eid,
        started_at="2026-10-20T00:00:00Z",
        ended_at="2026-10-20T00:01:00Z",
        build="b",
        model="m",
        effort="low",
        regime=regime,
        memory_main="0" * 40,
        worktree_before=worktree[0],
        worktree_after=worktree[1],
        request=list(request or ['Total these: {"rows": {"a": [1, 2, 3], "b": [4, 5]}}']),
        transcript=[],
        cells=[],
        actions=list(actions),
        replies=list(replies),
    )


def _write_action(path="out/report.txt", text="total 6", status="ok"):
    return Action(0, "files", "write", [path, text], {}, {"written": len(text)}, status, "write", kind="tool")


def _staging(root: Path, name: str, files: dict[str, str]) -> Path:
    d = root / "memory-staging" / name
    d.mkdir(parents=True)
    for rel, text in files.items():
        (d / rel).write_text(text)
    (d / "fork.json").write_text(json.dumps({"episode": name, "status": "ok"}))
    return d


def _box(result=None, *, outcome="returned", calls=None, form="value", **extra):
    """A stand-in for the confined box: it writes the result a box would; no model code runs."""
    seen: list = []

    def runner(argv, *, ro=None, rw=None, cwd="/tmp", timeout_s=120.0, env=None):
        [case] = [h for h, b in (ro or {}).items() if b == "/case"]
        doc_path = Path(case) / "case.json"  # the trace contract's case (the built contract writes request.txt)
        seen.append({"argv": argv, "ro": ro, "rw": rw, "env": env,
                     "case": json.loads(doc_path.read_text()) if doc_path.exists() else None})  # fmt: skip
        [out] = [h for h, b in (rw or {}).items() if b == "/out"]
        doc = {"outcome": outcome, "form": form, **extra}
        if outcome == "returned":
            doc["result"] = json.dumps(result)
            doc["result2"] = json.dumps(result)
        if calls is not None:
            doc["calls"] = calls
        (Path(out) / "result.json").write_text(json.dumps(doc))
        return SandboxResult(0, "", "", False)

    return runner, seen


# --- the form and the replay stand-in, host-side --------------------------------------------------------------------


def test_the_form_is_read_from_the_code():
    assert ct.entry_point(VALUE_PROC)["form"] == "value"
    assert ct.entry_point(STATE_PROC) == {"name": "run", "params": ["state"], "form": "state"}
    assert ct.entry_point(ENV_PROC) == {"name": "run", "params": ["state", "env"], "form": "state_env"}
    assert ct.entry_point("def main(x):\n    return x\n") is None
    assert ct.safety("def main(x):\n    return x\n") == "defines no run"
    assert ct.safety(STATE_PROC) is None and ct.safety("def run(:\n") .startswith("does not parse")
    # the old contract's safety still needs read_input: the trace contract alone takes the state forms
    assert cp.safety(STATE_PROC) == "defines no read_input"


def test_trace_pointers_value_and_effects():
    ep = _ep(request=["first", "second", "third"], actions=[_write_action(), _write_action("x", "y")])
    ep.cells = []
    t = ct.trace_of(ep)
    assert t == {"episode": "e1", "parts": ["request", "observation:0..1", "action:0..1"]}
    assert ct.effects_seen(ep) == {"read": 0, "write": 2, "unknown": 0}
    absent, _ = ct.value_of(_ep(replies=[], worktree=("a", "b")))
    assert absent == {"present": False}
    present, value = ct.value_of(_ep(actions=[Action(0, "env", "submit", [[6, 9]], {}, "ok", "ok", "unknown",
                                                     kind="dialogue")]))  # fmt: skip
    assert present["present"] and present["digest"] == ct.digest([6, 9]) and value == [6, 9]
    assert ct.state_data(['{"a": 1} then [2]', 'none', '{"b": 3}']) == [{"a": 1}, [2], {"b": 3}]


# --- 1. each form is captured, exported and shown to the writer -------------------------------------------------------


@pytest.mark.parametrize("form", ["value", "state", "state_env"])
def test_each_form_is_captured_exported_and_shown_to_the_writer(tmp_path, form):
    proc = FORMS[form]
    staging = _staging(tmp_path, "e1", {"procedure.py": proc, "summary.md": "## Asked\nTotal rows.\n"})
    store = cp.store_dir(tmp_path)
    calls = {"matched": 1, "differed": 0, "missing": 0, "unrecorded": 0, "repeats": 0} if form == "state_env" else None
    runner, seen = _box([6, 9] if form != "state_env" else None, form=form, calls=calls)
    ep = _ep(actions=[_write_action()] if form == "state_env" else [])
    row = cp.capture_trace_step(staging, ep, store)
    cid = cp.capture_id(proc)
    assert row["v"] == 2 and row["capture_id"] == cid and row["safety"] == "ok"
    assert row["entry_point"]["form"] == form
    assert row["trace"]["episode"] == "e1" and row["trace"]["parts"][0] == "request"
    assert "same_as" not in row and "reproduces" not in row
    # rev 2: nothing ran, and the record says so
    assert seen == [] and row["own_check"] == ct.NOT_RUN and row["run"] == "not run"
    rec = json.loads((store / cid / "record.json").read_text())
    assert rec["v"] == 2 and rec["entry_point"]["form"] == form and rec["own_check"] == row["own_check"]
    assert "same_as" not in rec and "recurrence" not in rec and "reproduces" not in rec
    # exported to the actor's copy and to the writer's inputs with its labels
    exported = cp.export_files(store)
    assert json.loads(exported[f".memory/captures/{cid}/record.json"])["own_check"] == row["own_check"]
    dest = tmp_path / "inputs" / "captures"
    assert cp.export_for_writer(store, [cid], dest) == [cid]
    shown = json.loads((dest / cid / "record.json").read_text())
    assert shown["entry_point"]["form"] == form and "trace" in shown and "value" in shown


# --- 2. an episode with only side effects --------------------------------------------------------------------------------


def test_a_side_effect_only_episode_is_captured_with_no_value_and_its_calls(tmp_path):
    staging = _staging(tmp_path, "e1", {"procedure.py": ENV_PROC})
    store = cp.store_dir(tmp_path)
    calls = {"matched": 1, "differed": 0, "missing": 0, "unrecorded": 0, "repeats": 0}
    runner, _ = _box(None, form="state_env", calls=calls)
    ep = _ep(actions=[_write_action()], replies=["done"], worktree=("a", "b"), regime="sparse")
    end = Signal("s1", "e1", "provenance", "support", "2026-10-20T00:00:30Z", regime="sparse")
    row = cp.capture_trace_step(staging, ep, store, signals=[end])
    assert row["value"] == {"present": False}
    assert row["own_check"] == ct.NOT_RUN
    assert row["outcome"] == "positive"  # from the signal only
    assert row["effects_seen"] == {"read": 0, "write": 1, "unknown": 0}
    rec = json.loads((store / row["capture_id"] / "record.json").read_text())
    assert rec["outcome"] == "positive" and not cp.negative(rec)
    # the old contract would have refused it: no value and no work-tree change is not a completed attempt
    old = cp.capture_trace_step(_staging(tmp_path / "old", "e1", {"procedure.py": VALUE_PROC}), _ep(), cp.store_dir(tmp_path / "old"))  # fmt: skip
    assert old["safety"] == "not a completed attempt: nothing was handed back"


def test_a_contrary_signal_makes_negative_knowledge_and_nothing_runs(tmp_path):
    store = cp.store_dir(tmp_path)
    runner, seen = _box([1], form="value", outcome="error", error_class="ValueError", stage="run")
    ep = _ep(actions=[Action(0, "env", "submit", [[6, 9]], {}, "ok", "ok", "unknown", kind="dialogue")], regime="dense")
    row = cp.capture_trace_step(_staging(tmp_path, "e1", {"procedure.py": VALUE_PROC}), ep, store)
    assert seen == [] and row["own_check"] == ct.NOT_RUN and row["outcome"] == "none"
    rec = json.loads((store / row["capture_id"] / "record.json").read_text())
    assert rec["outcome"] == "none" and not cp.negative(rec)
    fail = Signal("s-e2", "e2", "checker", "fail", "2026-10-20T00:00:30Z", refers_to="e2/actions/0",
                  regime="dense", visible_to_actor=True)  # fmt: skip
    other = VALUE_PROC.replace("Each row's total", "Each row's sum")
    ep2 = _ep("e2", actions=ep.actions, regime="dense")
    row2 = cp.capture_trace_step(_staging(tmp_path, "e2", {"procedure.py": other}), ep2, store, 
                      signals=[fail])  # fmt: skip
    assert row2["outcome"] == "contrary" and row2["own_check"] == ct.NOT_RUN
    # its procedure differs from the first only in a docstring: the same shape, so it folded into the first capture
    assert row2["folded"] and row2["capture_id"] == row["capture_id"]
    rec2 = json.loads((store / row2["capture_id"] / "record.json").read_text())
    assert rec2["outcome"] == "contrary" and cp.negative(rec2) and rec2["occurrences"] == 2
