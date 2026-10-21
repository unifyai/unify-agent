"""The capture contract under ``UNIFY_MEMORY_CAPTURE_TRACE=on``: a capture records what its episode did.

A procedure takes one of three forms (the request's value, the episode's state, or the state and the episode's tools).
Rev 2 (D1): the capture step runs nothing, so its own check is recorded as not run (``capture_trace.NOT_RUN``); the
confined own check (``run_box``, ``own_check``) is tested on its own below. The value is recorded only when one was
handed back; the outcome comes from signals only; no recurrence row is written. With the switch off everything is as
built.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from unify.memory import capture_trace as ct
from unify.memory import captures as cp
from unify.memory import replay_box as rb
from unify.memory.episodes import Action, Episode
from unify.memory.integration import fork as fork_mod
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


def test_the_replay_answers_only_identical_recorded_calls_and_never_executes():
    rows = [
        {"channel": "files", "method": "read", "args": ["a.txt"], "kwargs": {}, "response": "x", "status": "ok",
         "effect": "read", "kind": "tool"},
        {"channel": "files", "method": "write", "args": ["b.txt", "y"], "kwargs": {}, "response": {"written": 1},
         "status": "ok", "effect": "write", "kind": "tool"},
        {"channel": "worktree:workspace", "method": "write", "args": ["c.txt"], "kwargs": {}, "response": None,
         "status": "ok", "effect": "write", "kind": "worktree"},
        {"channel": "env", "method": "say", "args": ["hi"], "kwargs": {}, "response": "ok", "status": "ok",
         "effect": "unknown", "kind": "dialogue"},
        {"channel": "files", "method": "delete", "args": ["d"], "kwargs": {}, "response": None, "status": "error",
         "effect": "write", "error": "no such file", "kind": "tool"},
    ]  # fmt: skip
    env = rb.ReplayEnv(rows)
    assert env.files.read("a.txt") == "x"
    assert env.files.read("a.txt") == "x"  # a repeated read: answered again
    assert env.files.write("b.txt", "y") == {"written": 1}
    env.files.write("b.txt", "y")  # a repeated write: one more effect than the record holds
    assert env.worktree_workspace.write("c.txt") is None  # by its memory-channel name
    assert env.env.say("hi") == "ok"  # an unknown effect: answered from the record, never executed
    with pytest.raises(rb.RecordedError, match="no such file"):
        env.files.delete("d")
    with pytest.raises(rb.ReplayMiss):
        env.files.write("b.txt", "z")  # a recorded method with other arguments
    with pytest.raises(rb.ReplayMiss):
        env.mail.send("x")  # never recorded
    stats = env.stats()
    assert stats == {"matched": 6, "differed": 2, "missing": 0, "unrecorded": 1, "repeats": 2, "issued": 9}
    assert [c["outcome"] for c in env.calls()][-2:] == ["differed", "unrecorded"]
    # the replay holds no handle to anything real: only the recorded rows
    assert set(vars(env)) == {"_rows", "_table", "_names", "_next", "_used", "max_calls", "issued"}
    with pytest.raises(rb.TooManyCalls):
        small = rb.ReplayEnv(rows, max_calls=1)
        small.files.read("a.txt")
        small.files.read("a.txt")
    # missing: a recorded call never issued
    assert rb.ReplayEnv(rows).stats()["missing"] == 5


def test_the_box_s_channel_rule_is_the_episodes():
    from unify.memory.episodes import env_channel

    for kind, channel in (("tool", "files"), ("worktree", "worktree:workspace"), ("shell", "shell:python3.12"),
                          ("dialogue", "dialogue:user"), ("shell", "worktree:x"), ("dialogue", "env")):  # fmt: skip
        assert rb.alias(kind, channel) == env_channel(kind, channel)


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
    row = cp.capture(staging, ep, store, runner=runner, trace=True)
    cid = cp.capture_id(proc)
    assert row["v"] == 2 and row["capture_id"] == cid and row["safety"] == "ok"
    assert row["entry_point"]["form"] == form
    assert row["trace"]["episode"] == "e1" and row["trace"]["parts"][0] == "request"
    assert "same_as" not in row and "reproduces" not in row
    # rev 2 (D1): nothing ran, and the record says so
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
    row = cp.capture(staging, ep, store, runner=runner, trace=True, signals=[end])
    assert row["value"] == {"present": False}
    assert row["own_check"] == ct.NOT_RUN
    assert row["outcome"] == "positive"  # from the signal only
    assert row["effects_seen"] == {"read": 0, "write": 1, "unknown": 0}
    rec = json.loads((store / row["capture_id"] / "record.json").read_text())
    assert rec["outcome"] == "positive" and not cp.negative(rec)
    # the old contract would have refused it: no value and no work-tree change is not a completed attempt
    old = cp.capture(_staging(tmp_path / "old", "e1", {"procedure.py": VALUE_PROC}), _ep(), cp.store_dir(tmp_path / "old"),
                     runner=_box([6, 9])[0])  # fmt: skip
    assert old["safety"] == "not a completed attempt: nothing was handed back"


def test_a_contrary_signal_makes_negative_knowledge_and_nothing_runs(tmp_path):
    store = cp.store_dir(tmp_path)
    runner, seen = _box([1], form="value", outcome="error", error_class="ValueError", stage="run")
    ep = _ep(actions=[Action(0, "env", "submit", [[6, 9]], {}, "ok", "ok", "unknown", kind="dialogue")], regime="dense")
    row = cp.capture(_staging(tmp_path, "e1", {"procedure.py": VALUE_PROC}), ep, store, runner=runner, trace=True)
    assert seen == [] and row["own_check"] == ct.NOT_RUN and row["outcome"] == "none"
    rec = json.loads((store / row["capture_id"] / "record.json").read_text())
    assert rec["outcome"] == "none" and not cp.negative(rec)
    fail = Signal("s-e2", "e2", "checker", "fail", "2026-10-20T00:00:30Z", refers_to="e2/actions/0",
                  regime="dense", visible_to_actor=True)  # fmt: skip
    other = VALUE_PROC.replace("Each row's total", "Each row's sum")
    ep2 = _ep("e2", actions=ep.actions, regime="dense")
    row2 = cp.capture(_staging(tmp_path, "e2", {"procedure.py": other}), ep2, store, runner=_box([6, 9])[0],
                      trace=True, signals=[fail], checker_visible=True)  # fmt: skip
    assert row2["outcome"] == "contrary" and row2["own_check"] == ct.NOT_RUN
    # its procedure differs from the first only in a docstring: the same shape, so it folded into the first capture
    assert row2["folded"] and row2["capture_id"] == row["capture_id"]
    rec2 = json.loads((store / row2["capture_id"] / "record.json").read_text())
    assert rec2["outcome"] == "contrary" and cp.negative(rec2) and rec2["occurrences"] == 2


# --- 3. a dialogue whose values arrived later ---------------------------------------------------------------------------


@needs_bwrap
def test_a_dialogue_with_later_values_is_a_state_capture_whose_own_check_passes_on_the_recorded_state(tmp_path):
    ep = _ep(
        request=['Please book for me: {"name": "Ada"}', 'I would like {"date": "2026-11-02"}'],
        actions=[Action(0, "env", "book", [], {"name": "Ada", "date": "2026-11-02"}, "ok", "ok", "unknown",
                        kind="dialogue")],  # fmt: skip
    )
    staging = _staging(tmp_path, "e1", {"procedure.py": STATE_PROC})
    row = cp.capture(staging, ep, cp.store_dir(tmp_path), trace=True, work=tmp_path / "w")
    assert row["entry_point"]["form"] == "state" and row["own_check"] == ct.NOT_RUN
    # the own check itself (never run by the capture step under rev 2), on the recorded state
    run = ct.run_box(STATE_PROC, "state", ct.case_doc("state", ep), work=tmp_path / "w2")
    oc = ct.own_check(run, ct.value_of(ep)[0])
    assert oc["ran"] and oc["error"] is None and oc["deterministic"] is True
    assert oc["value_match"] == "matched", oc  # it gives what the episode handed back, from the later message


# --- 4. no same_as row; off is as built ---------------------------------------------------------------------------------


def test_no_recurrence_runs_under_the_trace_and_off_is_as_built(tmp_path, monkeypatch):
    store = cp.store_dir(tmp_path)
    runner, seen = _box([6, 9])
    for eid in ("e1", "e2"):
        cp.capture(_staging(tmp_path, eid, {"procedure.py": VALUE_PROC.replace("Total", f"Total {eid}")}),
                   _ep(eid, actions=[Action(0, "env", "submit", [[6, 9]], {}, "ok", "ok", "unknown", kind="dialogue")]),
                   store, runner=runner, trace=True)  # fmt: skip
    assert seen == []  # rev 2: nothing is run, neither an own check nor a capture on another's request
    rows = [json.loads(x) for x in (store / cp.EVENTS_FILE).read_text().splitlines()]
    assert all("same_as" not in r and "reproduces" not in r for r in rows)
    assert [r["folded"] for r in rows] == [False, True]  # the two differ in a docstring only
    for d in store.iterdir():
        if d.is_dir():
            rec = json.loads((d / "record.json").read_text())
            assert "same_as" not in rec and "recurrence" not in rec
    # off: the same capture as built (recurrence runs the other capture on this request)
    off_store = cp.store_dir(tmp_path / "off")
    runner2, seen2 = _box([6, 9], value=json.dumps({"a": [1, 2, 3], "b": [4, 5]}))
    for eid in ("e1", "e2"):
        cp.capture(_staging(tmp_path / "off", eid, {"procedure.py": VALUE_PROC.replace("Total", f"Total {eid}")}),
                   _ep(eid, actions=[Action(0, "env", "submit", [[6, 9]], {}, "ok", "ok", "unknown", kind="dialogue")]),
                   off_store, runner=runner2)  # fmt: skip
    rows = [json.loads(x) for x in (off_store / cp.EVENTS_FILE).read_text().splitlines()]
    assert len(seen2) == 3 and rows[1]["same_as"] and "v" not in rows[1]
    # the fork's instruction is the built one with the switch off, the trace contract's with it on
    monkeypatch.setattr(fork_mod, "_settings", lambda: SimpleNamespace(UNIFY_MEMORY_CAPTURE_TRACE="off"))
    assert fork_mod.fork_instruction() in (fork_mod.INSTRUCTION, fork_mod.INSTRUCTION_IN_CODE)
    assert fork_mod.CAPTURE_BULLETS in fork_mod.fork_instruction()
    monkeypatch.setattr(fork_mod, "_settings", lambda: SimpleNamespace(UNIFY_MEMORY_CAPTURE_TRACE="on"))
    text = fork_mod.fork_instruction()
    assert fork_mod.CAPTURE_BULLETS_TRACE in text and fork_mod.CAPTURE_BULLETS not in text
    for needle in ("run(state, env)", "run(state)", "read_input(request)", "None when the outcome"):
        assert needle in text


# --- 5. an unrecorded env call is evidence --------------------------------------------------------------------------------


@needs_bwrap
def test_an_unrecorded_env_call_in_the_own_check_is_labelled_evidence_not_a_failure(tmp_path):
    ep = _ep(actions=[_write_action()])
    row = cp.capture(_staging(tmp_path, "e1", {"procedure.py": MISS_PROC}), ep, cp.store_dir(tmp_path), trace=True,
                     work=tmp_path / "w")  # fmt: skip
    assert row["capture_id"] and row["safety"] == "ok" and row["own_check"] == ct.NOT_RUN
    rec = json.loads((cp.store_dir(tmp_path) / row["capture_id"] / "record.json").read_text())
    assert not cp.negative(rec)
    # the own check itself (never run by the capture step under rev 2)
    oc = ct.own_check(ct.run_box(MISS_PROC, "state_env", ct.case_doc("state_env", ep), work=tmp_path / "w1"), {})
    assert oc["error"] == "ReplayMiss" and oc["calls"]["mode"] == "replay"
    assert oc["calls"]["differed"] == 1 and oc["calls"]["missing"] == 1 and oc["calls"]["matched"] == 0
    # and the matching procedure's replay matches every recorded call, nothing executed on the host
    ep2 = _ep("e2", actions=[_write_action()])
    oc2 = ct.own_check(ct.run_box(ENV_PROC, "state_env", ct.case_doc("state_env", ep2), work=tmp_path / "w2"), {})
    assert oc2["calls"] == {"mode": "replay", "matched": 1, "differed": 0, "missing": 0, "unrecorded": 0}
    assert oc2["error"] is None and oc2["deterministic"] is True
    assert not (tmp_path / "out").exists() and not Path("out/report.txt").exists()


# --- 6. confinement refuses the network and the host's files ---------------------------------------------------------------


@needs_bwrap
def test_a_procedure_reading_the_network_or_the_host_files_is_refused_by_confinement(tmp_path):
    secret = tmp_path / "host-only.txt"
    secret.write_text("host")
    host_read = f'''def run(state):
    """Read a host file."""
    with open({str(secret)!r}) as f:
        return f.read()
'''
    net = '''def run(state):
    """Reach the network."""
    import socket
    s = socket.create_connection(("1.1.1.1", 80), timeout=3)
    return "reached"
'''
    for i, proc in enumerate((host_read, net)):
        ep = _ep(f"e{i}")
        oc = ct.own_check(ct.run_box(proc, "state", ct.case_doc("state", ep), work=tmp_path / f"w{i}"), ct.value_of(ep)[0])
        assert oc["ran"] and oc["error"] is not None and oc["value_match"] in ("not_applicable", "unknown"), oc


# --- the bounds ---------------------------------------------------------------------------------------------------------------


def test_a_case_over_the_bound_is_not_run_and_says_why(tmp_path, monkeypatch):
    monkeypatch.setattr(ct, "MAX_CASE_BYTES", 10)
    runner, seen = _box([1])
    out = ct.run_box(VALUE_PROC, "value", ct.case_doc("value", _ep()), work=tmp_path, runner=runner)
    assert out == {"outcome": "not run", "why": "the recorded messages and actions are over the replay bound"}
    assert seen == []
    oc = ct.own_check(out, {"present": True, "digest": "x"})
    assert oc == {"ran": False, "error": None, "refused": False, "deterministic": None, "value_match": "unknown",
                  "calls": {"mode": "not_run", "matched": 0, "differed": 0, "missing": 0, "unrecorded": 0},
                  "why": "the recorded messages and actions are over the replay bound"}  # fmt: skip


@needs_bwrap
def test_each_form_runs_for_real_in_the_box(tmp_path):
    episodes = {
        "value": _ep(),
        "state": _ep(request=['{"name": "Ada"}', 'then {"date": "x"}']),
        "state_env": _ep(actions=[_write_action()]),
    }
    results = {"value": [6, 9], "state": {"name": "Ada", "date": "x"}, "state_env": None}
    for form, proc in FORMS.items():
        out = ct.run_box(proc, form, ct.case_doc(form, episodes[form]), work=tmp_path / form)
        assert out["outcome"] == "returned", (form, out)
        assert out["deterministic"] is True and out["result"] == results[form]
    ep = episodes["state_env"]
    noisy = ENV_PROC.replace("return None", "import random\n    return random.random()")
    assert ct.run_box(noisy, "state_env", ct.case_doc("state_env", ep), work=tmp_path / "n")["deterministic"] is False
