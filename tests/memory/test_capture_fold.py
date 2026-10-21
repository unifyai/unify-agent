"""Rev 2, item 6: captures fold by their shape, nothing is run to classify them, and a pause that computed nothing
starts no fork.

The shape address comes from a procedure's syntax alone: docstrings, comments and local names do not count; literals,
helpers, tool calls and the form do. Same shape: one capture, every occurrence kept (raw bytes included), one ledger
entry. Rev 2: the capture step runs nothing, so its own check is recorded as not run and no ``own_check_failed`` entry is
made. Three synthetic worlds (a dialogue, a tool namespace, a shell-like value job); no ARC; no model, network or key.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from tests.memory.test_attention_ledger import Clock, Fake, _fork, _ledger, _read_all
from tests.memory.test_capture_trace import _ep, _staging
from unify.memory import capture_shape as cs
from unify.memory import capture_trace as ct
from unify.memory import captures as cp
from unify.memory.episodes import Action, Cell

# three worlds, each a procedure and a variant that differs only in prose and local names
WORLDS = {
    "dialogue": (
        '''"""Book the date the user gave last."""


def run(state):
    """The latest date, with the first message's name."""
    first = state.data[0]
    # the name came first
    return {"name": first["name"], "date": state.data[-1]["date"]}
''',
        '''"""Reserve on the day the person asked for, as their final message says."""


def run(s):
    head = s.data[0]
    return {"name": head["name"], "date": s.data[-1]["date"]}
''',
    ),
    "tool": (
        '''"""Write the summary file."""


def run(state, env):
    """Write the report."""
    text = "total " + str(len(state.messages))
    env.files.write("out/report.txt", text)
    return None
''',
        '''def run(st, tools):
    """Save it."""
    body = "total " + str(len(st.messages))  # what to write
    tools.files.write("out/report.txt", body)
    return None
''',
    ),
    "shell": (
        '''"""Count the lines of the listing."""


def read_input(request):
    return request.text


def run(value):
    """Lines that are not blank."""
    rows = [r for r in value.splitlines() if r.strip()]
    return len(rows)
''',
        '''def read_input(req):
    """The listing."""
    return req.text


def run(listing):
    kept = [line for line in listing.splitlines() if line.strip()]
    return len(kept)
''',
    ),
}


# --- the shape address ------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("world", sorted(WORLDS))
def test_prose_comments_and_local_names_do_not_change_the_shape(world):
    a, b = WORLDS[world]
    assert cp.capture_id(a) != cp.capture_id(b)
    sa, sb = cs.shape_of(a), cs.shape_of(b)
    assert sa is not None and sa["shape"] == sb["shape"]


@pytest.mark.parametrize("world", sorted(WORLDS))
def test_a_literal_that_differs_never_folds(world):
    """Two captures that differ only in a literal are two captures."""
    a, _b = WORLDS[world]
    for old, new in (('"total "', '"sum "'), ("[0]", "[1]"), ("[-1]", "[-2]"), ("r.strip()", "r.strip('x')")):
        if old in a:
            assert cs.shape_of(a)["shape"] != cs.shape_of(a.replace(old, new))["shape"], (world, old)
    one = "def run(state):\n    return state.data[0] + 1\n"
    assert cs.shape_of(one)["shape"] != cs.shape_of(one.replace("+ 1", "+ 2"))["shape"]
    assert cs.shape_of(one)["shape"] != cs.shape_of(one.replace("+ 1", "+ 1.0"))["shape"]
    word = 'def run(state):\n    return state.last == "yes"\n'
    assert cs.shape_of(word)["shape"] != cs.shape_of(word.replace('"yes"', '"no"'))["shape"]


def test_an_extra_tool_call_a_helper_or_a_form_changes_the_shape():
    base = WORLDS["tool"][0]
    extra = base.replace("    return None", '    env.files.read("out/report.txt")\n    return None')
    assert cs.shape_of(base)["env_calls"] == ["env.files.write"]
    assert cs.shape_of(extra)["env_calls"] == ["env.files.read", "env.files.write"]
    assert cs.shape_of(base)["shape"] != cs.shape_of(extra)["shape"]
    # the same run with a different helper never folds: the whole module counts
    helper = "def norm(x):\n    return x.strip()\n\n\ndef run(state):\n    return norm(state.last)\n"
    assert cs.shape_of(helper)["shape"] != cs.shape_of(helper.replace("x.strip()", "x.lower()"))["shape"]
    # the form and the parameter count
    one = "def run(state):\n    return state.last\n"
    two = "def run(state, env):\n    return state.last\n"
    assert (cs.shape_of(one)["form"], cs.shape_of(two)["form"]) == ("state", "state_env")
    assert cs.shape_of(one)["shape"] != cs.shape_of(two)["shape"]
    # a global name (a builtin, a module helper) is kept as written
    assert cs.shape_of("def run(s):\n    return len(s.last)\n")["shape"] != cs.shape_of(
        "def run(s):\n    return sorted(s.last)\n"
    )["shape"]
    assert cs.shape_of("def run(:\n") is None
    # nested functions, lambdas and comprehensions rename consistently
    p1 = "def run(state):\n    def inner(a):\n        return [k for k in a]\n    f = lambda z: z + 1\n    return inner(state.data), f(2)\n"
    p2 = "def run(st):\n    def helper(b):\n        return [q for q in b]\n    g = lambda w: w + 1\n    return helper(st.data), g(2)\n"
    assert cs.shape_of(p1)["shape"] == cs.shape_of(p2)["shape"]


# --- folding in the store: one capture, every occurrence kept, nothing run ------------------------------------------------


@pytest.fixture
def nothing_runs(monkeypatch):
    """Fails the test if anything starts a process."""

    def refuse(*a, **k):
        raise AssertionError("a process was started")

    monkeypatch.setattr(subprocess, "Popen", refuse)
    monkeypatch.setattr(os, "fork", refuse)
    monkeypatch.setattr(os, "posix_spawn", refuse, raising=False)
    return refuse


def _episode(world, eid):
    if world == "tool":
        return _ep(eid, actions=[Action(0, "files", "write", ["out/report.txt", "total 1"], {}, {"written": 7}, "ok",
                                        "write", kind="tool")])  # fmt: skip
    if world == "dialogue":
        return _ep(eid, request=['{"name": "Ada"}', 'then {"date": "2026-11-02"}'],
                   actions=[Action(0, "env", "book", [], {"name": "Ada", "date": "2026-11-02"}, "ok", "ok", "unknown",
                                   kind="dialogue")])  # fmt: skip
    return _ep(eid, request=["a\n\nb\nc"], actions=[Action(0, "env", "submit", [3], {}, "ok", "ok", "unknown",
                                                           kind="dialogue")])  # fmt: skip


@pytest.mark.parametrize("world", sorted(WORLDS))
def test_two_occurrences_of_one_shape_are_one_capture_with_both_sources_kept(tmp_path, nothing_runs, world):
    a, b = WORLDS[world]
    store = cp.store_dir(tmp_path)
    rows = [
        cp.capture(_staging(tmp_path, eid, {"procedure.py": src}), _episode(world, eid), store, trace=True,
                   runner=nothing_runs)  # fmt: skip
        for eid, src in (("e1", a), ("e2.s1", b))
    ]
    first, second = rows
    assert first["capture_id"] == cp.capture_id(a) and not first["folded"]
    assert second["capture_id"] == first["capture_id"] and second["folded"]
    assert second["source_capture_id"] == cp.capture_id(b) and second["shape"] == first["shape"]
    dirs = [d.name for d in store.iterdir() if d.is_dir()]
    assert dirs == [first["capture_id"]]  # no second capture directory
    d = store / first["capture_id"]
    rec = json.loads((d / cp.RECORD_FILE).read_text())
    assert rec["occurrences"] == 2 and rec["shape"] == first["shape"] and rec["episodes"] == ["e1", "e2.s1"]
    assert rec["own_check"] == ct.NOT_RUN and all(c["run"] == "not run" for c in rec["cases"])
    assert (d / cp.PROCEDURE_FILE).read_text() == a  # the first occurrence's bytes are the capture's
    assert (d / cp.OCCURRENCES_DIR / "2" / cp.PROCEDURE_FILE).read_text() == b  # the fold's bytes, kept as evidence
    occ = cp.occurrences(store, first["capture_id"])
    assert [(o["n"], o["staging"], o["folded"], o["source_capture_id"]) for o in occ] == [
        (1, "e1", False, cp.capture_id(a)),
        (2, "e2.s1", True, cp.capture_id(b)),
    ]
    assert cp.shape_index(store) == {first["shape"]: first["capture_id"]}


def test_a_store_from_before_the_index_is_indexed_from_its_procedures_and_keeps_every_capture(tmp_path, nothing_runs):
    a, b = WORLDS["tool"]
    store = cp.store_dir(tmp_path)
    for cid, src in ((cp.capture_id(b), b), (cp.capture_id(a), a)):  # two earlier captures of one shape
        (store / cid).mkdir(parents=True)
        (store / cid / cp.PROCEDURE_FILE).write_text(src)
    (store / cp.EVENTS_FILE).write_text(json.dumps({"capture_id": cp.capture_id(a), "episode": "e0"}) + "\n")
    index = cp.shape_index(store)
    assert index == {cs.shape_of(a)["shape"]: cp.capture_id(a)}  # the first made (by its event) names the shape
    assert sorted(d.name for d in store.iterdir() if d.is_dir()) == sorted([cp.capture_id(a), cp.capture_id(b)])
    row = cp.capture(_staging(tmp_path, "e9", {"procedure.py": b}), _episode("tool", "e9"), store, trace=True,
                     runner=nothing_runs)  # fmt: skip
    assert row["capture_id"] == cp.capture_id(a) and row["folded"]


def test_d1_no_own_check_runs_and_no_entry_comes_from_one(tmp_path, nothing_runs):
    from unify.memory import attention as at

    store = cp.store_dir(tmp_path)
    row = cp.capture(_staging(tmp_path, "e1", {"procedure.py": WORLDS["shell"][0]}), _episode("shell", "e1"), store,
                     trace=True, runner=nothing_runs)  # fmt: skip
    assert row["own_check"] == {"ran": False, "why": "rev2: captures are never executed to classify"}
    assert not at.own_check_failed(row)
    # even a row from before rev 2 whose own check failed makes no entry now
    old = {"capture_id": "c" * 16, "episode": "e2", "segment": None, "outcome": "none",
           "own_check": {"ran": True, "error": "ValueError"}}  # fmt: skip
    led = _ledger(tmp_path / "led")
    made = led.sync(Fake(captures={"e2": old}))
    assert made == [] and "own_check_failed" not in {e.kind for e in led.entries().values()}


def test_a_constant_capture_is_served_as_a_worked_example(tmp_path, nothing_runs):
    from types import SimpleNamespace

    from unify.memory.integration import switch

    constant = 'def run(state):\n    """The answer."""\n    return [6, 9]\n'
    store = cp.store_dir(tmp_path)
    row = cp.capture(_staging(tmp_path, "e1", {"procedure.py": constant}), _episode("dialogue", "e1"), store,
                     trace=True, runner=nothing_runs)  # fmt: skip
    assert row["constant"] is True
    under_trace = SimpleNamespace(UNIFY_MEMORY="on", UNIFY_MEMORY_CAPTURE_TRACE="on")
    assert switch.constant_captures_as_examples(under_trace)
    exported = cp.export_files(store, examples=switch.constant_captures_as_examples(under_trace))
    rec = json.loads(exported[f"{cp.EXPORT_DIR}/{row['capture_id']}/{cp.RECORD_FILE}"])
    assert rec[cp.EXAMPLE_KEY]["returns"] == ["[6, 9]"]


# --- the ledger keys new experience on the folded capture -------------------------------------------------------------------


def _row(cid, eid, outcome="none"):
    return {"v": 2, "capture_id": cid, "episode": eid, "segment": None, "outcome": outcome, "own_check": ct.NOT_RUN}


def test_one_entry_per_fold_a_repeat_is_counted_and_a_contrary_occurrence_reopens(tmp_path):
    cid = "f" * 16
    clock = Clock()
    led = _ledger(tmp_path, clock)
    caps = {"e1": _row(cid, "e1")}
    [entry] = led.sync(Fake(forks=[_fork("e1")], captures=dict(caps)))
    caps["e2"] = _row(cid, "e2")
    assert led.sync(Fake(forks=[_fork("e1"), _fork("e2")], captures=dict(caps))) == []
    assert led.get(entry.entry).refs.episodes == ("e1", "e2")  # appended to the open entry
    handed = led.hand("p1", 4)
    assert led.verify_reads("p1", _read_all(handed)) == {entry.entry}
    led.close("p1", [entry.entry], "nothing_to_keep", (), "pass:p1/decisions.jsonl#0")
    # a repeat of the closed capture: counted, no new entry, not handed again
    caps["e3"] = _row(cid, "e3", "positive")
    forks = [_fork(e) for e in ("e1", "e2", "e3")]
    assert led.sync(Fake(forks=forks, captures=dict(caps))) == []
    e = led.get(entry.entry)
    assert e.state == "closed" and e.occurrences == 1 and len(led.entries()) == 1
    assert led.hand("p2", 4) == []
    assert led.sync(Fake(forks=forks, captures=dict(caps))) == []  # recorded once
    # a contrary occurrence reopens it, naming why, with the new episode
    caps["e4"] = _row(cid, "e4", "contrary")
    assert led.sync(Fake(forks=forks + [_fork("e4")], captures=dict(caps))) == []
    e = led.get(entry.entry)
    assert e.state == "open" and e.why == "contrary_occurrence" and "e4" in e.refs.episodes
    assert [x.entry for x in led.hand("p3", 4)] == [entry.entry]


# --- a pause that computed nothing starts no fork ------------------------------------------------------------------------


def test_cells_that_only_call_with_literals_compute_nothing():
    def cells(*codes, language="python"):
        return [Cell(i, c, "", language=language) for i, c in enumerate(codes)]

    assert not cs.computes(cells("env.files.list()", "tools.note(3)", 'env.chat.say("hi", n=-1)', "state.last"))
    for code in ("x = 1", "print(state.last)", "for r in rows: pass", "[r for r in rows]", "def f(): pass",
                 "env.files.write(path, text)", "if ok: env.a.b()", "import json", "env.a.b(*args)"):  # fmt: skip
        assert cs.computes(cells("env.files.list()", code)), code
    assert cs.computes(cells("ls -la", language="bash"))  # another language: counted as computing
    assert cs.computes(cells("def broken(:"))  # does not parse: counted as computing


def test_a_pause_with_no_computation_starts_no_fork_and_its_cells_wait_for_the_next(tmp_path, monkeypatch):
    from tests.memory.integration import test_in_episode as TIE
    from unify.memory.integration import fork as fork_mod
    from unify.memory.integration import in_episode as ie_mod
    from unify.settings import SETTINGS

    monkeypatch.setattr(SETTINGS, "UNIFY_MEMORY_LIBRARY", "on")
    monkeypatch.setattr(SETTINGS, "UNIFY_MEMORY_LIBRARY_FORK", "on")
    monkeypatch.setattr(ie_mod, "_ensure_service", lambda home: None)
    forks = TIE._Forks()
    monkeypatch.setattr(fork_mod, "start", forks.start)
    run, ie = TIE._ie(tmp_path, {TIE.C1: TIE._tree(TIE.LIB)})
    for code in ("env.files.list()", "tools.note(3)"):
        run.code.append(code)
        ie.cell_done()
    ie.mark("actor_call")
    event = ie.at_boundary(object())
    assert event["skipped"] == "no computation" and "segment" not in event and forks.calls == []
    row = json.loads((run.paths.state_dir / "memory-segments" / "checkpoints.jsonl").read_text().splitlines()[-1])
    assert row["skipped"] == "no computation"
    # mixed cells fork, and the fork covers the cells the skipped pause left
    run.code.append("x = 1")
    ie.cell_done()
    ie.mark("actor_call")
    event = ie.at_boundary(object())
    assert event["segment"] == 1 and event["cells"] == [0, 2] and len(forks.calls) == 1
