"""Capture quality and adoption, each behind its own switch and off by default.

``UNIFY_MEMORY_CAPTURE_CASES=on``: the fork may write the input->output pairs its own record shows, each citing the
recorded part it came from; a case that cites no part the episode observed is dropped; the capture step runs the
procedure confined on each kept case and records "fits k of n" as a label; a procedure that misses one gets one
re-capture turn told only the misses. ``UNIFY_MEMORY_ASK_PRERUN=on``: ``memory.ask`` runs each captured procedure it
lists once on the current request, confined and bounded, and shows what it gave beside the item. Off, every prompt,
generated file and record is as without them."""

from __future__ import annotations

import io
import json
import os
import shutil
import sys
import time
import types
from pathlib import Path

import pytest

from unify.memory import captures as cp
from unify.memory import library_export as lx
from unify.memory import library_helper as lh
from unify.memory.episodes import Action, Episode
from unify.memory.integration import fork as fork_mod
from unify.memory.integration import switch
from unify.memory.integration.adapters import audit as audit_mod
from unify.memory.sandbox_run import SandboxResult

needs_bwrap = pytest.mark.skipif(shutil.which("bwrap") is None, reason="bubblewrap required")


class _Lazy:
    """The ask section with the pre-run, imported when a test first uses it (so each test fails on its own where
    the section does not exist)."""

    def __getattr__(self, name):
        import importlib

        return getattr(importlib.import_module("unify.memory.library_ask_prerun"), name)


lap = _Lazy()


def _prerun_module():
    import importlib

    return importlib.import_module("unify.memory.library_ask_prerun")

# A world-neutral task: the request carries a list after "Input:", the answer doubles each number. The episode
# observed two more pairs in its actions' responses.
PROC = '''"""Double each number of the request's list. Accepted: "checker: pass"."""

import json


def read_input(request):
    """The list after 'Input:'."""
    return json.loads(request.split("Input:", 1)[1])


def run(value):
    """Each number doubled."""
    return [2 * x for x in value]
'''
# fits its own test, but stores the instance's answer as a procedure: misses the observed pairs
INSTANCE = PROC.replace("return [2 * x for x in value]", "return [2, 4]")
REQUEST = "Double each number. Input: [1, 2]"


def _ep(eid="e1", value=(2, 4), actions=None):
    acts = actions if actions is not None else [
        Action(0, "env", "examples", [], {}, "Input: [3] -> [6]", "ok", "read", kind="dialogue"),
        Action(1, "env", "examples", [], {}, "Input: [5, 1] -> [10, 2]", "ok", "read", kind="dialogue"),
        Action(2, "env", "answer", [list(value)], {}, "ok", "ok", "unknown", kind="dialogue"),
    ]
    return Episode(
        episode_id=eid,
        started_at="2026-10-18T00:00:00Z",
        ended_at="2026-10-18T00:01:00Z",
        build="b",
        model="m",
        effort="low",
        regime="dense",
        memory_main="0" * 40,
        worktree_before=None,
        worktree_after=None,
        request=[REQUEST, "Input: [3] -> [6]"],
        transcript=[],
        cells=[],
        actions=acts,
    )


def _cases(*rows):
    return json.dumps([{"request": f"Double each number. Input: {json.dumps(i)}", "expected": o, "from": f}
                       for i, o, f in rows])


CASES = _cases(([3], [6], "action:0"), ([5, 1], [10, 2], "observation:0"))


def _staging(root: Path, eid: str, files: dict[str, str], status: str = "ok") -> Path:
    d = root / "memory-staging" / eid
    d.mkdir(parents=True)
    for rel, text in files.items():
        (d / rel).write_text(text)
    (d / "fork.json").write_text(json.dumps({"episode": eid, "status": status}))
    return d


def _doubling_box(calls=None, broken=False):
    """A stand-in for the confined box: it reads the request text and the procedure it was handed and answers as
    the procedure would (doubling, or the instance answer [2, 4]). No model code runs here."""

    def runner(argv, *, ro=None, rw=None, cwd="/tmp", timeout_s=120.0, env=None):
        if calls is not None:
            calls.append({"argv": argv, "ro": ro, "rw": rw, "env": env, "timeout_s": timeout_s})
        [case] = [h for h, b in (ro or {}).items() if b == "/case"]
        [out] = [h for h, b in (rw or {}).items() if b == "/out"]
        text = (Path(case) / "request.txt").read_text()
        source = (Path(case) / "procedure.py").read_text()
        value = json.loads(text.split("Input:", 1)[1])
        result = [2, 4] if "return [2, 4]" in source else [2 * x for x in value]
        doc = {"outcome": "returned", "value": json.dumps(value), "result": json.dumps(result),
               "result2": json.dumps(result)}
        if broken:
            doc = {"outcome": "error", "error_class": "KeyError", "stage": "run"}
        (Path(out) / "result.json").write_text(json.dumps(doc))
        return SandboxResult(0, "", "", False)

    return runner


def _settings(**kw):
    base = {
        "UNIFY_MEMORY_LIBRARY": "on",
        "UNIFY_MEMORY_LIBRARY_FORK": "on",
        "UNIFY_MEMORY_LIBRARY_WAIT_SLOT": "on",
        "UNIFY_MEMORY_LIBRARIAN": "on",
    }
    base.update(kw)
    return types.SimpleNamespace(**base)


# --- the switches ---------------------------------------------------------------------------------------------


def test_both_switches_are_off_by_default_and_refused_without_what_they_need():
    assert not switch.capture_cases_on(types.SimpleNamespace())
    assert not switch.ask_prerun_on(types.SimpleNamespace())
    assert switch.parse_capture_cases("") == "off" and switch.parse_ask_prerun(None) == "off"
    with pytest.raises(ValueError):
        switch.parse_capture_cases("yes")
    assert switch.capture_quality_refusal(_settings(UNIFY_MEMORY_CAPTURE_CASES="on")) is None
    assert switch.capture_quality_refusal(_settings(UNIFY_MEMORY_ASK_PRERUN="on")) is None
    why = switch.capture_quality_refusal(_settings(UNIFY_MEMORY_CAPTURE_CASES="on", UNIFY_MEMORY_LIBRARY_FORK="off"))
    assert why and "UNIFY_MEMORY_LIBRARY_FORK=on" in why
    why = switch.capture_quality_refusal(_settings(UNIFY_MEMORY_ASK_PRERUN="on", UNIFY_MEMORY_LIBRARIAN="off"))
    assert why and "UNIFY_MEMORY_LIBRARIAN=on" in why
    assert switch.PARSERS[switch.CAPTURE_CASES] is switch.parse_capture_cases
    assert switch.PARSERS[switch.ASK_PRERUN] is switch.parse_ask_prerun


# --- (A) provenance: a case must cite a recorded part the episode observed --------------------------------------


def test_a_case_without_a_cited_observed_part_is_dropped_and_counted():
    parts = ["request", "observation:0", "cell:0", "action:0", "action:1", "diff"]
    text = json.dumps(
        [
            {"request": "Input: [3]", "expected": [6], "from": "action:0"},  # kept
            {"request": "Input: [4]", "expected": [8], "from": "`observation:0`"},  # kept: backticks as named
            {"request": "Input: [5]", "expected": [10]},  # no citation
            {"request": "Input: [6]", "expected": [12], "from": "action:9"},  # names no recorded part
            {"request": "Input: [7]", "expected": [14], "from": "cell:0"},  # the episode's own work, not observed
            {"request": "Input: [8]", "expected": [16], "from": "diff"},  # the episode's own work
            {"request": "Input: [9]", "expected": [18], "from": "the demo above"},  # prose is no citation
            {"expected": [2], "from": "request"},  # no request text: malformed
            "Input: [1] -> [2]",  # not an object: malformed
        ],
    )
    kept, counts = cp.read_cases(text, parts)
    assert [c["from"] for c in kept] == ["action:0", "observation:0"]
    assert counts == {"file": "read", "dropped": 7, "uncited": 5, "malformed": 2, "over_bound": 0}
    # no file, or not a JSON list: no cases, and said so
    assert cp.read_cases(None, parts) == ([], {"file": "none", "dropped": 0, "uncited": 0, "malformed": 0,
                                              "over_bound": 0})
    assert cp.read_cases("{not json", parts)[1]["file"] == "unreadable"
    # provenance is structural: a cited case is kept whatever its values, and nothing is matched against the record
    kept, _ = cp.read_cases(json.dumps([{"request": "x", "expected": "never seen anywhere", "from": "request"}]), parts)
    assert len(kept) == 1


def test_the_capture_records_fits_k_of_n_as_a_label_and_never_refuses_for_it(tmp_path):
    cases = _cases(([3], [6], "action:0"), ([5, 1], [10, 2], "action:1"), ([4], [8], "cell:3"))
    staging = _staging(tmp_path, "e1", {"procedure.py": INSTANCE, cp.CASES_FILE: cases})
    store = cp.store_dir(tmp_path)
    calls: list = []
    row = cp.capture(staging, _ep(), store, runner=_doubling_box(calls), own_cases=True, clock=lambda: 0.0)
    # its own test reproduces, so it is captured as before: the misses are a label, not a refusal
    assert row["reproduces"] == "reproduces" and row["outcome"] != cp.FAILED
    assert row["own_pairs"] == [0, 2]
    rec = json.loads((store / row["capture_id"] / cp.RECORD_FILE).read_text())
    assert rec["own_pairs"] == {"fits": 0, "of": 2}
    [case] = rec["cases"]
    own = case["own_pairs"]
    assert (own["fits"], own["of"], own["uncited"], own["file"]) == (0, 2, 1, "read")
    assert [r["from"] for r in own["rows"]] == ["action:0", "action:1"] and not any(r["fits"] for r in own["rows"])
    # every case ran confined as the capture's own request does: empty environment, read-only case, only /out
    assert len(calls) == 3 and all(c["env"] == {} and sorted(c["ro"].values()) == ["/case"] for c in calls)
    # the record keeps no value of any case
    text = (store / row["capture_id"] / cp.RECORD_FILE).read_text()
    assert "[10, 2]" not in text and "[10,2]" not in text and "Input:" not in text

    staging2 = _staging(tmp_path, "e2", {"procedure.py": PROC, cp.CASES_FILE: CASES})
    row2 = cp.capture(staging2, _ep("e2"), store, runner=_doubling_box(), own_cases=True, clock=lambda: 0.0)
    assert row2["own_pairs"] == [2, 2]
    rec2 = json.loads((store / row2["capture_id"] / cp.RECORD_FILE).read_text())
    assert rec2["own_pairs"] == {"fits": 2, "of": 2}


def test_the_card_shows_the_label_and_is_unchanged_without_one(tmp_path, monkeypatch):
    root = tmp_path / "copy"
    (root / ".memory" / "captures").mkdir(parents=True)
    (root / ".memory" / "items.json").write_text(json.dumps({"items": {}}))
    cid = "0123456789abcdef"
    d = root / ".memory" / "captures" / cid
    d.mkdir()
    (d / "procedure.py").write_text(PROC)
    (d / "saved.json").write_text(json.dumps({"docstring": "Double each number.", "asked": "Double a list."}))
    base_rec = {"capture_id": cid, "outcome": "positive", "level": "unchecked", "status": "on-trial"}
    (d / "record.json").write_text(json.dumps(base_rec))
    monkeypatch.setattr(lh, "_ROOT", str(root))
    card = str(lh._Catalog("t").item("capture:" + cid))
    expected = (
        f"capture:{cid}  [on-trial, evidence: unchecked]\n"
        "A captured procedure: calling it runs it in a separate confined process on text or JSON values.\n"
        "Asked: Double a list.\n\n" + PROC.rstrip("\n") + "\n"
    )
    assert card == expected  # no label: the card as built
    (d / "record.json").write_text(json.dumps({**base_rec, "own_pairs": {"fits": 1, "of": 3}}))
    card = str(lh._Catalog("t").item("capture:" + cid))
    assert "Fits 1 of 3 input->output pairs recorded in the episode it was captured from.\n" in card
    (d / "record.json").write_text(json.dumps({**base_rec, "own_pairs": {"fits": 0, "of": 0}}))
    assert str(lh._Catalog("t").item("capture:" + cid)) == expected  # nothing cited: no label


@needs_bwrap
def test_a_real_confined_check_fits_its_own_pairs(tmp_path):
    cases, _ = cp.read_cases(CASES, ["request", "observation:0", "action:0"])
    rows, misses = cp.check_cases(PROC, cases, work=tmp_path / "w")
    assert [r["fits"] for r in rows] == [True, True] and misses == []
    rows, misses = cp.check_cases(INSTANCE, cases, work=tmp_path / "w2")
    assert [r["fits"] for r in rows] == [False, False]
    assert [(m["expected"], m["got"]) for m in misses] == [([6], [2, 4]), ([10, 2], [2, 4])]


# --- (A) one re-capture turn ----------------------------------------------------------------------------------


class _Stdin(io.BytesIO):
    def close(self):  # keep the bytes readable after the writer thread closes it
        self.closed_by_writer = True


class _Proc:
    def __init__(self, write):
        self.stdin, self.pid, self.returncode, self._write = _Stdin(), os.getpid(), 0, write

    def wait(self, timeout=None):
        self._write()
        return 0


def _fork(tmp_path, files, eid="e1", status="ok", reply="```path: procedure.py\n...\n```"):
    staging = _staging(tmp_path, eid, files, status)
    (staging / "fork-reply.json").write_text(json.dumps({"episode": eid, "reply_text": reply}))
    f = fork_mod.Fork(proc=None, writer=None, staging=staging, eid=eid)
    f.request = {
        "episode": eid,
        "url": "http://127.0.0.1:9/fork/v1/chat/completions",
        "headers": {"x-unify-call-kind": "actor_fork"},
        "body": {"model": "m", "messages": [{"role": "user", "content": REQUEST}], "tools": []},
    }
    return f


def _launcher(new_source):
    started: list = []

    def popen(argv, **kw):
        d = Path(argv[-1])

        def write():
            if new_source is not None:
                (d / "procedure.py").write_text(new_source)
            (d / "fork.json").write_text(json.dumps({"episode": "e1", "status": "ok"}))

        proc = _Proc(write)
        started.append({"argv": argv, "proc": proc, "kw": kw})
        return proc

    return popen, started


def test_a_procedure_that_misses_its_own_pairs_gets_exactly_one_recapture_turn_told_only_the_misses(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(fork_mod, "box_argv", lambda d, die_with_parent: ["box", str(d)])
    f = _fork(tmp_path, {"procedure.py": INSTANCE, cp.CASES_FILE: CASES})
    paths = types.SimpleNamespace(state_dir=tmp_path)
    popen, started = _launcher(PROC)
    parts = ["request", "observation:0", "action:0", "action:1", "action:2"]
    got = fork_mod.recapture(paths, f, _settings(UNIFY_MEMORY_CAPTURE_CASES="on"), parts=parts, popen=popen,
                             runner=_doubling_box())
    assert got == {"fits": 0, "of": 2, "recaptured": True, "status": "ok"}
    [start] = started
    sent = json.loads(start["proc"].stdin.getvalue())
    msgs = sent["body"]["messages"]
    # the same conversation, then the fork's own reply, then one message with the misses only
    assert msgs[0] == f.request["body"]["messages"][0] and msgs[1]["role"] == "assistant"
    assert msgs[1]["content"] == "```path: procedure.py\n...\n```" and msgs[2]["role"] == "user"
    msg = msgs[2]["content"]
    assert "It gives the expected value for 0 of the 2 pairs" in msg
    assert "pair 0 (from `action:0`): expected [6]; it returned [2, 4]" in msg
    assert "pair 1 (from `observation:0`): expected [10, 2]; it returned [2, 4]" in msg
    assert sent["headers"] == f.request["headers"] and sent["url"] == f.request["url"]
    d = cp.recapture_dir(tmp_path, "e1")
    assert (d / "procedure.py").read_text() == PROC and json.loads((d / "fork.json").read_text())["status"] == "ok"
    # one turn per episode, never more: a second call finds the dir and starts nothing
    again = fork_mod.recapture(paths, f, _settings(UNIFY_MEMORY_CAPTURE_CASES="on"), parts=parts, popen=popen,
                               runner=_doubling_box())
    assert len(started) == 1 and again["recaptured"] is False and again["status"].startswith("refused")
    # the capture step keeps the re-capture (it fits more of the same pairs) and records both counts
    row = cp.capture_settled(tmp_path, ["e1"], episode=lambda e: _ep(e), signals=lambda e: [],
                             checker_visible=True, runner=_doubling_box(), own_cases=True)[0]
    assert row["capture_id"] == cp.capture_id(PROC) and row["own_pairs"] == [2, 2] and row["recapture"] == "recapture"
    rec = json.loads((cp.store_dir(tmp_path) / row["capture_id"] / cp.RECORD_FILE).read_text())
    note = rec["cases"][0]["own_pairs"]["recapture"]
    assert (note["fits_before"], note["fits_after"], note["of"], note["kept"]) == (0, 2, 2, "recapture")


def test_no_recapture_when_every_pair_fits_no_pair_is_cited_the_fork_did_not_settle_ok_or_the_switch_is_off(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(fork_mod, "box_argv", lambda d, die_with_parent: ["box", str(d)])
    paths = types.SimpleNamespace(state_dir=tmp_path)
    parts = ["request", "observation:0", "action:0", "action:1"]
    on = _settings(UNIFY_MEMORY_CAPTURE_CASES="on")
    popen, started = _launcher(PROC)
    f = _fork(tmp_path / "a", {"procedure.py": PROC, cp.CASES_FILE: CASES})
    assert fork_mod.recapture(paths, f, on, parts=parts, popen=popen, runner=_doubling_box()) == {
        "fits": 2, "of": 2, "recaptured": False, "status": None}
    uncited = _cases(([3], [6], "cell:0"))
    f = _fork(tmp_path / "b", {"procedure.py": INSTANCE, cp.CASES_FILE: uncited})
    assert fork_mod.recapture(paths, f, on, parts=parts, popen=popen, runner=_doubling_box()) is None
    f = _fork(tmp_path / "c", {"procedure.py": INSTANCE, cp.CASES_FILE: CASES}, status="timeout")
    assert fork_mod.recapture(paths, f, on, parts=parts, popen=popen, runner=_doubling_box()) is None
    f = _fork(tmp_path / "d", {"procedure.py": INSTANCE, cp.CASES_FILE: CASES})
    assert fork_mod.recapture(paths, f, _settings(), parts=parts, popen=popen, runner=_doubling_box()) is None
    assert started == [] and not (tmp_path / cp.RECAPTURE_DIR).exists()


def test_a_running_recapture_holds_the_capture_and_a_worse_one_is_not_kept(tmp_path):
    _staging(tmp_path, "e1", {"procedure.py": PROC.replace("2 * x", "x + x"), cp.CASES_FILE: CASES})
    d = cp.recapture_dir(tmp_path, "e1")
    d.mkdir(parents=True)
    import fcntl

    fd = os.open(d / "fork.lock", os.O_RDWR | os.O_CREAT, 0o600)
    fcntl.flock(fd, fcntl.LOCK_EX)  # the re-capture turn is still running
    kw = dict(episode=lambda e: _ep(e), signals=lambda e: [], checker_visible=True, runner=_doubling_box())
    assert cp.capture_settled(tmp_path, ["e1"], own_cases=True, **kw) == []
    os.close(fd)
    (d / "procedure.py").write_text(INSTANCE)  # the re-capture fits fewer pairs than the first turn
    (d / "fork.json").write_text(json.dumps({"episode": "e1", "status": "ok"}))
    [row] = cp.capture_settled(tmp_path, ["e1"], own_cases=True, **kw)
    assert row["own_pairs"] == [2, 2] and row["recapture"] == "first"
    assert row["capture_id"] == cp.capture_id(PROC.replace("2 * x", "x + x"))


# --- (A) off: every prompt and record as without it --------------------------------------------------------------


def test_off_the_fork_instruction_is_as_built_and_on_it_adds_only_the_cases_bullet(monkeypatch):
    monkeypatch.setattr(fork_mod, "_settings", lambda: _settings())
    off = fork_mod.fork_instruction()
    assert off in (fork_mod.INSTRUCTION, fork_mod.INSTRUCTION_IN_CODE)
    assert "procedure_cases" not in off
    monkeypatch.setattr(fork_mod, "_settings", lambda: _settings(UNIFY_MEMORY_CAPTURE_CASES="on"))
    on = fork_mod.fork_instruction()
    assert on == off.replace(fork_mod.CAPTURE_BULLETS, fork_mod.CAPTURE_BULLETS + fork_mod.CASES_BULLET, 1)
    # it asks the fork for facts from its own record; it never tells it to verify, submit or trust anything
    for word in ("verify", "submit", "trust", "check against"):
        assert word not in fork_mod.CASES_BULLET.lower()


def test_off_the_capture_record_and_row_are_byte_for_byte_as_without_cases(tmp_path):
    def one(root: Path, files: dict, **kw):
        staging = _staging(root, "e1", files)
        store = cp.store_dir(root)
        row = cp.capture(staging, _ep(), store, runner=_doubling_box(), clock=lambda: 0.0, **kw)
        return row, (store / row["capture_id"] / cp.RECORD_FILE).read_bytes(), (store / cp.EVENTS_FILE).read_bytes()

    plain = one(tmp_path / "plain", {"procedure.py": PROC})
    staged = one(tmp_path / "staged", {"procedure.py": PROC, cp.CASES_FILE: CASES})  # the file is there, switch off
    assert plain == staged
    assert "own_pairs" not in plain[0] and b"own_pairs" not in plain[1] and b"own_pairs" not in plain[2]
    assert one(tmp_path / "off", {"procedure.py": PROC}, own_cases=False) == plain


# --- (B) memory.ask runs each listed capture once on the request ------------------------------------------------


def _copy_with_capture(tmp_path, monkeypatch, source, record=None):
    root = tmp_path / "copy"
    cid = cp.capture_id(source)
    d = root / ".memory" / "captures" / cid
    d.mkdir(parents=True)
    (d / "procedure.py").write_text(source)
    (d / "record.json").write_text(json.dumps(record or {"outcome": "positive"}))
    monkeypatch.setattr(lh, "_ROOT", str(root))
    return cid, d


def test_prerun_is_confined_and_bounded(tmp_path, monkeypatch):
    probe = PROC.replace(
        "    return [2 * x for x in value]",
        "    import os\n"
        "    return [sorted(os.environ), os.getcwd(), os.getpgid(0) == os.getpid()] + [2 * x for x in value]",
    )
    cid, d = _copy_with_capture(tmp_path, monkeypatch, probe)
    rep = lap._prerun_one(cid, REQUEST, None, 30.0)
    assert rep["outcome"] == "returned"
    names, cwd, own_group, *doubled = rep["result"]
    # an empty environment (the interpreter's own locale coercion aside), its capture directory as its working
    # directory, its own process group
    assert set(names) <= {"LC_CTYPE"}, names
    assert os.path.realpath(cwd) == os.path.realpath(d) and own_group is True and doubled == [2, 4]
    slow = PROC.replace("    return [2 * x for x in value]", "    while True:\n        pass")
    cid2, _ = _copy_with_capture(tmp_path / "slow", monkeypatch, slow)
    t0 = time.monotonic()
    rep = lap._prerun_one(cid2, REQUEST, None, 1.0)
    assert rep["outcome"] == "timeout" and time.monotonic() - t0 < 10
    # a refusal is reported as one, never raised
    refusing = PROC.replace(
        "    return [2 * x for x in value]",
        "    class MemoryInputError(ValueError):\n        pass\n    raise MemoryInputError('not mine')",
    )
    cid3, _ = _copy_with_capture(tmp_path / "ref", monkeypatch, refusing)
    assert lap._prerun_one(cid3, REQUEST, None, 30.0)["outcome"] == "refused"


def test_ask_shows_each_listed_captures_output_and_label_beside_it_and_instructs_nothing(tmp_path, monkeypatch):
    cid, _ = _copy_with_capture(tmp_path, monkeypatch, PROC, {"outcome": "positive", "own_pairs": {"fits": 2, "of": 2}})
    item = lh.Item(id="capture:" + cid, kind="function", summary="", applies="", evidence="unchecked",
                   status="on-trial", example="", text="", links=(), history=(), reason="it doubles lists")
    note = lh.Item(id="notes/a/b.md", kind="note", summary="", applies="", evidence="unchecked", status="on-trial",
                   example="", text="", links=(), history=(), reason="related")
    base = lap.Answer([item, note], calls={"capture:" + cid: f"memory.get('capture:{cid}')(memory.request())"})
    monkeypatch.setattr(_prerun_module(), "_listing_ask", lambda q=None: base)
    monkeypatch.setattr(_prerun_module(), "_current_request", lambda: (REQUEST, None))
    events: list = []
    monkeypatch.setattr(_prerun_module(), "_emit_event", lambda *a: events.append(a))
    got = lap.ask()
    assert list(got) == [item, note] and got.calls == base.calls
    assert got.runs["capture:" + cid]["outcome"] == "returned" and got.runs["capture:" + cid]["result"] == [2, 4]
    assert "notes/a/b.md" not in got.runs  # only captured procedures run
    text = str(got)
    assert f"- capture:{cid} (function): it doubles lists\n  call: memory.get('capture:{cid}')(memory.request())\n" \
           "  run once on this request: returned [2, 4]\n" \
           "  Fits 2 of 2 input->output pairs recorded in the episode it was captured from.\n" in text
    # informs, never instructs
    for word in ("submit", "verify", "trust", "should", "use it", "correct"):
        assert word not in text.lower(), word
    # one audit row per listed capture: id, outcome, the digest of what it returned (never the value)
    [(event, rows)] = events
    assert event == lap.PRERUN_EVENT and rows == (("capture:" + cid, "returned", cp.digest([2, 4])),)


def test_ask_with_no_captured_item_listed_runs_nothing_and_prints_as_before(monkeypatch):
    note = lh.Item(id="notes/a/b.md", kind="note", summary="", applies="", evidence="unchecked", status="on-trial",
                   example="", text="", links=(), history=(), reason="related")
    base = lap.Answer([note])
    monkeypatch.setattr(_prerun_module(), "_listing_ask", lambda q=None: base)
    unread = lambda: pytest.fail("nothing to run: the request is not read")  # noqa: E731
    monkeypatch.setattr(_prerun_module(), "_current_request", unread)
    got = lap.ask()
    assert str(got) == str(base) and got.runs == {}


def test_the_audit_keeps_preruns_with_their_ask_and_nothing_without_them():
    a = audit_mod.CellAudit(())
    a.on = True
    a(audit_mod.ASK_EVENT, (("capture:0123456789abcdef",), False, ()))
    assert "preruns" not in a.ask_records[-1]
    a(audit_mod.PRERUN_EVENT, ((("capture:0123456789abcdef", "returned", "0123456789abcdef"),),))
    assert a.ask_records[-1]["preruns"] == [
        {"id": "capture:0123456789abcdef", "outcome": "returned", "digest": "0123456789abcdef"}]
    a(audit_mod.PRERUN_EVENT, ((("capture:0123456789abcdef", "returned", "[2, 4]"),),))  # a value: refused
    assert a.calls_malformed == 1


# --- (B) off: the generated package and its stub are as without it ---------------------------------------------


def test_off_the_package_and_stub_are_byte_for_byte_as_built_and_on_only_append(tmp_path, monkeypatch):
    helper = (Path(lh.__file__)).read_bytes()
    ask_src = lx.ASK_SOURCE.read_bytes()
    assert lx.package_source(librarian=True) == helper + b"\n\n" + ask_src
    assert lx.package_source(librarian=True, prerun=False) == helper + b"\n\n" + ask_src
    on = lx.package_source(librarian=True, prerun=True)
    assert on == helper + b"\n\n" + ask_src + b"\n\n" + lx.ASK_PRERUN_SOURCE.read_bytes()
    assert lx.package_source(librarian=False, prerun=True) == helper  # no librarian, no ask, nothing to run
    assert lx.stub_source(librarian=True, prerun=False) == lx.stub_source(librarian=True)
    stub_on = lx.stub_source(librarian=True, prerun=True).decode()
    assert "is run once on the current request" in stub_on
    monkeypatch.setattr(lx, "_ask_prerun", lambda: False)
    tree = tmp_path / "tree"
    tree.mkdir()
    files = lx.librarian_files(tree)
    assert files[lx.RESERVED_INIT] == helper + b"\n\n" + ask_src
    assert files[lx.RESERVED_STUB] == lx.stub_source(librarian=True)


def test_the_generated_package_with_prerun_imports_and_binds_the_new_ask(tmp_path):
    pkg = tmp_path / "memory"
    pkg.mkdir()
    (pkg / "__init__.py").write_bytes(lx.package_source(librarian=True, prerun=True))
    (pkg / "_contract.py").write_bytes(lx.CONTRACT_SOURCE.read_bytes())
    code = (
        "import sys; sys.path.insert(0, sys.argv[1]); import memory; "
        "print(memory.ask.__doc__.split('.')[0]); print(type(memory.PrerunAnswer).__name__)"
    )
    import subprocess

    out = subprocess.run([sys.executable, "-I", "-c", code, str(tmp_path)], capture_output=True, text=True,
                         timeout=60, env={})
    assert out.returncode == 0, out.stderr[-2000:]
    assert out.stdout.splitlines()[0].startswith("Ask the librarian")
