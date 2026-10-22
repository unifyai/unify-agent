"""The writer's repair rounds on rev 2's library layout: a refused round's full result for the writer, one commit on
the parent, the round budget and deadline, the check tool, and the drafts the next pass sees.

Rev 2's gate is labels only (a red test is a label on its item), so a round is refused here by what still refuses:
a key-shaped string or a file the layout has no place for (:mod:`tests.memory.writer_library_items`).
"""

import asyncio
import json
import time

from unify.memory.blobs import BlobStore
from unify.memory.drafts import draft_states
from unify.memory.evidence import EvidenceStore
from unify.memory.gate import Gate, GateResult
from unify.memory.library_checks import LibraryConfig
from unify.memory.gitio import Repo
from unify.memory.writer_pass import PassConfig, WriterPass
from unify.memory.trigger import PassRequest
from tests.memory import writer_library_items as wl
from tests.memory.test_writer_pass import (
    Turns,
    _call,
    _e1,
    _never,
    _run,
    _writer,
    needs_bwrap,
)

FILES_CODE = wl.files_code()
#: The library with a file the layout has no place for: refused (G1) before anything runs.
MISPLACED_CODE = wl.files_code({**wl.FILES, wl.MISPLACED: "x"})
#: The function reads a key the response does not have (its test fails: a label) and holds a key-shaped string
#: (refused, G6).
KEYED_BAD = {**wl.FILES, wl.MODULE: wl.MODULE_BAD + f'\nTOKEN = "{wl.KEY}"\n'}


def _count(gate, seen=None):
    """Count the pass's gate calls, passing each through to the real gate; *seen* collects (name, candidate) per
    call."""
    calls = {"check": 0, "merge": 0}
    for name in ("check", "merge"):
        real = getattr(gate, name)

        def wrapped(*args, _real=real, _name=name, **kwargs):
            calls[_name] += 1
            if seen is not None:
                seen.append((_name, args[1]))
            return _real(*args, **kwargs)

        setattr(gate, name, wrapped)
    return calls


def _store(tmp_path):
    mem = Repo.init_bare(tmp_path / "mem.git")
    ev = EvidenceStore(tmp_path / "e.sqlite")
    ep = _e1()
    ev.index_episode(ep, "1" * 40)
    gate = Gate(mem, ev, BlobStore(tmp_path / "b"), library_mode=LibraryConfig())
    return mem, ev, ep, gate


@needs_bwrap
def test_repair_round_lands_one_commit_on_the_parent(tmp_path):
    mem, ev, ep, gate = _store(tmp_path)
    seen: list = []
    calls = _count(gate, seen)
    model = Turns(
        [
            _call("dm", "read_episode", {"episode": "e1", "parts": ["request"]}),
            _call("w0", "execute_code", {"code": wl.files_code(KEYED_BAD)}),
            _call("f0", "finish", {"summary": "first try"}),
            _call("r1", "read", {"path": "/inputs/gate/result-0.md"}),
            _call("w1", "execute_code", {"code": wl.write(f"/memory/{wl.MODULE}", wl.MODULE_SOURCE)}),
            _call("f1", "finish", {"summary": "read the right key"}),
        ],
    )
    writer = WriterPass(
        mem,
        gate,
        ev,
        load=lambda eid: ep,
        model_turn=model,
        config=PassConfig(run_home=str(tmp_path / "run-home"), max_calls=20),
    )
    parent = mem.head()
    out = asyncio.run(
        writer.run(PassRequest("incremental", "acct", ["e1"], False), "p1"),
    )
    assert out.passed, out.reasons
    assert out.rounds == 2 and out.round_results == ["/inputs/gate/result-0.md"]
    # round 0's check refused; round 1's check passed, so its candidate went to the one merge
    assert calls == {"check": 2, "merge": 1}
    # what the writer read: the first 8,000 bytes, marked when there is more, after the notice that the refused
    # finish stopped the round's interpreter
    read = model.outputs["r1"]
    assert read.startswith(wl.STOPPED), read
    result = read[len(wl.STOPPED):]
    assert result.startswith("# Gate result: pass p1, round 0, check (not merged)"), result
    repair_msgs = [
        m for m in model.seen if m.get("role") == "user" and "This is repair round 1." in m["content"]
    ]
    assert len(repair_msgs) == 1 and "/inputs/gate/result-0.md" in repair_msgs[0]["content"]
    # one commit on the parent: the refused round never enters main's history
    assert mem.head() == out.commit
    assert mem.run("rev-list", "--count", f"{parent}..{out.commit}").strip() == "1"
    assert "Round:" not in mem.run("log", "-1", "--format=%B", out.commit)
    # the merge lands the last checked commit
    c0, c1, m = seen
    assert [n for n, _ in seen] == ["check", "check", "merge"]
    assert m[1] == c1[1] == out.commit
    (row,) = ev.pass_rounds()
    assert row["pass_id"] == "p1" and row["rounds"] == 2 and len(row["round_blobs"]) == 1
    assert row["patch_blob"] is None and draft_states(ev.pass_rounds()) == {}  # landed whole: no draft
    full = writer.gate.blobs.get(row["round_blobs"][0]).decode()  # the same text, kept whole
    assert full.startswith(result.split("\n[… shown bytes ")[0])
    assert f"G6: key-shaped string in {wl.MODULE}" in full and wl.KEY not in full
    assert "## Test output" not in full  # the gate runs no test (rev 2.1): there is no output to keep


@needs_bwrap
def test_gate_time_counts_against_the_deadline(tmp_path):
    model = Turns(
        [
            _call("w", "execute_code", {"code": FILES_CODE}),
            _call("f", "finish", {"summary": "s"}),
        ],
    )
    mem, ev, writer = _writer(tmp_path, model, deadline_s=6.0)
    seen = []

    def slow_check(parent, candidate, manifest, **kwargs):
        seen.append(candidate)
        time.sleep(4.0)  # the gate's time
        return GateResult(
            False,
            {"G1": False},
            [wl.MISPLACED_REASON],
            ["G1"],
            items_refused={wl.ITEM_ID: ["G1"]},
        )

    writer.gate.check = slow_check
    out = _run(writer)
    # the check ran once and was written for the writer; after it, less time was left than a round takes
    assert len(seen) == 1 and out.rounds == 1
    assert out.round_results == ["/inputs/gate/result-0.md"]
    assert any(
        r.startswith("repair: no round 1: ") and "below the round time reserve" in r
        for r in out.reasons
    ), out.reasons
    # the round's candidate is one commit on the parent that no ref names
    assert mem.run("rev-list", "--count", f"{mem.head()}..{seen[0]}").strip() == "1"
    assert mem.run("branch", "--contains", seen[0]).strip() == ""
    assert "Round:" not in mem.run("log", "-1", "--format=%B", seen[0])  # Amendment A: no round trailer at all


@needs_bwrap
def test_check_is_the_whole_gate_and_runs_no_test_under_library_mode(tmp_path):
    """``check`` is the whole gate (gate.check, no merge), which runs no code (rev 2.1): a wrong function's test is
    not run, so the check passes and says the writer's tests are its own; the pass lands."""
    mem, ev, ep, gate = _store(tmp_path)
    calls = _count(gate)
    model = Turns(
        [
            _call("dm", "read_episode", {"episode": "e1", "parts": ["request"]}),
            _call("w0", "execute_code", {"code": wl.files_code({**wl.FILES, wl.MODULE: wl.MODULE_BAD})}),
            _call("c0", "check", {"manifest": json.dumps(wl.MANIFEST)}),
            _call("w1", "execute_code", {"code": wl.write(f"/memory/{wl.MODULE}", wl.MODULE_SOURCE)}),
            _call("f1", "finish", {"summary": "read the right key"}),
        ],
    )
    writer = WriterPass(
        mem,
        gate,
        ev,
        load=lambda eid: ep,
        model_turn=model,
        config=PassConfig(run_home=str(tmp_path / "run-home"), max_calls=20),
    )
    out = asyncio.run(
        writer.run(PassRequest("incremental", "acct", ["e1"], False), "p1"),
    )
    assert model.outputs["c0"] == wl.CHECK_OK
    assert calls["check"] == 2 and out.passed and calls["merge"] == 1
    assert len(ev.pass_rounds()) == 1  # the check tool wrote nothing of its own to the evidence store


@needs_bwrap
def test_a_long_check_result_is_a_file_read_can_page(tmp_path):
    """P2T4-1: a check refusal over the view bound is written to /inputs/gate/check-<k>.md, so the marker's
    offset can be read; nothing in it is unreachable."""
    mem, ev, ep, gate = _store(tmp_path)
    long = f"G1: file {wl.MISPLACED}: " + "w" * 9000 + " THE-END"
    gate.check = lambda parent, candidate, manifest, **k: GateResult(
        False,
        {"G1": False},
        [long],
        ["G1"],
        items_refused={wl.ITEM_ID: ["G1"]},
    )
    model = Turns(
        [
            _call("dm", "read_episode", {"episode": "e1", "parts": ["request"]}),
            _call("w0", "execute_code", {"code": FILES_CODE}),
            _call("c0", "execute_code", {"code": wl.paged("check", {"manifest": json.dumps(wl.MANIFEST)})}),
            _call("r0", "read", {"path": "/inputs/gate/check-0.md", "offset": 8000}),
        ],
    )
    writer = WriterPass(
        mem,
        gate,
        ev,
        load=lambda eid: ep,
        model_turn=model,
        config=PassConfig(run_home=str(tmp_path / "run-home"), max_calls=20),
    )
    asyncio.run(writer.run(PassRequest("incremental", "acct", ["e1"], False), "p1"))
    reply = model.outputs["c0"]
    assert reply.startswith("# Gate result: pass p1, round 0, check (not merged)")
    assert "next: offset=8000]" in reply and reply.endswith("(full result: /inputs/gate/check-0.md)")
    assert "THE-END" not in reply and "THE-END" in model.outputs["r0"]  # the rest is reachable by read


@needs_bwrap
def test_next_pass_sees_open_drafts_and_the_last_full_result(tmp_path):
    first = Turns(
        [
            _call("w", "execute_code", {"code": MISPLACED_CODE}),
            _call("f", "finish", {"summary": "s"}),
        ],
        usd="0.30",
    )
    # its two calls are all it has, so no repair round follows the refused check: round 0 is the final merge
    mem, ev, writer = _writer(tmp_path, first, max_calls=2)
    assert not _run(writer, "p1").passed  # refused at the one merge: a draft
    probe = (
        "import json, os\n"
        "print(json.dumps([sorted(os.listdir('/inputs')), sorted(os.listdir('/inputs/drafts/p1')),\n"
        "    open('/inputs/drafts/p1/patch.diff').read().startswith('diff --git')]))\n"
    )
    second = Turns(
        [
            _call("ls", "execute_code", {"code": probe}),
            _call("g", "read", {"path": "/inputs/drafts/p1/gate.md"}),
            _call("pv", "read", {"path": "/inputs/gate/previous.md"}),
        ],
    )
    sol2 = WriterPass(
        mem,
        writer.gate,
        ev,
        load=_never,
        model_turn=second,
        config=PassConfig(run_home=str(tmp_path / "run-home"), max_calls=10),
    )
    _run(sol2, "p2")
    top, draft, is_patch = json.loads(second.outputs["ls"])
    # previous_gate.json (200-character lines through the notes filter) is replaced by drafts and full results
    assert "previous_gate.json" not in top and "drafts" in top and "gate" in top
    assert draft == ["gate.md", "patch.diff"] and is_patch
    assert second.outputs["g"].startswith("# Gate result: pass p1, round 0, final (merge)")
    assert second.outputs["pv"] == second.outputs["g"]  # p1 was the last pass
    # the drafts are files the brief points to; the first message lists none of them
    first_message = second.seen[1]["content"]
    assert "Open drafts" not in first_message and wl.ITEM_ID not in first_message
    assert "/inputs/drafts/" in second.seen[0]["content"]
