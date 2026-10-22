"""Identical parts answered with a pointer to where the same agent saw them, finish validating the manifest with
the gate's own parser, and the first repair round priced on the write phase. Built on test_writer_pass's and
test_writer_library_tools' helpers."""

import asyncio
import json
from decimal import Decimal

from unify.memory import writer_pass
from unify.memory.blobs import BlobStore
from unify.memory.evidence import EvidenceStore
from unify.memory.gate import Gate
from unify.memory.library_checks import LibraryConfig
from unify.memory.gitio import Repo
from unify.memory.writer_pass import FINISH_G1_REFUSALS, PassConfig, WriterPass
from unify.memory.trigger import PassRequest
from tests.memory import writer_library_items as wl
from tests.memory.test_writer_pass import Turns, _call, _e1, needs_bwrap
from tests.memory.test_writer_library_repair import KEYED_BAD
from tests.memory.test_writer_library_tools import EPS, _episode, _pass

# --- identical parts ---------------------------------------------------------------------------------------


def test_an_identical_part_is_answered_with_where_it_was_shown(tmp_path):
    turns = [
        _call("a", "read_episode", {"episode": "e1", "part": "request"}),
        _call(
            "b",
            "read_episode",
            {"episode": "e2", "part": "request"},
        ),  # the same bytes as e1's
    ]
    out, model = _pass(tmp_path, turns, max_calls=10)
    assert model.outputs["a"] == '== request ==\n"req"'
    assert model.outputs["b"].startswith(
        "== request ==\nidentical to e1 request (sha256 ",
    )
    assert model.outputs["b"].endswith('read e1 parts=["request"] to see it again')
    assert out.coverage["opened"] == ["e1"]


def test_what_one_agent_was_shown_never_counts_for_another():
    from unify.memory import writer_tools as wt

    helper_shown: dict = {}
    writer_shown: dict = {}
    log = wt.ReadLog()
    args = {"episode": "e1", "parts": ["request"]}
    wt.read_episode(
        EPS["e1"],
        "e1",
        args,
        blob=lambda s: b"",
        log=log,
        shown=helper_shown,
    )
    got = wt.read_episode(
        EPS["e2"],
        "e2",
        {"episode": "e2", "parts": ["request"]},
        blob=lambda s: b"",
        log=log,
        shown=writer_shown,
    )
    assert got.text == '== request ==\n"req"'  # the writer was never shown e1's request


def test_a_part_shown_only_in_part_earns_no_identical_credit(tmp_path):
    long = (
        "x" * 9000
    )  # over one view: e3's request is shown in part, so it is not "shown complete"
    # the same long observation (an identical request would now be one shared block, shown whole)
    EPS["e3"], EPS["e4"] = _episode("e3", request=("first", long)), _episode(
        "e4",
        request=("second", long),
    )
    try:
        turns = [
            _call("a", "read_episode", {"episode": "e3", "part": "observation:0"}),
            _call("b", "read_episode", {"episode": "e4", "part": "observation:0"}),
        ]
        # three unrelated episodes keep the long text in under half the batch: not shared text
        for e in ("e5", "e6", "e7"):
            EPS[e] = _episode(e, request=(f"other {e}",))
        out, model = _pass(
            tmp_path,
            turns,
            eids=("e3", "e4", "e5", "e6", "e7"),
            max_calls=10,
        )
    finally:
        for e in ("e3", "e4", "e5", "e6", "e7"):
            EPS.pop(e, None)
    assert "identical to" not in model.outputs["b"]


# --- finish validates the manifest ------------------------------------------------------------------------


def test_finish_refuses_a_missing_manifest_a_bounded_number_of_times(tmp_path):
    out, model = _pass(tmp_path, [], eids=("e2",), max_calls=10)
    refusal = wl.FINISH_CELL + "not finished: G1: no manifest: write /memory/.pass/manifest.json, then finish"
    refused = [
        m
        for m in model.seen
        if m.get("role") == "tool"
        # each refused finish cell ends its interpreter, so the next one opens with that notice
        and m["content"] in (refusal, wl.STOPPED + refusal)
    ]
    assert len(refused) == FINISH_G1_REFUSALS
    # past the bound finish is accepted, and the final check refuses by name
    assert out.finished and not out.nothing_to_store and not out.passed


def test_finish_names_the_gates_parse_error(tmp_path, monkeypatch):
    monkeypatch.setattr(
        writer_pass,
        "_read_manifest",
        lambda box: (True, {**wl.MANIFEST, "items": "x"}, None),
    )
    turns = [
        _call("f", "finish", {"summary": "s"}),
    ]
    _, model = _pass(tmp_path, turns, eids=("e2",), max_calls=10)
    assert (
        model.outputs["f"]
        == wl.FINISH_CELL + "not finished: G1: malformed manifest: items must be a list"
    )


def test_an_empty_manifest_at_finish_records_nothing_to_store(tmp_path, monkeypatch):
    empty = {**wl.MANIFEST, "items": [], "support": []}
    monkeypatch.setattr(writer_pass, "_read_manifest", lambda box: (True, empty, None))
    turns = [
        _call("f", "finish", {"summary": "nothing to store"}),
    ]
    out, _ = _pass(tmp_path, turns, eids=("e2",), max_calls=10)
    assert out.finished and out.nothing_to_store


# --- the first repair round is priced on the write phase ----------------------------------------------------


@needs_bwrap
def test_a_long_read_still_leaves_a_repair_round(tmp_path, monkeypatch):
    """Round 0 spends 13 turns reading and 2 writing: priced on all of it (0.75 USD) no repair round would fit in
    the 0.25 left; priced on the write phase (0.05), the gate's refusal reaches the writer and the repair lands.
    """
    from unify.common._async_tool import loop_stop

    # twelve identical reads make no progress: the no-progress rule, not under test here, would end them first
    monkeypatch.setattr(loop_stop, "MEMORY_K", 100)
    mem = Repo.init_bare(tmp_path / "mem.git")
    ev = EvidenceStore(tmp_path / "e.sqlite")
    ep = _e1()
    ev.index_episode(ep, "1" * 40)
    gate = Gate(mem, ev, BlobStore(tmp_path / "b"), library_mode=LibraryConfig())
    # round 0 is refused for a key-shaped string (rev 2's gate is labels only: what still refuses drives a round)
    reads = [_call(f"r{i}", "read", {"path": "/inputs"}) for i in range(12)]
    model = Turns(
        [
            _call("dm", "read_episode", {"episode": "e1", "parts": ["request"]}),
            *reads,
            _call("w0", "execute_code", {"code": wl.files_code(KEYED_BAD)}),
            _call("f0", "finish", {"summary": "first try"}),
            _call("w1", "execute_code", {"code": wl.write(f"/memory/{wl.MODULE}", wl.MODULE_SOURCE)}),
            _call("f1", "finish", {"summary": "read the right key"}),
        ],
        usd="0.05",
    )
    writer = WriterPass(
        mem,
        gate,
        ev,
        load=lambda eid: ep,
        model_turn=model,
        config=PassConfig(run_home=str(tmp_path / "run-home"), max_calls=40, max_usd=Decimal("1.00")),
    )
    out = asyncio.run(
        writer.run(PassRequest("incremental", "acct", ["e1"], False), "p1"),
    )
    assert out.rounds == 2, out.reasons
    assert out.passed and out.finished
    assert not any("below the round reserve" in r for r in out.reasons)
