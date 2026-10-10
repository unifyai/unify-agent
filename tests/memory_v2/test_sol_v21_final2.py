"""Memory v2.1 final2 (design r2 §1 and §3, as the lead kept them on 10 Oct): identical parts credited once with
their source, finish validating the manifest with the gate's own parser, and the first repair round priced on
the write phase. Built on test_sol_pass's and test_sol_v21_tools' helpers."""

import asyncio
import json
from decimal import Decimal

from unify.memory_v2 import sol_pass
from unify.memory_v2.blobs import BlobStore
from unify.memory_v2.evidence import EvidenceStore
from unify.memory_v2.gate import Gate
from unify.memory_v2.gitio import Repo
from unify.memory_v2.sol_pass import FINISH_G1_REFUSALS, PassConfig, SolPass
from unify.memory_v2.trigger import PassRequest
from tests.memory_v2.test_gate import ITEM, KIT, MAN, TEST
from tests.memory_v2.test_sol_pass import Turns, _call, _e1, _ME, _write, needs_bwrap
from tests.memory_v2.test_sol_v21_repair import MOD_BAD
from tests.memory_v2.test_sol_v21_tools import EPS, _episode, _pass

# --- §1: identical parts (RUNTIME P2) ----------------------------------------------------------------------


def test_an_identical_part_is_credited_once_with_its_source(tmp_path):
    turns = [
        _call("a", "read_episode", {"episode": "e1", "part": "request"}),
        _call(
            "b",
            "read_episode",
            {"episode": "e2", "part": "request"},
        ),  # the same bytes as e1's
        _call("d", "dismiss", {"episode": "e1", "reason": "nothing reusable"}),
    ]
    out, model = _pass(tmp_path, turns, max_calls=10)
    assert model.outputs["a"] == '"req"'
    assert model.outputs["b"].startswith("identical to e1/request (sha256 ")
    assert model.outputs["b"].endswith("already shown in full; covered")
    s = out.coverage
    assert s["identical_parts"] == {"e2/request": "e1/request"}
    assert "e2" not in s["missing"] and s["covered"] == 1


def test_the_parts_form_credits_an_identical_part_too(tmp_path):
    turns = [
        _call("a", "read_episode", {"episode": "e1", "parts": ["request"]}),
        _call("b", "read_episode", {"episode": "e2", "parts": ["request"]}),
    ]
    out, model = _pass(tmp_path, turns, max_calls=10)
    assert model.outputs["b"].startswith(
        "== request ==\nidentical to e1/request (sha256 ",
    )
    assert out.coverage["identical_parts"] == {"e2/request": "e1/request"}


def test_a_part_shown_only_in_part_earns_no_identical_credit(tmp_path):
    long = (
        "x" * 9000
    )  # over one view: e3's request is shown in part, so it is not "shown complete"
    # the same long observation (an identical request would now be one shared block, shown whole: r5 T2)
    EPS["e3"], EPS["e4"] = _episode("e3", request=("first", long)), _episode(
        "e4",
        request=("second", long),
    )
    try:
        turns = [
            _call("a", "read_episode", {"episode": "e3", "part": "observation:0"}),
            _call("b", "read_episode", {"episode": "e4", "part": "observation:0"}),
        ]
        out, model = _pass(tmp_path, turns, eids=("e3", "e4"), max_calls=10)
    finally:
        del EPS["e3"], EPS["e4"]
    assert not model.outputs["b"].startswith("identical to")
    assert "identical_parts" not in out.coverage


# --- §3: finish validates the manifest (RUNTIME P6) -------------------------------------------------------


def test_finish_refuses_a_missing_manifest_a_bounded_number_of_times(tmp_path):
    turns = [_call("d", "dismiss", {"episode": "e2", "reason": "nothing reusable"})]
    out, model = _pass(tmp_path, turns, eids=("e2",), max_calls=10)
    refused = [
        m
        for m in model.seen
        if m.get("role") == "tool"
        and m["content"].startswith("not finished: G1: no manifest")
    ]
    assert len(refused) == FINISH_G1_REFUSALS
    # past the bound finish is accepted, and the final check refuses by name
    assert out.finished and not out.nothing_to_store and not out.passed


def test_finish_names_the_gates_parse_error(tmp_path, monkeypatch):
    monkeypatch.setattr(
        sol_pass,
        "_read_manifest",
        lambda box: (True, {**MAN, "items": "x"}, None),
    )
    turns = [
        _call("d", "dismiss", {"episode": "e2", "reason": "nothing reusable"}),
        _call("f", "finish", {"summary": "s"}),
    ]
    _, model = _pass(tmp_path, turns, eids=("e2",), max_calls=10)
    assert (
        model.outputs["f"]
        == "not finished: G1: malformed manifest: items must be a list"
    )


def test_an_empty_manifest_at_finish_records_nothing_to_store(tmp_path, monkeypatch):
    empty = {**MAN, "items": [], "support": []}
    monkeypatch.setattr(sol_pass, "_read_manifest", lambda box: (True, empty, None))
    turns = [
        _call("d", "dismiss", {"episode": "e2", "reason": "nothing reusable"}),
        _call("f", "finish", {"summary": "nothing to store"}),
    ]
    out, _ = _pass(tmp_path, turns, eids=("e2",), max_calls=10)
    assert out.finished and out.nothing_to_store


def test_off_finish_is_unchanged(tmp_path):
    from tests.memory_v2.test_sol_pass import _run, _sol

    model = Turns([_call("f", "finish", {"summary": "s"})])
    _, _, sol = _sol(tmp_path, model)  # v21 off
    out = _run(sol)
    assert (
        "f" not in model.outputs
    )  # finish ended the pass at once: no tool reply, no G1 refusal
    assert not out.finished and not out.nothing_to_store


# --- §3: the first repair round is priced on the write phase ------------------------------------------------


@needs_bwrap
def test_a_long_read_still_leaves_a_repair_round(tmp_path):
    """Round 0 spends 13 turns reading and 2 writing: priced on all of it (0.75 USD) no repair round would fit in
    the 0.25 left; priced on the write phase (0.05), the gate's refusal reaches Sol and the repair lands.
    """
    mem = Repo.init_bare(tmp_path / "mem.git")
    ev = EvidenceStore(tmp_path / "e.sqlite")
    ep = _e1()
    ev.index_episode(ep, "1" * 40)
    lookup = lambda eid, i: _ME if (eid, i) == ("e1", 0) else None  # noqa: E731
    gate = Gate(mem, ev, BlobStore(tmp_path / "b"), action_lookup=lookup)
    manifest = {**MAN, "items": [ITEM], "summary": "s"}
    first_try = "\n".join(
        [
            _write("/memory/unify_memory_testkit.py", KIT),
            _write("/memory/env/venmo/tests/test_me.py", TEST),
            _write("/memory/env/venmo/__init__.py", MOD_BAD),
            _write("/memory/.pass/manifest.json", json.dumps(manifest)),
        ],
    )
    from tests.memory_v2.test_gate import MOD

    reads = [_call(f"r{i}", "read", {"path": "/inputs"}) for i in range(12)]
    model = Turns(
        [
            _call(
                "dm",
                "dismiss",
                {"episode": "e1", "reason": "the item below covers it"},
            ),
            *reads,
            _call("w0", "execute_code", {"code": first_try}),
            _call("f0", "finish", {"summary": "first try"}),
            _call(
                "w1",
                "execute_code",
                {"code": _write("/memory/env/venmo/__init__.py", MOD)},
            ),
            _call("f1", "finish", {"summary": "read the right key"}),
        ],
        usd="0.05",
    )
    sol = SolPass(
        mem,
        gate,
        ev,
        load=lambda eid: ep,
        model_turn=model,
        config=PassConfig(max_calls=40, max_usd=Decimal("1.00"), v21=True),
    )
    out = asyncio.run(sol.run(PassRequest("incremental", "venmo", ["e1"], False), "p1"))
    assert out.rounds == 2, out.reasons
    assert out.passed and out.finished
    assert not any("below the round reserve" in r for r in out.reasons)
