"""The CURATE pass kind: its brief, inputs, role, trailers and gate."""

import asyncio
import json

from unify.memory.curate import CurateState
from unify.memory.library_prompts import curate_brief_now
from unify.memory.writer_pass import PassConfig, WriterPass
from unify.memory.trigger import PassRequest
from tests.memory.test_gate import _merged
from tests.memory.test_gate_curate import (
    ACCOUNT,
    ALIAS_IDS,
    IDS,
    WHY,
    _curate_man,
    _parent_files,
)
from tests.memory.test_gate_library_mode import (
    EPS,
    ITEM_ID,
    library_gate_world,
)  # noqa: F401 (fixture)
from tests.memory.test_writer_pass import Turns, _call, _writer, _write, needs_bwrap

FIRED = {"overlap:f1": f"overlap (antiunify): {ACCOUNT}, {ITEM_ID}"}


def _state(commit):
    cand = {
        "rule": "antiunify",
        "items": [ACCOUNT, ITEM_ID],
        "reasons": ["bodies anti-unify: kept share 1.00, 0 hole(s)"],
        "fingerprint": "overlap:f1",
    }
    return CurateState(
        commit=commit,
        overlap={"version": 1, "candidates": [cand], "truncated": []},
        suspects={},
        index_tokens=120,
        fired=dict(FIRED),
    )


def test_a_curate_request_without_library_or_its_state_spends_nothing(tmp_path):
    model = Turns([])
    mem, ev, writer = _writer(tmp_path, model)  # library_mode off
    out = asyncio.run(writer.run(PassRequest("curate", None, [], False), "c0"))
    assert not out.passed and out.reasons == [
        "curate: needs the library mode and the library state it curates",
    ]
    assert model.seen == [] and not ev.pass_exists("c0")


@needs_bwrap
def test_a_curate_pass_merges_through_its_own_brief_inputs_role_and_gate(
    library_gate_world,
):
    mem, ev, blobs, gate = library_gate_world
    parent = _merged(mem, _parent_files(blobs))
    ev.add_cover(ITEM_ID, "e1", 0)
    ev.add_cover(ACCOUNT, "e2", 0)
    write = "\n".join(
        [
            _write(f"/memory/{IDS}", ALIAS_IDS),
            _write("/memory/.pass/manifest.json", json.dumps(_curate_man())),
        ],
    )
    model = Turns(
        [
            _call("t", "read", {"path": "/inputs/curate/trigger.json"}),
            _call(
                "ls",
                "execute_code",
                {"code": "import os\nprint(sorted(os.listdir('/inputs')))"},
            ),
            _call("w", "execute_code", {"code": write}),
            _call("who", "who", {}),
            _call("f", "finish", {"summary": "account_id is user_id"}),
        ],
    )
    writer = WriterPass(
        mem,
        gate(role="curate"),
        ev,
        load=lambda e: EPS[e],
        model_turn=model,
        config=PassConfig(max_calls=10, library_mode=True),
        curate=_state(parent),
    )
    out = asyncio.run(writer.run(PassRequest("curate", None, [], False), "c1"))
    assert out.passed, out.reasons
    # a curate pass runs on its own record too, with the curator as its agent
    assert writer._pass_record.root == "curator"
    assert (
        model.outputs["who"]
        .splitlines()[1]
        .startswith("- curator (curator, running): curates the library")
    )
    assert model.seen[0] == {
        "role": "system",
        "content": curate_brief_now(
            finish_capped=True,
        ),  # a capped pass states finish's refusal cap
    }
    first = model.seen[1]["content"]
    # why the pass runs is a file (trigger.json), never a list in the first message
    assert (
        first.startswith("Pass c1: a curate pass.") and FIRED["overlap:f1"] not in first
    )
    assert "/inputs/curate/trigger.json" in first
    assert json.loads(model.outputs["t"])["reasons"] == list(FIRED.values())
    assert "'curate'" in model.outputs["ls"] and "'drafts'" not in model.outputs["ls"]
    assert out.coverage["unread"] == [] and out.coverage["episodes"] == 0
    assert (
        ev.db.execute("SELECT kind FROM passes WHERE pass_id='c1'").fetchone()[0]
        == "curate"
    )
    assert [r["pass_id"] for r in ev.pass_rounds("curate")] == [
        "c1",
    ] and ev.pass_rounds("write") == []
    body = mem.run("log", "-1", "--format=%B", out.commit)
    assert (
        f"Why: {WHY}" in body
        and f"Items: {ACCOUNT}" in body
        and f"Items: {ITEM_ID}" in body
    )
    assert ev.aliases()[ACCOUNT]["target"] == ITEM_ID


@needs_bwrap
def test_finish_names_a_reason_the_curate_manifest_neither_changes_nor_keeps(
    library_gate_world,
    monkeypatch,
):
    """Rule 7 at finish: a curator that lands only notes leaves the overlap it was shown to fire again after a later
    WRITE pass, so finish names the reason while the pass can still change or keep an item of it.
    """
    from unify.memory import writer_pass

    mem, ev, blobs, gate = library_gate_world
    parent = _merged(mem, _parent_files(blobs))
    nothing = {
        **_curate_man(),
        "items": [],
        "deleted": [],
        "aliases": {},
        "skeleton": [],
    }
    keep = {
        **nothing,
        "kept": {ACCOUNT: "account_id and user_id read different tables"},
    }
    shown = iter([nothing, keep])
    monkeypatch.setattr(
        writer_pass,
        "_read_manifest",
        lambda box: (True, next(shown), None),
    )
    model = Turns(
        [
            _call("f1", "finish", {"summary": "notes only"}),
            _call("f2", "finish", {"summary": "kept with its reason"}),
        ],
    )
    writer = WriterPass(
        mem,
        gate(role="curate"),
        ev,
        load=lambda e: EPS[e],
        model_turn=model,
        config=PassConfig(max_calls=10, library_mode=True),
        curate=_state(parent),
    )
    out = asyncio.run(writer.run(PassRequest("curate", None, [], False), "c2"))
    refused = model.outputs["f1"]
    assert "G1: 1 reason(s)" in refused and FIRED["overlap:f1"] in refused
    assert "`kept`" in refused
    assert model.outputs.get("f2", "ok") == "ok" and out.finished
