"""The CURATE pass kind (P6; spec v2.1 §10.3-10.4, §12.3): its brief, inputs, role, trailers and gate."""

import asyncio
import json

from unify.memory_v2.curate import CurateState, curate_system
from unify.memory_v2.sol_pass import PassConfig, SolPass
from unify.memory_v2.trigger import PassRequest
from tests.memory_v2.test_gate import _merged
from tests.memory_v2.test_gate_curate import (
    ACCOUNT,
    ALIAS_IDS,
    IDS,
    WHY,
    _curate_man,
    _parent_files,
)
from tests.memory_v2.test_gate_v21 import EPS, ITEM_ID, world21  # noqa: F401 (fixture)
from tests.memory_v2.test_sol_pass import Turns, _call, _sol, _write, needs_bwrap

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


def test_a_curate_request_without_v21_or_its_state_spends_nothing(tmp_path):
    model = Turns([])
    mem, ev, sol = _sol(tmp_path, model)  # v21 off
    out = asyncio.run(sol.run(PassRequest("curate", None, [], False), "c0"))
    assert not out.passed and out.reasons == [
        "curate: needs memory v2.1 and the library state it curates",
    ]
    assert model.seen == [] and not ev.pass_exists("c0")


@needs_bwrap
def test_a_curate_pass_merges_through_its_own_brief_inputs_role_and_gate(world21):
    mem, ev, blobs, gate = world21
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
            _call("f", "finish", {"summary": "account_id is user_id"}),
        ],
    )
    sol = SolPass(
        mem,
        gate(role="curate"),
        ev,
        load=lambda e: EPS[e],
        model_turn=model,
        config=PassConfig(max_calls=10, v21=True),
        curate=_state(parent),
    )
    out = asyncio.run(sol.run(PassRequest("curate", None, [], False), "c1"))
    assert out.passed, out.reasons
    assert model.seen[0] == {"role": "system", "content": curate_system()}
    first = model.seen[1]["content"]
    assert first.startswith("Pass c1: curate\n") and f"- {FIRED['overlap:f1']}" in first
    assert json.loads(model.outputs["t"])["reasons"] == list(FIRED.values())
    assert "'curate'" in model.outputs["ls"] and "'drafts'" not in model.outputs["ls"]
    assert out.coverage["missing"] == [] and out.coverage["episodes"] == 0
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
