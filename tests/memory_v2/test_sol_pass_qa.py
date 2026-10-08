"""Sol's side of the stage-5 test checks (memory v2.1): the brief's paragraph and the export's blob references.

With every switch off, Sol's brief and the export are as at the screen build.
"""

import json

from unify.memory_v2.blobs import BlobStore
from unify.memory_v2.episodes import Action
from unify.memory_v2.integration.adapters.dialogue import cap_text
from unify.memory_v2.integration.adapters.tool import TRUNCATED
from unify.memory_v2.qa import REWRITES, QAConfig, system
from unify.memory_v2.sol_pass import SOL_SYSTEM, export_for_sol
from tests.memory_v2.test_episodes import _ep


def _dl(obs, channel="dialogue:user"):
    return Action(-1, channel, "reply", ["submit"], {}, obs, "ok", kind="dialogue")


def F(n):
    return {
        "type": "SubmitFeedback",
        "valid": True,
        "correct": False,
        "failed": False,
        "attempts_used": n,
    }


def test_every_switch_off_leaves_sols_brief_and_export_as_before(tmp_path):
    assert system(SOL_SYSTEM, QAConfig()) is SOL_SYSTEM
    ep = _ep(episode_id="e9", actions=[_dl(F(1))])
    export_for_sol(lambda eid: ep, ["e9"], tmp_path / "a")
    row = json.loads((tmp_path / "a" / "e9.json").read_text())
    assert "response_blobs" not in row and "truncated" not in row


def test_sols_brief_gains_one_paragraph_per_switch_and_its_rewrites():
    for _, old, _ in REWRITES:
        assert old in SOL_SYSTEM  # the rewrites follow the brief
    every = QAConfig(
        fixtures="strict",
        mutation=True,
        determinism=True,
        replay=True,
        fixture_size=True,
    )
    text = system(SOL_SYSTEM, every)
    assert text.startswith(SOL_SYSTEM.split("Run tests as the gate does")[0])
    for _, old, new in REWRITES:
        assert old not in text and new in text
    for phrase in (
        "Drawn inputs:",
        "Every drawn input must be read",
        "Mutants:",
        "at least 0.5",
        "Determinism:",
        "Replay:",
        "Fixture size:",
        "PYTHONPATH=/memory:/inputs",
    ):
        assert phrase in text
    only = system(SOL_SYSTEM, QAConfig(replay=True))
    assert "Replay:" in only and "Mutants:" not in only and "or every\nrecorded" in only


def test_the_export_lists_response_blobs_and_cuts_beside_the_actions(tmp_path):
    big = {"items": [{"sku": f"A-{i}"} for i in range(400)]}
    acts = [
        Action(0, "shop", "list", [], {}, big, "ok"),
        Action(1, "shop", "get", [], {"sku": "A-1"}, {"sku": "A-1"}, "ok"),
        Action(
            2,
            "shop",
            "list",
            [],
            {},
            {TRUNCATED: {"bytes": 9, "shape": "x", "preview": "{"}},
            "ok",
        ),
        _dl(cap_text("You see: " + "grass " * 500, 300), "dialogue:crafter"),
    ]
    ep = _ep(episode_id="e9", actions=acts)
    store = BlobStore(tmp_path / "store")
    export_for_sol(
        lambda eid: ep,
        ["e9"],
        tmp_path / "episodes",
        response_blobs=(store, tmp_path / "blobs"),
    )
    row = json.loads((tmp_path / "episodes" / "e9.json").read_text())
    sha = row["response_blobs"][0]
    assert row["response_blobs"][1:] == [None, None, None]
    assert json.loads((tmp_path / "blobs" / sha).read_bytes()) == big and store.has(sha)
    assert row["truncated"] == [None, None, "end", "middle"]
    for a in row["actions"]:  # the documented rebuild still works
        Action(**{k: v for k, v in a.items() if k != "index"})
