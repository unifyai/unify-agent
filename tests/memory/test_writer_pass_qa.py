"""The writer's side of the stage-5 test checks: the brief's paragraph and the export's blob references.

With every switch off, the writer's brief and the export are as at the screen build.
"""

import json

from unify.memory.blobs import BlobStore
from unify.memory.episodes import Action
from unify.memory.integration.adapters.dialogue import cap_text
from unify.memory.integration.adapters.tool import TRUNCATED
from unify.memory.writer_pass import export_for_writer
from tests.memory.test_episodes import _ep


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
    export_for_writer(
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


