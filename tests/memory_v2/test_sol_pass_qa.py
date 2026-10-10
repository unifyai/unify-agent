"""Sol's side of the stage-5 test checks (memory v2.1): the brief's paragraph and the export's blob references.

With every switch off, Sol's brief and the export are as at the screen build.
"""

import json

from unify.memory_v2.blobs import BlobStore
from unify.memory_v2.episodes import Action
from unify.memory_v2.integration.adapters.dialogue import cap_text
from unify.memory_v2.integration.adapters.tool import TRUNCATED
from unify.memory_v2.qa import FIXTURE_MAX_BYTES, REWRITES, QAConfig, system
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


def test_with_any_switch_on_sol_is_pointed_at_the_harness_replay_not_a_root_kit_of_its_own():
    """The kit (mounted whenever a switch is on) provides memlab.replay.env_from, so the brief no longer asks
    Sol to write unify_memory_testkit.py to fake the environment; with every switch off it is as before.
    """
    assert "may build the fake environment" in SOL_SYSTEM
    assert '"support":["unify_memory_testkit.py"]' in SOL_SYSTEM
    for cfg in (
        QAConfig(mutation=True),
        QAConfig(replay=True),
        QAConfig(fixtures="on"),
    ):
        text = system(SOL_SYSTEM, cfg)
        assert "may build the fake environment" not in text
        assert '"support":["unify_memory_testkit.py"]' not in text
        assert "memlab.replay.env_from(<recorded actions>)" in text
        # the unfiltered comparison over the function's own rows (I-Q1): an effect filter is vacuous where
        # the recordings' effect is unknown, as for every dialogue action
        assert (
            "env.issued() == memlab.replay.calls(<the function's own recorded actions>)"
            in text
        )
        assert 'env.issued(effect="write")' not in text and "UnknownEffect" in text
    assert "memlab.replay.env_from" not in system(SOL_SYSTEM, QAConfig())


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


def test_the_blob_threshold_is_relative_to_the_fixture_bound_and_catches_crafter_screens(
    tmp_path,
):
    cfg = QAConfig(fixture_size=True)
    assert cfg.response_blob_bytes == FIXTURE_MAX_BYTES // 64 == 1024
    assert QAConfig(fixture_max_bytes=8192).response_blob_bytes == 128
    # 64 inline copies of a response just under the threshold fill the bound; a handful (8) takes 1/8
    assert 64 * cfg.response_blob_bytes == cfg.fixture_max_bytes
    # a whole Crafter screen of ~3.9 KB, under the recorder's 4000-character cap: the old fixed 4096-byte
    # threshold left it inline, so Sol had to copy it into a fixture
    screen = "You see: " + "grass " * 640 + "\n\nYour status:\nhealth: 9"
    assert cfg.response_blob_bytes <= len(json.dumps(screen)) < 4096
    small = "You see: tree\n\nYour status:\nhealth: 9"
    ep = _ep(
        episode_id="e9",
        actions=[_dl(screen, "dialogue:crafter"), _dl(small, "dialogue:crafter")],
    )
    store = BlobStore(tmp_path / "store")
    export_for_sol(
        lambda eid: ep,
        ["e9"],
        tmp_path / "episodes",
        response_blobs=(store, tmp_path / "blobs"),
        blob_min_bytes=cfg.response_blob_bytes,
    )
    row = json.loads((tmp_path / "episodes" / "e9.json").read_text())
    sha, none = row["response_blobs"]
    assert (
        none is None and json.loads((tmp_path / "blobs" / sha).read_bytes()) == screen
    )
    assert "at least 1024 bytes" in system(SOL_SYSTEM, cfg)


def test_sols_brief_says_how_tests_read_blobs_and_that_library_code_never_imports_the_kit():
    text = system(SOL_SYSTEM, QAConfig(determinism=True))
    assert "memlab.inputs.blob(<id>), never by its /inputs path" in text
    assert "Library code outside tests never imports memlab" in text
    assert "pinned to different values on each run" in text
