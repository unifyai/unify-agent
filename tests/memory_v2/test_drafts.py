import json

from unify.memory_v2.blobs import BlobStore
from unify.memory_v2.drafts import (
    DRAFT_IDLE_PASSES,
    draft_states,
    drafts_message,
    stage_drafts,
)
from unify.memory_v2.evidence import EvidenceStore


def _row(pid, merged=(), refused=None, patch=None):
    return {
        "pass_id": pid,
        "rounds": 1,
        "round_blobs": [],
        "gate_blob": None,
        "patch_blob": patch,
        "passed": int(not refused),
        "items_merged": list(merged),
        "items_refused": dict(refused or {}),
    }


def test_draft_states_touch_finish_archive():
    assert DRAFT_IDLE_PASSES == 3
    a = _row("a", refused={"m:f": ["G3"], "m:g": ["G2"]}, patch="p" * 64)
    c = _row("c", refused={"m:f": ["G3"]}, patch="q" * 64)
    rows = [a, _row("b"), c, _row("d"), _row("e")]
    # b leaves a idle (1); c names m:f, a touch (0); d and e leave it idle (2): a stays open, and so does c
    assert draft_states(rows) == {"a": "open", "c": "open"}
    # a third idle pass after the last touch archives a; c has d, e and f idle: archived too
    assert draft_states(rows + [_row("f")]) == {"a": "archived", "c": "archived"}
    # landing every item a draft names finishes it, whatever its idle count
    done = [a, _row("b", merged=["m:f"]), _row("c2", merged=["m:g"])]
    assert draft_states(done) == {"a": "finished"}
    # landing one of two items is a touch, not a finish
    assert draft_states(done[:2]) == {"a": "open"}
    # a draft that names no item (its manifest never parsed) is never touched: archived after 3 passes
    blank = [_row("x", patch="r" * 64), _row("y"), _row("z"), _row("w")]
    assert draft_states(blank[:3]) == {"x": "open"}
    assert draft_states(blank) == {"x": "archived"}


def _pass_row(ev, pid, passed, merged, refused, patch):
    ev.record_pass(
        {
            "pass_id": pid,
            "kind": "batched",
            "channel": None,
            "parent": "0" * 40,
            "candidate": "1" * 40,
            "passed": passed,
            "reasons": "[]",
            "usd": "0",
            "patch_blob": patch,
            "merged": None,
            "items_merged": json.dumps(merged),
            "items_refused": json.dumps(refused),
        },
    )


def test_stage_drafts_writes_open_drafts_and_the_last_result(tmp_path):
    ev = EvidenceStore(tmp_path / "e.sqlite")
    blobs = BlobStore(tmp_path / "b")
    # no v2.1 pass yet: no table, the v2 schema
    assert ev.pass_rounds() == []
    assert (
        ev.db.execute("SELECT 1 FROM sqlite_master WHERE name='pass_rounds'").fetchone()
        is None
    )
    patch = blobs.put(b"diff --git a/x b/x\n")
    g_evil = blobs.put(b"# Gate result: pass ../evil, round 0, final (merge)\n")
    g1 = blobs.put(b"# Gate result: pass p1, round 1, final (merge)\n")
    g2 = blobs.put(b"# Gate result: pass p2, round 0, final (merge)\n")
    # a pass id that is not a safe file name is never staged as a directory
    _pass_row(ev, "../evil", 0, [], {}, patch)
    ev.record_pass_rounds(
        {
            "pass_id": "../evil",
            "role": "write",
            "rounds": 1,
            "round_blobs": "[]",
            "gate_blob": g_evil,
        },
    )
    _pass_row(ev, "p1", 0, [], {"memory.x:f": ["G3"]}, patch)
    assert ev.record_pass_rounds(
        {
            "pass_id": "p1",
            "role": "write",
            "rounds": 2,
            "round_blobs": json.dumps(["a" * 64]),
            "gate_blob": g1,
        },
    )
    # the first record of a pass wins
    assert not ev.record_pass_rounds(
        {
            "pass_id": "p1",
            "role": "write",
            "rounds": 9,
            "round_blobs": "[]",
            "gate_blob": None,
        },
    )
    _pass_row(ev, "p2", 1, ["memory.y:g"], {}, None)
    ev.record_pass_rounds(
        {
            "pass_id": "p2",
            "role": "write",
            "rounds": 1,
            "round_blobs": "[]",
            "gate_blob": g2,
        },
    )
    rows = ev.pass_rounds()
    assert [r["pass_id"] for r in rows] == ["../evil", "p1", "p2"]
    assert rows[1]["rounds"] == 2 and rows[1]["round_blobs"] == ["a" * 64]
    assert (
        rows[1]["items_refused"] == {"memory.x:f": ["G3"]}
        and rows[1]["patch_blob"] == patch
    )
    assert rows[2]["items_merged"] == ["memory.y:g"] and rows[2]["patch_blob"] is None
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    staged = stage_drafts(ev, blobs, inputs)
    assert staged == [{"pass_id": "p1", "items": ["memory.x:f"]}]
    assert (inputs / "drafts/p1/patch.diff").read_bytes() == b"diff --git a/x b/x\n"
    assert (
        (inputs / "drafts/p1/gate.md")
        .read_text()
        .startswith("# Gate result: pass p1, round 1")
    )
    # the last pass's full result, landed or not (spec §9.3)
    assert (
        (inputs / "gate/previous.md").read_text().startswith("# Gate result: pass p2")
    )
    assert json.loads((inputs / "drafts/index.json").read_text()) == {
        "open": staged,
        "skipped": ["../evil"],
    }
    assert not (tmp_path / "evil").exists()
    msg = drafts_message(staged, True)
    assert "- p1: memory.x:f" in msg and "/inputs/gate/previous.md" in msg
    assert drafts_message([], False).endswith("- none")


def test_a_corrupt_blob_is_marked_never_staged_as_whole(tmp_path):
    """RUNTIME (T3, m): a staged patch or result must hash to its id; a damaged one is a marker line."""
    from unify.memory_v2.drafts import _blob

    blobs = BlobStore(tmp_path / "b")
    sha = blobs.put(b"diff --git a/x b/x\n")
    assert _blob(blobs, sha, "patch") == b"diff --git a/x b/x\n"
    path = tmp_path / "b" / sha[:2] / sha[2:]  # where BlobStore keeps it
    path.write_bytes(b"diff --git a/x b/x\n+tampered\n")
    assert (
        _blob(blobs, sha, "patch")
        == f"(patch {sha} is corrupt in the blob store)\n".encode()
    )
    assert (
        _blob(blobs, "0" * 64, "patch")
        == f"(patch {'0' * 64} is not in the blob store)\n".encode()
    )
