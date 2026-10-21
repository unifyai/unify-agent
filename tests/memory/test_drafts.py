import json

from unify.memory.blobs import BlobStore
from unify.memory.drafts import DRAFT_IDLE_PASSES, draft_states, stage_drafts
from unify.memory.evidence import EvidenceStore


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


def test_a_corrupt_blob_is_marked_never_staged_as_whole(tmp_path):
    """A staged patch or result must hash to its id; a damaged one is a marker line."""
    from unify.memory.drafts import _blob

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


def test_with_no_earlier_pass_both_pointed_files_still_exist(tmp_path):
    """The brief points at the drafts index and the last gate result; neither pointer leads to a refusal."""
    ev = EvidenceStore(tmp_path / "e.sqlite")
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    assert stage_drafts(ev, BlobStore(tmp_path / "b"), inputs) == []
    assert json.loads((inputs / "drafts" / "index.json").read_text()) == {"open": []}
    assert (
        (inputs / "gate" / "previous.md")
        .read_text()
        .startswith("No earlier pass has a gate result yet")
    )
