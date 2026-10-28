"""Per-kind recording: the channel a recorded action maps to, and the export of recorded blobs."""

import json

import pytest

from unify.memory.blobs import BlobStore
from unify.memory.episodes import Action, env_channel
from unify.memory.writer_pass import export_blobs
from tests.memory.test_episodes import _ep


INVOICES = (
    b"vendor_id,invoice_no,amount,due_date,status\n"
    b"V-17,INV-0042,1250.50,2026-10-14,open\n"
    b"V-03,INV-0043,99,2026-10-21,paid\n"
)
INVENTORY = b"sku\tqty\nA-1\t4\n"


def _blobs(tmp_path):
    store = BlobStore(tmp_path / "blobs")
    return store, store.put(INVOICES), store.put(INVENTORY)


def _wt(path, sha, method="read", status="ok"):
    return Action(
        0,
        "worktree:workspace",
        method,
        [path],
        {},
        {"blob_before": sha, "blob_after": sha, "size": 1, "shape": {"format": "csv"}},
        status,
        "read",
        kind="worktree",
    )


# --- the channel mapping ---------------------------------------------------------------------------------


def test_env_channel_maps_kind_qualified_keys_and_keeps_tool_keys():
    assert env_channel("tool", "venmo") == "venmo"
    assert env_channel("tool", "a:b") == "a:b"  # tool keys are never rewritten
    assert env_channel("shell", "shell:uv") == "shell_uv"
    assert env_channel("shell", "shell:python3.12") == "shell_python3_12"
    assert env_channel("worktree", "worktree:workspace") == "worktree_workspace"
    assert env_channel("dialogue", "dialogue:user") == "dialogue_user"
    assert env_channel("shell", "uv") == "uv"  # a bare key names the channel itself
    assert (
        env_channel("shell", "dialogue:user") is None
    )  # a qualifier naming another kind
    assert env_channel("shell", "shell:..") is None


# --- the per-kind rule -----------------------------------------------------------------------------------


# --- the export ------------------------------------------------------------------------------------------


def test_export_blobs_caps_and_lists_skips(tmp_path):
    store, inv, tsv = _blobs(tmp_path)
    big = store.put(b"x" * 5000)
    ep = _ep(
        episode_id="o1",
        actions=[
            _wt("ap/inv.csv", inv),
            _wt("inv.tsv", tsv),
            _wt("big.csv", big),
            _wt("gone.csv", "e" * 64),
            Action(
                0,
                "venmo",
                "me",
                [],
                {},
                {"blob_before": tsv},
                "ok",
            ),  # a tool row: ignored
        ],
    )
    out = tmp_path / "out"
    res = export_blobs(
        lambda e: ep,
        ["o1"],
        store,
        out,
        per_blob_bytes=1000,
        total_bytes=len(INVOICES) + 1,
    )
    assert res["exported"] == [inv]
    assert set(res["skipped"]) == {tsv, big, "e" * 64}
    assert "total cap" in res["skipped"][tsv] and "per-blob cap" in res["skipped"][big]
    assert (out / inv).read_bytes() == INVOICES
    assert json.loads((out / "index.json").read_text()) == res
    with pytest.raises(ValueError):
        export_blobs(lambda e: ep, ["../o1"], store, tmp_path / "x")
