"""Per-kind admission: covers per action kind, channel mapping and recorded-blob export."""

import json
import shutil

import pytest

from unify.memory.blobs import BlobStore
from unify.memory.episodes import Action, env_channel
from unify.memory.evidence import EvidenceStore
from unify.memory.gate import Gate
from unify.memory.library_checks import LibraryConfig
from unify.memory.gitio import Repo
from unify.memory.writer_pass import export_blobs
from tests.memory.test_episodes import _ep

needs_bwrap = pytest.mark.skipif(
    shutil.which("bwrap") is None,
    reason="bubblewrap required",
)

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


def _sh(tail, code=1, channel="shell:uv"):
    return Action(
        0,
        channel,
        "run",
        ["uv run pytest"],
        {},
        None if tail is None else {"exit_code": code, "tail": tail},
        "error" if code else "ok",
        kind="shell",
    )


def _dl(obs, status="ok", channel="dialogue:user"):
    return Action(-1, channel, "reply", ["do"], {}, obs, status, kind="dialogue")


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


# --- G2 through the gate ---------------------------------------------------------------------------------

READER = '''__all__ = ["read_invoices"]


def read_invoices(data):
    """Parse an accounts-payable invoices CSV (bytes) into dicts; raise on another layout.

    Effect: read
    Input: bytes
    """
    lines = data.decode("utf-8").splitlines()
    cols = ["vendor_id", "invoice_no", "amount", "due_date", "status"]
    if not lines or lines[0].split(",") != cols:
        raise ValueError("expected comma-delimited columns " + ",".join(cols))
    return [dict(zip(cols, ln.split(","))) for ln in lines[1:] if ln]
'''
READER_TEST = """import pathlib

import pytest

from env.worktree_workspace import read_invoices

REC = pathlib.Path(__file__).parent / "rec"


def test_reads_every_recorded_invoice_file():
    rows = read_invoices((REC / "invoices.csv").read_bytes())
    assert [r["invoice_no"] for r in rows] == ["INV-0042", "INV-0043"]


def test_rejects_the_recorded_tab_delimited_file():
    with pytest.raises(ValueError):
        read_invoices((REC / "inventory.tsv").read_bytes())
"""
WT_ITEM = {
    "item": "env/worktree_workspace:read_invoices",
    "kind": "env_function",
    "source_episodes": ["o1"],
    "tests": ["env/worktree_workspace/tests/test_read_invoices.py"],
    "covers": [["o1", 0]],
    "input": "bytes",
}
WT_MAN = {
    "items": [WT_ITEM],
    "support": [
        "env/worktree_workspace/tests/rec/invoices.csv",
        "env/worktree_workspace/tests/rec/inventory.tsv",
    ],
}


@pytest.fixture
def office(tmp_path):
    store, inv, tsv = _blobs(tmp_path)
    acts = [
        _wt("finance/ap/invoices-2026-10.csv", inv),
        _wt("ops/inventory.csv", tsv),
        _wt("ops/missing.csv", "e" * 64),
        _sh("=== 1 failed, 2 passed in 0.31s ===", 1),
        _dl("Inventory: wood 1"),
        Action(0, "venmo", "me", [], {}, {"user_id": "u"}, "ok"),
    ]
    ep = _ep(episode_id="o1", actions=acts)
    mem = Repo.init_bare(tmp_path / "mem.git")
    ev = EvidenceStore(tmp_path / "e.sqlite")
    ev.index_episode(ep, "1" * 40)
    lookup = lambda eid, i: acts[i] if eid == "o1" and 0 <= i < len(acts) else None
    gate = Gate(mem, ev, store, action_lookup=lookup, library_mode=LibraryConfig())
    return mem, ev, gate, ep


def _commit(mem, files):
    with mem.temp_checkout("main") as wt:
        for rel, data in files.items():
            (wt / rel).parent.mkdir(parents=True, exist_ok=True)
            (wt / rel).write_bytes(data if isinstance(data, bytes) else data.encode())
        return mem.commit_all(wt, "pass", {"Pass": "p1"})


WT_FILES = {
    "env/worktree_workspace/__init__.py": READER,
    "env/worktree_workspace/tests/test_read_invoices.py": READER_TEST,
    "env/worktree_workspace/tests/rec/invoices.csv": INVOICES,
    "env/worktree_workspace/tests/rec/inventory.tsv": INVENTORY,
}


# --- a pass end to end: the writer reads recorded blobs, writes fixtures, and the gate admits the reader --

WRITER_LOOKS = (
    "import json\n"
    "from memlab.analysis.recorded import worktree_files, load_blob\n"
    "from memlab.analysis.shapes import shape\n"
    "from memlab.episodes import env_channel\n"
    "ep = json.load(open('/inputs/episodes/o1.json'))\n"
    "files = worktree_files(ep)\n"
    "print('KINDS', sorted({a['kind'] for a in ep['actions']}))\n"
    "print('CH', env_channel('worktree', ep['actions'][0]['channel']))\n"
    "for f in files[:2]:\n"
    "    s = shape(f['path'], load_blob(f['blob_before']))\n"
    "    print('SHAPE', f['index'], s['format'], repr(s['delimiter']))\n"
    "try:\n"
    "    load_blob(files[2]['blob_before'])\n"
    "except KeyError as e:\n"
    "    print('MISSING', 'not in the blob store' in str(e))\n"
    "import pathlib\n"
    "rec = pathlib.Path('/memory/env/worktree_workspace/tests/rec')\n"
    "rec.mkdir(parents=True, exist_ok=True)\n"
    "(rec / 'invoices.csv').write_bytes(load_blob(files[0]['blob_before']))\n"
    "(rec / 'inventory.tsv').write_bytes(load_blob(files[1]['blob_before']))\n"
)
