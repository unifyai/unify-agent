import hashlib
import json

import pytest

from unify.memory_v2 import fixtures as fx
from unify.memory_v2.episodes import Action
from tests.memory_v2.test_episodes import _ep

BLOB = b"month,total\n09,120\n"
SHA = hashlib.sha256(BLOB).hexdigest()


def _blob(sha):
    if sha != SHA:
        raise KeyError(sha)
    return BLOB


EP = _ep(
    episode_id="e1",
    request=["Export the payroll", "obs: done"],
    actions=[
        Action(
            0,
            "acct",
            "me",
            [],
            {},
            {"user_id": "u-1", "name": "Ada"},
            "ok",
            "read",
        ),
        Action(1, "acct", "note", [], {}, "plain text reply", "ok", "read"),
        Action(
            2,
            "worktree:ws",
            "read",
            ["pay.csv"],
            {},
            {"blob_before": SHA},
            "ok",
            "read",
            kind="worktree",
        ),
        Action(3, "acct", "post", [], {"total": 120}, None, "unrecorded", "write"),
    ],
)
LOAD = {"e1": EP}.get
DEST = "memory/office/tests/data/me.json"


def test_ids_and_paths():
    assert fx.module_path("memory.office.payroll:export") == "memory/office/payroll.py"
    assert fx.package_dir("memory.office.payroll:export") == "memory/office"
    assert (
        fx.inputs_file("memory.office.payroll:export")
        == "memory/office/tests/data/payroll.export.inputs.jsonl"
    )


def test_source_bytes_per_kind():
    assert (
        fx.source_bytes(EP, 0, _blob)
        == json.dumps(
            {"user_id": "u-1", "name": "Ada"},
            sort_keys=True,
            ensure_ascii=False,
        ).encode()
    )
    assert fx.source_bytes(EP, "1", _blob) == b"plain text reply"
    assert fx.source_bytes(EP, 2, _blob) == BLOB
    assert fx.source_bytes(EP, "request:1", _blob) == b"obs: done"
    assert json.loads(fx.source_bytes(EP, "args:3", _blob)) == {
        "args": [],
        "kwargs": {"total": 120},
    }
    with pytest.raises(fx.FixtureError, match="recorded no response"):
        fx.source_bytes(EP, 3, _blob)
    with pytest.raises(fx.FixtureError, match="out of range"):
        fx.source_bytes(EP, 9, _blob)
    with pytest.raises(fx.FixtureError, match="not an index"):
        fx.source_bytes(EP, "-1", _blob)
    assert (
        fx.trust_of("args:3") == 2
        and fx.trust_of("0") == 1
        and fx.trust_of("request:1") == 1
    )
    assert fx.sources_of(EP) == [
        "request:0",
        "request:1",
        "0",
        "args:0",
        "1",
        "args:1",
        "2",
        "args:2",
        "args:3",
    ]


def test_make_copy_slice_and_assembled_lines():
    data, entry = fx.make(EP, 2, "memory/office/tests/data/pay.csv", blob=_blob)
    assert (
        data == BLOB
        and entry["kind"] == "copy"
        and entry["sha256"] == SHA
        and entry["slice"] is None
    )
    part, entry2 = fx.make(
        EP,
        2,
        "memory/office/tests/data/head.csv",
        blob=_blob,
        cut=[0, 12],
    )
    assert part == b"month,total\n" and entry2["slice"] == [0, 12]
    rel = fx.inputs_file("memory.acct.users:user_id")
    line, e1 = fx.make(EP, 0, rel, blob=_blob, append=True)
    assert json.loads(line) == {
        "input": {"name": "Ada", "user_id": "u-1"},
        "source": {"episode": "e1", "source": "0", "slice": None},
    }
    line2, e2 = fx.make(EP, 1, rel, blob=_blob, append=True, existing=e1)
    assert json.loads(line2)["input"] == "plain text reply" and len(e2["lines"]) == 2


@pytest.mark.parametrize(
    "dest",
    [
        "memory/office/data/x.json",
        "tests/data/x.json",
        "memory/office/tests/data/_drawn/0/x",
        "memory/office/tests/data/../x",
    ],
)
def test_make_refuses_a_destination_outside_tests_data(dest):
    with pytest.raises(fx.FixtureError, match="not a file name under"):
        fx.make(EP, 0, dest, blob=_blob)


def test_make_refuses_bad_slices_and_appending_to_a_copy():
    with pytest.raises(fx.FixtureError, match="outside the"):
        fx.make(EP, 2, DEST, blob=_blob, cut=[0, 999])
    with pytest.raises(fx.FixtureError, match="start, end"):
        fx.make(EP, 2, DEST, blob=_blob, cut=[3])
    _, copy = fx.make(EP, 0, "memory/office/tests/data/x.jsonl", blob=_blob)
    with pytest.raises(fx.FixtureError, match="cannot be appended"):
        fx.make(
            EP,
            0,
            "memory/office/tests/data/x.jsonl",
            blob=_blob,
            append=True,
            existing=copy,
        )
    with pytest.raises(fx.FixtureError, match="must end in .jsonl"):
        fx.make(EP, 0, DEST, blob=_blob, append=True)


def test_verify_accepts_what_make_wrote():
    data, entry = fx.make(EP, 0, DEST, blob=_blob)
    assert fx.verify(DEST, data, entry, LOAD, _blob) is None
    rel = fx.inputs_file("memory.acct.users:user_id")
    l1, e1 = fx.make(EP, 0, rel, blob=_blob, append=True)
    l2, e2 = fx.make(EP, "request:1", rel, blob=_blob, append=True, existing=e1)
    assert fx.verify(rel, l1 + l2, e2, LOAD, _blob) is None


def test_verify_refuses_a_tampered_copy_and_line():
    data, entry = fx.make(EP, 0, DEST, blob=_blob)
    assert "differs from the recorded bytes" in fx.verify(
        DEST,
        data.replace(b"u-1", b"u-9"),
        entry,
        LOAD,
        _blob,
    )
    assert "did not come through fixture()" in fx.verify(DEST, data, None, LOAD, _blob)
    rel = fx.inputs_file("memory.acct.users:user_id")
    line, e1 = fx.make(EP, 0, rel, blob=_blob, append=True)
    forged = line.replace(b"u-1", b"u-9")
    assert "line 1 differs from the recorded value" in fx.verify(
        rel,
        forged,
        e1,
        LOAD,
        _blob,
    )
    assert "has 2 lines" in fx.verify(rel, line + line, e1, LOAD, _blob)
    assert "cannot be read" in fx.verify(
        DEST,
        data,
        {**entry, "episode": "e404"},
        LOAD,
        _blob,
    )


def test_hash_index_finds_identical_recorded_bytes():
    idx = fx.HashIndex()
    idx.add(EP, _blob)
    assert ("e1", "2") in idx.find(BLOB)
    assert idx.find(b"never recorded") == []
