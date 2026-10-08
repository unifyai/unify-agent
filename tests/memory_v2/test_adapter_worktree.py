import hashlib
import json
import os
import time

import pytest

from unify.memory_v2.blobs import BlobStore
from unify.memory_v2.gitio import Repo
from unify.memory_v2.integration.adapters.worktree import (
    REQUEST_END_CELL,
    WorkTreeRecorder,
    file_shape,
    tree_files,
)
from unify.memory_v2.redact import Redactor

KEY = "sk-or-v1-" + "ab" * 32


def _setup(tmp_path, **kw):
    wt = tmp_path / "work"
    wt.mkdir()
    (wt / "pay.csv").write_text(
        "id\tname\tamount\tdate\n1\tann\t3.5\t2026-01-02\n2\tbob\t4\t2026-01-03\n",
    )
    (wt / "conf.json").write_text(
        json.dumps({"db": {"host": "h", "port": 5432}, "tags": [{"k": "v"}]}),
    )
    (wt / "notes.txt").write_text("one\ntwo\nthree\n")
    (wt / "sub").mkdir()
    (wt / "sub" / "a.txt").write_text("a\n")
    repo = Repo.init_snapshot(tmp_path / "worktree.git", wt)
    rec = WorkTreeRecorder(wt, repo, BlobStore(tmp_path / "blobs"), **kw)
    return wt, repo, rec


def _row(rows, method, path):
    (hit,) = [r for r in rows if r.method == method and r.args == [path]]
    return hit


def test_reads_get_shapes_and_blobs_from_the_before_snapshot(tmp_path):
    wt, repo, rec = _setup(tmp_path)
    rec.begin()
    audit = [
        {"event": "open", "path": str(wt / "pay.csv"), "mode": "r", "cell": 0},
        {"event": "open", "path": "pay.csv", "mode": "rb", "cell": 0},  # deduplicated
        {"event": "open", "path": "conf.json", "mode": "r", "cell": 0},
        {"event": "open", "path": "notes.txt", "mode": "r", "cell": 0},
        {"event": "listdir", "path": str(wt), "cell": 0},
        {
            "event": "open",
            "path": "/etc/hostname",
            "mode": "r",
            "cell": 0,
        },  # another root
    ]
    rows = rec.record_cell(0, audit)
    assert len(rows) == 4
    assert all(
        r.kind == "worktree" and r.channel == "worktree:workspace" and r.kwargs == {}
        for r in rows
    )

    csv_row = _row(rows, "read", "pay.csv")
    assert csv_row.status == "ok" and csv_row.effect == "read"
    shape = csv_row.response["shape"]
    assert shape["delimiter"] == "\t" and shape["header"] is True
    assert shape["columns"] == ["id", "name", "amount", "date"]
    assert shape["types"] == ["int", "str", "float", "date"]
    assert "ann" not in json.dumps(shape)  # no values in a shape
    assert rec.blobs.get(csv_row.response["blob"]) == (wt / "pay.csv").read_bytes()
    assert csv_row.response["size"] == (wt / "pay.csv").stat().st_size

    js = _row(rows, "read", "conf.json").response["shape"]
    assert js["format"] == "json"
    assert js["keys"]["db"] == {
        "type": "object",
        "keys": {"host": "str", "port": "int"},
    }
    assert js["keys"]["tags"] == {
        "type": "array",
        "items": {"type": "object", "keys": {"k": "str"}},
    }

    txt = _row(rows, "read", "notes.txt").response["shape"]
    assert txt == {"format": "text", "encoding": "utf-8", "lines": 3}

    listing = _row(rows, "list", ".").response
    assert listing["entries"] == ["conf.json", "notes.txt", "pay.csv", "sub"]


def test_writes_carry_before_and_after_blobs(tmp_path):
    wt, repo, rec = _setup(tmp_path)
    before = rec.begin()
    old = (wt / "notes.txt").read_bytes()
    (wt / "notes.txt").write_text("changed\n")
    (wt / "out.json").write_text(json.dumps({"total": 7.5}))
    rec.record_cell(
        0,
        [
            {"event": "open", "path": "notes.txt", "mode": "w", "cell": 0},
            {"event": "open", "path": "out.json", "mode": "x", "cell": 0},
        ],
    )
    # a later read of a file written in this request sees the in-request state
    (read,) = rec.record_cell(
        1,
        [{"event": "open", "path": "out.json", "mode": "r", "cell": 1}],
    )
    assert read.response["shape"]["keys"] == {"total": "float"}
    os.remove(wt / "sub" / "a.txt")  # e.g. a subprocess: not in the audit
    rows = rec.finish()
    assert rec.after and rec.after != before

    notes = _row(rows, "write", "notes.txt")
    assert notes.cell == 0 and notes.effect == "write" and notes.status == "ok"
    assert rec.blobs.get(notes.response["blob_before"]) == old
    assert rec.blobs.get(notes.response["blob_after"]) == b"changed\n"

    out = _row(rows, "write", "out.json")
    assert out.response["blob_before"] is None and out.response["blob_after"]
    assert out.response["shape"]["format"] == "json"

    gone = _row(rows, "write", "sub/a.txt")
    assert gone.cell == REQUEST_END_CELL and gone.response["deleted"] is True
    assert gone.response["blob_after"] is None and gone.response["blob_before"]

    assert repo.changed_paths(before, rec.after) == [
        "notes.txt",
        "out.json",
        "sub/a.txt",
    ]
    assert rec.repo.blame_lines(rec.after, "notes.txt") == [rec.after]


def test_size_cap_keeps_the_shape_only(tmp_path):
    wt, repo, rec = _setup(tmp_path, blob_cap=1024)
    (wt / "big.csv").write_text("a,b\n" + "1,2\n" * 1000)
    rec.begin()
    (row,) = rec.record_cell(
        0,
        [{"event": "open", "path": "big.csv", "mode": "r", "cell": 0}],
    )
    assert row.response["blob"] is None and row.status == "ok"
    assert row.response["size"] == (wt / "big.csv").stat().st_size
    assert row.response["shape"]["columns"] == ["a", "b"]
    assert row.response["shape"]["types"] == ["int", "int"]


def test_snapshot_skips_large_files_symlinks_and_nested_git(tmp_path):
    wt, repo, rec = _setup(tmp_path, snapshot_cap=4096)
    (wt / "huge.bin").write_bytes(b"\0" * 8192)
    (wt / "link").symlink_to(wt / "notes.txt")
    (wt / "nested" / ".git").mkdir(parents=True)
    (wt / "nested" / ".git" / "config").write_text("x")
    (wt / ".gitattributes").write_text("* filter=x\n")
    sha = rec.begin()
    files = tree_files(repo, sha)
    assert "huge.bin" not in files and "link" not in files
    assert not any(".git/" in p for p in files)
    assert ".gitattributes" in files and "pay.csv" in files
    assert any("huge.bin" in s for s in rec.skipped)
    # no .git ever appears inside the work tree
    assert not (wt / ".git").exists()
    # a read of the skipped big file still gets a shape from disk, with no blob
    (row,) = rec.record_cell(
        0,
        [{"event": "open", "path": "huge.bin", "mode": "rb", "cell": 0}],
    )
    assert row.response["blob"] is None and row.response["shape"]["format"] == "binary"
    assert row.response["source"] == "disk"


def test_channel_is_kind_qualified():
    from unify.memory_v2.episodes import env_channel
    from unify.memory_v2.integration.adapters.worktree import CHANNEL

    assert CHANNEL == "worktree:workspace"
    assert env_channel("worktree", CHANNEL) == "worktree_workspace"


def test_paths_hidden_from_cells_are_never_snapshotted_read_or_listed(tmp_path):
    hide = {"secret.env", "vault"}

    def hidden(rel):
        if rel == "boom":
            raise RuntimeError("predicate failed")  # unknown counts as hidden
        return rel.split("/")[0] in hide

    wt, repo, rec = _setup(tmp_path, hidden=hidden)
    (wt / "secret.env").write_text("TOKEN=HIDDEN-VALUE-1\n")
    (wt / "vault").mkdir()
    (wt / "vault" / "k.txt").write_text("HIDDEN-VALUE-2\n")
    (wt / "boom").write_text("HIDDEN-VALUE-3\n")
    sha = rec.begin()
    files = tree_files(repo, sha)
    assert "secret.env" not in files and "boom" not in files
    assert not any(p.startswith("vault") for p in files)
    assert rec.skip_counts["worktree_before: hidden from cells"] == 3
    rows = rec.record_cell(
        0,
        [
            {"event": "open", "path": str(wt / "secret.env"), "mode": "r"},
            {"event": "open", "path": str(wt / "vault" / "k.txt"), "mode": "r"},
            {"event": "os.listdir", "path": str(wt / "vault")},
            {"event": "open", "path": str(wt / "boom"), "mode": "w"},
            {"event": "open", "path": str(wt / "pay.csv"), "mode": "r"},
        ],
    )
    assert [(r.method, r.args) for r in rows] == [("read", ["pay.csv"])]
    (wt / "secret.env").write_text("TOKEN=HIDDEN-VALUE-4\n")
    assert rec.finish() == []
    assert rec.diff() == ""
    for needle in (b"HIDDEN-VALUE-1", b"HIDDEN-VALUE-2", b"HIDDEN-VALUE-4"):
        assert not _blob_store_holds(tmp_path, needle)
        assert not _snapshot_holds(repo, rec.after, needle)


def test_snapshot_dir_inside_work_tree_is_refused(tmp_path):
    wt = tmp_path / "w"
    wt.mkdir()
    repo = Repo.init_snapshot(wt / "inner.git", wt)
    with pytest.raises(ValueError):
        WorkTreeRecorder(wt, repo, BlobStore(tmp_path / "b"))


def test_blob_content_is_redacted(tmp_path):
    wt, repo, rec = _setup(
        tmp_path,
        redactor=Redactor({"API_TOKEN": "hunter2-secret-value"}),
    )
    (wt / "env.txt").write_text(f"key={KEY}\ntoken=hunter2-secret-value\n")
    rec.begin()
    (row,) = rec.record_cell(
        0,
        [{"event": "open", "path": "env.txt", "mode": "r", "cell": 0}],
    )
    stored = rec.blobs.get(row.response["blob"]).decode()
    assert KEY not in stored and "hunter2-secret-value" not in stored
    assert "<redacted:key-shaped>" in stored and "<secret:API_TOKEN>" in stored
    (wt / "env.txt").write_text(f"again {KEY}\n")
    rec.record_cell(1, [{"event": "open", "path": "env.txt", "mode": "a", "cell": 1}])
    (w,) = rec.finish()
    for key in ("blob_before", "blob_after"):
        assert KEY not in rec.blobs.get(w.response[key]).decode()


def test_file_shape_is_deterministic_and_value_free():
    data = b"x,y\n1,2026-01-01\n2,2026-02-01\n"
    assert file_shape("t.csv", data) == file_shape("t.csv", data)
    assert file_shape("t.csv", data)["types"] == ["int", "date"]
    assert file_shape("a.txt", "caf\xe9\n".encode("latin-1"))["encoding"] == "latin-1"
    assert file_shape("x.yaml", b"a:\n  b: 1\n")["keys"] == {
        "a": {"type": "object", "keys": {"b": "int"}},
    }
    assert file_shape("x.bin", b"\0\1\2") == {"format": "binary"}
    deep = json.dumps({"a": {"b": {"c": {"d": 1}}}}).encode()
    assert file_shape("d.json", deep)["keys"]["a"]["keys"]["b"]["keys"]["c"] == {
        "type": "object",
    }


def test_gitattributes_filters_never_run(tmp_path):
    wt, repo, rec = _setup(tmp_path)
    marker = tmp_path / "filter-ran"
    repo.run("config", "filter.x.clean", f"touch {marker}; cat")
    (wt / ".gitattributes").write_text("* filter=x\n")
    sha = rec.begin()
    assert not marker.exists()
    assert tree_files(repo, sha)["pay.csv"]


# -- regressions for the review probes (C1, I1-I8, minor items) ----------------------------------


def _host_secret(tmp_path):
    host = tmp_path / "host"
    (host / "gcloud").mkdir(parents=True)
    (host / "gcloud" / "x.json").write_text(
        json.dumps({"private" + "_key": "HOSTSECRET-123"}),
    )
    (host / "id_rsa").write_text("HOSTSECRET-RSA")
    return host


def _blob_store_holds(tmp_path, needle: bytes) -> bool:
    root = tmp_path / "blobs"
    return any(needle in p.read_bytes() for p in root.rglob("*") if p.is_file())


def _snapshot_holds(repo, sha, needle: bytes) -> bool:
    from unify.memory_v2.integration.adapters.worktree import _git_raw

    return any(
        needle in _git_raw(repo, ["cat-file", "blob", oid])
        for oid in tree_files(repo, sha).values()
    )


@pytest.mark.parametrize("when", ["during", "at_begin"])
def test_symlinked_directory_is_never_followed(tmp_path, when):
    wt, repo, rec = _setup(tmp_path)
    host = _host_secret(tmp_path)
    if when == "at_begin":
        (wt / "link").symlink_to(host)
    rec.begin()
    if when == "during":
        (wt / "link").symlink_to(host)
    rows = rec.record_cell(
        0,
        [
            {"event": "open", "path": "link/gcloud/x.json", "mode": "r", "cell": 0},
            {"event": "listdir", "path": "link", "cell": 0},
            {"event": "open", "path": "link", "mode": "r", "cell": 0},
        ],
    )
    assert [r.status for r in rows] == ["unrecorded"] * 3
    assert all(r.response.get("refused") for r in rows)
    assert _row(rows, "list", "link").response["entries"] is None
    assert not _blob_store_holds(tmp_path, b"HOSTSECRET")
    rec.finish()
    assert not _snapshot_holds(repo, rec.after, b"HOSTSECRET")
    assert sum(rec.skip_counts.values()) >= 3


def test_swap_in_the_check_read_gap_is_refused(tmp_path, monkeypatch):
    from unify.memory_v2.integration.adapters import worktree as mod

    wt, repo, rec = _setup(tmp_path, snapshot_cap=4096)
    host = _host_secret(tmp_path)
    (wt / "grow.txt").write_text("small\n")

    def gap(stage, rel):
        if stage == "listed" and rel == "notes.txt":
            os.remove(wt / "notes.txt")
            (wt / "notes.txt").symlink_to(host / "id_rsa")
        if stage == "listed" and rel == "pay.csv":
            os.remove(wt / "pay.csv")
            os.mkfifo(wt / "pay.csv")
        if stage == "checked" and rel == "grow.txt":
            with open(wt / "grow.txt", "ab") as fh:
                fh.write(b"x" * 100_000)

    monkeypatch.setattr(mod, "_gap", gap)
    sha = rec.begin()
    files = tree_files(repo, sha)
    assert (
        "notes.txt" not in files and "pay.csv" not in files and "grow.txt" not in files
    )
    assert not _snapshot_holds(repo, sha, b"HOSTSECRET")
    assert any("notes.txt" in n for n in rec.skipped)
    # the same seam on the read path: a symlink swapped in after the name was taken
    (wt / "late.txt").write_text("fine\n")

    def gap_read(stage, rel):
        if stage == "listed" and rel == "late.txt":
            os.remove(wt / "late.txt")
            (wt / "late.txt").symlink_to(host / "id_rsa")

    monkeypatch.setattr(mod, "_gap", gap_read)
    (row,) = rec.record_cell(
        0,
        [{"event": "open", "path": "late.txt", "mode": "r", "cell": 0}],
    )
    assert row.status == "unrecorded" and not _blob_store_holds(tmp_path, b"HOSTSECRET")


def test_fifo_is_refused_without_blocking(tmp_path):
    wt, repo, rec = _setup(tmp_path)
    os.mkfifo(wt / "pipe")
    sha = rec.begin()  # would hang if the FIFO were opened for a blocking read
    assert "pipe" not in tree_files(repo, sha)
    (row,) = rec.record_cell(
        0,
        [{"event": "open", "path": "pipe", "mode": "r", "cell": 0}],
    )
    assert row.status == "unrecorded" and "regular" in row.response["refused"]


def test_non_utf8_and_cr_names_are_refused_and_counted(tmp_path):
    wt, repo, rec = _setup(tmp_path)
    os.close(
        os.open(
            os.path.join(os.fsencode(wt), b"bad\xff.txt"),
            os.O_CREAT | os.O_WRONLY,
        ),
    )
    (wt / "foo").write_text("PLAIN\n")
    (wt / "foo\r").write_text("CR\n")
    (wt / ".GIT").mkdir()
    (wt / ".GIT" / "x").write_text("x")
    sha = rec.begin()
    files = tree_files(repo, sha)
    assert "foo\r" not in files and ".GIT/x" not in files
    assert not any("bad" in p for p in files)
    from unify.memory_v2.integration.adapters.worktree import _git_raw

    assert _git_raw(repo, ["cat-file", "blob", files["foo"]]) == b"PLAIN\n"
    counts = rec.skip_counts
    assert counts["worktree_before: non-UTF-8 name"] == 1
    assert counts["worktree_before: control character in name"] == 1
    assert counts["worktree_before: .git component"] == 1
    rows = rec.record_cell(
        0,
        [
            {"event": "open", "path": "foo\r", "mode": "r", "cell": 0},
            {
                "event": "open",
                "path": os.fsdecode(b"bad\xff.txt"),
                "mode": "r",
                "cell": 0,
            },
            {"event": "open", "path": ".GIT/x", "mode": "r", "cell": 0},
            {"event": "listdir", "path": ".", "cell": 0},
        ],
    )
    assert [r.method for r in rows] == ["list"]
    json.dumps(rows[0].response)  # listing names are always serialisable
    assert ".GIT" not in rows[0].response["entries"]
    assert (
        rec.finish() == []
    )  # nothing changed; the refused names never break the snapshot


def test_hostile_shapes_never_abort_recording(tmp_path):
    wt, repo, rec = _setup(tmp_path)
    (wt / "deep.json").write_text("[" * 200_000)
    (wt / "deep.jsonl").write_text("[" * 200_000)
    (wt / "deep.txt").write_text("[" * 200_000)
    (wt / "wide.csv").write_text("a,b\n1," + "x" * 200_000 + "\n")
    rec.begin()
    rows = rec.record_cell(
        0,
        [
            {"event": "open", "path": p, "mode": "r", "cell": 0}
            for p in ("deep.json", "deep.jsonl", "deep.txt", "wide.csv")
        ],
    )
    assert all(r.status == "ok" for r in rows)
    assert _row(rows, "read", "deep.json").response["shape"] == {
        "format": "json",
        "error": "unparsed",
    }
    assert _row(rows, "read", "deep.jsonl").response["shape"]["error"] == "unparsed"
    assert _row(rows, "read", "deep.txt").response["shape"]["format"] == "text"
    assert _row(rows, "read", "wide.csv").response["shape"]["format"] == "csv"
    assert file_shape("x.csv", b'"' + b"x" * 3_000_000)["format"] in ("csv", "tsv")


def test_csv_without_header_is_positional_and_bounded():
    shape = file_shape(
        "p.csv",
        b"Ann Smith,ann.smith@example.com,London\nBob Jones,bob@example.com,Paris\n",
    )
    assert shape["header"] is False and shape["columns"] == ["c0", "c1", "c2"]
    assert "Ann" not in json.dumps(shape) and "example.com" not in json.dumps(shape)
    numeric = file_shape("n.csv", b"1,2.5,2026-01-01\n2,3.5,2026-01-02\n")
    assert numeric["header"] is False and numeric["types"] == ["int", "float", "date"]
    wide_header = ",".join(f"h{i}" + "y" * 200 for i in range(1000)).encode()
    wide = file_shape("w.csv", wide_header + b"\n" + b",".join([b"1"] * 1000) + b"\n")
    assert len(wide["columns"]) == 256 and wide["more_columns"] == 744
    assert (
        all(len(c) <= 128 for c in wide["columns"]) and wide["names_truncated"] is True
    )
    deep = json.dumps({"k" * 1000: 1}).encode()
    assert all(len(k) <= 128 for k in file_shape("k.json", deep)["keys"])


def test_default_redactor_knows_environment_secrets(tmp_path, monkeypatch):
    monkeypatch.setenv("MEMV2_TEST_API_TOKEN", "envsecret-value-42")
    wt, repo, rec = _setup(tmp_path)
    (wt / "leak.txt").write_text("t=envsecret-value-42\n")
    rec.begin()
    (row,) = rec.record_cell(
        0,
        [{"event": "open", "path": "leak.txt", "mode": "r", "cell": 0}],
    )
    assert b"envsecret-value-42" not in rec.blobs.get(row.response["blob"])


def test_git_runs_with_a_minimal_isolated_environment(tmp_path, monkeypatch):
    from unify.memory_v2.integration.adapters import worktree as mod

    monkeypatch.setenv("MEMV2_UNRELATED_SECRET", "zzz")
    wt, repo, rec = _setup(
        tmp_path,
    )  # repo creation goes through gitio, not this adapter
    seen = []
    real = mod.subprocess.run

    def spy(argv, **kw):
        seen.append((argv, kw["env"], kw["cwd"]))
        return real(argv, **kw)

    monkeypatch.setattr(mod.subprocess, "run", spy)
    rec.begin()
    assert seen
    for argv, env, cwd in seen:
        assert (
            env["GIT_CONFIG_GLOBAL"] == os.devnull and env["GIT_CONFIG_NOSYSTEM"] == "1"
        )
        assert "MEMV2_UNRELATED_SECRET" not in env and env["HOME"] != os.environ.get(
            "HOME",
        )
        assert "core.hooksPath=/dev/null" in argv and "--work-tree" in argv
        assert str(wt) not in argv and not any(a.startswith(f"{wt}/") for a in argv)
        assert not (cwd + "/").startswith(f"{wt}/")


def test_snapshot_budget_is_counted(tmp_path):
    wt, repo, rec = _setup(tmp_path, budget_files=2)
    sha = rec.begin()
    assert len(tree_files(repo, sha)) == 2
    assert rec.skip_counts["worktree_before: over the snapshot budget"] == 2


def test_writes_are_attributed_per_cell_and_mkdir_has_its_own_row(tmp_path):
    wt, repo, rec = _setup(tmp_path)
    rec.begin()
    (wt / "notes.txt").write_text("v1\n")
    rec.record_cell(0, [{"event": "open", "path": "notes.txt", "mode": "w", "cell": 0}])
    (wt / "newdir").mkdir()
    (mk,) = rec.record_cell(1, [{"event": "mkdir", "path": "newdir", "cell": 1}])
    assert (
        mk.method == "mkdir"
        and mk.status == "ok"
        and mk.response == {"exists_after_cell": True}
    )
    (wt / "notes.txt").write_text("v2\n")
    rec.record_cell(2, [{"event": "open", "path": "notes.txt", "mode": "a", "cell": 2}])
    # a second call for the same cell does not duplicate reads
    first = rec.record_cell(
        3,
        [{"event": "open", "path": "pay.csv", "mode": "r", "cell": 3}],
    )
    again = rec.record_cell(
        3,
        [{"event": "open", "path": "pay.csv", "mode": "r", "cell": 3}],
    )
    assert len(first) == 1 and again == []
    assert first[0].response["oid"] == tree_files(repo, rec.before)["pay.csv"]
    assert first[0].response["source"] == "before"
    rows = [r for r in rec.finish() if r.args == ["notes.txt"]]
    assert [r.cell for r in rows] == [0, 2]
    assert rows[0].response == rows[1].response
    assert rec.blobs.get(rows[0].response["blob_after"]) == b"v2\n"


# -- regressions for the re-review (N1-N5) --------------------------------------------------------


def test_root_and_nul_audit_paths_never_abort(tmp_path):
    wt, repo, rec = _setup(tmp_path)
    rec.begin()
    rows = rec.record_cell(
        0,
        [
            {"event": "open", "path": str(wt), "mode": "r"},  # open(<work tree>)
            {"event": "open", "path": ".", "mode": "r"},  # os.open(".") with cwd there
            {"event": "open", "path": "a\x00b", "mode": "r"},  # a forged sys.audit
            {"event": "os.listdir", "path": "a\x00b"},
            {"event": "os.mkdir", "path": "a\x00b"},
            {"event": "open", "path": "pay.csv", "mode": "r"},
        ],
    )
    (root,) = [r for r in rows if r.args == ["."]]
    assert root.method == "read" and root.status == "unrecorded"
    assert root.response["refused"] == "not a regular file"
    assert _row(rows, "read", "pay.csv").status == "ok"  # later events still recorded
    assert not any("\x00" in r.args[0] for r in rows)
    assert rec.skip_counts["control character in name"] == 3


def test_csv_shaping_is_linear_on_hostile_input():
    # the parser itself, called in the test process: its own cost, without the child's start-up
    from unify.memory_v2.integration.adapters.worktree_shapes import _csv_shape

    start = time.monotonic()
    _csv_shape(b',"a' * 22_000, None, False)  # csv.Sniffer took 15 s on 64 KiB of this
    file_shape("q.csv", b',"a' * 300_000)
    file_shape("q.tsv", b'\t"a' * 300_000)
    # one very wide row and many short ones: columns past the cap are counted, not typed
    wide = file_shape("w.csv", b"," * 500_000 + b"\n" + b"1\n" * 200)
    assert time.monotonic() - start < 2
    assert len(wide["columns"]) == 256 and wide["more_columns"] == 500_001 - 256
    assert file_shape("t.csv", b"a;b\n1;x\n2;y\n")["delimiter"] == ";"
    assert file_shape("t.csv", b"a|b|c\n1|2|3\n")["columns"] == ["a", "b", "c"]
    assert file_shape("one.csv", b"name\n2\n3\n")["delimiter"] == ","  # one column


def test_shaping_budget_is_spent_per_request(tmp_path):
    wt, repo, rec = _setup(tmp_path, shape_budget=200)  # bytes parsed per request
    (wt / "big.json").write_text(json.dumps({"k": "v" * 200}))
    rec.begin()
    rows = rec.record_cell(
        0,
        [
            {"event": "open", "path": p, "mode": "r"}
            for p in ("pay.csv", "big.json", "notes.txt")
        ],
    )
    assert _row(rows, "read", "pay.csv").response["shape"]["format"] == "tsv"
    assert _row(rows, "read", "big.json").response["shape"] == {
        "format": "json",
        "error": "budget",
    }
    assert _row(rows, "read", "notes.txt").response["shape"]["format"] == "text"
    assert rec.skip_counts["shape budget spent"] == 1


def test_yaml_alias_expansion_is_bounded_in_the_shape():
    pytest.importorskip("yaml")
    from unify.memory_v2.integration.adapters.worktree import SHAPE_BYTES

    def keys(prefix, value):
        return ", ".join(f"{prefix * 120}{i:04d}: {value}" for i in range(64))

    doc = (
        f"c: &c {{{keys('c', 1)}}}\n"
        f"b: &b {{{keys('b', '*c')}}}\n"
        f"top: {{{keys('t', '*b')}}}\n"
    ).encode()
    assert len(doc) < 30_000
    shape = file_shape("bomb.yaml", doc)
    assert len(json.dumps(shape)) <= SHAPE_BYTES
    assert shape["format"] == "yaml" and shape["shape_truncated"] is True
    assert shape == file_shape("bomb.yaml", doc)  # deterministic


def test_yaml_merge_keys_never_expand():
    pytest.importorskip("yaml")
    # each level merges the previous one nine times: safe_load copies 9**9 key pairs from 600 bytes
    lines = ["l0: &l0 {a: 1, b: 2}"]
    for i in range(1, 10):
        lines.append(f"l{i}: &l{i}\n  <<: [{', '.join([f'*l{i - 1}'] * 9)}]\n  k{i}: 1")
    doc = ("\n".join(lines) + "\n").encode()
    assert len(doc) < 1000
    start = time.monotonic()
    shape = file_shape("merge.yaml", doc)
    assert time.monotonic() - start < 2
    assert shape["keys"]["l1"]["keys"]["<<"]["type"] == "array"  # kept as a plain key
    explicit = file_shape("tag.yaml", b"a: &a {x: 1}\nb:\n  !!merge <<: *a\n  y: 2\n")
    assert set(explicit["keys"]["b"]["keys"]) == {"<<", "y"}


def test_shape_size_is_bounded_for_every_format(monkeypatch):
    from unify.memory_v2.integration.adapters import worktree as mod

    nested = {
        f"k{i:05d}": {
            f"m{j:03d}": {f"n{m:02d}": 1 for m in range(16)} for j in range(64)
        }
        for i in range(64)
    }
    data = json.dumps(nested).encode()
    for name, payload in (
        ("big.json", data),
        ("big.jsonl", data + b"\n" + data + b"\n"),
        ("big.txt", data),
    ):
        shape = file_shape(name, payload)
        assert len(json.dumps(shape)) <= mod.SHAPE_BYTES, name
        assert shape["shape_truncated"] is True, name
    small = file_shape("conf.json", json.dumps({"a": {"b": 1}}).encode())
    assert "shape_truncated" not in small
    # the final cap applies to every format: anything over it is refused with a marker
    monkeypatch.setattr(mod, "SHAPE_BYTES", 40)
    wide = file_shape("w.csv", b"aaaa,bbbb,cccc,dddd\n1,2,3,4\n")
    assert wide["format"] == "csv" and wide["error"] == "shape_too_large"
    assert wide["bytes"] > 40 and set(wide) == {"format", "error", "bytes"}


def _zip(entries):
    import io
    import zipfile

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for name, chunks in entries:
            with z.open(name, "w", force_zip64=True) as fh:
                for chunk in chunks:
                    fh.write(chunk)
    return buf.getvalue()


def test_xlsx_inflation_is_refused_before_loading():
    start = time.monotonic()
    bomb = _zip(
        [
            ("[Content_Types].xml", [b"<Types/>"]),
            ("xl/sharedStrings.xml", (b"<si><t>a</t></si>" * 65536 for _ in range(50))),
        ],
    )
    assert len(bomb) < 1024 * 1024  # small on disk, 50 MiB inflated
    assert file_shape("bomb.xlsx", bomb) == {
        "format": "xlsx",
        "error": "inflated_too_large",
    }
    many = _zip([(f"x/{i}.xml", [b""]) for i in range(5000)])
    assert file_shape("many.xlsx", many) == {
        "format": "xlsx",
        "error": "too_many_entries",
    }
    assert file_shape("not.xlsx", b"PK not a zip")["error"] == "unparsed"
    assert time.monotonic() - start < 5


def test_small_xlsx_gives_sheets_and_headers():
    openpyxl = pytest.importorskip("openpyxl")
    import io

    wb = openpyxl.Workbook()
    wb.active.title = "pay"
    wb.active.append(["id", "amount"])
    wb.active.append([1, 3.5])
    buf = io.BytesIO()
    wb.save(buf)
    assert file_shape("p.xlsx", buf.getvalue()) == {
        "format": "xlsx",
        "sheets": [{"name": "pay", "headers": ["id", "amount"]}],
    }


def test_audit_adapter_event_names_are_accepted(tmp_path):
    # the raw records of the audit adapter (CellAudit.drain()["records"]): sys.audit event names,
    # absolute paths, "dst" for a rename, tid / cell_thread on every record
    wt, repo, rec = _setup(tmp_path)
    (wt / "old.txt").write_text("o\n")
    rec.begin()
    os.remove(wt / "notes.txt")
    os.rename(wt / "old.txt", wt / "new.txt")
    (wt / "made").mkdir()
    import shutil

    shutil.rmtree(wt / "sub")

    def r(event, path, **kw):
        return {"event": event, "path": path, "tid": 7, "cell_thread": True, **kw}

    rows = rec.record_cell(
        4,
        [
            r("os.listdir", str(wt)),
            r("os.scandir", str(wt / "made")),
            r("os.remove", str(wt / "notes.txt")),
            r("os.rename", str(wt / "old.txt"), dst=str(wt / "new.txt")),
            r("os.rename", None, dst=str(wt / "pay.csv")),  # source outside every root
            r("os.mkdir", str(wt / "made")),
            r("shutil.rmtree", str(wt / "sub")),
            r("os.remove", str(wt / "sub" / "a.txt")),
            r("os.rmdir", str(wt / "sub")),
            r("open", str(wt / "conf.json") + "<clipped>", mode="r", clipped=True),
        ],
    )
    assert sorted((x.method, x.args[0]) for x in rows) == [
        ("list", "."),
        ("list", "made"),
        ("mkdir", "made"),
        ("rmdir", "sub"),
        ("rmtree", "sub"),
    ]
    assert rec.skip_counts["clipped path"] == 1
    # the note says what was clipped
    assert any("conf.json<clipped>" in n for n in rec.skipped)
    writes = {(x.args[0], x.cell) for x in rec.finish()}
    assert writes == {
        ("notes.txt", 4),
        ("old.txt", 4),
        ("new.txt", 4),
        ("pay.csv", 4),
        ("sub/a.txt", 4),
    }


def test_a_failed_record_is_not_deduplicated_away(tmp_path, monkeypatch):
    wt, repo, rec = _setup(tmp_path)
    rec.begin()
    real, calls = rec._content, []

    def flaky(rel, oid):
        calls.append(rel)
        if len(calls) == 1:
            raise OSError("transient")
        return real(rel, oid)

    monkeypatch.setattr(rec, "_content", flaky)
    rows = rec.record_cell(
        0,
        [{"event": "open", "path": "pay.csv", "mode": "r"}] * 2,
    )
    assert [r.status for r in rows] == ["ok"]  # the retry is recorded, not deduplicated
    assert rec.skip_counts["audit record failed"] == 1


# -- regressions for re-review 2 (I-A, I-B): every format is parsed in a bounded child ----------

# YAML 1.1 base-60 int: PyYAML evaluates it with quadratic big-int arithmetic (480 KB: about 7 s)
BASE60_BOMB = b"k: 1" + b":59" * 160_000


def test_normal_shapes_are_unchanged_by_the_child():
    # the shapes the in-process parsers gave at 96a57da00
    cases = {
        "pay.csv": (
            b"id,name,amount,date\n1,ann,3.5,2026-01-02\n2,bob,4,2026-01-03\n",
            {
                "format": "csv",
                "delimiter": ",",
                "header": True,
                "columns": ["id", "name", "amount", "date"],
                "types": ["int", "str", "float", "date"],
                "encoding": "utf-8",
            },
        ),
        "pay.tsv": (
            b"id\tname\n1\tann\n",
            {
                "format": "tsv",
                "delimiter": "\t",
                "header": True,
                "columns": ["id", "name"],
                "types": ["int", "str"],
                "encoding": "utf-8",
            },
        ),
        "conf.json": (
            json.dumps(
                {"db": {"host": "h", "port": 5432}, "tags": [{"k": "v"}], "n": None},
            ).encode(),
            {
                "format": "json",
                "type": "object",
                "keys": {
                    "db": {"type": "object", "keys": {"host": "str", "port": "int"}},
                    "tags": {
                        "type": "array",
                        "items": {"type": "object", "keys": {"k": "str"}},
                    },
                    "n": "null",
                },
            },
        ),
        "rows.jsonl": (
            b'{"a": 1, "b": [1]}\n{"a": 2}\n',
            {
                "format": "jsonl",
                "records": 2,
                "record": {
                    "type": "object",
                    "keys": {"a": "int", "b": {"type": "array", "items": "int"}},
                },
            },
        ),
        "notes.txt": (
            b"one\ntwo\n",
            {"format": "text", "encoding": "utf-8", "lines": 2},
        ),
        "arr.txt": (
            b'[{"x": 1}]',
            {
                "format": "json",
                "type": "array",
                "items": {"type": "object", "keys": {"x": "int"}},
            },
        ),
    }
    for name, (data, want) in cases.items():
        assert file_shape(name, data) == want, name
    pytest.importorskip("yaml")
    assert file_shape(
        "c.yaml",
        b"a:\n  b: 1\n  c: [x, 2.5]\nd: 2026-01-02\ne: null\n",
    ) == {
        "format": "yaml",
        "type": "object",
        "keys": {
            "a": {
                "type": "object",
                "keys": {"b": "int", "c": {"type": "array", "items": "str"}},
            },
            "d": "date",
            "e": "null",
        },
    }
    openpyxl = pytest.importorskip("openpyxl")
    import io

    wb = openpyxl.Workbook()
    wb.active.title = "pay"
    wb.active.append(["id", "amount"])
    wb.active.append([1, 3.5])
    wb.create_sheet("notes").append(["when", None, "who"])
    buf = io.BytesIO()
    wb.save(buf)
    assert file_shape("w.xlsx", buf.getvalue()) == {
        "format": "xlsx",
        "sheets": [
            {"name": "pay", "headers": ["id", "amount"]},
            {"name": "notes", "headers": ["when", None, "who"]},
        ],
    }


def test_yaml_base60_int_bomb_gets_the_timeout_marker():
    pytest.importorskip("yaml")
    from unify.memory_v2.integration.adapters import worktree as mod

    start = time.monotonic()
    shape = file_shape("bomb.yaml", BASE60_BOMB)
    elapsed = time.monotonic() - start
    assert shape == {"format": "yaml", "unparsed": "timeout"}
    assert elapsed < mod.PARSE_TIMEOUT + 1.5


def _lying_zip(name: str, real_mib: int, declared: int) -> bytes:
    """A zip whose central directory declares *declared* bytes (and their CRC) for an entry that
    inflates to *real_mib* MiB of zeros."""
    import io
    import struct
    import zipfile
    import zlib

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        with z.open(name, "w") as fh:
            for _ in range(real_mib):
                fh.write(bytes(1 << 20))
    data = bytearray(buf.getvalue())
    cd = data.rindex(b"PK\x01\x02")
    struct.pack_into("<I", data, cd + 16, zlib.crc32(bytes(declared)))
    struct.pack_into("<I", data, cd + 24, declared)
    return bytes(data)


def test_xlsx_lying_sizes_are_refused_without_inflating():
    import tracemalloc

    from unify.memory_v2.integration.adapters import worktree_shapes as shapes

    lie = _lying_zip("[Content_Types].xml", 64, 10)
    assert len(lie) < 256 * 1024  # 64 MiB inflated, 10 bytes declared
    start = time.monotonic()
    assert file_shape("lie.xlsx", lie) == {"format": "xlsx", "error": "size_mismatch"}
    assert time.monotonic() - start < 3
    # the defence itself, not the child's limits: inflation stops one chunk past the declared size
    tracemalloc.start()
    try:
        assert shapes._xlsx_shape(lie, shapes._Budget())["error"] == "size_mismatch"
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert peak < 4 * 1024 * 1024


def test_xlsx_lzma_and_bzip2_entries_are_refused():
    import io
    import zipfile

    for method in (zipfile.ZIP_LZMA, zipfile.ZIP_BZIP2):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr("[Content_Types].xml", b"<Types/>")
            z.writestr("xl/sharedStrings.xml", b"<sst/>", compress_type=method)
        assert file_shape("m.xlsx", buf.getvalue()) == {
            "format": "xlsx",
            "error": "unsafe_compression",
        }, method


def test_parse_wall_budget_stops_spawning(tmp_path, monkeypatch):
    pytest.importorskip("yaml")
    from unify.memory_v2.integration.adapters import worktree as mod

    wt, repo, rec = _setup(tmp_path, parse_seconds=1.0)
    (wt / "bomb.yaml").write_bytes(BASE60_BOMB)
    rec.begin()
    spawned = []
    real = mod._run_child

    def counting(*args, **kwargs):
        spawned.append(args)
        return real(*args, **kwargs)

    monkeypatch.setattr(mod, "_run_child", counting)
    start = time.monotonic()
    rows = rec.record_cell(
        0,
        [
            {"event": "open", "path": p, "mode": "r"}
            for p in ("bomb.yaml", "pay.csv", "conf.json", "notes.txt")
        ],
    )
    assert time.monotonic() - start < 1.0 + 1.5
    assert _row(rows, "read", "bomb.yaml").response["shape"] == {
        "format": "yaml",
        "unparsed": "timeout",
    }
    for path, fmt in (("pay.csv", "csv"), ("conf.json", "json"), ("notes.txt", "text")):
        assert _row(rows, "read", path).response["shape"] == {
            "format": fmt,
            "error": "budget",
        }
    assert len(spawned) == 1  # nothing is spawned once the time is spent
    assert rec.skip_counts["shape time budget spent"] == 3


def _probe(tmp_path, body: str):
    probe = tmp_path / "probe.py"
    probe.write_text("import json, os, resource, sys, time\n" + body)
    return probe


def test_shape_child_gets_no_environment_no_path_and_its_limits(tmp_path, monkeypatch):
    from unify.memory_v2.integration.adapters import worktree as mod

    probe = _probe(
        tmp_path,
        "data = sys.stdin.buffer.read()\n"
        "names = ('RLIMIT_AS', 'RLIMIT_CPU', 'RLIMIT_FSIZE', 'RLIMIT_NOFILE', 'RLIMIT_NPROC',\n"
        "    'RLIMIT_CORE')\n"
        "lim = {n: resource.getrlimit(getattr(resource, n))[0] for n in names}\n"
        "sys.stdout.write(json.dumps({'format': 'json', 'env': sorted(os.environ),\n"
        "    'limits': lim, 'isolated': sys.flags.isolated, 'no_site': sys.flags.no_site,\n"
        "    'argv': sys.argv[1:], 'cwd': os.getcwd(), 'got': len(data)}))\n",
    )
    monkeypatch.setenv("MEMV2_CHILD_MARKER", "visible-to-the-parent-only")
    monkeypatch.setattr(mod, "_CHILD_SCRIPT", probe)
    raw, real = [], mod._run_child

    def keep(*args):
        out = real(*args)
        raw.append(out)
        return out

    monkeypatch.setattr(mod, "_run_child", keep)
    # the probe's report is not a shape, so the record gets the marker; the raw report is read here
    assert file_shape("secret-name.json", b'{"a": 1}') == {
        "format": "json",
        "unparsed": "invalid",
    }
    seen = json.loads(raw[0])
    assert "MEMV2_CHILD_MARKER" not in seen["env"]
    assert set(seen["env"]) <= {"LC_CTYPE"}  # at most Python's own C-locale coercion
    assert seen["limits"] == {
        "RLIMIT_AS": mod.CHILD_AS_BYTES,
        "RLIMIT_CPU": mod.CHILD_CPU_S,
        "RLIMIT_FSIZE": 0,
        "RLIMIT_NOFILE": mod.CHILD_NOFILE,
        "RLIMIT_NPROC": mod.CHILD_NPROC,
        # 1, not 0: the kernel ignores RLIMIT_CORE for a piped core_pattern handler, except that 1
        # aborts the dump; a file core needs at least a page, so 1 stops that too
        "RLIMIT_CORE": 1,
    }
    assert mod.CHILD_CORE == 1
    assert seen["isolated"] == 1 and seen["no_site"] == 1 and seen["cwd"] == "/"
    assert seen["got"] == 8  # the bytes arrive on stdin
    assert not any("secret-name" in a for a in seen["argv"])  # never the path


def test_shape_child_failures_give_markers_and_the_child_is_gone(tmp_path, monkeypatch):
    from unify.memory_v2.integration.adapters import worktree as mod

    def alive(path):
        for pid in filter(str.isdigit, os.listdir("/proc")):
            try:
                with open(f"/proc/{pid}/cmdline", "rb") as fh:
                    if str(path).encode() in fh.read():
                        return True
            except OSError:
                pass
        return False

    for body, want in (
        ("time.sleep(30)\n", "timeout"),
        ("sys.stdout.write('x' * 4_000_000)\n", "limit"),
        ("raise ValueError\n", "error"),
        ("os._exit(3)\n", "limit"),
        ("sys.stdout.write('[1, 2]')\n", "invalid"),
        ("sys.stdout.write('{not json')\n", "invalid"),
        ("os.kill(os.getpid(), 9)\n", "limit"),
    ):
        probe = _probe(tmp_path, body)
        monkeypatch.setattr(mod, "_CHILD_SCRIPT", probe)
        start = time.monotonic()
        assert file_shape("a.json", b"{}", timeout=0.5) == {
            "format": "json",
            "unparsed": want,
        }, body
        assert time.monotonic() - start < 2.5, body
        assert not alive(probe), body


def test_shape_child_imports_only_the_standard_library_and_the_parsers():
    from unify.memory_v2.integration.adapters.worktree_shapes import _ParsersOnly

    for name in (
        "json",
        "xml.etree.ElementTree",
        "yaml",
        "openpyxl.reader",
        "defusedxml",
    ):
        assert _ParsersOnly.find_spec(name) is None, name  # left to the normal finders
    for name in ("numpy", "lxml.etree", "PIL", "pandas", "unify"):
        with pytest.raises(ModuleNotFoundError):
            _ParsersOnly.find_spec(name)


def test_secrets_are_redacted_before_the_parse_cut(tmp_path, monkeypatch):
    from unify.memory_v2.integration.adapters import worktree as mod

    # a key-shaped secret straddling PARSE_CAP: cut first, its head would no longer look like a key
    body = b"a,b\n" * ((mod.PARSE_CAP - 20) // 4) + KEY.encode() + b",1\n"
    sent = []
    real = mod._run_child

    def spy(fmt, data, *args):
        sent.append(data)
        return real(fmt, data, *args)

    monkeypatch.setattr(mod, "_run_child", spy)
    # over the blob cap (shaped from the snapshot), and over the snapshot cap (shaped from disk)
    for i, kw in enumerate(({"blob_cap": 1024}, {"snapshot_cap": 4096})):
        root = tmp_path / str(i)
        root.mkdir()
        wt, repo, rec = _setup(root, **kw)
        (wt / "cut.csv").write_bytes(body)
        rec.begin()
        (row,) = rec.record_cell(0, [{"event": "open", "path": "cut.csv", "mode": "r"}])
        assert row.response["shape"]["format"] == "csv"
    assert len(sent) == 2
    assert all(KEY[:12].encode() not in data for data in sent)


# -- regressions for re-review 3 (I-C, n2, n4, n6) and controller ruling R27 ---------------------

# fake secrets; the sheet's has at most 31 characters, as an Excel sheet title
SHEET_SECRET = "FAKESHEET-0123456789abcd"  # pragma: allowlist secret
HEADER_SECRET = "FAKEHEADER-not-a-real-secret-0123456789"  # pragma: allowlist secret
CELL_SECRET = "FAKECELL-not-a-real-secret-0123456789"  # pragma: allowlist secret


def _workbook(sheet, headers, body):
    openpyxl = pytest.importorskip("openpyxl")
    import io

    wb = openpyxl.Workbook()
    wb.active.title = sheet
    wb.active.append(headers)
    wb.active.append(body)
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


def _exported(tmp_path, rec) -> bytes:
    """Everything the adapter exports: its rows, its notes and every blob in the blob store."""
    rows = json.dumps(
        [vars(a) for a in rec.actions] + [rec.skipped, dict(rec.skip_counts)],
        default=str,
    )
    blobs = [p.read_bytes() for p in (tmp_path / "blobs").rglob("*") if p.is_file()]
    return b"\n".join([rows.encode()] + blobs)


_CONTAINER_MAGIC = (b"PK", b"\x1f\x8b", b"BZh", b"\xfd7zXZ\x00", b"MZ")


def test_xlsx_header_secret_is_redacted_in_the_shape(tmp_path, monkeypatch):
    monkeypatch.setenv("MEMV2_TEST_EXPORT_API_KEY", HEADER_SECRET)
    data = _workbook("S", [HEADER_SECRET, "id"], [1, 2])
    # deflated: no byte scan of the file can see the secret
    assert HEADER_SECRET.encode() not in data
    wt, repo, rec = _setup(tmp_path)
    (wt / "conf.xlsx").write_bytes(data)
    rec.begin()
    (row,) = rec.record_cell(0, [{"event": "open", "path": "conf.xlsx", "mode": "rb"}])
    assert HEADER_SECRET not in json.dumps(row.response)
    assert row.response["shape"]["sheets"] == [
        {"name": "S", "headers": ["<secret:MEMV2_TEST_EXPORT_API_KEY>", "id"]},
    ]


def test_xlsx_secrets_never_reach_a_record_or_an_exported_blob(tmp_path, monkeypatch):
    import hashlib

    monkeypatch.setenv("MEMV2_TEST_SHEET_TOKEN", SHEET_SECRET)
    monkeypatch.setenv("MEMV2_TEST_HEADER_SECRET", HEADER_SECRET)
    monkeypatch.setenv("MEMV2_TEST_CELL_PASSWORD", CELL_SECRET)
    long_header = "x" * 100 + CELL_SECRET  # over MAX_NAME: never cut before redaction
    data = _workbook(
        SHEET_SECRET,
        [HEADER_SECRET, KEY, long_header, "id"],
        [CELL_SECRET, 1, 2, 3],
    )
    wt, repo, rec = _setup(tmp_path)
    (wt / "conf.xlsx").write_bytes(data)
    rec.begin()
    (row,) = rec.record_cell(0, [{"event": "open", "path": "conf.xlsx", "mode": "rb"}])
    assert row.status == "ok" and row.response["blob"] is None
    assert row.response["withheld"] == "binary" and "container" not in row.response
    assert row.response["sha256"] == hashlib.sha256(data).hexdigest()
    assert row.response["size"] == len(data)
    assert row.response["shape"]["sheets"] == [
        {
            "name": "<secret:MEMV2_TEST_SHEET_TOKEN>",
            "headers": [
                "<secret:MEMV2_TEST_HEADER_SECRET>",
                "<redacted:key-shaped>",
                {"chars": len(long_header)},
                "id",
            ],
        },
    ]
    new = _workbook(SHEET_SECRET, [HEADER_SECRET, "id"], [CELL_SECRET, 4])
    (wt / "conf.xlsx").write_bytes(new)
    rec.record_cell(1, [{"event": "open", "path": "conf.xlsx", "mode": "wb"}])
    (write,) = rec.finish()
    assert (
        write.response["blob_before"] is None and write.response["blob_after"] is None
    )
    assert write.response["sha256_before"] == hashlib.sha256(data).hexdigest()
    assert write.response["sha256_after"] == hashlib.sha256(new).hexdigest()
    assert write.response["withheld_after"] == "binary"
    exported = _exported(tmp_path, rec)
    for secret in (SHEET_SECRET, HEADER_SECRET, CELL_SECRET, KEY):
        assert secret.encode() not in exported, secret
    blobs = [p.read_bytes() for p in (tmp_path / "blobs").rglob("*") if p.is_file()]
    assert not any(b.startswith(_CONTAINER_MAGIC) for b in blobs)


def _tar(members):
    import io
    import tarfile

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        for name, data in members:
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def _binary_files(secret: bytes) -> dict:
    """Files whose bytes must never be exported (controller ruling R28): every one fails the text test,
    and in most the secret is compressed, so no byte scan can see it."""
    import bz2
    import gzip
    import io
    import lzma
    import zipfile
    import zlib

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as z:
        z.writestr("a.txt", secret)
    zipped = buf.getvalue()
    gz = gzip.compress(secret, mtime=0)
    return {
        "a.zip": zipped,
        "b.gz": gz,
        "c.txt": zipped,  # a zip renamed .txt
        "d.log": bz2.compress(secret),
        "e.dat": lzma.compress(secret),
        "g.bin": b"MZ" + bytes(100) + zipped,  # a zip behind a prefix
        # what signature detection missed (re-review 4, m-A)
        "h.tar": _tar([("cfg.gz", gz), ("big.txt", b"x" * 80 * 1024)]),
        "i.log": b"started\nok\n" + gz,  # a gzip member appended to text
        "j.joblib": zlib.compress(secret, 3),
        "k.lzma": lzma.compress(secret, format=lzma.FORMAT_ALONE),
        # not UTF-8, and a NUL in otherwise plain text
        "l.txt": b"caf\xe9 " + secret,
        "m.txt": b"plain text\0" + secret,
    }


def test_only_utf8_text_without_nul_is_exported_as_a_blob(tmp_path):
    import hashlib

    secret = f"token={KEY}\n".encode()
    files = _binary_files(secret)
    text = {
        "f.docx": b"not a zip at all",  # text under any name is text
        "n.gz": b"",
        "o.txt": "na\u00efve caf\u00e9\n".encode(),
    }
    wt, repo, rec = _setup(tmp_path)
    for name, raw in {**files, **text}.items():
        (wt / name).write_bytes(raw)
    rec.begin()
    rows = rec.record_cell(
        0,
        [
            {"event": "open", "path": p, "mode": "rb"}
            for p in [*files, *text, "notes.txt"]
        ],
    )
    for name, raw in files.items():
        got = _row(rows, "read", name).response
        assert got["blob"] is None and got["withheld"] == "binary", name
        assert "container" not in got, name
        assert got["sha256"] == hashlib.sha256(raw).hexdigest(), name
        assert got["size"] == len(raw) and isinstance(got["shape"]["format"], str), name
    for name in [*text, "notes.txt"]:
        got = _row(rows, "read", name).response
        assert got["blob"] and "withheld" not in got and "sha256" not in got, name
    new = files["b.gz"][:-8] + b"\x01" * 8
    (wt / "b.gz").write_bytes(new)
    rec.record_cell(1, [{"event": "open", "path": "b.gz", "mode": "wb"}])
    (write,) = rec.finish()
    assert (
        write.response["blob_before"] is None and write.response["blob_after"] is None
    )
    assert write.response["sha256_before"] == hashlib.sha256(files["b.gz"]).hexdigest()
    assert write.response["sha256_after"] == hashlib.sha256(new).hexdigest()
    assert (
        write.response["withheld_before"]
        == write.response["withheld_after"]
        == "binary"
    )
    blobs = [p.read_bytes() for p in (tmp_path / "blobs").rglob("*") if p.is_file()]
    assert blobs
    for b in blobs:  # the blob store holds text only
        assert b"\0" not in b
        b.decode("utf-8")
    assert not any(b.startswith(_CONTAINER_MAGIC) for b in blobs)
    assert KEY.encode() not in _exported(tmp_path, rec)


def test_child_output_outside_the_schema_is_invalid(tmp_path, monkeypatch):
    from unify.memory_v2.integration.adapters import worktree as mod

    deep = b'{"format": "json", "keys": ' + b'{"a": ' * 10_000 + b"1" + b"}" * 10_001
    bad = (
        ("a.json", b'{"format": "json", "type": NaN}'),
        ("a.json", b'{"format": "json", "type": "object", "more_keys": Infinity}'),
        ("a.json", deep),  # 10,000 levels
        ("a.json", b"[" * 100_000),
        ("a.json", b'{"format": "json", "type": "object", "evil": "x"}'),  # unknown key
        (
            "a.json",
            b'{"format": "json", "type": "object", "keys": {"'
            + b"k" * 5000
            + b'": "int"}}',
        ),
        (
            "a.json",
            b'{"format": "json", "type": "object", "keys": {"a": "'
            + b"x" * 300
            + b'"}}',
        ),
        ("a.json", b'{"format": "json", "type": "object", "keys": {"a": "nonsense"}}'),
        ("a.json", b'{"format": "json", "type": "object", "more_keys": 1.5}'),
        ("a.json", b'{"format": "json", "type": "object", "more_keys": -1}'),
        # not a format a JSON file gives
        ("a.json", b'{"format": "xlsx", "sheets": []}'),
        ("a.json", b"\xff\xfe{}"),  # not UTF-8
        ("a.csv", b'{"format": "csv", "columns": [1, 2]}'),
        ("a.csv", b'{"format": "csv", "columns": ["a"], "delimiter": "x"}'),
        (
            "a.xlsx",
            b'{"format": "xlsx", "sheets": [{"name": "s", "headers": [], "rows": [["v"]]}]}',
        ),
        (
            "a.xlsx",
            b'{"format": "xlsx", "sheets": [{"name": {"chars": 3, "v": "x"}, "headers": []}]}',
        ),
        (
            "a.json",
            b'{"format": "json", "type": "object", "keys": {'
            + b", ".join(b'"k%d": "int"' % i for i in range(100))
            + b"}}",
        ),  # more keys than one level holds
    )
    good = b'{"format": "json", "type": "object", "keys": {"a": "int", "b": {"type": "array"}}}'
    for path, out in bad + (("a.json", good),):
        probe = _probe(tmp_path, f"sys.stdout.buffer.write({out!r})\n")
        monkeypatch.setattr(mod, "_CHILD_SCRIPT", probe)
        fmt = mod._format_of(path)
        want = (
            json.loads(good) if out is good else {"format": fmt, "unparsed": "invalid"}
        )
        assert file_shape(path, b"{}", timeout=2.0) == want, out[:80]


def test_every_shape_string_is_redacted_whatever_the_format(tmp_path):
    pytest.importorskip("yaml")
    # escapes the byte scan cannot see: the parsed name is the key itself
    escaped = "\\u0073" + KEY[1:]
    wt, repo, rec = _setup(tmp_path)
    (wt / "k.json").write_text('{"%s": 1}' % escaped)
    (wt / "k.yaml").write_text('"\\x73%s": 1\n' % KEY[1:])
    rec.begin()
    rows = rec.record_cell(
        0,
        [{"event": "open", "path": p, "mode": "r"} for p in ("k.json", "k.yaml")],
    )
    for row in rows:
        assert row.response["shape"]["keys"] == {
            "<redacted:key-shaped>": "int",
        }, row.args
    assert KEY.encode() not in _exported(tmp_path, rec)


def test_one_failing_path_in_finish_keeps_the_other_rows(tmp_path, monkeypatch):
    wt, repo, rec = _setup(tmp_path)
    rec.begin()
    (wt / "notes.txt").write_text("changed\n")
    (wt / "conf.json").write_text('{"a": 1}')
    (wt / "pay.csv").write_text("x,y\n1,2\n")
    real = rec._content

    def flaky(rel, oid, **kw):
        if rel == "conf.json":
            raise RuntimeError("shape child group 1 survived SIGKILL")
        if rel == "pay.csv":
            raise RecursionError("hostile")
        return real(rel, oid, **kw)

    monkeypatch.setattr(rec, "_content", flaky)
    rows = rec.finish()
    by = {r.args[0]: r for r in rows}
    assert set(by) == {"conf.json", "notes.txt", "pay.csv"}
    assert by["notes.txt"].status == "ok"
    assert rec.blobs.get(by["notes.txt"].response["blob_after"]) == b"changed\n"
    before, after = tree_files(repo, rec.before), tree_files(repo, rec.after)
    for path, exc in (("conf.json", "RuntimeError"), ("pay.csv", "RecursionError")):
        got = by[path]
        assert got.status == "unrecorded" and got.response["failed"] == exc, path
        # unknown whether a secret was redacted from it: no raw git id (R30)
        assert got.response["oid_before"] is None, path
        assert got.response["oid_after"] is None, path
    assert rec.skip_counts["write record failed"] == 2


def test_benign_yaml_near_the_parse_cap_shapes_within_the_timeout():
    pytest.importorskip("yaml")
    from unify.memory_v2.integration.adapters import worktree as mod

    doc = "".join(
        f"item{i:05d}:\n  name: n{i}\n  tags: [a, b, {i}]\n  when: 2026-01-02\n"
        f"  nested: {{x: 1, y: [1, 2]}}\n"
        for i in range(6500)
    ).encode()
    assert 550_000 < len(doc) < mod.PARSE_CAP
    start = time.monotonic()
    shape = file_shape("big.yaml", doc)
    elapsed = time.monotonic() - start
    assert "unparsed" not in shape and "error" not in shape, shape
    assert shape["more_keys"] == 6500 - 64
    assert shape["keys"]["item00000"]["keys"]["tags"] == {
        "type": "array",
        "items": "str",
    }
    assert elapsed < mod.PARSE_TIMEOUT


def test_deep_yaml_is_unparsed_not_a_crash():
    pytest.importorskip("yaml")
    for doc in (b"- " * 100_000 + b"x\n", b"[" * 100_000 + b"]" * 100_000):
        assert file_shape("deep.yaml", doc) == {"format": "yaml", "error": "unparsed"}


def test_finish_does_not_shape_the_before_side(tmp_path, monkeypatch):
    from unify.memory_v2.integration.adapters import worktree as mod

    wt, repo, rec = _setup(tmp_path)
    rec.begin()
    (wt / "conf.json").write_text(json.dumps({"b": 2}))
    spawned, real = [], mod._run_child

    def counting(fmt, data, *args):
        spawned.append(data)
        return real(fmt, data, *args)

    monkeypatch.setattr(mod, "_run_child", counting)
    (row,) = rec.finish()
    assert row.response["shape"]["keys"] == {"b": "int"}
    assert row.response["blob_before"] and row.response["blob_after"]
    assert [json.loads(d) for d in spawned] == [{"b": 2}]  # the after side only


def test_answer_nesting_is_refused_before_parsing(monkeypatch):
    from unify.memory_v2.integration.adapters import worktree_shapes as shapes

    def parsed(*args, **kwargs):
        raise AssertionError("json.loads ran on a too-deep answer")

    monkeypatch.setattr(shapes.json, "loads", parsed)
    for deep in (
        b"[" * 100_000,
        b"[" * 17 + b"]" * 17,
        b'{"a": ' * 17 + b"1" + b"}" * 17,
    ):
        assert shapes.parse_answer(deep) is None
    # brackets inside strings and escaped quotes are not nesting
    assert shapes._nesting_within('{"a": "[[[[[[[[[[[[[[[[[[[[{{{{{"}', 1)
    assert shapes._nesting_within('{"a": "\\\\", "b": "\\"[[[["}', 1)
    assert not shapes._nesting_within('{"a": [[1]]}', 2)


# -- pre-wiring round (re-review 4: m-D, m-B, m-C, m-E) ---------------------------------------------

PEM_SECRET = "-----BEGIN FAKE KEY-----\nFAKEPEM-not-a-real-secret-0123456789\n-----END FAKE KEY-----"  # pragma: allowlist secret


def test_worktree_diff_exports_text_hunks_and_binary_hashes_only(tmp_path, monkeypatch):
    import hashlib

    from unify.memory_v2.integration.adapters import worktree as mod

    monkeypatch.setenv("MEMV2_TEST_PEM_KEY", PEM_SECRET)
    files = _binary_files(f"token={KEY}\n".encode())
    wt, repo, rec = _setup(tmp_path)
    for name, raw in files.items():
        (wt / name).write_bytes(raw)
    (wt / "gone.txt").write_text("bye\n")
    (wt / "long.txt").write_text("".join(f"{i}\n" for i in range(mod.DIFF_LINES + 1)))
    rec.begin()
    # a multi-line secret: redacted in each whole side, so diff prefixes never split it
    (wt / "notes.txt").write_text("one\nchanged\nthree\n" + PEM_SECRET + "\n")
    (wt / "new.txt").write_text(f"fresh\ntoken={KEY}\n")
    (wt / "gone.txt").unlink()
    (wt / "sub" / "a.txt").write_bytes(b"a\n\0")  # text to binary
    (wt / "long.txt").write_text("x\n" * (mod.DIFF_LINES + 1))
    for name, raw in files.items():
        (wt / name).write_bytes(raw + b"\0more")
    rec.finish()
    argv = []
    real = mod._git_raw

    def spy(repo, args, **kw):
        argv.append(list(args))
        return real(repo, args, **kw)

    monkeypatch.setattr(mod, "_git_raw", spy)
    diff = rec.diff()
    # never `git diff`: --binary emits base85 of every file, and its text test only looks for a NUL
    assert argv and all("diff" not in args for args in argv)
    for line in ("-two\n", "+changed\n", "+fresh\n", "-bye\n"):
        assert line in diff, line
    assert "+<secret:MEMV2_TEST_PEM_KEY>\n" in diff and "FAKEPEM" not in diff
    assert KEY not in diff and "\0" not in diff
    # hunks for text pairs only; every other changed path is one line of hashes and sizes
    hunks = {ln[6:] for ln in diff.splitlines() if ln.startswith("+++ b/")}
    assert hunks == {"notes.txt", "new.txt"}
    assert "+++ /dev/null" in diff and "--- a/gone.txt" in diff
    for name, raw in files.items():
        new = raw + b"\0more"
        assert (
            f"binary {name}: sha256 {hashlib.sha256(raw).hexdigest()} -> "
            f"{hashlib.sha256(new).hexdigest()}, bytes {len(raw)} -> {len(new)}\n"
        ) in diff, name
    assert "binary sub/a.txt: sha256 " in diff
    assert "text long.txt: sha256 " in diff and "+x\n" not in diff  # over DIFF_LINES
    # redacted whole, then cut: a cut inside the key never leaves its head
    at = diff.index("<redacted:key-shaped>")
    for cap in (at + 3, at + 12, at + 40):
        cut = rec.diff(cap=cap)
        assert KEY[:12] not in cut, cap
        assert cut.endswith("changed paths not shown: size cap]\n"), cap


NAME_SECRET = "FAKENAME-not-a-real-secret-0123456789"  # pragma: allowlist secret


def _names(shape) -> list:
    """Every key and column name in a shape."""
    found = list(shape.get("columns", []))
    for node in (shape, shape.get("record"), shape.get("items")):
        while isinstance(node, dict):
            found += list(node.get("keys", {}))
            for child in node.get("keys", {}).values():
                if isinstance(child, dict):
                    found += list(child.get("keys", {}))
            node = node.get("items")
    return found


def test_long_names_are_redacted_whole_before_the_cut():
    pytest.importorskip("yaml")
    from unify.memory_v2.integration.adapters import worktree_shapes as ws

    redactor = Redactor({"MEMV2_TEST_NAME_TOKEN": NAME_SECRET})
    pad = "p" * 120  # the secret straddles MAX_NAME (128)
    for secret in (NAME_SECRET, KEY):
        name = pad + secret
        huge = "q" * ws.MAX_NAME_SENT + secret  # sent as its length only
        files = {
            "k.json": json.dumps({name: 1, "n": {name: [1]}, huge: 2}).encode(),
            "k.yaml": f'"{name}": 1\nn:\n  "{name}": [1]\n? "{huge}"\n: 2\n'.encode(),
            "k.jsonl": json.dumps({name: 1, huge: 2}).encode() + b"\n",
            "c.csv": f"{name},{huge},id\nx,y,1\n".encode(),
        }
        for path, data in files.items():
            shape = file_shape(path, data, redactor=redactor)
            dumped = json.dumps(shape)
            assert "unparsed" not in shape and "error" not in shape, (path, shape)
            assert secret[:6] not in dumped and "qqqq" not in dumped, (path, dumped)
            names = _names(shape)
            assert all(len(n) <= ws.MAX_NAME for n in names), path
            assert any(n.startswith(pad + "<") for n in names), (path, names)
            assert any(n.startswith("<name ") for n in names), (path, names)
            flags = [
                shape,
                shape.get("record") or {},
                shape.get("keys", {}).get("n", {}),
            ]
            assert any(f.get("names_truncated") for f in flags), path


def test_yaml_omap_and_pairs_shape_as_mappings():
    pytest.importorskip("yaml")
    # the safe loaders build !!omap and !!pairs as lists of (key, value) tuples
    assert file_shape("o.yaml", b"!!omap\n- a: 1\n- b: x\n- c: [1]\n") == {
        "format": "yaml",
        "type": "object",
        "keys": {"a": "int", "b": "str", "c": {"type": "array", "items": "int"}},
    }
    nested = b"cfg: !!omap [x: 1, y: {z: 2}]\npairs: !!pairs [k: 1, k: x, j: 2]\n"
    assert file_shape("n.yaml", nested) == {
        "format": "yaml",
        "type": "object",
        "keys": {
            "cfg": {
                "type": "object",
                "keys": {"x": "int", "y": {"type": "object", "keys": {"z": "int"}}},
            },
            "pairs": {"type": "object", "keys": {"k": "int", "j": "int"}},
        },
    }
    wide = file_shape(
        "w.yaml",
        ("!!omap\n" + "".join(f"- k{i}: {i}\n" for i in range(70))).encode(),
    )
    assert len(wide["keys"]) == 64 and wide["more_keys"] == 6
    # a plain list of one-key mappings stays an array
    assert file_shape("l.yaml", b"- a: 1\n- b: 2\n") == {
        "format": "yaml",
        "type": "array",
        "items": {"type": "object", "keys": {"a": "int"}},
    }


EMPTY_SHA = hashlib.sha256(b"").hexdigest()


def test_worktree_diff_lists_every_changed_path(tmp_path):
    import hashlib

    from unify.memory_v2.integration.adapters import worktree as mod

    other = "sk-or-v1-" + "cd" * 32  # pragma: allowlist secret
    wt, repo, rec = _setup(tmp_path)
    (wt / "empty-del.txt").write_text("")
    (wt / "rot.txt").write_text(f"token={KEY}\n")
    rec.begin()
    (wt / "empty-new.txt").write_text("")
    (wt / "empty-del.txt").unlink()
    (wt / "rot.txt").write_text(f"token={other}\n")  # only a redacted span changes
    rec.finish()
    diff = mod.worktree_diff(repo, rec.before, rec.after, redactor=Redactor())
    old, new = (len(f"token={k}\n") for k in (KEY, other))
    red = hashlib.sha256(b"token=<redacted:key-shaped>\n").hexdigest()  # R30
    assert diff == (
        f"text empty-del.txt: sha256 {EMPTY_SHA} -> absent, bytes 0 -> absent\n"
        f"text empty-new.txt: sha256 absent -> {EMPTY_SHA}, bytes absent -> 0\n"
        f"text rot.txt: changed; content redacted, sha256 {red} -> {red}, "
        f"bytes {old} -> {new}\n"
    )


def test_worktree_diff_is_bounded_and_survives_git_errors(tmp_path, monkeypatch):
    from unify.memory_v2.gitio import GitError
    from unify.memory_v2.integration.adapters import worktree as mod

    wt, repo, rec = _setup(tmp_path)
    for i in range(30):
        (wt / f"f{i:02d}.txt").write_text(f"old {i}\n")
    rec.begin()
    for i in range(30):
        (wt / f"f{i:02d}.txt").write_text(f"new {i}\n")
    rec.finish()
    r = Redactor()

    def hunks(diff):
        return [ln[6:] for ln in diff.splitlines() if ln.startswith("+++ b/")]

    full = mod.worktree_diff(repo, rec.before, rec.after, redactor=r)
    assert len(hunks(full)) == 30 and "not shown" not in full
    # a path bound: every changed path counts, and the rest are counted in a marker
    few = mod.worktree_diff(repo, rec.before, rec.after, redactor=r, max_paths=5)
    assert hunks(few) == [f"f{i:02d}.txt" for i in range(5)]
    assert few.endswith("[25 changed paths not shown: path limit]\n")
    # a size cap: the paths not reached are counted too
    small = mod.worktree_diff(repo, rec.before, rec.after, redactor=r, cap=200)
    hidden = int(small.rsplit("[", 1)[1].split(" ")[0])
    assert small.endswith(f"[{hidden} changed paths not shown: size cap]\n")
    assert len(hunks(small)) + hidden == 30 and len(small) < 200 + 50
    # a wall-clock deadline over every path, git included
    real = mod._git_raw
    oids = mod.tree_files(repo, rec.before)

    def slow(repo_, args, **kw):
        time.sleep(0.05)
        return real(repo_, args, **kw)

    monkeypatch.setattr(mod, "_git_raw", slow)
    start = time.monotonic()
    late = mod.worktree_diff(repo, rec.before, rec.after, redactor=r, seconds=0.3)
    assert time.monotonic() - start < 1.0
    assert 0 < len(hunks(late)) < 30 and late.endswith(": deadline]\n")

    # one failing path gives a marker line; the others are still shown
    def flaky(repo_, args, **kw):
        if args[-1] == oids["f03.txt"]:
            raise GitError("cat-file failed")
        return real(repo_, args, **kw)

    monkeypatch.setattr(mod, "_git_raw", flaky)
    hurt = mod.worktree_diff(repo, rec.before, rec.after, redactor=r)
    assert "failed f03.txt: GitError\n" in hurt and "cat-file failed" not in hurt
    assert len(hunks(hurt)) == 29

    def broken(*args, **kw):
        raise GitError("no such commit")

    monkeypatch.setattr(mod, "tree_files", broken)
    assert mod.worktree_diff(repo, "x", "y", redactor=r) == (
        "[work-tree diff unavailable: GitError]\n"
    )


def test_worktree_diff_uses_the_recorders_redactor(tmp_path):
    from unify.memory_v2.integration.adapters import worktree as mod

    secret = "FAKEDIFF-not-a-real-secret-0123456789"  # pragma: allowlist secret
    wt, repo, rec = _setup(tmp_path, redactor=Redactor({"RUN_SECRET": secret}))
    rec.begin()
    with pytest.raises(RuntimeError):
        rec.diff()  # before finish()
    (wt / "notes.txt").write_text(f"pw={secret}\n")
    rec.finish()
    diff = rec.diff()
    assert "+pw=<secret:RUN_SECRET>\n" in diff and secret not in diff
    # no default: a caller cannot fall back to a redactor that knows only the environment
    with pytest.raises(TypeError):
        mod.worktree_diff(repo, rec.before, rec.after)


def test_worktree_diff_splits_lines_on_newline_only(tmp_path):
    wt, repo, rec = _setup(tmp_path)
    tail = "x\x0by\x0cz\x1c\x85  w\n"  # other Unicode line boundaries stay inside lines
    (wt / "cr.txt").write_bytes(("a\rb\rc\nkeep\n" + tail).encode())
    rec.begin()
    (wt / "cr.txt").write_bytes(("a\rB\rc\nkeep\n" + tail + "end").encode())
    rec.finish()
    diff = rec.diff()
    assert "-a\rb\rc\n+a\rB\rc\n keep\n " + tail in diff
    assert diff.endswith("+end\n\\ No newline at end of file\n")
    # every line is a header or starts with its marker
    for line in diff.split("\n")[:-1]:
        assert line[:1] in (" ", "-", "+", "@", "\\"), repr(line)


def test_yaml_keys_that_are_not_scalars_are_never_names():
    pytest.importorskip("yaml")
    doc = b"o: !!omap [? [1, [2, [3]]] : 1, ? {a: hidden-value} : 2, b: 3]\n"
    shape = file_shape("k.yaml", doc + b"? !!binary aGlkZGVuLWJ5dGVz\n: 1\n")
    assert shape["keys"] == {
        "o": {
            "type": "object",
            "keys": {"<key 0: array>": "int", "<key 1: object>": "int", "b": "int"},
        },
        "<key 1: bytes>": "int",
    }
    dumped = json.dumps(shape)
    assert "hidden" not in dumped and "[2" not in dumped
    # an alias-expanded key is never expanded into a name: one key of the budget, at once
    levels = ["l0: &a0 [x, x, x, x, x, x, x, x, x]"]
    levels += [
        f"l{i}: &a{i} [" + ", ".join([f"*a{i - 1}"] * 9) + "]" for i in range(1, 9)
    ]
    bomb = "\n".join(levels) + "\no: !!omap [? *a8 : 1]\n"
    start = time.monotonic()
    shape = file_shape("b.yaml", bomb.encode())
    assert time.monotonic() - start < 1.5
    assert shape["keys"]["o"] == {"type": "object", "keys": {"<key 0: array>": "int"}}


def test_worktree_diff_counts_every_path_the_size_cap_hides(tmp_path):
    wt, repo, rec = _setup(tmp_path)
    for name in ("f1.txt", "f2.txt", "f3.txt"):
        (wt / name).write_text("old\n")
    rec.begin()
    for name in ("f1.txt", "f2.txt", "f3.txt"):
        (wt / name).write_text("new\n")
    rec.finish()
    full = rec.diff()
    first = full.index("--- a/f2.txt")
    # the cap falls exactly at the end of f1's part: f2 and f3 are both hidden, and counted
    assert (
        rec.diff(cap=first) == full[:first] + "[2 changed paths not shown: size cap]\n"
    )
    assert rec.diff(cap=first + 5) == rec.diff(cap=first)  # no part is shown in pieces
    # only a first part longer than the cap is cut, after redaction
    root = tmp_path / "k"
    root.mkdir()
    wt, repo, rec = _setup(root)
    rec.begin()
    (wt / "notes.txt").write_text(f"token={KEY}\n")
    rec.finish()
    full = rec.diff()
    at = full.index("<redacted:key-shaped>")
    for cap in (at + 3, at + 12):
        cut = rec.diff(cap=cap)
        assert cut == full[:cap] + f"\n[work-tree diff cut at {cap} characters]\n"
        assert KEY[:12] not in cut


def test_worktree_diff_deadline_covers_listing_and_timeouts(tmp_path, monkeypatch):
    from unify.memory_v2.integration.adapters import worktree as mod

    wt, repo, rec = _setup(tmp_path)
    for i in range(5):
        (wt / f"f{i}.txt").write_text(f"old {i}\n")  # distinct blobs
    rec.begin()
    for i in range(5):
        (wt / f"f{i}.txt").write_text(f"new {i}\n")
    rec.finish()
    oids = mod.tree_files(repo, rec.before)
    real = mod._git_raw
    # a git call past its time is a GitTimeout (a GitError), not a generic failure
    with pytest.raises(mod.GitTimeout):
        real(repo, ["cat-file", "blob", oids["f0.txt"]], timeout=1e-6)
    seen = []

    def spy(repo_, args, **kw):
        seen.append((args[0], kw.get("timeout")))
        return real(repo_, args, **kw)

    monkeypatch.setattr(mod, "_git_raw", spy)
    rec.diff(seconds=5.0)
    assert {name for name, _ in seen} == {"ls-tree", "cat-file"}
    assert all(t is not None and t <= 5.0 for _, t in seen), seen

    def list_times_out(repo_, args, **kw):
        if args[0] == "ls-tree":
            raise mod.GitTimeout("git ls-tree timed out")
        return real(repo_, args, **kw)

    monkeypatch.setattr(mod, "_git_raw", list_times_out)
    assert rec.diff() == "[work-tree diff not shown: deadline]\n"

    def blob_times_out(repo_, args, **kw):
        if args[-1] == oids["f2.txt"]:
            raise mod.GitTimeout("git cat-file timed out")
        return real(repo_, args, **kw)

    monkeypatch.setattr(mod, "_git_raw", blob_times_out)
    out = rec.diff()
    shown = [ln[6:] for ln in out.splitlines() if ln.startswith("+++ b/")]
    assert shown == ["f0.txt", "f1.txt"] and "failed" not in out
    assert out.endswith("[3 changed paths not shown: deadline]\n")


def _raw_ids(data: bytes) -> set:
    """The SHA-256 and the git blob id of *data*."""
    oid = hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()  # noqa: S324
    return {hashlib.sha256(data).hexdigest(), oid}


def test_no_exported_hash_or_id_is_of_raw_text_that_was_redacted(tmp_path, monkeypatch):
    from unify.memory_v2.integration.adapters import worktree as mod

    pw = "hunter2pass"  # pragma: allowlist secret - short and guessable: an offline hash test finds it
    monkeypatch.setenv("MEMV2_TEST_DB_PASSWORD", pw)
    other = "sk-or-v1-" + "cd" * 32  # pragma: allowlist secret
    # over DIFF_LINES: a hash line
    long_head = "".join(f"{i}\n" for i in range(mod.DIFF_LINES))
    before = {
        "keep.txt": f"pw={pw}\n",
        ".env": f"pw={pw}\nmode=a\n",
        "big.txt": long_head + f"pw={pw}\n",
        "tok.txt": f"token={KEY}\n",
    }
    after = {
        ".env": f"pw={pw}\nmode=b\n",
        "big.txt": long_head + f"pw={pw}\nmore\n",
        "tok.txt": f"token={other}\n",  # the change is hidden by redaction
    }
    wt, repo, rec = _setup(tmp_path)
    for name, text in before.items():
        (wt / name).write_text(text)
    rec.begin()
    reads = rec.record_cell(
        0,
        [{"event": "open", "path": p, "mode": "r"} for p in ("keep.txt", "notes.txt")],
    )
    for name, text in after.items():
        (wt / name).write_text(text)
    rec.record_cell(
        1,
        [{"event": "open", "path": p, "mode": "w"} for p in after],
    )
    writes = rec.finish()
    diff = rec.diff()
    blob_names = " ".join(
        p.parent.name + p.name for p in (tmp_path / "blobs").rglob("*") if p.is_file()
    )
    exported = _exported(tmp_path, rec).decode("utf-8", "replace") + diff + blob_names
    assert pw not in exported
    for text in [*before.values(), *after.values()]:
        for raw_id in _raw_ids(text.encode()):
            assert raw_id not in exported, text[-30:]
    # rows drop the raw git id of redacted text, and say so; the blob id is of the redacted text
    keep = _row(reads, "read", "keep.txt").response
    assert keep["oid"] is None and keep["redacted"] is True and keep["blob"]
    assert (
        keep["blob"]
        == hashlib.sha256(b"pw=<secret:MEMV2_TEST_DB_PASSWORD>\n").hexdigest()
    )
    for row in writes:
        got = row.response
        assert got["oid_before"] is None and got["oid_after"] is None, row.args
        assert (
            got["redacted_before"] is True and got["redacted_after"] is True
        ), row.args
    # read from disk (written earlier in the request): no raw id either way, and the flag
    disk = rec.record_cell(2, [{"event": "open", "path": ".env", "mode": "r"}])
    assert disk[0].response["source"] == "disk" and disk[0].response["redacted"] is True
    # text with no redaction keeps its raw id
    notes = _row(reads, "read", "notes.txt").response
    assert notes["oid"] == mod.tree_files(repo, rec.before)["notes.txt"]
    assert "redacted" not in notes
    # the diff's hash lines hash the redacted text: equal sides when only a secret changed
    red = hashlib.sha256(b"token=<redacted:key-shaped>\n").hexdigest()
    assert f"text tok.txt: changed; content redacted, sha256 {red} -> {red}," in diff
    assert "text big.txt: sha256 " in diff


# -- review fix round (online Track A, I2): one blob process and deadlines ------------------------------


def _many(tmp_path, n=30):
    wt, repo, rec = _setup(tmp_path)
    for i in range(n):
        (wt / f"f{i:02d}.txt").write_text(f"file {i}\n")
    return wt, repo, rec


def test_blob_reads_share_one_cat_file_batch_process(tmp_path, monkeypatch):
    import unify.memory_v2.integration.adapters.worktree as mod

    wt, repo, rec = _many(tmp_path)
    rec.begin()
    started, real_popen = [], mod.subprocess.Popen
    raw_cats, real_raw = [], mod._git_raw

    def popen(argv, **kw):
        started.append(list(argv))
        return real_popen(argv, **kw)

    def raw(repo_, args, **kw):
        if args[:1] == ["cat-file"]:
            raw_cats.append(args)
        return real_raw(repo_, args, **kw)

    monkeypatch.setattr(mod.subprocess, "Popen", popen)
    monkeypatch.setattr(mod, "_git_raw", raw)
    for i in range(30):
        (wt / f"f{i:02d}.txt").write_text(f"changed {i}\n")
    rows = rec.record_cell(
        0,
        [{"event": "open", "path": f"f{i:02d}.txt", "mode": "r"} for i in range(30)],
    )
    assert len(rows) == 30 and all(r.status == "ok" for r in rows)
    assert rec.blobs.get(rows[3].response["blob"]) == b"file 3\n"  # the before snapshot
    writes = rec.finish()
    assert len(writes) == 30
    assert rec.blobs.get(writes[0].response["blob_after"]) == b"changed 0\n"
    batches = [a for a in started if "cat-file" in a]
    assert len(batches) == 1 and batches[0][-2:] == ["cat-file", "--batch"]
    assert "core.hooksPath=/dev/null" in batches[0]
    assert raw_cats == []
    rec.close()
    rec.close()  # idempotent


def test_records_past_the_deadline_are_counted_and_get_no_row(tmp_path):
    import time

    wt, repo, rec = _many(tmp_path, 5)
    rec.begin()
    recs = [{"event": "open", "path": f"f{i:02d}.txt", "mode": "r"} for i in range(5)]
    assert rec.record_cell(0, recs, deadline=time.monotonic() - 1) == []
    assert rec.skip_counts["deadline: audit records not recorded"] == 5
    assert len(rec.record_cell(1, recs, deadline=time.monotonic() + 60)) == 5
    rec.close()


def test_write_rows_past_the_deadline_are_counted(tmp_path, monkeypatch):
    import time

    wt, repo, rec = _many(tmp_path, 30)
    rec.begin()
    for i in range(30):
        (wt / f"f{i:02d}.txt").write_text(f"changed {i}\n")
    real = rec._write_response

    def slow(*a, **kw):
        time.sleep(0.1)
        return real(*a, **kw)

    monkeypatch.setattr(rec, "_write_response", slow)
    start = time.monotonic()
    rows = rec.finish(deadline=start + 1.5)
    took = time.monotonic() - start
    cut = rec.skip_counts["deadline: write rows not recorded"]
    assert rec.after and cut > 0 and len(rows) + cut == 30
    assert took < 3.0
    rec.close()


def test_a_snapshot_past_its_deadline_raises_and_commits_nothing(tmp_path):
    import time

    from unify.memory_v2.integration.adapters.worktree import GitTimeout

    wt, repo, rec = _many(tmp_path, 5)
    before = rec.begin()
    try:
        rec.finish(deadline=time.monotonic() - 1)
    except GitTimeout:
        pass
    else:
        raise AssertionError("expected GitTimeout")
    assert rec.after is None
    from unify.memory_v2.integration.adapters.worktree import _git_raw

    head = _git_raw(repo, ["rev-parse", "refs/heads/main"]).decode().strip()
    assert head == before
    rec.close()
