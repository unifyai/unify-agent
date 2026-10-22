"""The library mode's candidate pool: proposals persist across passes on the memory repo's ``refs/heads/pool`` until
the writer closes them. The design's seven tests: persistence, accumulation, status only from the writer, dedup,
isolation, determinism and the off path.
"""

import asyncio
import json
import os
import subprocess

import pytest

from unify.memory.library_checks import LibraryConfig
from unify.memory import pool, writer_pass
from unify.memory.blobs import BlobStore
from unify.memory.evidence import EvidenceStore
from unify.memory.gate import Gate
from unify.memory.gitio import Repo
from unify.memory.integration.checkout import export_checkout
from unify.memory.memory_writer import publish
from unify.memory.writer_pass import PassConfig, WriterPass
from unify.memory.staging import StagingView
from unify.memory.trigger import PassRequest
from tests.memory import writer_library_items as wl
from tests.memory.test_writer_pass import Turns, _call, _e1, _ME, needs_bwrap
from tests.memory.test_writer_library_tools import EPS

#: A manifest that lists nothing (the library item's manifest keys, no item)
MAN = {"items": [wl.ITEM], "unlisted": [], "deleted": [], "support": []}
ITEM = wl.ITEM
EMPTY = {**MAN, "items": [], "support": []}
INDEX = "/inputs/pool/INDEX.md"


class _Stores:
    """One memory repo and evidence store across several passes."""

    def __init__(self, tmp_path, **gate_kw):
        self.mem = Repo.init_bare(tmp_path / "mem.git")
        self.ev = EvidenceStore(tmp_path / "e.sqlite")
        self.gate = Gate(self.mem, self.ev, BlobStore(tmp_path / "b"), library_mode=LibraryConfig(), **gate_kw)
        self.run_home = str(tmp_path / "run-home")

    def run(self, turns, pass_id, eids=("e2",), load=EPS.__getitem__, **cfg):
        model = Turns(turns)
        cfg.setdefault("max_calls", 12)
        writer = WriterPass(
            self.mem,
            self.gate,
            self.ev,
            load=load,
            model_turn=model,
            config=PassConfig(run_home=self.run_home, **cfg),
        )
        out = asyncio.run(
            writer.run(PassRequest("incremental", "svc", list(eids), False), pass_id),
        )
        return out, model

    def pool(self):
        return pool.Pool(self.mem).load()


def _finish_with(monkeypatch, manifest):
    monkeypatch.setattr(
        writer_pass,
        "_read_manifest",
        lambda box: (True, manifest, None),
    )


def _defer(eid, reason="a rule that may come back"):
    """A manifest that defers *eid* (what ``pool`` decide defer with id 'new' writes)."""
    return {
        **EMPTY,
        "pool": {"defer": [{"id": "new", "episodes": [eid], "reason": reason}]},
    }


def _open(eid):
    """A read of *eid*, so the pass does not record it as unread."""
    return _call(f"o-{eid}", "read_episode", {"episode": eid, "parts": ["request"]})


# --- 1. persistence -------------------------------------------------------------------------------------------


@needs_bwrap
def test_a_deferred_episode_is_listed_next_pass_merged_there_and_then_gone(
    tmp_path,
    monkeypatch,
):
    ep = _e1()
    lookup = lambda eid, i: _ME if (eid, i) == ("e1", 0) else None  # noqa: E731
    st = _Stores(tmp_path, action_lookup=lookup)
    st.ev.index_episode(ep, "1" * 40)
    did = pool.entry_id("deferred", "e1")

    with monkeypatch.context() as m:
        _finish_with(m, _defer("e1"))
        out1, _ = st.run([], "p1", eids=("e1",), load=lambda e: ep)
    assert out1.finished and out1.pool["added"] == 1 and out1.pool["commit"]
    assert st.pool().entries[did].status == "open"

    manifest = {
        **MAN,
        "items": [ITEM],
        "summary": "s",
        "pool": {"merged": {did: ITEM["item"]}},
    }
    files = wl.files_code(manifest=manifest)
    out2, model2 = st.run(
        [
            _call("r", "read", {"path": INDEX}),
            _open("e1"),
            _call("w", "execute_code", {"code": files}),
            _call("f", "finish", {"summary": "merged the deferred rule"}),
        ],
        "p2",
        eids=("e1",),
        load=lambda e: ep,
        max_calls=20,
    )
    assert (
        f"`{did}` deferred: a rule that may come back (1 episode; opened in p1)"
        in model2.outputs["r"]
    )
    assert out2.passed and ITEM["item"] in out2.items_merged, out2.reasons
    e = st.pool().entries[did]
    assert e.status == "merged" and e.status_detail == {
        "item": ITEM["item"],
        "pass": "p2",
    }

    with monkeypatch.context() as m:
        _finish_with(m, EMPTY)
        _, model3 = st.run(
            [
                _call("r", "read", {"path": INDEX}),
                _open("e1"),
            ],
            "p3",
            eids=("e1",),
            load=lambda e: ep,
        )
    assert did not in model3.outputs["r"] and "(no open entries)" in model3.outputs["r"]

    # 5. isolation: the actor's served checkout holds no pool, and the pool is on no served or main history
    served = publish(st.mem)
    export_checkout(st.mem.git_dir, served, tmp_path / "co")
    assert not (tmp_path / "co" / "pool").exists()
    assert not any(
        p.startswith("pool/")
        for p in st.mem.run("ls-tree", "-r", "--name-only", served).split()
    )
    for ref in ("main", "served"):
        r = subprocess.run(
            [
                "git",
                "--git-dir",
                str(st.mem.git_dir),
                "merge-base",
                "--is-ancestor",
                "refs/heads/pool",
                ref,
            ],
        )
        assert r.returncode == 1


# --- 2. accumulation ------------------------------------------------------------------------------------------


# --- 3. status only from the writer ---------------------------------------------------------------------------


def test_an_entry_stays_open_through_passes_that_do_not_name_it(tmp_path, monkeypatch):
    st = _Stores(tmp_path)
    _finish_with(monkeypatch, _defer("e2"))
    st.run([], "p1")
    _finish_with(monkeypatch, EMPTY)
    for k in range(2, 5):
        out, model = st.run(
            [
                _call("r", "read", {"path": INDEX}),
                _open("e2"),
            ],
            f"p{k}",
        )
        assert out.finished and out.pool["changed"] == [] and out.pool["commit"] is None
        assert pool.entry_id("deferred", "e2") in model.outputs["r"]
    assert [e.status for e in st.pool().entries.values()] == ["open"]


def test_retired_needs_a_reason_and_applies_once_finish_is_accepted(
    tmp_path,
    monkeypatch,
):
    st = _Stores(tmp_path)
    did = pool.entry_id("deferred", "e2")
    _finish_with(monkeypatch, _defer("e2"))
    st.run([], "p1")
    _finish_with(monkeypatch, {**EMPTY, "pool": {"retired": {did: " "}}})
    out, model = st.run(
        [
            _open("e2"),
            _call("f", "finish", {"summary": "s"}),
        ],
        "p2",
    )
    assert model.outputs["f"] == wl.FINISH_CELL + f"not finished: G1: pool.retired: {did} needs a reason"
    assert st.pool().entries[did].status == "open"
    _finish_with(
        monkeypatch,
        {**EMPTY, "pool": {"retired": {did: "the request never needed it"}}},
    )
    out, _ = st.run([_open("e2")], "p3")
    e = st.pool().entries[did]
    assert out.finished and out.pool["changed"] == [did]
    assert e.status == "retired" and e.status_detail == {
        "reason": "the request never needed it",
        "pass": "p3",
    }


def test_merged_applies_only_to_an_item_the_gate_landed(tmp_path):
    p = pool.Pool(Repo.init_bare(tmp_path / "mem.git")).load()
    did = p.upsert(pool.deferred("e1", "r"), "p1").id
    man = {**MAN, "pool": {"merged": {did: ITEM["item"]}}}
    assert pool.block_problems(man, p) == []
    assert (
        pool.apply_block(p, man, "p2", landed=set(), finished=True) == []
    )  # the gate refused the item
    assert p.entries[did].status == "open"
    assert pool.block_problems(
        {**EMPTY, "pool": {"merged": {did: ITEM["item"]}}},
        p,
    ) == [
        f"pool.merged: {did} must name an item this manifest lists",
    ]
    assert pool.block_problems(
        man,
        p,
        "curate",
    )  # CURATE supersedes or retires; it does not merge
    assert pool.block_problems({**EMPTY, "pool": {"retired": {"nope": "r"}}}, p) == [
        "pool.retired: nope is not an open entry",
    ]
    assert pool.apply_block(p, man, "p3", landed={ITEM["item"]}, finished=True) == [did]
    assert pool.block_problems(man, p) == [
        f"pool.merged: {did} is not an open entry",
    ]  # closed stays closed


def test_superseded_by_an_entry_needs_an_accepted_finish(tmp_path):
    p = pool.Pool(Repo.init_bare(tmp_path / "mem.git")).load()
    a = p.upsert(pool.deferred("e1", "r"), "p1").id
    b = p.upsert(pool.deferred("e2", "r"), "p1").id
    man = {**EMPTY, "pool": {"superseded": {a: b}}}
    assert pool.apply_block(p, man, "p2", landed=set(), finished=False) == []
    assert pool.apply_block(p, man, "p2", landed=set(), finished=True) == [a]
    assert (
        p.entries[a].status_detail == {"by": b, "pass": "p2"}
        and p.entries[b].status == "open"
    )


# --- 4. dedup -------------------------------------------------------------------------------------------------


def test_the_same_staged_candidate_from_two_episodes_is_one_entry(tmp_path):
    repo = Repo.init_bare(tmp_path / "mem.git")
    files = {
        "candidates/f.py": b"def f(x):\n    return x\n",
        "cases/f.json": b"[[1, 1]]",
    }
    p = pool.Pool(repo).load()
    for eid, src in (("e1", "fork"), ("e2", "replay_fork")):
        for e in pool.from_staging(eid, StagingView("ok", src, dict(files))):
            p.upsert(e, "p1")
    p.save("p1")
    p = pool.Pool(repo).load()
    (e,) = p.entries.values()
    assert e.kind == "candidate" and e.files == {
        "candidate.py": "def f(x):\n    return x\n",
        "cases.json": "[[1, 1]]",
    }
    assert e.provenance["episodes"] == ["e1", "e2"] and e.provenance["sources"] == [
        "fork",
        "replay_fork",
    ]


# --- 6. determinism -------------------------------------------------------------------------------------------


def test_the_index_is_the_same_bytes_for_the_same_pool_and_long_indexes_page(
    tmp_path,
    monkeypatch,
):
    entries = [pool.deferred(f"e{i:03d}", "a rule " * 20) for i in range(120)]
    one, two = pool.Pool(None), pool.Pool(None)
    for e in entries:
        one.upsert(pool.deferred(e.provenance["episodes"][0], e.data["reason"]), "p1")
    for e in reversed(entries):
        two.upsert(pool.deferred(e.provenance["episodes"][0], e.data["reason"]), "p1")
    assert one.stage(tmp_path / "a") == two.stage(tmp_path / "b") == 120
    text = (tmp_path / "a" / "INDEX.md").read_bytes()
    assert text == (tmp_path / "b" / "INDEX.md").read_bytes() and len(text) > 8000

    st = _Stores(tmp_path / "s")
    p = st.pool()
    for e in entries:
        p.upsert(pool.deferred(e.provenance["episodes"][0], e.data["reason"]), "p0")
    p.save("p0")
    _finish_with(monkeypatch, EMPTY)
    _, model = st.run(
        [
            _call("r", "read", {"path": INDEX}),
            _open("e2"),
        ],
        "p1",
    )
    assert (
        "[… shown bytes 0–" in model.outputs["r"]
        and "; next: offset=" in model.outputs["r"]
    )


# --- 7. the off path ------------------------------------------------------------------------------------------


# --- a crafted pool commit never writes outside /inputs/pool -----------------------------------------------------


class _Crafted:
    """A memory repo whose pool commit holds the given (mode, type, path) -> content records."""

    def __init__(self, records):
        self.records = records

    def run(self, *args, input=None):
        if args[0] == "rev-parse":
            return "c" * 40
        assert args[:3] == ("ls-tree", "-r", "-z"), args
        return "".join(f"{m} {t} {'0' * 40}\t{path}\0" for (m, t, path) in self.records)

    def show(self, rev, path):
        return next(c for (m, t, p), c in self.records.items() if p == path)


def _meta(eid, kind):
    return json.dumps(
        {"id": eid, "kind": kind, "title": "t", "status": "open"},
    ).encode()


def test_a_crafted_pool_commit_never_writes_outside_inputs_pool(tmp_path):
    good, trav, link, other = (
        pool.entry_id("deferred", e) for e in ("e1", "e2", "e3", "e4")
    )
    records = {
        ("100644", "blob", f"pool/deferred/{good}/entry.json"): _meta(good, "deferred"),
        ("100644", "blob", f"pool/deferred/{trav}/entry.json"): _meta(trav, "deferred"),
        ("100644", "blob", f"pool/deferred/{trav}/../../../x"): b"escaped",
        ("100644", "blob", f"pool/candidate/{link}/entry.json"): _meta(
            link,
            "candidate",
        ),
        ("120000", "blob", f"pool/candidate/{link}/candidate.py"): b"/etc/passwd",
        ("100644", "blob", "pool/lesson/../entry.json"): _meta("..", "lesson"),
        ("100644", "blob", "pool/lesson/NOT-AN-ID/entry.json"): _meta(
            "NOT-AN-ID",
            "lesson",
        ),
        ("100644", "blob", f"pool/lesson/{other}/entry.json"): _meta(
            good,
            "lesson",
        ),  # names another entry
        ("100644", "blob", f"pool/deferred/{other}/a\\..\\y"): b"escaped",
        ("100644", "blob", "pool/x.json"): b"{}",
    }
    p = pool.Pool(_Crafted(records)).load()
    assert set(p.entries) == {good} and len(p.skipped) >= 7
    p.entries["../../z"] = pool.Entry(id="../../z", kind="lesson", title="t")
    p.entries[good].files = {
        "../../w": "escaped",
        "/abs": "escaped",
        "ok/f.txt": "fine",
    }
    dest = tmp_path / "inputs" / "pool"
    assert p.stage(dest) == 1
    written = {str(f.relative_to(tmp_path)) for f in tmp_path.rglob("*") if f.is_file()}
    assert written == {
        "inputs/pool/INDEX.md",
        f"inputs/pool/{pool.RELATED_FILE}",
        f"inputs/pool/{good}/entry.json",
        f"inputs/pool/{good}/ok/f.txt",
    }


def test_the_pool_is_staged_only_into_a_fresh_directory(tmp_path):
    p = pool.Pool(None)
    (tmp_path / "elsewhere").mkdir()
    os.symlink(tmp_path / "elsewhere", tmp_path / "link")
    with pytest.raises(FileExistsError):
        p.stage(tmp_path / "link")
    with pytest.raises(FileExistsError):
        p.stage(tmp_path / "elsewhere")
    assert list((tmp_path / "elsewhere").iterdir()) == []


def test_a_block_decides_each_entry_once_and_keeps_a_replacement_open(tmp_path):
    p = pool.Pool(Repo.init_bare(tmp_path / "mem.git")).load()
    a = p.upsert(pool.deferred("e1", "r"), "p1").id
    b = p.upsert(pool.deferred("e2", "r"), "p1").id
    both = {**EMPTY, "pool": {"superseded": {a: b}, "retired": {b: "gone"}}}
    assert pool.block_problems(both, p) == [
        f"pool.superseded: {a}'s replacement {b} is closed by this block too",
    ]
    twice = {**EMPTY, "pool": {"superseded": {a: b}, "retired": {a: "gone"}}}
    assert pool.block_problems(twice, p) == [
        f"pool: {a} is named in both superseded and retired",
    ]
    assert pool.block_problems({**EMPTY, "pool": {"superseded": {a: a}}}, p) == [
        f"pool.superseded: {a} cannot supersede itself",
    ]
