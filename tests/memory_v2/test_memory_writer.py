"""P7 Amendment A: one writer for memory main; requests pin a served head that advances only after a commit's item
records exist; a pass re-targets over status commits, and is refused as stale over code commits; archives keep
refs/notes/items.

The first tests need only git: their status commits have the shape P5's ``MemoryRepo.status_commit`` writes. The
last ones use P4's gate fixtures and P5's records and status commits, and run once those are integrated.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

from unify.memory_v2 import memory_writer as mw
from unify.memory_v2.gitio import Repo

try:  # P4's v2.1 gate fixture, visible to request.getfixturevalue once P4 is integrated
    from tests.memory_v2.test_gate_v21 import world21  # noqa: F401
except ImportError:
    pass

ITEM = "memory.text.dates:parse_date"
SEED = {
    "memory/text/__init__.py": '"""Text helpers."""\n',
    "memory/text/dates.py": 'def parse_date(s):\n    """A date."""\n    return s\n',
}
EXTRA = {"memory/text/extra.py": 'def more(x):\n    """More."""\n    return x\n'}


@pytest.fixture
def mem(tmp_path):
    repo = Repo.init_bare(tmp_path / "memory")
    base = repo.head()
    with repo.temp_checkout() as wt:
        for rel, text in SEED.items():
            (wt / rel).parent.mkdir(parents=True, exist_ok=True)
            (wt / rel).write_text(text)
        sha = repo.commit_all(wt, "seed", {})
    repo.fast_forward("main", sha, expected_old=base)
    return repo


def _candidate_on(repo, parent, files):
    with repo.temp_checkout(parent) as wt:
        for rel, text in files.items():
            (wt / rel).parent.mkdir(parents=True, exist_ok=True)
            (wt / rel).write_text(text)
        return repo.commit_all(wt, "consolidation pass p1: x", {"Pass": "p1"})


def _status(repo, changes):
    """A status commit as P5 writes it: empty, subject ``status: ``, ``Status:`` and ``Evidence:`` trailers."""
    base = repo.head()
    with repo.temp_checkout(base) as wt:
        sha = repo.commit_all(
            wt,
            f"status: {len(changes)} item(s)",
            {
                "Status": [f"{i} {s}" for i, s in sorted(changes.items())],
                "Evidence": ["e1"],
            },
            allow_empty=True,
        )
    repo.fast_forward("main", sha, expected_old=base)
    return sha


def _tree(repo, sha):
    return repo.run("rev-parse", f"{sha}^{{tree}}").strip()


def test_served_is_created_at_main_before_main_moves_and_never_moves_without_records(
    mem,
):
    base = mem.head()
    assert mw.served_head(mem) == base  # no v2.1 write yet: main itself
    c1 = mw.land(mem, _candidate_on(mem, base, EXTRA), base)
    assert mem.head() == c1 and mw.served_head(mem) == base
    assert mw.publish(mem, has_records=lambda c: False) == base
    c2 = mw.land(mem, _candidate_on(mem, c1, {"memory/text/more.py": "X = 1\n"}), c1)
    # the newest first-parent commit with its own records map, not the head
    assert mw.publish(mem, has_records=lambda c: c == c1) == c1 == mw.served_head(mem)
    assert mw.publish(mem, has_records=lambda c: c in (c1, c2)) == c2
    assert mw.served_head(mem) == c2


def test_status_only_moves_retarget_without_the_gate(mem):
    parent = mem.head()
    candidate = _candidate_on(mem, parent, EXTRA)
    status = _status(mem, {ITEM: "suspect"})
    assert mw.is_status_commit(mem, status) and mw.moved_only_by_status(
        mem,
        parent,
        status,
    )
    landed = mw.land(mem, candidate, parent)
    assert landed != candidate and mem.head() == landed
    assert mem.run("rev-parse", f"{landed}^").strip() == status
    assert _tree(mem, landed) == _tree(mem, candidate)
    assert mem.run("show", "-s", "--format=%B", landed) == mem.run(
        "show",
        "-s",
        "--format=%B",
        candidate,
    )


def test_a_status_subject_that_changes_the_tree_is_not_a_status_commit(mem):
    parent = mem.head()
    with mem.temp_checkout(parent) as wt:
        (wt / "memory/text/dates.py").write_text("X = 1\n")
        sha = mem.commit_all(wt, "status: 1 item(s)", {"Status": f"{ITEM} suspect"})
    mem.fast_forward("main", sha, expected_old=parent)
    assert not mw.is_status_commit(mem, sha)
    with pytest.raises(mw.StaleParent, match="stale"):
        mw.land(mem, _candidate_on(mem, parent, EXTRA), parent)


def test_a_code_commit_makes_the_pass_stale_and_writes_nothing(mem):
    parent = mem.head()
    candidate = _candidate_on(mem, parent, EXTRA)
    code = mw.land(
        mem,
        _candidate_on(mem, parent, {"memory/text/other.py": "Y = 2\n"}),
        parent,
    )
    assert not mw.is_status_commit(mem, code) and not mw.moved_only_by_status(
        mem,
        parent,
        code,
    )
    with pytest.raises(mw.StaleParent, match="stale"):
        mw.land(mem, candidate, parent)
    assert mem.head() == code


def test_writers_are_serialised(mem):
    holder = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import fcntl, os, sys, time\n"
            "fd = os.open(sys.argv[1], os.O_RDWR | os.O_CREAT, 0o600)\n"
            "fcntl.flock(fd, fcntl.LOCK_EX)\nprint('held', flush=True)\ntime.sleep(30)\n",
            str(mw.lock_path(mem)),
        ],
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert holder.stdout.readline().strip() == "held"
        parent = mem.head()
        with pytest.raises(TimeoutError, match="main lock"):
            mw.land(mem, _candidate_on(mem, parent, EXTRA), parent, timeout_s=0.3)
        assert mem.head() == parent
    finally:
        holder.kill()
        holder.wait()
    with mw.main_lock(mem, timeout_s=5):  # free again once its holder is gone
        pass


def test_an_archive_carries_the_notes_and_the_served_head(mem, tmp_path):
    base = mem.head()
    c1 = mw.land(mem, _candidate_on(mem, base, EXTRA), base)
    mem.add_note(c1, '{"item": "x"}', ref="items")
    assert mw.publish(mem, has_records=lambda c: bool(mem.notes(c, ref="items"))) == c1
    out = mw.archive(mem, tmp_path / "out" / "memory.bundle")
    assert {"refs/heads/main", "refs/heads/served", "refs/notes/items"} <= set(
        out["refs"],
    )
    subprocess.run(
        ["git", "clone", "-q", "--mirror", out["path"], str(tmp_path / "restored")],
        check=True,
    )
    restored = Repo(tmp_path / "restored")
    assert restored.notes(c1, ref="items") == mem.notes(c1, ref="items")
    assert restored.head("served") == c1


# --- with P5's records and status commits, and P4's v2.1 gate (run once integrated) -------------------------


def test_publish_reads_p5s_records_and_status_commits_are_p5s(mem):
    ir = pytest.importorskip("unify.memory_v2.item_records")
    from unify.memory_v2.memory_repo import MemoryRepo

    base = mem.head()
    c1 = mw.land(mem, _candidate_on(mem, base, EXTRA), base)
    assert mw.publish(mem) == base  # c1 has no records map: not servable
    ir.write_records(mem, c1, {ITEM: ir.empty_record(ITEM, "function")})
    assert mw.publish(mem) == c1
    status = MemoryRepo(mem).status_commit({ITEM: "suspect"}, ["e1"])
    assert mw.is_status_commit(mem, status) and mw.moved_only_by_status(mem, c1, status)


def _gate_v21():
    return pytest.importorskip("tests.memory_v2.test_gate_v21")


def test_the_gate_refuses_a_stale_pass_and_keeps_its_patch(request):
    """P4's v2.1 gate fixture; the candidate is made on a parent that a code commit then moves past."""
    g = _gate_v21()
    mem, ev, blobs, gate = request.getfixturevalue("world21")
    parent = mem.head()
    files, fixtures = g._files_and_fixtures(blobs)
    candidate = g._candidate(mem, files)
    mw.land(
        mem,
        _candidate_on(mem, parent, {"memory/zz/__init__.py": '"""Other."""\n'}),
        parent,
    )
    res = gate().merge(parent, candidate, g._man(fixtures), "p1", "write", None, "0.01")
    assert not res.passed and any("stale" in r for r in res.reasons)
    (patch,) = ev.db.execute(
        "SELECT patch_blob FROM passes WHERE pass_id='p1'",
    ).fetchone()
    assert patch and b"memory/acct/users.py" in blobs.get(patch)


def test_the_gate_retargets_over_a_status_commit(request):
    g = _gate_v21()
    from unify.memory_v2.memory_repo import MemoryRepo

    mem, ev, blobs, gate = request.getfixturevalue("world21")
    parent = mem.head()
    files, fixtures = g._files_and_fixtures(blobs)
    candidate = g._candidate(mem, files)
    status = MemoryRepo(mem).status_commit({ITEM: "suspect"}, ["e1"])
    res = gate().merge(parent, candidate, g._man(fixtures), "p1", "write", None, "0.01")
    assert res.passed, res.reasons
    assert mem.head() == res.merged
    assert mem.run("rev-parse", f"{res.merged}^").strip() == status
    assert _tree(mem, res.merged) == _tree(mem, candidate)
