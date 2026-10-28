"""Requests never wait for a pass, see only their pinned commit, and learn of a landed pass once."""

from __future__ import annotations

import pytest

from unify.memory.gitio import Repo
from unify.memory.integration import async_pass as ap
from unify.memory.integration.paths import Paths
from unify.memory.integration.state import State
from unify.memory.memory_writer import land, publish, served_head


def _snapshot(root):
    return {
        str(p.relative_to(root)): p.read_bytes()
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


def test_a_request_pins_the_served_head_and_never_sees_an_unrecorded_pass(tmp_path):
    """Needs the actor export, the library fixtures and the item records: runs once they are integrated."""
    checkout = pytest.importorskip("unify.memory.integration.checkout")
    if not hasattr(
        checkout,
        "export_actor_library_mode",
    ):  # importorskip skips only a missing module
        pytest.skip("needs P3's checkout.export_actor_library_mode")
    export_actor_library_mode = checkout.export_actor_library_mode
    write_records = pytest.importorskip("unify.memory.item_records").write_records
    LIB = pytest.importorskip("tests.memory.test_layout").LIB
    _commit = pytest.importorskip("tests.memory.test_library_helper")._commit

    mem = Repo.init_bare(tmp_path / "memory")
    old = _commit(mem, LIB, "seed")
    export_actor_library_mode(
        mem.git_dir,
        served_head(mem),
        tmp_path / "a",
    )  # a request in progress
    before = _snapshot(tmp_path / "a")
    with mem.temp_checkout() as wt:  # a pass's candidate: a commit that no ref names
        (wt / "memory/text/extra.py").write_text(
            'def more(x):\n    """More."""\n    return x\n',
        )
        candidate = mem.commit_all(wt, "candidate", {})
    assert served_head(mem) == old  # a request starting now pins the old commit
    assert land(mem, candidate, old) == candidate  # one writer, one compare-and-swap
    # no records map yet: still not servable
    assert mem.head() == candidate and served_head(mem) == old
    export_actor_library_mode(mem.git_dir, served_head(mem), tmp_path / "b")
    assert _snapshot(tmp_path / "b") == before
    write_records(mem, candidate, {})  # after_passes writes the map
    # the last step: the pin-able head advances
    assert publish(mem) == candidate == served_head(mem)
    assert (
        _snapshot(tmp_path / "a") == before
    )  # the earlier request's copy is untouched
    export_actor_library_mode(mem.git_dir, served_head(mem), tmp_path / "c")
    assert "memory/text/extra.py" in _snapshot(tmp_path / "c")


def test_a_half_written_results_line_waits_for_its_end(tmp_path):
    paths = Paths.under(tmp_path)
    paths.state_dir.mkdir(parents=True)
    ap.results_path(paths).write_text(
        '{"pass_id": "e1.p0", "ended": "landed"}\n{"pass_id": "e2',
    )
    state = State.load(paths.state)
    assert [r["pass_id"] for r in ap.apply_results(state, paths)] == ["e1.p0"]
    with open(ap.results_path(paths), "a") as fh:
        fh.write('.p0", "ended": "refused"}\n')
    assert [r["pass_id"] for r in ap.apply_results(state, paths)] == ["e2.p0"]
