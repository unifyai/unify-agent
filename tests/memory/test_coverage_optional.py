"""The reading rule is gone: finish never waits on reading, a batch episode the writer never opened is recorded as
unread (in the read log and, with the pool, as an ``unread`` entry), and nothing reads those measures back.
"""

from unify.memory import pool
from unify.memory.writer_pass import writer_tools
from tests.memory.test_pool import EMPTY, _defer, _finish_with, _open, _Stores
from tests.memory.test_writer_pass import _call, needs_bwrap
from tests.memory.test_writer_library_tools import _pass

FINISH = [_call("f", "finish", {"summary": "s"})]


def test_there_is_no_dismiss_and_finish_is_accepted_without_reading(
    tmp_path,
    monkeypatch,
):
    names = {t["function"]["name"] for t in writer_tools(library_mode=True)}
    assert "dismiss" not in names and "read_episode" in names
    _finish_with(monkeypatch, EMPTY)
    out, model = _pass(tmp_path, FINISH, eids=("e2",), max_calls=3)
    # accepted at its first call: the pass ended there, so no refusal was ever fed back to the model
    assert out.finished and "f" not in model.outputs
    assert out.coverage["unread"] == ["e2"] and out.coverage["opened"] == []
    assert out.coverage_measures is None


@needs_bwrap
def test_no_unread_entry_enters_the_pool(tmp_path, monkeypatch):
    uid = pool.entry_id("unread", "e2")
    with monkeypatch.context() as m:
        _finish_with(m, EMPTY)
        st = _Stores(tmp_path / "o")
        out, _ = st.run(FINISH, "p1")
    entry = st.pool().entries[uid]
    assert entry.kind == "unread" and entry.status == "open"
    assert entry.provenance["episodes"] == ["e2"] and "coverage" not in entry.title
    assert out.pool["unread_open"] == 1
    with monkeypatch.context() as m:
        _finish_with(m, EMPTY)
        st2 = _Stores(tmp_path / "r")
        out2, _ = st2.run([_open("e2"), *FINISH], "p1")
    assert uid not in st2.pool().entries and out2.pool["unread_open"] == 0


@needs_bwrap
def test_a_deferred_episode_is_not_unread(tmp_path, monkeypatch):
    with monkeypatch.context() as m:
        _finish_with(m, _defer("e2", "nothing here yet"))
        st = _Stores(tmp_path)
        out, _ = st.run(FINISH, "p1")
    assert out.coverage["unread"] == ["e2"]  # the read log: never opened
    entries = st.pool().entries
    assert pool.entry_id("unread", "e2") not in entries
    assert entries[pool.entry_id("deferred", "e2")].status == "open"
