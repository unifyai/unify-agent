"""The reading rule is gone: finish never waits on reading, a batch episode the writer never opened is recorded as
unread (in the read log and, with the pool, as an ``unread`` entry), and nothing reads those measures back.
"""

from unify.memory import pool
from tests.memory.test_pool import EMPTY, _defer, _finish_with, _open, _Stores
from tests.memory.test_writer_pass import _call, needs_bwrap

FINISH = [_call("f", "finish", {"summary": "s"})]


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
