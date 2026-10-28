"""The reading rule is gone: finish never waits on reading, a batch episode the writer never opened is recorded as
unread in the read log only (the pool has no ``unread`` kind), and nothing reads that measure back.
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
    assert uid not in st.pool().entries and "unread" not in pool.KINDS
    assert out.coverage["unread"] == ["e2"] and "unread_open" not in out.pool
    with monkeypatch.context() as m:
        _finish_with(m, EMPTY)
        st2 = _Stores(tmp_path / "r")
        out2, _ = st2.run([_open("e2"), *FINISH], "p1")
    assert uid not in st2.pool().entries and out2.coverage["unread"] == []


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
