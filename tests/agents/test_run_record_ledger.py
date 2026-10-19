"""The run record feeds the attention ledger (the redesign's A1 test 8): a message to @writer, or a reply to the
writer's entry, makes one ``message`` entry, whether the writer service is running or idle; the writer's own entries
and messages to anyone else make none. The read is verified from the record's own delivery log.
"""

from __future__ import annotations

from unify.agents import run_record as rr
from unify.agents.options import Options
from unify.memory import attention

FAST = Options(
    max_live=16,
    max_total=64,
    max_entries=None,
    max_record_bytes=rr.MAX_RECORD_BYTES,
    min_post_interval_s=0.0,
)


def _messages(home) -> list:
    ledger = attention.Ledger(home)
    ledger.sync()
    return [e for e in ledger.entries().values() if e.kind == "message"]


def test_an_actor_message_to_the_idle_writer_makes_one_message_entry(tmp_path):
    home = tmp_path / "home"
    rec = rr.join_run_record(home, options=FAST)
    rec.join("actor-3f9a", "actor", episode="20261020T210311-3f9a0c1d")
    rec.join("writer", "writer", state="idle")
    rec.join("h1", "helper")
    view = rr.RunRecordView(rec, "actor-3f9a", "actor")
    view.post(
        "actor-3f9a",
        "@writer the served item item:notes/totals gave a wrong sum here: action:2",
    )
    view.post("actor-3f9a", "@h1 not for the writer")
    got = _messages(home)
    assert len(got) == 1 and got[0].state == "open" and got[0].refs.record_seq == 1
    assert (
        "20261020T210311-3f9a0c1d" in got[0].refs.episodes
        and "notes/totals" in got[0].refs.items
    )
    # syncing again makes no second entry for the same record entry
    assert len(_messages(home)) == 1


def test_a_reply_to_the_writers_entry_is_a_message_and_the_writers_own_entries_are_not(
    tmp_path,
):
    home = tmp_path / "home"
    rec = rr.join_run_record(home, options=FAST)
    rec.join("writer", "writer", pass_id="ab" * 8)
    rec.join("fork-3f9a", "fork", episode="ep-1")
    writer = rr.RunRecordView(rec, "writer", "writer")
    asked = writer.post("writer", "@fork-3f9a what did cell:0 do?")
    fork = rr.RunRecordView(rec, "fork-3f9a", "fork")
    fork.post("fork-3f9a", "@writer it read the first line", reply_to=asked["seq"])
    got = _messages(home)
    assert len(got) == 1 and got[0].refs.record_seq == asked["seq"] + 1


def test_the_writers_read_of_a_message_is_verified_from_the_delivery_log(tmp_path):
    """B2 verifies a ``message`` entry as read when the writer's pass was given it (a block, a wait or messages), from
    the record's delivery rows, never from what the writer says."""
    home = tmp_path / "home"
    rec = rr.join_run_record(home, options=FAST)
    rec.join("actor-3f9a", "actor", episode="ep-1")
    rec.join("writer", "writer", pass_id="cd" * 8)
    rr.RunRecordView(rec, "actor-3f9a", "actor").post(
        "actor-3f9a", "@writer a question about item:x"
    )
    ledger = attention.Ledger(home)
    ledger.sync()
    (entry,) = [e for e in ledger.entries().values() if e.kind == "message"]
    handed = ledger.hand("cd" * 8, 4)
    assert [e.entry for e in handed] == [entry.entry]
    assert rec.take_block("writer") is not None
    delivered = attention.delivered_seqs(home, "cd" * 8)
    assert entry.refs.record_seq in delivered
    assert entry.entry in ledger.verify_reads(
        "cd" * 8, attention.PassReads(), delivered
    )
