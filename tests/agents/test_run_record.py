"""The run record (UNIFY_AGENTS_RUN_RECORD): one record for the whole run, shared across processes.

Numbering under a file lock, crash-cut lines, waiting across processes, the roster and its addresses, the delivery
log, the storm notice, the byte backstop and scrubbing (the redesign's A1 tests 1-4 and 9, and the failure modes they
rule out).
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

from unify.agents import run_record as rr
from unify.agents.options import Options
from unify.agents.record import PostRefused

FAST = Options(
    max_live=16,
    max_total=64,
    max_entries=None,
    max_record_bytes=rr.MAX_RECORD_BYTES,
    min_post_interval_s=0.0,
)
REPO = Path(__file__).resolve().parents[2]


def _record(home, **kw):
    return rr.join_run_record(home, options=kw.pop("options", FAST))


def _child(code: str, *args: str, env: dict | None = None) -> subprocess.Popen:
    """A Python process importing this checkout's unify (the run record's other process)."""
    return subprocess.Popen(
        [sys.executable, "-c", textwrap.dedent(code), *args],
        cwd=str(REPO),
        env={**os.environ, **(env or {})},
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


APPENDER = """
    import sys
    from unify.agents import run_record as rr
    from unify.agents.options import Options
    home, name, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
    rec = rr.join_run_record(home, options=Options(max_entries=None, min_post_interval_s=0.0))
    rec.join(name, "helper")
    for i in range(n):
        rec.append(name, f"{name} {i}", mentions=[])
    print("done")
"""


def test_two_processes_appending_at_once_number_every_entry_once_with_no_gap(tmp_path):
    """A1 test 1: 1,000 appends from each of two processes; seq strictly increasing, no gap, no duplicate."""
    home = tmp_path / "home"
    _record(home)  # made once; both processes reopen it
    procs = [_child(APPENDER, str(home), name, "1000") for name in ("pa", "pb")]
    for p in procs:
        out, err = p.communicate(timeout=300)
        assert p.returncode == 0, err
        assert out.strip() == "done"
    rows = rr.read_entries(home)
    assert [r["seq"] for r in rows] == list(range(1, 2001))
    assert sorted(r["author"] for r in rows).count("pa") == 1000
    # every line of the file is one whole entry: no interleaved halves
    lines = rr.record_path(home).read_text().splitlines()
    assert len(lines) == 2000 and all(json.loads(line)["v"] == 1 for line in lines)


def test_a_line_cut_by_a_crash_is_skipped_and_numbering_continues_from_the_last_whole_entry(
    tmp_path,
):
    """A1 test 2."""
    home = tmp_path / "home"
    rec = _record(home)
    rec.join("a", "helper")
    rec.append("a", "one", mentions=[])
    rec.append("a", "two", mentions=[])
    with open(rr.record_path(home), "a") as fh:
        fh.write('{"v": 1, "seq": 3, "ts": "2026-10-20T21:0')  # the crash
    again = _record(home)
    assert [e.seq for e in again.entries] == [1, 2]
    again.participants["a"] = rec.participants["a"]
    entry, _ = again.append("a", "three", mentions=[])
    assert entry.seq == 3
    assert [r["seq"] for r in rr.read_entries(home)] == [1, 2, 3]
    # the cut line stays in the file as one unreadable line of its own
    assert len(rr.record_path(home).read_text().splitlines()) == 4


def test_a_whole_entry_left_without_its_newline_is_numbered_after(tmp_path):
    """A crash between an entry's last brace and its newline: the next appender closes the line first, so the entry
    keeps its number and the new one follows it (no duplicate number)."""
    home = tmp_path / "home"
    rec = _record(home)
    rec.join("a", "helper")
    rec.append("a", "one", mentions=[])
    row = dict(rr.read_entries(home)[0], seq=2, text="two")
    with open(rr.record_path(home), "a") as fh:
        fh.write(json.dumps(row))
    entry, _ = rec.append("a", "three", mentions=[])
    assert entry.seq == 3
    assert [r["seq"] for r in rr.read_entries(home)] == [1, 2, 3]


WAITER = """
    import asyncio, sys, time
    from unify.agents import run_record as rr
    home = sys.argv[1]
    rec = rr.join_run_record(home)
    rec.join("bee", "helper")
    print("ready", flush=True)
    found, why = asyncio.run(rec.wait_outcome("bee", timeout=30))
    print(f"{time.time()!r} {why} {found[0].text if found else ''}", flush=True)
"""


def test_a_wait_in_one_process_returns_within_two_seconds_of_a_mention_from_another(
    tmp_path,
):
    """A1 test 3."""
    home = tmp_path / "home"
    rec = _record(home)
    rec.join("ann", "helper")
    rec.join(
        "cat", "helper"
    )  # running: the waiter has someone who could answer, so it waits
    child = _child(WAITER, str(home))
    assert child.stdout.readline().strip() == "ready"
    time.sleep(1.0)
    posted = time.time()
    rec.append("ann", "@bee are you there?")
    out, err = child.communicate(timeout=30)
    assert child.returncode == 0, err
    at, why, text = out.strip().split(" ", 2)
    assert why == "message" and text == "@bee are you there?"
    assert float(at) - posted < 2.0


def test_wait_returns_at_once_when_everyone_who_could_answer_is_idle_waiting_or_ended(
    tmp_path,
):
    """A1 test 4."""
    home = tmp_path / "home"
    rec = _record(home)
    rec.join("me", "actor")
    rec.join("writer", "writer", state="idle")
    rec.join("h1", "helper")
    rec.leave("h1")
    started = time.monotonic()
    found, why = asyncio.run(rec.wait_outcome("me", timeout=30))
    assert found == [] and why == "nothing_can_arrive"
    assert time.monotonic() - started < 2.0


def test_a_waiting_agent_with_something_unread_could_still_answer(tmp_path):
    """Two agents waiting: one has an unread message (it will wake and may answer), so the other keeps waiting until
    its own message arrives or its timeout."""
    home = tmp_path / "home"
    rec = _record(home)
    rec.join("me", "helper")
    rec.join("you", "helper")
    rec.set_state("you", "waiting")
    rec.append("me", "@you what did you find?")
    assert rec.has_unseen("you")
    assert not rec.nobody_can_answer("me")
    found, why = asyncio.run(rec.wait_outcome("me", timeout=0.5))
    assert found == [] and why == "timeout"


def test_a_process_that_died_leaves_no_running_agent_to_wait_on(tmp_path):
    """An agent whose process is gone is ended on the roster, whatever its row says."""
    home = tmp_path / "home"
    rec = _record(home)
    rec.join("me", "helper")
    gone = subprocess.Popen([sys.executable, "-c", "pass"])
    gone.wait()
    rec.join("ghost", "helper", pid=gone.pid)
    assert {r["name"]: r["state"] for r in rec.roster_rows(include_ended=True)}[
        "ghost"
    ] == "ended"
    assert rec.nobody_can_answer("me")


def test_names_are_never_given_twice_in_a_run(tmp_path):
    home = tmp_path / "home"
    rec = _record(home)
    rec.join("h1", "helper")
    with pytest.raises(PostRefused, match="taken"):
        rec.join("h1", "helper")
    rec.leave("h1")
    with pytest.raises(PostRefused, match="taken"):
        _record(home).join("h1", "helper")
    assert rec.next_name("h") == "h2"


def test_the_actor_is_named_from_its_episode_with_more_digits_only_on_a_clash(tmp_path):
    home = tmp_path / "home"
    rec = _record(home)
    first = rec.unique_name("actor", "20261020T210311-3f9a0c1d")
    assert first == "actor-3f9a"
    rec.join(first, "actor")
    assert rec.unique_name("actor", "20261020T210500-3f9a77aa") == "actor-3f9a77"


def test_role_addresses_resolve_to_the_running_agents_of_that_role(tmp_path):
    home = tmp_path / "home"
    rec = _record(home)
    rec.join("actor-aaaa", "actor", episode="ep-a")
    rec.join("actor-bbbb", "actor", episode="ep-b")
    rec.join("librarian-aaaa", "librarian", episode="ep-a")
    rec.join("librarian-bbbb", "librarian", episode="ep-b")
    rec.join("fork-aaaa", "fork", episode="ep-a")
    rec.join("writer", "writer", state="idle")
    rec.join("h1", "helper")
    rec.join("h2", "helper")
    rec.leave("h2")
    to, unresolved, ended = rec.resolve_for(
        "actor-aaaa", "@writer @librarian @fork @helpers @actor @nobody"
    )
    # rev 2: @librarian names the role, which this run has not seeded: no librarian instance is addressed by it
    assert to == ["writer", "fork-aaaa", "h1", "actor-bbbb"]
    assert unresolved == ["librarian", "nobody"] and ended == []
    # once the run seeds the role, @librarian is the role itself (a question for its next instance)
    rr.seed_role(home, "librarian", by="test")
    to, unresolved, ended = rec.resolve_for("actor-aaaa", "@writer @librarian @fork")
    assert to == ["writer", "librarian", "fork-aaaa"] and unresolved == [] and ended == []
    # a named agent that ended is resolved and reported: nobody reads for it now
    to, unresolved, ended = rec.resolve_for("h1", "@h2 done?")
    assert to == ["h2"] and ended == ["h2"]


def test_a_message_to_the_writer_while_its_service_is_idle_is_kept_for_it(tmp_path):
    home = tmp_path / "home"
    rec = _record(home)
    rec.join("actor-aaaa", "actor", episode="ep-a")
    rec.join("writer", "writer", state="idle")
    view = rr.RunRecordView(rec, "actor-aaaa", "actor")
    out = view.post(
        "actor-aaaa", "@writer the served item gave a wrong sum here: action:7"
    )
    assert out["to"] == ["writer"]
    row = rr.read_entries(home)[-1]
    assert (
        row["mentions"] == ["writer"]
        and row["role"] == "actor"
        and row["episode"] == "ep-a"
    )
    assert row["refs"] == ["ep-a/action:7"]


def test_a_message_that_addresses_no_one_is_refused_and_a_helpers_goes_to_its_starter(
    tmp_path,
):
    from unify.common.tool_registry import RecordRefused

    home = tmp_path / "home"
    rec = _record(home)
    rec.join("writer", "writer")
    rec.join("h1", "helper", spawner="writer")
    with pytest.raises(RecordRefused) as got:
        rr.RunRecordView(rec, "writer", "writer").post("writer", "nobody named")
    assert got.value.form == "no_recipient"
    out = rr.RunRecordView(rec, "h1", "helper", spawner="writer").post(
        "h1", "found it", default_to="writer"
    )
    assert out["to"] == ["writer"]


def test_delivery_is_logged_per_way_and_a_block_moves_the_cursor(tmp_path):
    home = tmp_path / "home"
    rec = _record(home)
    rec.join("a", "helper", pass_id="p" * 16)
    rec.join("b", "helper")
    rec.append("b", "@a first")
    block = rec.take_block("a")
    assert "#1 b: @a first" in block
    rec.append("b", "@a second")
    got = rec.take_unseen("a", "messages")
    assert [e.text for e in got] == ["@a second"]
    rows = rr.read_deliveries(home)
    assert [(r["agent"], r["via"], r["seqs"]) for r in rows if r["seqs"]] == [
        ("a", "block", [1]),
        ("a", "messages", [2]),
    ]
    assert all(r["pass_id"] == "p" * 16 for r in rows if r["agent"] == "a")
    # the next block shows the entry returned through messages as already returned, not again in full
    rec.append("b", "@a third")
    block = rec.take_block("a")
    assert "#2 b — already returned to your code" in block and "#3 b: @a third" in block


def test_messages_all_pages_through_every_entry(tmp_path):
    home = tmp_path / "home"
    rec = _record(home)
    rec.join("a", "helper")
    rec.join("b", "helper")
    for i in range(60):
        rec.append("b", f"note {i}", mentions=[])
    view = rr.RunRecordView(rec, "a", "helper")
    page, more = view.messages_all()
    assert len(page) == 50 and more == 10 and page[0]["text"] == "note 0"
    page, more = view.messages_all()
    assert len(page) == 10 and more == 0


def test_a_pair_that_only_messages_each_other_with_no_evidence_gets_one_notice(
    tmp_path,
):
    """The storm guard (§10.3): 20 entries of a pair only to each other with no tool call or file read in between."""
    home = tmp_path / "home"
    rec = _record(home)
    rec.join("x", "helper")
    rec.join("y", "helper")
    for i in range(rr.STORM_RUN - 1):
        rec.append("x" if i % 2 == 0 else "y", "@y again" if i % 2 == 0 else "@x again")
    assert not any(e.author == "harness" for e in rec.entries)
    rec.append("y", "@x again")
    notices = [e for e in rec.entries if e.author == "harness"]
    assert (
        len(notices) == 1
        and set(notices[0].mentions) == {"x", "y"}
        and notices[0].kind == "system"
    )
    assert rr.STORM_MARK in notices[0].text and "20 times" in notices[0].text
    # one notice per run: the next entry does not repeat it
    rec.append("x", "@y again")
    assert len([e for e in rec.entries if e.author == "harness"]) == 1


def test_evidence_read_in_between_means_no_storm_notice(tmp_path):
    home = tmp_path / "home"
    rec = _record(home)
    rec.join("x", "helper")
    rec.join("y", "helper")
    for i in range(rr.STORM_RUN + 4):
        who = "x" if i % 2 == 0 else "y"
        if i == 10:
            rec.note_act(who)  # a tool call or a file read
        rec.append(who, "@y again" if who == "x" else "@x again")
    assert not any(e.author == "harness" for e in rec.entries)


def test_past_the_byte_backstop_posts_are_refused_and_the_harness_is_cut_short(
    tmp_path,
):
    home = tmp_path / "home"
    small = Options(max_entries=None, max_record_bytes=4000, min_post_interval_s=0.0)
    rec = rr.join_run_record(home, options=small)
    rec.join("a", "helper")
    rec.join("b", "helper")
    rec.append("a", "@b " + "x" * 1000)
    with pytest.raises(PostRefused, match="size backstop"):
        rec.append("a", "@b " + "y" * 2000)
    entry = rec.append_harness(
        "harness",
        "@b " + "z" * 9000,
        kind="system",
        mentions=["b"],
        full_text_at="the log",
    )
    assert (
        len(entry.text.encode()) <= rr.CUT_TEXT_BYTES and "size backstop" in entry.text
    )


def test_a_credential_shaped_string_never_reaches_the_file(tmp_path, monkeypatch):
    """A1 test 9: scrubbed on append, by value and by shape."""
    secret = "sk-or-v1-" + "ab" * 32  # pragma: allowlist secret
    held = "tok-held-by-this-process-1234"  # pragma: allowlist secret
    monkeypatch.setenv("FAKE_SERVICE_API_KEY", held)
    home = tmp_path / "home"
    rec = _record(home)
    rec.join("a", "helper")
    rec.join("b", "helper")
    rec.append(
        "a", f"@b the key is {secret} and {held} and Authorization: Bearer abc.def.ghi"
    )
    text = rr.record_path(home).read_text()
    assert secret not in text and held not in text and "abc.def.ghi" not in text
    assert "redacted" in text.lower()


def test_the_record_files_are_private_and_never_under_a_bound_folder(tmp_path):
    home = tmp_path / "home"
    rec = _record(home)
    rec.join("a", "helper")
    rec.append("a", "hello", mentions=[])
    assert oct(os.stat(rr.record_path(home)).st_mode & 0o777) == "0o600"
    assert oct(os.stat(rr.agents_dir(home)).st_mode & 0o777) == "0o700"
    # under the memory state directory, which no box binds
    assert rr.agents_dir(home).parent == home / "memory-state"


def test_create_refuses_a_record_that_exists(tmp_path):
    """Two cells never share one record: a new run's record is created exclusively."""
    home = tmp_path / "home"
    rr.open_run_record(home, create=True)
    with pytest.raises(FileExistsError):
        rr.open_run_record(home, create=True)
    with pytest.raises(FileNotFoundError):
        rr.open_run_record(tmp_path / "other", create=False)


def test_rows_carry_the_schema_fields(tmp_path):
    home = tmp_path / "home"
    rec = _record(home)
    rec.join("actor-3f9a", "actor", episode="3f9a0c1d2e4b5a69")
    rec.join("writer", "writer", state="idle")
    rec.append(
        "actor-3f9a",
        "@writer the served item notes/invoices/totals gave a wrong sum here: action:7",
    )
    row = rr.read_entries(home)[0]
    assert set(row) >= {"v", "seq", "ts", "author", "kind", "text", "mentions", "role", "episode", "pass_id",
                        "reply_to", "refs"}  # fmt: skip
    assert row["ts"].endswith("Z") and row["kind"] == "post" and row["reply_to"] is None
    roster = rr.read_roster(home)
    actor = next(a for a in roster["agents"] if a["name"] == "actor-3f9a")
    assert actor["can_mention"] == ["*"] and actor["state"] == "running"
    assert [m["who"] for m in actor["should_mention"]] == ["writer", "librarian"]
