"""Symbolic: the record file is append-only, fsynced, scrubbed and reloadable."""

import json
import os

from unify.agents.entry import Entry
from unify.agents.log import RecordLog


def _e(seq, text="x", author="root"):
    return Entry(seq, f"2026-10-06T11:00:0{seq % 10}.000Z", author, "post", text, ())


def test_append_writes_one_line_per_entry_and_load_reads_them_back(tmp_path):
    log = RecordLog(tmp_path / "records" / "run.jsonl", create=True)
    log.append(_e(1))
    log.append(_e(2, "second"))
    lines = log.path.read_text().splitlines()
    assert [json.loads(line)["seq"] for line in lines] == [1, 2]
    entries, cursors = log.load()
    assert entries == [_e(1), _e(2, "second")] and cursors == {}


def test_cursors_go_to_their_own_file_and_the_highest_wins(tmp_path):
    log = RecordLog(tmp_path / "run.jsonl", create=True)
    log.append_cursor("root", 3)
    log.append_cursor("root", 7)
    log.append_cursor("h1", 2)
    assert log.cursor_path.name == "run.cursors.jsonl"
    assert log.load()[1] == {"root": 7, "h1": 2}


def test_a_cut_last_line_is_skipped_and_numbering_continues(tmp_path):
    log = RecordLog(tmp_path / "run.jsonl", create=True)
    log.append(_e(1))
    with open(log.path, "a") as f:
        f.write('{"seq": 2, "ts": "2026-10-06T11:0')  # a crash mid-write
    entries, _ = log.load()
    assert [e.seq for e in entries] == [1]


def test_secret_values_never_reach_the_file(tmp_path, monkeypatch):
    secret = "sk-test-0123456789abcdef"  # pragma: allowlist secret
    monkeypatch.setenv("FAKE_SERVICE_API_KEY", secret)
    log = RecordLog(tmp_path / "run.jsonl", create=True)
    log.append(_e(1, f"the key is {secret}"))
    assert secret not in log.path.read_text()


def test_the_file_is_private(tmp_path):
    log = RecordLog(tmp_path / "run.jsonl", create=True)
    log.append(_e(1))
    assert oct(os.stat(log.path).st_mode & 0o777) == "0o600"
