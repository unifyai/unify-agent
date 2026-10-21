"""Rev 2: a landed procedure is listed by its structure; nothing landed runs to decide its listing or to dedupe it.

Before rev 2 a landing ran the landed reader, ``applies``, procedure and ``holds`` on the store's other recorded
requests to decide whether ``memory.match`` and the librarian listed it, and ran each reader the library held to find
one to reuse. Both were code run to classify or dedupe. Now the landing records ``recognition`` as ``{"ran": false,
"why"}``, every landed procedure is listed, and a reader is reused only when it is the same code (its shape, from its
syntax). No model, network or key.
"""

from __future__ import annotations

import inspect
import json
import os
import subprocess

import pytest

from tests.memory.test_land_capture import ROWS, _cid, _no_blob, _store
from tests.memory.test_capture_end_to_end import PROCEDURE, recorded_episode
from tests.memory.test_tool_channel_box import needs_box
from unify.memory import captures
from unify.memory import land_capture as lc
from unify.memory import recognition
from unify.memory.sandbox_run import run_confined


def test_nothing_in_the_listing_or_the_landing_can_run_code_to_list():
    src = inspect.getsource(recognition)
    for word in ("run_confined", "subprocess", "sandbox_run", "exec(", "eval("):
        assert word not in src, word
    landing = inspect.getsource(lc)
    assert "recognise_landing" not in landing and "_recognition(" not in landing
    assert '_run_reader(cand' not in landing  # no reader runs to be reused


READER_A = '''def read_rows(text):
    """The rows."""
    out = []
    for line in text.splitlines():
        out.append([int(x) for x in line.split()])
    return out
'''
CAPTURE_SAME = '''def read_input(request):
    # the same reading, other names and prose
    rows = []
    for row in request.splitlines():
        rows.append([int(v) for v in row.split()])
    return rows


def run(value):
    return [sum(r) for r in value]
'''


def test_a_reader_is_reused_only_when_it_is_the_same_code_and_nothing_runs(tmp_path, monkeypatch):
    def refuse(*a, **k):
        raise AssertionError("a process was started")

    monkeypatch.setattr(subprocess, "Popen", refuse)
    monkeypatch.setattr(os, "fork", refuse)
    pkg = tmp_path / "memory" / "text"
    pkg.mkdir(parents=True)
    (pkg / "readers.py").write_text(READER_A)
    assert lc._same_reader(tmp_path, CAPTURE_SAME) == "memory.text.readers:read_rows"
    # a reading that gives the same value by other code is not the same code
    other = CAPTURE_SAME.replace("[int(v) for v in row.split()]", "list(map(int, row.split()))")
    assert lc._same_reader(tmp_path, other) == ""
    # and a literal that differs is other code
    assert lc._same_reader(tmp_path, CAPTURE_SAME.replace("row.split()", "row.split(',')")) == ""


@needs_box
@pytest.mark.timeout(600)
def test_a_landing_runs_nothing_to_list_or_reuse_and_records_the_absence(tmp_path):
    calls: list = []

    def spy(argv, **kw):
        calls.append(json.dumps({k: kw.get(k) for k in ("ro", "rw", "cwd")}, default=str))
        return run_confined(argv, **kw)

    ep = recorded_episode("e7", rows=ROWS)
    cid = _store(tmp_path / "store")
    box = tmp_path / "box"
    box.mkdir()
    reply = lc.land({"package": "readings", "name": "row_sums", "capture_id": cid}, store=tmp_path / "store",
                    box=box, episode={"e7": ep}.get, blob=_no_blob, batch=["e7"], runner=spy)  # fmt: skip
    assert reply.status == "ok", reply.text
    assert recognition.LISTED_BY_STRUCTURE in reply.text
    assert not any("/recognition" in c or "/reader-" in c for c in calls), calls
    landing = captures.landings(tmp_path / "store")[cid]
    assert landing["recognition"] == recognition.NOT_RUN


def test_the_capture_ids_of_this_file_match_the_store():
    assert _cid(PROCEDURE) == captures.capture_id(PROCEDURE)
