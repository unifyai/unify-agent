"""The episode-end fork writes a short account of its episode, ``summary.md``, beside its drafts.

What is checked here, keyless and with no model call:

* the fork's instruction asks for ``summary.md`` first, in one fixed format (five headed sections, each line ending
  with a pointer into the record), and its format example parses as a staged ``summary.md``;
* the fork worker stages ``summary.md`` under the same rules and records as its other files (``fork.json``,
  ``fork-blocks.json``), and ``fork.json`` records its size and the fork's wall time;
* the fork's request carries the recorded episode's part list after the actor's own messages, so the pointers in
  its summary can name real parts;
* staging lists the summary in the writer's batch-map row;
* a fork that did not finish hands the writer nothing for its episode, says why in the row, and keeps its files on
  the host;
* the consolidation end event counts, per episode, the summary's pointers and how many of them resolve, and records
  the writer's input tokens.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from unify.memory import library_prompts, staging
from unify.memory.integration import fork as fork_mod
from unify.memory.integration import fork_worker as fw
from tests.memory.integration.test_fork import (  # noqa: F401
    _reply,
    _source,
    lane,
)
from tests.memory.integration.test_request import (  # noqa: F401
    _begin,
    _finish,
    _transcript,
    mv2,
)

# --- assumed names (the spec leaves these open; the integrator adapts them here, in one place) ---------------------
#: the summary file's staged path and its kind in the block ledger (H.4, H.5 item 6)
SUMMARY_PATH = "summary.md"
SUMMARY_KIND = "summary"
#: the one format constant inserted into the fork's instruction (H.4)
SUMMARY_FORMAT_NAME = "SUMMARY_FORMAT"
#: the five headed sections, in order (H.4)
SECTIONS = ("## Asked", "## Approach", "## Worked", "## Failed", "## Outcome")
#: the fork's "What to write" bullet for the summary (H.4) and the one for candidates that follows it today
SUMMARY_BULLET = "- `summary.md`"
CANDIDATE_BULLET = "- `candidates/<name>.py`"
#: fork.json's new fields (H.5)
FORK_SUMMARY_BYTES = "summary_bytes"
FORK_SECONDS = "seconds"
#: the batch-map row's new key: the summary's path in the writer's box, or None (H.4)
ROW_SUMMARY = "summary"
#: an unfinished reader's row status prefix and the reason each withheld path carries (H.3.5, H.9 item 2)
NO_OUTPUT = "no reader output: "
WITHHELD_WHY = "the reader did not finish"
#: the end event's per-episode rows and their fields (H.5, narrowed to the fork)
STAGING_EPISODES = "staging_episodes"
POINTERS = "summary_pointers"
RESOLVED = "summary_pointers_resolved"
#: the end event's writer token fields (H.5)
INPUT_TOKENS_CALLS = "input_tokens_calls"
WRITER_CALLS = "writer_calls"
FIRST_PROMPT = "writer_first_prompt_tokens"
# ------------------------------------------------------------------------------------------------------------------

SUMMARY = (
    "## Asked\n"
    "Pay back friends through a payments service.\n"
    "## Approach\n"
    '- Looked up the user (`action:0` "u-1").\n'
    "## Worked\n"
    '- Reading the request (`request` "Venmo friends").\n'
    "## Failed\n"
    '- A step the record does not hold (`cell:4` "missing").\n'
    "## Outcome\n"
    'not shown (`action:0` "not in the response").\n'
)


def _fence(rel: str, body: str) -> str:
    return f"```path: {rel}\n{body}```\n"


def _req(eid="e1"):
    return {
        "episode": eid,
        "url": "u",
        "headers": {},
        "body": {"model": "m", "messages": []},
    }


# --- the format and the fork's instruction ---------------------------------------------------------------------------


def _summary_format() -> str:
    return getattr(library_prompts, SUMMARY_FORMAT_NAME)


def test_the_summary_format_has_five_sections_in_order_and_pointer_rules():
    text = _summary_format()
    at = [text.find(s) for s in SECTIONS]
    assert all(i >= 0 for i in at) and at == sorted(at), at
    for part in ("`request`", "`observation:<n>`", "`cell:<n>`", "`action:<n>`", "`diff`"):
        assert part in text, part
    # the outcome is never guessed
    assert "not shown" in text
    # general across domains, and nothing about checking a task's examples
    assert library_prompts.benchmark_words(text) == []
    assert library_prompts.example_checks(text) == []


@pytest.mark.parametrize("which", ["INSTRUCTION", "INSTRUCTION_IN_CODE"])
def test_the_fork_instruction_asks_for_the_summary_first_in_the_one_format(which):
    text = getattr(fork_mod, which)
    assert _summary_format() in text
    s, c = text.find(SUMMARY_BULLET), text.find(CANDIDATE_BULLET)
    assert 0 <= s < c, (s, c)
    # the rest of what it writes is still asked for
    for rel in ("`cases/<name>.json`", "`notes.md`"):
        assert rel in text
    assert library_prompts.benchmark_words(text) == []
    assert library_prompts.example_checks(text) == []


def test_the_format_example_stages_a_summary_first():
    blocks, record = fw.parse_reply(fork_mod.FORMAT_EXAMPLE)
    parsed = [b for b in blocks if b["status"] == "parsed"]
    assert parsed[0]["path"] == SUMMARY_PATH and parsed[0]["form"] == "fence"
    assert parsed[0]["kind"] == SUMMARY_KIND
    body = parsed[0]["content"]
    at = [body.find(s) for s in SECTIONS]
    assert all(i >= 0 for i in at) and at == sorted(at), at
    # the example's lines point into the record as the format says: (`part` "quote")
    assert re.search(r"\(`(request|observation:\d+|cell:\d+|action:\d+|diff)` \"[^\"]+\"\)", body)
    # the other example files still parse
    assert {b["path"] for b in parsed} >= {"candidates/double.py", "cases/double.json", "notes.md"}
    assert record["counts"]["refused"] == 0


# --- the worker: the same rules and records as every staged file --------------------------------------------------


def test_summary_md_is_an_allowed_staged_path_of_its_own_kind():
    assert fw.allowed(SUMMARY_PATH) is None
    assert fw.kind(SUMMARY_PATH) == SUMMARY_KIND
    for bad in ("summaries/summary.md", "summary.txt", "../summary.md", "/summary.md", "notes/summary.md"):
        assert fw.allowed(bad) is not None, bad


def test_the_worker_stages_the_summary_with_its_other_files_and_records_it(tmp_path):
    text = _fence(SUMMARY_PATH, SUMMARY) + _fence("notes.md", "A lesson.\n")
    doc = fw.run(_req(), tmp_path, post=lambda u, h, b: _reply(content=text))
    assert doc["status"] == "ok"
    assert (tmp_path / SUMMARY_PATH).read_text() == SUMMARY
    assert {f["path"]: f["from"] for f in doc["files"]} == {SUMMARY_PATH: "text", "notes.md": "text"}
    # fork.json, written last, records the summary's size and the fork's own wall time
    assert doc[FORK_SUMMARY_BYTES] == len(SUMMARY.encode())
    assert isinstance(doc[FORK_SECONDS], (int, float)) and doc[FORK_SECONDS] >= 0
    assert json.loads((tmp_path / fw.FORK_JSON).read_text()) == doc
    # the block ledger has the summary as its own kind, parsed, with the content digest the fork's way
    ledger = json.loads((tmp_path / fw.BLOCKS_JSON).read_text())
    (row,) = [b for b in ledger["blocks"] if b["path"] == SUMMARY_PATH]
    assert row["kind"] == SUMMARY_KIND and row["status"] == "parsed" and row["form"] == "fence"
    assert "content" not in row


def test_a_fork_with_no_summary_records_no_summary_bytes(tmp_path):
    doc = fw.run(_req(), tmp_path, post=lambda u, h, b: _reply(content=_fence("notes.md", "n\n")))
    assert doc["status"] == "ok" and FORK_SUMMARY_BYTES in doc and doc[FORK_SUMMARY_BYTES] in (None, 0)


# --- staging lists the summary in the writer's row ------------------------------------------------------------------


def _settled(root: Path, eid: str, status: str, files: dict[str, str]) -> Path:
    d = root / eid
    d.mkdir(parents=True)
    for rel, body in files.items():
        (d / rel).parent.mkdir(parents=True, exist_ok=True)
        (d / rel).write_text(body)
    (d / fw.FORK_JSON).write_text(json.dumps({"episode": eid, "status": status}))
    return d


def test_the_row_names_the_summary_and_the_box_holds_it(tmp_path):
    _settled(tmp_path / "root", "e1", "ok", {SUMMARY_PATH: SUMMARY, "notes.md": "n\n"})
    view = staging.read(tmp_path / "root", "e1")
    assert set(view.files) == {SUMMARY_PATH, "notes.md"}
    entry = staging.export(view, tmp_path / "inputs" / "staging" / "e1")
    assert entry["status"] == "ok" and SUMMARY_PATH in entry["files"]
    # a finished fork hands over everything it staged
    assert not [r for r in entry["refused"] if r.get("why") == WITHHELD_WHY]
    assert entry[ROW_SUMMARY] == f"/inputs/staging/e1/{SUMMARY_PATH}"
    assert (tmp_path / "inputs" / "staging" / "e1" / SUMMARY_PATH).read_text() == SUMMARY


def test_a_row_without_a_summary_says_none(tmp_path):
    _settled(tmp_path / "root", "e1", "ok", {"notes.md": "n\n"})
    entry = staging.export(staging.read(tmp_path / "root", "e1"), tmp_path / "out" / "e1")
    assert entry[ROW_SUMMARY] is None


def test_the_row_key_description_names_the_summary():
    key, doc = library_prompts.STAGING_ROW_KEY
    assert key == "staging" and f"`{ROW_SUMMARY}`" in doc


# --- an unfinished fork hands the writer nothing --------------------------------------------------------------------


@pytest.mark.parametrize(
    "status",
    [
        "timeout",
        "lost",
        "error: TimeoutError",
        "error: operational turn guard (16) reached",
        "refused: no fork lane",
    ],
)
def test_an_unfinished_fork_stages_nothing_and_its_row_says_why(tmp_path, status):
    files = {SUMMARY_PATH: SUMMARY, "notes.md": "n\n", "candidates/f.py": "def f(x):\n    return x\n"}
    d = _settled(tmp_path / "root", "e1", status, files)
    view = staging.read(tmp_path / "root", "e1")
    assert view is not None  # settled: the episode is admitted, with no reader output
    assert view.files == {}
    dest = tmp_path / "inputs" / "staging" / "e1"
    entry = staging.export(view, dest)
    assert entry["status"] == NO_OUTPUT + status
    assert entry["files"] == [] and entry.get(ROW_SUMMARY) is None
    withheld = {r["path"] for r in entry["refused"] if r.get("why") == WITHHELD_WHY}
    assert withheld == set(files)
    assert not dest.exists() or not [p for p in dest.rglob("*") if p.is_file()]
    # the fork's files stay on the host, for the record
    for rel, body in files.items():
        assert (d / rel).read_text() == body


def test_a_fork_stopped_by_its_wall_bound_keeps_its_files_but_hands_over_none(tmp_path):
    now = [0.0]

    def post(u, h, b):
        now[0] += 10.0
        return _reply(
            content=_fence(SUMMARY_PATH, SUMMARY) + _fence("notes.md", "n\n"),
            calls=[{"id": "c", "function": {"name": "x", "arguments": "{}"}}],
        )

    d = tmp_path / "root" / "e1"
    d.mkdir(parents=True)
    doc = fw.run(_req(), d, post=post, wall_s=15.0, clock=lambda: now[0])
    assert doc["status"] == "timeout"
    assert (d / SUMMARY_PATH).is_file() and (d / "notes.md").is_file()  # kept on the host
    entry = staging.export(staging.read(tmp_path / "root", "e1"), tmp_path / "out" / "e1")
    assert entry["status"] == NO_OUTPUT + "timeout" and entry["files"] == []
    assert {r["path"] for r in entry["refused"] if r.get("why") == WITHHELD_WHY} == {SUMMARY_PATH, "notes.md"}


# --- the fork is given the episode's part list ----------------------------------------------------------------------


def test_the_fork_request_carries_the_recorded_part_list_after_the_cached_prefix(mv2, monkeypatch, lane):  # noqa: F811
    from unify.memory.blobs import BlobStore
    from unify.memory.episodes import episode_dir, load_episode
    from unify.memory.evidence import EvidenceStore
    from unify.memory.gitio import Repo
    from unify.memory.writer_tools import part_names

    import unify.actor.code_act_actor as caa

    src = _source()
    monkeypatch.setattr(caa, "_session_fork_source", lambda inner, actor: (src, None))
    monkeypatch.setattr(fork_mod.shutil, "which", lambda n: f"/usr/bin/{n}")
    sent: list[bytes] = []

    def launch(reservation, argv, data, log_path, **kw):
        sent.append(data)
        reservation.abandon("ok")
        return None

    monkeypatch.setattr(fork_mod, "launch", launch)
    run = _begin(mv2, "Double 21.")
    _transcript(
        run,
        {"seq": 0, "type": "message", "role": "user", "content": "Double 21."},
        {"seq": 1, "type": "message", "role": "assistant", "content": "Which number?"},
        {"seq": 2, "type": "message", "role": "user", "content": "Twenty-one."},
        {"seq": 3, "type": "message", "role": "assistant", "content": "42"},
    )
    _finish(mv2, run, consolidate=False)
    eid = run.episode_id
    # the episode as recorded (read from the episodes repo; the request lifecycle's consolidate is a stand-in here)
    sha, started_at = EvidenceStore(mv2.paths.evidence).episode_ref(eid)
    rel = episode_dir(SimpleNamespace(episode_id=eid, started_at=started_at))
    ep = load_episode(Repo(mv2.paths.episodes), sha, rel, BlobStore(mv2.paths.blobs))
    names = part_names(ep)
    assert "request" in names and len(names) > 1, names  # the fixture records more than the request
    (data,) = sent
    req = json.loads(data.decode())
    messages = req["body"]["messages"]
    prefix = src["sent_messages"]
    # the actor's own messages come first, unchanged, so the provider serves them from cache
    assert messages[: len(prefix)] == prefix
    added = "\n".join(
        m["content"] if isinstance(m.get("content"), str) else json.dumps(m.get("content"))
        for m in messages[len(prefix) :]
    )
    assert fork_mod.fork_instruction() in added
    for name in names:
        assert re.search(rf"(?<![\w:]){re.escape(name)}(?![\w:])", added), name
    # nothing of the host reaches the request
    assert str(mv2.home) not in added


# --- the consolidation end event: pointers per episode and the writer's input tokens ------------------------------


