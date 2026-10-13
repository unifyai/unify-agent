"""Capture end to end, on a recorded episode (design I.2, with the decisions of I.10 and I.11).

The episode handed back a result: a text request whose rows of numbers it summed, answered by one dialogue action
(``{"sums": [7, 11]}``) that the counterpart accepted. Its path through the build:

1. the episode-end fork's reply (a recorded reply, posted by a scripted proxy through ``fork_worker.run``) stages
   ``summary.md``, ``procedure.py``, ``broke.md`` and ``notes.md`` in the episode's staging dir;
2. the harness's capture step (``captures.capture``) checks it, runs it confined on the recorded request and
   writes the capture: ``procedure.py``, ``signature.json`` (an applicability signature with no identifier) and
   ``record.json`` (the capture's own evidence level and status);
3. a writer pass, tools called from code, in real boxes, sees the capture in the episode's batch-map row and as a
   ``capture`` pool entry, lands it with ``tools.land_capture``, and the gate admits the procedure into the
   library at the evidence level the gate gives it, with a new item's status.

Keyless: the fork's model reply is recorded text; the writer is scripted. The boxes need bubblewrap and prlimit.
"""

from __future__ import annotations

import asyncio
import hashlib
import importlib
import json
import re
from dataclasses import fields as dataclass_fields
from pathlib import Path
from typing import Any

import pytest

from tests.memory.test_episodes import _ep
from tests.memory.test_tool_channel_box import needs_box
from tests.memory.test_writer_pass import _call
from unify.memory.blobs import BlobStore
from unify.memory.episodes import Action, Cell
from unify.memory.evidence import EvidenceStore
from unify.memory.gate import Gate
from unify.memory.gitio import Repo
from unify.memory.integration import fork_worker as fw
from unify.memory.library_checks import LibraryConfig
from unify.memory.trigger import PassRequest
from unify.memory.writer_pass import PassConfig, WriterPass
from unify.settings import SETTINGS

# ── assumed names: the integrator adapts these to the build in this one block ─────────────────────────────────
#: Design I.2.3: ``captures.capture(staging_dir, episode, store)`` runs at once when a fork settles.
CAPTURES_MODULE = "unify.memory.captures"


def run_capture(staging_dir: Path, episode, store: Path) -> str:
    """The capture step on one settled staging dir; returns the capture id (whatever the step returns: the id, a
    row holding ``capture_id``, or the capture's folder)."""
    got = importlib.import_module(CAPTURES_MODULE).capture(staging_dir, episode, store)
    if isinstance(got, str):
        return got
    if isinstance(got, Path):
        return got.name
    if isinstance(got, dict):
        return got["capture_id"]
    return getattr(got, "capture_id")


#: The capture store under the state dir (``<state>/memory-captures/<capture id>/``) and its files.
CAPTURE_STORE = "memory-captures"
CAPTURE_FILES = ("procedure.py", "signature.json", "record.json")
CAPTURE_EVENTS = "captures-events.jsonl"
SIGNATURE_KEYS = {"input", "output", "relations", "constraints", "goal"}
RECORD_KEYS = {"episodes", "reproduces", "deterministic", "applies", "holds", "outcome", "level", "status", "track"}
#: Design I.11 decision 3: evidence levels (lowest first) and a new item's status.
LEVELS = ("unchecked", "reproduces-its-case", "confirmed-by-outcomes")
LEVEL_CAPTURE = LEVELS[0]
#: A procedure landed from a capture whose episode carries no verdict visible to the actor reaches the lower library
#: level only (design I.3.5: the ``request`` cover needs a visible positive signal for the top level).
LEVEL_LANDED_WITHOUT_SIGNAL = LEVELS[1]
STATUS_NEW = "on-trial"
#: Design I.2.2: the fork's new staged files and their ``fork-blocks.json`` kinds.
PROCEDURE_FILE, PROCEDURE_KIND = "procedure.py", "procedure"
BROKE_FILE, BROKE_KIND = "broke.md", "broke"
#: Design I.2.6: the batch-map row's key, the pool entry kind, the writer tool and where it puts the procedure.
ROW_CAPTURE = "capture"
POOL_KIND = "capture"
PROCEDURE_ITEM = "memory.readings.procedures:row_sums"
#: The pass configuration field naming the capture store (None: the pass finds it beside ``staging_root``).
PASS_CAPTURE_FIELD = "capture_root"


def gate_level(result, item: str) -> str:
    """Design I.3.2: the gate result's per-item evidence level."""
    return result.levels[item]


# ── end of assumed names ──────────────────────────────────────────────────────────────────────────────────────

REFERENCE = "ref-5f2c19"


def request_text(rows: list[list[int]], reference: str = REFERENCE) -> str:
    lines = "\n".join("row: " + " ".join(str(x) for x in row) for row in rows)
    return f"Reference: {reference}\nReadings for the week, one row per day:\n{lines}\nReply with the sum of each row."


def recorded_episode(eid: str = "e7", rows=((3, 4), (5, 6)), reference: str = REFERENCE):
    """A recorded episode (the shape of ``tests/memory/test_procedures.py``'s dialogue episodes): a text request,
    one cell, and the dialogue action that handed the result back, accepted by the counterpart."""
    rows = [list(r) for r in rows]
    sums = [sum(r) for r in rows]
    text = request_text(rows, reference)
    return _ep(
        episode_id=eid,
        request=[text],
        transcript=[{"role": "user", "content": text}],
        cells=[Cell(0, f"rows = {rows!r}\nprint([sum(r) for r in rows])", f"{sums!r}")],
        actions=[
            Action(-1, "dialogue:user", "answer", [], {"sums": sums}, "Accepted.", "ok", "unknown", kind="dialogue"),
        ],
    )


PROCEDURE = '''"""Sum each row of numbers read from the request text.

Accepted: the counterpart answered "Accepted." (`action:0` "Accepted.").
"""


def read_input(text):
    """The rows of numbers: every line of the text that starts with "row:"."""
    rows = []
    for line in text.splitlines():
        if line.startswith("row:"):
            rows.append([int(x) for x in line[len("row:"):].split()])
    return rows


def run(value):
    """The sum of each row."""
    return {"sums": [sum(row) for row in value]}


def applies(value):
    """At least one row, each a non-empty list."""
    return bool(value) and all(isinstance(row, list) and row for row in value)


def holds(value, result):
    """One sum per row."""
    return len(result["sums"]) == len(value)
'''
ASKED = "Sum each row of numbers given as lines of text, and reply with the sums."
SUMMARY = (
    "## Asked\n"
    f"{ASKED}\n"
    "## Approach\n"
    '- Read the rows from the lines that start with row: (`cell:0` "rows =").\n'
    "## Worked\n"
    '- Summing each row (`cell:0` "print([sum(r) for r in rows])").\n'
    "## Failed\n"
    "none\n"
    "## Outcome\n"
    'Replied with the sums, and the reply was accepted (`action:0` "Accepted.").\n'
)
BROKE = "none\n"
NOTES = "- Rows of numbers came one per line after a row: label.\n"


def _fence(rel: str, body: str) -> str:
    return f"```path: {rel}\n{body}```\n\n"


def fork_reply_text(procedure: str = PROCEDURE) -> str:
    """The fork's reply as the fork writes it: one fenced block per file."""
    return (
        _fence("summary.md", SUMMARY)
        + _fence(PROCEDURE_FILE, procedure)
        + _fence(BROKE_FILE, BROKE)
        + _fence("notes.md", NOTES)
    )


def stage_fork(staging_dir: Path, episode, procedure: str = PROCEDURE) -> dict:
    """The episode-end fork for *episode*, run by the real worker logic against a scripted proxy whose one reply is
    :func:`fork_reply_text`; it settles the staging dir (``fork.json``)."""
    staging_dir.mkdir(parents=True, exist_ok=True)
    answer = {
        "choices": [{"message": {"role": "assistant", "content": fork_reply_text(procedure)}}],
        "usage": {"prompt_tokens": 1000, "completion_tokens": 300, "cost": "0.0012"},
    }
    req = {
        "episode": episode.episode_id,
        "url": "http://127.0.0.1:1/v1/chat/completions",
        "headers": {"x-unify-session": f"fork.{episode.episode_id}"},
        "body": {"model": "m", "messages": [{"role": "user", "content": episode.request[0]}]},
    }
    return fw.run(req, staging_dir, post=lambda url, headers, body: answer)


def capture_episode(state_dir: Path, episode, procedure: str = PROCEDURE) -> tuple[str, Path]:
    """Fork, then capture: the capture id and the store."""
    staging = state_dir / "memory-staging" / episode.episode_id
    doc = stage_fork(staging, episode, procedure)
    assert (staging / PROCEDURE_FILE).is_file(), f"the fork did not stage {PROCEDURE_FILE}: {doc.get('refused_files')}"
    assert (staging / PROCEDURE_FILE).read_text() == procedure, doc
    store = state_dir / CAPTURE_STORE
    return run_capture(staging, episode, store), store


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text())


# ── 1-2. the fork stages the procedure; the capture step records it ──────────────────────────────────────────


@needs_box
@pytest.mark.timeout(600)
def test_the_fork_stages_a_procedure_and_the_capture_step_records_it_confined(tmp_path):
    ep = recorded_episode()
    staging = tmp_path / "memory-state" / "memory-staging" / ep.episode_id
    doc = stage_fork(staging, ep)
    assert doc["status"] == "ok", doc
    staged = {f["path"] for f in doc["files"]}
    assert {"summary.md", PROCEDURE_FILE, BROKE_FILE, "notes.md"} <= staged, doc
    assert not doc.get("refused_files"), doc["refused_files"]
    kinds = {b["path"]: b["kind"] for b in _read_json(staging / "fork-blocks.json")["blocks"]}
    assert kinds[PROCEDURE_FILE] == PROCEDURE_KIND and kinds[BROKE_FILE] == BROKE_KIND, kinds
    fork_json = _read_json(staging / "fork.json")
    assert fork_json.get("procedure") is True and fork_json.get("broke") is True, fork_json

    store = tmp_path / "memory-state" / CAPTURE_STORE
    cid = run_capture(staging, ep, store)
    # the id is the code's digest: the same code from another attempt joins this capture
    assert cid == hashlib.sha256(PROCEDURE.encode()).hexdigest()[:16]
    folder = store / cid
    assert sorted(p.name for p in folder.iterdir() if p.is_file()) == sorted(CAPTURE_FILES)
    assert (folder / "procedure.py").read_text() == PROCEDURE
    record = _read_json(folder / "record.json")
    assert RECORD_KEYS <= set(record), record
    assert record["episodes"] == [ep.episode_id]
    # run confined on the recorded request: it gave the handed-back value, twice alike, its conditions held
    assert record["reproduces"] in (True, "reproduces"), record
    assert record["deterministic"] is True and record["applies"] is True and record["holds"] is True, record
    assert record["outcome"] != "failed", record
    assert (record["level"], record["status"]) == (LEVEL_CAPTURE, STATUS_NEW), record
    # the applicability signature: computed by the harness, with no identifier and no request text
    signature = _read_json(folder / "signature.json")
    assert SIGNATURE_KEYS <= set(signature), signature
    assert ASKED in json.dumps(signature["goal"]), signature["goal"]
    flat = json.dumps(signature)
    assert REFERENCE not in flat and f'"{ep.episode_id}"' not in flat, flat
    assert "Readings for the week" not in flat and "row: 3 4" not in flat, flat
    # one event row per capture step, with when the actor could first see it
    (row,) = [json.loads(x) for x in (store / CAPTURE_EVENTS).read_text().splitlines() if x.strip()]
    assert (row["capture_id"], row["episode"], row["safety"]) == (cid, ep.episode_id, "ok"), row
    assert row["published_at"], row


@needs_box
@pytest.mark.timeout(600)
def test_the_same_work_under_another_identifier_joins_one_capture_with_the_same_signature(tmp_path):
    """A second attempt at the same work, whose request differs only in its numbers and its reference line, writes
    the same procedure: one capture holds both episodes, and its signature does not change."""
    state = tmp_path / "memory-state"
    first = recorded_episode("e7")
    again = recorded_episode("e8", rows=((1, 2), (10, 20)), reference="ref-0a9d44")
    cid, store = capture_episode(state, first)
    signature = (store / cid / "signature.json").read_bytes()
    cid2, _ = capture_episode(state, again)
    assert cid2 == cid
    record = _read_json(store / cid / "record.json")
    assert sorted(record["episodes"]) == ["e7", "e8"], record
    assert (store / cid / "signature.json").read_bytes() == signature
    assert b"ref-0a9d44" not in signature and b"ref-5f2c19" not in signature


@needs_box
@pytest.mark.timeout(600)
def test_a_procedure_with_a_key_shaped_string_is_refused_before_anything_runs(tmp_path):
    state = tmp_path / "memory-state"
    marker = tmp_path / "ran"
    leaky = PROCEDURE.replace(
        '    """The sum of each row."""\n',
        f'    """The sum of each row."""\n    open({str(marker)!r}, "w").write("x")\n'
        '    token = "sk-proj-AbCdEfGhIjKlMnOpQrStUvWxYz0123456789"\n',  # pragma: allowlist secret
    )
    ep = recorded_episode()
    staging = state / "memory-staging" / ep.episode_id
    stage_fork(staging, ep, leaky)
    assert (staging / PROCEDURE_FILE).is_file(), f"the fork did not stage {PROCEDURE_FILE}"
    store = state / CAPTURE_STORE
    importlib.import_module(CAPTURES_MODULE)  # the step exists; only its refusal may raise below
    try:
        run_capture(staging, ep, store)
    except Exception:  # noqa: BLE001 - a refusal may be raised or recorded; both are checked below
        pass
    assert not marker.exists()
    rows = [json.loads(x) for x in (store / CAPTURE_EVENTS).read_text().splitlines() if x.strip()]
    assert rows and rows[-1]["safety"] != "ok", rows
    assert not any((store / r["capture_id"]).is_dir() for r in rows if r.get("capture_id")), rows


# ── 3. a writer pass lands the capture as a library item ─────────────────────────────────────────────────────


LAND_CELL = """
import json
bm = json.load(open('/inputs/batch_map.json'))
row = next(r for r in bm['episodes'] if 'e7' in (r.get('episode_id'), r.get('episode')))
print('CAPTURE ' + json.dumps(row['capture'], sort_keys=True))
print('POOL ' + tools.pool(op='list', kind='capture').replace(chr(10), ' | '))
print('LANDED ' + tools.land_capture(row['capture']['id'], 'readings', 'row_sums').replace(chr(10), ' | '))
print('CHECK ' + tools.check()[:3000].replace(chr(10), ' | '))
"""
FINISH_CELL = "tools.finish('landed the captured procedure')\nprint('NOT REACHED')\n"


class _Cells:
    """A scripted writer: one cell per turn, then the end call; keeps every cell's output."""

    def __init__(self, cells: list[str]) -> None:
        self.cells, self.outputs = list(cells), {}

    async def __call__(self, messages, tools):
        for m in messages:
            if m.get("role") == "tool":
                self.outputs[m["tool_call_id"]] = str(m.get("content"))
        n = len(self.outputs) + 1
        cell = self.cells.pop(0) if self.cells else FINISH_CELL
        return _call(f"c{n}", "execute_code", {"code": cell}), "0.001"


def _line(text: str, mark: str) -> str:
    return next((x[len(mark) :] for x in text.splitlines() if x.startswith(mark)), "")


@needs_box
@pytest.mark.timeout(1200)
def test_a_writer_pass_lands_the_capture_as_a_library_item(tmp_path, monkeypatch):
    monkeypatch.setattr(SETTINGS, "UNIFY_TOOLS_IN_CODE", "on")
    state = tmp_path / "memory-state"
    ep = recorded_episode()
    cid, store = capture_episode(state, ep)
    episodes = {ep.episode_id: ep}
    mem = Repo.init_bare(tmp_path / "mem.git")
    ev = EvidenceStore(tmp_path / "e.sqlite")
    ev.index_episode(ep, "7" * 40)
    gate = Gate(
        mem,
        ev,
        BlobStore(tmp_path / "b"),
        action_lookup=lambda eid, i: episodes[eid].actions[i] if eid in episodes else None,
        library_mode=LibraryConfig(checks=True, episodes=episodes.get),
    )
    results: list = []
    merge = gate.merge

    def spy(*args, **kwargs):
        res = merge(*args, **kwargs)
        results.append(res)
        return res

    gate.merge = spy
    extra = {}
    if PASS_CAPTURE_FIELD and PASS_CAPTURE_FIELD in {f.name for f in dataclass_fields(PassConfig)}:
        extra[PASS_CAPTURE_FIELD] = str(store)
    model = _Cells([LAND_CELL])
    writer = WriterPass(
        mem,
        gate,
        ev,
        load=episodes.__getitem__,
        model_turn=model,
        config=PassConfig(
            library_mode=True,
            tools_in_code=True,
            fork=True,
            pool=True,
            staging_root=str(state / "memory-staging"),
            pass_record_root=str(tmp_path / "out"),
            max_calls=6,
            cell_timeout_s=300.0,
            deadline_s=1500.0,
            **extra,
        ),
    )
    out = asyncio.run(writer.run(PassRequest("incremental", "readings", [ep.episode_id], False), "p1"))
    shown = model.outputs.get("c1", "")
    row = json.loads(_line(shown, "CAPTURE ") or "null")
    assert row is not None, shown[:3000]
    assert row["id"] == cid and row["level"] == LEVEL_CAPTURE, row
    assert row["reproduces"] in (True, "reproduces") and "same_as" in row and "outcome" in row, row
    assert cid in _line(shown, "POOL ") or POOL_KIND in _line(shown, "POOL "), shown[:3000]
    landed = _line(shown, "LANDED ")
    assert "readings/procedures.py" in landed and "readings/readers.py" in landed, landed
    assert out.finished and out.passed, (out.reasons, shown[:3000])
    assert PROCEDURE_ITEM in out.items_merged, (out.items_merged, out.items_refused)
    # the gate admitted it at its evidence level: reproduced on its recorded case, no verdict the actor saw
    final = results[-1]
    assert gate_level(final, PROCEDURE_ITEM) == LEVEL_LANDED_WITHOUT_SIGNAL, getattr(final, "level_reasons", None)
    # the landed library lists it as a new item: its status is the new-item status
    from unify.memory.library_export import generated_library_mode

    with mem.temp_checkout(out.commit) as wt:
        index = generated_library_mode(wt)["INDEX.md"].decode()
    line = next((x for x in index.splitlines() if "row_sums" in x), "")
    assert line and f"[{STATUS_NEW}" in line, index[:3000]
    # nothing in what the writer was shown names the episode's identifier line through the capture row
    assert REFERENCE not in json.dumps(row)
    assert re.fullmatch(r"[0-9a-f]{16}", cid)
