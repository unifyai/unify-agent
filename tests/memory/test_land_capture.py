"""The writer's ``land_capture`` tool: a captured attempt becomes a reader, a procedure, fixtures, a test and manifest
entries in the writer's /memory (design I.2.6, I.2.7).

The capture is read from the harness's own store, never from a path the box names, and checked first: its id, its
files (regular, no link, bounded), its procedure's digest against its id, its outcome and status, the safety checks.
Model-written code runs only confined. The gate admitting a landing at its evidence level, through a whole writer
pass, is ``tests/memory/test_capture_end_to_end.py::test_a_writer_pass_lands_the_capture_as_a_library_item``.
Keyless; the landings that run code need bubblewrap and prlimit.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.memory.test_capture_end_to_end import PROCEDURE, recorded_episode
from tests.memory.test_tool_channel_box import needs_box
from unify.common import tool_registry as tr
from unify.memory import land_capture as lc
from unify.memory import tool_handlers
from unify.memory import writer_tools as wt
from unify.memory.manifest import ManifestError, parse_manifest

PLANTED = "planted-land-capture-value-0451"
ROWS = [[1, 2, 2], [3, 4, 5]]


def _cid(source: str) -> str:
    return hashlib.sha256(source.encode("utf-8")).hexdigest()[:16]


def _store(root: Path, source: str = PROCEDURE, **record) -> str:
    cid = _cid(source)
    d = root / cid
    d.mkdir(parents=True)
    (d / "procedure.py").write_text(source, encoding="utf-8")
    rec = {
        "capture_id": cid,
        "episodes": ["e7"],
        "outcome": "none",
        "status": "on-trial",
        "same_as": [],
        **record,
    }
    (d / "record.json").write_text(json.dumps(rec), encoding="utf-8")
    return cid


def _no_blob(sha: str) -> bytes:
    raise KeyError(sha)


def _land(tmp_path: Path, episodes: dict, **args) -> wt.Reply:
    box = tmp_path / "box"
    box.mkdir(exist_ok=True)
    return lc.land(
        {"package": "readings", "name": "row_sums", **args},
        store=tmp_path / "store",
        box=box,
        episode=episodes.get,
        blob=_no_blob,
        batch=sorted(episodes),
    )


def _manifest(tmp_path: Path) -> dict:
    return json.loads((tmp_path / "box" / ".pass" / "manifest.json").read_text())


def _no_leak(reply: wt.Reply, tmp_path: Path) -> None:
    assert str(tmp_path) not in reply.text and os.path.expanduser("~") not in reply.text
    assert PLANTED not in reply.text
    for row in reply.paths:
        assert str(tmp_path) not in json.dumps(row)


# --- the landing --------------------------------------------------------------------------------------------------


@needs_box
@pytest.mark.timeout(600)
def test_landing_writes_the_reader_procedure_fixtures_test_and_manifest_entries(tmp_path, monkeypatch):
    monkeypatch.setenv("UNIFY_PLANTED_FOR_THE_LANDING_TEST", PLANTED)
    ep = recorded_episode("e7", rows=ROWS)
    cid = _store(tmp_path / "store")
    reply = _land(tmp_path, {"e7": ep}, capture_id=cid)
    assert reply.status == "ok", reply.text
    mem = tmp_path / "box" / "memory" / "readings"
    procedures = (mem / "procedures.py").read_text()
    readers = (mem / "readers.py").read_text()
    # the procedure under its name, its conditions kept private, the capture named in its marker line
    assert "def row_sums(value):" in procedures and "def _row_sums_applies(value):" in procedures
    assert "def _row_sums_holds(value, result):" in procedures and "def run(" not in procedures
    assert f"# Landed from capture {cid}: procedure memory.readings.procedures:row_sums" in procedures
    assert "Example: memory/readings/tests/test_row_sums.py::test_row_sums_case1_gives_the_recorded_result" in (
        procedures
    )
    # the reader under read_<name>, one text parameter
    assert "def read_row_sums(text):" in readers and "def read_input(" not in readers
    assert (mem / "__init__.py").read_text().startswith('"""')
    # the request text, and its parsed value through the reader, as fixtures the gate re-derives
    data = mem / "tests" / "data"
    assert (data / "row_sums.case1.txt").read_text() == ep.request[0]
    assert json.loads((data / "row_sums.case1.json").read_text()) == ROWS
    listing = json.loads((tmp_path / "box" / ".pass" / "fixtures.json").read_text())
    parsed = json.dumps(listing)
    assert "memory.readings.readers:read_row_sums" in parsed and "parsed" in parsed
    # the test: the recorded result, the reader on the recorded text, and one refusal
    test = (mem / "tests" / "test_row_sums.py").read_text()
    assert "def test_row_sums_case1_gives_the_recorded_result():" in test
    assert json.dumps({"sums": [sum(r) for r in ROWS]}, sort_keys=True) in test
    assert "def test_row_sums_case1_reader_reads_the_recorded_request():" in test
    assert "def test_row_sums_reader_refuses_text_it_was_not_built_for():" in test
    # the reader returns [] on empty text, so its refusal test records a known failure
    assert '@pytest.mark.xfail(strict=True, reason="the reader does not refuse text it was not built for yet")' in test
    # the manifest entries, with their roles
    items = {i["item"]: i for i in _manifest(tmp_path)["items"]}
    proc = items["memory.readings.procedures:row_sums"]
    assert proc["role"] == "procedure" and proc["reader"] == "memory.readings.readers:read_row_sums"
    assert proc["tests"] == ["memory/readings/tests/test_row_sums.py"] and proc["source_episodes"] == ["e7"]
    assert "covers" not in proc  # the request runner's episode cover is not built in this build
    assert items["memory.readings.readers:read_row_sums"]["role"] == "reader"
    parse_manifest(_manifest(tmp_path), library_mode=True)
    # the reply lists the files written, as the box sees them
    for rel in (
        "readings/procedures.py",
        "readings/readers.py",
        "readings/tests/data/row_sums.case1.txt",
        "readings/tests/data/row_sums.case1.json",
        "readings/tests/test_row_sums.py",
    ):
        assert f"- wrote /memory/memory/{rel}" in reply.text, reply.text
    assert "known failure" in reply.text
    _no_leak(reply, tmp_path)


@needs_box
@pytest.mark.timeout(600)
def test_a_landed_reader_that_gives_the_same_value_is_reused(tmp_path):
    ep = recorded_episode("e7", rows=ROWS)
    cid = _store(tmp_path / "store")
    text_pkg = tmp_path / "box" / "memory" / "text"
    text_pkg.mkdir(parents=True)
    (text_pkg / "__init__.py").write_text('"""Readers of text."""\n')
    (text_pkg / "readers.py").write_text(
        "def labelled_rows(text):\n"
        '    """Every line that starts with row: as a list of integers."""\n'
        "    return [[int(x) for x in line[4:].split()] for line in text.splitlines() if line.startswith('row:')]\n",
    )
    reply = _land(tmp_path, {"e7": ep}, capture_id=cid)
    assert reply.status == "ok", reply.text
    assert "reader memory.text.readers:labelled_rows (reused" in reply.text
    assert not (tmp_path / "box" / "memory" / "readings" / "readers.py").exists()
    items = {i["item"]: i for i in _manifest(tmp_path)["items"]}
    assert items["memory.readings.procedures:row_sums"]["reader"] == "memory.text.readers:labelled_rows"
    assert set(items) == {"memory.readings.procedures:row_sums"}
    fixture = tmp_path / "box" / "memory" / "readings" / "tests" / "data" / "row_sums.case1.json"
    assert json.loads(fixture.read_text()) == ROWS
    assert "from memory.text.readers import labelled_rows" in (
        tmp_path / "box" / "memory" / "readings" / "tests" / "test_row_sums.py"
    ).read_text()


@needs_box
@pytest.mark.timeout(900)
def test_same_as_naming_a_landed_capture_adds_a_new_case_of_that_procedure(tmp_path):
    first = recorded_episode("e7", rows=ROWS)
    second = recorded_episode("e8", rows=((9, 1), (2, 2), (0, 5)))
    episodes = {"e7": first, "e8": second}
    cid = _store(tmp_path / "store")
    assert _land(tmp_path, episodes, capture_id=cid).status == "ok"
    # the same work, captured again with other code: its record names the landed capture
    other = PROCEDURE + "\n\n# written again in another attempt\n"
    cid2 = _store(tmp_path / "store", other, episodes=["e8"], same_as=[cid])
    reply = _land(tmp_path, episodes, capture_id=f"capture:{cid2}", package="elsewhere", name="other_name")
    assert reply.status == "ok", reply.text
    assert "a new case of memory.readings.procedures:row_sums" in reply.text
    mem = tmp_path / "box" / "memory" / "readings"
    assert not (tmp_path / "box" / "memory" / "elsewhere").exists()
    assert (mem / "procedures.py").read_text().count("# Landed from capture ") == 1
    data = mem / "tests" / "data"
    assert json.loads((data / "row_sums.case2.json").read_text()) == [[9, 1], [2, 2], [0, 5]]
    test = (mem / "tests" / "test_row_sums.py").read_text()
    assert "def test_row_sums_case1_gives_the_recorded_result():" in test
    assert "def test_row_sums_case2_gives_the_recorded_result():" in test
    assert test.count("def test_row_sums_reader_refuses_text_it_was_not_built_for():") == 1
    proc = {i["item"]: i for i in _manifest(tmp_path)["items"]}["memory.readings.procedures:row_sums"]
    assert proc["source_episodes"] == ["e7", "e8"]


# --- refusals, all before anything runs ---------------------------------------------------------------------------


def _refusal(reply: wt.Reply) -> str:
    assert reply.status == "refused", reply.text
    return reply.form


@pytest.mark.parametrize(
    "capture_id",
    ["nope", "../../etc/passwd", "0123456789ABCDEF", "0123456789abcdef", "capture:../0123456789abcdef", 7],
)
def test_a_forged_or_unknown_id_is_refused(tmp_path, capture_id):
    _store(tmp_path / "store")
    reply = _land(tmp_path, {"e7": recorded_episode()}, capture_id=capture_id)
    assert _refusal(reply) == "unknown_capture"
    assert reply.text.startswith(tr.REFUSED) and "is not a capture of the store; nothing was written" in reply.text
    assert not (tmp_path / "box" / "memory").exists()


@pytest.mark.parametrize(
    "args",
    [
        {"package": "Bad-Pkg"},
        {"package": "../escape"},
        {"package": "tests"},
        {"name": "class"},
        {"name": "row sums"},
        {"name": "x/../../y"},
        {"name": 3},
    ],
)
def test_a_name_or_package_that_is_not_a_plain_identifier_is_refused(tmp_path, args):
    cid = _store(tmp_path / "store")
    reply = _land(tmp_path, {"e7": recorded_episode()}, capture_id=cid, **args)
    assert _refusal(reply) == "bad_name"
    assert not (tmp_path / "box" / "memory").exists()


@pytest.mark.parametrize("record", [{"outcome": "failed"}, {"status": "retired"}])
def test_a_failed_or_retired_capture_is_refused(tmp_path, record):
    cid = _store(tmp_path / "store", **record)
    assert _refusal(_land(tmp_path, {"e7": recorded_episode()}, capture_id=cid)) == "failed_capture"


def test_a_procedure_that_does_not_match_its_id_is_refused(tmp_path):
    cid = _store(tmp_path / "store")
    (tmp_path / "store" / cid / "procedure.py").write_text(PROCEDURE.replace("sum(row)", "max(row)"))
    reply = _land(tmp_path, {"e7": recorded_episode()}, capture_id=cid)
    assert _refusal(reply) == "not_landed" and "does not match its id" in reply.text


@pytest.mark.parametrize("which", ["procedure.py", "record.json", "dir"])
def test_a_linked_store_file_is_refused(tmp_path, which):
    cid = _store(tmp_path / "store")
    d = tmp_path / "store" / cid
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    if which == "dir":
        real = tmp_path / "store" / "real"
        d.rename(real)
        d.symlink_to(real)
    else:
        (elsewhere / which).write_bytes((d / which).read_bytes())
        (d / which).unlink()
        (d / which).symlink_to(elsewhere / which)
    assert _refusal(_land(tmp_path, {"e7": recorded_episode()}, capture_id=cid)) == "unknown_capture"


def test_a_key_shaped_string_in_the_procedure_is_refused(tmp_path):
    source = PROCEDURE + '\nTOKEN = "sk-' + "a1b2c3d4" * 6 + '"\n'
    cid = _store(tmp_path / "store", source)
    reply = _land(tmp_path, {"e7": recorded_episode()}, capture_id=cid)
    assert _refusal(reply) == "not_landed" and "safety" in reply.text
    assert "a1b2c3d4" not in reply.text
    assert not (tmp_path / "box" / "memory").exists()


def test_no_store_is_an_unknown_capture(tmp_path):
    (tmp_path / "box").mkdir()
    reply = lc.land(
        {"capture_id": "0123456789abcdef", "package": "readings", "name": "row_sums"},
        store=None,
        box=tmp_path / "box",
        episode={}.get,
        blob=_no_blob,
    )
    assert _refusal(reply) == "unknown_capture"


# --- who has the tool, and the quota --------------------------------------------------------------------------------


def test_the_tool_is_on_the_writer_allow_list_only():
    spec = tr.registry()["land_capture"]
    assert tuple(spec.roles) == ("writer",)
    assert spec.handler == "unify.memory.tool_handlers:land_capture"
    flags = frozenset(s.requires for s in tr.registry().values() if s.requires is not None)
    for agent in tr.AGENTS:
        assert ("land_capture" in tr.allow_list(agent, flags)) == (agent == "writer"), agent
    assert set(spec.refusals) == {"unknown_capture", "failed_capture", "bad_name", "not_landed"}
    assert "land_capture" in {s.name for s in tr.tools_for("write")}
    assert "land_capture" not in {s.name for s in tr.tools_for("curate")}
    assert "land_capture" not in {s.name for s in tr.tools_for("helper")}


def test_the_quota_is_measured_after_a_landing(tmp_path, monkeypatch):
    calls = []

    def fake_land(args, **kw):
        calls.append(kw)
        return wt.Reply("Landed capture x", paths=())

    monkeypatch.setattr(lc, "land", fake_land)

    async def over():
        return "the /memory quota of 2000 entries is exceeded"

    writer = SimpleNamespace(
        _captures_root=lambda: tmp_path / "store",
        _episode=lambda eid: None,
        gate=SimpleNamespace(_blob=_no_blob, python=None),
        _batch=["e7"],
    )
    st = SimpleNamespace(writer=writer, box=tmp_path, memory_lock=threading.Lock(), measure_quota=over, stop=None)
    ctx = SimpleNamespace(state=st)
    reply = asyncio.run(tool_handlers.land_capture(ctx, {"capture_id": "x", "package": "p", "name": "n"}))
    assert "the pass is refused" in reply.text and st.stop is not None
    # the store is the harness's own, never a path from the call
    assert calls and calls[0]["store"] == tmp_path / "store" and calls[0]["box"] == tmp_path


def test_the_manifest_takes_a_procedure_role_and_its_reader():
    item = {
        "item": "memory.readings.procedures:row_sums",
        "kind": "function",
        "source_episodes": ["e7"],
        "tests": ["memory/readings/tests/test_row_sums.py"],
        "role": "procedure",
        "reader": "memory.readings.readers:read_row_sums",
    }
    parse_manifest({"items": [item]}, library_mode=True)
    for bad in ({"role": "oracle"}, {"role": "reader", "reader": item["reader"]}, {"reader": "../x"}):
        with pytest.raises(ManifestError):
            parse_manifest({"items": [{**item, **bad}]}, library_mode=True)


def test_an_actor_curator_or_helper_call_is_refused_by_the_allow_list():
    from tests.memory.test_tool_handlers import _channel_for, _thread_call

    flags = frozenset(s.requires for s in tr.registry().values() if s.requires is not None)
    args = {"capture_id": "0123456789abcdef", "package": "readings", "name": "row_sums"}

    async def main():
        out, rows = {}, []
        for agent in ("actor", "curator", "helper"):
            ch, box, got = _channel_for(agent, tr.allow_list(agent, flags), SimpleNamespace())
            task = asyncio.ensure_future(ch.serve())
            out[agent] = await _thread_call(box, "land_capture", args)
            await ch.close("ended", record=False)
            await asyncio.gather(task, return_exceptions=True)
            rows += got
        return out, rows

    out, rows = asyncio.run(main())
    for agent, exc in out.items():
        assert getattr(exc, "form", None) == "unknown_tool", (agent, exc)
    assert [r["refusal"]["form"] for r in rows if r.get("refusal")] == ["unknown_tool"] * 3
