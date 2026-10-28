"""The writer's ``land_capture`` tool: a captured attempt becomes a reader, a procedure, fixtures, a test and manifest
entries in the writer's /memory (design I.2.6, I.2.7).

The capture is read from the harness's own store, never from a path the box names, and checked first: its id, its
files (regular, no link, bounded), its procedure's digest against its id, its outcome (a capture record holds no
status), the safety checks.
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
    # the request text as a fixture the gate re-derives; rev 2 makes no parsed fixture (that would run the reader)
    data = mem / "tests" / "data"
    assert (data / "row_sums.case1.txt").read_text() == ep.request[0]
    assert not (data / "row_sums.case1.json").exists()
    listing = json.loads((tmp_path / "box" / ".pass" / "fixtures.json").read_text())
    parsed = json.dumps(listing)
    assert "row_sums.case1.json" not in parsed and "read_row_sums" not in parsed
    # the handed-back value as a fixture of the recording: a slice of the answering dialogue action's arguments
    result = data / "row_sums.case1.result.json"
    assert json.loads(result.read_text()) == {"sums": [sum(r) for r in ROWS]}
    made = listing["memory/readings/tests/data/row_sums.case1.result.json"]
    assert (made["kind"], made["episode"], made["source"], made["trust"]) == ("copy", "e7", "args:0", 2), made
    assert made["sha256"] == hashlib.sha256(result.read_bytes()).hexdigest()
    # the test: the procedure on the reader's value of the recorded text against that fixture, and the reader on the
    # recorded text; the expected value is loaded from the fixture, never written in the test
    test = (mem / "tests" / "test_row_sums.py").read_text()
    assert "def test_row_sums_case1_gives_the_recorded_result():" in test
    assert 'expected = json.loads((DATA / "row_sums.case1.result.json").read_text(encoding="utf-8"))' in test
    assert "assert json.loads(json.dumps(row_sums(read_row_sums(request(text))))) == expected" in test
    # the reader is handed the request as the harness hands it: the text with its data (request.data)
    assert "\ndef request(text):" in test and "return Request(text, json_values(text))" in test
    assert '"sums"' not in test
    assert "def test_row_sums_case1_reader_reads_the_recorded_request():" in test
    # no refusal test: whether the reader refuses text it was not built for is known only by running it, which a
    # landing never does; the reply asks the writer to add one once it does
    assert "refuses_text_it_was_not_built_for" not in test and "xfail" not in test
    assert "no refusal test was written" in reply.text
    # the manifest entries, with their roles
    items = {i["item"]: i for i in _manifest(tmp_path)["items"]}
    proc = items["memory.readings.procedures:row_sums"]
    assert proc["role"] == "procedure" and proc["reader"] == "memory.readings.readers:read_row_sums"
    assert proc["tests"] == ["memory/readings/tests/test_row_sums.py"] and proc["source_episodes"] == ["e7"]
    assert "covers" not in proc  # the request runner's episode cover is not built in this build
    assert items["memory.readings.readers:read_row_sums"]["role"] == "reader"
    # the recipe: the capture's run(read_input(request)) lifted into the library, its applies the precondition
    assert proc["recipe"] == "memory.readings.procedures:apply_row_sums"
    assert "def apply_row_sums(request):" in procedures
    recipe = procedures[procedures.index("def apply_row_sums(request):") :]
    assert "from memory.readings.readers import read_row_sums as read" in recipe
    assert "if not _row_sums_applies(value):" in recipe and "raise MemoryInputError(" in recipe
    assert "return row_sums(value)" in recipe
    assert items["memory.readings.procedures:apply_row_sums"]["tests"] == ["memory/readings/tests/test_row_sums.py"]
    assert "def test_row_sums_case1_recipe_gives_the_recorded_result():" in test
    assert "assert json.loads(json.dumps(apply_row_sums(request(text)))) == expected" in test
    # its source case: the recorded request with the value the episode handed back, re-derived by the gate
    assert proc["cases"] == ["memory/readings/tests/data/row_sums.case1.source.case.json"]
    case = json.loads((data / "row_sums.case1.source.case.json").read_text())
    assert (case["input"], case["expected"], case["kind"]) == (ep.request[0], {"sums": [5, 12]}, "source")
    assert "recipe memory.readings.procedures:apply_row_sums" in reply.text
    parse_manifest(_manifest(tmp_path))
    # the reply lists the files written, as the box sees them
    for rel in (
        "readings/procedures.py",
        "readings/readers.py",
        "readings/tests/data/row_sums.case1.txt",
        "readings/tests/data/row_sums.case1.result.json",
        "readings/tests/test_row_sums.py",
    ):
        assert f"- wrote /memory/memory/{rel}" in reply.text, reply.text
    assert "no refusal test was written" in reply.text
    _no_leak(reply, tmp_path)


@needs_box
@pytest.mark.timeout(600)
def test_a_landed_reader_that_is_the_same_code_is_reused(tmp_path):
    """Rev 2: a reader is reused when it is the capture's reader code (its shape: names, prose and local names aside),
    read from the syntax; no reader runs to find one that gives the same value."""
    ep = recorded_episode("e7", rows=ROWS)
    cid = _store(tmp_path / "store")
    text_pkg = tmp_path / "box" / "memory" / "text"
    text_pkg.mkdir(parents=True)
    (text_pkg / "__init__.py").write_text('"""Readers of text."""\n')
    (text_pkg / "readers.py").write_text(
        "def other_rows(text):\n"
        '    """Every line that starts with row: as a list of integers, another way: never reused."""\n'
        "    return [[int(x) for x in line[4:].split()] for line in text.splitlines() if line.startswith('row:')]\n"
        "\n\n"
        "def labelled_rows(source):\n"
        '    """Every line that starts with row: as a list of integers."""\n'
        "    found = []\n"
        "    for each in source.splitlines():\n"
        '        if each.startswith("row:"):\n'
        '            found.append([int(n) for n in each[len("row:"):].split()])\n'
        "    return found\n",
    )
    reply = _land(tmp_path, {"e7": ep}, capture_id=cid)
    assert reply.status == "ok", reply.text
    assert "reader memory.text.readers:labelled_rows (reused" in reply.text
    assert not (tmp_path / "box" / "memory" / "readings" / "readers.py").exists()
    items = {i["item"]: i for i in _manifest(tmp_path)["items"]}
    assert items["memory.readings.procedures:row_sums"]["reader"] == "memory.text.readers:labelled_rows"
    assert set(items) == {"memory.readings.procedures:row_sums", "memory.readings.procedures:apply_row_sums"}
    assert "from memory.text.readers import labelled_rows" in (
        tmp_path / "box" / "memory" / "readings" / "tests" / "test_row_sums.py"
    ).read_text()


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


@pytest.mark.parametrize("record", [{"outcome": "failed"}, {"outcome": "contrary"}])
def test_a_failed_capture_is_refused(tmp_path, record):
    cid = _store(tmp_path / "store", **record)
    assert _refusal(_land(tmp_path, {"e7": recorded_episode()}, capture_id=cid)) == "failed_capture"


def test_only_a_negative_outcome_refuses_a_capture_never_a_status_an_old_record_kept(tmp_path):
    # a capture record holds no status (its uses and the status they give it are the evidence store's): a record an
    # earlier build wrote with one is landed on its outcome alone
    cid = _store(tmp_path / "store", status="retired")
    reply = _land(tmp_path, {"e7": recorded_episode()}, capture_id=cid)
    assert reply.status == "ok" and reply.text.startswith(f"Landed capture {cid} from episode e7:"), reply.text


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


def test_the_final_landed_text_is_scanned_for_a_key_shape_too(tmp_path):
    # defence in depth: whatever the rename or the generated lines produce, no file with a key shape is written
    box = tmp_path / "box"
    box.mkdir()
    with pytest.raises(lc._NotLanded) as err:
        lc._write(box, "memory/readings/procedures.py", 'X = "sk-' + "a1b2c3d4" * 6 + '"\n', [])
    assert err.value.form == "not_landed"
    assert "key-shaped" in err.value.fields["reason"] and "a1b2c3d4" not in err.value.fields["reason"]
    assert not (box / "memory").exists()


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
    st = SimpleNamespace(
        writer=writer, box=tmp_path, memory_lock=threading.Lock(), measure_quota=over, stop=None, pass_id="p1"
    )
    ctx = SimpleNamespace(state=st)
    reply = asyncio.run(tool_handlers.land_capture(ctx, {"capture_id": "x", "package": "p", "name": "n"}))
    assert "the pass is refused" in reply.text and st.stop is not None
    # the store is the harness's own, never a path from the call
    assert calls and calls[0]["store"] == tmp_path / "store" and calls[0]["box"] == tmp_path


def test_the_manifest_takes_a_procedures_recipe_in_its_module_and_its_case_files():
    item = {
        "item": "memory.readings.procedures:row_sums",
        "kind": "function",
        "source_episodes": ["e7"],
        "tests": ["memory/readings/tests/test_row_sums.py"],
        "role": "procedure",
        "recipe": "memory.readings.procedures:apply_row_sums",
        "cases": ["memory/readings/tests/data/row_sums.case1.source.case.json"],
    }
    got = parse_manifest({"items": [item]}).items[0]
    assert (got.role, got.recipe, got.cases) == ("procedure", item["recipe"], item["cases"])
    for bad in (
        {"recipe": "memory.readings.other:apply_row_sums"},  # another module
        {"recipe": "apply_row_sums"},
        {"cases": ["memory/elsewhere/tests/data/x.case.json"]},  # another package
        {"cases": ["memory/readings/tests/data/x.json"]},  # not a case file
        {"cases": ["memory/readings/tests/data/x.case.json"] * 2},
        {"role": "reader"},  # recipe and cases are a procedure's
    ):
        with pytest.raises(ManifestError):
            parse_manifest({"items": [{**item, **bad}]})


def test_the_manifest_takes_a_procedure_role_and_its_reader():
    item = {
        "item": "memory.readings.procedures:row_sums",
        "kind": "function",
        "source_episodes": ["e7"],
        "tests": ["memory/readings/tests/test_row_sums.py"],
        "role": "procedure",
        "reader": "memory.readings.readers:read_row_sums",
    }
    parse_manifest({"items": [item]})
    for bad in ({"role": "oracle"}, {"role": "reader", "reader": item["reader"]}, {"reader": "../x"}):
        with pytest.raises(ManifestError):
            parse_manifest({"items": [{**item, **bad}]})


def test_every_agent_but_the_writer_is_refused_by_the_allow_list():
    # rev 2: the curator role is gone; every other agent of the run (tool_registry.AGENTS) is refused
    from tests.memory.test_tool_handlers import _channel_for, _thread_call

    flags = frozenset(s.requires for s in tr.registry().values() if s.requires is not None)
    args = {"capture_id": "0123456789abcdef", "package": "readings", "name": "row_sums"}

    async def main():
        out, rows = {}, []
        for agent in ("actor", "teammate", "helper", "librarian"):
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
    assert [r["refusal"]["form"] for r in rows if r.get("refusal")] == ["unknown_tool"] * 4
