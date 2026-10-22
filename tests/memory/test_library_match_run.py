"""``memory.match`` (recognition by content) and ``memory.run`` (try a stored procedure), as a cell runs them; the
index's ``input:`` line; the guide's look-first and try-first bullets; the pointer section; and the use record's
match, run and file-lookup evidence.

The library's item records are what production writes (:mod:`tests.memory.gate_records`): a pass lands the capture
of a procedure that sums the prices of priced rows as :data:`LANDED`, the library gate checks it, and the records are
``lifecycle.next_records`` with the gate's verification and the capture store's landing. Only a landed procedure has
an applicability signature, so it is the library item ``memory.match`` recognises. Building the records runs code
confined, so those tests need bubblewrap."""

from __future__ import annotations

import atexit
import copy
import functools
import json
import shutil
import tempfile
from pathlib import Path

import pytest

from tests.memory.gate_records import Built, Landing, build, needs_box
from tests.memory.test_episodes import _ep
from tests.memory.test_layout import LIB, _tree
from tests.memory.test_library_helper import _py
from unify.memory import library_export as lx
from unify.memory import memory_helper
from unify.memory.catalogue import write_files
from unify.memory.integration.adapters.audit import CellAudit
from unify.memory.integration.request import (
    add_runtime_calls,
    library_file_lookups,
    library_file_lookups_total,
    library_helper_calls,
    runtime_calls_empty,
)
from unify.memory.analysis import use
from unify.memory.episodes import Action

DATE = "memory.text.dates:parse_date"
TOKENS = "memory.text.parse:tokens"
#: The procedure the pass lands from its capture, as ``land_capture`` names it.
LANDED = "memory.prices.procedures:total_price"
ROWS = [{"price": 3, "name": "pen"}, {"price": 5, "name": "cup"}]
PLANTED = "alias-7f3a9c2e"  # an identifier the request printed: rev 2.1 keeps it wherever it is written


def _signature(value, relations=("output_within_input",)) -> dict:
    return {
        "input": memory_helper.value_shape(value),
        "output": memory_helper.value_shape({"total": 1}),
        "relations": list(relations),
        "constraints": {"applies": {"exists": True, "held": True}},
    }


SUMMARY = (
    f"## Asked\nSum the prices of a list of priced rows for {PLANTED}.\n"
    "## Approach\n- Added the price column (`cell:0` \"sum(r['price'] for r in rows)\").\n"
    "## Worked\n- the sum\n## Failed\nnone\n## Outcome\nnot shown\n"
)
PROCEDURE = '''"""Sum the prices of priced rows."""
import json


def read_input(text):
    """The JSON value the request starts with."""
    return json.JSONDecoder().raw_decode(text.strip())[0]


def run(value):
    """The total price of the rows."""
    return {"total": sum(r["price"] for r in value)}
'''
FAILING = '''def run(value):
    raise ValueError("no rows")
'''
TOUCHES = '''open("ran.txt", "w").write("ran")


def run(value):
    return value
'''


def _capture(root: Path, cid: str, *, procedure: str, signature: dict, outcome: str = "reproduces",
             broke: str = "none\n") -> Path:
    d = root / ".memory" / "captures" / cid
    d.mkdir(parents=True)
    (d / "procedure.py").write_text(procedure)
    (d / "broke.md").write_text(broke)
    (d / "signature.json").write_text(json.dumps(signature))
    (d / "record.json").write_text(
        json.dumps({"level": "unchecked", "status": "on-trial", "outcome": outcome}),
    )
    from unify.memory import captures

    # saved.json as the copy's export writes it (captures.export_files): what the capture was saved against, and
    # what broke, each line as written
    saved = {
        **captures.saved_against(procedure, SUMMARY),
        "broke": list(captures._broke_lines(broke)),
    }
    (d / "saved.json").write_bytes(captures._json(saved))
    return d


#: The library stored before the pass: the layout tests' library, its note without the link to a function it lacks
#: (the gate refuses a library with a dangling link).
BASE = {**LIB, "notes/text/dates.md": LIB["notes/text/dates.md"].replace(", memory.text.dates:gone", "")}


@functools.lru_cache(maxsize=1)
def _built() -> Built:
    """The pass that lands the capture of episode ``ep1`` (priced rows, as JSON text, whose total it handed back) as
    :data:`LANDED`, the gate's check and the item records; once per process."""
    root = Path(tempfile.mkdtemp(prefix="match-run-world-"))
    atexit.register(shutil.rmtree, root, True)
    request = json.dumps(ROWS)
    episode = _ep(
        episode_id="ep1",
        request=[request],
        transcript=[{"role": "user", "content": request}],
        cells=[],
        actions=[Action(-1, "dialogue:user", "answer", [], {"total": 8}, "Accepted.", "ok", "unknown", kind="dialogue")],
    )
    return build(
        root,
        episodes={"ep1": episode},
        base=BASE,
        landings=[Landing("ep1", PROCEDURE, SUMMARY, "prices", "total_price")],
    )


def _library(tmp_path: Path, *, effect: str | None = None) -> Path:
    """The library the gate checked, with :data:`LANDED` saved against priced rows (its record's signature, from the
    capture it was landed from), one capture that sums prices, one that failed, and one whose module would write a
    file if it were ever run. *effect*: the effect the gate derives from covers (none here), for the tests about
    it."""
    built = _built()
    root = _tree(tmp_path / "lib", built.files)
    records = copy.deepcopy(built.records)
    if effect is not None:
        records[LANDED]["effect"] = effect
    saved = lx.saved_against_of(records, lambda e: SUMMARY if e == "ep1" else None)
    write_files(root, lx.generated_library_mode(root, records=records, saved_against=saved))
    _capture(root, "a" * 16, procedure=PROCEDURE, signature=_signature(ROWS))
    _capture(root, "b" * 16, procedure=FAILING, signature=_signature(ROWS), outcome="failed",
             broke=f"- read the rows as text, not JSON (`cell:1` \"{PLANTED}\")\n")
    _capture(root, "c" * 16, procedure=TOUCHES, signature=_signature({"other": [1, 2]}))
    return root


# --- match -------------------------------------------------------------------------------------------------------


@needs_box
def test_match_lists_procedures_whose_recorded_input_has_the_same_structure(tmp_path):
    root = _library(tmp_path)
    code = (
        "import json, memory\n"
        f"r = memory.match(json.dumps({ROWS!r}))\n"
        "print(json.dumps([[x.id, x.matched, x.level, x.status] for x in r], sort_keys=True))\n"
        "print(json.dumps([x.id for x in r.failed]))\n"
        "print(json.dumps(repr(r)))\n"
    )
    rows, failed, text = (json.loads(line) for line in _py(root, code).splitlines())
    assert _built().records[LANDED]["level"] == "reproduces-its-case"  # the gate's, for a landing that reproduces
    # rev 2: the landed procedure's signature comes from its trace capture, which never runs read_input, so its input
    # is the request's data (the list of JSON values the request holds: here one, the rows), not the rows the reader
    # would read; the structural query of the same rows does not match it (KNOWN-ISSUES.md), while the capture saved
    # against the rows themselves does
    assert _built().records[LANDED]["signature"]["input"] == memory_helper.value_shape([ROWS])
    assert rows == [
        ["capture:" + "a" * 16, "exact", "unchecked", "on-trial"],
    ]
    assert failed == ["capture:" + "b" * 16]  # tried on input like this and failed: apart, never runnable
    assert "Tried on input like this and failed:" in text and "read the rows as text" in text
    assert "compares structure only" in text


@needs_box
def test_the_default_never_matches_by_words(tmp_path):
    """Text that shares every word with what a procedure was saved against, its docstring and its summary, but has
    another structure (or none), matches nothing."""
    root = _library(tmp_path)
    words = "Sum the prices of a list of priced rows: price name total rows"
    code = (
        "import json, memory\n"
        f"a = memory.match({words!r})\n"
        f"b = memory.match(json.dumps({words.split()!r}))\n"
        "c = memory.match(inputs={'price name': 'Sum the prices', 'rows': 'price'})\n"
        "print(json.dumps([[x.id for x in m] + [x.id for x in m.failed] for m in (a, b, c)]))\n"
        "print(json.dumps(a.note))\n"
    )
    out, note = (json.loads(line) for line in _py(root, code).splitlines())
    assert out == [[], [], []]
    assert "no structure" in note


@needs_box
def test_match_runs_nothing(tmp_path):
    root = _library(tmp_path)
    _py(root, "import json, memory\nmemory.match(inputs={'other': [1, 2]})\n")
    assert not (root / ".memory" / "captures" / ("c" * 16) / "ran.txt").exists()
    assert not (root / "ran.txt").exists()


@needs_box
def test_the_matcher_hook_can_be_replaced_and_receives_what_each_item_was_saved_against(tmp_path):
    root = _library(tmp_path)
    code = (
        "import json, memory\n"
        "seen = []\n"
        "def mine(query, candidates):\n"
        "    seen.append([query.request, [c._asdict() for c in candidates]])\n"
        "    return [(c.id, 'by-hand') for c in candidates if c.id.endswith('aaaa') or c.id.endswith('bbbb')]\n"
        "memory.MATCHER = mine\n"
        "r = memory.match('any text at all')\n"
        "print(json.dumps([[x.id, x.matched] for x in r] + [[x.id] for x in r.failed]))\n"
        "print(json.dumps(seen, sort_keys=True, default=list))\n"
    )
    rows, seen = (json.loads(line) for line in _py(root, code).splitlines())
    assert rows == [["capture:" + "a" * 16, "by-hand"], ["capture:" + "b" * 16]]
    (request, candidates), = seen
    assert request == "any text at all"
    by_id = {c["id"]: c for c in candidates}
    # the library's procedure is the landed one: the only library item whose record holds a signature
    assert set(by_id) == {LANDED, "capture:" + "a" * 16, "capture:" + "b" * 16, "capture:" + "c" * 16}
    lib = by_id[LANDED]
    assert lib["source"] == "library" and lib["signature"]["input"] == _built().records[LANDED]["signature"]["input"]
    assert lib["docstring"].startswith("The total price of the rows.")
    assert lib["asked"] == f"Sum the prices of a list of priced rows for {PLANTED}."
    assert lib["approach"].startswith("- Added the price column")
    cap = by_id["capture:" + "a" * 16]
    assert cap["docstring"] == "The total price of the rows." and cap["asked"] == lib["asked"]
    # what broke, as the copy's export saves it: each line of broke.md, its bullet kept, as written
    assert by_id["capture:" + "b" * 16]["broke"] == [f"- read the rows as text, not JSON (`cell:1` \"{PLANTED}\")"]
    # rev 2.1: the identifier reaches the matcher and the printed rows as written; nothing strips or masks it
    assert PLANTED in json.dumps(seen) and PLANTED in (root / ".memory" / "find.json").read_text()
    assert "<id>" not in json.dumps(seen) and "<id>" not in (root / ".memory" / "find.json").read_text()


def test_the_exported_texts_keep_every_identifier_as_written():
    """Rev 2.1: no identifier is stripped or masked; a summary's sections are exported as written."""
    text = "Sum the prices for task_00576224 in 3f2b9c1d-77aa-4b0e-9a1c-0d5e2f6a7b8c on 2026-10-22, invoice 100000"
    out = lx.summary_sections(f"## Asked\n{text}\n## Approach\n- x\n")["asked"]
    assert out == text


# --- run ---------------------------------------------------------------------------------------------------------


@needs_box
def test_run_calls_an_item_and_reports_its_outcome(tmp_path):
    root = _library(tmp_path)
    code = (
        "import json, memory\n"
        f"r = memory.run({LANDED!r}, {ROWS!r})\n"
        "bad = memory.run('memory.text.dates:nope', 1)\n"
        f"err = memory.run({TOKENS!r}, 5)\n"
        "print(json.dumps([r.outcome, r.result, r.level, r.status, hasattr(r, 'repeatable'), hasattr(r, 'agrees_with')]))\n"
        "print(json.dumps([bad.outcome, bad.error, err.outcome, err.error]))\n"
        "print(json.dumps(repr(r)))\n"
    )
    first, failures, text = (json.loads(line) for line in _py(root, code).splitlines())
    # the capture of the same work gives the same result on these rows
    # rev 2.1: the report is the one call: no repeat and no other procedure run beside it
    assert first == ["returned", {"total": 8}, "reproduces-its-case", "on-trial", False, False]
    assert failures == ["error", "LookupError", "error", "AttributeError"]  # reported, never raised
    assert f"run of {LANDED}: returned" in text and "repeatable" not in text and "agrees" not in text
    for word in ("correct", "verified", "right answer"):
        assert word not in text.lower()


@needs_box
@pytest.mark.parametrize("effect", [None, "write"])
def test_run_calls_a_library_procedure_once_whatever_its_effect(tmp_path, effect):
    """Rev 2.1, fix 4: rev 2 called an item a second time unless its recorded effect was ``write``; an item whose
    effect was never derived then acted twice. Now every item is called once."""
    root = _library(tmp_path, effect=effect)
    code = (
        "import json, sys, memory\n"
        "seen = []\n"
        "sys.addaudithook(lambda ev, a: seen.append([ev, list(a)]) if ev == 'memory.call' else None)\n"
        f"r = memory.run({LANDED!r}, {ROWS!r})\n"
        "print(json.dumps([r.outcome, [a[0] for _, a in seen]]))\n"
    )
    assert json.loads(_py(root, code)) == ["returned", [LANDED]]


@needs_box
def test_run_runs_a_capture_confined_and_reports_failures(tmp_path):
    root = _library(tmp_path)
    code = (
        "import json, memory\n"
        f"ok = memory.run('capture:' + 'a' * 16, json.dumps({ROWS!r}))\n"
        f"failed = memory.run('capture:' + 'b' * 16, json.dumps({ROWS!r}))\n"
        "missing = memory.run('capture:' + 'd' * 16, 'x')\n"
        "print(json.dumps([ok.outcome, ok.result, ok.conditions, ok.level, ok.status]))\n"
        "print(json.dumps([failed.outcome, failed.error, missing.outcome, missing.error]))\n"
    )
    ok, failures = (json.loads(line) for line in _py(root, code).splitlines())
    assert ok == ["returned", {"total": 8}, "none stated", "unchecked", "on-trial"]
    assert failures == ["error", "LookupError", "error", "LookupError"]


@needs_box
def test_a_capture_that_raises_is_reported_and_its_process_ends(tmp_path):
    root = _library(tmp_path)
    d = _capture(root, "e" * 16, procedure=FAILING, signature=_signature({"q": 1}))
    code = (
        "import json, memory\n"
        "r = memory.run('capture:' + 'e' * 16, {'q': 1})\n"
        "print(json.dumps([r.outcome, r.error, r.detail]))\n"
    )
    assert json.loads(_py(root, code)) == ["error", "ValueError", "no rows"]
    assert d.is_dir()


@needs_box
def test_match_and_run_raise_their_audit_events(tmp_path):
    root = _library(tmp_path)
    code = (
        "import json, sys, memory\n"
        "seen = []\n"
        "sys.addaudithook(lambda ev, a: seen.append([ev, list(a)]) if ev.startswith('memory.') else None)\n"
        f"memory.match(json.dumps({ROWS!r}))\n"
        f"memory.run('capture:' + 'a' * 16, json.dumps({ROWS!r}))\n"
        "print(json.dumps(seen, default=list))\n"
    )
    seen = json.loads(_py(root, code))
    helper = [(a[0], a[2]) for ev, a in seen if ev == "memory.helper_call"]
    assert helper == [("memory:match", "ok"), ("memory:run", "ok")]
    (rows, failed), = [a for ev, a in seen if ev == "memory.match_result"]
    # the landed procedure records no input structure in rev 2 (test_match_lists_procedures_...): only the capture
    assert [r[0] for r in rows] == ["capture:" + "a" * 16] and [r[0] for r in failed] == ["capture:" + "b" * 16]
    (run_args,) = [a for ev, a in seen if ev == "memory.run_result"]
    # rev 2.1: the run's event is its target, outcome, level and conditions (no repeat, no agreement)
    assert run_args == ["capture:" + "a" * 16, "returned", "unchecked", "none"]
    calls = [a for ev, a in seen if ev == "memory.call"]
    assert [c[0] for c in calls] == ["capture:" + "a" * 16] and calls[0][2] == "ok"
    assert all(PLANTED not in json.dumps(a) for _, a in seen)


# --- the evidence ------------------------------------------------------------------------------------------------


def _cell(audit: CellAudit, *events) -> dict:
    audit.begin()
    for ev, args in events:
        audit(ev, args)
    return audit.drain()


def test_match_and_run_are_counted_with_the_helpers_and_their_rows_kept():
    audit = CellAudit([])
    cap = "capture:" + "a" * 16
    first = _cell(
        audit,
        ("memory.helper_call", ("memory:match", "ab12", "ok")),
        ("memory.match_result", (((DATE, "unknown", "experimental", "exact"), (cap, "unchecked", "on-trial", "exact")),
                                 ())),
        ("memory.call", (DATE, "ab12", "ok")),
    )
    second = _cell(
        audit,
        ("memory.call", (cap, "ab12", "ok")),
        ("memory.run_result", (cap, "returned", "unchecked", "none")),
        ("memory.helper_call", ("memory:run", "ab12", "ok")),
        ("memory.run_result", ("not an id", "returned", "x", "none")),
    )
    assert second["calls"]["malformed"] == 1
    acc = runtime_calls_empty()
    add_runtime_calls(acc, first)
    add_runtime_calls(acc, second)
    counts = library_helper_calls(acc)
    assert counts["match"] == {"ok": 1, "refused": 0, "error": 0}
    assert counts["run"] == {"ok": 1, "refused": 0, "error": 0}
    lines = [
        {"type": "message", "message": {"role": "user", "content": "request"}},
        *[
            row
            for i in range(2)
            for row in (
                {"type": "message", "message": {"role": "assistant", "tool_calls": [
                    {"id": f"c{i}", "function": {"name": "execute_code", "arguments": "{}"}}]}},
                {"type": "message", "message": {"role": "tool", "tool_call_id": f"c{i}", "content": ""}},
            )
        ],
        {"type": "message", "message": {"role": "user", "content": "more"}},
    ]
    fields = use.match_run_fields(acc, lines)
    assert fields["match_calls"] == 1 and fields["run_calls"] == 1
    (row,) = fields["match_rows"]
    assert [r["id"] for r in row["listed"]] == [DATE, cap]
    assert row["called_after"] == [cap, DATE]  # the import-style call in its cell, the run in the next
    (run_row,) = fields["run_rows"]
    assert run_row == {
        "cell": 1,
        "target": cap,
        "level": "unchecked",
        "outcome": "returned",
        "conditions": "none",
        "listed_by_match": True,
        "listed_by_ask": False,
    }
    assert fields["ran_before_more_input"] is True and fields["run_first_share"] == 1.0
    late = use.match_run_fields(acc, lines[:3] + lines[-1:] + lines[3:5])
    assert late["ran_before_more_input"] is False and late["run_first_share"] == 0.0
    assert use.match_run_fields(runtime_calls_empty(), lines)["run_first_share"] is None


def test_file_reads_and_searches_of_the_library_copy_are_counted_apart_from_helpers():
    from unify.sandbox import LIBRARY_MOUNT

    lib = LIBRARY_MOUNT
    records = [
        {"event": "open", "path": f"{lib}/memory/text/dates.py", "mode": "r"},
        {"event": "open", "path": f"{lib}/INDEX.md", "mode": "rb"},
        {"event": "open", "path": f"{lib}/memory/text/parse.py", "mode": "r", "via": "import"},
        {"event": "open", "path": f"{lib}/.memory/items.json", "mode": "r", "via": "library"},
        {"event": "open", "path": "/workspace/notes.txt", "mode": "r"},
        {"event": "open", "path": f"{lib}/x.txt", "mode": "w"},
        {"event": "subprocess.Popen", "exe": "/bin/grep", "argv": ["grep", "-rn", "total", f"{lib}/memory"]},
        {"event": "subprocess.Popen", "exe": "/bin/sh", "argv": ["/bin/sh", "-c", f"rg price {lib}"]},
        {"event": "os.system", "exe": None, "argv": [f"git grep -n price -- {lib}/notes"]},
        {"event": "subprocess.Popen", "exe": "/bin/grep", "argv": ["grep", "x", "/workspace"]},
        {"event": "subprocess.Popen", "exe": "/bin/cat", "argv": ["cat", f"{lib}/INDEX.md"]},
    ]
    assert library_file_lookups(records, [lib]) == {"reads": 2, "searches": 3}
    acc = runtime_calls_empty()
    add_runtime_calls(acc, {"records": records}, [lib])
    add_runtime_calls(acc, {"records": records[:1]}, [lib])
    assert library_file_lookups_total(acc) == {"reads": 3, "searches": 3}
    assert library_file_lookups_total(runtime_calls_empty()) == {"reads": 0, "searches": 0}


def test_the_audit_hook_marks_opens_by_the_import_system_and_the_library(tmp_path):
    lib = tmp_path / "lib"
    (lib / "pkgx").mkdir(parents=True)
    (lib / "pkgx" / "__init__.py").write_text("VALUE = 1\n")
    (lib / "data.txt").write_text("x\n")
    from unify.memory.integration.adapters import audit as audit_module

    code = (
        "import importlib.util, json, sys\n"
        f"spec = importlib.util.spec_from_file_location('cell_audit', {audit_module.__file__!r})\n"
        "A = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(A)\n"
        f"hook = A.install([{str(lib)!r}])\n"
        "hook.begin()\n"
        f"sys.path.insert(0, {str(lib)!r})\n"
        "import pkgx\n"
        f"open({str(lib / 'data.txt')!r}).read()\n"
        "out = hook.drain()\n"
        "print(json.dumps([[r.get('path', '').rsplit('/', 1)[-1], r.get('via')] for r in out['records']"
        " if r['event'] == 'open']))\n"
    )
    import subprocess
    import sys

    got = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, timeout=120)
    assert got.returncode == 0, got.stderr
    opens = json.loads(got.stdout)
    assert ["data.txt", None] in opens
    assert ["__init__.py", "import"] in opens


# --- the index line, the guide and the pointer section -----------------------------------------------------------


