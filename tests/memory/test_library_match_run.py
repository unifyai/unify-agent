"""An item's own check (``item.fits``, recognition by structure) and its call (``item(...)``, try a stored
procedure), as a cell runs them; the index's ``input:`` line; the guide's look-first and try-first bullets; the
pointer section; and the use record's match, run and file-lookup evidence.

The library's item records are what production writes (:mod:`tests.memory.gate_records`): a pass lands the capture
of a procedure that sums the prices of priced rows as :data:`LANDED`, the library gate checks it, and the records are
``lifecycle.next_records`` with the gate's verification and the capture store's landing. Only a landed procedure has
an applicability signature, so it is the library item a check can recognise. Building the records runs code
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
PLANTED = "alias-7f3a9c2e"  # an identifier the request printed: it is kept wherever it is written


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


# --- an item's own check -----------------------------------------------------------------------------------------

A, B, C = ("capture:" + x * 16 for x in "abc")


@needs_box
def test_a_procedure_fits_when_its_recorded_input_has_the_same_structure(tmp_path):
    root = _library(tmp_path)
    code = (
        "import json, memory\n"
        f"v = json.dumps({ROWS!r})\n"
        f"fits = [memory.get(i).fits(v) for i in {[LANDED, A, B, C]!r}]\n"
        "print(json.dumps([[f.ok, f.why] for f in fits]))\n"
        f"print(json.dumps(str(memory.get({B!r}))))\n"
    )
    fits, failed_card = (json.loads(line) for line in _py(root, code).splitlines())
    assert _built().records[LANDED]["level"] == "unchecked"  # the gate runs no code, so it checks no level
    # the landed procedure's signature comes from its trace capture, which never runs read_input, so its input is the
    # request's data (the list of JSON values the request holds: here one, the rows), not the rows the reader would
    # read; the structural check of the same rows does not fit it (KNOWN-ISSUES.md), while the capture saved against
    # the rows themselves does
    assert _built().records[LANDED]["signature"]["input"] == memory_helper.value_shape([ROWS])
    (landed, landed_why), (a, a_why), (b, b_why), (c, c_why) = fits
    assert landed is False and landed_why.startswith("the input it was saved against has a different structure")
    assert a is True and "(exact)" in a_why and "A structural match can be wrong." in a_why
    # tried on input like this and failed: never offered to run, and its card shows what broke
    assert b is False and b_why.startswith("it was tried on input like this and failed")
    assert "Tried and failed:" in failed_card and "read the rows as text" in failed_card
    assert c is False


@needs_box
def test_the_check_never_fits_by_words(tmp_path):
    """Text that shares every word with what a procedure was saved against, its docstring and its summary, but has
    another structure (or none), fits nothing."""
    root = _library(tmp_path)
    words = "Sum the prices of a list of priced rows: price name total rows"
    code = (
        "import json, memory\n"
        f"inputs = [{words!r}, json.dumps({words.split()!r}), {{'price name': 'Sum the prices', 'rows': 'price'}}]\n"
        f"print(json.dumps([[memory.get(i).fits(x).ok for x in inputs] for i in {[LANDED, A, B, C]!r}]))\n"
        f"print(json.dumps(memory.get({A!r}).fits({words!r}).why))\n"
    )
    out, why = (json.loads(line) for line in _py(root, code).splitlines())
    assert out == [[False, False, False]] * 4
    assert "no structure" in why


@needs_box
def test_the_check_runs_nothing(tmp_path):
    root = _library(tmp_path)
    code = f"import json, memory\nprint(json.dumps(memory.get({C!r}).fits({{'other': [1, 2]}}).ok))\n"
    fitted = json.loads(_py(root, code))
    assert fitted is True  # it fits, and still nothing of it ran
    assert not (root / ".memory" / "captures" / ("c" * 16) / "ran.txt").exists()
    assert not (root / "ran.txt").exists()


@needs_box
def test_each_candidate_carries_what_it_was_saved_against(tmp_path):
    root = _library(tmp_path)
    code = (
        "import json, memory\n"
        "candidates, unreadable = memory._candidates_all()\n"
        "print(json.dumps([[c._asdict() for c in candidates], unreadable], sort_keys=True, default=list))\n"
    )
    candidates, unreadable = json.loads(_py(root, code))
    assert unreadable == 0
    by_id = {c["id"]: c for c in candidates}
    # the library's procedure is the landed one: the only library item whose record holds a signature
    assert set(by_id) == {LANDED, A, B, C}
    lib = by_id[LANDED]
    assert lib["source"] == "library" and lib["signature"]["input"] == _built().records[LANDED]["signature"]["input"]
    assert lib["docstring"].startswith("The total price of the rows.")
    assert lib["asked"] == f"Sum the prices of a list of priced rows for {PLANTED}."
    assert lib["approach"].startswith("- Added the price column")
    cap = by_id[A]
    assert cap["docstring"] == "The total price of the rows." and cap["asked"] == lib["asked"]
    # what broke, as the copy's export saves it: each line of broke.md, its bullet kept, as written
    assert by_id[B]["broke"] == [f"- read the rows as text, not JSON (`cell:1` \"{PLANTED}\")"]
    # the identifier reaches the candidates as written; nothing strips or masks it
    assert PLANTED in json.dumps(candidates) and PLANTED in (root / ".memory" / "find.json").read_text()
    assert "<id>" not in json.dumps(candidates) and "<id>" not in (root / ".memory" / "find.json").read_text()


def test_the_exported_texts_keep_every_identifier_as_written():
    """No identifier is stripped or masked; a summary's sections are exported as written."""
    text = "Sum the prices for task_00576224 in 3f2b9c1d-77aa-4b0e-9a1c-0d5e2f6a7b8c on 2026-10-22, invoice 100000"
    out = lx.summary_sections(f"## Asked\n{text}\n## Approach\n- x\n")["asked"]
    assert out == text


# --- an item's call and the tracked run under it -----------------------------------------------------------------


@needs_box
def test_the_tracked_run_calls_an_item_and_reports_its_outcome(tmp_path):
    root = _library(tmp_path)
    code = (
        "import json, memory\n"
        f"r = memory._tracked_run({LANDED!r}, ({ROWS!r},), {{}})\n"
        "bad = memory._tracked_run('memory.text.dates:nope', (1,), {})\n"
        f"err = memory._tracked_run({TOKENS!r}, (5,), {{}})\n"
        "print(json.dumps([r.outcome, r.result, r.level, r.status, hasattr(r, 'repeatable'), hasattr(r, 'agrees_with')]))\n"
        "print(json.dumps([bad.outcome, bad.error, err.outcome, err.error]))\n"
        "print(json.dumps(repr(r)))\n"
        f"print(json.dumps(memory.get({LANDED!r})({ROWS!r})))\n"
    )
    first, failures, text, called = (json.loads(line) for line in _py(root, code).splitlines())
    # the capture of the same work gives the same result on these rows
    # the report is the one call: no repeat and no other procedure run beside it
    assert first == ["returned", {"total": 8}, "unchecked", "on-trial", False, False]
    assert failures == ["error", "LookupError", "error", "AttributeError"]  # reported, never raised
    assert f"run of {LANDED}: returned" in text and "repeatable" not in text and "agrees" not in text
    for word in ("correct", "verified", "right answer"):
        assert word not in text.lower()
    assert called == {"total": 8}  # the item's call returns what the function returned


@needs_box
@pytest.mark.parametrize("effect", [None, "write"])
def test_a_call_runs_a_library_procedure_once_whatever_its_effect(tmp_path, effect):
    """An item was once called a second time unless its recorded effect was ``write``; an item whose effect was never
    derived then acted twice. Now every item is called once."""
    root = _library(tmp_path, effect=effect)
    code = (
        "import json, sys, memory\n"
        "seen = []\n"
        "sys.addaudithook(lambda ev, a: seen.append([ev, list(a)]) if ev == 'memory.call' else None)\n"
        f"r = memory.get({LANDED!r})({ROWS!r})\n"
        "print(json.dumps([r, [a[0] for _, a in seen]]))\n"
    )
    assert json.loads(_py(root, code)) == [{"total": 8}, [LANDED]]


@needs_box
def test_a_capture_runs_confined_and_its_failures_are_reported(tmp_path):
    root = _library(tmp_path)
    code = (
        "import json, memory\n"
        f"ok = memory._tracked_run('capture:' + 'a' * 16, (json.dumps({ROWS!r}),), {{}})\n"
        f"failed = memory._tracked_run('capture:' + 'b' * 16, (json.dumps({ROWS!r}),), {{}})\n"
        "missing = memory._tracked_run('capture:' + 'd' * 16, ('x',), {})\n"
        "print(json.dumps([ok.outcome, ok.result, ok.conditions, ok.level, ok.status]))\n"
        "print(json.dumps([failed.outcome, failed.error, missing.outcome, missing.error]))\n"
        f"called = memory.get('capture:' + 'a' * 16)(json.dumps({ROWS!r}))\n"
        "try:\n"
        f"    memory.get('capture:' + 'b' * 16)(json.dumps({ROWS!r}))\n"
        "    raised = None\n"
        "except Exception as exc:\n"
        "    raised = [type(exc).__name__, str(exc).splitlines()[0]]\n"
        "print(json.dumps([called, raised]))\n"
    )
    ok, failures, (called, raised) = (json.loads(line) for line in _py(root, code).splitlines())
    assert ok == ["returned", {"total": 8}, "none stated", "unchecked", "on-trial"]
    assert failures == ["error", "LookupError", "error", "LookupError"]
    # the item's call returns the result, and raises the failed attempt's refusal to run
    assert called == {"total": 8}
    assert raised[0] == "ArgumentError" and "this captured attempt failed" in raised[1]


@needs_box
def test_a_capture_that_raises_is_reported_and_its_process_ends(tmp_path):
    root = _library(tmp_path)
    d = _capture(root, "e" * 16, procedure=FAILING, signature=_signature({"q": 1}))
    code = (
        "import json, memory\n"
        "r = memory._tracked_run('capture:' + 'e' * 16, ({'q': 1},), {})\n"
        "print(json.dumps([r.outcome, r.error, r.detail]))\n"
    )
    assert json.loads(_py(root, code)) == ["error", "ValueError", "no rows"]
    assert d.is_dir()


@needs_box
def test_a_check_and_a_call_raise_their_audit_events(tmp_path):
    root = _library(tmp_path)
    code = (
        "import json, sys, memory\n"
        "seen = []\n"
        "sys.addaudithook(lambda ev, a: seen.append([ev, list(a)]) if ev.startswith('memory.') else None)\n"
        f"a, b, v = memory.get({A!r}), memory.get({B!r}), json.dumps({ROWS!r})\n"
        "a.fits(v)\n"
        "b.fits(v)\n"
        "a(v)\n"
        "print(json.dumps(seen, default=list))\n"
    )
    seen = json.loads(_py(root, code))
    helper = [(a[0], a[2]) for ev, a in seen if ev == "memory.helper_call"]
    assert helper == [
        ("memory:get", "ok"),
        ("memory:get", "ok"),
        ("memory:match", "ok"),
        ("memory:match", "ok"),
        ("memory:run", "ok"),
    ]
    (rows_a, failed_a), (rows_b, failed_b) = [a for ev, a in seen if ev == "memory.match_result"]
    # each check's event names its own item: the capture that fits, and the failed attempt apart
    assert [r[0] for r in rows_a] == [A] and failed_a == []
    assert rows_b == [] and [r[0] for r in failed_b] == [B]
    (run_args,) = [a for ev, a in seen if ev == "memory.run_result"]
    # the run's event is its target, outcome, level and conditions (no repeat, no agreement)
    assert run_args == [A, "returned", "unchecked", "none"]
    calls = [a for ev, a in seen if ev == "memory.call"]
    assert [c[0] for c in calls] == [A] and calls[0][2] == "ok"
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


