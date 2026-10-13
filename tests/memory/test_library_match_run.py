"""``memory.match`` (recognition by content) and ``memory.run`` (try a stored procedure), as a cell runs them; the
index's ``input:`` line; the guide's look-first and try-first bullets; the pointer section; and the use record's
match, run and file-lookup evidence."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.memory.test_layout import LIB, _tree
from tests.memory.test_library_helper import _py
from unify.memory import library_export as lx
from unify.memory import library_prompts as lp
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

DATE = "memory.text.dates:parse_date"
TOKENS = "memory.text.parse:tokens"
ROWS = [{"price": 3, "name": "pen"}, {"price": 5, "name": "cup"}]
PLANTED = "alias-7f3a9c2e"  # a printed identifier the stand-in rule removes


def _strip(text: str) -> str:
    """A stand-in for the one identifier rule (``signatures.strip_identifiers``, the capture store's): it removes
    the planted identifier only, so these tests check the plumbing, not the rule."""
    return text.replace(PLANTED, "<id>")


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
    return json.loads(text)


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
    (d / "saved.json").write_bytes(lx.capture_saved_bytes(d, SUMMARY, _strip))
    return d


def _library(tmp_path: Path, *, effect: str = "read") -> Path:
    """The fixture library with ``parse_date`` saved against priced rows, one capture that sums prices, one that
    failed, and one whose module would write a file if it were ever run."""
    from unify.memory.item_records import empty_record

    root = _tree(tmp_path / "lib")
    records = {
        DATE: {
            **empty_record(DATE, "function"),
            "signature": _signature(ROWS),
            "level": "reproduces-its-case",
            "provenance": {"episodes": ["ep1"]},
            "effect": effect,
        },
    }
    saved = lx.saved_against_of(records, lambda e: SUMMARY if e == "ep1" else None, _strip)
    write_files(root, lx.generated_library_mode(root, records=records, saved_against=saved, strip=_strip))
    _capture(root, "a" * 16, procedure=PROCEDURE, signature=_signature(ROWS))
    _capture(root, "b" * 16, procedure=FAILING, signature=_signature(ROWS), outcome="failed",
             broke=f"- read the rows as text, not JSON (`cell:1` \"{PLANTED}\")\n")
    _capture(root, "c" * 16, procedure=TOUCHES, signature=_signature({"other": [1, 2]}))
    return root


# --- match -------------------------------------------------------------------------------------------------------


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
    assert rows == [
        [DATE, "exact", "reproduces-its-case", "on-trial"],
        ["capture:" + "a" * 16, "exact", "unchecked", "on-trial"],
    ]
    assert failed == ["capture:" + "b" * 16]  # tried on input like this and failed: apart, never runnable
    assert "Tried on input like this and failed:" in text and "read the rows as text" in text
    assert "compares structure only" in text


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


def test_match_runs_nothing(tmp_path):
    root = _library(tmp_path)
    _py(root, "import json, memory\nmemory.match(inputs={'other': [1, 2]})\n")
    assert not (root / ".memory" / "captures" / ("c" * 16) / "ran.txt").exists()
    assert not (root / "ran.txt").exists()


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
    assert set(by_id) == {DATE, "capture:" + "a" * 16, "capture:" + "b" * 16, "capture:" + "c" * 16}
    lib = by_id[DATE]
    assert lib["source"] == "library" and lib["signature"]["input"] == memory_helper.value_shape(ROWS)
    assert lib["docstring"].startswith("Parse a date written as YYYY-MM-DD.")
    assert lib["asked"] == "Sum the prices of a list of priced rows for <id>."
    assert lib["approach"].startswith("- Added the price column")
    cap = by_id["capture:" + "a" * 16]
    assert cap["docstring"] == "The total price of the rows." and cap["asked"] == lib["asked"]
    assert by_id["capture:" + "b" * 16]["broke"] == ["read the rows as text, not JSON (`cell:1` \"<id>\")"]
    # no identifier reaches the matcher or the printed rows
    assert PLANTED not in json.dumps(seen) and PLANTED not in (root / ".memory" / "find.json").read_text()


def test_the_exported_texts_use_the_one_identifier_rule_when_it_is_built():
    signatures = pytest.importorskip("unify.memory.signatures")
    text = "Sum the prices for task_00576224 in 3f2b9c1d-77aa-4b0e-9a1c-0d5e2f6a7b8c"
    out = lx.summary_sections(f"## Asked\n{text}\n## Approach\n- x\n")["asked"]
    assert out == signatures.strip_identifiers(text)


# --- run ---------------------------------------------------------------------------------------------------------


def test_run_calls_an_item_and_reports_its_outcome(tmp_path):
    root = _library(tmp_path)
    code = (
        "import json, memory\n"
        f"r = memory.run({DATE!r}, '2026-10-09')\n"
        "bad = memory.run('memory.text.dates:nope', 1)\n"
        f"err = memory.run({TOKENS!r}, 5)\n"
        "print(json.dumps([r.outcome, r.result, r.repeatable, r.level, r.status, r.agrees_with]))\n"
        "print(json.dumps([bad.outcome, bad.error, err.outcome, err.error]))\n"
        "print(json.dumps(repr(r)))\n"
    )
    first, failures, text = (json.loads(line) for line in _py(root, code).splitlines())
    assert first == ["returned", "2026-10-09", True, "reproduces-its-case", "on-trial", []]
    assert failures == ["error", "LookupError", "error", "AttributeError"]  # reported, never raised
    assert "run of memory.text.dates:parse_date: returned" in text and "repeatable: yes" in text
    for word in ("correct", "verified", "right answer"):
        assert word not in text.lower()


def test_run_never_repeats_a_procedure_whose_effect_is_write(tmp_path):
    root = _library(tmp_path, effect="write")
    code = (
        "import json, memory\n"
        f"r = memory.run({DATE!r}, '2026-10-09')\n"
        "print(json.dumps([r.outcome, r.repeatable, r.repeat_note]))\n"
    )
    assert json.loads(_py(root, code)) == [
        "returned",
        None,
        "a procedure whose recorded effect is write is called once",
    ]


def test_run_runs_a_capture_confined_and_reports_failures(tmp_path):
    root = _library(tmp_path)
    code = (
        "import json, memory\n"
        f"ok = memory.run('capture:' + 'a' * 16, json.dumps({ROWS!r}))\n"
        f"failed = memory.run('capture:' + 'b' * 16, json.dumps({ROWS!r}))\n"
        "missing = memory.run('capture:' + 'd' * 16, 'x')\n"
        "print(json.dumps([ok.outcome, ok.result, ok.repeatable, ok.conditions, ok.level, ok.status]))\n"
        "print(json.dumps([failed.outcome, failed.error, missing.outcome, missing.error]))\n"
    )
    ok, failures = (json.loads(line) for line in _py(root, code).splitlines())
    assert ok == ["returned", {"total": 8}, True, "none stated", "unchecked", "on-trial"]
    assert failures == ["error", "LookupError", "error", "LookupError"]


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


def test_agrees_with_names_other_matching_procedures_with_the_same_result(tmp_path):
    root = _library(tmp_path)
    _capture(root, "f" * 16, procedure=PROCEDURE, signature=_signature(ROWS))
    code = (
        "import json, memory\n"
        f"r = memory.run('capture:' + 'a' * 16, json.dumps({ROWS!r}))\n"
        "print(json.dumps([r.agrees_with, r.agreement_note]))\n"
    )
    agrees, note = json.loads(_py(root, code))
    assert agrees == ["capture:" + "f" * 16] and note.startswith("checked against")


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
    assert [r[0] for r in rows] == [DATE, "capture:" + "a" * 16] and [r[0] for r in failed] == ["capture:" + "b" * 16]
    (run_args,) = [a for ev, a in seen if ev == "memory.run_result"]
    assert run_args[:3] == ["capture:" + "a" * 16, "returned", "yes"]
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
        ("memory.run_result", (cap, "returned", "yes", "unchecked", "none", ())),
        ("memory.helper_call", ("memory:run", "ab12", "ok")),
        ("memory.run_result", ("not an id", "returned", "yes", "x", "none", ())),
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
        "repeatable": "yes",
        "agrees_with": 0,
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


def test_the_index_line_shows_the_signature_compactly_without_identifiers(tmp_path):
    from unify.memory import layout
    from unify.memory.library_index import INDEX_HEAD, INDEX_HEAD_IN_CODE, build_links, render_index

    lib = layout.discover(_tree(tmp_path))
    links = build_links(lib)
    sig = _signature([{"price": 3, "<id>": "pen"}])
    index = render_index(lib, links, None, None, None, None, {DATE: sig}.get)
    paths = memory_helper.signature(sig["input"])
    line = (
        "  input: {" + ", ".join(f"{k}: {v}" for k, v in sorted(paths.items()))
        + "}; relations: output_within_input\n"
    )
    assert "[].price: int" in line and "[].<id>" in line
    assert line in index and index.count("  input: ") == 1
    assert render_index(lib, links, None, None, None, None, {DATE: sig}.get) == index  # byte-stable
    assert "`input:` line" in INDEX_HEAD and "`input:` line" in INDEX_HEAD_IN_CODE
    # no signature: no line
    assert "  input: " not in render_index(lib, links, None, None, None, None, lambda i: None)


def test_the_guides_nudge_to_match_and_try_first_and_never_mention_examples():
    texts = {
        "guide": lp.GUIDE_LIBRARY,
        "guide in code": lp.GUIDE_LIBRARY_IN_CODE,
        "pointer": lp.GUIDE_POINTER,
        "pointer in code": lp.GUIDE_POINTER_IN_CODE,
    }
    for name, text in texts.items():
        assert "At the start of a request, `memory.match(request)`" in text, name
        assert "pass any data you were given as `inputs=` too" in text, name
        assert "memory.match(request, inputs)" not in text, name
        assert "`memory.run(\"<id>\", ...)`" in text, name
        assert "Trying a matching stored procedure costs one call, and rediscovering it usually costs many." in text
        assert "It compares structure only" in text and "it runs nothing" in text, name
        assert "example" not in text.lower().replace("e.g.", ""), name
        assert lp.example_checks(text) == [] and lp.benchmark_words(text) == [], name
        assert "stored function" not in text.lower(), name
    for name in ("pointer", "pointer in code"):
        assert lp.POINTER_BOUND in texts[name]
        assert "already available in your cells as `memory`, no import needed" in texts[name]


def test_the_pointer_section_is_the_same_bytes_as_the_library_grows(tmp_path, monkeypatch):
    from unify.memory.integration import prompt
    from unify.settings import SETTINGS

    small = _tree(tmp_path / "small", {k: v for k, v in LIB.items() if not k.startswith("memory/web")})
    big = _tree(tmp_path / "big")
    (big / "memory/more").mkdir(parents=True)
    for i in range(30):
        (big / f"memory/more/m{i}.py").write_text(f'def f{i}(x):\n    """Item {i}."""\n    return x\n')
    for root in (small, big):
        write_files(root, lx.generated_library_mode(root))
    assert (small / "INDEX.md").read_text() != (big / "INDEX.md").read_text()
    for mode in ("off", "on"):
        monkeypatch.setattr(SETTINGS, "UNIFY_TOOLS_IN_CODE", mode)
        a = prompt.render_memory_library_mode(small, actor_index="pointer")[0]
        b = prompt.render_memory_library_mode(big, actor_index="pointer")[0]
        assert a == b
        assert a.startswith(lp.GUIDE_POINTER_IN_CODE if mode == "on" else lp.GUIDE_POINTER)
