"""``memory.match`` and ``memory.run`` in the generated library package, run as a cell runs them.

``memory.match(request=None, inputs=None)`` is a seam: it tells the agent which stored procedures and captures
fit what it holds, through a pluggable matching hook whose default compares structure only. It never matches by
words and never runs stored code; what it shows of a stored procedure is as it was recorded, identifiers included
(rev 2.1: none is stripped or masked).

``memory.run(item, *args, **kwargs)`` runs one stored procedure (a library function, or a captured procedure) on
the agent's inputs in the caller's process and reports how the run went: its outcome, its result, whether its
stated conditions held. It calls the procedure once: never a second time and never another procedure beside it
(rev 2.1). A failing procedure is reported, never raised past the cell.

Each test runs the generated package in a fresh isolated interpreter with the copy first on its path, as the
actor's cells import it. No model is called.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

from tests.memory.test_layout import LIB, _tree
from unify.memory import library_export as lx
from unify.memory import memory_helper
from unify.memory.catalogue import write_files

# ==== ASSUMED NAMES (the integrator adapts these to the code, here only) ======================================
#: Where the actor's copy of the library holds the capture store's snapshot (design I.2.6).
CAPTURES_DIR = ".memory/captures"
#: The files of one capture (design I.2.3-I.2.5), and the source episode's summary beside them (H.4, J.3).
CAPTURE_PROCEDURE = "procedure.py"
CAPTURE_SIGNATURE = "signature.json"
CAPTURE_RECORD = "record.json"
CAPTURE_SUMMARY = "summary.md"
#: How ``memory.run`` and ``memory.match`` name a capture (design I.4.1).
CAPTURE_TARGET = "capture:{id}"
#: The audit event and ids of the package's own helpers (``library_helper.HELPER_EVENT``; design J.3 / lane).
HELPER_EVENT = "memory.helper_call"
CALL_EVENT = "memory.call"
MATCH_ID = "memory:match"
RUN_ID = "memory:run"
#: The renamed evidence level and status of a new capture (design I.11 decision 3).
NEW_LEVEL = "unchecked"
NEW_STATUS = "on-trial"
#: The report's outcomes (design I.5.1).
RETURNED, REFUSED, ERROR = "returned", "refused", "error"
#: Code run first in every cell of these tests: how a test reads a report's field (an attribute, else a key),
#: how it installs a matching hook (the package's ``MATCHER``, called as ``MATCHER(query, candidates)`` with a
#: ``Query`` and ``Candidate`` tuples and returning ``(id, how matched)`` pairs; ``set_hook`` adapts a test's
#: ``hook(request, inputs, candidates)``, given one mapping per candidate, that returns the candidates that fit),
#: and the audit hook that keeps the package's events.
PRELUDE = """
import json, threading
events = []
sys.addaudithook(lambda e, a: events.append([e, *[str(x) for x in a]]) if e in ("memory.call", "memory.helper_call") else None)
import memory


def field(report, name):
    if hasattr(report, name):
        return getattr(report, name)
    try:
        return report[name]
    except Exception:
        return None


def text(value):
    if isinstance(value, (list, tuple)):
        return "\\n".join(str(v) + " " + repr(v) for v in value)
    return str(value) + "\\n" + repr(value)


def set_hook(fn):
    def matcher(query, candidates):
        chosen = fn(query.request, query.inputs, [c._asdict() for c in candidates])
        return [(c["id"], "hook") for c in (chosen or [])]

    memory.MATCHER = matcher


def plain(value):
    try:
        json.dumps(value)
        return value
    except TypeError:
        return repr(value)
"""
#: What a capture was saved against besides its signature, in the actor's copy (``captures.SAVED_FILE``).
CAPTURE_SAVED = "saved.json"


def export_capture_files(d: Path, signature, summary: str) -> None:
    """Write a capture's signature and what it was saved against as the harness exports them into the actor's copy
    (``captures.export_files``): the signature as the store built it, and ``saved.json`` with the procedure's
    docstring and the summary's Asked and Approach (``captures.saved_against``, as the capture step writes it). Rev
    2.1: nothing strips an identifier, so all of them are as written."""
    from unify.memory import captures

    if isinstance(signature, str):  # a malformed file, as written
        (d / CAPTURE_SIGNATURE).write_text(signature)
    else:
        (d / CAPTURE_SIGNATURE).write_text(json.dumps(signature))
    (d / CAPTURE_SUMMARY).write_text(summary)
    proc = d / "procedure.py"
    source = proc.read_text() if proc.is_file() else ""
    (d / CAPTURE_SAVED).write_bytes(captures._json(captures.saved_against(source, summary)))


# ==============================================================================================================

#: A request in plain prose with a table of records in it, and a printed identifier (rev 2.1: kept as written
#: wherever a stored procedure recorded it, never a placeholder).
ALIAS = "3f2c8a10-9b7e-4c51-a2d4-6e0f1b9c7d35"
ALIAS_SHORT = "TASK-4471"
ROWS = [{"item": "pen", "amount": 3}, {"item": "ink", "amount": 4}]
REQUEST = (
    f"Request {ALIAS} ({ALIAS_SHORT}).\n"
    "Add up the amount of every item in the order table below and report the total.\n"
    "TABLE\n" + json.dumps(ROWS) + "\n"
)
#: The same words as the request's goal and the procedure's docstring, with no table: word overlap only.
WORDS_ONLY = (
    "Add up the amount of every item in the order table and report the total. "
    "Sum the amount column of an order table of items."
)
GRID = [[1, 2, 3], [4, 5, 6]]

READ_TABLE = '''
def read_input(text):
    """The order table that follows the line TABLE in the request."""
    return json.loads(text.split("TABLE\\n", 1)[1].splitlines()[0])
'''


def _procedure(body: str, mark: str | None = None, doc: str = "Sum the amount column of an order table.") -> str:
    """A captured ``procedure.py``. With *mark*, importing it or calling any of its functions appends to *mark*."""
    lines = [f'"""{doc}"""', "", "import json", ""]
    if mark is not None:
        lines += [f"open({mark!r}, 'a').write('imported\\n')", ""]
        body = body.replace("    return", f"    open({mark!r}, 'a').write('called\\n')\n    return")
    return "\n".join(lines) + body


SUM_TABLE = READ_TABLE + '''

def applies(value):
    """A list of records, each with a whole-number amount."""
    return isinstance(value, list) and all(isinstance(r, dict) and isinstance(r.get("amount"), int) for r in value)


def run(value):
    """Sum the amount column of an order table of items."""
    return sum(r["amount"] for r in value)


def holds(value, result):
    """The total is a whole number."""
    return isinstance(result, int)
'''
TRANSPOSE = '''
def read_input(text):
    """The grid that follows the line GRID in the request."""
    return json.loads(text.split("GRID\\n", 1)[1].splitlines()[0])


def run(value):
    """Turn a grid's rows into its columns."""
    return [list(c) for c in zip(*value)]
'''


def _capture_id(source: str) -> str:
    """As the capture step names a capture (design I.2.3): the first 16 hex digits of the sha256 of its code."""
    return hashlib.sha256(source.encode("utf-8")).hexdigest()[:16]


def _signature(value, result, goal: str = "Add up the amount of every item and report the total.") -> dict:
    return {
        "input": memory_helper.value_shape(value),
        "output": memory_helper.value_shape(result) if isinstance(result, (list, dict)) else {"kind": "scalar"},
        "relations": ["scalar_output"] if not isinstance(result, (list, dict)) else [],
        "constraints": {},
        "goal": goal,
    }


def _record(**kw) -> dict:
    return {
        "episodes": ["episode-a"],
        "reproduces": "reproduces",
        "deterministic": True,
        "applies": True,
        "holds": True,
        "outcome": "none",
        "level": NEW_LEVEL,
        "status": NEW_STATUS,
        "track": {},
        **kw,
    }


SUMMARY = (
    "## Asked\n\nAdd up the amount of every item in an order table and report the total.\n\n"
    "## Approach\n\nRead the table as records and summed the amount column.\n\n"
    "## Worked\n\nThe sum.\n\n## Failed\n\nnone\n\n## Outcome\n\nnot shown\n"
)


def _add_capture(
    root: Path,
    source: str,
    signature: dict | str,
    record: dict | None = None,
    summary: str = SUMMARY,
) -> str:
    cid = _capture_id(source)
    d = root / CAPTURES_DIR / cid
    d.mkdir(parents=True, exist_ok=True)
    (d / CAPTURE_PROCEDURE).write_text(source)
    (d / CAPTURE_RECORD).write_text(json.dumps(record if record is not None else _record()))
    export_capture_files(d, signature, summary)
    return cid


GUARD = '''class MemoryInputError(ValueError):
    pass


def double(x):
    """Double a whole number."""
    if not isinstance(x, int):
        raise MemoryInputError("not an int")
    if x < 0:
        raise ValueError("negative")
    return x * 2
'''


def _library(tmp_path: Path, extra: dict | None = None, shapes=None) -> Path:
    root = _tree(tmp_path / "lib", {**LIB, "memory/text/guard.py": GUARD, **(extra or {})})
    write_files(root, lx.generated_library_mode(root, shapes=shapes))
    return root


def _py(root: Path, code: str, timeout: float = 120) -> dict:
    """Run *code* after :data:`PRELUDE` in a fresh isolated interpreter with *root* first on its path; the code
    prints one JSON value last."""
    out = subprocess.run(
        [sys.executable, "-I", "-c", f"import sys\nsys.path.insert(0, {str(root)!r})\n{PRELUDE}\n{code}"],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    assert out.returncode == 0, out.stderr[-4000:]
    return json.loads(out.stdout.strip().splitlines()[-1])


def _two_captures(root: Path) -> tuple[str, str]:
    table = _add_capture(root, _procedure(SUM_TABLE), _signature(ROWS, 7))
    grid = _add_capture(
        root,
        _procedure(TRANSPOSE, doc="Turn a grid's rows into its columns."),
        _signature(GRID, [[1, 4], [2, 5], [3, 6]], goal="Turn the rows of a grid into columns."),
    )
    return table, grid


# --- match: the structural default ---------------------------------------------------------------------------


def test_match_lists_a_capture_whose_input_has_the_same_structure_and_not_one_of_another(tmp_path):
    root = _library(tmp_path)
    table, grid = _two_captures(root)
    got = _py(
        root,
        f"rows = {ROWS!r}\n"
        f"print(json.dumps([text(memory.match(inputs=rows)), text(memory.match(request={REQUEST!r}, inputs=rows)),"
        f" text(memory.match(inputs={GRID!r}))]))",
    )
    by_inputs, with_request, by_grid = got
    assert table in by_inputs and grid not in by_inputs
    assert table in with_request and grid not in with_request
    assert grid in by_grid and table not in by_grid


def test_a_request_that_shares_only_words_does_not_match(tmp_path):
    """The goal's and the docstring's words, copied into the request, never make a match: the default compares
    structure only, and plain text has none."""
    root = _library(tmp_path)
    table, grid = _two_captures(root)
    got = _py(
        root,
        f"words = {WORDS_ONLY!r}\n"
        "print(json.dumps([text(memory.match(request=words)), text(memory.match(inputs=words)),"
        " text(memory.match(request=words, inputs={'colour': 'red', 'size': 2}))]))",
    )
    for listed in got:
        assert table not in listed and grid not in listed, listed


def test_match_with_nothing_given_or_an_empty_store_lists_nothing_and_does_not_raise(tmp_path):
    root = _library(tmp_path)  # no capture store in the copy at all
    got = _py(
        root,
        f"print(json.dumps([text(memory.match()), text(memory.match(inputs={ROWS!r})), text(memory.match(request=''))]))",
    )
    assert all(isinstance(t, str) for t in got)
    table, _ = _two_captures(root)
    assert table not in _py(root, "print(json.dumps(text(memory.match())))")


def test_a_malformed_capture_is_skipped_and_the_others_still_match(tmp_path):
    root = _library(tmp_path)
    table, _ = _two_captures(root)
    broken = _add_capture(root, _procedure(SUM_TABLE) + "\n# another copy\n", "{not json")
    got = _py(root, f"print(json.dumps(text(memory.match(inputs={ROWS!r}))))")
    assert table in got and broken not in got


def test_match_never_runs_stored_code(tmp_path):
    """Listing candidates imports no capture and no library module and calls no stored function: no ``memory.call``
    event, no side effect of their code."""
    mark = tmp_path / "ran.txt"
    root = _library(
        tmp_path,
        extra={
            "memory/text/stamp.py": (
                f"open({str(mark)!r}, 'a').write('library module imported\\n')\n\n\n"
                'def total(rows):\n    """Sum the amount of each row."""\n'
                f"    open({str(mark)!r}, 'a').write('library function called\\n')\n"
                '    return sum(r["amount"] for r in rows)\n'
            ),
        },
        shapes=lambda item, digest: (
            ([{**memory_helper.value_shape(ROWS), "param": "rows"}], False) if item.endswith(":total") else None
        ),
    )
    table = _add_capture(root, _procedure(SUM_TABLE, mark=str(mark)), _signature(ROWS, 7))
    got = _py(
        root,
        f"listed = text(memory.match(request={REQUEST!r}, inputs={ROWS!r}))\n"
        "import sys as _s\n"
        "print(json.dumps([listed, events, [m for m in _s.modules if m.startswith('memory.')]]))",
    )
    listed, evs, modules = got
    assert table in listed  # it was a candidate, and it fitted
    assert not mark.exists(), mark.read_text()
    assert [e for e in evs if e[0] == CALL_EVENT] == []
    assert "memory.text.stamp" not in modules


def test_match_raises_its_helper_event_and_no_use(tmp_path):
    root = _library(tmp_path)
    _two_captures(root)
    evs = _py(root, f"memory.match(inputs={ROWS!r})\nmemory.match(request='x')\nprint(json.dumps(events))")
    helpers = [e for e in evs if e[0] == HELPER_EVENT]
    assert [e[1] for e in helpers].count(MATCH_ID) == 2
    assert all(e[3] == "ok" for e in helpers if e[1] == MATCH_ID)
    assert [e for e in evs if e[0] == CALL_EVENT] == []


def test_what_match_shows_keeps_the_identifiers_a_procedure_recorded(tmp_path):
    """Rev 2.1: identifiers are never stripped or masked; what matching shows of a stored procedure carries the ids it
    recorded (in its source episode's summary, in its goal line, or as a key of its recorded input) as written, so
    an item keyed on another request's id is visibly so and never a wildcard."""
    root = _library(tmp_path)
    keyed = {ALIAS: ROWS}
    sig = _signature(keyed, 7, goal=f"Total the order table of {ALIAS_SHORT} ({ALIAS}).")
    summary = SUMMARY.replace("an order table", f"the order table of {ALIAS_SHORT}, id {ALIAS},")
    cid = _add_capture(root, _procedure(SUM_TABLE), sig, summary=summary)
    got = _py(
        root,
        f"print(json.dumps([text(memory.match(inputs={keyed!r})), text(memory.match(request={REQUEST!r}, "
        f"inputs={ROWS!r}))]))",
    )
    for shown in got:
        assert "<id>" not in shown, shown
        assert str(root) not in shown  # no host path
    assert any(ALIAS in shown and ALIAS_SHORT in shown for shown in got), got
    assert cid  # the capture exists; whether it is listed for a keyed input is the matcher's to decide


def test_a_very_large_input_is_matched_in_bounded_time(tmp_path):
    root = _library(tmp_path)
    table, _ = _two_captures(root)
    got = _py(
        root,
        "import time\nrows = [{'item': 'x%d' % i, 'amount': i} for i in range(200000)]\n"
        "t = time.monotonic()\nlisted = text(memory.match(inputs=rows))\n"
        "print(json.dumps([listed, time.monotonic() - t]))",
        timeout=180,
    )
    listed, seconds = got
    assert table in listed
    assert seconds < 60


def test_concurrent_matches_agree(tmp_path):
    root = _library(tmp_path)
    table, _ = _two_captures(root)
    got = _py(
        root,
        "out = [None] * 8\n"
        "def one(i):\n"
        "    try:\n"
        f"        out[i] = text(memory.match(inputs={ROWS!r}))\n"
        "    except Exception as exc:\n"
        "        out[i] = 'raised ' + type(exc).__name__ + ': ' + str(exc)\n"
        "ts = [threading.Thread(target=one, args=(i,)) for i in range(8)]\n"
        "[t.start() for t in ts]\n[t.join() for t in ts]\nprint(json.dumps(out))",
    )
    assert not any(g.startswith("raised ") for g in got), got[0]
    assert len(set(got)) == 1 and table in got[0]


# --- match: the pluggable hook -------------------------------------------------------------------------------


def test_the_hook_gets_each_candidates_signature_docstring_and_summary_as_written(tmp_path):
    root = _library(
        tmp_path,
        shapes=lambda item, digest: (
            ([{**memory_helper.value_shape(ROWS), "param": "s"}], False) if item.endswith(":parse_date") else None
        ),
    )
    summary = (
        f"## Asked\n\nAdd up the amounts in the order table of {ALIAS_SHORT} (request {ALIAS}).\n\n"
        "## Approach\n\nRead the records and summed the amount column with a plain loop.\n\n"
        "## Worked\n\nThe loop.\n\n## Failed\n\nnone\n\n## Outcome\n\nnot shown\n"
    )
    table = _add_capture(root, _procedure(SUM_TABLE, doc="Totals of order tables, by summing amounts."), _signature(ROWS, 7), summary=summary)
    got = _py(
        root,
        "seen = []\n"
        "def hook(request, inputs, candidates):\n"
        "    seen.append([plain(request), plain(inputs), [plain(c) for c in candidates]])\n"
        "    return list(candidates)\n"
        "set_hook(hook)\n"
        f"listed = text(memory.match(request={REQUEST!r}, inputs={ROWS!r}))\n"
        "print(json.dumps([listed, json.dumps(seen, default=repr)]))",
    )
    listed, seen = got
    assert seen != "[]", "the hook was not called"
    # each candidate's applicability signature, docstring and source summary reach the hook
    assert "amount" in seen  # a key path of the recorded input's signature
    assert "Totals of order tables" in seen or "Sum the amount column" in seen
    assert "Add up the amounts in the order table" in seen  # the summary's Asked
    assert "summed the amount column with a plain loop" in seen  # the summary's Approach
    # rev 2.1: with the summary's identifiers as written (never a placeholder), and no host path
    beyond_request = seen.replace(json.dumps(REQUEST)[1:-1], "").replace(REQUEST, "")
    assert ALIAS in beyond_request and ALIAS_SHORT in beyond_request and "<id>" not in seen
    assert str(root) not in seen
    # what the hook returns is what match tells the agent
    assert table in listed


def test_a_hook_that_judges_nothing_fits_lists_nothing_and_a_failing_hook_never_breaks_the_cell(tmp_path):
    root = _library(tmp_path)
    table, _ = _two_captures(root)
    got = _py(
        root,
        "set_hook(lambda request, inputs, candidates: [])\n"
        f"none = text(memory.match(inputs={ROWS!r}))\n"
        "def broken(request, inputs, candidates):\n    raise RuntimeError('hook failed')\n"
        "set_hook(broken)\n"
        "try:\n"
        f"    after = text(memory.match(inputs={ROWS!r}))\n"
        "    raised = None\n"
        "except Exception as exc:\n"
        "    after, raised = '', type(exc).__name__\n"
        "print(json.dumps([none, after, raised]))",
    )
    none, after, raised = got
    assert table not in none
    assert raised is None, f"match raised {raised} past the cell when its hook failed"


# --- run -----------------------------------------------------------------------------------------------------


def _target(cid: str) -> str:
    return CAPTURE_TARGET.format(id=cid)


RUN_FIELDS = (
    "[str(field(r, 'outcome')), plain(field(r, 'result')), plain(field(r, 'conditions')), "
    "str(field(r, 'level')), str(field(r, 'status')), "
    "str(r)]"
)


def _run(root: Path, target: str, *args_src: str, setup: str = "") -> dict:
    """``memory.run(target, *args)`` in a cell: the report's fields, its printed text, whether it raised and the
    package's events."""
    args = ", ".join(args_src)
    got = _py(
        root,
        setup
        + "try:\n"
        + f"    r = memory.run({target!r}, {args})\n"
        + f"    fields = {RUN_FIELDS}\n"
        + "    raised = None\n"
        + "except BaseException as exc:\n"
        + "    fields, raised = None, type(exc).__name__ + ': ' + str(exc)\n"
        + "print(json.dumps([fields, raised, events], default=repr))",
    )
    fields, raised, evs = got
    assert raised is None, f"memory.run raised past the cell: {raised}"
    keys = ("outcome", "result", "conditions", "level", "status", "printed")
    return {**dict(zip(keys, fields)), "events": evs}


def test_run_a_capture_on_the_request_reports_each_field(tmp_path):
    root = _library(tmp_path)
    table, _ = _two_captures(root)
    rep = _run(root, _target(table), repr(REQUEST))
    assert rep["outcome"] == RETURNED
    assert rep["result"] == 7
    assert "repeatable" not in rep["printed"] and "agrees with" not in rep["printed"]
    cond = json.dumps(rep["conditions"]).lower()
    assert "applies" in cond and "holds" in cond
    assert "false" not in cond and "not" not in cond.replace("none stated", "")
    assert rep["level"] == NEW_LEVEL and rep["status"] == NEW_STATUS
    assert len(rep["printed"].strip().splitlines()) >= 3  # it prints as a few lines
    assert ALIAS not in rep["printed"] and ALIAS_SHORT not in rep["printed"]
    assert str(root) not in rep["printed"]


def test_run_raises_its_helper_event_and_counts_the_procedure_call_as_a_use(tmp_path):
    root = _library(tmp_path)
    table, _ = _two_captures(root)
    evs = _run(root, _target(table), repr(REQUEST))["events"]
    assert [e[1] for e in evs if e[0] == HELPER_EVENT].count(RUN_ID) == 1
    uses = [e for e in evs if e[0] == CALL_EVENT]
    assert uses and all(e[1] == _target(table) for e in uses), uses
    assert all(e[3] == "ok" for e in uses)


def test_run_a_library_function_on_the_agents_inputs(tmp_path):
    root = _library(tmp_path)
    rep = _run(root, "memory.text.guard:double", "2")
    assert rep["outcome"] == RETURNED and rep["result"] == 4
    calls = [e for e in rep["events"] if e[0] == CALL_EVENT]
    # rev 2.1: one call, never a second one to check that it repeats
    assert len(calls) == 1 and calls[0][1] == "memory.text.guard:double" and calls[0][3] == "ok"
    assert RUN_ID in [e[1] for e in rep["events"] if e[0] == HELPER_EVENT]


def test_run_reports_a_refusal_and_an_error_without_raising(tmp_path):
    root = _library(tmp_path)
    refused = _run(root, "memory.text.guard:double", "'a word'")
    assert refused["outcome"] == REFUSED
    failed = _run(root, "memory.text.guard:double", "-1")
    assert str(failed["outcome"]).startswith(ERROR)
    assert "ValueError" in failed["printed"] or "ValueError" in str(failed["outcome"])
    assert [e[3] for e in failed["events"] if e[0] == CALL_EVENT][:1] == ["error"]


def test_run_reports_a_capture_that_raises_or_whose_reader_raises(tmp_path):
    root = _library(tmp_path)
    boom = _add_capture(
        root,
        _procedure(SUM_TABLE.replace('return sum(r["amount"] for r in value)', 'raise RuntimeError("broke")')),
        _signature(ROWS, 7),
    )
    rep = _run(root, _target(boom), repr(REQUEST))
    assert str(rep["outcome"]).startswith(ERROR) and "RuntimeError" in rep["printed"]
    unreadable = _run(root, _target(boom), repr("no table in this text"))
    assert unreadable["outcome"] in (REFUSED,) or str(unreadable["outcome"]).startswith(ERROR)
    deep = _add_capture(
        root,
        _procedure(SUM_TABLE + "\n\ndef _down(n):\n    return _down(n + 1)\n").replace(
            'return sum(r["amount"] for r in value)', "return _down(0)"
        ),
        _signature(ROWS, 7),
    )
    rep = _run(root, _target(deep), repr(REQUEST))
    assert str(rep["outcome"]).startswith(ERROR) and "RecursionError" in rep["printed"]


#: A captured procedure that leaves one mark in its own directory each time its run is called (the work tree the
#: run acts on), so a test counts its calls from the outside.
MARKING_RUN = (
    'def run(value):\n    """Sum the amount column, and mark that it ran."""\n'
    '    with open("runs.log", "a") as fh:\n        fh.write("x")\n'
    '    return sum(r["amount"] for r in value)'
)


def test_run_calls_a_captured_procedure_once(tmp_path):
    """Rev 2.1, fix 4: an item that acts (writes, sends, steps) acts once; nothing runs it a second time."""
    root = _library(tmp_path)
    marking = _add_capture(
        root,
        _procedure(
            SUM_TABLE.replace(
                'def run(value):\n    """Sum the amount column of an order table of items."""\n'
                '    return sum(r["amount"] for r in value)',
                MARKING_RUN,
            ),
        ),
        _signature(ROWS, 7),
    )
    rep = _run(root, _target(marking), repr(REQUEST))
    assert rep["outcome"] == RETURNED and rep["result"] == 7
    assert (root / CAPTURES_DIR / marking / "runs.log").read_text() == "x"


def test_run_reports_conditions_that_do_not_hold(tmp_path):
    root = _library(tmp_path)
    fails_holds = _add_capture(
        root,
        _procedure(SUM_TABLE.replace("return isinstance(result, int)", "return result < 0")),
        _signature(ROWS, 7),
    )
    rep = _run(root, _target(fails_holds), repr(REQUEST))
    cond = rep["conditions"]
    if isinstance(cond, dict):
        assert cond.get("holds") in (False, "False", "no", "did not hold")
    else:
        low = str(cond).lower()
        assert "holds" in low and ("false" in low or "not" in low), cond
    not_applies = _add_capture(
        root,
        _procedure(SUM_TABLE.replace("return isinstance(value, list) and", "return False and")),
        _signature(ROWS, 7),
    )
    rep = _run(root, _target(not_applies), repr(REQUEST))
    low = json.dumps(rep["conditions"]).lower()
    assert "applies" in low and ("false" in low or "not" in low), rep["conditions"]


def test_run_never_runs_another_matching_procedure(tmp_path):
    """Rev 2.1, fix 4: no other stored procedure runs on the caller's argument to compare results (rev 2 ran every
    matching one to report which agreed); the report says only what the one call did."""
    root = _library(tmp_path)
    first = _add_capture(root, _procedure(SUM_TABLE), _signature(ROWS, 7))
    other = _add_capture(
        root,
        _procedure(
            SUM_TABLE.replace(
                'def run(value):\n    """Sum the amount column of an order table of items."""\n'
                '    return sum(r["amount"] for r in value)',
                MARKING_RUN,
            ),
        ),
        _signature(ROWS, 7),
    )
    rep = _run(root, _target(first), repr(REQUEST))
    assert rep["outcome"] == RETURNED and rep["result"] == 7
    assert not (root / CAPTURES_DIR / other / "runs.log").exists()
    assert "agrees" not in rep["printed"] and "checked against" not in rep["printed"]
    assert [e[1] for e in rep["events"] if e[0] == CALL_EVENT] == [_target(first)]


def test_the_report_never_claims_a_result_is_correct_without_verdicts(tmp_path):
    import re

    root = _library(tmp_path)
    table, _ = _two_captures(root)
    rep = _run(root, _target(table), repr(REQUEST))
    assert not re.search(r"\b(correct|right|verified|accepted)\b", rep["printed"], re.I), rep["printed"]


def test_run_through_the_bound_name_needs_no_import_of_the_procedure(tmp_path):
    """Running a capture is one call: nothing of the capture is importable as a library module, and running it
    leaves the library's own modules unchanged."""
    root = _library(tmp_path)
    table, _ = _two_captures(root)
    got = _py(
        root,
        f"r = memory.run({_target(table)!r}, {REQUEST!r})\n"
        "import sys as _s\nprint(json.dumps(sorted(m for m in _s.modules if m.startswith('memory.'))))",
    )
    assert not any(table in m for m in got)
