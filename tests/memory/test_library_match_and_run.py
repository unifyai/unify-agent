"""An item's own check, ``item.fits(inputs)``, and its call, ``item(...)``, in the generated library package, run as
a cell runs them.

``memory.get(ref).fits(inputs)`` tells the agent whether a stored procedure or capture fits what it holds. It compares
structure only: it never matches by words and never runs stored code; what an item shows of a stored procedure is as
it was recorded, identifiers included (none is stripped or masked).

``memory.get(ref)(*args, **kwargs)`` runs one stored procedure (a library function, or a captured procedure) on the
agent's inputs, tracked, and returns what it returned; the tracked run under it (``_tracked_run``) reports how the run
went: its outcome, its result, whether its stated conditions held. It calls the procedure once: never a second time
and never another procedure beside it. The tracked run reports a failing procedure and never raises it.

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
#: How a capture is named as an item (design I.4.1).
CAPTURE_TARGET = "capture:{id}"
#: The audit event and ids of the package's own helpers (``library_helper.HELPER_EVENT``; design J.3 / lane).
HELPER_EVENT = "memory.helper_call"
CALL_EVENT = "memory.call"
MATCH_ID = "memory:match"
RUN_ID = "memory:run"
#: The evidence level and status of a capture no use has decided yet (its evidence row's defaults).
NEW_LEVEL = "unchecked"
NEW_STATUS = "on-trial"
#: The report's outcomes (design I.5.1).
RETURNED, REFUSED, ERROR = "returned", "refused", "error"
#: Code run first in every cell of these tests: how a test reads a report's field (an attribute, else a key), and
#: the audit hook that keeps the package's events.
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
    docstring and the summary's Asked and Approach (``captures.saved_against``, as the capture step writes it).
    Nothing strips an identifier, so all of them are as written."""
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

#: A request in plain prose with a table of records in it, and a printed identifier (kept as written wherever a
#: stored procedure recorded it, never a placeholder).
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
        "deterministic": True,
        "applies": True,
        "holds": True,
        "outcome": "none",
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


# --- an item's own check: the structural default ---------------------------------------------------------------


def _fits(cid: str, value_src: str) -> str:
    """Cell code for one check, ``memory.get(capture:<id>).fits(value)``: its ``ok`` and its text."""
    return f"(lambda f: [f.ok, str(f)])(memory.get({_target(cid)!r}).fits({value_src}))"


def test_a_capture_fits_an_input_of_the_same_structure_and_not_one_of_another(tmp_path):
    root = _library(tmp_path)
    table, grid = _two_captures(root)
    got = _py(
        root,
        f"rows = {ROWS!r}\n"
        f"print(json.dumps([{_fits(table, 'rows')}, {_fits(grid, 'rows')}, {_fits(grid, repr(GRID))},"
        f" {_fits(table, repr(GRID))}]))",
    )
    (table_rows, _), (grid_rows, _), (grid_grid, _), (table_grid, _) = got
    assert table_rows is True and grid_rows is False
    assert grid_grid is True and table_grid is False


def test_an_input_that_shares_only_words_does_not_fit(tmp_path):
    """The goal's and the docstring's words, passed as the input, never make a fit: the check compares structure
    only, and plain text has none."""
    root = _library(tmp_path)
    table, grid = _two_captures(root)
    other = "{'colour': 'red', 'size': 2}"
    got = _py(
        root,
        f"words = {WORDS_ONLY!r}\n"
        f"print(json.dumps([{_fits(table, 'words')}, {_fits(grid, 'words')}, {_fits(table, other)},"
        f" {_fits(grid, other)}]))",
    )
    for ok, shown in got:
        assert ok is False, shown
    assert "no structure to compare" in got[0][1]


def test_a_check_given_nothing_does_not_fit_and_does_not_raise(tmp_path):
    root = _library(tmp_path)
    table, _ = _two_captures(root)
    got = _py(root, f"print(json.dumps([{_fits(table, 'None')}, {_fits(table, repr(''))}]))")
    for ok, shown in got:
        assert ok is False and shown.startswith("does not fit: "), shown


def test_a_malformed_capture_does_not_fit_and_the_others_still_do(tmp_path):
    root = _library(tmp_path)
    table, _ = _two_captures(root)
    broken = _add_capture(root, _procedure(SUM_TABLE) + "\n# another copy\n", "{not json")
    got = _py(root, f"print(json.dumps([{_fits(table, repr(ROWS))}, {_fits(broken, repr(ROWS))}]))")
    (table_ok, _), (broken_ok, broken_why) = got
    assert table_ok is True and broken_ok is False
    assert "no input it was saved against is recorded" in broken_why


def test_a_check_never_runs_stored_code(tmp_path):
    """A check imports no capture and no library module and calls no stored function: no ``memory.call`` event, no
    side effect of their code."""
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
        f"fitted = memory.get({_target(table)!r}).fits({ROWS!r}).ok\n"
        f"memory.get('memory.text.stamp:total').fits({ROWS!r})\n"
        "import sys as _s\n"
        "print(json.dumps([fitted, events, [m for m in _s.modules if m.startswith('memory.')]]))",
    )
    fitted, evs, modules = got
    assert fitted is True  # it was checked, and it fitted
    assert not mark.exists(), mark.read_text()
    assert [e for e in evs if e[0] == CALL_EVENT] == []
    assert "memory.text.stamp" not in modules


def test_a_check_raises_its_helper_event_and_no_use(tmp_path):
    root = _library(tmp_path)
    table, _ = _two_captures(root)
    evs = _py(
        root,
        f"item = memory.get({_target(table)!r})\nitem.fits({ROWS!r})\nitem.fits('x')\nprint(json.dumps(events))",
    )
    helpers = [e for e in evs if e[0] == HELPER_EVENT]
    assert [e[1] for e in helpers].count(MATCH_ID) == 2
    assert all(e[3] == "ok" for e in helpers if e[1] == MATCH_ID)
    assert [e for e in evs if e[0] == CALL_EVENT] == []


def test_what_a_capture_shows_keeps_the_identifiers_it_recorded(tmp_path):
    """Identifiers are never stripped or masked; what a stored procedure's card and its check show carry the ids it
    recorded (in its source episode's summary, in its goal line, or as a key of its recorded input) as written, so
    an item keyed on another request's id is visibly so and never a wildcard."""
    root = _library(tmp_path)
    keyed = {ALIAS: ROWS}
    sig = _signature(keyed, 7, goal=f"Total the order table of {ALIAS_SHORT} ({ALIAS}).")
    summary = SUMMARY.replace("an order table", f"the order table of {ALIAS_SHORT}, id {ALIAS},")
    cid = _add_capture(root, _procedure(SUM_TABLE), sig, summary=summary)
    got = _py(
        root,
        f"item = memory.get({_target(cid)!r})\n"
        f"print(json.dumps([str(item), str(item.fits({keyed!r})), str(item.fits({ROWS!r}))]))",
    )
    for shown in got:
        assert "<id>" not in shown, shown
        assert str(root) not in shown  # no host path
    assert ALIAS in got[0] and ALIAS_SHORT in got[0], got[0]


def test_a_very_large_input_is_checked_in_bounded_time(tmp_path):
    root = _library(tmp_path)
    table, _ = _two_captures(root)
    got = _py(
        root,
        "import time\nrows = [{'item': 'x%d' % i, 'amount': i} for i in range(200000)]\n"
        f"item = memory.get({_target(table)!r})\n"
        "t = time.monotonic()\nfitted = item.fits(rows).ok\n"
        "print(json.dumps([fitted, time.monotonic() - t]))",
        timeout=180,
    )
    fitted, seconds = got
    assert fitted is True
    assert seconds < 60


def test_concurrent_checks_agree(tmp_path):
    root = _library(tmp_path)
    table, _ = _two_captures(root)
    got = _py(
        root,
        f"item = memory.get({_target(table)!r})\n"
        "out = [None] * 8\n"
        "def one(i):\n"
        "    try:\n"
        f"        out[i] = str(item.fits({ROWS!r}))\n"
        "    except Exception as exc:\n"
        "        out[i] = 'raised ' + type(exc).__name__ + ': ' + str(exc)\n"
        "ts = [threading.Thread(target=one, args=(i,)) for i in range(8)]\n"
        "[t.start() for t in ts]\n[t.join() for t in ts]\nprint(json.dumps(out))",
    )
    assert not any(g.startswith("raised ") for g in got), got[0]
    assert len(set(got)) == 1 and got[0].startswith("fits: ")


# --- an item's tracked call --------------------------------------------------------------------------------------


def _target(cid: str) -> str:
    return CAPTURE_TARGET.format(id=cid)


RUN_FIELDS = (
    "[str(field(r, 'outcome')), plain(field(r, 'result')), plain(field(r, 'conditions')), "
    "str(field(r, 'level')), str(field(r, 'status')), "
    "str(r)]"
)


def _run(root: Path, target: str, *args_src: str, setup: str = "") -> dict:
    """The tracked run under an item's call, ``memory._tracked_run(target, args, {})``, in a cell: the report's
    fields, its printed text, whether it raised and the package's events."""
    args = ", ".join(args_src)
    got = _py(
        root,
        setup
        + "try:\n"
        + f"    r = memory._tracked_run({target!r}, ({args},), {{}})\n"
        + f"    fields = {RUN_FIELDS}\n"
        + "    raised = None\n"
        + "except BaseException as exc:\n"
        + "    fields, raised = None, type(exc).__name__ + ': ' + str(exc)\n"
        + "print(json.dumps([fields, raised, events], default=repr))",
    )
    fields, raised, evs = got
    assert raised is None, f"the tracked run raised past the cell: {raised}"
    keys = ("outcome", "result", "conditions", "level", "status", "printed")
    return {**dict(zip(keys, fields)), "events": evs}


def _call(root: Path, target: str, *args_src: str) -> dict:
    """``memory.get(target)(*args)`` in a cell: what it returned, what it raised (type and text) and the package's
    events."""
    args = ", ".join(args_src)
    result, raised, evs = _py(
        root,
        "try:\n"
        + f"    result, raised = plain(memory.get({target!r})({args})), None\n"
        + "except Exception as exc:\n"
        + "    result, raised = None, [type(exc).__name__, str(exc)]\n"
        + "print(json.dumps([result, raised, events], default=repr))",
    )
    return {"result": result, "raised": raised, "events": evs}


def test_the_tracked_run_of_a_capture_on_the_request_reports_each_field(tmp_path):
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


def test_a_call_raises_its_helper_event_and_counts_the_procedure_call_as_a_use(tmp_path):
    root = _library(tmp_path)
    table, _ = _two_captures(root)
    got = _call(root, _target(table), repr(REQUEST))
    assert got["raised"] is None and got["result"] == 7
    evs = got["events"]
    assert [e[1] for e in evs if e[0] == HELPER_EVENT].count(RUN_ID) == 1
    uses = [e for e in evs if e[0] == CALL_EVENT]
    assert uses and all(e[1] == _target(table) for e in uses), uses
    assert all(e[3] == "ok" for e in uses)


def test_a_library_function_item_is_called_on_the_agents_inputs(tmp_path):
    root = _library(tmp_path)
    got = _call(root, "memory.text.guard:double", "2")
    assert got["raised"] is None and got["result"] == 4
    calls = [e for e in got["events"] if e[0] == CALL_EVENT]
    # one call, never a second one to check that it repeats
    assert len(calls) == 1 and calls[0][1] == "memory.text.guard:double" and calls[0][3] == "ok"
    assert RUN_ID in [e[1] for e in got["events"] if e[0] == HELPER_EVENT]


def test_the_tracked_run_reports_a_refusal_and_an_error_without_raising(tmp_path):
    root = _library(tmp_path)
    refused = _run(root, "memory.text.guard:double", "'a word'")
    assert refused["outcome"] == REFUSED
    failed = _run(root, "memory.text.guard:double", "-1")
    assert str(failed["outcome"]).startswith(ERROR)
    assert "ValueError" in failed["printed"] or "ValueError" in str(failed["outcome"])
    assert [e[3] for e in failed["events"] if e[0] == CALL_EVENT][:1] == ["error"]


def test_the_tracked_run_reports_a_capture_that_raises_or_whose_reader_raises(tmp_path):
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
    # the item's call raises the same failure, with its detail
    got = _call(root, _target(boom), repr(REQUEST))
    assert got["raised"][0] == "ArgumentError" and "RuntimeError: broke" in got["raised"][1]


#: A captured procedure that leaves one mark in its own directory each time its run is called (the work tree the
#: run acts on), so a test counts its calls from the outside.
MARKING_RUN = (
    'def run(value):\n    """Sum the amount column, and mark that it ran."""\n'
    '    with open("runs.log", "a") as fh:\n        fh.write("x")\n'
    '    return sum(r["amount"] for r in value)'
)


def test_a_call_runs_a_captured_procedure_once(tmp_path):
    """An item that acts (writes, sends, steps) acts once; nothing runs it a second time."""
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
    got = _call(root, _target(marking), repr(REQUEST))
    assert got["raised"] is None and got["result"] == 7
    assert (root / CAPTURES_DIR / marking / "runs.log").read_text() == "x"


def test_the_tracked_run_reports_conditions_that_do_not_hold(tmp_path):
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


def test_the_tracked_run_never_runs_another_matching_procedure(tmp_path):
    """No other stored procedure runs on the caller's argument to compare results; the report says only what the
    one call did."""
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


def test_a_call_through_the_bound_name_needs_no_import_of_the_procedure(tmp_path):
    """Calling a capture is one call: nothing of the capture is importable as a library module, and running it
    leaves the library's own modules unchanged."""
    root = _library(tmp_path)
    table, _ = _two_captures(root)
    got = _py(
        root,
        f"r = memory.get({_target(table)!r})({REQUEST!r})\n"
        "import sys as _s\nprint(json.dumps(sorted(m for m in _s.modules if m.startswith('memory.'))))",
    )
    assert not any(table in m for m in got)
