"""Symbolic: ``UNIFY_FUNCTION_VALUE_NOTICE``, one line when a value a stored function matches on no longer occurs in what it read.

A reused function runs cleanly on a stale value -- the request says "meal"
where the export says "meals", the code keeps last month's error codes -- and
the answer comes out empty or short. With the switch on, the files a recorded
call reads are scanned when it returns for its short string arguments and the
string literals its code compares against, as whole values. A value that
matched in every accepted earlier call of the same source and now occurs 0
times gets one line after the call, with the closest values of a small table
column. Silent on a cut scan, a started process, no file reads, or no accepted
history. Functions run in-process on files the test writes; no model or
network is called.
"""

from __future__ import annotations

import json

import pytest

from tests.function_manager.core import test_function_cases as tfc
from tests.function_manager.core.test_function_cases import (  # noqa: F401 (fixtures)
    cases_on,
    music_env,
)
from tests.helpers import _handle_project
from unify.actor.execution import capture, worker_child
from unify.function_manager import run_summary, task_origin, value_notice
from unify.settings import ProductionSettings, SETTINGS

TOTAL_FOR = (
    "def total_for(path: str, category: str) -> float:\n"
    "    import csv, io, pathlib\n"
    "    rows = list(csv.DictReader(io.StringIO(pathlib.Path(path).read_text())))\n"
    "    return round(sum(float(r['amount']) for r in rows if r['category'] == category), 2)\n"
)
COUNT_CODES = (
    "def count_codes(path: str) -> dict:\n"
    "    codes = ['E101', 'E202']\n"
    "    import pathlib\n"
    "    counts = {}\n"
    "    for line in pathlib.Path(path).read_text().splitlines():\n"
    "        for code in codes:\n"
    "            if code in line.split():\n"
    "                counts[code] = counts.get(code, 0) + 1\n"
    "    return counts\n"
)
TOTAL_VIA_SHELL = (
    "def total_via_shell(path: str, category: str) -> int:\n"
    "    import pathlib, subprocess\n"
    "    subprocess.run(['true'], check=False)\n"
    "    lines = pathlib.Path(path).read_text().splitlines()\n"
    "    return sum(1 for line in lines if line.split(',')[0] == category)\n"
)
LOGIN_COUNT = (
    "def logins_for(path: str, api_key: str) -> int:\n"
    "    import pathlib\n"
    "    return sum(1 for line in pathlib.Path(path).read_text().splitlines() if api_key in line)\n"
)


@pytest.fixture
def notice_on(monkeypatch, cases_on):
    monkeypatch.setattr(SETTINGS, "UNIFY_FUNCTION_SUMMARY", True)
    monkeypatch.setattr(SETTINGS, "UNIFY_FUNCTION_VALUE_NOTICE", True)
    monkeypatch.setattr(SETTINGS, "UNIFY_TASK_ORIGIN", True)
    # Outcomes are kept under requests while request records are on.
    monkeypatch.setattr(SETTINGS, "UNIFY_ENTRY_RECORD", True)


def _in_cell(request: str, fn, *args, solved=None):
    origin = task_origin.enter(request)
    parts = capture._stdout_parts.set([])
    try:
        value = fn(*args)
        if solved is not None:
            task_origin.record_outcome(solved)
        return value, "".join(p.text for p in capture._stdout_parts.get())
    finally:
        capture._stdout_parts.reset(parts)
        task_origin.leave(origin)


def _function(name: str, source: str):
    fm = tfc._FM()
    fm.add_functions(implementations=[source])
    return tfc._load(fm)[name]


def _expenses(path, categories):
    rows = ["category,amount", *(f"{c},{10 + i}" for i, c in enumerate(categories))]
    path.write_text("\n".join(rows) + "\n")
    return str(path)


def test_the_switch_is_off_by_default():
    assert (
        ProductionSettings.model_fields["UNIFY_FUNCTION_VALUE_NOTICE"].default is False
    )


def test_values_count_as_whole_values_only(tmp_path):
    table = _expenses(tmp_path / "x.csv", ["meals", "meals", "travel"])
    log = tmp_path / "app.log"
    log.write_text("ERROR E101 disk\nWARN E1010 slow\n")
    doc = tmp_path / "d.json"
    doc.write_text(json.dumps({"E202": 3, "level": "meals"}))
    out = worker_child.count_values(
        [table, str(log), str(doc)],
        ["meal", "meals", "E101", "E202"],
        ("category",),
    )
    assert out["counts"] == {"meal": 0, "meals": 3, "E101": 1, "E202": 1}
    assert out["where"]["meals"] == ["category", "level"]
    assert out["distinct"]["category"] == {"meals": 2, "travel": 1}
    assert out["sampled"] is False


def test_a_cut_scan_says_so(tmp_path, monkeypatch):
    monkeypatch.setattr(worker_child, "WATCH_MAX_BYTES", 8)
    table = _expenses(tmp_path / "x.csv", ["meals"] * 5)
    assert worker_child.count_values([table], ["meals"])["sampled"] is True


def test_literals_are_the_strings_the_code_compares_against():
    assert value_notice.literals(COUNT_CODES, "count_codes") == ["E101", "E202"]
    assert value_notice.literals(
        "def f(df):\n    return df[df.status.isin(['paid', 'open']) & (df.kind != 'x')]\n",
        "f",
    ) == ["paid", "open", "x"]


@_handle_project
def test_an_argument_that_no_longer_occurs_is_remarked_on_with_closest_values(
    notice_on,
    music_env,
    tmp_path,
):
    fn = _function("total_for", TOTAL_FOR)
    first = _expenses(tmp_path / "q2.csv", ["meals", "travel", "meals"])
    value, out = _in_cell(
        "Total approved meals spend.",
        fn,
        first,
        "meals",
        solved=True,
    )
    assert value == 22.0 and out == ""
    later = _expenses(tmp_path / "q3.csv", ["meals", "travel", "meals", "meals"])
    value, out = _in_cell("Total approved meal spend for Q3.", fn, later, "meal")
    assert value == 0
    assert (
        "[total_for: category='meal' occurs 0 times as a whole value in the files this "
        "call read; in its one earlier accepted call it occurred 2 times. Closest values "
        "now in column category: 'meals' (3).]"
    ) in out


@_handle_project
def test_a_literal_that_no_longer_occurs_is_remarked_on(
    notice_on,
    music_env,
    tmp_path,
):
    fn = _function("count_codes", COUNT_CODES)
    september = tmp_path / "sep.log"
    september.write_text("ERROR E101 a\nERROR E202 b\nERROR E101 c\n")
    _in_cell("Count the error codes in the log.", fn, str(september), solved=True)
    october = tmp_path / "oct.log"
    october.write_text("ERROR E101 a\nERROR E909 b\n")
    _, out = _in_cell("Count the error codes in October's log.", fn, str(october))
    assert "the literal 'E202' in its code occurs 0 times" in out
    assert "'E101'" not in out


@_handle_project
def test_without_an_accepted_call_it_says_nothing(notice_on, music_env, tmp_path):
    fn = _function("total_for", TOTAL_FOR)
    _in_cell("Total meals.", fn, _expenses(tmp_path / "a.csv", ["meals"]), "meals")
    _, out = _in_cell(
        "Total meal.",
        fn,
        _expenses(tmp_path / "b.csv", ["meals"]),
        "meal",
    )
    assert "occurs 0 times" not in out


@_handle_project
def test_a_value_that_still_occurs_is_not_remarked_on(notice_on, music_env, tmp_path):
    fn = _function("total_for", TOTAL_FOR)
    _in_cell(
        "Meals?",
        fn,
        _expenses(tmp_path / "a.csv", ["meals"]),
        "meals",
        solved=True,
    )
    _, out = _in_cell(
        "Meals now?",
        fn,
        _expenses(tmp_path / "b.csv", ["meals"]),
        "meals",
    )
    assert out == ""


@_handle_project
def test_a_cut_scan_keeps_it_silent(notice_on, music_env, tmp_path, monkeypatch):
    fn = _function("total_for", TOTAL_FOR)
    _in_cell(
        "Meals?",
        fn,
        _expenses(tmp_path / "a.csv", ["meals"]),
        "meals",
        solved=True,
    )
    monkeypatch.setattr(worker_child, "WATCH_MAX_BYTES", 8)
    _, out = _in_cell(
        "Meal?",
        fn,
        _expenses(tmp_path / "b.csv", ["meals"] * 9),
        "meal",
    )
    assert "occurs 0 times" not in out


@_handle_project
def test_a_call_that_starts_a_process_keeps_it_silent(notice_on, music_env, tmp_path):
    fn = _function("total_via_shell", TOTAL_VIA_SHELL)
    _in_cell(
        "Meals?",
        fn,
        _expenses(tmp_path / "a.csv", ["meals"]),
        "meals",
        solved=True,
    )
    _, out = _in_cell("Meal?", fn, _expenses(tmp_path / "b.csv", ["meals"]), "meal")
    assert "occurs 0 times" not in out


@_handle_project
def test_a_credential_argument_is_never_watched_or_shown(
    notice_on,
    music_env,
    tmp_path,
):
    fn = _function("logins_for", LOGIN_COUNT)
    log = tmp_path / "a.log"
    log.write_text("sk-live-abcdef123456 login\n")
    _in_cell("Count my logins.", fn, str(log), "sk-live-abcdef123456", solved=True)
    other = tmp_path / "b.log"
    other.write_text("nothing here\n")
    _, out = _in_cell("Count my logins today.", fn, str(other), "sk-live-zzzzzz999999")
    assert "sk-live" not in out
    (row,) = [
        r
        for r in run_summary._connect().execute(
            "SELECT inputs FROM function_runs ORDER BY seq DESC LIMIT 1",
        )
    ]
    assert row[0] is None or "api_key" not in row[0]


@_handle_project
def test_argument_values_are_not_kept(notice_on, music_env, tmp_path):
    fn = _function("total_for", TOTAL_FOR)
    _in_cell("Travel?", fn, _expenses(tmp_path / "a.csv", ["travel"]), "travel")
    (row,) = list(
        run_summary._connect().execute("SELECT inputs FROM function_runs"),
    )
    kept = json.loads(row[0])
    assert kept["keys"]["arg:category"] == {"n": 1, "where": ["category"]}
    assert "travel" not in json.dumps(kept["keys"])


@_handle_project
def test_off_scans_nothing(notice_on, music_env, tmp_path, monkeypatch):
    monkeypatch.setattr(SETTINGS, "UNIFY_FUNCTION_VALUE_NOTICE", False)
    fn = _function("total_for", TOTAL_FOR)
    _in_cell("Travel?", fn, _expenses(tmp_path / "a.csv", ["travel"]), "travel")
    (row,) = list(
        run_summary._connect().execute("SELECT inputs FROM function_runs"),
    )
    assert row[0] is None
