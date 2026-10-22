"""Tests the writer states before it runs them (``tools.test``, ``UNIFY_MEMORY_WRITER_TESTS``).

The expectation is on record before anything runs and never changes; a value test runs an item on a recorded input
(rev 2 has no replay test: nothing replays an episode's recorded calls). The boxes are real (bubblewrap); no model is
called.
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import pytest

from unify.memory import writer_tests as wt
from unify.memory.episodes import Action, Cell, Episode
from unify.memory.sandbox_run import SandboxResult

needs_bwrap = pytest.mark.skipif(shutil.which("bwrap") is None, reason="bubblewrap required")

REPORT = '''"""Reports."""


def write_report(state, env):
    """Write the totals report of the request's rows."""
    rows = state.data[0]["rows"]
    total = sum(sum(v) for v in rows.values())
    env.files.write("out/report.txt", f"total {total}")
    return None


def write_wrong(state, env):
    """Write a report with another total."""
    env.files.write("out/report.txt", "total 7")


def mail_it(state, env):
    """Mail the report: a call this episode never made."""
    env.mail.send("ops", "total 6")


def handles(state, env):
    """What the env holds, after an unknown-effect call."""
    env.env.say("hi")
    return sorted(vars(env))


def total(request):
    """The request's total."""
    return sum(sum(v) for v in request.data[0]["rows"].values())


def boom(request):
    """Raises."""
    raise KeyError("x")
'''


def _library(root: Path) -> Path:
    for rel, text in {
        "memory/__init__.py": "",
        "memory/files/__init__.py": '"""Files."""\n',
        "memory/files/report.py": REPORT,
        "notes/files/how.md": "---\ntitle: How\ndescription: d\nuses: []\n---\nWrite it.\n",
    }.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
    return root


def _ep() -> Episode:
    return Episode(
        episode_id="e1",
        started_at="2026-10-20T00:00:00Z",
        ended_at="2026-10-20T00:01:00Z",
        build="b",
        model="m",
        effort="low",
        regime="none",
        memory_main="0" * 40,
        worktree_before=None,
        worktree_after=None,
        request=['Report: {"rows": {"a": [1, 2], "b": [3]}}'],
        transcript=[],
        cells=[Cell(0, "print(6)", "6\n")],
        actions=[
            Action(0, "files", "write", ["out/report.txt", "total 6"], {}, {"written": 7}, "ok", "write"),
            Action(0, "env", "say", ["hi"], {}, "ok", "ok", "unknown", kind="dialogue"),
        ],
    )


def _ctx(tmp_path, modes=("value",), runner=None) -> wt.TestContext:
    """*modes* is kept for the callers: rev 2 has one mode, a value test (no replay)."""
    lib = _library(tmp_path / "memory-copy")
    kw = {"runner": runner} if runner is not None else {}
    return wt.TestContext(
        pass_id="p1",
        tests_path=tmp_path / "passes" / "p1" / "tests.jsonl",
        library=lib,
        episode=lambda eid: _ep() if eid == "e1" else None,
        work=tmp_path / "work",
        **kw,
    )


# --- 4. unknown effects are compared, never executed --------------------------------------------------------------------------


# --- 5. the expectation is on record first, and a changed one is a new test --------------------------------------------------


def test_the_expectation_is_recorded_before_the_run_and_a_changed_one_is_a_new_test(tmp_path):
    seen: list = []

    def runner(argv, *, ro=None, rw=None, cwd="/tmp", timeout_s=120.0, env=None):
        # at run time the stated row is already on record
        rows = [json.loads(x) for x in ctx.tests_path.read_text().splitlines()]
        seen.append([r["phase"] for r in rows])
        [out] = [h for h, b in (rw or {}).items() if b == "/out"]
        (Path(out) / "result.json").write_text(json.dumps({"outcome": "returned", "result": "6"}))
        return SandboxResult(0, "", "", False)

    ctx = _ctx(tmp_path, runner=runner)
    first = wt.run_test(ctx, "memory.files.report:total", "e1/request", {"kind": "returns", "value": 6})
    second = wt.run_test(ctx, "memory.files.report:total", "e1/request", {"kind": "returns", "value": 7})
    assert seen == [["stated"], ["stated", "result", "stated"]]
    assert (first.outcome, second.outcome) == ("as_stated", "not_as_stated")
    assert first.test_id != second.test_id
    rows = [json.loads(x) for x in ctx.tests_path.read_text().splitlines()]
    assert [r["phase"] for r in rows] == ["stated", "result", "stated", "result"]
    assert [r["stated"]["value"] for r in rows if r["phase"] == "stated"] == [6, 7]
    assert [r.test_id for r in wt.results(ctx.tests_path)] == [first.test_id, second.test_id]


# --- value tests and the refusals ---------------------------------------------------------------------------------------------


@needs_bwrap
def test_value_tests_run_on_the_recorded_input_named(tmp_path):
    ctx = _ctx(tmp_path, modes=("value",))
    assert wt.run_test(ctx, "memory.files.report:total", "e1/request", {"kind": "returns", "value": 6}).outcome == (
        "as_stated"
    )
    raised = wt.run_test(ctx, "memory.files.report:boom", "e1/request", {"kind": "raises", "exception": "KeyError"})
    assert raised.outcome == "as_stated"
    wrong = wt.run_test(ctx, "memory.files.report:boom", "e1/request", {"kind": "returns", "value": 1})
    assert wrong.outcome == "error" and "raised KeyError" in wrong.detail


def test_calls_that_are_not_tests_are_refused_and_nothing_is_recorded(tmp_path):
    ctx = _ctx(tmp_path, modes=("value",))
    # rev 2: one mode (a value test), so run_test takes no mode and the mode refusals go; an item that does not take
    # one input (a state form) is refused with what to do instead
    cases = [
        (("notes/files/how.md", "e1", {"kind": "returns", "value": 1}), "tested by reading"),
        (("memory.files.report:nope", "e1/request", {"kind": "returns", "value": 1}), "defines no function"),
        (("memory.files.report:total", "e9/request", {"kind": "returns", "value": 1}), "no recorded episode"),
        (("memory.files.report:total", "e1/cell:7", {"kind": "returns", "value": 1}), "no such part"),
        (("memory.files.report:total", "e1/request", {"kind": "maybe"}), "expect['kind'] is one of"),
        (
            ("memory.files.report:write_report", "e1", {"kind": "returns", "value": None}),
            "takes 2 positional arguments (state, env); a test hands it one recorded input",
        ),
    ]
    for args, words in cases:
        with pytest.raises(wt.WriterTestRefused, match=re.escape(words)):
            wt.run_test(ctx, *args)
    assert not ctx.tests_path.exists()




