"""In library mode every actor cell has the library bound as ``memory``, imported by the first cell that uses it,
and the request's use record counts the library calls that failed and the cells whose error named ``memory``.

Cells run in the real sandboxed worker (bubblewrap); no model is called.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from tests.actor.code_act.sandbox_world import needs_bwrap, world  # noqa: F401
from tests.memory.integration.test_actor_export_library_mode import _library_repo
from tests.memory.test_episodes import _ep
from tests.memory.cells import Cells
from unify.memory.integration import hooks
from unify.memory.integration import request as request_mod
from unify.memory.integration.adapters.audit import CellAudit
from unify.memory.integration.checkout import export_actor_library_mode
from unify.memory.integration.paths import Paths
from unify.memory.integration.request import (
    RequestRun,
    add_runtime_calls,
    library_cell_error,
    library_failures,
    memory_use,
    runtime_calls_empty,
)
from unify.memory.library_export import generated_library_mode
from unify.settings import SETTINGS

PARSE_DATE = "memory.text.dates:parse_date"


def _library_run(monkeypatch, paths: Paths) -> RequestRun:
    monkeypatch.setattr(SETTINGS, "UNIFY_MEMORY", "on")
    run = RequestRun("r", paths)
    monkeypatch.setattr(request_mod, "_CURRENT", run)
    return run


def _write_copy(root: Path, files: dict) -> None:
    for rel, data in files.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_bytes(data if isinstance(data, bytes) else data.encode())


async def _run(ex: Cells, run: RequestRun | None, code: str) -> Any:
    res = await asyncio.wait_for(
        ex.execute(code=code, state_mode="stateful", session_id=0),
        timeout=90,
    )
    if run is not None:
        run.note_result(f"c{len(run.cell_status)}", res)
    return res


def _ok(res: Any) -> Any:
    assert res["error"] is None, res["error"]
    return res["result"]


# --- a real worker -------------------------------------------------------------------------------------------


@needs_bwrap
@pytest.mark.asyncio
@pytest.mark.timeout(240)
async def test_cells_use_memory_with_no_import_line_and_failed_calls_are_counted(
    world,  # noqa: F811
    monkeypatch,
):
    paths = Paths.under(world["state"])
    mem, sha = _library_repo(paths.memory)
    export_actor_library_mode(mem.git_dir, sha, paths.checkout)
    run = _library_run(monkeypatch, paths)
    ex = Cells()
    try:
        out = _ok(
            await _run(
                ex,
                run,
                "first = repr(memory)\n"
                "head = memory.index().splitlines()[0]\n"
                f"shown = memory.show({PARSE_DATE!r}).splitlines()[0]\n"
                "from memory.text.dates import parse_date\n"
                "[first, type(memory).__name__, head == memory.index().splitlines()[0], "
                "'parse_date' in shown, parse_date('2026-10-09')]",
            ),
        )
        # the first cell that names it imports it before its first line runs: the name holds the real package from
        # the start, and ``import memory`` is the same module
        assert out[0].startswith("<module 'memory' from ") and "first use" not in out[0], out[0]
        assert list(out)[1:] == ["module", True, True, "2026-10-09"]
        assert _ok(await _run(ex, run, "import memory as m\nm is memory")) is True
        # a library function that raises, and a helper that raises: both counted
        res = await _run(ex, run, "parse_date(None)")
        assert "AttributeError" in res["error"]
        res = await _run(ex, run, "memory.show('memory.nothing.here:f')")
        assert "NotFound" in res["error"] and "no item named 'memory.nothing.here:f'" in res["error"]
    finally:
        await ex.close()
    assert run.runtime_calls["counts"][PARSE_DATE]["ok"] >= 1
    assert run.runtime_calls["counts"][PARSE_DATE]["error"] == 1
    assert run.runtime_calls["helpers_failed"] == 1
    assert run.runtime_calls["helpers"]["index"]["ok"] >= 2  # memory.index() with no import line, twice
    assert library_failures(run.runtime_calls, run.cell_errors) == {
        "library_calls_failed": 2,
        "library_name_errors": 0,
        "library_import_errors": 0,
    }
    # and each failure is listed by the cell it happened in: the item's, then the helper's with its argument shape
    shown_id = "memory.nothing.here:f"
    assert [(e["kind"], e["target"], e["exception"], e["call"]) for e in run.failures["entries"]] == [
        ("item", PARSE_DATE, "AttributeError", "c2"),
        ("show", f"(str[{len(shown_id)}])", "NotFound", "c3"),
    ]
    assert run.failures["pending"] == [] and run.failures["dropped"] == 0
    assert run.failures["entries"][1]["message"].startswith(
        "memory.show('memory.nothing.here:f'): no item named 'memory.nothing.here:f'.",
    )


@needs_bwrap
@pytest.mark.asyncio
@pytest.mark.timeout(240)
async def test_an_empty_library_is_bound_too(world, monkeypatch, tmp_path):  # noqa: F811
    paths = Paths.under(world["state"])
    empty = tmp_path / "empty"
    empty.mkdir()
    _write_copy(paths.checkout, generated_library_mode(empty))
    assert (paths.checkout / "memory" / "__init__.py").is_file()
    run = _library_run(monkeypatch, paths)
    ex = Cells()
    try:
        out = _ok(
            await _run(
                ex,
                run,
                "['The library is empty.' in memory.index(), memory.find({'a': 1})]",
            ),
        )
    finally:
        await ex.close()
    assert list(out) == [True, []]


@needs_bwrap
@pytest.mark.asyncio
@pytest.mark.timeout(240)
async def test_a_library_that_fails_to_import_never_breaks_the_session(
    world,  # noqa: F811
    monkeypatch,
):
    paths = Paths.under(world["state"])
    # the copy's root is first on the import path: modules named like the worker's own must not shadow them
    _write_copy(
        paths.checkout,
        {
            "memory/__init__.py": "raise RuntimeError('this library is broken')\n",
            "json.py": "raise RuntimeError('shadowed json')\n",
            "os.py": "raise RuntimeError('shadowed os')\n",
        },
    )
    run = _library_run(monkeypatch, paths)
    ex = Cells()
    try:
        assert (
            _ok(await _run(ex, run, "import json, os\njson.dumps({'a': os.sep})"))
            == '{"a": "/"}'
        )
        res = await _run(ex, run, "memory.index()")
        assert "ImportError" in res["error"] and "this library is broken" in res["error"]
        assert _ok(await _run(ex, run, "1 + 1")) == 2
        # the next use tries again, and fails the same way
        res = await _run(ex, run, "memory.index()")
        assert "this library is broken" in res["error"]
    finally:
        await ex.close()
    assert run.cell_errors == {"name": 0, "import": 2}
    assert library_failures(run.runtime_calls, run.cell_errors)["library_import_errors"] == 2
    assert [(e["kind"], e["target"]) for e in run.failures["entries"]] == [("import", None), ("import", None)]
    assert {e["exception"] for e in run.failures["entries"]} <= {"ImportError", "ModuleNotFoundError"}


@needs_bwrap
@pytest.mark.asyncio
@pytest.mark.timeout(240)
async def test_a_library_that_never_finishes_importing_is_stopped_by_the_cell_timeout(
    world,  # noqa: F811
    monkeypatch,
):
    paths = Paths.under(world["state"])
    _write_copy(paths.checkout, {"memory/__init__.py": "while True:\n    pass\n"})
    run = _library_run(monkeypatch, paths)
    ex = Cells(limit=5)
    try:
        assert _ok(await _run(ex, run, "x = 1\nx")) == 1  # starting the worker imports nothing of it
        res = await _run(ex, run, "memory.index()")
        # rev 2's interpreter: the cell is stopped at its limit and the interpreter restarted, said in a note the
        # actor reads beside the cell's output (the cell has no error of its own)
        shown = "".join(p.text for p in res["stdout"])
        assert res["error"] is None and "(the cell timed out after 5 s; the interpreter was restarted" in shown, shown
        assert _ok(await _run(ex, run, "2 + 2")) == 4  # a fresh worker
    finally:
        await ex.close()


@needs_bwrap
@pytest.mark.asyncio
@pytest.mark.timeout(240)
async def test_without_a_library_cells_have_no_memory_and_a_name_error_is_counted(
    world,  # noqa: F811
    monkeypatch,
):
    monkeypatch.setattr(SETTINGS, "UNIFY_MEMORY", "off")  # rev 2: empty means on
    ex = Cells()
    try:
        probe = "try:\n    memory\n    bound = True\nexcept NameError:\n    bound = False\nbound"
        assert _ok(await _run(ex, None, probe)) is False
    finally:
        await ex.close()
    # a library-mode request whose copy has no package: no ``memory`` either, and its NameError is counted
    paths = Paths.under(world["state"])
    paths.checkout.mkdir(parents=True)
    run = _library_run(monkeypatch, paths)
    assert hooks.worker_globals() == {}
    ex = Cells()
    try:
        res = await _run(ex, run, "memory.index()")
    finally:
        await ex.close()
    assert "NameError" in res["error"]
    assert run.cell_errors == {"name": 1, "import": 0}


# --- no worker -----------------------------------------------------------------------------------------------


def test_the_init_global_only_in_library_mode_with_a_package(
    world,  # noqa: F811
    monkeypatch,
):
    from unify.actor.execution.worker import PythonWorker

    monkeypatch.setattr(SETTINGS, "UNIFY_MEMORY", "on")
    paths = Paths.under(world["state"])
    (paths.checkout / "memory").mkdir(parents=True)
    monkeypatch.setattr(
        request_mod,
        "_CURRENT",
        SimpleNamespace(index="", paths=paths, library_mode=True),
    )
    assert hooks.worker_globals() == {}  # no memory/__init__.py
    (paths.checkout / "memory" / "__init__.py").write_text("")
    assert hooks.worker_globals() == {"memory": ["lazy_module", "memory"]}
    assert PythonWorker()._init_message()["globals"]["memory"] == [
        "lazy_module",
        "memory",
    ]
    # memory off (rev 2: UNIFY_MEMORY=off; empty means on): no memory name
    monkeypatch.setattr(SETTINGS, "UNIFY_MEMORY", "off")
    assert "memory" not in PythonWorker()._init_message()["globals"]


def test_helper_calls_are_counted_apart_from_items():
    audit = CellAudit([])
    audit.begin()
    audit("memory.helper_call", ("memory:show", "ab12", "error"))
    audit("memory.helper_call", ("memory:index", "ab12", "ok"))
    audit("memory.helper_call", ("memory.text.dates:parse_date", "ab12", "ok"))
    audit("memory.call", (PARSE_DATE, "ab12", "refused"))
    calls = audit.drain()["calls"]
    assert calls["helpers"] == {
        "memory:show": {"ok": 0, "refused": 0, "error": 1},
        "memory:index": {"ok": 1, "refused": 0, "error": 0},
    }
    assert calls["malformed"] == 1 and set(calls["counts"]) == {PARSE_DATE}
    audit.begin()
    audit("memory.call", (PARSE_DATE, "ab12", "ok"))
    assert "helpers" not in audit.drain()["calls"]  # the block is unchanged without helper calls
    acc = runtime_calls_empty()
    add_runtime_calls(acc, {"calls": calls})
    assert acc["helpers_failed"] == 1
    assert library_failures(acc, {"name": 1})["library_calls_failed"] == 2


@pytest.mark.parametrize(
    "error, kind",
    [
        ("Traceback ...\nNameError: name 'memory' is not defined\n", "name"),
        ("NameError: name 'memoryx' is not defined", None),
        ("ModuleNotFoundError: No module named 'memory'", "import"),
        ("ModuleNotFoundError: No module named 'memory.text.nope'", "import"),
        ("ImportError: cannot import name 'f' from 'memory.text.dates' (/x)", "import"),
        ("ImportError: importing the module 'memory' failed (RuntimeError)\n", "import"),
        ("ModuleNotFoundError: No module named 'pandas'", None),
        ("NameError: name 'memory' is not defined\nValueError: later", None),
        (None, None),
    ],
)
def test_cell_errors_naming_the_library_are_classified(error, kind):
    assert library_cell_error(error) == kind


def test_the_library_mode_use_record_carries_the_counts():
    acc = runtime_calls_empty()
    add_runtime_calls(
        acc,
        {
            "calls": {
                "counts": {PARSE_DATE: {"ok": 1, "refused": 1, "error": 2}},
                "helpers": {"memory:find": {"ok": 0, "refused": 0, "error": 1}},
            },
        },
    )
    rec = memory_use(
        _ep(transcript=[], actions=[]),
        [PARSE_DATE],
        runtime_calls=acc,
        cell_errors={"name": 3, "import": 1},
    )
    assert "error" not in rec, rec
    assert (
        rec["library_calls_failed"],
        rec["library_name_errors"],
        rec["library_import_errors"],
    ) == (4, 3, 1)
    # rev 2 counts the librarian's memory.ask beside the other helpers
    assert rec["library_helper_calls"] == {
        "ask": {"ok": 0, "refused": 0, "error": 0},
        "get": {"ok": 0, "refused": 0, "error": 0},
        "items": {"ok": 0, "refused": 0, "error": 0},
        "index": {"ok": 0, "refused": 0, "error": 0},
        "show": {"ok": 0, "refused": 0, "error": 0},
        "find": {"ok": 0, "refused": 0, "error": 1},
        "match": {"ok": 0, "refused": 0, "error": 0},
        "run": {"ok": 0, "refused": 0, "error": 0},
    }
