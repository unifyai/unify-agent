"""How the request's use record counts the actor's lookups in the library: calls of ``memory.match`` and
``memory.run`` among the package's helper calls, and ``library_file_lookups``, the file reads and greps of the
library copy (``/memory-library``), which no helper call records.

The unit tests feed the use record both sources it may count from: the cells' code in the transcript, and the
cells' drained audit records (``adapters.audit`` record schema). One test runs the cells in the real sandboxed
worker. No model is called.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from tests.actor.code_act.sandbox_world import needs_bwrap, world  # noqa: F401
from tests.memory.test_episodes import _ep
from tests.memory.test_use_telemetry import _lines
from unify.memory.integration.request import (
    add_runtime_calls,
    library_failures,
    library_helper_calls,
    memory_use,
    runtime_calls_empty,
)

# ==== ASSUMED NAMES (the integrator adapts these to the code, here only) ======================================
#: The new helpers' audit ids and their names in ``library_helper_calls``.
NEW_HELPERS = {"memory:match": "match", "memory:run": "run"}
#: The use record's field for file reads and greps of the library copy.
LOOKUPS_FIELD = "library_file_lookups"
#: Where the actor's cells see the library copy (``unify.sandbox.LIBRARY_MOUNT``).
LIBRARY = "/memory-library"


def lookups(record: dict) -> int:
    """The record's count of file lookups: the field as a number, or the sum of its parts when it counts reads and
    greps apart."""
    value = record[LOOKUPS_FIELD]
    if isinstance(value, dict):
        return sum(v for v in value.values() if isinstance(v, int))
    assert isinstance(value, int), value
    return value


# ==============================================================================================================

PARSE_DATE = "memory.text.dates:parse_date"
OUTCOMES = ("ok", "refused", "error")


# --- match and run among the helper calls --------------------------------------------------------------------


def test_match_and_run_are_counted_among_the_helper_calls_by_outcome():
    acc = runtime_calls_empty()
    zero = {k: 0 for k in OUTCOMES}
    empty = library_helper_calls(acc)
    for name in ("index", "show", "find", *NEW_HELPERS.values()):
        assert empty[name] == zero, name
    add_runtime_calls(
        acc,
        {
            "calls": {
                "helpers": {
                    "memory:match": {"ok": 2},
                    "memory:run": {"ok": 1, "error": 1},
                    "memory:index": {"ok": 1},
                },
            },
        },
    )
    add_runtime_calls(acc, {"calls": {"helpers": {"memory:run": {"refused": 1}}}})
    got = library_helper_calls(acc)
    assert got["match"] == {"ok": 2, "refused": 0, "error": 0}
    assert got["run"] == {"ok": 1, "refused": 1, "error": 1}
    assert got["index"] == {"ok": 1, "refused": 0, "error": 0}
    assert acc["malformed"] == 0  # both ids are the package's helpers
    assert library_failures(acc, {})["library_calls_failed"] == 2


def test_the_library_mode_use_record_carries_match_and_run_counts():
    acc = runtime_calls_empty()
    add_runtime_calls(acc, {"calls": {"helpers": {"memory:match": {"ok": 1}, "memory:run": {"error": 1}}}})
    rec = memory_use(_ep(transcript=[], actions=[]), [PARSE_DATE], runtime_calls=acc)
    assert "error" not in rec, rec
    assert rec["library_helper_calls"]["match"] == {"ok": 1, "refused": 0, "error": 0}
    assert rec["library_helper_calls"]["run"] == {"ok": 0, "refused": 0, "error": 1}


def test_get_and_items_count_as_lookups_and_their_failures_are_listed():
    """``memory.get`` and ``memory.items`` raise ``memory.helper_call`` as ``memory:get`` and ``memory:items``: the
    audit hook keeps them, the use record counts them among the lookups, and a failed one is listed by its kind."""
    from unify.memory.integration.adapters.audit import CellAudit
    from unify.memory.integration.request import add_library_failures, library_failures_empty

    audit = CellAudit([])
    audit.begin()
    audit("memory.helper_call", ("memory:get", "ab12", "ok"))
    audit("memory.helper_call", ("memory:items", "ab12", "ok"))
    audit("memory.helper_call", ("memory:get", "ab12", "error"))
    audit("memory.call_failed", ("memory:get", "NotFound", "memory.get('x'): no item named 'x'.", "(str[1])"))
    drained = audit.drain()
    assert drained["calls"]["malformed"] == 0
    acc = runtime_calls_empty()
    add_runtime_calls(acc, drained)
    assert acc["malformed"] == 0
    got = library_helper_calls(acc)
    assert got["get"] == {"ok": 1, "refused": 0, "error": 1} and got["items"] == {"ok": 1, "refused": 0, "error": 0}
    assert library_failures(acc, {})["library_calls_failed"] == 1
    failures = library_failures_empty()
    add_library_failures(failures, drained)
    assert [(e["kind"], e["target"], e["exception"]) for e in failures["pending"]] == [
        ("get", "(str[1])", "NotFound"),
    ]


# --- file reads and greps of the library copy ----------------------------------------------------------------

READ_INDEX = f"print(open({LIBRARY + '/INDEX.md'!r}).read()[:200])\n"
GREP_NOTES = (
    "import subprocess\n"
    f"print(subprocess.run(['grep', '-rn', 'total', {LIBRARY + '/notes'!r}], capture_output=True, text=True).stdout)\n"
)
ELSEWHERE = (
    "print(open('/workspace/data.csv').read()[:10])\n"
    "import subprocess\n"
    "print(subprocess.run(['grep', '-rn', 'total', '/workspace'], capture_output=True, text=True).stdout)\n"
)
HELPERS_ONLY = "print(memory.index())\nprint(memory.show('memory.text.dates:parse_date'))\n"


def _drained(records: list[dict]) -> dict:
    """One cell's drained audit buffer (``CellAudit.drain``) holding *records*."""
    return {"records": records, "dropped": 0, "failed": 0, "bytes": 100}


def _open(path: str) -> dict:
    return {"event": "open", "path": path, "mode": "r", "tid": 1, "cell_thread": True}


def _spawn(argv: list[str]) -> dict:
    return {"event": "subprocess.Popen", "exe": "/usr/bin/" + argv[0], "argv": argv, "tid": 1, "cell_thread": True}


def _record(codes: list[str], drained: list[dict]) -> dict:
    acc = runtime_calls_empty()
    for d in drained:
        add_runtime_calls(acc, d)
    rec = memory_use(
        _ep(transcript=_lines([(c, None) for c in codes]), actions=[]),
        [PARSE_DATE],
        runtime_calls=acc,
        export_roots=[LIBRARY],
    )
    assert "error" not in rec, rec
    return rec


def test_reads_and_greps_of_the_library_copy_are_counted():
    rec = _record(
        [READ_INDEX, GREP_NOTES],
        [
            _drained([_open(LIBRARY + "/INDEX.md")]),
            _drained([_spawn(["grep", "-rn", "total", LIBRARY + "/notes"]), _open(LIBRARY + "/notes/text/dates.md")]),
        ],
    )
    assert lookups(rec) >= 2


def test_reads_elsewhere_and_helper_calls_are_not_file_lookups():
    rec = _record(
        [ELSEWHERE, HELPERS_ONLY],
        [
            _drained([_open("/workspace/data.csv"), _spawn(["grep", "-rn", "total", "/workspace"])]),
            _drained([]),
        ],
    )
    assert lookups(rec) == 0


def test_a_request_with_no_cells_records_zero_lookups():
    assert lookups(_record([], [])) == 0


# --- in the real worker --------------------------------------------------------------------------------------


@needs_bwrap
@pytest.mark.asyncio
@pytest.mark.timeout(240)
async def test_lookups_made_in_the_real_worker_reach_the_use_record(world, monkeypatch):  # noqa: F811
    from tests.memory.integration.test_actor_export_library_mode import _library_repo
    from tests.memory.cells import Cells
    from unify.memory.integration import request as request_mod
    from unify.memory.integration.checkout import export_actor_library_mode
    from unify.memory.integration.paths import Paths
    from unify.memory.integration.request import RequestRun
    from unify.settings import SETTINGS

    paths = Paths.under(world["state"])
    mem, sha = _library_repo(paths.memory)
    export_actor_library_mode(mem.git_dir, sha, paths.checkout)
    monkeypatch.setattr(SETTINGS, "UNIFY_MEMORY", "on")
    run = RequestRun("r", paths)
    monkeypatch.setattr(request_mod, "_CURRENT", run)
    codes = [
        f"text = open({LIBRARY + '/INDEX.md'!r}).read()\nlen(text) > 0",
        "import os\n"
        f"hits = [os.path.join(d, f) for d, _, fs in os.walk({LIBRARY!r}) for f in fs if f.endswith('.md')]\n"
        "found = [p for p in hits if 'memory' in open(p).read()]\nlen(found) > 0",
        "1 + 1",
    ]
    ex = Cells()
    results: list[Any] = []
    try:
        for i, code in enumerate(codes):
            res = await asyncio.wait_for(ex.execute(code=code, state_mode="stateful", session_id=0), timeout=90)
            run.note_result(f"c{i}", res)
            results.append(res)
    finally:
        await ex.close()
    assert [r["error"] for r in results] == [None, None, None]
    assert list(r["result"] for r in results) == [True, True, 2]
    rec = memory_use(
        _ep(transcript=_lines([(c, None) for c in codes]), actions=[]),
        run.item_ids,
        export_roots=run.export_roots or [LIBRARY],
        cell_status=dict(run.cell_status),
        runtime_calls=run.runtime_calls,
        cell_errors=dict(run.cell_errors),
    )
    assert "error" not in rec, rec
    assert lookups(rec) >= 2
