"""Symbolic, end to end: a hierarchy of stored functions, stored and reused from code.

Under ``UNIFY_TOOL_SURFACE=core`` the libraries are the sandbox's
``functions`` and ``guidance`` objects. The lead's storage doctrine puts
composed functions first: a reusable entry point built from helpers that are
stored functions of their own. These tests store such a hierarchy (and the
guidance that explains it) from cells, then, in a later session of another
actor process on the same ``UNIFY_HOME`` store, find it from code, run it
with ``functions.run`` and by name, and check the results, state modes,
errors and tracebacks, recursion and cycles, the cases, trust and usage the
harness records for every level, what a patch, a rename and a delete do to
the callers, and readers running while another process writes.

Cells run in the real sandboxed worker (``tests/actor/code_act/library_world.py``),
search ranks with a deterministic embedder, and no model is called. The
tests are skipped where bubblewrap is missing.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
import time

import pytest

from tests.actor.code_act.core_world import core_world, world  # noqa: F401
from tests.actor.code_act.library_world import (
    ENTRY,
    FORMAT,
    GUIDANCE_CONTENT,
    GUIDANCE_TITLE,
    HELPERS,
    HIERARCHY,
    PARSE,
    REPO,
    SUMMARY,
    TEXT,
    TOTAL,
    Cells,
    install_fake_embed,
    new_actor,
    run_in_another_process,
    stdout,
)
from tests.actor.code_act.sandbox_world import needs_bwrap
from tests.helpers import _handle_project
from unify import db
from unify.actor import core_surface
from unify.settings import SETTINGS

NAMES = sorted([*HELPERS, "summarize_pairs"])


@pytest.fixture
def library(core_world, monkeypatch):  # noqa: F811
    """The core world with a deterministic embedder, cases and trust on."""
    install_fake_embed(monkeypatch.setattr)
    monkeypatch.setattr(SETTINGS, "UNIFY_FUNCTION_CASES", True)
    monkeypatch.setattr(SETTINGS, "UNIFY_STORE_TRUST", "ramp")
    monkeypatch.setattr(SETTINGS, "UNIFY_BUILTIN_GUIDANCE", False)
    return core_world


def _store_hierarchy() -> None:
    """The hierarchy, stored by this process's function manager."""
    from unify.function_manager.function_manager import FunctionManager

    FunctionManager(include_primitives=False).add_functions(
        implementations=list(HIERARCHY),
    )


def _rows(sql: str, *params) -> list[dict]:
    return [dict(r) for r in db.query(sql, params)]


def _cases_by_name() -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for row in _rows(
        "SELECT f.name AS name, c.* FROM function_cases c "
        "JOIN functions f ON f.function_id = c.function_id ORDER BY c.case_id",
    ):
        out.setdefault(row["name"], []).append(row)
    return out


def _usage() -> dict[str, int]:
    return {
        r["name"]: int(r["usage_calls"] or 0)
        for r in _rows("SELECT name, usage_calls FROM functions")
    }


async def _settled_usage(expected) -> dict[str, int]:
    """The usage counts once the off-loop usage writes have landed (or after 10 s)."""
    deadline = time.monotonic() + 10
    while True:
        usage = _usage()
        if expected(usage) or time.monotonic() > deadline:
            return usage
        await asyncio.sleep(0.1)


def _trust() -> dict[str, tuple[int, int]]:
    return {
        r["name"]: (int(r["passes"]), int(r["failures"]))
        for r in _rows(
            "SELECT f.name AS name, t.passes AS passes, t.failures AS failures "
            "FROM function_trust t JOIN functions f ON f.function_id = t.function_id",
        )
    }


# ── 1 & 2: store from code, find from code in another process ───────────────


@needs_bwrap
@pytest.mark.asyncio
@pytest.mark.timeout(300)
@_handle_project
async def test_a_hierarchy_stored_from_code_is_found_from_code_in_another_process(
    library,
):
    hierarchy = json.dumps(list(HIERARCHY))
    first = run_in_another_process(
        [
            f"await functions.add({hierarchy})",
            "entry = await functions.get('summarize_pairs')\n"
            "await guidance.add(title="
            + repr(GUIDANCE_TITLE)
            + ", content="
            + repr(GUIDANCE_CONTENT)
            + ", function_ids=[entry['function_id']])",
            # Stored, the entry point is callable at once by name and by run.
            "(summarize_pairs(" + repr(TEXT) + "), "
            "await functions.run('summarize_pairs', text=" + repr(TEXT) + "))",
        ],
    )
    added, guided, ran = first
    assert added["error"] is None, added["error"]
    assert added["result"] == {name: "added" for name in NAMES}
    assert guided["error"] is None, guided["error"]
    assert guided["result"]["outcome"] == "guidance created successfully"
    assert ran["error"] is None and ran["result"] == [SUMMARY, SUMMARY], ran

    # A later session: a new actor and a new worker over the same store file.
    db.reset_store()
    cells = Cells(new_actor(can_store=False))
    try:
        out = await cells("sorted(await functions.list())")
        assert out.error is None and out.result == NAMES, out.error
        out = await cells("await functions.get('summarize_pairs')")
        assert out.error is None, out.error
        entry = out.result
        assert entry["implementation"] == ENTRY
        assert entry["argspec"] == "(text: str, sep: str = '; ') -> str"
        assert entry["docstring"].startswith("Summarise comma-separated key=value")
        assert sorted(entry["depends_on"]) == sorted(HELPERS)
        for helper, source in zip(HELPERS, (PARSE, TOTAL, FORMAT)):
            out = await cells(f"(await functions.get({helper!r}))['implementation']")
            assert out.result == source, (helper, out.error)
        out = await cells(
            "[r['name'] for r in await functions.search("
            "'summarise key=value pairs as per-key totals', n=4)]",
        )
        assert out.error is None and out.result[0] == "summarize_pairs", out
        out = await cells(
            "[(g.title, g.function_ids) for g in await guidance.search('summarising key value pairs')]",
        )
        assert out.error is None, out.error
        assert out.result[0] == (GUIDANCE_TITLE, [entry["function_id"]])
        out = await cells(
            "g = (await guidance.search('summarising key value pairs'))[0]\n"
            "(await guidance.get(g.guidance_id)).content",
        )
        assert out.result == GUIDANCE_CONTENT, out.error
        # help() prints the library's contracts and the stored function's own doc.
        out = await cells("help(functions.get)\nhelp(summarize_pairs)")
        text = stdout(out)
        assert out.error is None, out.error
        assert "await functions.get(name: str)" in text
        assert "summarize_pairs(text: 'str', sep: 'str' = '; ') -> 'str'" in text
        assert "Composes parse_pairs, total_by_key and format_totals." in text
    finally:
        await cells.close()


@needs_bwrap
@pytest.mark.asyncio
@pytest.mark.timeout(240)
@_handle_project
async def test_store_time_checks_refuse_or_warn_from_code(library, monkeypatch):
    """What ``functions.add`` refuses (a third-party import without a declared
    dependency, a dangerous call, two functions in one source, an invalid
    requirement) and what it stores with a warning (the async check, the
    instance lint) reaches the cell as it reaches the JSON tool."""
    from unify.function_manager import instance_lint
    from unify.function_manager.primitives import (
        EnvironmentMethod,
        EnvironmentNamespace,
        EnvironmentSurface,
        register_environment,
    )
    from unify.function_manager.primitives.environment import (
        clear_environment_namespaces,
    )

    monkeypatch.setattr(SETTINGS, "UNIFY_STORE_ASYNC_CHECK", True)
    monkeypatch.setattr(SETTINGS, "UNIFY_STORE_INSTANCE_LINT", True)
    clear_environment_namespaces()
    register_environment(
        EnvironmentSurface(
            namespaces=(
                EnvironmentNamespace(
                    name="shop",
                    methods=(
                        EnvironmentMethod(
                            name="list_items",
                            call=lambda: [1, 2],
                            effect="read",
                        ),
                    ),
                ),
            ),
        ),
        source="tests:shop",
    )
    token = instance_lint.enter("Total the order task-7a4cf12e for the shop.")
    cells = Cells(new_actor())
    try:
        out = await cells(
            "await functions.add('def use_numpy(x):\\n    import numpy_ext_e2e\\n"
            "    return x\\n', raise_on_error=False)",
        )
        assert "no dependencies were provided" in out.result["use_numpy"], out
        out = await cells(
            "await functions.add('def bad(x):\\n    return eval(x)\\n', "
            "raise_on_error=False)",
        )
        assert out.result["bad"].startswith("error:"), out
        out = await cells(
            "await functions.add('def a():\\n    return 1\\ndef b():\\n    return 2\\n',"
            " raise_on_error=False)",
        )
        assert out.error is not None or any(
            v.startswith("error") for v in out.result.values()
        ), out
        out = await cells(
            "await functions.add('def f():\\n    return 1\\n', "
            "dependencies=['not a requirement !!'])",
        )
        assert "ValueError" in out.error and "not a valid requirement" in out.error
        out = await cells(
            "await functions.add('async def count_items() -> int:\\n"
            '    """Count the items of task-7a4cf12e."""\\n'
            "    return len(await primitives.shop.list_items())\\n')",
        )
        assert out.error is None, out.error
        status = out.result["count_items"]
        assert status.startswith("added"), status
        assert "warning" in status and "task-7a4cf12e" in status, status
        assert "primitives.shop.list_items" in status, status
        # A name that already exists is skipped unless overwritten.
        out = await cells(
            "await functions.add('async def count_items() -> int:\\n    return 0\\n')",
        )
        assert out.result == {"count_items": "skipped: already exists"}, out
        # Nothing refused was stored.
        out = await cells("sorted(await functions.list())")
        assert out.result == ["count_items"], out
    finally:
        await cells.close()
        instance_lint.leave(token)
        clear_environment_namespaces()


# ── 3: run accurately ───────────────────────────────────────────────────────


@needs_bwrap
@pytest.mark.asyncio
@pytest.mark.timeout(240)
@_handle_project
@pytest.mark.parametrize("state", ["stateless", "stateful", "read_only"])
async def test_functions_run_resolves_stored_helpers_in_a_fresh_session(
    library,
    state,
):
    """The first cell of a new session runs the entry point by name: its
    helpers are stored functions the session has not read, and are found."""
    _store_hierarchy()
    cells = Cells(new_actor(can_store=False))
    try:
        out = await cells(
            f"await functions.run('summarize_pairs', state={state!r}, text={TEXT!r})",
        )
        assert out.error is None, out.error
        assert out.result == SUMMARY
        out = await cells(
            f"await functions.run('summarize_pairs', state={state!r}, "
            f"text={TEXT!r}, sep=' | ')",
        )
        assert out.result == "a: 4 | b: 2", out.error
    finally:
        await cells.close()


@needs_bwrap
@pytest.mark.asyncio
@pytest.mark.timeout(240)
@_handle_project
async def test_stored_functions_call_each_other_by_name_in_cells(library):
    _store_hierarchy()
    cells = Cells(new_actor(can_store=False))
    try:
        out = await cells("await functions.get('summarize_pairs')")
        assert out.error is None, out.error
        out = await cells(
            f"(summarize_pairs({TEXT!r}), parse_pairs('x=1'), "
            "format_totals(total_by_key([('k', 2), ('k', 3)])))",
        )
        assert out.error is None, out.error
        assert out.result == (SUMMARY, [["x", 1]], "k: 5")
        # Code written in a cell composes stored functions like any others.
        out = await cells(
            "def doubled_summary(text):\n"
            "    totals = total_by_key(parse_pairs(text))\n"
            "    return format_totals({k: 2 * v for k, v in totals.items()})\n"
            f"doubled_summary({TEXT!r})",
        )
        assert out.result == "a: 8; b: 4", out.error
    finally:
        await cells.close()


@needs_bwrap
@pytest.mark.asyncio
@pytest.mark.timeout(240)
@_handle_project
async def test_a_helpers_exception_reaches_the_caller_with_its_traceback(library):
    _store_hierarchy()
    cells = Cells(new_actor(can_store=False))
    try:
        out = await cells("await functions.run('summarize_pairs', text='a=1, b=x')")
        assert out.result is None
        error = out.error
        assert "ValueError" in error and "invalid literal for int()" in error
        # Every level of the stored hierarchy is a frame, with its source line.
        assert "<function:summarize_pairs>" in error, error
        assert "<function:parse_pairs>" in error, error
        assert "pairs.append([key.strip(), int(value)])" in error, error
        # Called by name (bound by a read, from the next cell on), the same.
        await cells("await functions.get('summarize_pairs')")
        out = await cells(
            "try:\n"
            "    summarize_pairs('a=1, b=x')\n"
            "except ValueError as exc:\n"
            "    caught = type(exc).__name__\n"
            "caught",
        )
        assert out.result == "ValueError", out.error
        # A missing argument is the caller's TypeError, raised in the cell.
        out = await cells("await functions.run('summarize_pairs')")
        assert "TypeError" in out.error and "text" in out.error, out.error
        # The session goes on.
        out = await cells(f"await functions.run('summarize_pairs', text={TEXT!r})")
        assert out.result == SUMMARY, out.error
    finally:
        await cells.close()


FACT = (
    "def factorial(n: int) -> int:\n"
    '    """n! by recursion."""\n'
    "    return 1 if n <= 1 else n * factorial(n - 1)\n"
)
EVEN = (
    "def is_even(n: int) -> bool:\n"
    '    """Whether n is even, by mutual recursion with is_odd."""\n'
    "    return True if n == 0 else is_odd(n - 1)\n"
)
ODD = (
    "def is_odd(n: int) -> bool:\n"
    '    """Whether n is odd, by mutual recursion with is_even."""\n'
    "    return False if n == 0 else is_even(n - 1)\n"
)
FOREVER = (
    "def descend(n: int) -> int:\n"
    '    """Never stops: no base case."""\n'
    "    return descend(n + 1)\n"
)


@needs_bwrap
@pytest.mark.asyncio
@pytest.mark.timeout(300)
@_handle_project
async def test_recursion_and_cycles_resolve_and_end(library):
    from unify.function_manager.function_manager import FunctionManager

    fm = FunctionManager(include_primitives=False)
    assert fm.add_functions(implementations=[FACT, EVEN, ODD, FOREVER]) == {
        "factorial": "added",
        "is_even": "added",
        "is_odd": "added",
        "descend": "added",
    }
    cells = Cells(new_actor(can_store=False))
    try:
        out = await cells(
            "(await functions.run('factorial', n=6), "
            "await functions.run('is_even', n=10), "
            "await functions.run('is_odd', n=7))",
        )
        assert out.error is None, out.error
        assert out.result == (720, True, True)
        out = await cells(
            "await functions.get('is_even')\nawait functions.get('factorial')",
        )
        out = await cells("(factorial(5), is_even(3), is_odd(3))")
        assert out.result == (120, False, True), out.error
        # Unbounded recursion ends in RecursionError, not a hang, and the
        # worker survives it.
        started = time.monotonic()
        out = await cells("await functions.run('descend', n=0)")
        assert "RecursionError" in out.error, out.error
        assert time.monotonic() - started < 120
        out = await cells("1 + 1")
        assert out.result == 2, out.error
    finally:
        await cells.close()


# ── 5: recording ────────────────────────────────────────────────────────────


@needs_bwrap
@pytest.mark.asyncio
@pytest.mark.timeout(300)
@_handle_project
async def test_every_level_of_the_hierarchy_is_recorded(library):
    """``functions.run`` records the entry point (usage, case, trust) and each
    helper call it makes; a call by name does the same."""
    _store_hierarchy()
    cells = Cells(new_actor(can_store=False))
    try:
        out = await cells(f"await functions.run('summarize_pairs', text={TEXT!r})")
        assert out.result == SUMMARY, out.error
        cases = _cases_by_name()
        assert sorted(cases) == NAMES, cases
        entry = cases["summarize_pairs"][0]
        assert entry["args_shown"] == f"text={TEXT!r}"
        assert json.loads(entry["result"]) == SUMMARY or SUMMARY in entry["result"]
        usage = await _settled_usage(lambda u: u == {n: 1 for n in NAMES})
        assert usage == {name: 1 for name in NAMES}, usage
        assert _trust() == {name: (1, 0) for name in NAMES}
        # A stateless run binds nothing; a read binds the entry point (and
        # its helpers) for calls by name, which are recorded the same way.
        out = await cells("await functions.get('summarize_pairs')")
        out = await cells("summarize_pairs('z=5')")
        assert out.result == "z: 5", out.error
        usage = await _settled_usage(
            lambda u: u["summarize_pairs"] == 2 and u["parse_pairs"] == 2,
        )
        assert usage == {name: 2 for name in NAMES}, usage
        failing = await cells("await functions.run('summarize_pairs', text='q=nope')")
        assert "ValueError" in failing.error
        trust = _trust()
        assert trust["parse_pairs"][1] == 1 and trust["summarize_pairs"][1] == 1, trust
        assert trust["total_by_key"][1] == 0, trust
    finally:
        await cells.close()


async def _calls_both_ways(monkeypatch, *, core: bool) -> dict:
    """The entry point found, then run twice, through ``execute_function``
    (switch off) or ``functions.run`` (switch on), in a fresh store."""
    from unify.actor.execution import PythonExecutionSession, _CURRENT_SANDBOX

    db.clear()
    monkeypatch.setattr(SETTINGS, "UNIFY_TOOL_SURFACE", "core" if core else "")
    _store_hierarchy()
    actor = new_actor(can_store=False)
    tools = actor.get_tools("act")
    sandbox = PythonExecutionSession(environments={})
    if core:
        objects = core_surface.sandbox_objects(actor, policy=core_surface.WritePolicy())
        sandbox.global_state.update(objects)
        sandbox.core_globals = objects
    token = _CURRENT_SANDBOX.set(sandbox)
    outs = []
    try:
        if core:
            await tools["execute_code"].fn(
                thought="Find it.",
                code="await functions.search('summarise key value pairs')",
            )
        else:
            await tools["FunctionManager_search_functions"].fn(
                query="summarise key value pairs",
            )
        for text in (TEXT, "c=7"):
            if core:
                out = await tools["execute_code"].fn(
                    thought="Run it.",
                    code=f"await functions.run('summarize_pairs', text={text!r})",
                )
            else:
                out = await tools["execute_function"].fn(
                    thought="Run it.",
                    function_name="summarize_pairs",
                    call_kwargs={"text": text},
                    state_mode="stateless",
                )
            outs.append((out.result, out.error))
    finally:
        _CURRENT_SANDBOX.reset(token)
        await sandbox.close()
        await actor.close()
    cases = {
        name: [(c["args_shown"], c["error"] is None) for c in rows]
        for name, rows in _cases_by_name().items()
    }
    usage = await _settled_usage(lambda u: u.get("summarize_pairs", 0) >= 2)
    return {"outs": outs, "cases": cases, "usage": usage, "trust": _trust()}


@needs_bwrap
@pytest.mark.asyncio
@pytest.mark.timeout(300)
@_handle_project
async def test_functions_run_records_the_entry_point_as_execute_function_does(
    library,
    monkeypatch,
):
    """Parity on the entry point with ``execute_function`` on the default
    surface (worker Python both ways); the helper calls are recorded only
    under core, where they run as recorded stored functions."""
    shipped = await _calls_both_ways(monkeypatch, core=False)
    core = await _calls_both_ways(monkeypatch, core=True)
    assert core["outs"] == shipped["outs"] == [(SUMMARY, None), ("c: 7", None)]
    assert core["cases"]["summarize_pairs"] == shipped["cases"]["summarize_pairs"]
    assert core["usage"]["summarize_pairs"] == shipped["usage"]["summarize_pairs"]
    assert core["trust"]["summarize_pairs"] == shipped["trust"]["summarize_pairs"]
    for helper in HELPERS:
        assert len(core["cases"][helper]) == 2, core["cases"]
        assert core["trust"][helper] == (2, 0), core["trust"]


# ── 6: patch, rename, delete, concurrent readers ────────────────────────────


@needs_bwrap
@pytest.mark.asyncio
@pytest.mark.timeout(300)
@_handle_project
async def test_a_helper_patch_is_replayed_against_the_calls_the_entry_point_made(
    library,
    monkeypatch,
):
    """The helper's cases were recorded through the entry point, so a patch
    that changes what the helper did there is refused, naming the new-name
    route; one that keeps it is stored and the entry point runs it."""
    monkeypatch.setattr(SETTINGS, "UNIFY_FUNCTION_PATCH", True)
    _store_hierarchy()
    cells = Cells(new_actor())
    try:
        out = await cells(f"await functions.run('summarize_pairs', text={TEXT!r})")
        assert out.result == SUMMARY, out.error
        out = await cells(
            "await functions.patch('total_by_key', "
            "'totals[key] = totals.get(key, 0) + value', "
            "'totals[key] = totals.get(key, 0) + 10 * value', "
            "why='scale the totals')",
        )
        assert out.error is None, out.error
        refusal = json.dumps(out.result)
        assert "error" in out.result, refusal
        assert "does something else" in refusal, refusal
        out = await cells(
            "await functions.patch('total_by_key', "
            "'totals[key] = totals.get(key, 0) + value', "
            "'totals[key] = value + totals.get(key, 0)', "
            "why='same sum, operands swapped')",
        )
        assert out.result.get("status") == "patched", out.result
        out = await cells(f"await functions.run('summarize_pairs', text={TEXT!r})")
        assert out.result == SUMMARY, out.error
        # A patch starts the helper's trust over, and its callers': a caller's
        # trust covers the sources of the stored functions it calls.
        trust = _trust()
        assert trust["total_by_key"] == (1, 0), trust
        assert trust["summarize_pairs"] == (1, 0), trust
        assert trust["parse_pairs"] == (2, 0), trust
    finally:
        await cells.close()


@needs_bwrap
@pytest.mark.asyncio
@pytest.mark.timeout(300)
@_handle_project
async def test_renaming_and_deleting_a_helper_the_entry_point_uses(library):
    _store_hierarchy()
    cells = Cells(new_actor())
    try:
        # A rename is a new function under the new name; the old one stays
        # until it is deleted, and the entry point still calls it.
        renamed = PARSE.replace("def parse_pairs(", "def parse_key_values(")
        out = await cells(f"await functions.add({renamed!r})")
        assert out.result == {"parse_key_values": "added"}, out
        out = await cells(f"await functions.run('summarize_pairs', text={TEXT!r})")
        assert out.result == SUMMARY, out.error
        # Deleting the helper but keeping its callers marks them stale ...
        out = await cells(
            "fid = (await functions.get('parse_pairs'))['function_id']\n"
            "await functions.delete(fid, delete_dependents=False)",
        )
        assert out.result == {"parse_pairs": "deleted"}, out
        out = await cells("(await functions.get('summarize_pairs'))['stale_reasons']")
        assert "parse_pairs" in json.dumps(out.result), out
        # ... and a later session running the entry point is told which
        # name is missing. (A session that loaded the helper keeps its copy
        # until it ends, as a session keeps any function it defined.)
        later = Cells(new_actor())
        try:
            out = await later(f"await functions.run('summarize_pairs', text={TEXT!r})")
            assert "NameError" in out.error and "parse_pairs" in out.error, out.error
            # Deleting a helper with its dependents (the default) deletes the
            # entry point too, and says so.
            out = await later(
                "fid = (await functions.get('format_totals'))['function_id']\n"
                "await functions.delete(fid)",
            )
            assert out.result == {
                "format_totals": "deleted",
                "summarize_pairs": "deleted",
            }, out
            out = await later("sorted(await functions.list())")
            assert out.result == ["parse_key_values", "total_by_key"], out
        finally:
            await later.close()
        last = Cells(new_actor())
        try:
            out = await last(f"await functions.run('summarize_pairs', text={TEXT!r})")
            assert "NameError" in out.error and "summarize_pairs" in out.error
        finally:
            await last.close()
    finally:
        await cells.close()


WRITER = """
import sys
from unify.function_manager.function_manager import FunctionManager
fm = FunctionManager(include_primitives=False)
for i in range(int(sys.argv[1])):
    fm.add_functions(
        implementations=[f"def added_{i}(x: int) -> int:\\n    return x + {i}\\n"],
    )
print("wrote", flush=True)
"""


@needs_bwrap
@pytest.mark.asyncio
@pytest.mark.timeout(300)
@_handle_project
async def test_readers_see_a_consistent_library_while_another_process_writes(
    library,
):
    import os

    _store_hierarchy()
    env = {
        **os.environ,
        "PYTHONPATH": str(REPO),
        "UNIFY_VALIDATE_LLM_PROVIDERS": "false",
    }
    env.pop("UNIFY_STORE_PATH", None)
    writer = subprocess.Popen(
        [sys.executable, "-c", WRITER, "40"],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    cells = Cells(new_actor(can_store=False))
    counts = []
    try:
        while writer.poll() is None and len(counts) < 200:
            out = await cells(
                "names = await functions.list()\n"
                f"(len(names), await functions.run('summarize_pairs', text={TEXT!r}))",
            )
            assert out.error is None, out.error
            counts.append(out.result[0])
            assert out.result[1] == SUMMARY
        _, err = writer.communicate(timeout=120)
        assert writer.returncode == 0, err[-2000:]
        out = await cells("len(await functions.list())")
        assert out.result == 44, out
    finally:
        await cells.close()
        if writer.poll() is None:
            writer.kill()
    assert counts == sorted(counts), counts
    assert all(4 <= c <= 44 for c in counts), counts
