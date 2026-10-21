"""In library mode the request's use record (``memory_use.json``) lists the library failures its cells met
(``library_failures``), beside the counts it already had. Each entry holds:

* ``kind``: ``item``, ``index``, ``show``, ``find``, ``match``, ``import`` or ``name``;
* ``target``: an item id, or the argument shape, never argument values;
* ``exception``: the exception's type name;
* ``message``: the first line of its message, redacted;
* ``cell``: the index of the cell it happened in.

A refused call (an item raising its ``MemoryInputError``) is listed too. The list is capped, with a count of the
entries dropped, and never holds a full traceback or a planted secret. Outside library mode the record has no list.

Cells run in the real sandboxed worker (bubblewrap); no model is called.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from tests.actor.code_act.sandbox_world import needs_bwrap, world  # noqa: F401
from tests.memory.integration.test_actor_export_library_mode import _library_repo
from tests.memory.integration.test_library_name_in_cells import _library_run, _run
from tests.memory.test_episodes import _ep
from unify.actor.execution.session import SessionExecutor
from unify.memory.integration.checkout import export_actor_library_mode
from unify.memory.integration.paths import Paths
from unify.memory.integration.request import memory_use

# ── ASSUMED NAMES (the integrator adapts them here, in one place) ────────────────────────────────────────────────
LIST_FIELD = "library_failures"
#: The count of entries left out once the list is full.
DROPPED_FIELD = "library_failures_dropped"
ENTRY_FIELDS = {"kind", "target", "exception", "message", "cell"}
KINDS = {"item", "index", "show", "find", "match", "import", "name"}


def _transcript(run: Any) -> list[dict]:
    """The episode's transcript lines for the cells the run noted, in order: each ``execute_code`` call and its
    result, under the call id ``_run`` gave it, so a failure's ``cell`` resolves as it does for a recorded episode."""
    lines: list[dict] = []
    for call in run.cell_status:
        lines.append(
            {
                "type": "message",
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {"id": call, "type": "function", "function": {"name": "execute_code", "arguments": "{}"}},
                    ],
                },
            },
        )
        lines.append({"type": "message", "message": {"role": "tool", "tool_call_id": call, "content": ""}})
    return lines


def use_record(run: Any, *, library_mode: bool = True) -> dict:
    """The request's use record built from what the run collected, as ``RequestRun`` builds ``memory_use.json``
    at the end of a request (the integrator adapts this call if the failures reach ``memory_use`` another way)."""
    return memory_use(
        _ep(transcript=_transcript(run), actions=[]),
        list(getattr(run, "item_ids", None) or []),
        runtime_calls=run.runtime_calls,
        cell_errors=dict(run.cell_errors),
        library_mode=library_mode,
        failures=run.failures,
    )


# ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────

EXPLODE = "memory.data.checks:explode"
PICKY = "memory.data.checks:picky"
PLANTED_KEY = "sk-planted-key-for-the-library-failure-test"  # pragma: allowlist secret
PLANTED_API_VALUE = "planted-api-key-value-0042"  # pragma: allowlist secret
PLANTED_ARG = "planted-argument-value-77"
SECOND_LINE = "second line of the failure"
CHECKS = (
    '"""Checks that fail on purpose."""\n\n\n'
    "class MemoryInputError(ValueError):\n"
    '    """The caller\'s input is not usable."""\n\n\n'
    "def explode(value):\n"
    '    """Raise with the argument in the message."""\n'
    f"    raise ValueError('cannot use {PLANTED_KEY} here\\n{SECOND_LINE} ' + repr(value))\n\n\n"
    "def picky(value):\n"
    '    """Refuse every input."""\n'
    f"    raise MemoryInputError('refused: api_key={PLANTED_API_VALUE}\\n{SECOND_LINE} ' + repr(value))\n"
)
NEVER = (PLANTED_KEY, PLANTED_API_VALUE, PLANTED_ARG, SECOND_LINE, "Traceback", 'File "')


def _library(world, monkeypatch):  # noqa: F811
    paths = Paths.under(world["state"])
    mem, sha = _library_repo(paths.memory, {"memory/data/checks.py": CHECKS})
    export_actor_library_mode(mem.git_dir, sha, paths.checkout)
    return _library_run(monkeypatch, paths)


def _entries(record: dict) -> list[dict]:
    assert "error" not in record, record
    entries = record[LIST_FIELD]
    assert isinstance(entries, list)
    for e in entries:
        assert ENTRY_FIELDS <= set(e), e
        assert e["kind"] in KINDS, e
        assert isinstance(e["exception"], str) and e["exception"].isidentifier(), e
        assert isinstance(e["message"], str) and "\n" not in e["message"], e
        assert isinstance(e["cell"], int) and not isinstance(e["cell"], bool) and e["cell"] >= 0, e
    text = json.dumps(record)
    for planted in NEVER:
        assert planted not in text, planted
    return entries


@needs_bwrap
@pytest.mark.asyncio
@pytest.mark.timeout(300)
async def test_failures_are_listed_by_kind_target_exception_message_and_cell(world, monkeypatch):  # noqa: F811
    run = _library(world, monkeypatch)
    ex = SessionExecutor(environments={})
    try:
        # one cell: an item that raises and an item that refuses, both caught
        res = await _run(
            ex,
            run,
            "from memory.data.checks import explode, picky\n"
            "for f in (explode, picky):\n"
            "    try:\n"
            f"        f({PLANTED_ARG!r})\n"
            "    except Exception as exc:\n"
            "        pass\n"
            "'caught'",
        )
        assert res["error"] is None, res["error"]
        # a later cell: the item again, uncaught, on a dict holding the planted value
        res = await _run(ex, run, f"explode({{'name': {PLANTED_ARG!r}, 'rows': [1, 2]}})")
        assert "ValueError" in res["error"]
        res = await _run(ex, run, "memory.show('memory.nothing.here:f')")
        assert "NotFound" in res["error"]
        res = await _run(ex, run, f"memory.index({PLANTED_KEY!r})")
        assert "NotFound" in res["error"]
        res = await _run(ex, run, "import memory.data.nope")
        assert "ModuleNotFoundError" in res["error"]
    finally:
        await ex.close()
    entries = _entries(use_record(run))
    items = [e for e in entries if e["kind"] == "item"]
    assert [(e["target"], e["exception"]) for e in items] == [
        (EXPLODE, "ValueError"),
        (PICKY, "MemoryInputError"),  # a refused call is listed too
        (EXPLODE, "ValueError"),
    ]
    # the first line only, redacted: the key-shaped value and the credential structure are gone
    assert items[0]["message"].startswith("cannot use ") and items[0]["message"].endswith(" here")
    assert items[1]["message"].startswith("refused: api_key=")
    # the two failures of one cell share its index; the later cell's is greater
    assert items[0]["cell"] == items[1]["cell"] < items[2]["cell"]
    (show,) = [e for e in entries if e["kind"] == "show"]
    assert show["exception"] == "NotFound"
    (index,) = [e for e in entries if e["kind"] == "index"]
    assert index["exception"] == "NotFound" and items[2]["cell"] < show["cell"] < index["cell"]
    (imported,) = [e for e in entries if e["kind"] == "import"]
    assert imported["exception"] == "ModuleNotFoundError"
    # targets are item ids or argument shapes, never values
    for e in entries:
        assert e["target"] is None or isinstance(e["target"], str)
    # the counts are still there and agree
    record = use_record(run)
    assert record["library_calls_failed"] == 5
    assert record["library_import_errors"] == 1
    assert record.get(DROPPED_FIELD, 0) == 0


@needs_bwrap
@pytest.mark.asyncio
@pytest.mark.timeout(300)
async def test_a_name_error_on_memory_is_listed(world, monkeypatch):  # noqa: F811
    paths = Paths.under(world["state"])
    paths.checkout.mkdir(parents=True)  # a library-mode request whose copy has no package
    run = _library_run(monkeypatch, paths)
    ex = SessionExecutor(environments={})
    try:
        res = await _run(ex, run, f"memory.find({{'name': {PLANTED_ARG!r}}})")
    finally:
        await ex.close()
    assert "NameError" in res["error"]
    (entry,) = _entries(use_record(run))
    assert entry["kind"] == "name" and entry["exception"] == "NameError"


@needs_bwrap
@pytest.mark.asyncio
@pytest.mark.timeout(300)
async def test_the_list_is_capped_and_counts_what_it_dropped(world, monkeypatch):  # noqa: F811
    total = 1000
    run = _library(world, monkeypatch)
    ex = SessionExecutor(environments={})
    try:
        res = await _run(
            ex,
            run,
            "from memory.data.checks import explode\n"
            f"for i in range({total}):\n"
            "    try:\n"
            f"        explode({PLANTED_ARG!r} + str(i))\n"
            "    except ValueError:\n"
            "        pass\n"
            "'done'",
        )
        assert res["error"] is None, res["error"]
    finally:
        await ex.close()
    record = use_record(run)
    entries = _entries(record)
    assert 0 < len(entries) < total
    assert len(entries) + record[DROPPED_FIELD] == total
    assert len(json.dumps(entries)) < 200_000


@needs_bwrap
@pytest.mark.asyncio
@pytest.mark.timeout(300)
async def test_outside_library_mode_there_is_no_list(world, monkeypatch):  # noqa: F811
    run = _library(world, monkeypatch)
    ex = SessionExecutor(environments={})
    try:
        res = await _run(ex, run, f"from memory.data.checks import explode\nexplode({PLANTED_ARG!r})")
    finally:
        await ex.close()
    assert "ValueError" in res["error"]
    assert LIST_FIELD in use_record(run)
    off = use_record(run, library_mode=False)
    assert LIST_FIELD not in off and DROPPED_FIELD not in off


def test_no_failures_give_an_empty_list():
    from unify.memory.integration.request import runtime_calls_empty

    record = memory_use(
        _ep(transcript=[], actions=[]),
        [],
        runtime_calls=runtime_calls_empty(),
        cell_errors={"name": 0, "import": 0},
    )
    assert record[LIST_FIELD] == []
    assert record.get(DROPPED_FIELD, 0) == 0
