"""The use record when the worker binds ``memory`` in every cell before any runs (library mode with a library
copy): calls through that name with no import line count as they do after ``import memory``; a cell that rebinds
the name stops them; a request without the binding is not seeded."""

from __future__ import annotations

import json

from tests.memory.test_episodes import _ep
from tests.memory.test_use_library_mode import (  # noqa: F401
    ITEMS_LIBRARY,
    PD,
    TOK,
    WEEK,
    library_root,
)
from tests.memory.test_use_telemetry import _lines, _result
from unify.memory.analysis import use
from unify.memory.integration.request import RequestRun, memory_use

LOOKUPS = (
    "memory.index()\n"
    "memory.index('text')\n"
    "memory.show('memory.text.dates:parse_date')\n"
    "memory.text.dates.week('2026-10-09')\n"
)


def _status(cells, metas=None):
    run = RequestRun("r", None)
    run.item_ids = list(ITEMS_LIBRARY)
    for i, _ in enumerate(cells):
        run.note_result(f"c{i}", _result(None, metas[i] if metas else None))
    return run.cell_status


def _use(root, codes, metas=None, **kw):
    cells = [(code, None) for code in codes]
    return use.request_use(
        _lines(cells, metas=metas),
        ITEMS_LIBRARY,
        cell_status=_status(cells, metas),
        surface=use.library_surface(root),
        **kw,
    )


COUNTED = (
    "items",
    "module_imports",
    "unknown_calls",
    "cell_exposure",
    "items_outcome_unknown",
)


def test_calls_through_the_bound_name_count_with_no_import_line(library_root):
    rec = _use(library_root, [LOOKUPS])
    assert rec["memory_bound"] is True
    assert rec["cell_exposure"] == {
        "items": [PD, WEEK, TOK],
        "channels": ["text", "text.dates"],
        "calls": {"show": 1, "index": 2, "find": 0, "match": 0, "run": 0, "get": 0, "items": 0, "help": 0},
    }
    assert set(rec["items"]) == {WEEK} and rec["items"][WEEK]["called"] == 1
    assert rec["items"][WEEK]["cells"] == [0]


def test_the_bound_name_counts_exactly_as_after_an_import(library_root):
    bound = _use(library_root, [LOOKUPS])
    imported = _use(library_root, ["import memory\n" + LOOKUPS])
    unbound_imported = _use(
        library_root, ["import memory\n" + LOOKUPS], memory_bound=False
    )
    for key in COUNTED:
        assert bound[key] == imported[key] == unbound_imported[key], key
    # a later cell of the same session keeps the binding, as it keeps an import
    split = _use(library_root, ["x = 1\n", LOOKUPS])
    assert split["cell_exposure"] == bound["cell_exposure"]
    assert split["items"][WEEK]["cells"] == [1]


def test_a_request_without_the_binding_is_not_seeded(library_root):
    rec = _use(library_root, [LOOKUPS], memory_bound=False)
    assert rec["memory_bound"] is False
    assert rec["items"] == {}
    assert rec["cell_exposure"]["calls"] == {
        "show": 0,
        "index": 0,
        "find": 0,
        "match": 0,
        "run": 0,
        "get": 0,
        "items": 0,
        "help": 0,
    }
    assert rec["cell_exposure"]["items"] == []


def test_a_cell_that_rebinds_the_name_stops_the_counting(library_root):
    rec = _use(library_root, ["memory = 3\n" + LOOKUPS, LOOKUPS])
    assert rec["items"] == {}
    assert set(rec["cell_exposure"]["calls"].values()) == {0}
    # a new session starts with the name bound again
    metas = [
        {"session_id": 0, "session_created": True},
        {"session_id": 0, "session_created": False},
        {"session_id": 1, "session_created": True},
    ]
    rec = _use(library_root, ["memory = 3\n", LOOKUPS, LOOKUPS], metas)
    assert rec["items"][WEEK]["called"] == 1 and rec["items"][WEEK]["cells"] == [2]
    assert rec["cell_exposure"]["calls"]["index"] == 2


def test_the_request_record_passes_the_binding_through(library_root):
    cells = [(LOOKUPS, None)]
    ep = _ep(transcript=_lines(cells), actions=[])
    kw = {
        "cell_status": _status(cells),
        "surface": use.library_surface(library_root),
    }
    seeded = memory_use(ep, ITEMS_LIBRARY, **kw)
    assert seeded["memory_bound"] is True and seeded["items"][WEEK]["called"] == 1
    plain = memory_use(ep, ITEMS_LIBRARY, memory_bound=False, **kw)
    assert plain["memory_bound"] is False and plain["items"] == {}


def test_the_offline_analysis_reads_the_binding_from_the_record(library_root, tmp_path):
    cells = [(LOOKUPS, None)]
    lines = _lines(cells)
    live = _use(library_root, [LOOKUPS])

    def replay(name, recorded):
        d = tmp_path / name
        d.mkdir()
        (d / "transcript.jsonl").write_text(
            "".join(json.dumps(ln) + "\n" for ln in lines),
        )
        (d / "memory_use.json").write_text(json.dumps(recorded))
        return use.use_from_episode_dir(d)

    base = {k: live[k] for k in ("layout", "items_at_pin", "surface", "cell_status")}
    again = replay("bound", {**base, "memory_bound": True})
    assert again["memory_bound"] is True
    for key in COUNTED:
        assert again[key] == live[key], key
    off = replay("unbound", {**base, "memory_bound": False})
    assert off["memory_bound"] is False and off["items"] == {}
    # a record from before the flag was kept: library mode, as the worker decides it
    older = replay("older", base)
    assert older["memory_bound"] is True and older["items"] == live["items"]
