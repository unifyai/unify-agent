"""Use telemetry (memory v2.1 stage 1): per-request use records, the evidence index, Sol's table.

The tracebacks here are real: each cell runs against a real ``env/<channel>/__init__.py`` package on a
temporary import path, and its traceback is formatted by the traceback module, as the worker does.
"""

from __future__ import annotations

import ast
import hashlib
import json
import random
import re
import sys
import traceback
from pathlib import Path

import pytest

from tests.memory_v2.test_episodes import _ep
from unify.actor.execution.types import ExecutionResult, TextPart
from unify.actor.notebook_cells import NotebookCellResult
from unify.memory_v2.analysis import use
from unify.memory_v2.blobs import BlobStore
from unify.memory_v2.episodes import (
    Action,
    EpisodeWriter,
    env_channel,
    episode_dir,
    load_episode,
)
from unify.memory_v2.evidence import _USE_COUNTS, EvidenceStore, _use_rows
from unify.memory_v2.gitio import Repo
from unify.memory_v2.index import HEADER, build_index, index_with_names
from unify.memory_v2.integration.prompt import render_index, render_memory
from unify.memory_v2.integration import hooks
from unify.memory_v2.integration.request import RequestRun, memory_use
from unify.memory_v2.redact import Redactor

MODULE = '''
class MemoryInputError(ValueError):
    pass


def parse(text):
    """Split a line.

    Effect: read
    Input: text
    """
    if not isinstance(text, str):
        raise MemoryInputError("expected a line of text")
    return text.split()


def lookup(key):
    """Look a key up.

    Effect: read
    """
    return {"a": 1}[key]


def _inner(key):
    raise KeyError(key)


def strict(key):
    """Refuse an unknown key after catching the lookup's error inside the module.

    Effect: read
    """
    try:
        return _inner(key)
    except KeyError as exc:
        raise MemoryInputError("unknown key") from exc
'''
ITEMS = ["env/x:lookup", "env/x:parse", "env/x:strict"]


# --- helpers ---------------------------------------------------------------------------------------------


@pytest.fixture
def lib(tmp_path, monkeypatch):
    """A memory export with one channel ``x`` on the import path; ``env`` modules are dropped after."""
    root = tmp_path / "checkout"
    (root / "env" / "x").mkdir(parents=True)
    (root / "env" / "x" / "__init__.py").write_text(MODULE)
    monkeypatch.syspath_prepend(str(root))

    def drop():
        for name in [m for m in sys.modules if m == "env" or m.startswith("env.")]:
            sys.modules.pop(name, None)

    drop()
    yield root
    drop()


def _session(codes: list[str]) -> list[tuple[str, str | None]]:
    """Run each cell in one namespace (a session); each cell's code and formatted traceback (or None)."""
    ns: dict = {}
    out = []
    for i, code in enumerate(codes):
        try:
            exec(
                compile(code, f"<cell {i}>", "exec"),
                ns,
            )  # noqa: S102 - the test's own cells
            err = None
        except Exception:  # noqa: BLE001
            err = traceback.format_exc()
        out.append((code, err))
    return out


PROJECTIONS = {"legacy": ExecutionResult, "notebook": NotebookCellResult}


def _result(
    err: str | None,
    meta: dict | None = None,
    printed: str = "printed\n",
    projection: str = "legacy",
) -> ExecutionResult:
    """The code tool's result for one cell, in either projection (``UNIFY_CODE_PROJECTION``)."""
    meta = meta or {}
    return PROJECTIONS[projection](
        stdout=[TextPart(text=printed)] if printed else [],
        error=err,
        duration_ms=3,
        session_id=meta.get("session_id"),
        session_created=meta.get("session_created"),
    )


def _lines(
    cells: list[tuple[str, str | None]],
    system: str = "core prompt",
    metas: list[dict] | None = None,
    projection: str = "legacy",
    printed: list[str] | None = None,
) -> list[dict]:
    """Transcript lines in the line format of ``unify.transcripts``, each cell's result rendered as the
    runtime renders it in *projection* (the legacy one starts with a metadata block, the notebook one with
    what the cell printed and ends with its traceback)."""
    out: list[dict] = [{"seq": 0, "type": "system_prompt", "content": system}]
    for i, (code, err) in enumerate(cells):
        out.append(
            {
                "seq": len(out),
                "type": "message",
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": f"c{i}",
                            "type": "function",
                            "function": {
                                "name": "execute_code",
                                "arguments": json.dumps({"code": code}),
                            },
                        },
                    ],
                },
            },
        )
        result = _result(
            err,
            metas[i] if metas else None,
            printed[i] if printed else "printed\n",
            projection,
        )
        out.append(
            {
                "seq": len(out),
                "type": "message",
                "message": {
                    "role": "tool",
                    "tool_call_id": f"c{i}",
                    "name": "execute_code",
                    "content": result.to_llm_content(),
                },
            },
        )
    return out


def _status(
    cells: list[tuple[str, str | None]],
    metas: list[dict] | None = None,
    *,
    items=ITEMS,
    roots=(),
    projection: str = "legacy",
    skip: tuple[int, ...] = (),
) -> dict:
    """Each cell's status as the harness notes it (``RequestRun.note_result``), by call id; cells in
    *skip* have none (a recording without it, or a tool call that failed outright)."""
    run = RequestRun("r", None)
    run.item_ids, run.export_roots = list(items), list(roots)
    for i, (_, err) in enumerate(cells):
        if i not in skip:
            meta = metas[i] if metas else None
            run.note_result(f"c{i}", _result(err, meta, projection=projection))
    return run.cell_status


def _use(codes: list[str], actions=()) -> dict:
    cells = _session(codes)
    return use.request_use(_lines(cells), ITEMS, actions, cell_status=_status(cells))


# --- imports and calls -----------------------------------------------------------------------------------


def test_a_cell_importing_and_calling_an_item_counts_one_import_and_one_call(lib):
    rec = _use(["from env.x import parse\nwords = parse('a b')\n"])
    assert rec["items"]["env/x:parse"] == {
        "imported": 1,
        "called": 1,
        "referenced": 0,
        "guarded": 0,
        "refused": 0,
        "errored": 0,
        "refused_modified": 0,
        "errored_modified": 0,
        "refused_then_accepted": 0,
        "modified_in_request": False,
        "cells": [0],
    }
    assert set(rec["items"]) == {"env/x:parse"}
    assert rec["unknown_calls"] == {} and rec["cells"] == 1


def test_an_alias_import_resolves_and_bindings_persist_across_cells(lib):
    rec = _use(["from env.x import parse as g\n", "g('a')\nh = g\nh('b')\n"])
    row = rec["items"]["env/x:parse"]
    assert (row["imported"], row["called"], row["cells"]) == (1, 2, [0, 1])
    assert row["referenced"] == 1  # ``h = g``


def test_module_and_package_access_resolve_to_the_item(lib):
    rec = _use(
        [
            "import env.x\nenv.x.parse('a')\n",
            "from env import x as m\nm.lookup('a')\n",
            "import env.x as mod\nmod.parse('b')\n",
        ],
    )
    assert rec["items"]["env/x:parse"]["called"] == 2
    assert rec["items"]["env/x:lookup"]["called"] == 1
    assert (
        rec["items"]["env/x:parse"]["imported"] == 0
    )  # the channel was imported, not the name
    assert rec["module_imports"] == {"x": 3}


def test_dynamic_calls_are_unknown_not_guessed(lib):
    rec = _use(
        [
            "import env.x as m\nname = 'pa' + 'rse'\ngetattr(m, name)('a')\n",
            "import env.x as m\nf = getattr(m, 'parse')\nf('b')\n",
            "import importlib\nimportlib.import_module('env.x').parse('c')\n",
            "import env.x as m\nvars(m)['lookup']('a')\n",
        ],
    )
    assert rec["unknown_calls"] == {"x": 4}
    assert all(r["called"] == 0 for r in rec["items"].values())


def test_shadowed_names_are_not_the_item(lib):
    rec = _use(
        [
            "from env.x import parse\n"
            "def use(parse):\n    return parse('a')\n"
            "use(str.split)\n"
            "[parse for parse in [len]]\n"
            "parse = len\nparse('abc')\n",
        ],
    )
    assert rec["items"]["env/x:parse"]["called"] == 0


def test_calls_inside_a_try_with_a_handler_are_marked_guarded(lib):
    rec = _use(
        ["from env.x import parse\ntry:\n    parse(1)\nexcept Exception:\n    pass\n"],
    )
    row = rec["items"]["env/x:parse"]
    assert (row["called"], row["guarded"], row["refused"]) == (1, 1, 0)


def test_bash_and_unparsable_cells_are_skipped_and_counted(lib):
    cells = [("ls -la", None), ("def (:\n", None)]
    lines = _lines(cells)
    lines[1]["message"]["tool_calls"][0]["function"]["arguments"] = json.dumps(
        {"code": "ls -la", "language": "bash"},
    )
    rec = use.request_use(lines, ITEMS)
    assert rec["cells"] == 2 and rec["unparsed_cells"] == 1 and rec["items"] == {}


# --- errors ----------------------------------------------------------------------------------------------


def test_a_memory_input_error_from_the_item_counts_as_refused_by_its_frames(lib):
    ((_, err),) = _session(["from env.x import parse\nparse(3)\n"])
    assert "MemoryInputError" in err.splitlines()[-1]
    assert use.attribute_errors(err, ITEMS) == [("refused", "env/x:parse")]
    cells = [("from env.x import parse\nparse(3)\n", err)]
    rec = use.request_use(_lines(cells), ITEMS, cell_status=_status(cells))
    assert rec["items"]["env/x:parse"]["refused"] == 1
    assert rec["cell_status"] == [
        {
            "call": "c0",
            "status": "error",
            "exits": [["refused", "x", "parse"]],
            "session": None,
            "fresh": False,
        },
    ]


def test_the_message_is_never_read(lib):
    """A cell's own exception whose message names the item, its file and the refusal type is not the item's;
    the item's refusal with an empty message is."""
    own = (
        "from env.x import parse\nparse('a')\n"
        "raise ValueError('MemoryInputError in env/x/__init__.py, in parse')\n"
    )
    silent = "from env.x import parse\nimport env.x\nraise env.x.MemoryInputError()\n"
    (_, own_err), (_, silent_err) = _session([own, silent])
    assert use.attribute_errors(own_err, ITEMS) == []
    assert (
        use.attribute_errors(silent_err, ITEMS) == []
    )  # raised in the cell, not in the item
    blocks = use.parse_traceback(own_err)
    assert [b["type"] for b in blocks] == ["ValueError"]


def test_an_error_raised_in_the_cells_own_code_is_not_attributed_to_the_item(lib):
    rec = _use(["from env.x import parse\nwords = parse('a b')\nwords[5]\n"])
    row = rec["items"]["env/x:parse"]
    assert (row["called"], row["refused"], row["errored"]) == (1, 0, 0)
    assert rec["unattributed_errors"] == {}


def test_other_exceptions_from_inside_the_item_count_as_errored(lib):
    rec = _use(["from env.x import lookup\nlookup('zz')\n"])
    row = rec["items"]["env/x:lookup"]
    assert (row["errored"], row["refused"]) == (1, 0)


def test_an_exception_caught_inside_the_item_counts_once_as_its_refusal(lib):
    ((_, err),) = _session(["from env.x import strict\nstrict('zz')\n"])
    assert len(use.parse_traceback(err)) == 2  # the KeyError, then the MemoryInputError
    assert use.attribute_errors(err, ITEMS) == [("refused", "env/x:strict")]


def test_a_refusal_caught_by_the_cell_that_then_fails_is_still_counted(lib):
    code = (
        "from env.x import parse\n"
        "try:\n    parse(3)\nexcept Exception:\n    raise RuntimeError('fallback failed')\n"
    )
    ((_, err),) = _session([code])
    assert use.attribute_errors(err, ITEMS) == [("refused", "env/x:parse")]


def test_a_refusal_then_an_accepted_action_on_its_channel(lib):
    codes = ["from env.x import parse\nparse(3)\n", "print('handled directly')\n"]
    ok = Action(1, "x", "parse_line", [], {}, None, "ok", "read")
    other = Action(1, "y", "call", [], {}, None, "ok", "read")
    assert _use(codes, [other])["items"]["env/x:parse"]["refused_then_accepted"] == 0
    rec = _use(codes, [ok])
    assert rec["items"]["env/x:parse"]["refused_then_accepted"] == 1


def test_env_channel_copy_matches_the_episodes_mapping():
    for kind, channel in [
        ("tool", "venmo"),
        ("shell", "shell:python3.12"),
        ("worktree", "worktree:workspace"),
        ("dialogue", "dialogue:user"),
        ("shell", "worktree:workspace"),
        ("shell", ""),
        ("shell", "shell:--"),
    ]:
        assert use.env_channel(kind, channel) == env_channel(kind, channel)


# --- the memory section ----------------------------------------------------------------------------------


def test_a_legacy_recording_finds_the_section_by_the_v2_header_and_says_so(lib):
    """Without the renderer's record (a recording from before it), the section is found by the v2 index's
    opening line, and the record says the shown lists came from that legacy reading."""
    index = build_index(lib)
    assert HEADER.startswith(use.HEADER_PREFIX)
    rec = use.request_use(_lines([], system=f"core prompt\n\n{index}"), ITEMS)
    shown = rec["memory_section_shown"]
    assert shown["shown"] and shown["distinct"] == 1 and shown["prompts"] == 1
    assert shown["sha256"] == hashlib.sha256(index.encode()).hexdigest()
    assert shown["bytes"] == len(index.encode()) and shown["est_tokens"] == -(
        -len(index) // 4
    )
    assert rec["exposure_source"] == "legacy_text" and shown["source"] == "legacy_text"
    assert rec["shown_record"] is None and shown["prompt_confirmed"] is None
    none = use.request_use(_lines([]), ITEMS)
    assert not none["memory_section_shown"]["shown"]
    assert none["memory_section_shown"]["sha256"] is None
    # nothing says whether a section was shown: unknown, never "shown nothing" from a record
    assert none["exposure_source"] == "unknown" and none["shown_items"] == []


# --- bounds, determinism, vendoring ----------------------------------------------------------------------


def test_the_record_is_deterministic_and_bounded(lib):
    codes = ["from env.x import parse\n" + "parse('a')\n" * 50]
    assert _use(codes) == _use(codes)
    many = [f"env/c{i}:f" for i in range(use.MAX_ITEMS_AT_PIN + 5)]
    rec = use.request_use(_lines([]), many)
    assert len(rec["items_at_pin"]) == use.MAX_ITEMS_AT_PIN and rec["truncated"]
    assert rec["items_at_pin_count"] == use.MAX_ITEMS_AT_PIN + 5
    huge = use.request_use(_lines([("x = 1\n" * 100_000, None)]), ITEMS)
    assert huge["unparsed_cells"] == 1
    rows = use.request_use(
        _lines([("from env.zz import *\nimport env.qq\n", None)]),
        ITEMS,
    )
    assert rows["module_imports"] == {"?": 1}  # no channel name outside the pin is kept


def test_the_analysis_module_imports_only_the_standard_library():
    tree = ast.parse(Path(use.__file__).read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            assert node.level == 0
            names = [node.module]
        elif isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        else:
            continue
        for name in names:
            top = name.split(".")[0]
            assert top == "__future__" or top in sys.stdlib_module_names, name


# --- the episode and post-hoc parity ---------------------------------------------------------------------


def test_the_record_is_written_with_the_episode_and_recomputed_from_its_directory(
    lib,
    tmp_path,
):
    cells = _session(
        ["from env.x import parse as p\np(1)\n", "import env.x\nenv.x.lookup('a')\n"],
    )
    acts = [Action(1, "x", "get", [], {}, None, "ok", "read")]
    text, shown = render_memory(lib)
    ep = _ep(transcript=_lines(cells, system=f"core\n\n{text}"), actions=acts, cells=[])
    ep.memory_use = memory_use(
        ep,
        ITEMS,
        redactor=Redactor(),
        export_roots=use.roots_of(lib),
        surface=use.library_surface(lib),
        shown=shown,
        shown_text=text,
        cell_status=_status(cells, roots=use.roots_of(lib)),
    )
    assert ep.memory_use["export_roots"][0] == str(lib)
    assert ep.memory_use["exposure_source"] == "record"
    assert (
        ep.memory_use["outcomes_known"] and ep.memory_use["cells_without_metadata"] == 0
    )
    assert ep.memory_use["shown_items"] == ITEMS
    assert ep.memory_use["items"]["env/x:parse"]["refused_then_accepted"] == 1
    repo, blobs = Repo.init_bare(tmp_path / "episodes.git"), BlobStore(tmp_path / "b")
    sha = EpisodeWriter(repo, blobs, Redactor()).write(ep)
    rel = episode_dir(ep)
    assert load_episode(repo, sha, rel, blobs).memory_use == ep.memory_use
    out = tmp_path / "export"
    out.mkdir()
    for name in ("transcript.jsonl", "actions.jsonl", "memory_use.json", "memory.diff"):
        (out / name).write_bytes(repo.show(sha, f"{rel}/{name}"))
    assert use.use_from_episode_dir(out) == ep.memory_use
    assert use.use_from_episode_dir(out, ITEMS) == ep.memory_use


def test_an_episode_without_a_record_is_written_as_before(tmp_path):
    """The OFF path never computes a record; an episode without one has the old file set and no rows."""
    repo, blobs = Repo.init_bare(tmp_path / "episodes.git"), BlobStore(tmp_path / "b")
    ep = _ep()
    assert ep.memory_use is None
    sha = EpisodeWriter(repo, blobs, Redactor()).write(ep)
    files = repo.run("ls-tree", "--name-only", sha, f"{episode_dir(ep)}/").split()
    assert sorted(f.rsplit("/", 1)[-1] for f in files) == sorted(
        [
            "meta.json",
            "request.json",
            "replies.json",
            "transcript.jsonl",
            "cells.jsonl",
            "actions.jsonl",
            "memory.diff",
            "worktree.diff",
            "cost.jsonl",
        ],
    )
    ev = EvidenceStore(tmp_path / "e.sqlite")
    ev.index_episode(ep, sha)
    assert ev.db.execute("SELECT COUNT(*) FROM item_use").fetchone() == (0,)
    assert load_episode(repo, sha, episode_dir(ep), blobs).memory_use is None


# --- the evidence index and the signals ------------------------------------------------------------------


def _record(
    items: dict,
    pin=ITEMS,
    unknown=None,
    shown=(),
    channels=(),
    modified=(),
    source="record",
    known=True,
) -> dict:
    base = dict.fromkeys(
        (
            "imported",
            "called",
            "referenced",
            "guarded",
            "refused",
            "errored",
            "refused_then_accepted",
        ),
        0,
    )
    return {
        "version": use.VERSION,
        "items_at_pin": list(pin),
        "items": {k: {**base, **v, "cells": [0]} for k, v in items.items()},
        "unknown_calls": unknown or {},
        "shown_items": list(shown),
        "shown_channels": list(channels),
        "modified_channels": list(modified),
        **({"exposure_source": source} if source is not None else {}),
        **({"outcomes_known": known} if known is not None else {}),
    }


def _index(ev: EvidenceStore, eid: str, rec: dict, at: str) -> None:
    ep = _ep(episode_id=eid, started_at=at, memory_use=rec)
    ev.index_episode(ep, "1" * 40)


def test_the_evidence_table_aggregates_per_item(tmp_path):
    ev = EvidenceStore(tmp_path / "e.sqlite")
    _index(
        ev,
        "e1",
        _record({"env/x:parse": {"imported": 1, "called": 2, "refused": 1}}),
        "2026-10-08T01:00:00Z",
    )
    _index(
        ev,
        "e2",
        _record(
            {"env/x:parse": {"called": 1, "refused_then_accepted": 1, "refused": 1}},
            unknown={"x": 2},
        ),
        "2026-10-08T02:00:00Z",
    )
    _index(ev, "e3", _record({}, pin=ITEMS[:2]), "2026-10-08T03:00:00Z")
    totals = ev.item_use()
    assert list(totals) == ITEMS
    p = totals["env/x:parse"]
    assert (p["requests"], p["used_requests"], p["called"], p["refused"]) == (
        3,
        2,
        3,
        2,
    )
    assert (p["refused_accepted"], p["unknown_calls"], p["imported"]) == (1, 2, 1)
    assert totals["env/x:strict"]["requests"] == 2  # not at e3's pin
    assert ev.item_use(["e2"])["env/x:parse"]["called"] == 1
    assert ev.last_call_seq("env/x:parse") == ev.seq_of("e2")
    assert ev.last_call_seq("env/x:lookup") is None
    assert (
        totals["env/x:parse"]["shown"] == 0
    )  # records without shown lists show nothing


def test_the_evidence_counts_where_each_requests_shown_lists_came_from(tmp_path):
    ev = EvidenceStore(tmp_path / "e.sqlite")
    sources = {
        "e1": "record",
        "e2": "record",
        "e3": "legacy_text",
        "e4": "unknown",
        "e5": None,
    }
    for n, (eid, source) in enumerate(sorted(sources.items())):
        _index(
            ev,
            eid,
            _record({}, shown=ITEMS[:1], channels=["x"], source=source),
            f"2026-10-08T0{n}:00:00Z",
        )
    row = ev.item_use()["env/x:lookup"]
    # a record from before the field counts as unknown
    assert (
        row["exposure_record"],
        row["exposure_legacy_text"],
        row["exposure_unknown"],
    ) == (
        2,
        1,
        2,
    )
    assert row["shown"] == 5  # the shown lists are counted whatever their source
    assert ev.request_flags() == {
        "exposure_record": 2,
        "exposure_legacy_text": 1,
        "exposure_unknown": 2,
        "outcome_unknown": 0,
    }
    assert ev.request_flags(["e1", "e3"])["exposure_legacy_text"] == 1
    assert ev.request_flags([]) == dict.fromkeys(ev.request_flags(), 0)


def test_a_store_from_before_the_table_or_its_columns_opens_and_is_migrated(tmp_path):
    import sqlite3

    old = tmp_path / "old.sqlite"
    con = sqlite3.connect(old)
    con.executescript(
        "CREATE TABLE episodes(seq INTEGER PRIMARY KEY AUTOINCREMENT, episode_id TEXT UNIQUE, "
        "commit_sha TEXT, started_at TEXT, regime TEXT, memory_main TEXT, request TEXT);"
        "INSERT INTO episodes(episode_id, commit_sha, started_at) VALUES('e0', 'x', '2026-10-07');",
    )
    con.commit()
    con.close()
    ev = EvidenceStore(old)
    _index(
        ev,
        "e1",
        _record({"env/x:parse": {"called": 1}}, shown=ITEMS),
        "2026-10-08T01:00:00Z",
    )
    assert ev.seq_of("e0") == 1 and ev.item_use()["env/x:parse"]["shown"] == 1
    # a store from this lane's first commit: item_use with its first ten columns and a row
    early = tmp_path / "early.sqlite"
    con = sqlite3.connect(early)
    con.executescript(
        "CREATE TABLE item_use(item TEXT, episode_id TEXT, imported INTEGER, called INTEGER, "
        "refused INTEGER, errored INTEGER, refused_accepted INTEGER, referenced INTEGER, "
        "guarded INTEGER, unknown_calls INTEGER, PRIMARY KEY(item, episode_id));"
        "INSERT INTO item_use VALUES('env/x:parse', 'e9', 1, 2, 0, 0, 0, 0, 0, 0);",
    )
    con.commit()
    con.close()
    ev = EvidenceStore(early)
    row = ev.item_use()["env/x:parse"]
    assert (row["called"], row["shown"], row["refused_modified"]) == (2, 0, 0)
    _index(
        ev,
        "e1",
        _record({"env/x:parse": {"refused_modified": 1}}, modified=["x"]),
        "2026-10-08T01:00:00Z",
    )
    row = ev.item_use()["env/x:parse"]
    assert (row["refused_modified"], row["modified"], row["refused"]) == (1, 1, 0)


# --- what the prompt showed ------------------------------------------------------------------------------


def test_shown_items_are_the_items_whose_own_lines_the_section_carries(lib):
    """Legacy reading: the items whose own lines the v2 index carries."""
    (lib / "env/x/__init__.py").write_text(MODULE + '\n__all__ = ["parse", "strict"]\n')
    index = build_index(lib)
    rec = use.request_use(_lines([], system=f"core\n\n{index}"), ITEMS)
    assert rec["shown_items"] == ["env/x:parse", "env/x:strict"]  # lookup is unlisted
    assert rec["shown_channels"] == ["x"] and rec["exposure_source"] == "legacy_text"
    assert use.request_use(_lines([]), ITEMS)["shown_items"] == []


def test_a_catalogue_of_channels_shows_channels_and_no_items(lib):
    """Legacy reading of a catalogue that kept the v2 header."""
    catalogue = f"{use.HEADER_PREFIX} Channels:\n- env.x: 3 functions for parsing lines\n- env.zz: 1\n"
    rec = use.request_use(_lines([], system=f"core\n\n{catalogue}"), ITEMS)
    assert rec["shown_items"] == [] and rec["shown_channels"] == ["x"]
    assert rec["exposure_source"] == "legacy_text"


def test_the_real_renderer_records_what_the_index_shows(lib):
    """The record comes from the renderer itself (``prompt.render_memory``), not from the prompt's text."""
    (lib / "env/x/__init__.py").write_text(MODULE + '\n__all__ = ["parse", "strict"]\n')
    text, shown = render_memory(lib)
    assert text == render_index(lib) and text.startswith(build_index(lib))
    assert build_index(lib) == index_with_names(lib)[0]
    assert shown["renderer"] == "index" and shown["channels"] == ["x"]
    assert shown["items"] == ["env/x:parse", "env/x:strict"]  # lookup is unlisted
    assert shown["sha256"] == hashlib.sha256(text.encode()).hexdigest()
    assert text not in json.dumps(shown)  # names and a digest, never the text
    rec = use.request_use(_lines([], system=f"core\n\n{text}"), ITEMS, shown=shown)
    assert rec["exposure_source"] == "record" and rec["shown_record"] == shown
    assert rec["shown_items"] == ["env/x:parse", "env/x:strict"]
    assert rec["shown_channels"] == ["x"]
    section = rec["memory_section_shown"]
    assert section["shown"] and section["prompt_confirmed"] is True
    assert (section["sha256"], section["bytes"]) == (shown["sha256"], shown["bytes"])
    # an empty library renders nothing and records that nothing was shown
    empty = lib.parent / "empty"
    (empty / "env").mkdir(parents=True)
    text0, shown0 = render_memory(empty)
    assert text0 == "" and shown0["sha256"] is None and shown0["channels"] == []
    rec0 = use.request_use(_lines([]), ITEMS, shown=shown0)
    assert rec0["exposure_source"] == "record" and rec0["shown_items"] == []
    assert not rec0["memory_section_shown"]["shown"]


S1_GUIDE = (
    "Memory: a library of Python functions distilled from earlier work, at `/mem/checkout` (first on "
    "the import path; import with `from env.<channel> import <function>`). Its functions are candidates "
    "to check, not authority.\n\nChannels:\n- `env.x`: 3 functions. Parse and look up lines.\n"
)


def test_an_s1_style_section_without_the_old_header_still_counts_what_it_showed(lib):
    """A channel catalogue in new wording (S1): the old header is absent, so the legacy reading finds
    nothing, but the renderer's record still counts every channel and item it showed."""
    assert use.HEADER_PREFIX not in S1_GUIDE
    lines = _lines([], system=f"core prompt\n\n{S1_GUIDE}")
    blind = use.request_use(lines, ITEMS)
    assert blind["exposure_source"] == "unknown" and blind["shown_channels"] == []
    catalogue = use.record_shown(S1_GUIDE, channels=["x"], renderer="catalogue")
    rec = use.request_use(lines, ITEMS, shown=catalogue)
    assert rec["exposure_source"] == "record"
    assert rec["shown_channels"] == ["x"] and rec["shown_items"] == []
    assert rec["memory_section_shown"]["prompt_confirmed"] is True
    listing = use.record_shown(
        S1_GUIDE,
        channels=["x"],
        items=["env/x:parse", "env/x:lookup"],
        renderer="catalogue",
    )
    rec = use.request_use(lines, ITEMS, shown=listing)
    assert rec["shown_items"] == ["env/x:lookup", "env/x:parse"]
    # the evidence row of each shown item says so, and where it came from
    rows = {r[0]: r for r in _use_rows("e1", rec)}
    assert rows["env/x:parse"][2 + _USE_COUNTS.index("shown")] == 1
    assert rows["env/x:strict"][2 + _USE_COUNTS.index("shown")] == 0
    assert rows["env/x:strict"][2 + _USE_COUNTS.index("channel_shown")] == 1
    assert rows["env/x:parse"][2 + _USE_COUNTS.index("exposure_record")] == 1


def test_a_recorded_section_the_prompt_does_not_end_with_still_counts_and_says_so(lib):
    shown = use.record_shown(
        S1_GUIDE,
        channels=["x"],
        items=["env/x:parse"],
        renderer="catalogue",
    )
    later = _lines([], system=f"core\n\n{S1_GUIDE}\nsomething appended after")
    rec = use.request_use(later, ITEMS, shown=shown)
    assert rec["shown_items"] == ["env/x:parse"]
    assert rec["memory_section_shown"]["prompt_confirmed"] is False
    # no system prompt was sent at all: nothing was shown
    unsent = [ln for ln in _lines([]) if ln["type"] != "system_prompt"]
    rec = use.request_use(unsent, ITEMS, shown=shown)
    assert rec["exposure_source"] == "record" and rec["shown_items"] == []
    assert not rec["memory_section_shown"]["shown"]
    assert rec["memory_section_shown"]["prompt_confirmed"] is None
    # only items and channels of the pin count
    other = use.record_shown(
        S1_GUIDE,
        channels=["zz"],
        items=["env/zz:f"],
        renderer="catalogue",
    )
    rec = use.request_use(_lines([], system=S1_GUIDE), ITEMS, shown=other)
    assert rec["shown_items"] == [] and rec["shown_channels"] == []


def test_record_shown_keeps_names_and_a_digest_only_for_random_inputs():
    """Seeded random names and texts: the record is sorted, bounded and deterministic, drops malformed
    names, counts each item's channel, shows nothing for an empty text and never holds the text.
    """
    rng = random.Random(20261008)
    alphabet = "abcxyz_019"
    for _ in range(200):
        chans = [
            "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 6)))
            for _ in range(rng.randint(0, 8))
        ]
        ids = [
            f"env/{rng.choice(chans or ['q'])}:{rng.choice(['f', 'g', '9bad', 'h-i', ''])}"
            for _ in range(rng.randint(0, 8))
        ]
        secret = f"VALUE-{rng.getrandbits(64):x}"
        text = rng.choice(["", f"Memory section holding {secret}\n"])
        rec = use.record_shown(text, channels=chans, items=ids, renderer="index")
        assert rec == use.record_shown(
            text,
            channels=list(reversed(chans)),
            items=ids,
            renderer="index",
        )
        assert secret not in json.dumps(rec)
        valid_ids = sorted(
            {i for i in ids if re.fullmatch(r"env/[A-Za-z_]\w*:[A-Za-z_]\w*", i)},
        )
        if not text:
            assert (
                rec["channels"] == [] and rec["items"] == [] and rec["sha256"] is None
            )
            continue
        assert rec["items"] == valid_ids
        assert rec["channels"] == sorted(
            {c for c in chans if re.fullmatch(r"[A-Za-z_]\w*", c)}
            | {i.split(":")[0][4:] for i in valid_ids},
        )
        assert rec["sha256"] == hashlib.sha256(text.encode()).hexdigest()
        assert rec["bytes"] == len(text.encode())
    # a malformed record read back (a hand-edited memory_use.json) is ignored, never trusted
    for bad in ({}, {"version": 99}, {"version": 1, "sha256": "nothex", "bytes": 1}):
        assert (
            use.request_use(_lines([]), ITEMS, shown=bad)["exposure_source"]
            == "unknown"
        )


# --- the agent's own edits -------------------------------------------------------------------------------

DIFF = "diff --git a/env/x/__init__.py b/env/x/__init__.py\nindex 1..2 100644\n--- a/env/x/__init__.py\n+++ b/env/x/__init__.py\n@@ -1 +1 @@\n-a\n+b\n"


def test_errors_from_an_edited_channel_are_not_charged_to_the_stored_item(lib):
    codes = [
        "from env.x import parse, lookup\nparse(3)\n",
        "lookup('zz')\n",
        "print(1)\n",
    ]
    cells = _session(codes)
    lines, status = _lines(cells), _status(cells)
    ok = Action(2, "x", "get", [], {}, None, "ok", "read")
    rec = use.request_use(lines, ITEMS, [ok], memory_diff=DIFF, cell_status=status)
    assert rec["modified_channels"] == ["x"] and not rec["memory_diff_truncated"]
    parse, lookup = rec["items"]["env/x:parse"], rec["items"]["env/x:lookup"]
    assert parse["modified_in_request"] and lookup["modified_in_request"]
    assert (
        parse["refused"],
        parse["refused_modified"],
        parse["refused_then_accepted"],
    ) == (0, 1, 0)
    assert (lookup["errored"], lookup["errored_modified"]) == (0, 1)
    assert parse["called"] == 1  # the call is still recorded
    plain = use.request_use(lines, ITEMS, [ok], cell_status=status)
    assert plain["items"]["env/x:parse"]["refused"] == 1
    assert not plain["items"]["env/x:parse"]["modified_in_request"]
    other = "diff --git a/env/other/__init__.py b/env/other/__init__.py\n"
    assert use.request_use(lines, ITEMS, memory_diff=other)["modified_channels"] == []


def test_a_truncated_diff_or_a_root_file_marks_every_channel(lib):
    cut = (
        DIFF.replace("env/x/", "env/zz/")
        + "\n[memory.diff truncated at 10 of 99 bytes; blob abc123]\n"
    )
    assert use.modified_channels(cut, ITEMS) == (["x"], True)
    root = "diff --git a/env/__init__.py b/env/__init__.py\n"
    assert use.modified_channels(root, ITEMS) == (["x"], False)


# --- export roots, star imports, re-exports --------------------------------------------------------------


def test_frames_are_tied_to_the_export_root(lib, tmp_path):
    ((_, err),) = _session(["from env.x import parse\nparse(3)\n"])
    assert use.attribute_errors(err, ITEMS, roots=use.roots_of(lib)) == [
        ("refused", "env/x:parse"),
    ]
    elsewhere = tmp_path / "elsewhere"
    assert use.attribute_errors(err, ITEMS, roots=[str(elsewhere)]) == []
    forged = err.replace(str(lib), str(elsewhere))
    assert use.attribute_errors(forged, ITEMS, roots=use.roots_of(lib)) == []


def _channel_y(lib) -> list[str]:
    (lib / "env" / "y").mkdir()
    (lib / "env" / "y" / "__init__.py").write_text(
        "from env.x import parse as yparse\n"
        '__all__ = ["shout", "yparse"]\n\n\n'
        "def shout(t):\n    return t.upper()\n\n\n"
        "def hidden(t):\n    return t\n",
    )
    return ITEMS + ["env/y:hidden", "env/y:shout"]


def test_star_imports_bind_the_modules_all_and_re_exports_resolve_to_the_definer(lib):
    items = _channel_y(lib)
    surface = use.library_surface(lib)
    assert surface == {
        "star": {
            "x": ["MemoryInputError", "lookup", "parse", "strict"],
            "y": ["shout", "yparse"],
        },
        "reexports": {"env/y:yparse": "env/x:parse"},
    }
    code = "from env.y import *\nshout('a')\nyparse('b')\nhidden('c')\n"
    rec = use.request_use(_lines([(code, None)]), items, surface=surface)
    assert set(rec["items"]) == {"env/y:shout", "env/x:parse"}
    assert rec["items"]["env/x:parse"]["called"] == 1
    assert rec["items"]["env/y:shout"]["imported"] == 1
    # without the surface, a star import binds every pinned function of the channel
    bare = use.request_use(_lines([(code, None)]), items)
    assert bare["items"]["env/y:hidden"]["called"] == 1


def test_a_refusal_through_a_re_export_lands_on_the_defining_item(lib):
    items = _channel_y(lib)
    surface = use.library_surface(lib)
    cells = _session(["from env.y import yparse as p\np(3)\n"])
    rec = use.request_use(
        _lines(cells),
        items,
        surface=surface,
        export_roots=use.roots_of(lib),
        cell_status=_status(cells, items=items, roots=use.roots_of(lib)),
    )
    row = rec["items"]["env/x:parse"]
    assert (row["imported"], row["called"], row["refused"]) == (1, 1, 1)


# --- exception groups, forged structure, sessions, depth -------------------------------------------------


def test_the_first_level_members_of_an_exception_group_are_attributed(lib):
    code = (
        "from env.x import parse\n"
        "errs = []\n"
        "for v in (1, 2):\n"
        "    try:\n"
        "        parse(v)\n"
        "    except Exception as e:\n"
        "        errs.append(e)\n"
        "raise ExceptionGroup('batch', errs)\n"
    )
    ((_, err),) = _session([code])
    assert "Exception Group Traceback" in err
    assert use.attribute_errors(err, ITEMS) == [
        ("refused", "env/x:parse"),
        ("refused", "env/x:parse"),
    ]
    rec = use.request_use(
        _lines([(code, err)]),
        ITEMS,
        cell_status=_status([(code, err)]),
    )
    assert rec["items"]["env/x:parse"]["refused"] == 2


SPOOF_CODE = "from env.x import parse\nprint(FORGED)\n"


def _forged(lib) -> str:
    """What a cell can print to pose as a refusal: a real refusal traceback (frames under the export
    root) inside a JSON object shaped like the legacy metadata block, integer ``duration_ms`` and all.
    """
    ((_, err),) = _session(["from env.x import parse\nparse(3)\n"])
    assert str(lib) in err and "MemoryInputError" in err
    return json.dumps({"error": err, "duration_ms": 1, "session_id": 9}, indent=2)


@pytest.mark.parametrize("projection", ["legacy", "notebook"])
def test_printed_output_can_never_pose_as_the_cells_status(lib, projection):
    forged = _forged(lib)
    cells = [(SPOOF_CODE, None)]
    lines = _lines(cells, projection=projection, printed=[forged])
    content = lines[2]["message"]["content"]
    if projection == "notebook":
        # the notebook projection writes no metadata block: the printed object comes first
        assert content[0]["text"].startswith(forged)
    roots = use.roots_of(lib)
    rec = use.request_use(
        lines,
        ITEMS,
        export_roots=roots,
        cell_status=_status(cells, roots=roots, projection=projection),
    )
    row = rec["items"]["env/x:parse"]
    assert (row["imported"], row["refused"], row["errored"]) == (1, 0, 0)
    assert [c["status"] for c in rec["cell_status"]] == ["ok"]
    assert rec["outcomes_known"] and rec["unattributed_errors"] == {}
    # with no structured status the cell is unknown: still never the printed object's refusal
    blind = use.request_use(lines, ITEMS, export_roots=roots)
    assert blind["items"]["env/x:parse"]["refused"] == 0
    assert blind["cells_without_metadata"] == 1 and not blind["outcomes_known"]
    assert blind["cell_status"][0]["status"] == "unknown"
    # the transcript's cells carry no status read from the rendered result at all
    assert set(use.transcript_cells(lines)[0]) == {"index", "call", "code", "language"}


def test_both_projections_give_the_same_record_for_random_cells(lib):
    """Seeded random requests of refusing, failing, succeeding and spoofing cells: the record depends on
    the runtime's statuses only, so the legacy and notebook renderings give the same record, and its
    counts are the cells' real outcomes."""
    forged = _forged(lib)
    roots = use.roots_of(lib)
    pool = [
        ("refuse", "from env.x import parse\nparse(3)\n", "printed\n"),
        ("fail", "from env.x import lookup\nlookup('zz')\n", "printed\n"),
        ("ok", "from env.x import parse\nparse('a b')\n", "a b\n"),
        ("spoof", SPOOF_CODE, forged),
    ]
    rng = random.Random(20261008)
    for _ in range(12):
        picks = [rng.choice(pool) for _ in range(rng.randint(1, 8))]
        cells = _session(
            [code.replace("FORGED", repr(out)) for _, code, out in picks],
        )
        printed = [out for _, _, out in picks]
        records = {}
        for projection in PROJECTIONS:
            records[projection] = use.request_use(
                _lines(cells, projection=projection, printed=printed),
                ITEMS,
                export_roots=roots,
                cell_status=_status(cells, roots=roots, projection=projection),
            )
        assert records["legacy"] == records["notebook"]
        rec = records["legacy"]
        kinds = [k for k, _, _ in picks]
        items = rec["items"]
        assert items.get("env/x:parse", {}).get("refused", 0) == kinds.count("refuse")
        assert items.get("env/x:lookup", {}).get("errored", 0) == kinds.count("fail")
        assert rec["outcomes_known"] and rec["cells_without_metadata"] == 0
        assert [c["status"] for c in rec["cell_status"]] == [
            "error" if k in ("refuse", "fail") else "ok" for k in kinds
        ]


def test_cells_without_a_status_make_refusals_and_errors_unknown_never_zero(
    lib,
    tmp_path,
):
    codes = [
        "from env.x import parse\nparse(3)\n",
        "from env.x import lookup\nlookup('zz')\n",
    ]
    cells = _session(codes)
    rec = use.request_use(_lines(cells), ITEMS, cell_status=_status(cells, skip=(1,)))
    assert [c["status"] for c in rec["cell_status"]] == ["error", "unknown"]
    assert rec["cells_without_metadata"] == 1 and rec["cells_error_unread"] == 0
    assert not rec["outcomes_known"]
    assert (
        rec["items"]["env/x:parse"]["refused"] == 1
    )  # what was recorded, a lower bound
    # a recording with no statuses at all (before the field): every cell unknown
    old = use.request_use(_lines(cells), ITEMS)
    assert old["cells_without_metadata"] == 2 and not old["outcomes_known"]
    # forged or malformed recorded entries are never trusted
    junk = [
        {
            "call": "c0",
            "status": "error",
            "exits": [["refused", "zz", "f"], ["boom", "x", "parse"]],
        },
        {"call": "c1", "status": "weird"},
    ]
    rec_junk = use.request_use(_lines(cells), ITEMS, cell_status=junk)
    assert [c["status"] for c in rec_junk["cell_status"]] == ["error", "unknown"]
    assert rec_junk["cell_status"][0]["exits"] == []
    assert all(v["refused"] == v["errored"] == 0 for v in rec_junk["items"].values())
    # an error too long to read is counted apart, and also leaves the outcomes unknown
    huge = use.runtime_status("x" * (use.MAX_TRACEBACK_CHARS + 1), items=ITEMS)
    assert huge["status"] == "unread" and huge["exits"] == []
    unread = use.request_use(_lines(cells[:1]), ITEMS, cell_status={"c0": huge})
    assert unread["cells_error_unread"] == 1 and not unread["outcomes_known"]
    # the evidence store, the signals and Sol's table say unknown, never 0
    shown = use.record_shown("SECTION", channels=["x"], items=ITEMS, renderer="index")
    lines = _lines(cells, system="core\n\nSECTION")
    partial = use.request_use(
        lines,
        ITEMS,
        cell_status=_status(cells, skip=(1,)),
        shown=shown,
    )
    complete = use.request_use(lines, ITEMS, cell_status=_status(cells), shown=shown)
    assert partial["shown_items"] == ITEMS and complete["outcomes_known"]
    ev = EvidenceStore(tmp_path / "e.sqlite")
    _index(ev, "e1", partial, "2026-10-08T01:00:00Z")
    row = ev.item_use()["env/x:lookup"]
    assert row["outcome_unknown"] == 1 and row["errored"] == 0
    assert ev.request_flags()["outcome_unknown"] == 1
    from unify.memory_v2 import usage

    sig = usage.item_signals("env/x:lookup", ev)
    assert sig["errors"] is None and sig["errors_at_least"] == 0
    assert not sig["outcomes_known"] and sig["requests_outcome_unknown"] == 1
    assert usage.item_signals("env/x:parse", ev)["refusals_at_least"] == 1
    # shown and untouched, but a cell's outcome is unknown: not "never used"
    assert not usage.item_signals("env/x:strict", ev)["never_used"]
    known = EvidenceStore(tmp_path / "known.sqlite")
    _index(known, "e1", complete, "2026-10-08T01:00:00Z")
    assert usage.item_signals("env/x:strict", known)["never_used"]
    assert usage.item_signals("env/x:lookup", known)["errors"] == 1
    table = usage.usage_table(ev, ["e1"], ITEMS)
    assert " | 1+? | 0+? | 0+? | " in table  # parse: refused 1, at least
    assert "(+?: 1 of these requests did not record every cell's outcome" in table


def test_the_tool_result_hook_notes_only_code_results_of_the_current_run(monkeypatch):
    run = RequestRun("r", None)
    run.item_ids = list(ITEMS)
    monkeypatch.setattr(hooks, "_run", lambda: None)
    hooks.tool_result("execute_code", "c0", _result(None))
    assert run.cell_status == {}  # no run: inert
    monkeypatch.setattr(hooks, "_run", lambda: run)
    hooks.tool_result("execute_code", "c0", _result(None, {"session_id": 4}))
    hooks.tool_result("execute_code", "c1", _result(None, projection="notebook"))
    hooks.tool_result("other_tool", "c2", {"error": "Traceback"})  # not a code result
    hooks.tool_result("execute_code", "c3", "a plain string")
    assert sorted(run.cell_status) == ["c0", "c1"]
    assert run.cell_status["c0"] == {
        "status": "ok",
        "exits": [],
        "session": 4,
        "fresh": False,
    }

    def broken(call_id, result):
        raise RuntimeError("boom")

    monkeypatch.setattr(run, "note_result", broken)
    hooks.tool_result("execute_code", "c4", _result(None))  # never raises


def test_bindings_follow_the_execution_session(lib):
    cells = [
        ("from env.x import parse\n", None),
        ("parse('a')\n", None),  # same session: counted
        ("parse('b')\n", None),  # another session: not the item
        ("parse('c')\n", None),  # the first session restarted: not the item
    ]
    metas = [
        {"session_id": 1},
        {"session_id": 1},
        {"session_id": 2},
        {"session_id": 1, "session_created": True},
    ]
    rec = use.request_use(
        _lines(cells, metas=metas),
        ITEMS,
        cell_status=_status(cells, metas),
    )
    assert rec["items"]["env/x:parse"]["called"] == 1
    assert [c["session"] for c in rec["cell_status"]] == [1, 1, 2, 1]
    # a cell without a status stays in the previous cell's session
    gap = use.request_use(
        _lines(cells[:2], metas=metas[:2]),
        ITEMS,
        cell_status=_status(cells[:2], metas[:2], skip=(1,)),
    )
    assert gap["items"]["env/x:parse"]["called"] == 1
    assert gap["cell_status"][1] == {
        "call": "c1",
        "status": "unknown",
        "exits": [],
        "session": 1,
        "fresh": False,
    }


def test_a_deep_syntax_tree_is_counted_unparsed_and_the_rest_still_counts(lib):
    deep = "x = " + "+".join(["1"] * 20000)  # ast.parse raises RecursionError
    wide = "y = " + "+".join(
        ["1"] * 5000,
    )  # parses; walking it may exceed the recursion limit
    cells = [
        (deep, None),
        (wide, None),
        ("from env.x import parse\nparse('a')\n", None),
    ]
    rec = use.request_use(_lines(cells), ITEMS)
    assert rec["cells"] == 3 and rec["unparsed_cells"] >= 1
    assert rec["items"]["env/x:parse"]["called"] == 1


def test_a_failing_record_keeps_the_items_at_the_pin():
    ep = _ep(memory_use=None)
    ep.transcript = None  # not iterable: request_use raises
    rec = memory_use(ep, ITEMS)
    assert rec["error"] == "TypeError" and rec["items_at_pin"] == ITEMS
