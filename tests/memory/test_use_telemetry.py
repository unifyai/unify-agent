"""Use telemetry (the library mode's stage 1): per-request use records, the evidence index, the writer's table.

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

from tests.memory.test_episodes import _ep
from unify.actor.execution.types import ExecutionResult, TextPart
from unify.memory.analysis import use
from unify.memory.blobs import BlobStore
from unify.memory.episodes import (
    Action,
    EpisodeWriter,
    env_channel,
    episode_dir,
    load_episode,
)
from unify.memory.evidence import _USE_COUNTS, EvidenceStore, _use_rows
from unify.memory.gitio import Repo
from unify.memory.index import HEADER, build_index
from unify.memory.integration import hooks
from unify.memory.integration.request import RequestRun, memory_use
from unify.memory.redact import Redactor

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


PROJECTIONS = {"legacy": ExecutionResult}


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


def test_bash_and_unparsable_cells_are_skipped_and_counted(lib):
    cells = [("ls -la", None), ("def (:\n", None)]
    lines = _lines(cells)
    lines[1]["message"]["tool_calls"][0]["function"]["arguments"] = json.dumps(
        {"code": "ls -la", "language": "bash"},
    )
    rec = use.request_use(lines, ITEMS)
    assert rec["cells"] == 2 and rec["unparsed_cells"] == 1 and rec["items"] == {}


# --- errors ----------------------------------------------------------------------------------------------


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
        "prompt_unconfirmed": 0,
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
    # the row indexed before the columns existed reads as unknown, never as known (DEFAULT 1)
    assert (row["outcome_unknown"], row["exposure_unknown"]) == (1, 1)
    assert (row["exposure_record"], row["prompt_unconfirmed"]) == (0, 0)
    assert ev.db.execute(
        "SELECT outcome_unknown, exposure_unknown FROM item_use WHERE episode_id='e9'",
    ).fetchone() == (1, 1)
    from unify.memory import usage

    sig = usage.item_signals("env/x:parse", ev)
    assert sig["refusals"] is None and not sig["outcomes_known"]
    assert sig["refusals_report"] == "refusals 0 of 0 known (+1 unknown)"
    _index(
        ev,
        "e1",
        _record({"env/x:parse": {"refused_modified": 1}}, modified=["x"]),
        "2026-10-08T01:00:00Z",
    )
    row = ev.item_use()["env/x:parse"]
    assert (row["refused_modified"], row["modified"], row["refused"]) == (1, 1, 0)
    # rows indexed after the migration set every column: the new request is known
    assert row["outcome_unknown"] == 1 and row["requests"] == 2


# --- what the prompt showed ------------------------------------------------------------------------------


S1_GUIDE = (
    "Memory: a library of Python functions distilled from earlier work, at `/mem/checkout` (first on "
    "the import path; import with `from env.<channel> import <function>`). Its functions are candidates "
    "to check, not authority.\n\nChannels:\n- `env.x`: 3 functions. Parse and look up lines.\n"
)


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


def test_a_truncated_diff_or_a_root_file_marks_every_channel(lib):
    cut = (
        DIFF.replace("env/x/", "env/zz/")
        + "\n[memory.diff truncated at 10 of 99 bytes; blob abc123]\n"
    )
    assert use.modified_channels(cut, ITEMS) == (["x"], True)
    root = "diff --git a/env/__init__.py b/env/__init__.py\n"
    assert use.modified_channels(root, ITEMS) == (["x"], False)


# --- export roots, star imports, re-exports --------------------------------------------------------------


def _channel_y(lib) -> list[str]:
    (lib / "env" / "y").mkdir()
    (lib / "env" / "y" / "__init__.py").write_text(
        "from env.x import parse as yparse\n"
        '__all__ = ["shout", "yparse"]\n\n\n'
        "def shout(t):\n    return t.upper()\n\n\n"
        "def hidden(t):\n    return t\n",
    )
    return ITEMS + ["env/y:hidden", "env/y:shout"]


# --- exception groups, forged structure, sessions, depth -------------------------------------------------


SPOOF_CODE = "from env.x import parse\nprint(FORGED)\n"


def _forged(lib) -> str:
    """What a cell can print to pose as a refusal: a real refusal traceback (frames under the export
    root) inside a JSON object shaped like the legacy metadata block, integer ``duration_ms`` and all.
    """
    ((_, err),) = _session(["from env.x import parse\nparse(3)\n"])
    assert str(lib) in err and "MemoryInputError" in err
    return json.dumps({"error": err, "duration_ms": 1, "session_id": 9}, indent=2)


def test_the_tool_result_hook_notes_only_code_results_of_the_current_run(monkeypatch):
    run = RequestRun("r", None)
    run.item_ids = list(ITEMS)
    monkeypatch.setattr(hooks, "_run", lambda: None)
    hooks.tool_result("execute_code", "c0", _result(None))
    assert run.cell_status == {}  # no run: inert
    monkeypatch.setattr(hooks, "_run", lambda: run)
    hooks.tool_result("execute_code", "c0", _result(None, {"session_id": 4}))
    hooks.tool_result("execute_code", "c1", _result(None))
    hooks.tool_result("other_tool", "c2", {"error": "Traceback"})  # not a code result
    hooks.tool_result("execute_code", "c3", "a plain string")
    assert sorted(run.cell_status) == ["c0", "c1"]
    # rev 2's ExecutionResult carries no session id (the actor's cells share one interpreter; 295ff16d5), so the
    # status has none
    assert run.cell_status["c0"] == {
        "status": "ok",
        "exits": [],
        "session": None,
        "fresh": False,
    }

    def broken(call_id, result):
        raise RuntimeError("boom")

    monkeypatch.setattr(run, "note_result", broken)
    hooks.tool_result("execute_code", "c4", _result(None))  # never raises


def test_a_failing_record_keeps_the_items_at_the_pin():
    ep = _ep(memory_use=None)
    ep.transcript = None  # not iterable: request_use raises
    rec = memory_use(ep, ITEMS)
    assert rec["error"] == "TypeError" and rec["items_at_pin"] == ITEMS


# --- fix round 3: names, plain results, causes, per-item unknowns, prompts ---------------------------------


def _frame_traceback(root, func: str, kind: str = "ValueError") -> str:
    """A traceback leaving ``env/x/__init__.py`` under *root* through a frame named *func*."""
    return (
        "Traceback (most recent call last):\n"
        '  File "<cell 0>", line 1, in <module>\n'
        f'  File "{root}/env/x/__init__.py", line 3, in {func}\n'
        "    raise\n"
        f"{kind}: boom\n"
    )


def _executor_traceback(exc: BaseException) -> str:
    try:
        raise exc
    except (
        BaseException
    ):  # noqa: BLE001 - the harness's own traceback, as execute_code keeps it
        return traceback.format_exc()


class SandboxGone(RuntimeError):
    pass


def _plain(error: str | None, session_id=None) -> dict:
    """The legacy ``execute_code``'s own plain result (an empty cell, or its session executor raised)."""
    return {
        "stdout": "",
        "stderr": "",
        "result": None,
        "error": error,
        "state_mode": "stateful",
        "session_id": session_id,
        "session_name": None,
        "session_created": False,
        "duration_ms": 0,
    }


def test_plain_results_and_raised_calls_reach_the_record_with_a_status(monkeypatch):
    run = RequestRun("r", None)
    run.item_ids = list(ITEMS)
    monkeypatch.setattr(hooks, "_run", lambda: run)
    assert hooks.result_hook() is hooks.tool_result
    hooks.tool_result("execute_code", "c0", _plain(None, session_id=3))
    hooks.tool_result(
        "execute_code",
        "c1",
        _plain(_executor_traceback(TimeoutError("worker gone"))),
    )
    hooks.tool_result("execute_code", "c2", _plain(_executor_traceback(SandboxGone())))
    hooks.tool_result("execute_code", "c3", None, raised=TypeError("bad keyword"))
    hooks.tool_result("execute_code", "c4", None, raised=SandboxGone())
    hooks.tool_result("other_tool", "c5", _plain(None))  # not the code tool
    hooks.tool_result("other_tool", "c6", None, raised=TypeError("x"))
    # a runtime result left as a dict (a list stdout) is read as one
    hooks.tool_result(
        "execute_code",
        "c7",
        {**_plain(None, session_id=2), "stdout": [], "duration_ms": 4},
    )
    assert sorted(run.cell_status) == ["c0", "c1", "c2", "c3", "c4", "c7"]
    assert run.cell_status["c0"] == {
        "status": "empty",
        "exits": [],
        "session": None,
        "fresh": False,
        "cause": "empty",
    }
    assert run.cell_status["c1"]["status"] == "error"
    assert run.cell_status["c1"]["cause"] == "executor_error"
    assert run.cell_status["c1"]["exc"] == "TimeoutError"
    assert run.cell_status["c2"]["exc"] == use.OTHER_NAME  # not a builtin exception
    assert run.cell_status["c3"] == {
        "status": "unknown",
        "exits": [],
        "session": None,
        "fresh": False,
        "cause": "tool_raised",
        "exc": "TypeError",
    }
    assert run.cell_status["c4"]["exc"] == use.OTHER_NAME
    assert run.cell_status["c7"] == {
        "status": "ok",
        "exits": [],
        "session": 2,
        "fresh": False,
    }
    assert "worker gone" not in json.dumps(run.cell_status)  # names only
    # no run: no hook at all
    monkeypatch.setattr(hooks, "_run", lambda: None)
    assert hooks.result_hook() is None


_HARNESS_REPLIES = {
    "cancelled": "Cancelled: the requester cancelled the request before this call finished.",
    "timeout": "Cancelled: the timeout (300s) was reached before this call finished.",
    "step_cap": "Cancelled: the step limit (max_steps (500) exceeded) ended the request before "
    "this call finished.",
    "not_run_timeout": "Not run: timeout (300s) exceeded before this call started.",
    "not_run_cap": "Not run: max_steps (500) exceeded before this call started.",
    "stopped": "Cancelled: the request was stopped for making no progress before this call "
    "finished.",
    "limit": "⚠️ Error: 'execute_code' was not run: its call limit is reached.",
}


def _random_request(rng: random.Random, channels: dict[str, list[str]]):
    """A seeded random request: cells built from fragments whose reach is known by construction, each
    with a known or unknown status (and, when known, random exits). Returns the transcript lines, the
    statuses by call id and the brute-force expectation: the items whose outcome is unknown and the
    refusals and errors per item."""
    ids = [f"env/{c}:{n}" for c, names in channels.items() for n in names]
    defined: dict[str, set[str]] = {}
    cells, statuses = [], {}
    unknown: set[str] = set()
    refused: dict[str, int] = {}
    errored: dict[str, int] = {}
    for i in range(rng.randint(1, 9)):
        parts, reach = [], set()
        for _ in range(rng.randint(1, 3)):
            kind = rng.choice(
                ["call", "attr", "dynamic", "module", "def", "callh", "plain"],
            )
            ch = rng.choice(sorted(channels))
            fn = rng.choice(channels[ch])
            item = f"env/{ch}:{fn}"
            if kind == "call":
                parts.append(f"from env.{ch} import {fn}\n{fn}(1)\n")
                reach.add(item)
            elif kind == "attr":
                parts.append(f"import env.{ch}\nenv.{ch}.{fn}(1)\n")
                reach.add(item)
            elif kind == "dynamic":
                parts.append(
                    f"import env.{ch} as m_{ch}\ngetattr(m_{ch}, 'f' + '0')(1)\n",
                )
                reach |= {f"env/{ch}:{n}" for n in channels[ch]}
            elif kind == "module":
                parts.append(f"import env.{ch}\nhelp(env.{ch})\n")
                reach |= {f"env/{ch}:{n}" for n in channels[ch]}
            elif kind == "def":
                name = f"h{len(defined)}"
                parts.append(
                    f"from env.{ch} import {fn}\ndef {name}():\n    return {fn}(1)\n",
                )
                defined[name] = {item}
                reach.add(item)
            elif kind == "callh" and defined:
                name = rng.choice(sorted(defined))
                parts.append(f"{name}()\n")
                reach |= defined[name]
            else:
                parts.append("v = 1 + 1\n")
        cells.append(("".join(parts), None))
        mode = rng.choice(["ok", "error", "none", "raised", "executor", "unread"])
        call = f"c{i}"
        if mode == "ok":
            statuses[call] = use.runtime_status(None, items=ids)
        elif mode == "error":
            exits = []
            for _ in range(rng.randint(1, 3)):
                ch = rng.choice(sorted(channels))
                fn = rng.choice(channels[ch])
                k = rng.choice(["refused", "errored"])
                exits.append([k, ch, fn])
                bucket = refused if k == "refused" else errored
                bucket[f"env/{ch}:{fn}"] = bucket.get(f"env/{ch}:{fn}", 0) + 1
            statuses[call] = {
                "status": "error",
                "exits": exits,
                "session": None,
                "fresh": False,
            }
        else:
            unknown |= reach
            if mode == "raised":
                statuses[call] = use.failed_status("RuntimeError")
            elif mode == "executor":
                statuses[call] = use.dict_status(
                    _plain("Traceback\nOSError: x\n"),
                    items=ids,
                )
            elif mode == "unread":
                statuses[call] = use.runtime_status(
                    "x" * (use.MAX_TRACEBACK_CHARS + 1),
                    items=ids,
                )
    return _lines(cells), statuses, unknown, refused, errored


def test_an_unknown_cell_marks_its_items_for_that_request_only(tmp_path):
    """One request with an unknown cell over parse leaves parse unknown in that request only; lookup
    stays known there, and parse is known again in the next request."""
    from unify.memory import usage

    ev = EvidenceStore(tmp_path / "e.sqlite")
    first = _record(
        {"env/x:parse": {"called": 1}, "env/x:lookup": {"called": 1, "refused": 1}},
        shown=ITEMS,
    )
    first.update(
        {
            "outcomes_known": False,
            "items_outcome_unknown": ["env/x:parse"],
            "items_outcome_unknown_count": 1,
        },
    )
    second = _record({"env/x:parse": {"called": 1, "refused": 1}}, shown=ITEMS)
    second.update({"items_outcome_unknown": [], "items_outcome_unknown_count": 0})
    _index(ev, "e1", first, "2026-10-08T01:00:00Z")
    _index(ev, "e2", second, "2026-10-08T02:00:00Z")
    parse = usage.item_signals("env/x:parse", ev)
    assert parse["refusals_report"] == "refusals 1 of 1 known (+1 unknown)"
    assert parse["refusals"] is None and parse["refusals_known"] == 1
    lookup = usage.item_signals("env/x:lookup", ev)
    assert lookup["refusals_report"] == "refusals 1 of 2 known (+0 unknown)"
    assert lookup["refusals"] == 1 and lookup["outcomes_known"]
    strict = usage.item_signals("env/x:strict", ev)
    assert strict["outcomes_known"] and strict["never_used"]
    # a list cut short (more unknown items than it holds) marks every item of the request
    cut = _record({}, shown=ITEMS)
    cut.update(
        {"items_outcome_unknown": ["env/x:parse"], "items_outcome_unknown_count": 9},
    )
    rows = {r[0]: r for r in _use_rows("e3", cut)}
    col = 2 + _USE_COUNTS.index("outcome_unknown")
    assert {i: rows[i][col] for i in ITEMS} == dict.fromkeys(ITEMS, 1)
    # a failed record (no list, no outcomes_known) marks every item
    rows = {
        r[0]: r
        for r in _use_rows(
            "e4",
            {"version": use.VERSION, "error": "TypeError", "items_at_pin": ITEMS},
        )
    }
    assert {i: rows[i][col] for i in ITEMS} == dict.fromkeys(ITEMS, 1)


def test_a_pre_v4_record_with_a_section_counts_as_legacy_text():
    col = 2 + _USE_COUNTS.index("exposure_legacy_text")
    unknown = 2 + _USE_COUNTS.index("exposure_unknown")
    for version, shown, want in (
        (3, True, (1, 0)),
        (2, False, (0, 1)),
        (4, True, (0, 1)),
    ):
        rec = _record({}, shown=ITEMS[:1], source=None)
        rec.update({"version": version, "memory_section_shown": {"shown": shown}})
        row = {r[0]: r for r in _use_rows("e1", rec)}["env/x:parse"]
        assert (row[col], row[unknown]) == want, version


# --- in-cell exposure under the catalogue (memory-int2) -----------------------------------------------


def _static(codes: list[str]) -> dict:
    """The record of cells that are only read (the counts are static: nothing needs to run)."""
    cells = [(code, None) for code in codes]
    return use.request_use(_lines(cells), ITEMS, cell_status=_status(cells))
