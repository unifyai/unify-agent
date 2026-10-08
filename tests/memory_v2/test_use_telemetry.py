"""Use telemetry (memory v2.1 stage 1): per-request use records, the evidence index, Sol's table.

The tracebacks here are real: each cell runs against a real ``env/<channel>/__init__.py`` package on a
temporary import path, and its traceback is formatted by the traceback module, as the worker does.
"""

from __future__ import annotations

import ast
import hashlib
import json
import sys
import traceback
from pathlib import Path

import pytest

from tests.memory_v2.test_episodes import _ep
from unify.memory_v2.analysis import use
from unify.memory_v2.blobs import BlobStore
from unify.memory_v2.episodes import (
    Action,
    EpisodeWriter,
    env_channel,
    episode_dir,
    load_episode,
)
from unify.memory_v2.evidence import EvidenceStore
from unify.memory_v2.gitio import Repo
from unify.memory_v2.index import HEADER, build_index
from unify.memory_v2.integration.request import memory_use
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


def _lines(
    cells: list[tuple[str, str | None]],
    system: str = "core prompt",
) -> list[dict]:
    """Transcript lines in the line format of ``unify.transcripts``: each cell's result starts with the
    JSON metadata block that holds its ``error``, as ``ExecutionResult.to_llm_content`` writes it.
    """
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
        meta = {"duration_ms": 3, **({"error": err} if err else {})}
        out.append(
            {
                "seq": len(out),
                "type": "message",
                "message": {
                    "role": "tool",
                    "tool_call_id": f"c{i}",
                    "name": "execute_code",
                    "content": [
                        {"type": "text", "text": json.dumps(meta, indent=2)},
                        {"type": "text", "text": "\n--- stdout ---\n"},
                        {"type": "text", "text": "printed\n"},
                    ],
                },
            },
        )
    return out


def _use(codes: list[str], actions=()) -> dict:
    return use.request_use(_lines(_session(codes)), ITEMS, actions)


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
        "refused_then_accepted": 0,
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
    rec = use.request_use(_lines([("from env.x import parse\nparse(3)\n", err)]), ITEMS)
    assert rec["items"]["env/x:parse"]["refused"] == 1


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


def test_the_memory_section_is_hashed_and_counted_from_the_system_prompt(lib):
    index = build_index(lib)
    assert HEADER.startswith(use.HEADER_PREFIX)
    rec = use.request_use(_lines([], system=f"core prompt\n\n{index}"), ITEMS)
    shown = rec["memory_section_shown"]
    assert shown["shown"] and shown["distinct"] == 1 and shown["prompts"] == 1
    assert shown["sha256"] == hashlib.sha256(index.encode()).hexdigest()
    assert shown["bytes"] == len(index.encode()) and shown["est_tokens"] == -(
        -len(index) // 4
    )
    none = use.request_use(_lines([]), ITEMS)["memory_section_shown"]
    assert not none["shown"] and none["sha256"] is None


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
    ep = _ep(transcript=_lines(cells), actions=acts, cells=[])
    ep.memory_use = memory_use(ep, ITEMS)
    assert ep.memory_use["items"]["env/x:parse"]["refused_then_accepted"] == 1
    repo, blobs = Repo.init_bare(tmp_path / "episodes.git"), BlobStore(tmp_path / "b")
    sha = EpisodeWriter(repo, blobs, Redactor()).write(ep)
    rel = episode_dir(ep)
    assert load_episode(repo, sha, rel, blobs).memory_use == ep.memory_use
    out = tmp_path / "export"
    out.mkdir()
    for name in ("transcript.jsonl", "actions.jsonl", "memory_use.json", "cells.jsonl"):
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


def _record(items: dict, pin=ITEMS, unknown=None) -> dict:
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
