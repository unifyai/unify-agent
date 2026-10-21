"""Every refusal form of every memory tool called in code, and every error path, with no secret, host path or route
token in any reply or row.

``COVERAGE`` names every (tool, form) the memory tools' specs list (and the general forms, under ``*``): each is
either reached by the scripted passes below, through the real channel and handlers, or listed as data with the
reason it cannot be reached by a call. A forced exception in every tool's handler is answered as an error that
carries none of what the exception said. The planted values: an environment variable, a fake helper route token on
the helper route's state and in the environment, the host's temporary folders, the pass's state folder and home.
Keyless and boxless (the in-process interpreter stand-in of ``test_tool_handlers``).
"""

from __future__ import annotations

import asyncio
import json
import os
import threading
import time
from pathlib import Path

import pytest

from unify.common import tool_channel as tc
from unify.memory import tool_handlers, tool_spec
from unify.memory.trigger import PassRequest
from unify.memory.writer_pass import WriterPass
from tests.memory.test_tool_handlers import (
    PLANTED_ENV,
    PLANTED_TOKEN,
    FakeBox,
    ToolsSeen,
    _around,
    _client,
    _thread_call,
)
from tests.memory.test_writer_library_tools import EPS
from tests.memory.test_writer_pass import _call, _writer

R = "reached"

COVERAGE: dict[tuple[str, str], str] = {
    ("read", "outside_roots"): R,
    ("read", "outside_box"): R,
    ("read", "not_found"): R,
    ("read", "unreadable"): (
        "only an I/O error other than a missing or forbidden entry gives it (the writer answers any other read "
        "failure as not_found, a helper a forbidden entry as outside_box); no call's arguments make one"
    ),
    ("read", "outside_library"): (
        "given only by the librarian's read, for a path outside the library; no pass agent gets it (reached in "
        "test_librarian_navigation.py)"
    ),
    ("grep", "outside_library"): (
        "given only by the librarian's grep, for a path outside the library; no pass agent gets it (reached in "
        "test_librarian_navigation.py)"
    ),
    ("grep", "outside_roots"): R,
    ("grep", "outside_box"): R,
    ("grep", "bad_pattern"): R,
    ("grep", "not_found"): (
        "grep runs in a child process that reports only an outside path or a bad pattern; a missing path is "
        "searched as nothing, so the handler never sees a missing file"
    ),
    ("grep", "unreadable"): "as read's unreadable: only an I/O error other than a missing or forbidden entry",
    ("grep", "search_failed"): (
        "the search's child process fails only for a reason outside the call (it could not start, or it crashed); "
        "listed for builds whose grep has this form"
    ),
    ("overview", "unknown_episode"): R,
    ("overview", "not_episode_id"): R,
    ("overview", "not_recorded"): R,
    ("read_episode", "unknown_episode"): R,
    ("read_episode", "bad_offset"): R,
    ("read_episode", "bad_parts"): "the schema requires parts to be an array (bad_arguments first)",
    ("read_episode", "unknown_part"): R,
    ("read_episode", "not_episode_id"): R,
    ("read_episode", "not_recorded"): R,
    ("fixture", "unknown_episode"): R,
    ("fixture", "no_source"): R,
    ("fixture", "bad_fixture"): R,
    ("fixture", "link"): R,
    ("fixture", "not_input_lines"): R,
    ("replay", "not_run"): R,
    ("land_capture", "unknown_capture"): R,
    ("land_capture", "bad_name"): R,
    ("land_capture", "failed_capture"): (
        "needs a capture store holding a failed attempt, which this pass (fork off) has none of; reached in "
        "test_land_capture.py"
    ),
    ("land_capture", "not_landed"): (
        "needs a capture store, which this pass (fork off) has none of; reached in test_land_capture.py"
    ),
    ("pool", "bad_op"): "the schema's enum refuses any other op as bad_arguments before the handler runs",
    ("pool", "bad_status"): "the schema's enum refuses any other status as bad_arguments before the handler runs",
    ("pool", "unknown_entry"): R,
    ("pool", "closed_entry"): R,
    ("pool", "open_entry"): R,
    ("pool", "bad_decision"): R,
    ("pool", "manifest_unreadable"): R,
    ("similar", "args"): R,
    ("similar", "unknown_episode"): R,
    ("similar", "unknown_cell"): R,
    ("similar", "no_parse"): R,
    ("similar", "not_available"): (
        "an agent is offered similar only when the pass has an index, and with an index the handler is available"
    ),
    ("check", "manifest_problem"): R,
    ("check", "too_late"): R,
    ("check", "stopped"): R,
    ("finish", "manifest_problem"): R,
    ("finish", "bad_answer"): (
        "given only by the librarian's finish, for an answer that does not match its form; no pass agent gets it"
    ),
    ("finish", "answered"): (
        "given only by the librarian's finish, for a second answer to one question; no pass agent gets it"
    ),
    ("write", "bad_path"): R,
    ("write", "reserved"): R,
    ("write", "link"): R,
    ("write", "not_regular"): R,
    ("write", "not_directory"): R,
    ("write", "not_written"): (
        "an OS failure creating the file or a directory under /memory (a full disk, a read-only mount); the box's "
        "user owns /memory, so no call's arguments make one"
    ),
    ("write", "bad_content"): "the schema refuses content that is not a string as bad_arguments first",
    ("edit", "bad_path"): R,
    ("edit", "reserved"): R,
    ("edit", "link"): R,
    ("edit", "not_regular"): R,
    ("edit", "not_directory"): R,
    ("edit", "not_written"): "as write's not_written: an OS failure replacing the file",
    ("edit", "not_found"): R,
    ("edit", "not_text"): R,
    ("edit", "bad_old"): R,
    ("edit", "match_count"): R,
    ("delegate", "no_task"): R,
    ("delegate", "no_episodes"): "the schema requires episodes as a list of strings (bad_arguments first)",
    ("delegate", "unknown_episode"): R,
    ("delegate", "no_lane"): "delegate is offered only when a helper lane exists, and then its pool exists too",
    ("delegate", "no_time"): R,
    ("delegate", "index_unreadable"): R,
    ("delegate", "not_allowed"): "the pass's pool lets its agent start helpers; the refusal is for a pool that does not",
    ("delegate", "helper_limit"): R,
    ("delegate", "bad_request"): "helper names are made by the host (call-<n>); the pool takes every such request",
    ("send", "no_text"): R,
    ("send", "unknown_name"): R,
    ("send", "no_recipient"): R,
    ("send", "not_posted"): R,
    ("send", "no_role"): "the run record's send only (unify.agents.run_record_tools): a pass's own record has no roles",
    ("send", "ended"): "the run record's send only: a pass's own record ends with the pass, so no agent on it has ended",
    ("reply", "no_text"): R,
    ("reply", "unknown_name"): R,
    ("reply", "no_recipient"): "reply always addresses the named agent: an agent that is not running is unknown_name",
    ("reply", "not_posted"): R,
    ("stop", "unknown_helper"): R,
    ("stop", "not_running"): R,
    ("ask", "no_text"): R,
    ("ask", "unknown_name"): R,
    ("ask", "not_posted"): R,
    ("stage", "path"): R,
    ("stage", "quota"): R,
    ("done", "claims"): R,
    ("done", "no_sources"): R,
    ("done", "bad_source"): R,
    ("*", "unknown_tool"): R,
    ("*", "bad_arguments"): R,
    ("*", "readers_per_turn"): R,
    ("*", "cells_per_turn"): R,
    ("*", "reads_limit"): R,
    ("*", "stopped"): R,
    ("*", "pass_finished"): R,
    ("*", "task_done"): R,
    ("*", "check_limit"): R,
    ("*", "call_limit"): R,
}

#: The model's own JSON tool keeps today's refusals; it is not a channel tool.
NOT_CHANNEL_TOOLS = {"execute_code"}

#: Forms the shared specs list for the actor's and a teammate's handlers (their agent record), which the memory
#: agents' handlers never give: the pass's record answers a refused post as not_posted.
ACTOR_ONLY = "given only by the actor's and a teammate's handlers (the agent record)"


def _actor_only() -> dict[tuple[str, str], str]:
    from unify.common import tool_registry as reg

    out = {("delegate", f): ACTOR_ONLY for f in reg._SPAWN_FORMS}
    for tool in ("send", "reply", "ask"):
        out.update({(tool, f): ACTOR_ONLY for f in reg._RECORD_POST_FORMS})
    out.update(
        {
            ("wait", "bad_timeout"): ACTOR_ONLY,
            ("stop", "not_allowed"): ACTOR_ONLY,
        },
    )
    return out


def _required() -> set[tuple[str, str]]:
    codes = tool_spec.refusal_codes()
    out = {("*", f) for f in codes[tool_spec.GENERAL]}
    for tool, forms in codes.items():
        if tool in tool_handlers.HANDLED:
            out.update((tool, f) for f in forms)
    return out


def test_the_coverage_table_names_every_form_of_every_memory_tool():
    # every form is in the table; a form some builds do not have may be listed, as unreachable data
    table = {**_actor_only(), **COVERAGE}
    assert _required() <= set(table), sorted(_required() - set(table))
    assert all(table[k] != R for k in set(table) - _required())
    assert set(tool_handlers.HANDLED) | NOT_CHANNEL_TOOLS >= {t for t, _ in _required()} - {"*"}
    for key, how in COVERAGE.items():
        assert how == R or len(how) > 40, key  # an unreachable form says why


# --- the scripted passes ------------------------------------------------------------------------------------------------

BIG = "x" * 1_000_000
POST_GAP = ("__sleep__", 1.1)  # the record takes one post a second from each agent
LONG_POST = "y" * 20_000  # over the record's limit for one post

HELPER_CELL = [
    ("delegate", {"task": "more", "episodes": []}),
    ("finish", {"summary": "s"}),
    ("check", {}),
    ("read", {"path": "/etc/passwd"}),
    ("read", {"path": "/work/missing.txt"}),
    ("grep", {"pattern": "(", "path": "/work"}),
    ("grep", {"pattern": "x", "path": "/elsewhere"}),
    ("overview", {"episode": "../x"}),
    ("overview", {"episode": "e999"}),
    ("read_episode", {"episode": "../x"}),
    ("read_episode", {"episode": "e999"}),
    ("stage", {"path": "../escape.py", "text": "x"}),
    *[("stage", {"path": f"candidates/big{i}.py", "text": BIG}) for i in range(5)],
    ("send", {"text": ""}),
    ("send", {"text": "@nobody hello"}),
    POST_GAP,
    ("send", {"text": "@writer " + LONG_POST}),
    ("ask", {"question": ""}),
    ("ask", {"question": "@nobody which one?"}),
    POST_GAP,
    ("ask", {"question": "@writer " + LONG_POST}),
    POST_GAP,
    ("ask", {"question": "@writer which one?"}),
    ("done", {"summary": "s", "claims": [{"claim": "", "sources": []}]}),
    ("done", {"summary": "s", "claims": [{"claim": "x", "sources": []}]}),
    ("done", {"summary": "s", "claims": [{"claim": "x", "sources": ["e999/request"]}]}),
    ("done", {"summary": "handing back", "claims": []}),
]

MANIFEST_LINK = ".pass/manifest.json"

WRITER_CELL = [
    ("read", {"path": "/etc/passwd"}),
    ("read", {"path": "/memory/missing.py"}),
    ("grep", {"pattern": "(", "path": "/inputs"}),
    ("grep", {"pattern": "x", "path": "/elsewhere"}),
    ("overview", {"episode": "e999"}),
    ("read_episode", {"episode": "e999"}),
    ("read_episode", {"episode": "e1", "parts": ["nope"]}),
    ("read_episode", {"episode": "e1", "parts": [{"part": "request", "offset": -1}]}),
    ("read_episode", {"episode": "e1", "nope": 1}),
    ("bogus_tool", {}),
    ("fixture", {"episode": "e999", "dest": "memory/pkg/tests/data/a.json"}),
    ("fixture", {"episode": "e1", "dest": "memory/pkg/tests/data/a.json"}),
    ("fixture", {"episode": "e1", "dest": "memory/pkg/tests/data/a.json", "source": "zzz"}),
    ("__setup__", {"link": ["memory/lnk", "nowhere"]}),
    ("fixture", {"episode": "e1", "dest": "memory/lnk/tests/data/a.json", "source": "request:0"}),
    ("write", {"path": "memory/pkg/tests/data/in.jsonl", "content": "not an input line\n"}),
    (
        "fixture",
        {"episode": "e1", "dest": "memory/pkg/tests/data/in.jsonl", "source": "request:0", "append": True},
    ),
    ("replay", {"item": "nothing"}),
    ("land_capture", {"capture_id": "nope", "package": "pkg", "name": "proc"}),
    ("land_capture", {"capture_id": "0123456789abcdef", "package": "Bad-Pkg", "name": "proc"}),
    ("__setup__", {"input": ["pool/openone/entry.json", json.dumps({"id": "openone", "status": "open"})]}),
    ("__setup__", {"input": ["pool/closedone/entry.json", json.dumps({"id": "closedone", "status": "retired"})]}),
    ("pool", {"op": "show", "id": "nope"}),
    ("pool", {"op": "decide", "id": "openone", "decision": "reopened", "reason": "again"}),
    ("pool", {"op": "decide", "id": "closedone", "decision": "retired", "reason": "wrong"}),
    ("pool", {"op": "decide", "id": "new", "decision": "defer"}),
    ("write", {"path": MANIFEST_LINK, "content": "not json"}),
    ("pool", {"op": "decide", "id": "new", "decision": "defer", "reason": "later", "episodes": ["e1"]}),
    ("similar", {}),
    ("similar", {"code": "def ("}),
    ("similar", {"episode": "e999", "cell": 0}),
    ("similar", {"episode": "e1", "cell": 99}),
    *[("check", {"manifest": "{not json"}) for _ in range(6)],
    ("write", {"path": "../escape.py", "content": "x"}),
    ("write", {"path": ".git/config", "content": "x"}),
    ("__setup__", {"link": ["lnkdir", "nowhere"]}),
    ("write", {"path": "lnkdir/x.py", "content": "x"}),
    ("write", {"path": "memory/ok.py", "content": "x = 1\n"}),
    ("write", {"path": "memory/ok.py/x.py", "content": "x"}),
    ("write", {"path": "memory", "content": "x"}),
    ("write", {"path": "n" * 300 + ".py", "content": "x"}),  # an OS error the handler does not name: an error
    ("edit", {"path": "memory/missing.py", "old": "a", "new": "b"}),
    ("__setup__", {"bytes": ["memory/bin.py", "fffe00"]}),
    ("edit", {"path": "memory/bin.py", "old": "a", "new": "b"}),
    ("edit", {"path": "memory/ok.py", "old": "", "new": "b"}),
    ("edit", {"path": "memory/ok.py", "old": "zzz", "new": "y"}),
    ("edit", {"path": "../escape.py", "old": "a", "new": "b"}),
    ("edit", {"path": ".git/config", "old": "a", "new": "b"}),
    ("edit", {"path": "lnkdir/x.py", "old": "a", "new": "b"}),
    ("edit", {"path": "memory", "old": "a", "new": "b"}),
    ("edit", {"path": "memory/ok.py/x.py", "old": "a", "new": "b"}),
    ("send", {"text": ""}),
    ("send", {"text": "@nobody hello"}),
    ("send", {"text": "hello"}),
    ("reply", {"agent": "nobody", "text": "x"}),
    ("reply", {"agent": "writer", "text": ""}),
    ("stop", {"helper": "nobody"}),
    ("delegate", {"task": "", "episodes": []}),
    ("delegate", {"task": "t", "episodes": ["e999"]}),
    ("delegate", {"task": "look at e1", "episodes": []}),
    ("wait", {}),  # the helper's question
    ("reply", {"agent": "call-0", "text": LONG_POST}),
    POST_GAP,
    ("reply", {"agent": "call-0", "text": "the first one"}),
    ("wait", {}),  # the helper's end (or its question, when the reply came first)
    ("__sleep__", 5.0),
    ("stop", {"helper": "call-0"}),
    ("write", {"path": "memory/big.bin", "content": "z" * (1024**2 + 1)}),  # over the per-file quota
    ("check", {}),
    ("edit", {"path": "memory/ok.py", "old": "x", "new": "y"}),
    ("finish", {"summary": "done"}),
]


def _cell(cid, calls):
    return _call(cid, "execute_code", {"code": json.dumps([list(c) for c in calls])})


def _turn(*cells):
    calls = []
    for cid, script in cells:
        calls.extend(_cell(cid, script)["tool_calls"])
    return {"role": "assistant", "content": "", "tool_calls": calls}


def _helper(script, live_s=0.0):
    def factory(label=None):
        state = {"step": 0}

        async def turn(messages, tools):
            state["step"] += 1
            if state["step"] == 1:
                head = [("__sleep__", live_s)] if live_s else []
                # a second cell in the same turn after done: task_done
                return _turn(("h1", head + list(script)), ("h2", [])), "0.001"
            return _cell("hx", [("done", {"summary": "again", "claims": []})]), "0.001"

        turn.route_token = PLANTED_TOKEN
        return turn

    factory.route_token = PLANTED_TOKEN
    return factory


def _run(tmp_path, name, turns, *, helper=None, patch=None, **cfg):
    folder = tmp_path / name
    folder.mkdir()
    FakeBox.started = []
    model = ToolsSeen(turns)
    cfg.setdefault("max_calls", 80)
    _, _, writer = _writer(
        folder,
        model,
        library_mode=True,
        tools_in_code=True,
        pass_record_root=str(folder / "out"),
        **cfg,
    )
    writer.load = EPS.__getitem__
    writer.interpreter_start = FakeBox.start
    if helper is not None:
        writer.helper_turn = helper
    if patch is not None:
        patch(writer)
    out = asyncio.run(writer.run(PassRequest("incremental", "svc", ["e1"], False), "p1"))
    rows = [json.loads(x) for x in (folder / "out" / "p1" / "tool-events.jsonl").read_text().splitlines()]
    texts = [str(m.get("content")) for m in writer.messages if m.get("role") == "tool"]
    texts += [str(m.get("content")) for msgs, _ in (getattr(writer._delegate_ctx, "transcripts", {}) or {}).values()
              for m in msgs if m.get("role") == "tool"]
    boxes = [p for b in FakeBox.started for p in b.host_paths()]
    return out, rows, texts, boxes, folder


def _forms(rows):
    out = set()
    for r in rows:
        form = (r.get("refusal") or {}).get("form")
        if r.get("event") == "tool_call" and form:
            out.add((r["tool"], form))
            out.add(("*", form))
    return out


def _no_leak(texts, rows, folder, boxes):
    events = (folder / "out" / "p1" / "tool-events.jsonl").read_text()
    call_rows = [json.dumps(r) for r in rows if r.get("event") in ("tool_call", "channel_closed")]
    host = {str(folder), os.path.expanduser("~"), *boxes}
    host.discard("")
    for blob in (*texts, events):
        for bad in (PLANTED_ENV, PLANTED_TOKEN):
            assert bad not in blob, (bad, _around(blob, bad))
    for blob in (*texts, *call_rows):
        for bad in host:
            assert bad not in blob, (bad, _around(blob, bad))


def test_every_reachable_form_is_answered_and_nothing_leaks(tmp_path, monkeypatch):
    monkeypatch.setenv("UNIFY_PLANTED_FOR_THE_CHANNEL_TEST", PLANTED_ENV)
    monkeypatch.setenv("UNIFY_MEMORY_LIBRARY_HELPER_TOKEN", PLANTED_TOKEN)
    (tmp_path / "similar").mkdir()
    seen: set = set()
    runs = []

    # the main pass: every tool's refusals, a helper's, the record's, and a quota stop
    runs.append(
        _run(
            tmp_path,
            "main",
            [_cell("c1", WRITER_CELL)],
            helper=_helper(HELPER_CELL),
            edit_tools=True,
            pool=True,
            similar_index=str(tmp_path / "similar"),
            deadline_s=300.0,
        ),
    )

    # too little time left for a check; an unreadable episode index; a helper started in the last second
    from unify.memory import helpers as _helpers

    real_context = _helpers.context_for

    def unreadable_index(*a, **k):
        ctx = real_context(*a, **k)

        def broken():
            raise OSError(f"cannot read {tmp_path} ({PLANTED_ENV})")

        ctx.recorded = broken
        return ctx

    def no_index(writer):
        monkeypatch.setattr(_helpers, "context_for", unreadable_index)

    runs.append(
        _run(
            tmp_path,
            "late",
            [_cell("c1", [("check", {}), ("delegate", {"task": "t", "episodes": []})])],
            helper=_helper([("done", {"summary": "s", "claims": []})]),
            patch=no_index,
            deadline_s=50.0,
        ),
    )
    monkeypatch.setattr(_helpers, "context_for", real_context)
    runs.append(
        _run(
            tmp_path,
            "last-second",
            [_cell("c1", [("__until_left__", 0.5), ("delegate", {"task": "t", "episodes": []})])],
            helper=_helper([("done", {"summary": "s", "claims": []})]),
            deadline_s=20.0,
        ),
    )

    # the per-turn and per-pass limits
    runs.append(
        _run(
            tmp_path,
            "limits",
            [
                _cell("c1", [("read", {"path": "/inputs"})] * 17),
                _cell("c2", [("read", {"path": "/inputs"})] * 5),
                _turn(*[(f"k{i}", []) for i in range(9)]),
                _cell("c4", [("check", {"manifest": "{x"})] * 3),
            ],
            max_calls=5,
            max_reads=20,
        ),
    )

    # an accepted finish, then a second cell in the same turn
    def accept(writer):
        writer._finish_manifest_problem = lambda box: (None, False)

    runs.append(
        _run(
            tmp_path,
            "finished",
            [_turn(("f1", [("finish", {"summary": "done"})]), ("f2", []))],
            patch=accept,
            max_calls=3,
        ),
    )

    # seventeen helpers alive at once: the seventeenth is refused
    runs.append(
        _run(
            tmp_path,
            "many",
            [_cell("c1", [("delegate", {"task": f"t{i}", "episodes": []}) for i in range(17)])],
            helper=_helper([("done", {"summary": "s", "claims": []})], live_s=3.0),
            deadline_s=120.0,
            max_calls=2,
        ),
    )

    for out, rows, texts, boxes, folder in runs:
        seen |= _forms(rows)
        _no_leak(texts, rows, folder, boxes)
    missing = sorted(k for k, how in COVERAGE.items() if how == R and k not in seen)
    assert not missing, missing
    # the long name is an OS error the handler does not catch: an error reply with only its type
    main_rows = runs[0][1]
    long_name = [r for r in main_rows if r.get("tool") == "write" and r["status"] == "error"]
    assert long_name, "no error reply for the over-long name"


# --- a forced exception in every handler ---------------------------------------------------------------------------------


def test_a_forced_exception_in_any_handler_is_an_error_reply_that_carries_nothing_it_said(tmp_path):
    secret = f"{PLANTED_ENV} {PLANTED_TOKEN} {tmp_path} {os.path.expanduser('~')}"
    from types import SimpleNamespace

    from unify.common import tool_registry

    specs = {n: s for n, s in tool_registry.registry().items() if n in tool_handlers.HANDLED}

    def boom_sync(ctx, args):
        raise OSError(secret)

    async def boom_async(ctx, args):
        raise RuntimeError(secret)

    forced = {}
    for name, spec in specs.items():
        real = getattr(tool_handlers, name)
        fn = boom_async if asyncio.iscoroutinefunction(real) else boom_sync
        agent = next(a for a in ("writer", "helper", "curator") if a in spec.roles)
        forced[name] = SimpleNamespace(
            name=name,
            parameters=tool_registry.strict_parameters(agent, name),
            kind=spec.kind,
            blocking=spec.blocking,
            deadline_s=spec.deadline_s,
            handler=fn,
        )
    sample = {
        name: {k: _sample(v) for k, v in ((spec.parameters or {}).get("properties") or {}).items()
               if k in ((spec.parameters or {}).get("required") or [])}
        for name, spec in forced.items()
    }

    async def main():
        binding = tc.CallContext(
            agent="writer", name="writer", role="write", pass_id="p1", request_id=None, roots={},
            limits=None, counters=None, record=None, episodes=None, deadline=time.monotonic() + 60, state=None,
        )
        rows: list[dict] = []
        ch = tc.Channel.open(binding, tuple(forced), rows.append, specs=forced)
        box = _client()
        req, rep = ch.box_fds()
        box._STATE["fds"] = (os.dup(req), os.dup(rep))
        ch.after_spawn()
        ch.cell_started(0)
        serve = asyncio.ensure_future(ch.serve())
        out = {}
        for name in forced:
            out[name] = await _thread_call(box, name, sample[name])
        await ch.close("ended")
        await asyncio.gather(serve, return_exceptions=True)
        return out, rows

    out, rows = asyncio.run(main())
    assert set(out) == set(tool_handlers.HANDLED)
    for name, exc in out.items():
        assert type(exc).__name__ == "ToolError", (name, exc)
        text = str(exc)
        assert text.startswith("error: ") and name in text
        for bad in secret.split(" "):
            assert bad not in text, (name, text)
    assert all(r["status"] == "error" for r in rows)
    assert not any(bad in json.dumps(rows) for bad in secret.split(" "))


def _sample(schema):
    if schema.get("enum"):
        return schema["enum"][0]
    if schema.get("anyOf"):
        return _sample(schema["anyOf"][0])
    t = schema.get("type")
    t = t[0] if isinstance(t, list) else t
    return {"string": "x", "integer": 0, "number": 0, "boolean": False, "array": [], "object": {}}.get(t, "x")


_ = (threading, pytest, Path, WriterPass)
