"""The memory agents' sandboxes, in real boxes, with tools called from code: the writer, the curator and a helper
each find ``tools``, ``memory`` and ``help`` bound in their first cell with no import (``memory`` imported only by
the first cell that uses it, under that cell's open log), even when the library is empty; none finds a model or
network helper. Their cells keep their variables by default (``x = 41`` in one cell, ``x + 1`` in the next), and
under ``UNIFY_CELL_STATE=fresh`` each cell starts a new interpreter, for every role, and the prompt says so. The
restarts that lose variables are told in the cell's output. A library module rewritten after it was imported is
imported again with the new code, and ``memory`` is bound lazily again. Each role's first request opens with the
shared opening, which says the sandbox is persistent and stateful and that ``memory`` and ``tools`` are already
available as variables.

The model is scripted (no key, no network); the boxes need bubblewrap and prlimit, as the other box tests do.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import re
from pathlib import Path

import pytest

from tests.memory.test_tool_channel_box import needs_box
from tests.memory.test_writer_pass import Turns, _call, _e1, _ME
from unify.common.box_interpreter import BoxInterpreter
from unify.memory.blobs import BlobStore
from unify.memory.curate import CurateState
from unify.memory.evidence import EvidenceStore
from unify.memory.gate import Gate
from unify.memory.gitio import Repo
from unify.memory.trigger import PassRequest
from unify.memory.writer_pass import PassConfig, WriterPass
from unify.settings import SETTINGS

# ==== names and sentences this file assumes (the spec gives them; adapt them here, in one place) ====================
#: design D.1, persistent form, its first sentence
PERSISTENT_SENTENCE = (
    "Your Python sandbox is persistent and stateful: variables, imports and functions you define in one cell are "
    "still there in the next, so build on what you already have instead of defining or loading it again."
)
#: design D.1, fresh form
FRESH_SENTENCE = (
    "Each cell runs in a fresh Python sandbox: files you write stay, but variables, imports and functions do not "
    "carry over to the next cell."
)
#: design D.1, the names sentence
NAMES_SENTENCE = re.compile(
    r"((?:`[A-Za-z_][A-Za-z0-9_]*`(?:, | and |, and )?)+) (?:is|are) already available as (?:a )?variables? in every "
    r"cell; no import needed\.",
)
#: design B.2: the memory agents' pre-bound set, and what they must not have
PREBOUND = ("tools", "memory", "help")
MUST_NOT = (
    "query_llm",
    "list_llms",
    "unillm",
    "run_coro_sync",
    "display",
    "ModelRequestError",
    "install",
    "read_file",
    "grep",
    "record",
    "agents",
    "functions",
    "guidance",
)
#: the restart notes (box_interpreter._LOST and its callers; design C.3)
LOST = "the interpreter was restarted: variables are gone, files remain"
# =====================================================================================================================

CHECK = (*PREBOUND, *MUST_NOT)


def probe(names) -> str:
    """A first cell, before any import line: ``x = 41``, then which of *names* are bound (a bare name read in a
    ``try``, so it works with any builtins the box keeps; a lazy ``memory`` is not imported by it)."""
    lines = ["x = 41", "_bound = []"]
    for n in names:
        lines += ["try:", f"    {n}", f"    _bound.append({n!r})", "except NameError:", "    pass"]
    lines.append("print('BOUND', sorted(_bound))")
    return "\n".join(lines) + "\n"

#: the first cell of every role: before any import line, which names are bound, whether `memory` is still lazy
PROBE = probe(CHECK) + (
    "try:\n"
    "    _lazy = 'imported on first use' in repr(memory)\n"
    "except NameError:\n"
    "    _lazy = False\n"
    "print('LAZY', _lazy)\n"
    "import sys\n"
    "print('LOADED', 'memory' in sys.modules)\n"
)
#: the second cell: the variable is still there, and the library and help work with no import line
NEXT = (
    "print('NEXT', x + 1)\n"
    "print('INDEX', isinstance(memory.index(), str))\n"
    "help(len)\n"
    "print('HELPED')\n"
)
#: a second cell that only reads the variable (for the fresh opt-out)
ONLY_NEXT = "print('NEXT', x + 1)\n"


def flat(text: str) -> str:
    return " ".join(str(text).split())


def _text(content) -> str:
    if isinstance(content, list):
        return "\n".join(str(p.get("text", "")) if isinstance(p, dict) else str(p) for p in content)
    return "" if content is None else str(content)


def _system(messages: list[dict]) -> str:
    systems = [_text(m.get("content")) for m in messages if m.get("role") == "system"]
    assert systems, [m.get("role") for m in messages]
    return systems[0]


def _names(text: str) -> set[str]:
    found = NAMES_SENTENCE.findall(flat(text))
    assert len(found) == 1, f"expected one names sentence, found {found}"
    return set(re.findall(r"`([A-Za-z_][A-Za-z0-9_]*)`", found[0]))


def _bound(output: str) -> list[str]:
    line = next(ln for ln in output.splitlines() if ln.startswith("BOUND "))
    return json.loads(line[len("BOUND ") :].replace("'", '"'))


def _assert_first_request(messages: list[dict], role: str, persistent: bool = True) -> None:
    system = flat(_system(messages))
    assert system.startswith("## How you work here"), (role, system[:300])
    if persistent:
        assert system.count(PERSISTENT_SENTENCE) == 1, (role, system[:2000])
        assert FRESH_SENTENCE not in system, role
    else:
        assert system.count(FRESH_SENTENCE) == 1, (role, system[:2000])
        assert "persistent and stateful" not in system, role
    assert {"tools", "memory"} <= _names(system), role
    assert not re.search(r"\bimport memory\b", system), role


def _assert_probe(output: str, role: str) -> None:
    assert sorted(_bound(output)) == sorted(PREBOUND), (role, output)
    assert "LAZY True" in output, (role, output)
    assert "LOADED False" in output, (role, output)  # no library code ran before a cell used it


def _assert_next(output: str, role: str) -> None:
    assert "NEXT 42" in output, (role, output)
    assert "INDEX True" in output and "HELPED" in output, (role, output)
    assert "NameError" not in output, (role, output)


class Recording(Turns):
    """The scripted model, keeping every request's messages as sent; *delays* (request number -> seconds) holds a
    reply back, as a slow model would."""

    def __init__(self, messages, delays=None):
        super().__init__(messages)
        self.requests: list[list[dict]] = []
        self.delays = dict(delays or {})

    async def __call__(self, messages, tools):
        self.requests.append(json.loads(json.dumps(messages, default=str)))
        delay = self.delays.get(len(self.requests))
        if delay:
            await asyncio.sleep(delay)
        return await super().__call__(messages, tools)


def _cell(cid: str, code: str) -> dict:
    return _call(cid, "execute_code", {"code": code})


def _pass(tmp_path: Path, cells: list[str], *, role: str = "write", helper=None, delays=None, start=None, **cfg):
    """One library-mode pass over an empty library, tools called from code; the scripted cells c1, c2, ... Returns
    (writer, model, the cells' outputs by id)."""
    mem = Repo.init_bare(tmp_path / "mem.git")
    ev = EvidenceStore(tmp_path / "e.sqlite")
    ep = _e1()
    ev.index_episode(ep, "1" * 40)
    lookup = lambda eid, i: _ME if (eid, i) == ("e1", 0) else None  # noqa: E731
    gate = Gate(mem, ev, BlobStore(tmp_path / "b"), action_lookup=lookup)
    model = Recording([_cell(f"c{i + 1}", code) for i, code in enumerate(cells)], delays)
    cfg.setdefault("max_calls", len(cells) + 1)
    curate = None
    if role == "curate":
        curate = CurateState(
            commit=mem.head(),
            overlap={"version": 1, "candidates": [], "truncated": []},
            suspects={},
            index_tokens=None,
        )
    writer = WriterPass(
        mem,
        gate,
        ev,
        load=lambda eid: ep,
        model_turn=model,
        config=PassConfig(
            library_mode=True,
            tools_in_code=True,
            pass_record_root=str(tmp_path / "out"),
            **cfg,
        ),
    )
    if helper is not None:
        writer.helper_turn = helper
    if start is not None:
        writer.interpreter_start = start
    req = PassRequest("curate", None, [], False) if role == "curate" else PassRequest(
        "incremental",
        "venmo",
        ["e1"],
        False,
    )
    asyncio.run(writer.run(req, "p1"))
    outputs = {m["tool_call_id"]: _text(m["content"]) for m in writer.messages if m.get("role") == "tool"}
    return writer, model, outputs


DELEGATE_AND_WAIT = (
    "print('DELEGATE', tools.delegate('Run three cells in your box and hand back what they print.', episodes=[]))\n"
    "seen = ''\n"
    "for _ in range(8):\n"
    "    seen += str(tools.wait())\n"
    "    if 'probe finished' in seen:\n"
    "        break\n"
    "print('WAITED', 'probe finished' in seen)\n"
)


def _helper(cells: list[str], seen: list):
    """A helper model that runs *cells*, then ends with ``tools.done``; *seen* gets every request's messages."""

    def factory(label=None):
        state = {"n": 0}

        async def turn(messages, tools):
            seen.append(json.loads(json.dumps(messages, default=str)))
            n = state["n"]
            state["n"] += 1
            code = cells[n] if n < len(cells) else "tools.done('probe finished', [])"
            return _cell(f"h{n + 1}", code), "0.001"

        return turn

    return factory


def _helper_outputs(seen: list) -> dict[str, str]:
    out: dict[str, str] = {}
    for messages in seen:
        for m in messages:
            if m.get("role") == "tool":
                out[m["tool_call_id"]] = _text(m.get("content"))
    return out


# --- the writer and a helper it starts ------------------------------------------------------------------------------


@needs_box
@pytest.mark.timeout(600)
def test_writer_and_helper_have_their_names_bound_and_keep_variables(tmp_path, monkeypatch):
    monkeypatch.setattr(SETTINGS, "UNIFY_TOOLS_IN_CODE", "on")
    monkeypatch.delenv("UNIFY_CELL_STATE", raising=False)
    monkeypatch.setattr(SETTINGS, "UNIFY_CELL_STATE", "persistent")
    helper_seen: list = []
    writer, model, out = _pass(
        tmp_path,
        [PROBE, NEXT, DELEGATE_AND_WAIT],
        helper=_helper([PROBE, NEXT], helper_seen),
    )
    assert helper_seen, ("the helper never ran", out.get("c3"))
    helper_out = _helper_outputs(helper_seen)
    # what holds already: variables kept between cells, for both roles
    assert "NEXT 42" in out["c2"], out["c2"]
    assert "NEXT 42" in helper_out["h2"], helper_out["h2"]
    assert "WAITED True" in out["c3"], out["c3"]
    _assert_probe(out["c1"], "writer")
    _assert_next(out["c2"], "writer")
    _assert_probe(helper_out["h1"], "helper")
    _assert_next(helper_out["h2"], "helper")
    _assert_first_request(model.requests[0], "writer")
    _assert_first_request(helper_seen[0], "helper")


@needs_box
@pytest.mark.timeout(600)
def test_the_curator_has_its_names_bound_and_keeps_variables(tmp_path, monkeypatch):
    monkeypatch.setattr(SETTINGS, "UNIFY_TOOLS_IN_CODE", "on")
    monkeypatch.setattr(SETTINGS, "UNIFY_CELL_STATE", "persistent")
    writer, model, out = _pass(tmp_path, [PROBE, NEXT, "print('LAST')"], role="curate")
    assert writer._role == "curate"
    assert "NEXT 42" in out["c2"], out["c2"]
    _assert_probe(out["c1"], "curator")
    _assert_next(out["c2"], "curator")
    _assert_first_request(model.requests[0], "curator")


@needs_box
@pytest.mark.timeout(600)
def test_under_fresh_cell_state_every_memory_role_loses_its_variables(tmp_path, monkeypatch):
    monkeypatch.setattr(SETTINGS, "UNIFY_TOOLS_IN_CODE", "on")
    monkeypatch.setattr(SETTINGS, "UNIFY_CELL_STATE", "fresh")
    helper_seen: list = []
    writer, model, out = _pass(
        tmp_path / "write",
        ["x = 41\nprint('SET')", ONLY_NEXT, DELEGATE_AND_WAIT],
        helper=_helper(["x = 41\nprint('SET')", ONLY_NEXT], helper_seen),
    )
    (tmp_path / "curate").mkdir()
    _, curator, cout = _pass(tmp_path / "curate", ["x = 41\nprint('SET')", ONLY_NEXT, "print('LAST')"], role="curate")
    assert helper_seen, ("the helper never ran", out.get("c3"))
    helper_out = _helper_outputs(helper_seen)
    for role, shown in (("writer", out["c2"]), ("helper", helper_out["h2"]), ("curator", cout["c2"])):
        assert "NameError" in shown and "NEXT 42" not in shown, (role, shown)
    _assert_first_request(model.requests[0], "writer", persistent=False)
    _assert_first_request(helper_seen[0], "helper", persistent=False)
    _assert_first_request(curator.requests[0], "curator", persistent=False)


# --- what the model is told when the interpreter restarts -----------------------------------------------------------


@needs_box
@pytest.mark.timeout(600)
def test_the_writer_is_told_each_restart_and_then_has_lost_its_variables(tmp_path, monkeypatch):
    monkeypatch.setattr(SETTINGS, "UNIFY_TOOLS_IN_CODE", "on")
    monkeypatch.setattr(SETTINGS, "UNIFY_CELL_STATE", "persistent")
    cells = [
        "x = 41\nprint('SET')",
        "import time\ntime.sleep(60)",
        "print('AFTER TIMEOUT', 'x' in dir())\nx = 41",
        "import threading, time\nthreading.Thread(target=time.sleep, args=(60,), daemon=True).start()",
        "print('AFTER THREAD', 'x' in dir())",
    ]
    _, model, out = _pass(tmp_path, cells, cell_timeout_s=3.0)
    assert f"(the cell timed out after 3 s; {LOST})" in out["c2"], out["c2"]
    assert "AFTER TIMEOUT False" in out["c3"], out["c3"]
    assert f"(a thread this cell started was still running after it ended; {LOST})" in out["c4"], out["c4"]
    assert "AFTER THREAD False" in out["c5"], out["c5"]
    # the opening named these causes before they happened
    system = flat(_system(model.requests[0]))
    for cause in ("ran out of time", "left something running", "sat idle too long"):
        assert cause in system, cause


@needs_box
@pytest.mark.timeout(600)
def test_an_idle_writer_interpreter_is_restarted_and_says_so(tmp_path, monkeypatch):
    monkeypatch.setattr(SETTINGS, "UNIFY_TOOLS_IN_CODE", "on")
    monkeypatch.setattr(SETTINGS, "UNIFY_CELL_STATE", "persistent")

    async def start(**kw):
        kw["profile"] = dataclasses.replace(kw["profile"], idle_restart_s=0.5)
        return await BoxInterpreter.start(**kw)

    # the model takes 3 s before the second cell: longer than the interpreter may sit idle
    _, _, out = _pass(
        tmp_path,
        ["x = 41\nprint('SET')", "print('AFTER IDLE', 'x' in dir())"],
        start=start,
        delays={2: 3.0},
    )
    assert "(the interpreter was idle for more than 0.5 s and was restarted: " in out["c2"], out["c2"]
    assert "AFTER IDLE False" in out["c2"], out["c2"]


# --- `memory` in a persistent box: lazy, and never stale --------------------------------------------------------


SWEEP_WRITE = (
    "import os\n"
    "os.makedirs('/memory/memory/sweep', exist_ok=True)\n"
    "open('/memory/memory/sweep/__init__.py', 'w').write('')\n"
    "open('/memory/memory/sweep/value.py', 'w').write('def f():\\n    return 1\\n')\n"
    "import memory.sweep.value as early\n"
    "print('FIRST', early.f(), memory.sweep.value.f())\n"
)
SWEEP_EDIT = "open('/memory/memory/sweep/value.py', 'w').write('def f():\\n    return 22\\n')\nprint('EDITED')\n"
SWEEP_AGAIN = (
    "lazy = 'imported on first use' in repr(memory)\n"
    "import importlib\n"
    "print('AGAIN', importlib.import_module('memory.sweep.value').f(), early.f(), lazy)\n"
    "print('ATTR', memory.sweep.value.f())\n"
)


@needs_box
@pytest.mark.timeout(600)
def test_a_library_module_edited_after_its_import_is_imported_again_with_the_new_code(tmp_path, monkeypatch):
    monkeypatch.setattr(SETTINGS, "UNIFY_TOOLS_IN_CODE", "on")
    monkeypatch.setattr(SETTINGS, "UNIFY_CELL_STATE", "persistent")
    _, _, out = _pass(tmp_path, [SWEEP_WRITE, SWEEP_EDIT, SWEEP_AGAIN, "print('LAST')"])
    assert "FIRST 1 1" in out["c1"], out["c1"]
    assert "EDITED" in out["c2"], out["c2"]
    # the new code on a new import; the name bound earlier keeps its code; `memory` is lazy again
    assert "AGAIN 22 1 True" in out["c3"], out["c3"]
    assert "ATTR 22" in out["c3"], out["c3"]


@needs_box
@pytest.mark.timeout(600)
def test_the_first_use_of_memory_is_in_that_cell_s_open_log(tmp_path, monkeypatch):
    monkeypatch.setattr(SETTINGS, "UNIFY_TOOLS_IN_CODE", "on")
    monkeypatch.setattr(SETTINGS, "UNIFY_CELL_STATE", "persistent")
    writer, _, out = _pass(
        tmp_path,
        [
            "x = 1\nprint('NO USE', 'imported on first use' in repr(memory))",
            "print('USE', isinstance(memory.index(), str))",
            "print('LAST')",
        ],
    )
    assert "NO USE True" in out["c1"] and "USE True" in out["c2"], out
    rows = [
        json.loads(line)
        for line in (tmp_path / "out" / "p1" / "tool-events.jsonl").read_text().splitlines()
    ]
    cells = sorted(
        r["call"]
        for r in rows
        if r.get("event") == "tool_call" and r.get("tool") == "execute_code" and r.get("agent") == "writer"
    )
    first, second = cells[:2]
    opens = getattr(writer, "box_opens", [])
    init = "/memory/memory/__init__.py"
    assert not [o for o in opens if o["call"] == first and o["path"] == init], opens
    assert [o for o in opens if o["call"] == second and o["path"] == init], opens
