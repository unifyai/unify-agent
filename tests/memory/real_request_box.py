"""One real request over a fixture library, for the tests that check what the actor or a teammate can open.

:func:`run_session` runs the actor (or a teammate the actor starts) in its real sandboxed worker with
``InCodeTools`` and the real handlers, the models scripted (keyless), the library copy exported as a run exports it
(``RequestRun._open_library``) under ``UNIFY_MEMORY=on``. The agent's cells are, in order:

1. :data:`HELP_CELL`, the first cell: what ``memory`` is and ``help(memory)``, before any other use of it;
2. :func:`probe_cell`: reads and searches of the library by every name the texts give it, ``memory.get`` of paths, the
   cards, an answer of ``memory.ask``, the request as ``tools.request()`` and ``memory.request()`` give it, and the
   refusals of paths that do not exist or are outside every root;
3. :func:`open_cell`: every path and handle the agent was shown (the system prompt's memory section, its first
   message, the cards, the answer and the refusals: :func:`printed`) opened with its own ``tools.read`` and
   ``memory.get``.

The request's librarian (scripted) runs ``help(memory)`` as its own first cell too, then answers with two fits. Each
cell writes what it found to a JSON file in the workspace, which the test reads; nothing here mocks a handler, a
mount or the sandbox.
"""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
from typing import Any

from tests.actor.code_act.test_in_code_tools_wiring import _act, _Collect, _text, code
from tests.memory.test_layout import LIB
from tests.memory.test_library_helper import _commit
from tests.scripted_model import Always, ScriptedModel, reply
from unify.actor import in_code_tools
from unify.memory.gitio import Repo
from unify.memory.integration import request as request_mod
from unify.memory.integration.paths import Paths
from unify.memory.integration.state import State
from unify.settings import SETTINGS

REQUEST_TEXT = "Read the date in this note and report it: 2026-10-09."
TASK_TEXT = "Run the calls you are given in your own sandbox."
FUNCTION_ID = "memory.text.dates:parse_date"
NOTE_ID = "notes/text/dates.md"
#: A note whose Rule is longer than a card serves, so its card ends with where the rest is.
LONG_NOTE_ID = "notes/text/long.md"
LONG_NOTE = (
    "---\ntitle: Long\ndescription: How dates are written, at length.\nuses: [memory.text.dates:parse_date]\n---\n"
    "## Rule\n" + "Dates are written year, month, day, joined by hyphens. " * 60 + "\n\n## Evidence\nSeen in e1.\n"
)

HELP_CELL = (
    "import json\n"
    "print('TYPE', type(memory).__name__)\n"
    "help(memory)\n"
)

def assert_package_help(text: str) -> None:
    """*text* is the output of :data:`HELP_CELL`: ``memory`` was already the module when the cell's first line ran,
    and ``help(memory)`` printed the package's documentation, never the lazy stand-in's."""
    assert "TYPE module" in text, text[:600]
    assert ("package memory" in text or "module memory" in text) and "NAME" in text, text[:600]
    assert "_LazyModule" not in text and "imported on first use" not in text, text[:600]


LIBRARIAN_FITS = (
    "tools.finish({'fits': ["
    f"{{'id': '{FUNCTION_ID}', 'call': \"memory.get('{FUNCTION_ID}')('2026-10-09')\", 'why': 'it reads a date'}}, "
    f"{{'id': '{LONG_NOTE_ID}', 'call': \"print(memory.get('{LONG_NOTE_ID}'))\", 'why': 'it says how dates are written'}}"
    "]})\n"
)


def setup_library(core_world: dict, monkeypatch, mode: str) -> Any:
    """The fixture library (and :data:`LONG_NOTE`) committed and opened as the request's copy under *mode*."""
    monkeypatch.setattr(SETTINGS, "UNIFY_MEMORY", mode)
    monkeypatch.setattr(in_code_tools, "SESSIONS", _Collect())
    paths = Paths.under(core_world["state"])
    mem = Repo.init_bare(paths.memory)
    _commit(mem, {**LIB, LONG_NOTE_ID: LONG_NOTE}, "seed the library")
    run = request_mod.RequestRun("r", paths)
    run.state = State.load(paths.state)
    run.pin = mem.head()
    run._shape_rows = lambda consolidate, sha=None: {}  # no evidence store here
    run._open_library(None)
    monkeypatch.setattr(request_mod, "_CURRENT", run)
    return run


def probe_cell(tag: str, mode: str) -> str:
    """The cell that reads the library by every name, gathers the texts the agent is shown and the refusals."""
    reads = ["/memory-library", "/memory-library/INDEX.md", f"/memory-library/{NOTE_ID}", "/memory", f"/memory/{NOTE_ID}"]
    gets = [f"/memory-library/{NOTE_ID}", f"/memory/{NOTE_ID}", NOTE_ID]
    own = f"/memory-library/{NOTE_ID}"
    refused = [
        ("read", ["/workspace/none.txt"]),
        ("read", ["/memory-library/notes/none.md"]),
        ("read", ["/memory/notes/none.md"]),
        ("read", ["/etc/hostname"]),
        ("read", ["/memory-library/../etc/hostname"]),
        ("grep", ["x", "/workspace/none"]),
        ("grep", ["x", "/memory-library/none"]),
        ("grep", ["x", "/etc"]),
    ]
    return f"""
import json
_out = {{"reads": {{}}, "greps": {{}}, "gets": {{}}, "texts": {{}}, "refusals": []}}
def _try(fn):
    try:
        return ["ok", str(fn())]
    except tools.Refused as exc:
        return ["refused:" + str(getattr(exc, "form", "")), str(exc)]
    except Exception as exc:
        return ["raised:" + type(exc).__name__, str(exc)]
for _p in {reads!r}:
    _out["reads"][_p] = _try(lambda: tools.read(_p))
for _p in ("/memory-library", "/memory"):
    _out["greps"][_p] = _try(lambda: tools.grep("year, month", _p))
for _p in {gets!r}:
    _out["gets"][_p] = _try(lambda: memory.get(_p).id)
_out["own_open"] = _try(lambda: open({own!r}).read())
# the copy as the package itself names it (a traceback names its files so)
import os as _os
_root = _os.path.dirname(_os.path.dirname(memory.__file__))
_out["package root"] = _root
_out["reads"][_root + "/INDEX.md"] = _try(lambda: tools.read(_root + "/INDEX.md"))
_out["gets"][_root + "/{NOTE_ID}"] = _try(lambda: memory.get(_root + "/{NOTE_ID}").id)
_out["texts"]["function card"] = _try(lambda: memory.get({FUNCTION_ID!r}))[1]
_out["texts"]["note card"] = _try(lambda: memory.get({NOTE_ID!r}))[1]
_out["texts"]["long note card"] = _try(lambda: memory.get({LONG_NOTE_ID!r}))[1]
_out["texts"]["ask answer"] = _try(lambda: memory.ask("how is a date written?"))[1]
_out["request"] = _try(lambda: tools.request())
_out["memory_request"] = _try(lambda: memory.request())
for _tool, _args in {refused!r}:
    _out["refusals"].append([_tool, _args[-1], _try(lambda: getattr(tools, _tool)(*_args))])
open("/workspace/PROBE_{tag}.json", "w").write(json.dumps(_out))
print("PROBE DONE")
"""


#: A path the agent's texts print: a quoted argument of ``tools.read``/``tools.grep``, or an absolute path under one
#: of the roots the actor and its teammates are given.
_READ_ARG = re.compile(r"tools\.(?:read|grep)\((?:[^()'\"]*?,\s*)?['\"]([^'\"<>]+)['\"]")
_ABSOLUTE = re.compile(r"(?<![\w.:/-])(/(?:workspace|memory-library|memory)(?:/[\w.\-]+)*)")
_HANDLE_CALL = re.compile(r"memory\.get\(['\"]([^'\"<>]+)['\"]\)")
_HANDLE = re.compile(r"(?<![\w./-])(memory\.[a-z_][a-z0-9_]*\.[a-z_][a-z0-9_]*:[a-z_][a-z0-9_]*|notes/[\w./-]+?\.md|"
                     r"capture:[0-9a-f]{16})(?!\w)")
_NOT_IN_LIBRARY = " (not in the library)"


def harness_lines(card: str) -> str:
    """What the harness itself prints on an item's card: its head (id, status, what it rests on), the pointer to the
    rest of a note, and its links, history and records; never the item's own text (a note's body, a function's
    source), which its writer wrote."""
    head, _, rest = card.partition("\n\n")
    body, sep, tail = rest.rpartition("\n\nLinks to: ")
    pointers = [line for line in body.splitlines() if line.startswith("… (")]
    return "\n".join([head, *pointers, ("Links to: " + tail) if sep else ""])


def printed(text: str, skip: tuple[str, ...] = ()) -> tuple[list[str], list[str]]:
    """``(paths, handles)`` that *text* prints for its reader to open; *skip*: paths that are the subject of the text
    (the path a refusal refuses), never a pointer."""
    for s in skip:
        text = text.replace(s, " ")
    paths = [m.group(1) for m in _READ_ARG.finditer(text)]
    paths += [m.group(1).rstrip(".") for m in _ABSOLUTE.finditer(text)]
    handles = [m.group(1) for m in _HANDLE_CALL.finditer(text)]
    handles += [
        m.group(1)
        for m in _HANDLE.finditer(text)
        if not text[m.end(1) :].startswith(_NOT_IN_LIBRARY)
    ]
    return list(dict.fromkeys(p for p in paths if p)), list(dict.fromkeys(h for h in handles if h))


def open_cell(tag: str, paths: list[str], handles: list[str]) -> str:
    """The cell that opens every path with the agent's own ``tools.read`` and every handle with ``memory.get``."""
    return f"""
import json
_res = {{}}
for _p in {paths!r}:
    try:
        tools.read(_p)
        _res["read " + _p] = "ok"
    except Exception as exc:
        _res["read " + _p] = type(exc).__name__ + ": " + str(exc)[:300]
for _h in {handles!r}:
    try:
        memory.get(_h)
        _res["get " + _h] = "ok"
    except Exception as exc:
        _res["get " + _h] = type(exc).__name__ + ": " + str(exc)[:300]
open("/workspace/OPEN_{tag}.json", "w").write(json.dumps(_res))
print("OPEN DONE")
"""


def _tool_texts(call) -> list[str]:
    return [_text(m.get("content")) for m in call.messages if m.get("role") == "tool"]


def _shown(messages: list[dict]) -> dict[str, str]:
    """The agent's memory section (its system prompt from the memory guide on) and its first user message."""
    system = "\n".join(_text(m.get("content")) for m in messages if m.get("role") == "system")
    users = [_text(m.get("content")) for m in messages if m.get("role") == "user"]
    at = system.find("### Memory Library")
    return {"memory section": system[at:] if at >= 0 else "", "first message": users[0] if users else ""}


def _agent_turns(state: dict, workspace: Path, tag: str, mode: str, end: Any) -> list:
    def first(call):
        state["first"] = _shown(call.messages)
        state["system"] = "\n".join(_text(m.get("content")) for m in call.messages if m.get("role") == "system")
        return reply(calls=[code(HELP_CELL)])

    def probe(call):
        state["help"] = _tool_texts(call)[-1]
        return reply(calls=[code(probe_cell(tag, mode))])

    def opens(call):
        state["probe output"] = _tool_texts(call)[-1]
        probe_file = workspace / f"PROBE_{tag}.json"
        state["probe"] = json.loads(probe_file.read_text()) if probe_file.exists() else {}
        texts = dict(state["first"])
        for name, text in (state["probe"].get("texts") or {}).items():
            texts[name] = harness_lines(text) if name.endswith(" card") else text
        for tool, asked, (form, text) in state["probe"].get("refusals") or ():
            texts[f"refusal of {tool} {asked}"] = text
        state["shown texts"] = texts
        paths, handles, by_text = [], [], {}
        for name, text in texts.items():
            asked = name.rsplit(" ", 1)[-1] if name.startswith("refusal of ") else None
            p, h = printed(text, (asked,) if asked else ())
            by_text[name] = (p, h)
            paths += p
            handles += h
        state["printed"] = by_text
        return reply(calls=[code(open_cell(tag, list(dict.fromkeys(paths)), list(dict.fromkeys(handles))))])

    def done(call):
        state["open output"] = _tool_texts(call)[-1]
        open_file = workspace / f"OPEN_{tag}.json"
        state["opened"] = json.loads(open_file.read_text()) if open_file.exists() else {}
        return end

    return [first, probe, opens, done]


def _librarian_turns(state: dict) -> list:
    def first(call):
        return reply(calls=[code(HELP_CELL)])

    def answer(call):
        state.setdefault("librarian help", _tool_texts(call)[-1] if _tool_texts(call) else "")
        return reply(calls=[code(LIBRARIAN_FITS)])

    return [first, Always(answer)]


TEAMMATE_START = f"print(tools.delegate({TASK_TEXT!r}))\n"
TEAMMATE_WAIT = (
    "got = ''\n"
    "for _ in range(8):\n"
    "    got += tools.wait(timeout=240)\n"
    "    if '(done)' in got:\n"
    "        break\n"
    "print(got[-1500:])\n"
)


def run_session(core_world: dict, monkeypatch, role: str, mode: str) -> dict:
    """One request in real boxes, the actor's cells (*role* ``actor``) or a teammate's (``teammate``) running the
    module's three cells; returns what each step saw."""
    setup_library(core_world, monkeypatch, mode)
    workspace = Path(core_world["workspace"])
    tag = f"{role}_{mode}"
    state: dict = {"role": role, "mode": mode}
    if role == "actor":
        model = ScriptedModel(
            actor=_agent_turns(state, workspace, tag, mode, reply("done")),
            librarian=_librarian_turns(state),
            allow=("query_llm",),
        )
    else:
        model = ScriptedModel(
            actor=[reply(calls=[code(TEAMMATE_START)]), reply(calls=[code(TEAMMATE_WAIT)]), reply("done")],
            helper=_agent_turns(state, workspace, tag, mode, reply("the calls ran")),
            librarian=_librarian_turns(state),
            allow=("query_llm",),
        )
    try:
        asyncio.run(asyncio.wait_for(_act(model, REQUEST_TEXT), 1200))
    except BaseException as exc:  # noqa: BLE001 - the state says how far it got; the tests assert on it
        state["error"] = repr(exc)
    state["kinds"] = model.kinds()
    return state
