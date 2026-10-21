"""The memory agents with tools in code **off** (their tools stay JSON tools): by default ``execute_code`` runs in one
persistent interpreter per agent, with no tool channel and no ``tools`` stub, so the writer, the curator and a helper
keep their variables between calls and find ``memory`` and ``help`` bound with no import. Under
``UNIFY_CELL_STATE=fresh`` every call runs in a fresh process, as before. The ``execute_code`` text and the briefs say
which; no brief says ``import memory``, and the helper's no longer says that only files persist. A ``finish`` stops
the interpreter before the gate judges, so a refused finish's next call says the interpreter was stopped and has lost
its variables.

The model is scripted (no key, no network); the boxes need bubblewrap and prlimit, as the other box tests do.
"""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path

import pytest

from tests.memory.test_tool_channel_box import needs_box
from tests.memory.test_writer_pass import Turns, _call, _e1, _ME
from unify.memory import helpers
from unify.memory import library_prompts as lp
from unify.memory.blobs import BlobStore
from unify.memory.curate import CurateState
from unify.memory.evidence import EvidenceStore
from unify.memory.gate import Gate
from unify.memory.gitio import Repo
from unify.memory.trigger import PassRequest
from unify.memory.writer_pass import PassConfig, WriterPass
from unify.settings import SETTINGS

# ==== names and sentences this file assumes (the spec gives them; adapt them here, in one place) ====================
#: design D.3: the writer's and curator's execute_code text with the persistent default
WRITER_TOOL_TEXT = "Run Python code as a cell in your persistent interpreter, in your confined box"
#: design D.4: the helper's execute_code text with the persistent default
HELPER_TOOL_TEXT = "Run Python in your persistent interpreter, in your box"
#: design D.4: the helper's off brief
HELPER_BRIEF_TEXT = "in an interpreter that keeps its variables between calls"
#: what the tool texts and briefs say today, and must not with the persistent default
FRESH_TEXTS = ("fresh confined process", "fresh process", "Only files persist between calls")
#: design B.2 (tools in code off): the memory agents' pre-bound set
PREBOUND = ("memory", "help")
MUST_NOT = ("tools", "query_llm", "list_llms", "unillm", "run_coro_sync", "display", "ModelRequestError")
#: the note after an end call stopped the interpreter (tool_handlers.RESTARTED; design C.2)
STOPPED = "(the interpreter was stopped by the end of the last call; variables are gone, files remain)"
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
PROBE = probe(CHECK)
NEXT = "print('NEXT', x + 1)\nprint('INDEX', isinstance(memory.index(), str))\nhelp(len)\nprint('HELPED')\n"
ONLY_NEXT = "print('NEXT', x + 1)\n"


def flat(text: str) -> str:
    return " ".join(str(text).split())


def _text(content) -> str:
    if isinstance(content, list):
        return "\n".join(str(p.get("text", "")) if isinstance(p, dict) else str(p) for p in content)
    return "" if content is None else str(content)


def _bound(output: str) -> set[str]:
    line = next(ln for ln in output.splitlines() if ln.startswith("BOUND "))
    return set(json.loads(line[len("BOUND ") :].replace("'", '"')))


def _tool_text(tools: list, name: str = "execute_code") -> str:
    (spec,) = [t for t in tools if t.get("function", {}).get("name") == name]
    return flat(spec["function"].get("description", ""))


class Recording(Turns):
    """The scripted model, keeping every request's messages and tools as sent."""

    def __init__(self, messages):
        super().__init__(messages)
        self.requests: list[tuple[list[dict], list]] = []

    async def __call__(self, messages, tools):
        self.requests.append((json.loads(json.dumps(messages, default=str)), json.loads(json.dumps(tools))))
        return await super().__call__(messages, tools)


def _pass(tmp_path: Path, calls: list[dict], *, role: str = "write", helper=None, **cfg):
    mem = Repo.init_bare(tmp_path / "mem.git")
    ev = EvidenceStore(tmp_path / "e.sqlite")
    ep = _e1()
    ev.index_episode(ep, "1" * 40)
    lookup = lambda eid, i: _ME if (eid, i) == ("e1", 0) else None  # noqa: E731
    gate = Gate(mem, ev, BlobStore(tmp_path / "b"), action_lookup=lookup)
    model = Recording(calls)
    cfg.setdefault("max_calls", len(calls) + 1)
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
        config=PassConfig(library_mode=True, tools_in_code=False, pass_record_root=str(tmp_path / "out"), **cfg),
        curate=curate,
    )
    if helper is not None:
        writer.helper_turn = helper
    req = (
        PassRequest("curate", None, [], False)
        if role == "curate"
        else PassRequest("incremental", "venmo", ["e1"], False)
    )
    asyncio.run(writer.run(req, "p1"))
    outputs = {m["tool_call_id"]: _text(m["content"]) for m in writer.messages if m.get("role") == "tool"}
    return writer, model, outputs


def _cell(cid: str, code: str) -> dict:
    return _call(cid, "execute_code", {"code": code})


def _helper(cells: list[str], seen: list):
    def factory(label=None):
        state = {"n": 0}

        async def turn(messages, tools):
            seen.append((json.loads(json.dumps(messages, default=str)), json.loads(json.dumps(tools))))
            n = state["n"]
            state["n"] += 1
            if n < len(cells):
                return _cell(f"h{n + 1}", cells[n]), "0.001"
            return _call(f"h{n + 1}", "done", {"summary": "probe finished", "claims": []}), "0.001"

        return turn

    return factory


def _helper_outputs(seen: list) -> dict[str, str]:
    out: dict[str, str] = {}
    for messages, _ in seen:
        for m in messages:
            if m.get("role") == "tool":
                out[m["tool_call_id"]] = _text(m.get("content"))
    return out


def _assert_probe(output: str, role: str) -> None:
    bound = _bound(output)
    assert bound == set(PREBOUND), (role, sorted(bound), output)


def _assert_next(output: str, role: str) -> None:
    assert "NEXT 42" in output and "INDEX True" in output and "HELPED" in output, (role, output)


# --- persistent by default --------------------------------------------------------------------------------------


@needs_box
@pytest.mark.timeout(600)
def test_writer_and_helper_keep_variables_and_have_memory_bound(tmp_path, monkeypatch):
    monkeypatch.setattr(SETTINGS, "UNIFY_TOOLS_IN_CODE", "off")
    monkeypatch.setattr(SETTINGS, "UNIFY_CELL_STATE", "persistent")
    seen: list = []
    calls = [
        _cell("c1", PROBE),
        _cell("c2", NEXT),
        _call("d", "delegate", {"task": "Run two cells in your box and hand back what they print.", "episodes": []}),
        _call("w1", "wait", {}),
        _call("w2", "wait", {}),
    ]
    writer, model, out = _pass(tmp_path, calls, helper=_helper([PROBE, NEXT], seen))
    assert seen, ("the helper never ran", out)
    hout = _helper_outputs(seen)
    _assert_next(out["c2"], "writer")
    _assert_next(hout["h2"], "helper")
    _assert_probe(out["c1"], "writer")
    _assert_probe(hout["h1"], "helper")
    messages, tools = model.requests[0]
    text = _tool_text(tools)
    assert WRITER_TOOL_TEXT in text, text
    assert not any(t in text for t in FRESH_TEXTS), text
    system = "\n".join(_text(m.get("content")) for m in messages if m.get("role") == "system")
    assert not re.search(r"\bimport memory\b", system)
    _, helper_tools = seen[0]
    text = _tool_text(helper_tools)
    assert HELPER_TOOL_TEXT in text and not any(t in text for t in FRESH_TEXTS), text


@needs_box
@pytest.mark.timeout(600)
def test_the_curator_keeps_variables_and_has_memory_bound(tmp_path, monkeypatch):
    monkeypatch.setattr(SETTINGS, "UNIFY_TOOLS_IN_CODE", "off")
    monkeypatch.setattr(SETTINGS, "UNIFY_CELL_STATE", "persistent")
    _, model, out = _pass(tmp_path, [_cell("c1", PROBE), _cell("c2", NEXT), _cell("c3", "print('LAST')")], role="curate")
    _assert_next(out["c2"], "curator")
    _assert_probe(out["c1"], "curator")
    _, tools = model.requests[0]
    text = _tool_text(tools)
    assert WRITER_TOOL_TEXT in text and not any(t in text for t in FRESH_TEXTS), text


@needs_box
@pytest.mark.timeout(600)
def test_under_fresh_cell_state_each_call_starts_a_fresh_process(tmp_path, monkeypatch):
    monkeypatch.setattr(SETTINGS, "UNIFY_TOOLS_IN_CODE", "off")
    monkeypatch.setattr(SETTINGS, "UNIFY_CELL_STATE", "fresh")
    _, model, out = _pass(tmp_path, [_cell("c1", "x = 41\nprint('SET')"), _cell("c2", ONLY_NEXT), _cell("c3", "1")])
    assert "NameError" in out["c2"] and "NEXT 42" not in out["c2"], out["c2"]
    _, tools = model.requests[0]
    assert WRITER_TOOL_TEXT not in _tool_text(tools)


@needs_box
@pytest.mark.timeout(600)
def test_a_refused_finish_stops_the_interpreter_and_the_next_call_says_so(tmp_path, monkeypatch):
    monkeypatch.setattr(SETTINGS, "UNIFY_TOOLS_IN_CODE", "off")
    monkeypatch.setattr(SETTINGS, "UNIFY_CELL_STATE", "persistent")
    calls = [
        _cell("c1", "x = 41\nprint('SET')"),
        _call("f", "finish", {"summary": "nothing stored yet"}),
        _cell("c2", "print('AFTER FINISH', 'x' in dir())"),
        _cell("c3", "print('LAST')"),
    ]
    writer, _, out = _pass(tmp_path, calls)
    assert out["f"].startswith("not finished: G1: no manifest"), out["f"]
    assert STOPPED in out["c2"] and "AFTER FINISH False" in out["c2"], out["c2"]


# --- the briefs, without a box ----------------------------------------------------------------------------------


def test_the_off_briefs_say_memory_is_bound_and_drop_fresh_process_wording(monkeypatch):
    monkeypatch.setattr(SETTINGS, "UNIFY_TOOLS_IN_CODE", "off")
    monkeypatch.setattr(SETTINGS, "UNIFY_CELL_STATE", "persistent")
    texts = {
        "writer": lp.write_brief_now(cell_s=60.0),
        "curator": lp.curate_brief_now(cell_s=60.0),
        "helper": helpers.helper_brief(False),
    }
    for role, text in texts.items():
        body = flat(text)
        assert not re.search(r"\bimport memory\b", body), role
        assert "`memory` is already bound" in body or "is already available" in body, role
        assert "Only files persist between calls" not in body, role
    assert HELPER_BRIEF_TEXT in flat(texts["helper"])
    assert "fresh process" not in flat(texts["helper"])
