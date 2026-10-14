"""``memory.ask()`` and ``memory.match()`` on the current request (``UNIFY_MEMORY_LIBRARIAN=on``).

* The host gives the current request through ``current_request``: on the actor's and a teammate's channel, never in
  their ``tools`` module, ``tools.list()``, ``tools.help`` or ``dir(tools)``. The memory package calls it through the
  channel client (``tools._call``). With no request known it is refused, and nothing crashes.
* ``memory.ask()`` runs the stored readers on the current request's text and asks about the request; a question of
  the agent's own goes to the librarian while the readers still read the request.
* ``memory.match()`` lists a procedure only when its reader parsed the current request and its stated ``applies``
  held; a procedure saved against input with the request's structure whose reader did not parse is never listed.
* A reader child that cannot start is ``not-started``; one that never prints its marker, or runs past its time, is
  ``failed``. Each carries a fixed short reason that reaches the host's evidence and log; no value does.

The generated package runs in a fresh isolated interpreter, as a cell imports it, with a stand-in ``tools`` module for
the host. No model is called.
"""

from __future__ import annotations

import json
import logging
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.memory.test_librarian import (
    CAPTURE,
    FAILED,
    PLANTED_VALUE,
    READ_ROWS,
    REQUEST,
    TOTAL,
    _captures,
    _records,
    _summary,
    _tree,
)
from tests.memory.test_library_helper import _py
from unify.common import tool_registry as R
from unify.memory import library_export as lx
from unify.memory import memory_helper
from unify.memory.catalogue import write_files
from unify.memory.integration import request as request_mod
from unify.memory.integration.adapters.audit import CellAudit
from unify.memory.integration.request import add_runtime_calls, runtime_calls_empty
from unify.settings import SETTINGS

#: The current request's data, as the host gives a structured request; TOTAL was saved against input of this shape.
DATA = {"rows": [[1, 2, 3]], "unit": "kg"}
#: A capture saved against input with DATA's structure whose reader never parses: structure alone never lists it.
SAME_SHAPE = "aaaaaaaaaaaaaaaa"
#: A capture whose reader never returns.
SLOW = "bbbbbbbbbbbbbbbb"
NEVER_PARSES = '''"""Read a ledger."""


def read_input(text):
    raise ValueError("not a ledger")


def applies(value):
    return True


def run(value):
    """The value."""
    return value
'''
NEVER_RETURNS = '''"""Wait for a table."""

import time


def read_input(text):
    while True:
        time.sleep(1)


def applies(value):
    return True


def run(value):
    """The value."""
    return value
'''


def _capture(cid: str, procedure: str, shape: object) -> dict[str, bytes]:
    sig = {
        "input": memory_helper.value_shape(shape),
        "relations": [],
        "constraints": {"applies": {"stated": True, "held": True, "sha256": None}},
        "goal": "Read some input.",
    }
    return {
        f".memory/captures/{cid}/procedure.py": procedure.encode(),
        f".memory/captures/{cid}/record.json": json.dumps({"level": "unchecked", "status": "on-trial"}).encode(),
        f".memory/captures/{cid}/signature.json": json.dumps(sig).encode(),
    }


def _copy(tmp_path: Path, extra: dict | None = None, librarian: bool = True, name: str = "copy") -> Path:
    """An actor's library copy as the export writes it, with the captures' snapshot and *extra* captures."""
    root = _tree(tmp_path / name)
    captures = {**_captures(), **(extra or {})}
    files = lx.generated_library_mode(
        root, records=_records(), librarian=librarian, summary_of=_summary, captures=captures,
    )
    write_files(root, {**files, **captures})
    return root


def host(request: str | None, data: object = None) -> str:
    """Cell code: a stand-in host whose ``_call`` answers ``current_request`` with *request* and *data* (refused when
    *request* is None) and whose ``ask_library`` records what it is sent; then the package's events are kept."""
    return (
        "import json, sys, types\n"
        "tools = types.ModuleType('tools')\n"
        "sent, requests = [], []\n"
        "class Refused(Exception):\n"
        "    pass\n"
        "tools.Refused = Refused\n"
        "def ask_library(question, facts=None):\n"
        "    sent.append([question, facts])\n"
        "    return json.dumps({'fits': []})\n"
        "def _call(tool, args):\n"
        "    requests.append([tool, args])\n"
        f"    if {request!r} is None:\n"
        "        raise Refused('refused: the current request is not available here (no current request is known)')\n"
        f"    return json.dumps({{'text': {request!r}, 'data': {data!r}}})\n"
        "tools.ask_library = ask_library\n"
        "tools._call = _call\n"
        "sys.modules['tools'] = tools\n"
        "events = []\n"
        "sys.addaudithook(lambda e, a: events.append([e, a]) if e in ('memory.ask_result', 'memory.match_result') "
        "else None)\n"
        "import memory\n"
    )


def _tuples(value: object) -> object:
    """An event's arguments as the audit hook gets them (JSON turned its tuples into lists)."""
    return tuple(_tuples(v) for v in value) if isinstance(value, list) else value


class _Logged(logging.Handler):
    """The request log's warnings, as the host's log handlers receive them (whatever the logger's propagation)."""

    def __init__(self) -> None:
        super().__init__(logging.WARNING)
        self.lines: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.lines.append(record.getMessage())

    def __enter__(self) -> "_Logged":
        log = logging.getLogger(request_mod.__name__)
        self._level = log.level
        log.addHandler(self)
        log.setLevel(logging.WARNING)
        return self

    def __exit__(self, *exc: object) -> None:
        log = logging.getLogger(request_mod.__name__)
        log.removeHandler(self)
        log.setLevel(self._level)

    @property
    def text(self) -> str:
        return "\n".join(self.lines)


def _host_record(events: list) -> dict:
    """The events through the cell's audit hook and into the request's runtime evidence, as a cell's drain is."""
    audit = CellAudit([])
    audit.begin()
    for ev, args in events:
        audit(ev, _tuples(args))
    acc = runtime_calls_empty()
    add_runtime_calls(acc, audit.drain())
    return acc


# --- ask --------------------------------------------------------------------------------------------------------


def test_ask_reads_the_current_request_and_sends_the_question_given(tmp_path):
    root = _copy(tmp_path)
    code = host(REQUEST) + (
        "a = memory.ask()\n"
        "b = memory.ask('Which item totals a table?')\n"
        "print(json.dumps({'sent': sent, 'requests': requests, 'notes': [a.note, b.note]}))\n"
    )
    out = json.loads(_py(root, code))
    (first, first_facts), (second, second_facts) = out["sent"]
    # with no question the request itself is asked about; a question of the agent's own is sent as given
    assert first == REQUEST and second == "Which item totals a table?"
    # either way the readers read the current request: the table reader parsed it and its applies held
    for facts in (first_facts, second_facts):
        assert facts == [
            {"id": TOTAL, "reader": READ_ROWS, "read": "parsed", "applies": "held"},
            {"id": "capture:" + CAPTURE, "reader": "capture:" + CAPTURE, "read": "parsed", "applies": "did-not-hold"},
        ]
    assert out["requests"] == [["current_request", {}]] * 2 and out["notes"] == ["", ""]
    assert PLANTED_VALUE not in json.dumps(out)


def test_with_no_current_request_nothing_crashes_and_the_note_says_so(tmp_path):
    root = _copy(tmp_path)
    code = host(None) + (
        "a = memory.ask()\n"
        "b = memory.ask('Which item totals a table?')\n"
        "m = memory.match()\n"
        "print(json.dumps({'sent': sent, 'a': [list(a), a.note, str(a)], 'b': [list(b), b.note], "
        "'m': [list(m), m.note]}))\n"
    )
    out = json.loads(_py(root, code))
    # nothing to ask about: nothing is sent, and the answer says why
    assert out["a"][0] == [] and "not available here, so nothing was asked" in out["a"][1]
    assert out["a"][2] == out["a"][1] + "\n"
    # a question still goes, with no facts, and the note says no reader ran
    assert out["sent"] == [["Which item totals a table?", []]]
    assert "no stored reader ran" in out["b"][1]
    assert out["m"][0] == [] and "not available here, so no stored reader ran" in out["m"][1]


# --- match ------------------------------------------------------------------------------------------------------


def test_match_lists_only_what_its_reader_parsed_with_its_applies_holding(tmp_path):
    extra = _capture(SAME_SHAPE, NEVER_PARSES, DATA)
    root = _copy(tmp_path, extra)
    code = host(REQUEST, DATA) + (
        "m = memory.match()\n"
        "f = memory.match(inputs=['x', 'y'])\n"
        "print(json.dumps({'ids': [r.id for r in m], 'matched': [r.matched for r in m], 'text': str(m), "
        "'failed': [r.id for r in f.failed], 'f_ids': [r.id for r in f], 'events': events}, default=list))\n"
    )
    out = json.loads(_py(root, code))
    # the table reader parsed the request and its applies held; the word capture parsed it but its applies did not
    # hold; the capture saved against input with the request data's structure never parsed it: none of those two
    assert out["ids"] == [TOTAL] and out["matched"] == ["applies-held"]
    assert "1 stored procedure(s) recognise the request" in out["text"] and "compares structure only" not in out["text"]
    # an explicit input changes only the order and the failed attempts shown apart, never what is listed
    assert out["f_ids"] == [TOTAL] and out["failed"] == ["capture:" + FAILED]
    (ev, (listed, failed)), _ = out["events"]
    assert ev == "memory.match_result" and [row[0] for row in listed] == [TOTAL] and failed == []
    # without the librarian's dispatch file match is as it was: structure alone lists the same-shape capture
    off = _copy(tmp_path, extra, librarian=False, name="off")
    got = json.loads(_py(off, f"import json, memory\nm = memory.match({REQUEST!r}, {DATA!r})\n"
                               "print(json.dumps([[r.id for r in m], str(m)]))\n"))
    assert "capture:" + SAME_SHAPE in got[0] and "compares structure only" in got[1]


# --- a reader that gives no outcome -----------------------------------------------------------------------------


def test_a_child_that_cannot_start_or_never_prints_its_marker_is_labelled_with_its_reason(tmp_path):
    root = _copy(tmp_path)
    code = host(REQUEST) + (
        "real = sys.executable\n"
        "sys.executable = '/nonexistent/python3'\n"
        "a = memory.ask()\n"
        "sys.executable = real\n"
        "memory._RECOGNISER_RUNNER = 'import sys'\n"
        "b = memory.ask()\n"
        "print(json.dumps({'sent': sent, 'notes': [a.note, b.note], 'events': events}, default=list))\n"
    )
    out = json.loads(_py(root, code))
    (_, started), (_, marked) = out["sent"]
    # the host is told the labels only; the reason travels in the audit event
    assert [(f["id"], f["read"], f["applies"]) for f in started] == [
        (TOTAL, "not-started", "not-run"), ("capture:" + CAPTURE, "not-started", "not-run"),
    ]
    assert [f["read"] for f in marked] == ["failed", "failed"] and all("reason" not in f for f in started + marked)
    assert "start: FileNotFoundError" in out["notes"][0] and "no marker" in out["notes"][1]
    (_, one), (_, two) = out["events"]
    assert one[2][0] == [TOTAL, "not-started", "not-run", "start: FileNotFoundError"]
    assert two[2][0] == [TOTAL, "failed", "not-run", "no marker"]

    with _Logged() as log:
        acc = _host_record(out["events"])
    first, second = acc["asks"]
    assert first["facts"][0] == {"id": TOTAL, "read": "not-started", "applies": "not-run",
                                 "reason": "start: FileNotFoundError"}
    assert second["facts"][1] == {"id": "capture:" + CAPTURE, "read": "failed", "applies": "not-run",
                                  "reason": "no marker"}
    logged = log.text
    assert f"memory.ask: the stored reader of {TOTAL} not-started (start: FileNotFoundError)" in logged
    assert f"memory.ask: the stored reader of capture:{CAPTURE} failed (no marker)" in logged


def test_a_reader_that_cannot_be_loaded_has_failed_with_its_reason_and_never_reads_as_not_parsing(tmp_path):
    root = _copy(tmp_path)
    module = root / "memory" / "tables" / "readers.py"  # TOTAL's reader, READ_ROWS, lives here
    module.write_text(module.read_text() + "\nraise RuntimeError('this module breaks on import')\n")
    code = host(REQUEST) + (
        "a = memory.ask()\nprint(json.dumps({'sent': sent, 'note': a.note, 'events': events}, default=list))\n"
    )
    out = json.loads(_py(root, code))
    ((_, facts),) = out["sent"]
    total = next(f for f in facts if f["id"] == TOTAL)
    assert (total["read"], total["applies"]) == ("failed", "not-run"), total
    assert "load: RuntimeError" in out["note"]
    ((_, event),) = out["events"]
    assert [TOTAL, "failed", "not-run", "load: RuntimeError"] in event[2]


def test_a_reader_past_its_time_is_failed_and_the_others_keep_their_outcomes(tmp_path):
    root = _copy(tmp_path, _capture(SLOW, NEVER_RETURNS, DATA))
    code = host(REQUEST, DATA) + (
        "memory.READ_TIMEOUT_S, memory.READERS_TIMEOUT_S = 1.0, 20.0\n"
        "a = memory.ask()\n"
        "m = memory.match()\n"
        "print(json.dumps({'sent': sent, 'ids': [r.id for r in m], 'notes': [a.note, m.note], 'events': events}, "
        "default=list))\n"
    )
    out = json.loads(_py(root, code))
    ((_, facts),) = out["sent"]
    by_id = {f["id"]: (f["read"], f["applies"]) for f in facts}
    assert by_id["capture:" + SLOW] == ("failed", "not-run")
    assert by_id[TOTAL] == ("parsed", "held")
    assert out["ids"] == [TOTAL]
    for note in out["notes"]:
        assert f"capture:{SLOW} failed (timeout after 1 s)" in note, note
    (_, ask), (_, match) = out["events"]
    assert [row for row in ask[2] if len(row) == 4] == [["capture:" + SLOW, "failed", "not-run", "timeout after 1 s"]]
    assert match[2] == [["capture:" + SLOW, "failed", "not-run", "timeout after 1 s"]]

    with _Logged() as log:
        acc = _host_record(out["events"])
    assert acc["matches"][0]["listed"][0]["id"] == TOTAL
    assert f"memory.match: the stored reader of capture:{SLOW} failed (timeout after 1 s)" in log.text
    assert f"memory.ask: the stored reader of capture:{SLOW} failed (timeout after 1 s)" in log.text


def test_the_audit_hook_keeps_only_fixed_reasons_for_readers_without_an_outcome():
    good = ((TOTAL, "failed", "not-run", "timeout after 30 s"),)
    planted = ((TOTAL, "failed", "not-run", "the value was 42"),)
    labelled = ((TOTAL, "parsed", "held", "no marker"),)  # a reason only where no outcome was given
    bare = ((TOTAL, "not-started", "not-run"),)  # and never without one there
    audit = CellAudit([])
    audit.begin()
    for rows in (good, planted, labelled, bare):
        audit("memory.ask_result", ((), True, rows))
    audit("memory.match_result", ((), (), planted))
    audit("memory.match_result", ((), (), good))
    calls = audit.drain()["calls"]
    assert [a["facts"] for a in calls["asks"]] == [
        [{"id": TOTAL, "read": "failed", "applies": "not-run", "reason": "timeout after 30 s"}],
    ]
    assert [m.get("no_outcome") for m in calls["matches"]] == [
        [{"id": TOTAL, "read": "failed", "applies": "not-run", "reason": "timeout after 30 s"}],
    ]
    assert calls["malformed"] == 4


# --- the host tool ----------------------------------------------------------------------------------------------

CLIENT = '''
class Result(str):
    paths = ()
    next = None
class Refused(Exception):
    pass
class NotRun(Refused):
    pass
class ToolError(Exception):
    pass
class ChannelClosed(Exception):
    pass
CALLS = []
def _call(tool, args):
    CALLS.append((tool, args))
    return Result("%s answered" % tool)
'''


SHELL = '''
class ShellResult(str):
    returncode = 0
def sh(command, timeout=None):
    return ShellResult("ran " + command)
'''


def test_the_request_tool_is_on_the_channel_but_never_in_tools(monkeypatch, capsys):
    monkeypatch.setattr(R, "client_source", lambda: CLIENT)
    monkeypatch.setattr(R, "shell_source", lambda: SHELL)
    assert R.CHANNEL_ONLY == {"current_request"}
    assert R.channel_only("actor") == R.channel_only("teammate") == ("current_request",)
    assert R.channel_only("writer") == () and "current_request" not in R.allow_list("actor", frozenset({"librarian"}))
    mod = types.ModuleType("tools")
    names = R.allow_list("actor", frozenset({"librarian", "store", "helper"})) + R.channel_only("actor")
    exec(compile(R.render_stub(names, "actor"), "tools.py", "exec"), mod.__dict__)
    assert "current_request" not in dir(mod) and "current_request" not in mod.__all__
    mod.list()
    listed = capsys.readouterr().out
    mod.help("current_request")
    helped = capsys.readouterr().out
    assert "ask_library(" in listed and "current_request" not in listed
    assert helped.startswith("no tool named 'current_request'")
    assert mod._call("current_request", {}) == "current_request answered"


@pytest.mark.asyncio
async def test_the_host_gives_the_acts_request_and_refuses_when_none_is_known(tmp_path, monkeypatch):
    from unify.actor import in_code_tools
    from unify.actor import tool_handlers as H

    monkeypatch.setattr(SETTINGS, "UNIFY_TOOLS_IN_CODE", "on")
    monkeypatch.setattr(request_mod, "_CURRENT", None)

    def session(request: object, name: str) -> in_code_tools.InCodeTools:
        return in_code_tools.InCodeTools(
            agent="actor",
            name="root",
            record=SimpleNamespace(),
            workspace_root=str(tmp_path),
            shadow={},
            cell_limit_s=None,
            events_folder=tmp_path / name,
            request=request,
        )

    def ask(tools: in_code_tools.InCodeTools, agent: str = "actor") -> object:
        return H.current_request(SimpleNamespace(state=tools.state, agent=agent), {})

    text = session("Total the table, please.", "a")
    assert "current_request" in text.channel_allow and "current_request" in text.specs
    assert "current_request" not in text.allow and "current_request" not in text.stub
    assert json.loads(ask(text).text) == {"text": "Total the table, please.", "data": []}
    # exactly what the first user message shows: never a field of the request it does not show
    structured = session({"task": "total", "rows": [[1, 2]], "task_id": "arc-0451"}, "b")
    assert ask(structured).status == "refused"  # an object with no message text shows the model no text
    shown_request = {"role": "user", "content": "Total the table.", "task_id": "arc-0451", "label": "x"}
    shown = session(shown_request, "e")
    assert json.loads(ask(shown).text) == {"text": "Total the table.", "data": []}
    assert "arc-0451" not in ask(shown).text and "label" not in ask(shown).text
    grid = '{"train": [{"input": [[1, 2]], "output": [[2, 1]]}], "test": [{"input": [[3, 4]]}]}'
    arc = session(grid, "f")  # a request that is all JSON: its one value is the data, read from the text shown
    assert json.loads(ask(arc).text) == {"text": grid, "data": [json.loads(grid)]}
    # the JSON values a text carries among its prose, in order: the bound request's data
    prose = 'Totals for {"rows": [[1, 2]]} and then [3, 4], please.'
    mixed = session(prose, "g")
    assert json.loads(ask(mixed).text) == {"text": prose, "data": [{"rows": [[1, 2]]}, [3, 4]]}
    # each text is the one the loop's first user message carries (loop.py's single-message path wraps a request that
    # is not a message as the user's content)
    from unify.common._async_tool.bound_request import text_of

    for tools, req in ((text, "Total the table, please."), (shown, shown_request), (arc, grid)):
        first = req if isinstance(req, dict) else {"role": "user", "content": req}
        assert json.loads(ask(tools).text)["text"] == text_of(first)
    message = session([{"role": "user", "content": "first"}, {"role": "user", "content": "the latest"}], "c")
    assert json.loads(ask(message).text) == {"text": "the latest", "data": []}
    nothing = session(None, "d")
    refused = ask(nothing)
    assert refused.status == "refused" and refused.form == "unavailable"
    # the memory run's request is the actor's fallback only: a teammate gets only the task it was given
    monkeypatch.setattr(request_mod, "_CURRENT", SimpleNamespace(request="the memory run's request"))
    assert json.loads(ask(nothing).text) == {"text": "the memory run's request", "data": []}
    # the channel's real context (the request's binding, as the channel copies it per call) names its agent
    assert nothing.binding.agent == "actor"
    assert json.loads(H.current_request(nothing.binding, {}).text) == {"text": "the memory run's request", "data": []}
    teammate = ask(nothing, agent="teammate")
    assert teammate.status == "refused" and teammate.form == "unavailable"
    assert "the memory run's request" not in teammate.text
    for tools in (text, structured, message, nothing, shown, arc, mixed):
        await tools.close()
