"""``memory.ask()`` and ``memory.match()`` on the current request (``UNIFY_MEMORY_LIBRARIAN=on``).

* The host gives the current request through ``current_request``: on the actor's and a teammate's channel, never in
  their ``tools`` module, ``tools.list()``, ``tools.help`` or ``dir(tools)``. The memory package calls it through the
  channel client (``tools._call``). With no request known it is refused, and nothing crashes.
* ``memory.ask()`` asks about the current request; a question of the agent's own goes to the librarian alone. No
  stored reader runs (rev 2).
* ``memory.match()`` compares structure only, on the current request's data when it is given nothing; nothing runs.
* An ask's audit event carries the catalogue's row count, never reader facts.

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
    PLANTED_VALUE,
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
        "print(json.dumps({'sent': sent, 'requests': requests, 'notes': [a.note, b.note], 'events': events}, "
        "default=list))\n"
    )
    out = json.loads(_py(root, code))
    (first, first_facts), (second, second_facts) = out["sent"]
    # with no question the request itself is asked about; a question of the agent's own is sent as given
    assert first == REQUEST and second == "Which item totals a table?"
    # rev 2: no reader runs, so nothing but the question is sent, and a question of its own needs no request
    assert first_facts is None and second_facts is None
    assert out["requests"] == [["current_request", {}]] and out["notes"] == ["", ""]
    assert PLANTED_VALUE not in json.dumps(out)
    # the ask's event carries the catalogue's row count, never facts
    (ev, (listed, nothing, rows)), _ = out["events"]
    assert ev == "memory.ask_result" and listed == [] and nothing is True and isinstance(rows, int) and rows > 0


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
    # a question of the agent's own still goes, alone
    assert out["sent"] == [["Which item totals a table?", None]] and out["b"][1] == ""
    assert out["m"][0] == [] and "no structure the matcher reads" in out["m"][1]


# --- match ------------------------------------------------------------------------------------------------------


def test_match_compares_structure_only_on_the_current_request_and_runs_nothing(tmp_path):
    """Rev 2: no reader or applies runs to recognise the request; match lists what has the structure of the current
    request's data, in the catalogue's order, whatever a stored reader would have said."""
    extra = _capture(SAME_SHAPE, NEVER_PARSES, DATA)
    root = _copy(tmp_path, extra)
    # the host gives the request's data as the list of the JSON values its text carries (tool_handlers.current_request)
    code = host(REQUEST, [DATA]) + (
        "m = memory.match()\n"
        "print(json.dumps({'ids': [r.id for r in m], 'text': str(m), 'events': events}, default=list))\n"
    )
    out = json.loads(_py(root, code))
    # the capture whose reader never parses is listed by its structure (it was saved against the request's data):
    # nothing ran to tell otherwise. A library procedure saved against its reader's value is not listed by the
    # request's structure: nothing runs its reader to get that value
    assert "capture:" + SAME_SHAPE in out["ids"] and TOTAL not in out["ids"]
    assert "compares structure only" in out["text"] and "recognise the request" not in out["text"]
    (ev, (listed, _failed)), = out["events"]
    assert ev == "memory.match_result" and [row[0] for row in listed] == out["ids"]
    # without the librarian's catalogue match is the same
    off = _copy(tmp_path, extra, librarian=False, name="off")
    got = json.loads(_py(off, f"import json, memory\nm = memory.match({REQUEST!r}, {DATA!r})\n"
                               "print(json.dumps([[r.id for r in m], str(m)]))\n"))
    assert "capture:" + SAME_SHAPE in got[0] and "compares structure only" in got[1]


def test_the_audit_hook_keeps_the_catalogue_count_of_an_ask_and_refuses_facts():
    audit = CellAudit([])
    audit.begin()
    audit("memory.ask_result", ((TOTAL,), False, 7))
    audit("memory.ask_result", ((), True, ((TOTAL, "parsed", "held"),)))  # rev 1's facts: malformed now
    audit("memory.ask_result", ((), True, -1))
    audit("memory.ask_result", ((), True, True))
    calls = audit.drain()["calls"]
    assert [(a["listed"], a["catalogue_rows"]) for a in calls["asks"]] == [([TOTAL], 7)]
    assert calls["malformed"] == 3
    acc = runtime_calls_empty()
    audit2 = CellAudit([])
    audit2.begin()
    audit2("memory.ask_result", ((TOTAL,), False, 7))
    add_runtime_calls(acc, audit2.drain())
    [row] = acc["asks"]
    assert row["catalogue_rows"] == 7 and "facts" not in row


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
