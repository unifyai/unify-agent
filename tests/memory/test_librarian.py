"""The librarian (``UNIFY_MEMORY_LIBRARIAN``, design J(b)): the catalogue and dispatch file each export writes,
``memory.ask`` in the generated package, the ``ask_library`` tool, the librarian's session on the memory agents'
runtime, the actor's constant pointer, the use record's ask fields, and the switch itself. Keyless: the librarian's
model is the scripted transport (``tests/scripted_model.py``), and the boxes are real where they are named.

The library's item records are what production writes (:mod:`tests.memory.gate_records`): a pass lands the capture of
episode ``e1`` as :data:`TOTAL`, read by the reader it was landed with (:data:`READ_ROWS`) and with its ``applies``
(:data:`WIDE`); the library gate checks it, and the records are ``lifecycle.next_records`` with the gate's
verification and the capture store's landing. Building them runs code confined, so the tests that read the library
need bubblewrap."""

from __future__ import annotations

import atexit
import copy
import functools
import json
import os
import re
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

import pytest

from tests.memory.gate_records import Built, Landing, build
from tests.memory.test_episodes import _ep
from tests.memory.test_library_helper import _py
from tests.scripted_model import ScriptedModel, reply, scripted
from unify.common import tool_registry as R
from unify.memory import library_export as lx
from unify.memory import library_prompts as lp
from unify.memory import librarian as L
from unify.memory import memory_helper
from unify.memory import sandbox_run as sr
from unify.memory.analysis import use
from unify.memory.catalogue import write_files
from unify.memory.integration.adapters.audit import CellAudit
from unify.memory.integration.request import add_runtime_calls, library_helper_calls, runtime_calls_empty
from unify.memory.episodes import Action
from unify.memory.layout import RESERVED_INIT, RESERVED_STUB
from unify.memory.library_index import signature_text
from unify.settings import SETTINGS, ProductionSettings

needs_box = pytest.mark.skipif(
    shutil.which("bwrap") is None or not os.access(sr.PRLIMIT, os.X_OK),
    reason="bubblewrap and prlimit required",
)

#: The landed procedure, the reader it was landed with and its ``applies``, as ``land_capture`` names them.
TOTAL = "memory.tables.procedures:total"
READ_ROWS = "memory.tables.readers:read_total"
WIDE = "memory.tables.procedures:_total_applies"
COUNT = "memory.tables.count:count_rows"
NOTE = "notes/tables/sums.md"
CAPTURE = "0123456789abcdef"
FAILED = "fedcba9876543210"
#: What the reader gives on the test's request: never in the request text itself, so seen nowhere but in the box.
PLANTED_FACTOR = 7919
REQUEST = "4096 8191 3\n5 7 11"
PLANTED_VALUE = str(4096 * PLANTED_FACTOR)
#: Identifiers by their form alone (signatures.strip_identifiers): never in the catalogue.
IDS = ("run_000123", "0f3a9c7e1b", "123e4567-e89b-12d3-a456-426614174000")
#: The request the landed procedure's episode answered and the total it handed back: only that episode and the
#: library's fixtures hold them, never the catalogue.
RECORDED_REQUEST = "2 3 5\n7 11 13"
RECORDED_TOTAL = str(41 * PLANTED_FACTOR)

#: The procedure the episode-end fork staged for ``e1``, which the pass lands as :data:`TOTAL`.
PROCEDURE_TOTAL = f'''"""Sum every number of a table's rows."""


def read_input(text):
    """Read whitespace-separated rows of integers from text.

    Raises ValueError on text that holds no such rows."""
    rows = [[int(x) * {PLANTED_FACTOR} for x in line.split()] for line in text.splitlines() if line.strip()]
    if not rows:
        raise ValueError("no rows")
    return rows


def applies(value):
    """Whether every row holds more than two numbers."""
    return all(len(r) > 2 for r in value)


def run(value):
    """Sum every number of a table's rows.

    It takes the rows read_rows gives.

    Notes: notes/tables/sums.md"""
    return sum(sum(r) for r in value)
'''
COUNT_SRC = '''"""Counting rows."""


def count_rows(rows):
    """How many rows a table has."""
    return len(rows)
'''
NOTE_SRC = """---
title: Sums of tables
description: How table sums are taken
uses: [memory.tables.procedures:total]
---
Sum each row, then the row sums.
"""
PROCEDURE = '''"""Split comma-separated words and reverse them."""


def read_input(text):
    return text.split(",")


def applies(value):
    return len(value) > 1


def run(value):
    """The words in reverse order."""
    return value[::-1]
'''


def _summary(episode: str) -> str | None:
    if episode != "e1":
        return None
    return (
        f"## Asked\nAdd up a table of numbers from {IDS[0]} and report the total ({IDS[1]}).\n"
        f"## Approach\n- Read the rows (`cell:0` \"rows = read(text)\")\n- Summed them, run {IDS[2]} "
        "(`cell:1` \"sum(...)\")\n## Worked\n- the sum\n## Failed\nnone\n## Outcome\nnot shown\n"
    )


#: The library stored before the pass.
BASE = {
    "memory/tables/__init__.py": '"""Number tables."""\n',
    "memory/tables/count.py": COUNT_SRC,
    "notes/tables/sums.md": NOTE_SRC,
}


@functools.lru_cache(maxsize=1)
def _built() -> Built:
    """The pass that lands the capture of episode ``e1`` as :data:`TOTAL`, the gate's check and the item records;
    once per process."""
    root = Path(tempfile.mkdtemp(prefix="librarian-tables-"))
    atexit.register(shutil.rmtree, root, True)
    episode = _ep(
        episode_id="e1",
        request=[RECORDED_REQUEST],
        transcript=[{"role": "user", "content": RECORDED_REQUEST}],
        cells=[],
        actions=[
            Action(-1, "dialogue:user", "answer", [int(RECORDED_TOTAL)], {}, "Accepted.", "ok", "unknown", kind="dialogue"),
        ],
    )
    return build(
        root,
        episodes={"e1": episode},
        base=BASE,
        landings=[Landing("e1", PROCEDURE_TOTAL, _summary("e1"), "tables", "total")],
    )


def _records() -> dict:
    """The item records production writes for the library; a fresh copy each call."""
    return copy.deepcopy(_built().records)


def _captures() -> dict[str, bytes]:
    def j(obj: Any) -> bytes:
        return json.dumps(obj).encode()

    sig = {
        "input": memory_helper.value_shape(["a", "b"]),
        "relations": [],
        "constraints": {"applies": {"stated": True, "held": True, "sha256": None}},
        "goal": "Reverse a list of words.",
    }
    return {
        f".memory/captures/{CAPTURE}/procedure.py": PROCEDURE.encode(),
        f".memory/captures/{CAPTURE}/record.json": j({"level": "unchecked", "status": "on-trial", "outcome": "none"}),
        f".memory/captures/{CAPTURE}/signature.json": j(sig),
        f".memory/captures/{FAILED}/procedure.py": b"def run(value):\n    raise ValueError('no')\n",
        f".memory/captures/{FAILED}/record.json": j({"level": "unchecked", "status": "on-trial", "outcome": "failed"}),
        f".memory/captures/{FAILED}/signature.json": j({**sig, "goal": "Reverse words, tried."}),
        f".memory/captures/{FAILED}/broke.md": f"- reading the words raised ValueError {IDS[1]}\n".encode(),
    }


def _tree(root: Path) -> Path:
    """The library the gate checked, written under *root*."""
    for rel, text in _built().files.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text)
    return root


def _copy(tmp_path: Path, librarian: bool = True, name: str = "copy") -> Path:
    """An actor's library copy as the export writes it, with the captures' snapshot."""
    root = _tree(tmp_path / name)
    captures = _captures()
    files = lx.generated_library_mode(
        root,
        records=_records(),
        librarian=librarian,
        summary_of=_summary,
        captures=captures,
    )
    write_files(root, {**files, **captures})
    return root


# --- the catalogue and the dispatch file ------------------------------------------------------------------------


@needs_box
def test_the_catalogue_has_one_card_per_function_in_module_order_then_notes_then_captures(tmp_path):
    root = _copy(tmp_path)
    text = (root / lx.CATALOGUE_FILE).read_text()
    # its applies is private to the procedure's module: no card of its own
    assert L.catalogue_ids(text) == [COUNT, TOTAL, READ_ROWS, NOTE, "capture:" + CAPTURE, "capture:" + FAILED]
    assert text.index("## Functions") < text.index("## Notes") < text.index("## Captured procedures")
    card = text.split(f"### {TOTAL}\n", 1)[1].split("\n### ", 1)[0]
    assert f"`{TOTAL}(value)`" in card
    assert "Sum every number of a table's rows." in card and "It takes the rows" not in card  # first paragraph
    assert "- Asked: Add up a table of numbers from <id> and report the total (<id>)." in card
    assert "Approach: - Read the rows" in card and "run <id>" in card
    assert f"Evidence: level reproduces-its-case; status on-trial; reader {READ_ROWS}; applies: yes" in card
    sig = _records()[TOTAL]["signature"]
    assert f"Input: {signature_text(sig)}" in card and "Input: {" in card
    assert "Record: " in card and f"Notes: {NOTE}" in card
    failed = text.split(f"### capture:{FAILED}\n", 1)[1]
    assert "tried and failed: never offered to run" in failed and "Broke: reading the words raised ValueError <id>" in failed
    assert "level unchecked" in text.split(f"### capture:{CAPTURE}\n", 1)[1]


@needs_box
def test_the_catalogue_holds_no_identifier_answer_or_request_and_is_byte_stable(tmp_path):
    root = _copy(tmp_path)
    text = (root / lx.CATALOGUE_FILE).read_bytes()
    for value in (*IDS, RECORDED_TOTAL, RECORDED_REQUEST, REQUEST, PLANTED_VALUE):
        assert value.encode() not in text, value
    again = lx.librarian_files(root, records=_records(), summary_of=_summary, captures=_captures())
    assert again[lx.CATALOGUE_FILE] == text
    assert again[lx.DISPATCH_FILE] == (root / lx.DISPATCH_FILE).read_bytes()


@needs_box
def test_the_dispatch_file_lists_each_procedure_with_a_reader_or_an_applies(tmp_path):
    root = _copy(tmp_path)
    data = json.loads((root / lx.DISPATCH_FILE).read_text())
    rows = data["procedures"]
    assert [r["id"] for r in rows] == [TOTAL, "capture:" + CAPTURE]  # the failed capture is never offered
    sig = _records()[TOTAL]["signature"]
    assert rows[0] == {"id": TOTAL, "reader": READ_ROWS, "applies": True, "input": signature_text(sig)}
    assert rows[0]["input"].startswith("{")
    assert rows[1]["reader"] == "capture:" + CAPTURE and rows[1]["applies"] is True
    assert data["applies_of"] == {TOTAL: WIDE}  # where a landed procedure's applies lives
    assert all(set(r) == {"id", "reader", "applies", "input"} for r in rows)


@needs_box
def test_off_the_export_is_byte_for_byte_as_before(tmp_path):
    root = _tree(tmp_path / "lib")
    off = lx.generated_library_mode(root, records=_records())
    assert off == lx.generated_library_mode(root, records=_records(), librarian=False, summary_of=_summary)
    assert lx.CATALOGUE_FILE not in off and lx.DISPATCH_FILE not in off
    assert off[RESERVED_INIT] == (Path(lx.__file__).parent / "library_helper.py").read_bytes()
    on = lx.generated_library_mode(root, records=_records(), librarian=True)
    assert on[RESERVED_INIT].startswith(off[RESERVED_INIT]) and b"def ask(" in on[RESERVED_INIT]
    # the stub gains memory.ask with the package, and nothing else
    assert b"def ask(" in on[RESERVED_STUB] and b"def ask(" not in off[RESERVED_STUB]
    assert on[RESERVED_STUB] == lx.stub_source(librarian=True)
    assert {
        k: v for k, v in on.items() if k not in (RESERVED_INIT, RESERVED_STUB, lx.CATALOGUE_FILE, lx.DISPATCH_FILE)
    } == {k: v for k, v in off.items() if k not in (RESERVED_INIT, RESERVED_STUB)}


# --- memory.ask in the package, run as a cell runs it ------------------------------------------------------------

FAKE_ANSWER = json.dumps({"fits": [{"id": TOTAL, "call": f"memory.run({TOTAL!r}, text)", "why": "its reader parsed it"}]})
FAKE_TOOLS = (
    "import json, sys, types\n"
    "tools = types.ModuleType('tools')\n"
    "sent = []\n"
    "requests = []\n"
    "def ask_library(question, facts=None):\n"
    "    sent.append([question, facts])\n"
    f"    return {FAKE_ANSWER!r}\n"
    "def _call(tool, args):\n"
    "    requests.append([tool, args])\n"
    f"    return json.dumps({{'text': {REQUEST!r}, 'data': None}})\n"
    "tools.ask_library = ask_library\n"
    "tools._call = _call\n"
    "sys.modules['tools'] = tools\n"
)


@needs_box
def test_ask_computes_the_facts_in_its_own_process_and_sends_outcomes_only(tmp_path):
    root = _copy(tmp_path)
    code = (
        FAKE_TOOLS
        + "import memory\n"
        "seen = []\n"
        "sys.addaudithook(lambda ev, a: seen.append([ev, list(a)]) if ev.startswith('memory.') else None)\n"
        "answer = memory.ask()\n"
        "fits = [[i.id, i.kind, i.reason, type(i).__name__] for i in answer]\n"
        "print(json.dumps({'answer': fits, 'list': isinstance(answer, list), 'sent': sent, 'seen': seen, "
        "'requests': requests}, default=list))\n"
    )
    out = json.loads(_py(root, code))
    # the current request came from the host, through the channel client, and was asked about itself
    assert out["requests"] == [["current_request", {}]]
    # a list of items, each with the librarian's reason (the stand-in host's answer)
    assert out["list"] is True and out["answer"] == [[TOTAL, "function", "its reader parsed it", "Item"]]
    (question, facts), = out["sent"]
    assert question == REQUEST
    assert facts == [
        {"id": TOTAL, "reader": READ_ROWS, "read": "parsed", "applies": "held"},
        {"id": "capture:" + CAPTURE, "reader": "capture:" + CAPTURE, "read": "parsed", "applies": "did-not-hold"},
    ]
    # the readers' values never left their process: only the outcome labels did
    assert PLANTED_VALUE not in json.dumps(out)
    helper = [(a[0], a[2]) for ev, a in out["seen"] if ev == "memory.helper_call"]
    assert helper == [("memory:ask", "ok")]
    assert not [a for ev, a in out["seen"] if ev == "memory.call"]  # asking is never a use
    (result,) = [a for ev, a in out["seen"] if ev == "memory.ask_result"]
    assert result[0] == [TOTAL] and result[1] is False
    assert [r[0] for r in result[2]] == [TOTAL, "capture:" + CAPTURE]


@needs_box
def test_ask_without_the_tool_says_the_librarian_cannot_be_reached(tmp_path):
    root = _copy(tmp_path)
    code = (
        "import json, memory\n"
        "try:\n"
        "    memory.ask('x')\n"
        "except LookupError as exc:\n"
        "    print(json.dumps(str(exc)))\n"
    )
    assert "cannot be reached" in json.loads(_py(root, code))
    off = _copy(tmp_path, librarian=False, name="off")
    assert json.loads(_py(off, "import json, memory\nprint(json.dumps(hasattr(memory, 'ask')))")) is False


NEAREST_TOOLS = (
    "import json, sys, types\n"
    "tools = types.ModuleType('tools')\n"
    "class Refused(Exception):\n"
    "    pass\n"
    "tools.Refused = Refused\n"
    "def ask_library(question, facts=None):\n"
    "    return json.dumps({'fits': [], 'nearest': [{'id': %r, 'differs': 'it sums a table; this counts rows'}]})\n"
    "def refuse(question, facts=None):\n"
    "    raise Refused('refused: the librarian is not available here (no session); nothing was asked')\n"
    "def garbled(question, facts=None):\n"
    "    return 'not json'\n"
    "def _call(tool, args):\n"
    "    return json.dumps({'text': 'How many rows does the table have?', 'data': None})\n"
    "tools.ask_library = ask_library\n"
    "tools._call = _call\n"
    "sys.modules['tools'] = tools\n"
) % NOTE


@needs_box
def test_ask_gives_the_nearest_items_when_nothing_applies_and_errors_that_teach(tmp_path):
    root = _copy(tmp_path)
    code = (
        NEAREST_TOOLS
        + "import memory\n"
        "def err(fn, *a):\n"
        "    try:\n"
        "        fn(*a)\n"
        "    except Exception as exc:\n"
        "        return [type(exc).__name__, str(exc)]\n"
        "answer = memory.ask('How many rows does the table have?')\n"
        "out = [list(answer), [[i.id, i.kind, i.reason] for i in answer.nearest], str(answer)]\n"
        "out += [err(memory.ask, 3), err(memory.ask, '')]\n"
        "tools.ask_library = refuse\n"
        "out.append(err(memory.ask, 'q'))\n"
        "tools.ask_library = garbled\n"
        "out.append(err(memory.ask, 'q'))\n"
        "print(json.dumps(out))\n"
    )
    fits, nearest, shown, number, empty, refused, garbled = json.loads(_py(root, code))
    assert fits == [] and nearest == [[NOTE, "note", "it sums a table; this counts rows"]]
    assert shown == f"Nothing in the library applies.\n- nearest {NOTE} (note): it sums a table; this counts rows\n"
    expected = "Expected question: str non-empty | None, e.g. memory.ask()."
    assert number == ["ArgumentError", f"memory.ask(3): question is 3 (int).\n{expected}"]
    assert empty == ["ArgumentError", f"memory.ask(''): question is '' (str).\n{expected}"]
    assert refused == [
        "NotFound",
        "memory.ask('q'): the librarian gave no answer (refused: the librarian is not available here (no session); "
        "nothing was asked). Ask again later, or browse with memory.items() and open an item with memory.get(id).",
    ]
    assert garbled == [
        "NotFound",
        "memory.ask('q'): the librarian's answer could not be read. Ask again, or browse with memory.items().",
    ]


# --- the answer, the facts and the prompt ------------------------------------------------------------------------

IDS_IN_ORDER = [COUNT, TOTAL, READ_ROWS, "capture:" + CAPTURE, "capture:" + FAILED]


@pytest.mark.parametrize(
    "answer, why",
    [
        ("text", "must be an object"),
        ({"fits": "x"}, "`fits` must be a list"),
        ({"fits": [{"id": TOTAL, "call": "c"}]}, "non-empty texts"),
        ({"fits": [{"id": "memory.nope:x", "call": "c", "why": "w"}]}, "not an id the catalogue holds"),
        ({"fits": [{"id": "capture:" + FAILED, "call": "c", "why": "w"}]}, "never offered to run"),
        ({"fits": []}, "`nearest` must be a list"),
        ({"fits": [{"id": TOTAL, "call": "c", "why": "w"}], "nearest": []}, "only when nothing fits"),
        ({"fits": [], "nearest": [], "score": 1}, "unknown keys"),
    ],
)
def test_a_malformed_answer_or_an_unknown_id_is_refused_with_the_reason(answer, why):
    got, reason = L.check_answer(answer, IDS_IN_ORDER, {"capture:" + FAILED})
    assert got is None and why in reason


def test_an_answer_passes_as_the_librarian_gave_it_with_nothing_added_or_reordered():
    fits = [{"id": TOTAL, "call": "a", "why": "b"}, {"id": COUNT, "call": "c", "why": "d"}]
    got, _ = L.check_answer({"fits": fits}, IDS_IN_ORDER)
    assert got == {"fits": fits}
    got, _ = L.check_answer({"fits": [], "nearest": [{"id": "capture:" + FAILED, "differs": "it failed"}]}, IDS_IN_ORDER)
    assert got == {"fits": [], "nearest": [{"id": "capture:" + FAILED, "differs": "it failed"}]}


@needs_box
def test_only_dispatch_ids_with_their_own_reader_and_known_labels_reach_the_librarian(tmp_path):
    dispatch = L.read_dispatch(_copy(tmp_path))
    facts = [
        {"id": TOTAL, "reader": READ_ROWS, "read": "parsed", "applies": "held"},
        {"id": TOTAL, "reader": READ_ROWS, "read": "parsed", "applies": "held"},  # twice
        {"id": "memory.planted:path", "reader": READ_ROWS, "read": "parsed", "applies": "held"},
        {"id": "capture:" + CAPTURE, "reader": "/home/planted", "read": "parsed", "applies": "held"},
        {"id": "capture:" + CAPTURE, "reader": "capture:" + CAPTURE, "read": "planted-label", "applies": "held"},
        "not a fact",
    ]
    kept, dropped = L.checked_facts(facts, dispatch)
    assert kept == [{"id": TOTAL, "reader": READ_ROWS, "read": "parsed", "applies": "held"}] and dropped == 5
    text = L.question_message("q", kept, dropped)
    assert f"`{TOTAL}`: reader `{READ_ROWS}` parsed the text; its `applies` held on what the reader gave." in text
    assert "planted" not in text and "(5 fact(s) could not be read" in text


def test_the_brief_is_complete_general_and_says_nothing_of_ids_answers_or_examples():
    brief = L.BRIEF
    for part in ("## Your role", "## What you read", "## How you answer", "`memory.get`", "tools.finish(answer)",
                 "Read every card", "Name every item that applies", "nothing fits", "evidence only"):
        assert part in brief, part
    assert lp.benchmark_words(brief) == [] and lp.example_checks(brief) == []
    # the library's calls are named without arguments: their help holds those
    assert not re.search(r"memory\.[a-z_]+\(", brief) and "memory.show" not in brief and "memory.run" not in brief
    lowered = brief.lower()
    for word in ("task id", "identifier", "example", "store the answer", "expected"):
        assert word not in lowered, word
    assert R.allow_list("librarian", frozenset({"helper", "store", "edit", "pool", "similar"})) == ("finish",)
    opening = L.opening()
    assert opening.startswith("## How you work here") and "tools.finish(answer)" in opening
    assert "tools.sh" not in opening and "tools.install" not in opening and "Messages:" not in opening
    assert "/memory (read-only)" in opening and "no network" in opening
    system = L.system_prompt("CATALOGUE")
    assert system.index(opening) == 0 and system.index(brief) < system.index("CATALOGUE")


def test_the_first_user_message_line_counts_the_copy_and_is_absent_for_an_empty_library(tmp_path, monkeypatch):
    """The actor's first user message of a request opens with how many functions and notes its library copy holds
    (as ``memory.items()`` lists them: the retired are left out), with the library bound in its cells and the
    librarian on; never for an empty library, a teammate, or with the librarian off."""
    from types import SimpleNamespace

    from unify.memory.integration import hooks
    from unify.memory.integration import request as request_mod

    def copy_with(name: str, items: dict) -> Path:
        root = tmp_path / name
        (root / "memory").mkdir(parents=True)
        (root / "memory" / "__init__.py").write_text("")
        (root / ".memory").mkdir()
        (root / ".memory" / "items.json").write_text(json.dumps({"items": items}))
        return root

    def line_for(copy: Path, **kw) -> str | None:
        monkeypatch.setattr(request_mod, "_CURRENT", SimpleNamespace(paths=SimpleNamespace(checkout=copy),
                                                                      library_mode=True))
        return hooks.library_line(**kw)

    monkeypatch.setattr(SETTINGS, "UNIFY_MEMORY", "on")
    monkeypatch.setattr(SETTINGS, "UNIFY_MEMORY_LIBRARIAN", "on")
    tail = "; memory.ask() shows the ones that apply, memory.items() lists them all."
    # the world's catalogue: COUNT, TOTAL and READ_ROWS, and NOTE
    assert line_for(_copy(tmp_path)) == "Your memory library holds 3 functions and 1 note" + tail
    one = copy_with("one", {
        "memory.a.b:c": {"kind": "function"},
        "memory.a.b:d": {"kind": "function", "status": "retired"},
        "notes/a/x.md": {"kind": "note"},
        "notes/a/y.md": {"kind": "note", "status": "deprecated"},
    })
    assert line_for(one) == "Your memory library holds 1 function and 1 note" + tail
    assert line_for(one, actor=False) is None
    assert line_for(copy_with("empty", {})) is None
    assert line_for(copy_with("retired", {"memory.a.b:c": {"kind": "function", "status": "retired"}})) is None
    monkeypatch.setattr(SETTINGS, "UNIFY_MEMORY_LIBRARIAN", "off")
    assert line_for(one) is None
    monkeypatch.setattr(SETTINGS, "UNIFY_MEMORY_LIBRARIAN", "on")
    monkeypatch.setattr(SETTINGS, "UNIFY_MEMORY", "off")
    assert line_for(one) is None


@needs_box
def test_the_pointer_is_constant_across_libraries_and_replaces_the_index(tmp_path, monkeypatch):
    from unify.memory.integration.prompt import render_memory_library_mode

    monkeypatch.setattr(SETTINGS, "UNIFY_TOOLS_IN_CODE", "on")
    a = _copy(tmp_path, name="a")
    b = _tree(tmp_path / "b")  # another library: one module fewer, no records
    (b / "memory/tables/count.py").unlink()
    write_files(b, lx.generated_library_mode(b, librarian=True))
    text_a, shown = render_memory_library_mode(a, librarian=True)
    text_b, _ = render_memory_library_mode(b, librarian=True)
    assert text_a == text_b and lp.LIBRARIAN_POINTER in text_a
    assert "index at the end of this prompt" not in text_a and TOTAL not in text_a and "## memory." not in text_a
    assert shown["renderer"] == "librarian_pointer"
    assert lp.benchmark_words(lp.GUIDE_LIBRARIAN) == [] and lp.example_checks(lp.GUIDE_LIBRARIAN) == []
    off, _ = render_memory_library_mode(a, librarian=False)
    monkeypatch.setattr(SETTINGS, "UNIFY_MEMORY_LIBRARIAN", "off")
    assert render_memory_library_mode(a)[0] == off and lp.LIBRARIAN_POINTER not in off


# --- the switch ------------------------------------------------------------------------------------------------


def test_the_switch_is_off_by_default_and_on_needs_the_tools_in_code():
    assert ProductionSettings().UNIFY_MEMORY_LIBRARIAN == "off"
    assert ProductionSettings(UNIFY_MEMORY_LIBRARIAN=" On ", UNIFY_TOOLS_IN_CODE="on").UNIFY_MEMORY_LIBRARIAN == "on"
    with pytest.raises(ValueError, match="requires UNIFY_TOOLS_IN_CODE=on"):
        ProductionSettings(UNIFY_MEMORY_LIBRARIAN="on", UNIFY_TOOLS_IN_CODE="off")
    with pytest.raises(ValueError, match="must be 'on' or 'off'"):
        ProductionSettings(UNIFY_MEMORY_LIBRARIAN="sometimes")


def test_off_the_actor_has_no_ask_tool_and_its_stub_is_unchanged():
    from unify.actor import tool_handlers as H

    off = R.allow_list("actor", H.actor_flags("actor", True))
    on = R.allow_list("actor", H.actor_flags("actor", True, librarian=True))
    assert "ask_library" not in off and set(on) - set(off) == {"ask_library"}
    assert "ask_library" not in R.allow_list("teammate", H.actor_flags("teammate", False, librarian=True))
    assert "ask_library" not in R.render_stub(off, "actor")
    assert R.preamble("actor", {"flags": H.actor_flags("actor", True)}) == R.preamble(
        "actor", {"flags": H.actor_flags("actor", True, librarian=False)},
    )
    assert R.handler_for("actor", "ask_library") == ("unify.actor.tool_handlers:ask_library", False)
    assert R.handler_for("librarian", "finish") == ("unify.memory.librarian:finish", False)


# --- the evidence ------------------------------------------------------------------------------------------------


def _cell(audit: CellAudit, *events) -> dict:
    audit.begin()
    for ev, args in events:
        audit(ev, args)
    return audit.drain()


def test_ask_is_a_helper_call_and_its_rows_and_listed_by_ask_reach_the_use_record():
    audit = CellAudit([])
    cap = "capture:" + CAPTURE
    first = _cell(
        audit,
        ("memory.helper_call", ("memory:ask", "ab12", "ok")),
        ("memory.ask_result", ((TOTAL, cap), False, ((TOTAL, "parsed", "held"), (cap, "parsed", "did-not-hold")))),
        ("memory.call", (TOTAL, "ab12", "ok")),
        ("memory.ask_result", (("not an id",), False, ())),
    )
    assert first["calls"]["malformed"] == 1
    second = _cell(
        audit,
        ("memory.helper_call", ("memory:ask", "ab12", "ok")),
        ("memory.ask_result", ((), True, ())),
        ("memory.call", (cap, "ab12", "ok")),
    )
    acc = runtime_calls_empty()
    add_runtime_calls(acc, first)
    add_runtime_calls(acc, second)
    assert library_helper_calls(acc, librarian=True)["ask"] == {"ok": 2, "refused": 0, "error": 0}
    assert library_helper_calls(acc)["ask"] == {"ok": 2, "refused": 0, "error": 0}  # a seen ask is always counted
    assert "ask" not in library_helper_calls(runtime_calls_empty())  # no librarian and no ask: no ask row
    assert TOTAL in acc["counts"] and "memory:ask" not in acc["counts"]  # asking is never a use
    fields = use.ask_fields(acc)
    assert fields["ask_calls"] == 2
    one, two = fields["ask_rows"]
    assert one["listed"] == [TOTAL, cap] and one["nothing_fits"] is False
    assert one["facts"] == [{"id": TOTAL, "read": "parsed", "applies": "held"},
                            {"id": cap, "read": "parsed", "applies": "did-not-hold"}]
    assert one["called_after"] == sorted([TOTAL, cap]) and two == {**two, "nothing_fits": True, "listed": []}
    assert fields["listed_by_ask"] == sorted([TOTAL, cap])
    # a request that never asked keeps today's runtime evidence
    plain = runtime_calls_empty()
    add_runtime_calls(plain, _cell(CellAudit([]), ("memory.call", (TOTAL, "ab12", "ok"))))
    assert "asks" not in plain


def test_the_librarians_calls_count_as_the_actors_and_its_totals_keep_unknown_usd_unknown():
    from unify.memory.integration import cost

    assert cost._purpose(L.ORIGIN) == "actor"
    u = L.Usage()
    u.add({"input": 100, "output": 5, "cache_read": 80}, "0.0012")
    u.add({"input": None}, None)
    got = L.usage_fields([type("S", (), {"usage": u, "questions": 2})()])
    assert got["calls"] == 2 and got["input_tokens"] == 100 and got["cache_read_tokens"] == 80
    assert got["usd"] == "unknown" and got["usd_priced"] == "0.0012" and got["unpriced_calls"] == 1
    assert got["questions"] == 2 and got["unreported_calls"] == 1


# --- the librarian end to end, in real boxes, with a scripted model ----------------------------------------------


class HeaderModel(ScriptedModel):
    """A scripted model that also keeps each request's ``extra_headers``."""

    def __init__(self, **scripts: Any) -> None:
        super().__init__(**scripts)
        self.headers: list[dict] = []

    async def __call__(self, *, shared_session=None, client=None, **kw: Any) -> Any:
        self.headers.append(dict(kw.get("extra_headers") or {}))
        return await super().__call__(shared_session=shared_session, client=client, **kw)


def code(text: str):
    return ("execute_code", {"code": text})


LIBRARIAN_FIRST = (
    "import memory, os, socket\n"
    f"print(memory.show({TOTAL!r}))\n"
    "try:\n"
    "    open('/memory/planted.txt', 'w')\n"
    "except OSError as exc:\n"
    "    print('WRITE REFUSED', type(exc).__name__)\n"
    "try:\n"
    "    socket.create_connection(('1.1.1.1', 80), timeout=3)\n"
    "except OSError as exc:\n"
    "    print('NETWORK REFUSED', type(exc).__name__)\n"
    "print('ENV', sorted(os.environ))\n"
    "print('WORKSPACE', os.path.exists('/workspace'))\n"
    "tools.finish({'fits': [{'id': 'memory.nope:x', 'call': 'x', 'why': 'y'}]})\n"
)
LIBRARIAN_SECOND = "tools.finish(%r)\n" % {
    "fits": [{"id": TOTAL, "call": f"memory.run({TOTAL!r}, text)", "why": "reader read_rows parsed the text"}],
}
LIBRARIAN_THIRD = (
    "tools.finish({'fits': [], 'nearest': [{'id': 'capture:%s', 'differs': 'it reverses words, not tables'}]})\n"
) % FAILED

ACTOR_FIRST = (
    "import json, memory\n"
    "first = memory.ask()\n"
    "print('FIRST', json.dumps([[i.id, i.reason] for i in first]))\n"
)
ACTOR_SECOND = (
    "second = memory.ask('and a,b,c?')\n"
    "print('SECOND', json.dumps([[i.id for i in second], [[i.id, i.reason] for i in second.nearest]]))\n"
)


@needs_box
@pytest.mark.asyncio
@pytest.mark.timeout(300)
async def test_ask_end_to_end_facts_in_the_box_one_session_per_request_and_the_call_kind(tmp_path, monkeypatch):
    from unify.actor import tool_handlers as H
    from unify.common.tool_channel import HandlerThreads
    from unify.memory import tool_handlers as th
    from unify.memory.writer_pass import unillm_turn

    monkeypatch.setattr(SETTINGS, "UNIFY_REQUEST_METADATA_HEADERS", "on")
    monkeypatch.setenv("PLANTED_LIBRARIAN_SECRET", "planted-librarian-secret-value")
    copy = _copy(tmp_path)
    catalogue = (copy / lx.CATALOGUE_FILE).read_text()
    events: list[dict] = []
    opened: list[L.LibrarianSession] = []

    async def open_librarian() -> L.LibrarianSession:
        session = L.LibrarianSession(
            copy=copy,
            catalogue=catalogue,
            turn=unillm_turn("openai/gpt-5.6-sol@openrouter", "low", origin=L.ORIGIN),
            on_event=events.append,
            request_id="r1",
        )
        opened.append(session)
        return session

    state = H.ActorRequestState(open_librarian=open_librarian, request_text=REQUEST)
    actor_events: list[dict] = []
    # the actor's cells: the copy read-only at /library, its memory package importable, ask_library on its
    # allow-list and current_request on its channel
    actor = th.InterpreterSession(
        agent="actor",
        name="root",
        role="actor",
        pass_id="r1",
        state=state,
        allow=("ask_library", "current_request"),
        binds={"ro": {copy: "/library"}, "rw": {}},
        env={},
        roots={"/library": copy},
        deadline=time.monotonic() + 3600,
        cell_limit_s=200.0,
        on_event=actor_events.append,
        threads=HandlerThreads(),
        profile=th.memory_profile(
            cwd="/library", sys_path=("/library",), audit_roots=("/library",), open_areas=("/library",),
        ),
    )
    model = HeaderModel(actor=[reply(calls=[code(LIBRARIAN_FIRST)]), reply(calls=[code(LIBRARIAN_SECOND)]),
                               reply(calls=[code(LIBRARIAN_THIRD)])])
    try:
        with scripted(model):
            first = await actor.cell(ACTOR_FIRST, 0)
            second = await actor.cell(ACTOR_SECOND, 1)
    finally:
        await actor.close()
        if state.librarian is not None:
            await state.librarian.close()
    shown = first.output + (first.error or "")
    assert "FIRST" in shown, shown
    got = json.loads(shown.split("FIRST ", 1)[1].splitlines()[0])
    assert got == [[TOTAL, "reader read_rows parsed the text"]]  # an Item, with the librarian's reason
    later = json.loads((second.output or "").split("SECOND ", 1)[1].splitlines()[0])
    assert later == [[], [["capture:" + FAILED, "it reverses words, not tables"]]]

    # one session for the request: opened at the first question, kept for the follow-up, ended with the request
    (session,) = opened
    assert session.questions == 2 and session.closed and not session.box_alive()
    assert len(model.calls) == 3
    requests = [json.dumps(c.request) for c in model.calls]
    first_question = model.calls[0].messages[-1]["content"]
    assert f"`{TOTAL}`: reader `{READ_ROWS}` parsed the text; its `applies` held" in first_question
    assert f"`capture:{CAPTURE}`: reader `capture:{CAPTURE}` parsed the text; its `applies` did not hold" in first_question
    # the readers' values stayed in the actor's box; no host path reaches the librarian
    for sent in requests:
        assert PLANTED_VALUE not in sent and str(tmp_path) not in sent
    system = model.calls[0].messages[0]["content"]
    assert system.startswith(L.opening()) and system.endswith(catalogue)
    assert all(c.messages[0]["content"] == system for c in model.calls)  # the cached prefix
    # the librarian's box: the library read-only, no network, an empty environment, no workspace
    tool_out = "\n".join(m["content"] for m in model.calls[1].messages if m.get("role") == "tool")
    assert "def total(value):" in tool_out
    assert "WRITE REFUSED" in tool_out and "NETWORK REFUSED" in tool_out and "WORKSPACE False" in tool_out
    env_line = [line for line in tool_out.splitlines() if line.startswith("ENV ")][0]
    assert "PLANTED_LIBRARIAN_SECRET" not in env_line and "OPENROUTER" not in env_line
    assert "planted-librarian-secret-value" not in json.dumps(requests)
    assert "not an id the catalogue holds" in tool_out  # the malformed answer went back as a refusal
    # each librarian call carries the call kind
    assert [h.get("X-Unify-Call-Kind") for h in model.headers] == ["librarian"] * 3
    assert session.usage.calls == 3 and session.usage.input_tokens == 300
    asks = [r for r in actor_events if r.get("tool") == "ask_library"]
    assert [r["status"] for r in asks] == ["ok", "ok"]
    assert {r.get("agent") for r in events if r.get("tool") == "finish"} == {"librarian"}


@pytest.mark.asyncio
async def test_a_librarian_that_never_answers_is_a_refusal_not_a_crash(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from unify.actor import tool_handlers as H

    class Silent:
        questions = 0

        async def ask(self, question, facts, request=None):
            raise L.NoAnswer("no progress")

    state = H.ActorRequestState(librarian=Silent())
    ctx = SimpleNamespace(state=state)
    got = await H.ask_library(ctx, {"question": "q"})
    assert got.status == "refused" and got.form == "no_answer"
    got = await H.ask_library(SimpleNamespace(state=H.ActorRequestState()), {"question": "q"})
    assert got.form == "unavailable"
    got = await H.ask_library(ctx, {"question": " "})
    assert got.form == "no_question"


@pytest.mark.asyncio
async def test_the_request_closes_its_librarian(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from unify.actor import in_code_tools

    monkeypatch.setattr(SETTINGS, "UNIFY_TOOLS_IN_CODE", "on")
    monkeypatch.setattr(SETTINGS, "UNIFY_MEMORY_LIBRARIAN", "on")

    class Session:
        closed = False

        async def close(self):
            self.closed = True

        def box_alive(self):
            return False

    async def opener():
        return Session()

    tools = in_code_tools.InCodeTools(
        agent="actor",
        name="root",
        record=SimpleNamespace(),
        workspace_root=str(tmp_path),
        shadow={},
        cell_limit_s=None,
        events_folder=tmp_path / "events",
        librarian_opener=opener,
    )
    assert "ask_library" in tools.allow and "def ask_library(" in tools.stub
    tools.state.librarian = await tools.state.open_librarian()
    session = tools.state.librarian
    await tools.close(completed=True)
    assert session.closed and tools.state.librarian is None
    monkeypatch.setattr(SETTINGS, "UNIFY_MEMORY_LIBRARIAN", "off")
    off = in_code_tools.InCodeTools(
        agent="actor",
        name="root",
        record=SimpleNamespace(),
        workspace_root=str(tmp_path),
        shadow={},
        cell_limit_s=None,
        events_folder=tmp_path / "events-off",
    )
    try:
        assert "ask_library" not in off.allow and "ask_library" not in off.stub
        assert off.state.open_librarian is None
    finally:
        await off.close(completed=True)
