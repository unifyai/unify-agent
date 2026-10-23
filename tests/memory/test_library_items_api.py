"""The library's contract: ``memory.get``, ``memory.items`` and the ``Item`` type, run as a cell runs them.

Every way of naming one item resolves in one place; several matches list each exact id, and none gives the nearest ids
and the topic to browse. Each call checks its argument types, and a malformed generated file never escapes as a raw
exception. Calling a function item is a tracked call; calling a note raises. The export generates the package's
typed stub.
The error texts are the approved memory API's own examples (docs/design/memory-api-items-20261013.md in the research
repository).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from unify.memory import layout
from unify.memory import library_export as lx
from unify.memory import library_helper
from unify.memory.catalogue import write_files
from unify.memory.layout import RESERVED_STUB

CELLS = '''"""Grid cells."""


def fill(g, v):
    """Fill every cell of a grid with one value.

    Use when: a request asks for a grid of one colour.
    """
    return [[v for _ in row] for row in g]


def rows(g):
    """How many rows a grid has."""
    return len(g)
'''
STENCIL = '''"""Stencils."""


class MemoryInputError(ValueError):
    pass


def paint(g):
    """Paint a stencil onto a grid: each row reversed.

    Notes: notes/grid/stencil.md
    """
    if not isinstance(g, list):
        raise MemoryInputError("expected a grid: a list of rows")
    return [row[::-1] for row in g]
'''
COLOUR = '''"""Colour words."""


def paint(word):
    """Name the colour a word paints."""
    return word.upper()
'''
LIB = {
    "memory/grid/__init__.py": '"""Grid helpers."""\n',
    "memory/grid/cells.py": CELLS,
    "memory/grid/stencil.py": STENCIL,
    "memory/text/__init__.py": '"""Text helpers."""\n',
    "memory/text/colour.py": COLOUR,
    "notes/grid/stencil.md": (
        "---\ntitle: Stencils\ndescription: How stencils are drawn.\nuses: [memory.grid.stencil:paint]\n---\n"
        "A stencil is drawn row by row.\n"
    ),
    "notes/grid/paint_stencil.md": (
        "---\ntitle: Painting stencils\ndescription: How a stencil is painted.\n---\nPaint the stencil last.\n"
    ),
    "notes/text/colours.md": "---\ntitle: Colours\ndescription: Colour names.\n---\nColours are words.\n",
}
FILL = "memory.grid.cells:fill"
ROWS = "memory.grid.cells:rows"
PAINT = "memory.grid.stencil:paint"
WORD_PAINT = "memory.text.colour:paint"
NOTE = "notes/grid/stencil.md"
#: A cell's prelude: the package, and ``err(fn, ...)``, the type, text and bases of what a call raised (None if none).
PRELUDE = '''
import json, sys
import memory


def err(fn, *a, **k):
    try:
        fn(*a, **k)
    except Exception as exc:
        return [type(exc).__name__, str(exc), [c.__name__ for c in type(exc).__mro__]]
    return None
'''


def _library(tmp_path: Path, extra: dict | None = None, **kw) -> Path:
    """The fixture library (:data:`LIB`, with *extra* files) and its generated files, as an export writes them."""
    root = tmp_path / "lib"
    for rel, text in {**LIB, **(extra or {})}.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text)
    write_files(root, lx.generated_library_mode(root, **kw))
    return root


def _cell(root: Path, code: str) -> tuple[str, str]:
    """Run *code* after :data:`PRELUDE` in a fresh isolated interpreter with *root* first on its path (as a cell
    imports the package): its output and its error output."""
    out = subprocess.run(
        [sys.executable, "-I", "-c", f"import sys\nsys.path.insert(0, {str(root)!r})\n{PRELUDE}\n{code}"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert out.returncode == 0, out.stderr
    return out.stdout, out.stderr


def _json(root: Path, code: str) -> object:
    return json.loads(_cell(root, code)[0])


# --- resolution: every way of naming one item ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "ref, wanted",
    [
        (FILL, FILL),  # the full id
        ("fill", FILL),  # a function's plain name
        ("cells:fill", FILL),  # module:name
        ("grid.cells:fill", FILL),
        ("memory.grid.cells.fill", FILL),  # the dotted form of an id
        ("grid.stencil", PAINT),  # a module with one function
        (NOTE, NOTE),
        ("stencil", NOTE),  # a note's slug
        ("grid/stencil", NOTE),  # topic/slug
        ("grid/stencil.md", NOTE),
        ("notes/grid/stencil", NOTE),
        ("/memory/notes/grid/stencil.md", NOTE),  # a path under /memory
        ("/memory/memory/grid/stencil.py", PAINT),  # a module's path, one function in it
        ("  rows ", ROWS),
    ],
)
def test_each_way_of_naming_one_item_resolves_to_it(tmp_path, ref, wanted):
    root = _library(tmp_path)
    assert _json(root, f"print(json.dumps(memory.get({ref!r}).id))") == wanted


def test_the_copys_own_paths_the_function_itself_and_an_item_resolve_too(tmp_path):
    root = _library(tmp_path)
    got = _json(
        root,
        "from memory.grid.cells import rows\n"
        f"item = memory.get({FILL!r})\n"
        f"print(json.dumps([memory.get({str(root / NOTE)!r}).id, memory.get(rows).id, memory.get(item) == item]))\n",
    )
    assert got == [NOTE, ROWS, True]


def test_items_takes_a_topic_by_the_same_rules(tmp_path):
    root = _library(tmp_path)
    got = _json(
        root,
        "topics = ['grid', 'memory.grid', 'memory/grid', 'notes/grid', 'grid.cells', '/memory/notes/grid', None]\n"
        "out = {str(t): [i.id for i in memory.items(t)] for t in topics}\n"
        "out['note'] = [i.id for i in memory.items(kind='note')]\n"
        "out['grid functions'] = [i.id for i in memory.items('grid', kind='function')]\n"
        "print(json.dumps(out))\n",
    )
    functions = [FILL, ROWS, PAINT]
    assert got["grid"] == [*functions, "notes/grid/paint_stencil.md", NOTE]  # the package and the notes topic
    assert got["memory.grid"] == got["memory/grid"] == functions
    assert got["notes/grid"] == got["/memory/notes/grid"] == ["notes/grid/paint_stencil.md", NOTE]
    assert got["grid.cells"] == [FILL, ROWS]
    assert got["None"] == [*functions, WORD_PAINT, "notes/grid/paint_stencil.md", NOTE, "notes/text/colours.md"]
    assert got["note"] == ["notes/grid/paint_stencil.md", NOTE, "notes/text/colours.md"]
    assert got["grid functions"] == functions


def test_a_retired_item_is_left_out_of_items_and_still_opens(tmp_path):
    root = _library(tmp_path, status_of=lambda i: "retired" if i == ROWS else "on-trial")
    got = _json(
        root,
        f"print(json.dumps([[i.id for i in memory.items('grid.cells')], memory.get('rows').status,"
        f" str(memory.get('rows')).splitlines()[1]]))\n",
    )
    assert got == [[FILL], "retired", "Not in the index: its status is retired."]


def test_an_ambiguous_name_lists_each_exact_id_with_its_call(tmp_path):
    root = _library(tmp_path)
    kind, text, bases = _json(root, "print(json.dumps(err(memory.get, 'paint')))")
    assert kind == "Ambiguous" and "LookupError" in bases
    assert text == (
        "memory.get('paint'): 'paint' names 2 items.\n"
        f"Expected one item's full id, e.g. memory.get({PAINT!r}).\n"
        f"Nearest: memory.get({PAINT!r}), memory.get({WORD_PAINT!r})."
    )


def test_a_missing_name_gives_the_nearest_ids_and_the_topic_to_browse(tmp_path):
    """The approved design's first example, exactly."""
    root = _library(tmp_path)
    kind, text, bases = _json(root, "print(json.dumps(err(memory.get, 'stencil.md')))")
    assert kind == "NotFound" and "LookupError" in bases
    assert text == (
        "memory.get('stencil.md'): no item named 'stencil.md'.\n"
        "Expected an item id, a function name, or a note slug, e.g. memory.get('notes/grid/stencil.md').\n"
        "Nearest: notes/grid/stencil.md, notes/grid/paint_stencil.md. Browse with memory.items('notes/grid')."
    )


def test_a_topic_given_to_get_and_an_unknown_topic_name_the_right_call(tmp_path):
    root = _library(tmp_path)
    topic, unknown = _json(root, "print(json.dumps([err(memory.get, 'grid'), err(memory.items, 'gird')]))")
    assert topic[0] == "WrongCall" and topic[1] == (
        "memory.get('grid'): that is a topic, not an item. Browse it with memory.items('grid')."
    )
    assert unknown[0] == "NotFound" and unknown[1] == (
        "memory.items('gird'): no topic named 'gird'.\n"
        "Expected a package or a notes topic, e.g. memory.items('notes/grid').\n"
        "Nearest: notes/grid. Browse every topic with memory.items()."
    )


# --- the recorded failure, as regressions -------------------------------------------------------------------------


def test_the_recorded_failure_opens_the_note_by_get(tmp_path):
    """In the live runs a note's id was passed to a listing call in many visits; memory.get opens the note, and the
    wrong call to memory.items names memory.get (the design's second example, exactly)."""
    root = _library(tmp_path)
    by_get, wrong = _json(
        root,
        f"print(json.dumps([str(memory.get({NOTE!r})), err(memory.items, {NOTE!r})]))\n",
    )
    assert by_get.startswith("notes/grid/stencil.md  [on-trial, curated item, no record]  Stencils\n")
    assert "drawn row by row" in by_get
    assert wrong[0] == "WrongCall" and wrong[1] == (
        "memory.items('notes/grid/stencil.md'): that is an item, not a topic. "
        "Open it with memory.get('notes/grid/stencil.md')."
    )


# --- types at the call -------------------------------------------------------------------------------------------


def test_each_call_checks_its_argument_types_before_any_work(tmp_path):
    root = _library(tmp_path)
    got = _json(
        root,
        "item = memory.get('fill')\n"
        "print(json.dumps([err(memory.get, 3), err(memory.items, kind='notes'), err(memory.items, 3),"
        " err(memory.get), err(item.fits), err(memory.items().page, 0)]))\n",
    )
    for kind, _, bases in got:
        assert kind == "ArgumentError" and "TypeError" in bases
    by_number, by_kind, by_topic, missing, fits, page = (text for _, text, _ in got)
    assert by_number == (
        "memory.get(3): ref is 3 (int).\n"
        "Expected ref: str | Item | Callable, e.g. memory.get('notes/<topic>/<name>.md')."
    )
    assert by_kind == (
        "memory.items(kind='notes'): kind is 'notes' (str).\n"
        "Expected kind: 'function' | 'note' | None, e.g. memory.items('notes/<topic>', kind='note')."
    )
    assert by_topic.startswith("memory.items(3): topic is 3 (int).\nExpected topic: str | None, e.g. memory.items(")
    assert missing.startswith("memory.get(): missing a required argument: 'ref'.")
    assert fits.startswith("Item.fits(Item('memory.grid.cells:fill')): missing a required argument: 'inputs'.")
    assert "n is 0 (int).\nExpected n: int >= 1, e.g. print(memory.items().page(2))." in page


def test_no_raw_exception_escapes_a_malformed_items_file(tmp_path):
    root = _library(tmp_path)
    (root / library_helper.ITEMS_FILE).write_text("{not json")
    calls = "[err(memory.get, 'fill'), err(memory.items), err(memory.items, 'grid')]"
    got = _json(root, f"print(json.dumps({calls}))")
    for kind, text, bases in got:
        assert kind == "NotFound" and "LookupError" in bases, text
        assert "the library copy is unreadable: its .memory/items.json cannot be read (JSONDecodeError)" in text
        assert text.endswith("Library modules may still import directly; otherwise do the step yourself.")
        assert str(root) not in text  # never a host path
    assert got[0][1].startswith("memory.get('fill'): the library copy is unreadable")
    (root / library_helper.ITEMS_FILE).unlink()
    (gone,) = _json(root, "print(json.dumps([err(memory.get, 'fill')]))")
    assert gone[0] == "NotFound" and "(FileNotFoundError)" in gone[1]


def test_a_malformed_row_or_matcher_never_escapes_either(tmp_path):
    root = _library(tmp_path)
    data = json.loads((root / library_helper.ITEMS_FILE).read_text())
    data["items"][FILL] = {"kind": "function", "history": 3, "level": ["x"]}  # no path, signature or summary
    (root / library_helper.ITEMS_FILE).write_text(json.dumps(data))
    (root / library_helper.MATCHER_FILE).write_text("def broken(:\n")
    # a recorded procedure with a recorded input, so its check reaches the matcher
    capture = "capture:" + "a" * 16
    d = root / ".memory" / "captures" / ("a" * 16)
    d.mkdir(parents=True)
    (d / "procedure.py").write_text("def run(value):\n    return value\n")
    (d / "signature.json").write_text(json.dumps({"input": {}, "relations": [], "constraints": {}}))
    card, fitted = _json(
        root,
        f"print(json.dumps([str(memory.get('fill')), err(memory.get({capture!r}).fits, {{'a': 1}})]))\n",
    )
    assert card.startswith(f"{FILL}  [on-trial, curated item, no record]\n")
    assert "(its source cannot be read: IsADirectoryError)" in card
    kind, text, _ = fitted
    assert kind == "NotFound" and "its .memory/matcher.py cannot be read (SyntaxError)" in text


# --- Item, Items and Fit -----------------------------------------------------------------------------------------


def test_an_item_carries_every_member_and_prints_its_card(tmp_path):
    root = _library(tmp_path, history=({PAINT: ["abc123def456 add paint"]}, True))
    got = _json(
        root,
        "import dataclasses\n"
        f"paint, note = memory.get({PAINT!r}), memory.get({NOTE!r})\n"
        "fill = memory.get('fill')\n"
        "frozen = err(setattr, paint, 'id', 'x')\n"
        "fields = {f.name: getattr(paint, f.name) for f in dataclasses.fields(paint)}\n"
        "print(json.dumps([fields, repr(paint), str(paint), str(memory.get(paint)), fill.applies, note.kind,"
        " note.summary, note.links, note.evidence, frozen[0]]))\n",
    )
    fields, shown_repr, card, shown, applies, kind, summary, links, evidence, frozen = got
    assert fields["id"] == PAINT and fields["kind"] == "function" and fields["status"] == "on-trial"
    assert fields["summary"] == "Paint a stencil onto a grid: each row reversed."
    assert fields["evidence"] == "unchecked" and fields["example"] == "" and fields["reason"] == ""
    assert fields["text"].startswith("def paint(g):") and "Notes: notes/grid/stencil.md" in fields["text"]
    assert fields["links"] == [NOTE] and fields["history"] == ["abc123def456 add paint"]
    assert shown_repr == f"Item({PAINT!r})"
    assert card == shown and card.startswith(f"{PAINT}(g)  [on-trial, curated item, no record]\n")
    assert "Links to: notes/grid/stencil.md\nLinked from: notes/grid/stencil.md\n" in card
    assert "History (newest first):\n  abc123def456 add paint\n" in card
    assert applies == "a request asks for a grid of one colour."
    assert (kind, summary, links, evidence) == ("note", "How stencils are drawn.", [PAINT], "unchecked")
    assert frozen == "FrozenInstanceError"


def test_items_prints_a_compact_table_page_by_page(tmp_path):
    root = _library(tmp_path)
    table, last, page_three = _json(
        root,
        "memory.PAGE_ROWS = 3\n"
        "listing = memory.items()\n"
        "print(json.dumps([str(listing), str(listing.page(9)), repr(listing.page(3))]))\n",
    )
    lines = table.splitlines()
    assert lines[0] == (
        "memory.items(): 7 item(s), 4 function(s) and 3 note(s). Open one with memory.get(id); call a function "
        "item directly."
    )
    # what each item rests on in place of the evidence level (this library has no item records, and says so)
    assert lines[1].split() == ["id", "kind", "rests", "on", "summary"]
    assert lines[2].split()[:2] == [FILL, "function"] and lines[2].endswith(
        "  curated item, no record  Fill every cell of a grid with one value.",
    )
    assert len(lines) == 2 + 3 + 2
    assert lines[-2] == (
        "Topics: memory.grid (3), memory.text (1), notes/grid (2), notes/text (1); memory.items(topic) lists one."
    )
    assert lines[-1] == "Page 1 of 3. print(memory.items().page(2)) shows the next."
    assert last == page_three and last.splitlines()[-1] == "Page 3 of 3."
    assert last.splitlines()[2].split()[:2] == ["notes/text/colours.md", "note"]
    assert "  curated item, no record  " in last.splitlines()[2]


def test_items_is_a_frozen_sequence_of_items(tmp_path):
    root = _library(tmp_path)
    got = _json(
        root,
        "listing = memory.items('grid.cells')\n"
        "print(json.dumps([len(listing), listing[0].id, [i.id for i in listing[1:]], type(listing[1:]).__name__,"
        " err(listing.__getitem__, 'fill')[0], [i.id for i in listing]]))\n",
    )
    assert got == [2, FILL, [ROWS], "Items", "ArgumentError", [FILL, ROWS]]


# --- an item's own check and its call ----------------------------------------------------------------------------


EVENTS = """
seen = []
sys.addaudithook(lambda ev, a: seen.append([ev, [str(x) for x in a]]) if ev.startswith('memory.') else None)
"""


def _events(seen: list, *names: str) -> list:
    return [(ev, a[0], a[1] if ev == "memory.run_result" else a[2]) for ev, a in seen if ev in names]


def test_calling_a_function_item_is_tracked(tmp_path):
    root = _library(tmp_path)
    got = _json(
        root,
        EVENTS
        + "fill = memory.get('fill')\n"
        "start = len(seen)\n"
        "out = fill([[0, 0]], 7)\n"
        "by_call = seen[start:]\n"
        "print(json.dumps([out, by_call]))\n",
    )
    out, by_call = got
    assert out == [[7, 7]]
    names = ("memory.call", "memory.run_result", "memory.helper_call")
    assert _events(by_call, *names) == [
        ("memory.call", FILL, "ok"),  # called once; nothing calls it a second time
        ("memory.run_result", FILL, "returned"),
        ("memory.helper_call", "memory:run", "ok"),
    ]


#: A cell's code that calls *call*, catches what it raises and prints its type, text, the file and function of its
#: innermost frame, the classes it derives from, whether it was raised alone (not chained), and the call's events.
RAISED = (
    "start = len(seen)\n"
    "try:\n"
    "    {call}\n"
    "except Exception as exc:\n"
    "    tb = exc.__traceback__\n"
    "    while tb.tb_next is not None:\n"
    "        tb = tb.tb_next\n"
    "    caught = [type(exc).__name__, str(exc), tb.tb_frame.f_code.co_filename.rsplit('/', 1)[-1],\n"
    "              tb.tb_frame.f_code.co_name, [c.__name__ for c in type(exc).__mro__],\n"
    "              exc.__cause__ is None and exc.__context__ is None]\n"
    "print(json.dumps([*caught, seen[start:]]))\n"
)


def test_a_refusal_reaches_the_caller_as_the_functions_own_error(tmp_path):
    """A direct call's refusal is the function's own ``MemoryInputError``: the same exception, raised in the stored
    function's frame, unwrapped; the tracked call raises its events."""
    root = _library(tmp_path)
    kind, text, where, function, bases, alone, events = _json(
        root,
        EVENTS + f"paint = memory.get({PAINT!r})\n" + RAISED.format(call="paint('not a grid')"),
    )
    assert (kind, text) == ("MemoryInputError", "expected a grid: a list of rows")
    assert (where, function) == ("stencil.py", "paint") and "ValueError" in bases and alone is True
    assert "ContractError" not in bases
    assert _events(events, "memory.call", "memory.run_result", "memory.helper_call") == [
        ("memory.call", PAINT, "refused"),
        ("memory.run_result", PAINT, "refused"),
        ("memory.helper_call", "memory:run", "ok"),
    ]


LOOKUP = '''"""Looking values up."""


def cell(rows, key):
    """The row stored under key."""
    return rows[key]
'''
CELL = "memory.grid.lookup:cell"


def test_a_functions_own_error_reaches_the_caller_unchanged(tmp_path):
    """A ``KeyError`` inside a stored function is that function's error, not a bad argument: it propagates as a
    direct call raises it, from the function's own frame, and counts as the item's error."""
    root = _library(tmp_path, {"memory/grid/lookup.py": LOOKUP})
    kind, text, where, function, bases, alone, events = _json(
        root,
        EVENTS + f"cell = memory.get({CELL!r})\n" + RAISED.format(call="cell({}, 'row-9')"),
    )
    assert (kind, text) == ("KeyError", "'row-9'")
    assert (where, function) == ("lookup.py", "cell") and "ContractError" not in bases and alone is True
    assert _events(events, "memory.call", "memory.run_result", "memory.helper_call") == [
        ("memory.call", CELL, "error"),
        ("memory.run_result", CELL, "error"),
        ("memory.helper_call", "memory:run", "ok"),
    ]


def test_a_captured_procedures_failure_carries_its_detail_and_the_four_parts(tmp_path):
    """A captured procedure runs in a separate process, so no exception object comes back: its failure is one error
    that says what it did, what it takes with an example call, and where to see what it was built for."""
    root = _library(tmp_path)
    capture = "capture:" + "e" * 16
    d = root / ".memory" / "captures" / ("e" * 16)
    d.mkdir(parents=True)
    (d / "procedure.py").write_text("def run(value):\n    raise ValueError('no rows')\n")
    (d / "signature.json").write_text(json.dumps({"input": {}, "relations": [], "constraints": {}}))
    (d / "record.json").write_text(json.dumps({"level": "unchecked", "status": "on-trial", "outcome": "none"}))
    kind, text, _, _, bases, alone, events = _json(
        root,
        EVENTS + f"procedure = memory.get({capture!r})\n" + RAISED.format(call="procedure('x')"),
    )
    assert kind == "ArgumentError" and "TypeError" in bases and alone is True
    assert text == (
        f"memory.get({capture!r})('x'): it did not return (ValueError: no rows).\n"
        f"Expected no arguments (the current request), or a request's text, e.g. memory.get({capture!r})().\n"
        f"Its card shows what it was built for: print(memory.get({capture!r}))."
    )
    assert _events(events, "memory.call", "memory.run_result", "memory.helper_call") == [
        ("memory.call", capture, "error"),
        ("memory.run_result", capture, "error"),
        ("memory.helper_call", "memory:run", "ok"),
    ]


def test_calling_a_note_or_checking_its_fit_raises_with_the_right_call(tmp_path):
    root = _library(tmp_path)
    called, fitted, events = _json(
        root,
        EVENTS
        + f"note = memory.get({NOTE!r})\n"
        "start = len(seen)\n"
        "print(json.dumps([err(note), err(note.fits, {'a': 1}), seen[start:]]))\n",
    )
    assert called[0] == "WrongCall" and called[1] == (
        "memory.get('notes/grid/stencil.md')(): a note cannot be called; it is text to read. "
        "Read it with print(memory.get('notes/grid/stencil.md'))."
    )
    assert fitted[0] == "WrongCall" and fitted[1] == (
        "memory.get('notes/grid/stencil.md').fits({'a': 1}): a note has no inputs to check; it is text to read. "
        "Read it with print(memory.get('notes/grid/stencil.md'))."
    )
    assert _events(events, "memory.helper_call") == [
        ("memory.helper_call", "memory:run", "error"),
        ("memory.helper_call", "memory:match", "error"),
    ]
    assert [a[:2] for ev, a in events if ev == "memory.call_failed"] == [
        ["memory:run", "WrongCall"],
        ["memory:match", "WrongCall"],
    ]


def test_fits_says_why_when_nothing_is_recorded_to_compare(tmp_path):
    root = _library(tmp_path)
    ok, truthy, why, events = _json(
        root,
        EVENTS
        + "fit = memory.get('fill').fits({'grid': [[0]]})\n"
        "print(json.dumps([fit.ok, bool(fit), fit.why, seen]))\n",
    )
    assert ok is False and truthy is False
    assert why.startswith("no input it was saved against is recorded") and "print(memory.get(" in why
    assert [ev for ev, _ in events if ev in ("memory.match_result", "memory.helper_call")] == [
        "memory.helper_call",  # the lookup
        "memory.match_result",
        "memory.helper_call",  # the check, as memory:match
    ]


# --- the contract as Python ----------------------------------------------------------------------------------------


def test_the_stub_is_generated_valid_python_and_holds_the_contract(tmp_path):
    root = _library(tmp_path)
    off = (root / RESERVED_STUB).read_text()
    on = lx.stub_source(librarian=True).decode("utf-8")
    assert off == lx.stub_source().decode("utf-8") == lx.stub_source().decode("utf-8")  # same code, same bytes
    for stub in (off, on):
        compile(stub, "memory/__init__.pyi", "exec")
        assert stub.startswith("# memory: the library's typed contract")
        for line in (
            "@dataclass(frozen=True)\nclass Fit:",
            "@dataclass(frozen=True)\nclass Item:",
            "@dataclass(frozen=True)\nclass Items:",
            "    kind: Literal['function', 'note']",
            "    evidence: Literal['unchecked', 'reproduces-its-case', 'confirmed-by-outcomes']",
            "    status: Literal['on-trial', 'adopted', 'retired']",
            "    links: tuple[str, ...]",
            "    def fits(self, inputs: Any) -> Fit:",
            "    rows: tuple[Item, ...]",
            "    def page(self, n: int) -> Items:",
            "def items(topic: str | None = None, kind: Literal['function', 'note'] | None = None) -> Items:",
            "def get(ref: str | Item | Callable) -> Item:",
        ):
            assert line in stub, line
    # memory.ask takes an optional schema for a structured answer (a list of Item without one)
    assert "def ask(" not in off and "def ask(question: str | None = None, schema: Any = None) -> Any:" in on
    for gone in ("def index(", "def show(", "def find(", "def match(", "def run("):
        assert gone not in on
    assert layout.classify(RESERVED_STUB) == "generated" and RESERVED_STUB in layout.GENERATED_ROOT_FILES


def test_help_shows_the_contract(tmp_path):
    root = _library(tmp_path)
    text = _json(root, "import pydoc\nprint(json.dumps(pydoc.render_doc(memory, renderer=pydoc.plaintext)))")
    assert "memory.get(ref) -> Item" in text and "memory/__init__.pyi" in text
    for shown in ("get(ref:", "items(topic:", "class Item(", "class Items(", "class Fit(", "fits(self, inputs"):
        assert shown in text, shown
    assert "show(item:" not in text and "run(item:" not in text  # the earlier calls are gone


def test_every_public_call_is_checked_and_documented():
    """Each call of the contract and each public method of its types checks its arguments and documents its
    arguments, return, errors and one example call."""
    from unify.memory import library_ask

    calls = {
        "memory.get": library_helper.get,
        "memory.items": library_helper.items,
        "memory.ask": library_ask.ask,
        "Item.fits": library_helper.Item.fits,
        "Item.__call__": library_helper.Item.__call__,
        "Items.page": library_helper.Items.page,
    }
    for name, fn in calls.items():
        assert getattr(fn, "__contract_checked__", False), name
        doc = fn.__doc__ or ""
        for header in ("Args:", "Returns:", "Raises:", "Example:"):
            assert f"\n    {header}\n" in doc or f"\n        {header}\n" in doc, (name, header)
    assert set(library_helper.__all__) >= {"get", "items", "Item", "Items", "Fit"}
    # the errors are names of the package, named in its documentation, but out of __all__ so that help(memory) does
    # not print each one's inherited exception machinery
    errors = {"Ambiguous", "ArgumentError", "ContractError", "NotFound", "WrongCall"}
    assert not errors & set(library_helper.__all__)
    assert all(isinstance(getattr(library_helper, n), type) for n in errors)
    assert all(f"memory.{n}" in library_helper.__doc__ for n in errors)
    assert not {"index", "show", "find", "match", "run"} & set(library_helper.__all__)
    assert not [n for n in ("index", "show", "find", "match", "run") if hasattr(library_helper, n)]
    assert {"ask", "get", "items"} <= layout.RESERVED_PACKAGES
    assert layout.classify("memory/get/x.py") is None and layout.classify("memory/items/x.py") is None


def test_a_functions_error_carries_one_note_on_what_to_do_next(tmp_path):
    """The function's own exception reaches the caller with one note: for its ``MemoryInputError`` (whose message is
    its writer's), that the item was not built for this input and what to do; for any other error, where the card
    is. A second call adds no second note."""
    root = _library(tmp_path, {"memory/grid/lookup.py": LOOKUP})
    code = (
        "import json, memory\n"
        "out = []\n"
        f"for call in [\"memory.get({PAINT!r})('not a grid')\", \"memory.get({CELL!r})({{}}, 'row-9')\"]:\n"
        "    try:\n"
        "        eval(call)\n"
        "    except Exception as exc:\n"
        "        out.append(list(getattr(exc, '__notes__', [])))\n"
        "print(json.dumps(out))\n"
    )
    refusal, crash = _json(root, code)
    assert refusal == [
        f"{PAINT} was not built for this input; its card shows what it was built for: "
        f"print(memory.get({PAINT!r})). Do this step yourself, or look for another item with memory.items()."
    ]
    assert crash == [
        f"raised inside the stored function {CELL}; its card shows its records: "
        f"print(memory.get({CELL!r})), and its source: print(memory.get({CELL!r}).text)."
    ]
