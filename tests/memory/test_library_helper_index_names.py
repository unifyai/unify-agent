"""``memory.index(name)``, the alias kept for one build: a bare name is a package's section, else the notes topic of
that name; an item's id, as the index lists it, shows the item; a name that is none of these raises NotFound (a
LookupError) naming the nearest of the index's sections and the call that replaces it. Run as a cell runs it, from
the copied helper file."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.memory.test_library_helper import DATE, _library, _py
from unify.memory import library_helper
from unify.memory.layout import INDEX_FILE, RESERVED_CONTRACT, RESERVED_INIT
from unify.memory.library_helper import find_section

HEAD = "# Memory index\n\nOne line per item.\n"
INDEX = (
    HEAD
    + "\n## memory.grid — Grid helpers.\n\n- `memory.grid.cells:rows(g)` — Rows of a grid. [on-trial]\n"
    + "\n## memory.text — Text helpers.\n\n- `memory.text.parse:tokens(s)` — Split text. [on-trial]\n"
    + "\n## notes/grid\n\n- `notes/grid/shapes.md` — Grid shapes. [on-trial]\n"
    + "\n## notes/maps\n\n- `notes/maps/legend.md` — Map legends. [on-trial]\n"
)


def lx_contract() -> Path:
    from unify.memory import library_export

    return library_export.CONTRACT_SOURCE


def _copy(tmp_path: Path, index: str) -> Path:
    """A library copy holding only the helper and its contract runtime (byte for byte as generated) and *index*."""
    root = tmp_path / "copy"
    init = root / RESERVED_INIT
    init.parent.mkdir(parents=True)
    init.write_bytes(Path(library_helper.__file__).read_bytes())
    (root / RESERVED_CONTRACT).write_bytes(lx_contract().read_bytes())
    (root / INDEX_FILE).write_text(index)
    return root


def _cell(root: Path, names: list) -> list:
    """``memory.index(name)`` for each of *names* in a fresh interpreter: the text, or the LookupError's message."""
    code = (
        "import json, memory\n"
        "out = []\n"
        f"for name in {names!r}:\n"
        "    try:\n"
        "        out.append(['ok', memory.index(*([] if name is None else [name]))])\n"
        "    except LookupError as exc:\n"
        "        out.append(['missing', str(exc)])\n"
        "print(json.dumps(out))\n"
    )
    return json.loads(_py(root, code))


def test_a_bare_name_is_a_package_else_a_notes_topic(tmp_path):
    root = _copy(tmp_path, INDEX)
    (text, maps, notes_grid, grid, dotted, slashed) = _cell(
        root,
        ["text", "maps", "notes/grid", "grid", "memory.grid", "memory/grid"],
    )
    assert text == ["ok", find_section(INDEX, "memory.text")]
    assert text[1].startswith("## memory.text — Text helpers.\n")
    # no package is named maps: the notes topic
    assert maps == ["ok", "## notes/maps\n\n- `notes/maps/legend.md` — Map legends. [on-trial]\n"]
    assert notes_grid[1].startswith("## notes/grid\n")
    # both a package and a notes topic: the package, as before
    assert grid[1].startswith("## memory.grid — Grid helpers.\n")
    assert grid == dotted == slashed


def test_a_notes_only_library_answers_a_bare_topic_name(tmp_path):
    index = HEAD + "\n## notes/grid\n\n- `notes/grid/shapes.md` — Grid shapes. [on-trial]\n"
    (grid,) = _cell(_copy(tmp_path, index), ["grid"])
    assert grid == ["ok", "## notes/grid\n\n- `notes/grid/shapes.md` — Grid shapes. [on-trial]\n"]


def test_a_name_that_is_neither_names_the_nearest_sections_and_the_new_call(tmp_path):
    (nope, prefixed) = _cell(_copy(tmp_path, INDEX), ["nope", "memory.nope"])
    assert nope == [
        "missing",
        "memory.index('nope'): no section or item named 'nope'.\n"
        "Expected a package or a notes topic, e.g. memory.items('memory.grid').\n"
        "Browse every topic with memory.items(); memory.get(ref) opens one item.",
    ]
    assert prefixed == [
        "missing",
        "memory.index('memory.nope'): no section or item named 'memory.nope'.\n"
        "Expected a package or a notes topic, e.g. memory.items('memory.text').\n"
        "Nearest: memory.text, memory.grid. Browse every topic with memory.items(); memory.get(ref) opens one item.",
    ]
    with pytest.raises(LookupError, match="no index section 'notes/nope';"):
        find_section(INDEX, "notes/nope")


def test_an_empty_library_has_no_sections(tmp_path):
    (whole, grid) = _cell(_copy(tmp_path, HEAD), [None, "grid"])
    assert whole == ["ok", HEAD]
    assert grid == [
        "missing",
        "memory.index('grid'): no section or item named 'grid'.\n"
        "Expected a package or a notes topic, e.g. memory.items('notes/topic').\n"
        "Browse every topic with memory.items(); memory.get(ref) opens one item.",
    ]


def test_the_lookup_still_raises_its_helper_audit_event(tmp_path):
    root = _copy(tmp_path, INDEX)
    code = (
        "import json, sys, memory\n"
        "seen = []\n"
        "sys.addaudithook(lambda ev, args: seen.append([ev, args[0], args[2]]) if ev == 'memory.helper_call' else None)\n"
        "memory.index('maps')\n"
        "try:\n    memory.index('nope')\nexcept LookupError:\n    pass\n"
        "print(json.dumps(seen))\n"
    )
    assert json.loads(_py(root, code)) == [
        ["memory.helper_call", "memory:index", "ok"],
        ["memory.helper_call", "memory:index", "error"],
    ]


def test_an_item_id_given_as_a_name_shows_the_item(tmp_path):
    """A cell that passes an id it read in the index to memory.index, not memory.show, gets the item, recorded as one
    index lookup (in the live runs a note's id was passed so in many visits, and each call raised)."""
    root = _library(tmp_path)
    code = (
        "import json, sys, memory\n"
        "seen = []\n"
        "sys.addaudithook(lambda ev, args: seen.append([ev, args[0], args[2]]) if ev == 'memory.helper_call' else None)\n"
        "note = memory.index('notes/text/dates.md')\n"
        f"fn = memory.index({DATE!r})\n"
        "try:\n    memory.index('notes/text/nope.md')\nexcept LookupError as exc:\n    missing = str(exc)\n"
        "events = list(seen)\n"
        f"print(json.dumps([note, fn, missing, events, memory.show('notes/text/dates.md'), memory.show({DATE!r})]))\n"
    )
    note, fn, missing, events, shown_note, shown_fn = json.loads(_py(root, code))
    assert note == shown_note and note.startswith("notes/text/dates.md  [on-trial]")
    assert fn == shown_fn and fn.startswith("memory.text.dates:parse_date(")
    assert missing.startswith(
        "memory.index('notes/text/nope.md'): no section or item named 'notes/text/nope.md'.\n"
        "Expected a package or a notes topic, e.g. memory.items('notes/text').\nNearest: notes/text.",
    )
    assert events == [
        ["memory.helper_call", "memory:index", "ok"],
        ["memory.helper_call", "memory:index", "ok"],
        ["memory.helper_call", "memory:index", "error"],
    ]
