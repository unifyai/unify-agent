"""The use record of a library-mode export: every import form reaches the defining item, errors
leave a function by its frame, and the package's lookups (get and items, and the aliases show, index and find) are
lookups, not uses.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from tests.memory.test_episodes import _ep
from tests.memory.test_use_telemetry import _lines, _result, _session
from unify.memory.analysis import use
from unify.memory.integration.request import RequestRun, memory_use

HELPER = """def show(item):
    return ""


def index(package=None):
    return ""


def find(value):
    return []


def get(ref):
    return lambda *args, **kwargs: None


def items(topic=None, kind=None):
    return []
"""
PARSE = '''class MemoryInputError(ValueError):
    pass


def tokens(s):
    """Split text on whitespace."""
    if not isinstance(s, str):
        raise MemoryInputError("expected text")
    return s.split()
'''
DATES = '''from memory.text.parse import MemoryInputError, tokens


def parse_date(s):
    """Parse a date written as YYYY-MM-DD."""
    if not isinstance(s, str):
        raise MemoryInputError("expected a date")
    return "-".join(tokens(s.replace("-", " ")))


def week(s):
    """The ISO week of a date."""
    return {"2026-10-09": 41}[s]
'''
LIBRARY_FILES = {
    "memory/__init__.py": HELPER,  # a stand-in for the generated helper
    "memory/text/__init__.py": '"""Text helpers."""\nfrom .parse import tokens\n',
    "memory/text/parse.py": PARSE,
    "memory/text/dates.py": DATES,
}
ITEMS_LIBRARY = [
    "memory.text.dates:parse_date",
    "memory.text.dates:week",
    "memory.text.parse:tokens",
]
PD, WEEK, TOK = ITEMS_LIBRARY


@pytest.fixture
def library_root(tmp_path, monkeypatch):
    """A library-mode export on the import path; ``memory`` modules are dropped before and after."""
    root = tmp_path / "checkout"
    for rel, text in LIBRARY_FILES.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text)
    monkeypatch.syspath_prepend(str(root))

    def drop():
        for name in [
            m for m in sys.modules if m == "memory" or m.startswith("memory.")
        ]:
            sys.modules.pop(name, None)

    drop()
    yield root
    drop()


def _status_library(cells, roots=()):
    """Each cell's status as a library-mode request notes it (``RequestRun.note_result`` with ``library_mode``)."""
    run = RequestRun("r", None)
    run.library_mode, run.item_ids, run.export_roots = (
        True,
        list(ITEMS_LIBRARY),
        list(roots),
    )
    for i, (_, err) in enumerate(cells):
        run.note_result(f"c{i}", _result(err))
    return run.cell_status


def _use_library(root: Path, codes: list[str]) -> dict:
    cells = _session(codes)
    return use.request_use(
        _lines(cells),
        ITEMS_LIBRARY,
        cell_status=_status_library(cells),
        surface=use.library_surface(root),
        library_mode=True,
    )


def test_a_by_name_import_and_call_count_on_the_defining_item(library_root):
    rec = _use_library(
        library_root,
        ["from memory.text.dates import parse_date\nparse_date('2026-10-09')\n"],
    )
    assert rec["layout"] == "library_mode"
    row = rec["items"][PD]
    assert (row["imported"], row["called"], row["referenced"], row["cells"]) == (
        1,
        1,
        0,
        [0],
    )
    assert set(rec["items"]) == {PD} and rec["unknown_calls"] == {}


def test_every_import_form_resolves_to_the_defining_item(library_root):
    rec = _use_library(
        library_root,
        [
            "import memory.text.dates\nmemory.text.dates.parse_date('2026-10-09')\n",
            "import memory.text.dates as d\nd.parse_date('2026-10-09')\n",
            "from memory.text import dates\ndates.week('2026-10-09')\n",
            "from memory import text\ntext.dates.parse_date('2026-10-09')\n",
            "from memory.text import tokens\ntokens('a b')\n",  # the package re-exports text.parse's tokens
            "import memory.text as t\nt.tokens('c')\n",
        ],
    )
    assert (rec["items"][PD]["imported"], rec["items"][PD]["called"]) == (0, 3)
    assert rec["items"][WEEK]["called"] == 1
    assert (rec["items"][TOK]["imported"], rec["items"][TOK]["called"]) == (1, 2)
    assert rec["module_imports"] == {"text": 2, "text.dates": 3}


def test_a_star_import_binds_the_modules_public_names(library_root):
    rec = _use_library(
        library_root,
        ["from memory.text.dates import *\nweek('2026-10-09')\n"],
    )
    assert {i: r["imported"] for i, r in rec["items"].items()} == {
        PD: 1,
        WEEK: 1,
        TOK: 1,
    }
    assert rec["items"][WEEK]["called"] == 1


def test_helper_lookups_are_lookups_not_uses(library_root):
    rec = _use_library(
        library_root,
        [
            "import memory\nmemory.show('memory.text.dates:parse_date')\nmemory.index('text')\n"
            "memory.find({'a': 1})\n",
            "import memory.text.dates\nfrom memory import show\nshow(memory.text.dates.week)\n",
        ],
    )
    assert rec["items"] == {}
    assert rec["cell_exposure"] == {
        "items": [PD, WEEK, TOK],
        "channels": ["text", "text.dates"],
        "calls": {"show": 2, "index": 1, "find": 1, "match": 0, "run": 0, "get": 0, "items": 0, "help": 0},
    }


def test_get_and_items_are_lookups_as_show_and_index_were(library_root):
    """``memory.get`` shows one item as ``show`` did and ``memory.items`` a topic as ``index`` did; calling the item
    ``get`` returned is a tracked call the audit hook counts, not a use the code shows."""
    rec = _use_library(
        library_root,
        [
            "import memory\nmemory.get('memory.text.dates:parse_date')\nmemory.items('text')\n",
            "import memory.text.dates\nfrom memory import get\nget(memory.text.dates.week)\n"
            "memory.get('memory.text.dates:week')('2026-10-09')\nmemory.items(topic='text')\n",
        ],
    )
    assert rec["items"] == {}
    assert rec["cell_exposure"] == {
        "items": [PD, WEEK, TOK],
        "channels": ["text", "text.dates"],
        "calls": {"show": 0, "index": 0, "find": 0, "match": 0, "run": 0, "get": 3, "items": 2, "help": 0},
    }


def test_refusals_and_errors_leave_the_function_by_its_frame(library_root):
    rec = _use_library(
        library_root,
        [
            "from memory.text.dates import parse_date, week\n",
            "parse_date(3)\n",
            "week('nope')\n",
        ],
    )
    assert (rec["items"][PD]["refused"], rec["items"][PD]["errored"]) == (1, 0)
    assert (rec["items"][WEEK]["refused"], rec["items"][WEEK]["errored"]) == (0, 1)
    tb = _session(["from memory.text.dates import parse_date\n", "parse_date(3)\n"])[1][
        1
    ]
    assert use.attribute_errors(
        tb,
        ITEMS_LIBRARY,
        roots=[str(library_root)],
    ) == [
        ("refused", PD),
    ]
    assert (
        use.attribute_errors(tb, ITEMS_LIBRARY, roots=[str(library_root)]) == []
    )  # v2's reading knows no memory.* item


def test_the_library_surface_reads_packages_modules_and_reexports(library_root):
    s = use.library_surface(library_root)
    assert s["star"] == {
        "text": ["tokens"],
        "text.dates": ["MemoryInputError", "parse_date", "tokens", "week"],
        "text.parse": ["MemoryInputError", "tokens"],
    }
    assert s["reexports"] == {
        "memory.text.dates:MemoryInputError": "memory.text.parse:MemoryInputError",
        "memory.text.dates:tokens": "memory.text.parse:tokens",
        "memory.text:tokens": "memory.text.parse:tokens",
    }
    assert use.library_surface(library_root) == {
        "star": {},
        "reexports": {},
    }  # v2 reads env/ only


def test_with_library_off_the_record_is_v2s(library_root):
    cells = _session(
        ["from memory.text.dates import parse_date\nparse_date('2026-10-09')\n"],
    )
    rec = use.request_use(
        _lines(cells),
        ITEMS_LIBRARY,
        cell_status=_status_library(cells),
    )
    assert "layout" not in rec and rec["items"] == {} and rec["items_at_pin"] == []


def test_the_request_records_library_use(library_root):
    cells = _session(["from memory.text.dates import week\nweek('2026-10-09')\n"])
    ep = _ep(transcript=_lines(cells), actions=[])
    rec = memory_use(
        ep,
        ITEMS_LIBRARY,
        cell_status=_status_library(cells),
        surface=use.library_surface(library_root),
    )
    assert rec["layout"] == "library_mode" and rec["items"][WEEK]["called"] == 1
