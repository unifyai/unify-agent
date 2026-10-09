"""The use record of a v2.1 library (spec §4.2, §5 item 4): every import form reaches the defining item, errors
leave a function by its frame, and the helper's lookups (show, index, find) are lookups, not uses.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from tests.memory_v2.test_episodes import _ep
from tests.memory_v2.test_use_telemetry import _lines, _result, _session
from unify.memory_v2.analysis import use
from unify.memory_v2.integration.request import RequestRun, memory_use

HELPER = """def show(item):
    return ""


def index(package=None):
    return ""


def find(value):
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
LIB21 = {
    "memory/__init__.py": HELPER,  # a stand-in for P3's generated helper
    "memory/text/__init__.py": '"""Text helpers."""\nfrom .parse import tokens\n',
    "memory/text/parse.py": PARSE,
    "memory/text/dates.py": DATES,
}
ITEMS21 = [
    "memory.text.dates:parse_date",
    "memory.text.dates:week",
    "memory.text.parse:tokens",
]
PD, WEEK, TOK = ITEMS21


@pytest.fixture
def lib21(tmp_path, monkeypatch):
    """A v2.1 export on the import path; ``memory`` modules are dropped before and after."""
    root = tmp_path / "checkout"
    for rel, text in LIB21.items():
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


def _status21(cells, roots=()):
    """Each cell's status as a v2.1 request notes it (``RequestRun.note_result`` with ``v21``)."""
    run = RequestRun("r", None)
    run.v21, run.item_ids, run.export_roots = True, list(ITEMS21), list(roots)
    for i, (_, err) in enumerate(cells):
        run.note_result(f"c{i}", _result(err))
    return run.cell_status


def _use21(root: Path, codes: list[str]) -> dict:
    cells = _session(codes)
    return use.request_use(
        _lines(cells),
        ITEMS21,
        cell_status=_status21(cells),
        surface=use.library_surface(root, v21=True),
        v21=True,
    )


def test_a_by_name_import_and_call_count_on_the_defining_item(lib21):
    rec = _use21(
        lib21,
        ["from memory.text.dates import parse_date\nparse_date('2026-10-09')\n"],
    )
    assert rec["layout"] == "v21"
    row = rec["items"][PD]
    assert (row["imported"], row["called"], row["referenced"], row["cells"]) == (
        1,
        1,
        0,
        [0],
    )
    assert set(rec["items"]) == {PD} and rec["unknown_calls"] == {}


def test_every_import_form_resolves_to_the_defining_item(lib21):
    rec = _use21(
        lib21,
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


def test_a_star_import_binds_the_modules_public_names(lib21):
    rec = _use21(lib21, ["from memory.text.dates import *\nweek('2026-10-09')\n"])
    assert {i: r["imported"] for i, r in rec["items"].items()} == {
        PD: 1,
        WEEK: 1,
        TOK: 1,
    }
    assert rec["items"][WEEK]["called"] == 1


def test_helper_lookups_are_lookups_not_uses(lib21):
    rec = _use21(
        lib21,
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
        "calls": {"show": 2, "index": 1, "find": 1, "help": 0},
    }


def test_refusals_and_errors_leave_the_function_by_its_frame(lib21):
    rec = _use21(
        lib21,
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
    assert use.attribute_errors(tb, ITEMS21, roots=[str(lib21)], v21=True) == [
        ("refused", PD),
    ]
    assert (
        use.attribute_errors(tb, ITEMS21, roots=[str(lib21)]) == []
    )  # v2's reading knows no memory.* item


def test_the_v21_surface_reads_packages_modules_and_reexports(lib21):
    s = use.library_surface(lib21, v21=True)
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
    assert use.library_surface(lib21) == {
        "star": {},
        "reexports": {},
    }  # v2 reads env/ only


def test_with_v21_off_the_record_is_v2s(lib21):
    cells = _session(
        ["from memory.text.dates import parse_date\nparse_date('2026-10-09')\n"],
    )
    rec = use.request_use(_lines(cells), ITEMS21, cell_status=_status21(cells))
    assert "layout" not in rec and rec["items"] == {} and rec["items_at_pin"] == []


def test_the_request_records_v21_use(lib21):
    cells = _session(["from memory.text.dates import week\nweek('2026-10-09')\n"])
    ep = _ep(transcript=_lines(cells), actions=[])
    rec = memory_use(
        ep,
        ITEMS21,
        cell_status=_status21(cells),
        surface=use.library_surface(lib21, v21=True),
        v21=True,
    )
    assert rec["layout"] == "v21" and rec["items"][WEEK]["called"] == 1
