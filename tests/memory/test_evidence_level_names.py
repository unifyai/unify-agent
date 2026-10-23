"""The renamed evidence levels and statuses, in the item records, the index line and ``memory.show``.

An item has two separate axes. Its evidence level says how much its evidence shows:
``unchecked`` -> ``reproduces-its-case`` -> ``confirmed-by-outcomes`` (rev 2.1: the gate runs no code, so only
outcomes after use move it; the index line and ``memory.show`` show what an item rests on in its place: its kind,
the episodes it comes from and its confirmed uses). Its status says where it is in its life:
``on-trial`` -> ``adopted`` -> ``retired``. The old status names (``experimental``, ``stable``) never reach a record, an
index line or ``memory.show`` again, but a record stored under an old name still loads, as the new name.
"""

from __future__ import annotations

import json
import re

from tests.memory.test_layout import _tree
from tests.memory.test_library_helper import _commit, _py
from unify.memory import item_records as ir
from unify.memory import item_states
from unify.memory import library_export as lx
from unify.memory import library_index
from unify.memory import lifecycle as lc
from unify.memory.catalogue import write_files
from unify.memory.gitio import Repo

# --- assumed names (adapt here) ---------------------------------------------------------------------------------
#: the evidence levels, lowest first, beside STATUSES and EFFECTS
LEVELS_OF = lambda: item_states.LEVELS  # noqa: E731
LEVELS = ("unchecked", "reproduces-its-case", "confirmed-by-outcomes")
#: the statuses, in an item's life
ON_TRIAL, ADOPTED, RETIRED = "on-trial", "adopted", "retired"
#: what a record stored before the rename may hold, and what it loads as
OLD_STATUS_LOADS_AS = {"experimental": ON_TRIAL, "stable": ADOPTED}
#: the record fields the harness writes from the gate run
LEVEL_FIELD, LEVEL_REASONS_FIELD = "level", "level_reasons"
#: what the brackets show in its place (rev 2.1): an item whose record names no provenance episode
SHOWN_SUPPORT = "curated item from 0 episodes"
#: the level a record stored before evidence levels existed reads as, when nothing in it shows its examples ran on
#: recorded values: the lowest (a missing level fails closed; decided for this revision)
KEPT_LEVEL = "unchecked"
# -----------------------------------------------------------------------------------------------------------------

DATE = "memory.text.dates:parse_date"
WEEK = "memory.text.dates:week"
FLAG = (
    "G2: [examples:special-cased] memory.text.dates:parse_date handles memory/text/tests/data/sample.json only "
    "for its recorded values"
)


def _record(item: str, **over) -> dict:
    rec = ir.empty_record(item, "function")
    rec.update(over)
    return rec


def _library(tmp_path, records):
    root = _tree(tmp_path / "lib")
    write_files(
        root,
        lx.generated_library_mode(
            root,
            status_of=ir.status_of(records),
            records=records,
        ),
    )
    return root


def _index_line(root, item: str) -> str:
    module, name = item.split(":", 1)
    lines = [
        ln
        for ln in (root / "INDEX.md").read_text().splitlines()
        if ln.startswith(f"- `{module}:{name}(")
    ]
    assert len(lines) == 1, (root / "INDEX.md").read_text()
    return lines[0]


def _brackets(line: str) -> str:
    found = re.findall(r"\[([^\]]*)\]", line)
    assert found, line
    return found[-1]


# --- the names --------------------------------------------------------------------------------------------------


def test_the_levels_are_the_three_renamed_ones_lowest_first():
    assert tuple(LEVELS_OF()) == LEVELS
    # no level shares a name with a status
    assert not set(LEVELS) & set(library_index.STATUSES)


def test_the_statuses_are_renamed_and_the_old_names_are_gone():
    statuses = set(library_index.STATUSES)
    assert {ON_TRIAL, ADOPTED, RETIRED} <= statuses
    assert "experimental" not in statuses and "stable" not in statuses


def test_a_new_record_is_on_trial_and_carries_a_level_and_its_reasons():
    rec = ir.empty_record(DATE, "function")
    assert rec["status"] == ON_TRIAL
    assert LEVEL_FIELD in rec and LEVEL_REASONS_FIELD in rec
    assert rec[LEVEL_REASONS_FIELD] == []
    assert rec[LEVEL_FIELD] is None or rec[LEVEL_FIELD] in LEVELS


def test_a_status_decided_from_no_use_is_on_trial_and_an_adopted_one_stays():
    assert lc.decide_status(WEEK, None, [])[0] == ON_TRIAL
    assert lc.decide_status(WEEK, ON_TRIAL, [])[0] == ON_TRIAL
    assert lc.decide_status(WEEK, ADOPTED, [])[0] == ADOPTED


def test_an_old_stored_status_still_loads_as_its_new_name():
    records = {
        DATE: {"item": DATE, "status": "experimental"},
        WEEK: {"item": WEEK, "status": "stable"},
    }
    get = ir.status_of(records)
    assert get(DATE) == OLD_STATUS_LOADS_AS["experimental"]
    assert get(WEEK) == OLD_STATUS_LOADS_AS["stable"]
    # no record, or an unreadable one: on trial
    assert get("memory.text.parse:tokens") == ON_TRIAL
    assert ir.status_of({DATE: {"item": DATE, "status": "nonsense"}})(DATE) == ON_TRIAL
    # the lifecycle reads a prior decided under the old names as the new ones
    assert lc.decide_status(WEEK, "experimental", [])[0] == ON_TRIAL
    assert lc.decide_status(WEEK, "stable", [])[0] == ADOPTED


def test_a_map_written_under_the_old_names_still_loads(tmp_path):
    """A note on refs/notes/items written before the rename (raw lines, as an older harness wrote them) is read,
    and its statuses come back under the new names."""
    mem = Repo.init_bare(tmp_path / "mem.git")
    _commit(mem, {"memory/text/dates.py": "def week(s):\n    return 41\n"}, "library")
    head = mem.head()
    old = [{"item": WEEK, "kind": "function", "status": "stable"}]
    lines = [json.dumps({"items_map": ir.RECORD_VERSION, "commit": head, "count": 1})]
    lines += [json.dumps(r, sort_keys=True) for r in old]
    mem.append_note_lines(head, lines, ir.NOTES_REF)
    records, at = ir.records_at(mem, head)
    assert at == head and WEEK in records
    assert ir.status_of(records)(WEEK) == ADOPTED


def test_a_record_with_the_new_status_and_a_level_is_written_and_read_back(tmp_path):
    mem = Repo.init_bare(tmp_path / "mem.git")
    _commit(mem, {"memory/text/dates.py": "def week(s):\n    return 41\n"}, "library")
    head = mem.head()
    recs = {
        WEEK: _record(
            WEEK,
            status=ON_TRIAL,
            **{LEVEL_FIELD: "unchecked", LEVEL_REASONS_FIELD: [FLAG]},
        ),
    }
    assert ir.write_records(mem, head, recs) is True
    back = ir.read_records(mem, head)
    assert back[WEEK]["status"] == ON_TRIAL
    assert back[WEEK][LEVEL_FIELD] == "unchecked"
    assert back[WEEK][LEVEL_REASONS_FIELD] == [FLAG]


# --- what the actor and the writer see ----------------------------------------------------------------------------


def test_the_index_line_shows_the_status_and_the_support_apart(tmp_path):
    records = {
        DATE: _record(
            DATE,
            status=ON_TRIAL,
            **{LEVEL_FIELD: "unchecked", LEVEL_REASONS_FIELD: [FLAG]},
        ),
        WEEK: _record(
            WEEK,
            status=ADOPTED,
            **{LEVEL_FIELD: "reproduces-its-case", LEVEL_REASONS_FIELD: []},
        ),
    }
    root = _library(tmp_path, records)
    date = _brackets(_index_line(root, DATE))
    week = _brackets(_index_line(root, WEEK))
    assert ON_TRIAL in date and SHOWN_SUPPORT in date and "evidence:" not in date
    assert ADOPTED in week and SHOWN_SUPPORT in week and "evidence:" not in week
    index = (root / "INDEX.md").read_text()
    assert "[experimental" not in index and "[stable" not in index


def test_show_gives_the_status_the_support_and_the_gates_finding(tmp_path):
    """An item the check with unseen values flagged lands at the lowest level; the flag is shown to the agent
    (``memory.show``, as a gate finding) as well as to the writer (the gate's reply)."""
    records = {
        DATE: _record(
            DATE,
            status=ON_TRIAL,
            **{LEVEL_FIELD: "unchecked", LEVEL_REASONS_FIELD: [FLAG]},
        ),
    }
    root = _library(tmp_path, records)
    out = json.loads(
        _py(root, f"import json, memory\nprint(json.dumps(memory.show({DATE!r})))"),
    )
    first = out.splitlines()[0]
    assert first.startswith("memory.text.dates:parse_date(")
    assert ON_TRIAL in first and SHOWN_SUPPORT in first and "evidence:" not in first
    # the gate's findings are the writer's: never on a card the actor or the librarian reads
    assert "Gate findings:" not in out and "[examples:special-cased]" not in out
    assert "experimental" not in out


def test_a_record_from_before_levels_reads_as_the_lowest_level(tmp_path):
    """A record with no level, and no verification showing its examples ran on recorded values, fails closed: it
    reads as the lowest level, and shows its old status under its new name, with what it rests on."""
    old = {"item": WEEK, "kind": "function", "status": "stable"}
    assert item_states.level_of_record(old.get("verification")) == KEPT_LEVEL
    root = _library(tmp_path, {WEEK: old})
    week = _brackets(_index_line(root, WEEK))
    assert ADOPTED in week and SHOWN_SUPPORT in week
    out = json.loads(
        _py(root, f"import json, memory\nprint(json.dumps(memory.show({WEEK!r})))"),
    )
    assert ADOPTED in out.splitlines()[0] and "stable" not in out.splitlines()[0]
