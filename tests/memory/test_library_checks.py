"""The library gate's checks (``unify.memory.library_checks``) on a pass's candidate, read as data: what a pass may not
do to the library (fewer items, a declared deletion, an alias it drops, an alias or retirement it declares, a rename
that does not name two items), and what every new or changed note must hold (its ``## Rule`` section, which refuses,
and its ``When to use`` section, which only labels). Which findings refuse is the gate policy's to say
(``gate_policy.GATE``); each test asks it of the findings the checks made. Nothing here runs library code.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from tests.memory.test_gate_library_checks import _checks, _Run
from tests.memory.test_layout import LIB, _tree
from unify.memory.gate_policy import GATE
from unify.memory.library_checks import LibraryChecks, LibraryConfig

NOTE_ID = "notes/text/rows.md"
WHEN = (
    "## When to use\n"
    "- **Saved for:** reading the rows a request carries.\n"
    "- **Recognise:** a request with a table of rows.\n"
    "- **Not for:** sorting the rows.\n"
    "- **Example request:** Add up the amounts of these rows.\n"
)
RULE = "## Rule\n- the rows follow the line ROWS and end at END\n"
FRONT = "---\ntitle: Rows\ndescription: how rows are read\nuses: []\n---\n"


def _files(tree: Path) -> dict[str, bytes]:
    return {str(p.relative_to(tree)): p.read_bytes() for p in sorted(tree.rglob("*")) if p.is_file()}


def _note_run(tmp_path: Path, text: str) -> LibraryChecks:
    note = SimpleNamespace(item=NOTE_ID, kind="note", tests=[], source_episodes=["e1"], covers=[], typed_covers=[])
    v = _checks(tmp_path, [note], c_bodies={NOTE_ID: ("note", text)}, p_bodies={})
    v._when_to_use()
    return v


def _refusing(fails: list) -> list:
    return [(check, reason) for check, reason, owners in fails if GATE.hard(check, reason, [owners] if isinstance(
        owners, str) else owners or [])]


def test_a_new_note_without_a_rule_section_is_refused_and_one_without_when_to_use_is_labelled(tmp_path):
    v = _note_run(tmp_path, FRONT + WHEN + "\nThe rows follow the line ROWS.\n")
    assert [(c, o) for c, _, o in v.run.fails] == [("G1", NOTE_ID)]
    assert "has no `## Rule` section" in v.run.fails[0][1]
    assert _refusing(v.run.fails) == [("G1", v.run.fails[0][1])]

    v = _note_run(tmp_path, FRONT + RULE)
    assert v.run.fails and all(c == "G1" and o == NOTE_ID for c, _, o in v.run.fails)
    assert all("When to use" in r for _, r, _ in v.run.fails), v.run.fails
    assert _refusing(v.run.fails) == []  # a missing When to use section labels the note; it still lands

    v = _note_run(tmp_path, FRONT + RULE + "\n" + WHEN)
    assert v.run.fails == []


def test_an_empty_rule_section_is_refused_as_a_missing_one_is(tmp_path):
    v = _note_run(tmp_path, FRONT + "## Rule\n\n## Evidence\n- case 1\n" + WHEN)
    assert [r for _, r, _ in v.run.fails] == [f"{NOTE_ID} has an empty `## Rule` section"]
    assert _refusing(v.run.fails)


def test_an_unchanged_note_is_not_checked_again(tmp_path):
    old = FRONT + "The rows follow the line ROWS.\n"
    note = SimpleNamespace(item=NOTE_ID, kind="note", tests=[], source_episodes=["e1"], covers=[], typed_covers=[])
    v = _checks(tmp_path, [note], c_bodies={NOTE_ID: ("note", old)}, p_bodies={NOTE_ID: ("note", old)})
    v._when_to_use()
    assert v.run.fails == []


def test_a_pass_with_fewer_items_or_a_declared_deletion_is_refused(tmp_path):
    parent = _tree(tmp_path / "p")
    child = dict(LIB)
    del child["notes/text/dates.md"]
    cand = _tree(tmp_path / "c", child)
    v = _checks(tmp_path, [], p_tree=parent, c_tree=cand, p_files=_files(parent), c_files=_files(cand))
    v._test_changes()
    g5 = [r for c, r, _ in v.run.fails if c == "G5"]
    assert g5 == ["a pass may not reduce the number of items (7 to 6); retire an item instead: it stays on record"]
    assert _refusing(v.run.fails) == [("G5", g5[0])]

    v = _checks(tmp_path, [], p_tree=parent, c_tree=parent, p_files=_files(parent), c_files=_files(parent),
                man=SimpleNamespace(items=[], deleted=["notes/text/dates.md"], unlisted=[], skeleton=[]))
    v._test_changes()
    assert [(c, r) for c, r, _ in v.run.fails] == [
        ("G5", "a pass may not delete an item (notes/text/dates.md); retire it instead: it stays on record"),
    ]
    assert _refusing(v.run.fails)


def test_a_pass_that_keeps_every_item_has_no_item_finding(tmp_path):
    parent = _tree(tmp_path / "p")
    more = {**LIB, NOTE_ID: FRONT + RULE + "\n" + WHEN}
    cand = _tree(tmp_path / "c", more)
    v = _checks(tmp_path, [], p_tree=parent, c_tree=cand, p_files=_files(parent), c_files=_files(cand))
    v._test_changes()
    assert [f for f in v.run.fails if f[0] == "G5"] == []


def _alias_checks(tmp_path: Path, raw: dict, aliases: dict) -> LibraryChecks:
    tree = _tree(tmp_path / "c")
    gate = SimpleNamespace(library_mode=LibraryConfig(episodes=lambda e: None), ev=SimpleNamespace(
        aliases=lambda: aliases))
    run = _Run(fails=[], notes=[], labels=[], manifest_raw=raw, c_tree=tree, p_tree=_tree(tmp_path / "p"),
               man=SimpleNamespace(items=[], deleted=[]))
    v = LibraryChecks(gate, run)
    v.aliases_kept()
    return v


def test_a_pass_may_not_declare_aliases_or_retirements_and_keeps_every_live_alias(tmp_path):
    v = _alias_checks(tmp_path, {"retired": {"memory.text.dates:week": "unused"}}, {})
    assert [(c, r) for c, r, _ in v.run.fails] == [("G1", "a pass may not declare aliases or retirements")]
    v = _alias_checks(tmp_path / "a", {"aliases": {"memory.text.dates:day": "memory.text.dates:parse_date"}}, {})
    assert [(c, r) for c, r, _ in v.run.fails] == [("G1", "a pass may not declare aliases or retirements")]
    # an alias earlier passes made must still forward in the candidate
    v = _alias_checks(tmp_path / "b", {}, {"memory.text.dates:old": "memory.text.dates:parse_date"})
    assert [(c, r) for c, r, _ in v.run.fails] == [
        ("G5", "alias memory.text.dates:old is no longer forwarded; a pass may not drop an alias"),
    ]
    assert _refusing(v.run.fails)
    v = _alias_checks(tmp_path / "c", {}, {})
    assert v.run.fails == [] and v.run.curations == []


def test_a_rename_names_an_item_of_the_parent_and_a_new_item_of_the_candidate(tmp_path):
    renamed = {**LIB, "memory/text/dates.py": LIB["memory/text/dates.py"].replace("def week(", "def iso_week(")}
    parent, cand = _tree(tmp_path / "p"), _tree(tmp_path / "c", renamed)

    def renames(mapping: dict) -> tuple[list, list]:
        run = _Run(fails=[], p_tree=parent, c_tree=cand)
        rows = LibraryChecks(SimpleNamespace(library_mode=None), run)._renames({"renames": mapping})
        return rows, [r for _, r, _ in run.fails]

    rows, fails = renames({"memory.text.dates:week": "memory.text.dates:iso_week"})
    assert fails == [] and rows == [
        {"item": "memory.text.dates:week", "action": "rename", "target": "memory.text.dates:iso_week", "reason": ""},
    ]
    assert renames({"memory.text.dates:gone": "memory.text.dates:iso_week"})[1] == [
        "rename of memory.text.dates:gone: it is not an item of the parent",
    ]
    assert renames({"memory.text.dates:week": "memory.text.dates:nowhere"})[1] == [
        "rename of memory.text.dates:week to memory.text.dates:nowhere: memory.text.dates:nowhere is not an item of "
        "the candidate",
    ]
    # two items become one only by an alias, never by a rename
    assert "may not merge two items" in renames({"memory.text.dates:week": "memory.text.dates:parse_date"})[1][0]
    assert renames({"memory.text.dates:week": 3})[1] == ["renames maps old item ids to new item ids (strings)"]
