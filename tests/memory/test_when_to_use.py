"""The "when to use" section of every stored function and note, and the README of every library folder.

The writer writes both; the library-mode gate (G1) checks their shape only: the section is there with its four
labelled lines, each with some text, and each folder a pass changes holds a README.md with some text. The wording
is never judged.
"""

import ast
import textwrap

from unify.memory import docstrings, layout, when_to_use
from unify.memory.library_prompts import WHEN_TO_USE_TEXT
from tests.memory.test_gate_library_checks import _checks, _item

SECTION = """\
When to use:
    Saved for: totalling the amounts of payment rows per currency.
    Recognise: the request gives rows with an amount and a currency
        and asks for totals.
    Not for: totals per date.
    Example request: Sum these payments per currency."""

FUNCTION = f'''def total(rows):
    """Total the amounts per currency.

    Args:
        rows: the payment rows.

    Returns:
        A dict of totals.

    Raises:
        MemoryInputError: on rows without an amount.

    Example: memory/calc/tests/test_sums.py::test_total

    {SECTION.replace(chr(10), chr(10) + "    ")}
    """
    return {{}}
'''

NOTE = """---
title: Totals per currency
description: how to total payments
uses: []
---
## Rule
Group the rows first.

## When to use
- **Saved for:** totalling payments per currency.
- **Recognise:** rows with amounts and currencies.
- **Not for:** totals per date.
- **Example request:** Sum these payments per currency.

## Pitfalls
Amounts may be strings.
"""


def _doc(source: str) -> str:
    return ast.get_docstring(ast.parse(source).body[0]) or ""


def test_a_full_section_has_no_problem_and_keeps_the_example_section_whole():
    doc = _doc(FUNCTION)
    assert when_to_use.function_problems(doc) == []
    found, twice = when_to_use.parts(when_to_use.docstring_section(doc))
    assert twice == []
    assert found["Recognise"] == "the request gives rows with an amount and a currency and asks for totals."
    # parsed as a section of its own: the Example: line before it stays one line
    assert docstrings.parse(doc).sections["Example"] == "memory/calc/tests/test_sums.py::test_total"


def test_each_missing_or_empty_part_is_its_own_reason():
    assert when_to_use.function_problems("Total.\n\nArgs:\n    rows: rows.") == [
        f"has no `When to use:` section ({when_to_use.SHAPE})",
    ]
    assert when_to_use.function_problems("Total.\n\nWhen to use:\n") == [
        "`When to use:` section has no `Saved for:` line",
        "`When to use:` section has no `Recognise:` line",
        "`When to use:` section has no `Not for:` line",
        "`When to use:` section has no `Example request:` line",
    ]
    partial = (
        "Total.\n\nWhen to use:\n    Saved for: totals.\n    Recognize: rows.\n    Not for: ...\n"
        "    Saved for: again.\n\nNotes: notes/a/b.md"
    )
    assert when_to_use.function_problems(partial) == [
        "`When to use:` section gives `Saved for:` more than once",
        "`When to use:` section has an empty `Not for:` line",
        "`When to use:` section has no `Example request:` line",
    ]


def test_a_note_holds_the_section_under_its_heading_up_to_the_next():
    assert when_to_use.note_problems(NOTE) == []
    assert "Amounts may be strings" not in when_to_use.note_section(NOTE)
    plain = NOTE.replace("- **", "").replace(":**", ":").replace("## When to use", "### When to use:")
    assert when_to_use.note_problems(plain) == []
    without = NOTE.replace("## When to use", "## Use")
    assert when_to_use.note_problems(without) == [
        f"has no `## When to use` section ({when_to_use.SHAPE})",
    ]
    cut = NOTE.replace("- **Example request:** Sum these payments per currency.\n", "")
    assert when_to_use.note_problems(cut) == ["`## When to use` section has no `Example request:` line"]


def test_folders_and_their_readmes():
    assert when_to_use.folder("memory/calc/sums.py") == "memory/calc"
    assert when_to_use.folder("memory/calc/tests/data/rows.json") == "memory/calc"
    assert when_to_use.folder("notes/payments/totals.md") == "notes/payments"
    assert when_to_use.folder("INDEX.md") is None
    assert when_to_use.readme_problem(None) == "has no README.md"
    assert when_to_use.readme_problem(b"  \n# \n") == "has an empty README.md"
    assert when_to_use.readme_problem(b"Payment totals.\n") is None


def test_the_layout_places_a_readme_in_each_folder_and_exports_it():
    for path in ("memory/calc/README.md", "notes/payments/README.md"):
        assert layout.classify(path) == "readme" and layout.misplaced(path) is None
        assert layout.exported(path)
    assert layout.classify("README.md") is None  # the root's is the harness's, never the writer's
    assert layout.classify("memory/calc/tests/README.md") is None
    assert layout.classify("notes/Bad Topic/README.md") is None
    assert layout.classify("notes/payments/readme.md") == "note"  # a note named readme, as before


def _function_run(tmp_path, source: str, files: dict[str, bytes], parent: str | None = None):
    item = "memory.calc.sums:total"
    body = ast.unparse(ast.parse(source).body[0])
    c_tree = tmp_path / "c"
    for rel, data in files.items():
        (c_tree / rel).parent.mkdir(parents=True, exist_ok=True)
        (c_tree / rel).write_bytes(data)
    return _checks(
        tmp_path,
        [_item(item=item)],
        changed=sorted(files),
        c_files={p: "" for p in files},
        p_bodies={} if parent is None else {item: ("function", parent, True)},
        c_bodies={item: ("function", body, True)},
    )


def test_the_gate_refuses_a_new_function_without_the_section(tmp_path):
    files = {"memory/calc/sums.py": b"", "memory/calc/README.md": b"Sums of rows.\n"}
    bare = FUNCTION.split("When to use:")[0] + '"""\n    return {}\n'
    v = _function_run(tmp_path, bare, files)
    v._when_to_use()
    assert v.run.fails == [
        (
            "G1",
            f"memory.calc.sums:total has no `When to use:` section ({when_to_use.SHAPE})",
            "memory.calc.sums:total",
        ),
    ]
    ok = _function_run(tmp_path, FUNCTION, files)
    ok.g1()
    assert ok.run.fails == []


def test_an_unchanged_function_is_not_judged(tmp_path):
    bare = FUNCTION.split("When to use:")[0] + '"""\n    return {}\n'
    body = ast.unparse(ast.parse(bare).body[0])
    v = _function_run(tmp_path, bare, {"memory/calc/sums.py": b""}, parent=body)
    v._when_to_use()
    assert v.run.fails == []


def test_a_new_note_without_the_section_is_refused(tmp_path):
    item = "notes/payments/totals.md"
    text = NOTE.replace("## When to use", "## Use")
    v = _checks(
        tmp_path,
        [_item(item=item, kind="note")],
        c_bodies={item: ("note", text, True)},
    )
    v._when_to_use()
    assert v.run.fails == [
        ("G1", f"{item} has no `## When to use` section ({when_to_use.SHAPE})", item),
    ]
    v = _checks(tmp_path, [_item(item=item, kind="note")], c_bodies={item: ("note", NOTE, True)})
    v._when_to_use()
    assert v.run.fails == []


def test_a_new_or_changed_note_without_a_rule_section_is_refused(tmp_path):
    """``memory.get`` serves a note's front matter and its ``## Rule`` section: the gate requires one with text."""
    assert when_to_use.rule_problems(NOTE) == []
    item = "notes/payments/totals.md"
    for text, why in (
        (NOTE.replace("## Rule\n", ""), "has no `## Rule` section"),
        (NOTE.replace("Group the rows first.\n", ""), "has an empty `## Rule` section"),
    ):
        v = _checks(tmp_path, [_item(item=item, kind="note")], c_bodies={item: ("note", text, True)})
        v._when_to_use()
        assert [(c, r.split(" (")[0], i) for c, r, i in v.run.fails] == [("G1", f"{item} {why}", item)]


def test_a_changed_folder_without_a_readme_refuses_its_items(tmp_path):
    files = {"memory/calc/sums.py": b"", "notes/payments/totals.md": NOTE.encode()}
    v = _function_run(tmp_path, FUNCTION, files)
    v.run.man.items.append(_item(item="notes/payments/totals.md", kind="note"))
    v._readmes()
    reasons = {(r, i) for _, r, i in v.run.fails}
    assert {i for _, i in reasons} == {"memory.calc.sums:total", "notes/payments/totals.md"}
    assert all("has no README.md" in r for r, _ in reasons)


def test_a_readme_only_change_or_an_emptied_folder_needs_no_item(tmp_path):
    files = {"memory/calc/README.md": b"   \n"}
    v = _function_run(tmp_path, FUNCTION, files)
    v.run.man.items.clear()
    v._readmes()
    assert [(c, i) for c, _, i in v.run.fails] == [("G1", None)]
    assert "has an empty README.md" in v.run.fails[0][1]
    gone = _checks(tmp_path, [], changed=["memory/old/sums.py"], c_files={})
    gone._readmes()
    assert gone.run.fails == []


def test_the_writer_is_told_the_shape_the_gate_checks():
    flat = " ".join(WHEN_TO_USE_TEXT.split())
    for label in when_to_use.PARTS:
        assert f"`{label}:`" in flat
    assert "`When to use:`" in flat and "`## When to use`" in flat and "README.md" in flat
    from unify.memory.library_prompts import _WHAT_TO_WRITE_IN_CODE

    notes = " ".join(_WHAT_TO_WRITE_IN_CODE.split())
    assert "and a `## Rule` section" in notes and "without a `## Rule` section with text" in notes
    example = WHEN_TO_USE_TEXT.split("For example:\n", 1)[1].split("  The gate", 1)[0]
    assert when_to_use.function_problems("Summary.\n\n" + textwrap.dedent(example)) == []
