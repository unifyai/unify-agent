"""What the episode-end fork and the kinship judge are shown is true for them.

* The fork's part list marks a cell "raised an error" from the cell's recorded error status, never from a word of its
  printed output.
* The fork's instruction points to the record's section where it is (its first message), and keeps no wording that
  steers what is kept.
* The judge is shown behaviour only: a capture's code without its docstrings and comments (a docstring may quote the
  record, so it may carry request or environment text), its calls, its input key paths and its outcome.
"""

from __future__ import annotations

from types import SimpleNamespace

from tests.memory.test_episodes import _ep
from unify.memory import kinship
from unify.memory.episodes import Cell
from unify.memory.integration import fork as fork_mod


def test_a_cell_that_prints_a_traceback_without_raising_is_not_marked_raised():
    ep = _ep(
        episode_id="e1",
        request=["Show what an error looks like."],
        cells=[
            Cell(0, "print(text)\n", "Traceback (most recent call last):\n  File x\nValueError: shown\n"),
            Cell(1, "int('x')\n", "", "ValueError: invalid literal for int() with base 10: 'x'"),
        ],
        actions=[],
    )
    lines = fork_mod.part_list(ep).splitlines()
    assert "- `cell:0`: print(text)" in lines
    assert "- `cell:1`: int('x')  (raised an error)" in lines


def test_the_fork_instruction_names_where_the_record_section_is_and_does_not_steer_what_is_kept():
    flat = " ".join(fork_mod.fork_instruction().split())
    assert "the record's section in your first message" in flat and "section below" not in flat
    assert "could be reused" not in flat.lower()
    assert "one of the `run(state, env)` form is not run that way" in flat


QUOTE = "The environment replied: ORDER 7731 ACCEPTED for the customer"
PROCEDURE = f'''"""Price an order line.

The record shows the work accepted: "{QUOTE}".
"""


def run(value):
    """Quantity times unit price; accepted ("{QUOTE}")."""
    # the request said: {QUOTE}
    return value["quantity"] * value["unit"]
'''


def test_the_judge_sees_code_without_its_docstrings_or_comments(tmp_path):
    d = tmp_path / "cap"
    d.mkdir()
    (d / "procedure.py").write_text(PROCEDURE, encoding="utf-8")
    (d / "saved.json").write_text('{"docstring": "%s"}' % QUOTE, encoding="utf-8")
    profile = SimpleNamespace(sequence=("env.orders.price",), input=frozenset({"quantity:int"}), outcome="positive")
    card = kinship.card(profile, d, code_chars=kinship.NEW_CODE)
    assert QUOTE not in card and "as its author described it" not in card
    assert "return value['quantity'] * value['unit']" in card
    assert "calls in order: env.orders.price" in card and "outcome of its work: positive" in card
    messages, labels = kinship.question((profile, d), [("x1", profile, d)])
    assert QUOTE not in messages[1]["content"] and labels == {"[1]": "x1"}
    assert "one piece of finished work" not in kinship.BRIEF


def _long_procedure(statements: int, comments: int) -> str:
    """A procedure with *statements* lines of code in ``run`` and *comments* comment lines before it."""
    lines = ['"""Add the values up, one step at a time."""', ""]
    lines += [f"# step {i}: every value is kept and only the total changes." for i in range(comments)]
    lines += ["", "", "def run(value):", '    """The total."""', "    total = 0"]
    lines += [f"    total = total + value[{i}]" for i in range(statements)]
    lines += ["    return total", ""]
    return "\n".join(lines)


def test_the_judge_is_shown_bounded_code_and_told_where_it_was_cut(tmp_path):
    """The bounds apply to the code the judge is shown, docstrings and comments already left out: long code is cut at
    the bound with a marker, while comments alone never make code long."""
    profile = SimpleNamespace(sequence=(), input=frozenset(), outcome="positive")
    long_code = tmp_path / "long"
    long_code.mkdir()
    (long_code / "procedure.py").write_text(_long_procedure(statements=80, comments=0), encoding="utf-8")
    commented = tmp_path / "commented"
    commented.mkdir()
    (commented / "procedure.py").write_text(_long_procedure(statements=3, comments=200), encoding="utf-8")
    shown = kinship.card(profile, long_code, code_chars=kinship.BRIEF_CODE).split("code:\n", 1)[1]
    assert shown.endswith("\n# ... (the rest of the code is not shown)")
    assert len(shown) == kinship.BRIEF_CODE + len("\n# ... (the rest of the code is not shown)")
    whole = kinship.card(profile, long_code, code_chars=kinship.NEW_CODE).split("code:\n", 1)[1]
    assert "not shown" not in whole and whole.endswith("return total")
    short = kinship.card(profile, commented, code_chars=kinship.BRIEF_CODE).split("code:\n", 1)[1]
    assert "not shown" not in short and "step 199" not in short and short.endswith("return total")
    assert (kinship.CANDIDATES, kinship.NEW_CODE, kinship.BRIEF_CODE) == (40, 4000, 600)
