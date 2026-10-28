"""The writer's brief describes the gate exactly as the gate decides: what refuses, what only labels, and that it runs
no test. A sentence that says the gate reruns tests, replays a procedure, lands a procedure only if its cases pass, or
refuses what it only labels describes behaviour that does not exist and misleads the writer; and no test result sets
an item's level, status or rank. The brief's gate section is built from :data:`unify.memory.gate_policy.GATE`, the
object :func:`unify.memory.gate_policy.hard` decides with; ``test_gate_behaviour_table.py`` checks real merges.
"""

from __future__ import annotations

import re

from unify.memory import gate_policy as gp
from unify.memory import library_prompts as lp

#: Each refusal the brief names, with a finding of that kind as the gate writes it: (check, reason, owners).
REFUSALS = {
    "safety (G6)": ("G6", "key-shaped string in memory/a/b.py", []),
    "a finding about the pass as a whole": ("G1", "undeclared change memory/a/b.py", []),
    "a fixture that did not come through `tools.fixture` unchanged": (
        "G1", "fixture provenance: memory/a/tests/data/x.json was not made with fixture()", ["memory.a.b:c"],
    ),
    "an item that names no source episode, or cites one that is not recorded": (
        "G1", "memory.a.b:c cites unknown episode e9", ["memory.a.b:c"],
    ),
    "a new or changed note without a `## Rule` section with text": (
        "G1", "notes/a/b.md has no `## Rule` section (what memory.get serves of a note)", ["notes/a/b.md"],
    ),
    "a pass with fewer items than the library it started from, or a dropped alias (G5)": (
        "G5", "a pass may not reduce the number of items (3 to 2)", [],
    ),
    "code that does not parse": ("G1", "memory/a/b.py does not parse (line 3)", ["memory.a.b:c"]),
    "a listed item, test or case file missing from the candidate": (
        "G3", "listed test memory/a/tests/test_b.py is not in the candidate", ["memory.a.b:c"],
    ),
}
#: Findings the brief says only label (or note), as the gate writes them.
LABELS = {
    "When to use": ("G1", "memory.a.b:c has no `When to use:` section", ["memory.a.b:c"]),
    "README": ("G1", "folder memory/a/ has no README.md: each library folder holds a short README.md", ["memory.a.b:c"]),
    "dangling link": ("G1", "dangling link: notes/a/b.md links to memory.a.b:x (uses)", ["notes/a/b.md"]),
    "G2": ("G2", "memory.a.b:c covers cell 0 of e1, which defines no Python function", ["memory.a.b:c"]),
    "failed source": ("G2", gp.FAILED_SOURCE_LABEL.format(item="memory.a.p:f", episodes="e1"), ["memory.a.p:f"]),
    "no recipe": ("G2", gp.NO_RECIPE_LABEL.format(item="memory.a.p:f"), ["memory.a.p:f"]),
    "G3": ("G3", "[library_mode:exact] memory.a.b:c's tests hold no exact assertion of its result", ["memory.a.b:c"]),
    "fewer tests": ("G3", "the pass has fewer tests than its parent (4 to 3)", []),
    "G4": ("G4", "note: G4 hygiene due", []),
}
#: Wording of behaviour the gate does not have.
NEVER = (
    "reruns the tests",
    "unseen values",
    "lands the procedure only if",
    "still refuse",
    "judged by its own tests",
    "verified by replaying",
    "cannot be verified, so it stays a draft",
    "the gate refuses a candidate",
    "the gate refuses the items",
    "has passed the gate",
    "does not parse or import",
    "never removes library items and never deletes or weakens a test",
)


def _flat(text: str) -> str:
    return " ".join(text.split())


def test_each_refusal_the_brief_names_is_a_hard_check_and_nothing_else_is():
    gate = _flat(lp.gate_text())
    named = gate.split("Only these refuse: ", 1)[1].split(". The gate runs no code", 1)[0]
    assert len(REFUSALS) == len(gp.GATE.refusals)
    for phrase, (check, reason, owners) in REFUSALS.items():
        assert phrase in named, phrase
        assert gp.hard(check, reason, owners), (phrase, check, reason)
    for words in gp.GATE.refusal_words():
        assert _flat(words) in named, words
    for name, (check, reason, owners) in LABELS.items():
        assert not gp.hard(check, reason, owners), (name, check, reason)


def test_the_brief_says_the_gate_runs_no_test_and_no_test_result_sets_trust():
    assert gp.GATE.runs == ()
    for brief in (lp.write_brief_now(helper=False), lp.write_brief_now(helper=True)):
        flat = _flat(brief)
        low = flat.lower()
        assert not [w for w in NEVER if w.lower() in low], [w for w in NEVER if w.lower() in low]
        assert "The gate runs no code: no item, test or recipe is run" in flat
        assert "no test result, yours or one recorded on a label, sets an item's level, status or rank" in flat
        assert _flat(lp.gate_text()) in flat
        # every sentence about the gate refusing names one of the hard checks or the repair round
        for sentence in re.split(r"(?<=[.;])\s+", flat):
            if re.search(r"\bgate refuses\b", sentence, re.I):
                assert "repair" in sentence.lower(), sentence


def test_rule_four_and_the_note_rule_say_what_the_gate_does():
    flat = _flat(lp.write_brief_now(helper=True))
    assert "A new or changed note without a `## Rule` section with text is refused." in flat
    assert (
        "a candidate with fewer items than the library it started from, a deleted item or a dropped alias is refused"
        in flat
    )
    assert "or a candidate with fewer tests, is labelled so" in flat
    assert "name it in `tests_changed` with the reason" in flat
    assert "the item stays on record with its file" in flat
    # the When to use and README findings only label, as the gate decides
    assert "without the section or one of its four lines is labelled so" in flat
    assert "are labelled when the folder has no README.md" in flat
