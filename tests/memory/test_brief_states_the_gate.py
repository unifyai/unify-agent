"""The writer's brief describes the gate exactly as the gate decides: what refuses, what only labels, and that it runs
no test. A sentence that says the gate reruns tests, replays a procedure, lands a procedure only if its cases pass, or
refuses what it only labels describes behaviour that does not exist and misleads the writer; and no test result sets
an item's level, status or rank.
"""

from __future__ import annotations

import re

from unify.memory import gate_policy as gp
from unify.memory import library_prompts as lp

#: Each refusal the brief names, with a finding of that kind as the gate writes it: (check, reason, owners).
REFUSALS = {
    "safety (G6)": ("G6", "memory/a/b.py holds a key-shaped string", ["memory.a.b:c"]),
    "a change outside `memory/` and `notes/` or to the generated files": ("G1", "file INDEX.md is generated", []),
    "a manifest that does not parse or leaves a change undeclared": ("G1", "memory/a/b.py changed and not declared", []),
    "a fixture that did not come through `tools.fixture` unchanged": (
        "G1", "fixture provenance: memory/a/tests/data/x.json was not made with fixture()", ["memory.a.b:c"],
    ),
    "a cited episode that is not recorded": ("G1", "memory.a.b:c cites unknown episode e9", ["memory.a.b:c"]),
    "code that does not parse": ("G3", "memory/a/b.py does not parse", ["memory.a.b:c"]),
    "a listed item, test or case file missing from the candidate": (
        "G3", "listed test memory/a/tests/test_b.py is not in the candidate", ["memory.a.b:c"],
    ),
    "a test that imports the gate's own test kit": (
        "G3", "[library_mode:plain] memory/a/tests/test_b.py imports the gate's test kit", ["memory.a.b:c"],
    ),
}
#: Findings the brief says only label (or note), as the gate writes them.
LABELS = {
    "When to use": ("G1", "memory.a.b:c has no `When to use:` section", ["memory.a.b:c"]),
    "Rule": ("G1", "notes/a/b.md has no `## Rule` section (what memory.get serves of a note)", ["notes/a/b.md"]),
    "README": ("G1", "folder memory/a/ has no README.md: each library folder holds a short README.md", ["memory.a.b:c"]),
    "G2": ("G2", "memory.a.b:c covers cell 0 of e1, which defines no Python function", ["memory.a.b:c"]),
    "G3": ("G3", "[library_mode:exact] memory.a.b:c's tests hold no exact assertion of its result", ["memory.a.b:c"]),
    "G4": ("G4", "note: G4 hygiene due", []),
    "G5": ("G5", "a pass may not reduce the number of items (3 to 2)", []),
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
    "the gate refuses a new or changed",
    "the gate refuses a candidate",
    "the gate refuses the items",
    "has passed the gate",
    "does not parse or import",
)


def _flat(text: str) -> str:
    return " ".join(text.split())


def test_each_refusal_the_brief_names_is_a_hard_check_and_nothing_else_is():
    gate = _flat(lp.LABELS_ONLY_GATE)
    named = gate.split("Only these refuse: ", 1)[1].split(". The gate runs no code", 1)[0]
    for phrase, (check, reason, owners) in REFUSALS.items():
        assert phrase in named, phrase
        assert gp.hard(check, reason, owners), (phrase, check, reason)
    for name, (check, reason, owners) in LABELS.items():
        assert not gp.hard(check, reason, owners), (name, check, reason)


def test_the_brief_says_the_gate_runs_no_test_and_no_test_result_sets_trust():
    for brief in (lp.write_brief_now(helper=False), lp.write_brief_now(helper=True)):
        flat = _flat(brief)
        low = flat.lower()
        assert not [w for w in NEVER if w.lower() in low], [w for w in NEVER if w.lower() in low]
        assert "The gate runs no code: no item, test or recipe is run" in flat
        assert "no test result, yours or one recorded on a label, sets an item's level, status or rank" in flat
        # every sentence about the gate refusing names one of the hard checks or the repair round
        for sentence in re.split(r"(?<=[.;])\s+", flat):
            if re.search(r"\bgate refuses\b", sentence, re.I):
                assert "repair" in sentence.lower(), sentence
