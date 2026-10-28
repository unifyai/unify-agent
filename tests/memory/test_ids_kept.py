"""Rev 2.1, fix 1: no mechanism strips, masks or placeholders an identifier.

Rev 2 replaced identifier-shaped tokens (and values a request labelled with words such as ``task`` or ``ticket``) by
``<id>`` in signatures, goals, docstrings, catalogue rows, the librarian's question and its notes. Both sides of a
comparison then read ``<id>``, so an item keyed on one request's identifier fitted every request. Real work carries
meaningful references (invoice and ticket numbers, dates, versions, account ids); they are now kept as written
everywhere, and the librarian sees the same text the item carries.
"""

from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path

from unify.memory import captures
from unify.memory import librarian as lib_mod

ROOT = Path(__file__).resolve().parents[2] / "unify"
#: References of several worlds' forms: a ticket, a date, an amount, a UUID and a hex digest.
REFS = ("TKT-004211", "2026-10-22", "100000", "3f2b9c1d-77aa-4b0e-9a1c-0d5e2f6a7b8c", "a1b2c3d4e5f6")


def test_the_identifier_rule_is_gone():
    assert importlib.util.find_spec("unify.memory.signatures") is None
    names = re.compile(r"\b(strip_identifiers|printed_ids|ID_MARK|ID_LABELS|default_strip|strip_signature)\b")
    hits = [
        f"{p.relative_to(ROOT)}:{n}"
        for p in sorted(ROOT.rglob("*.py"))
        for n, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1)
        if names.search(line)
    ]
    assert hits == []


def test_a_shape_keeps_every_key_as_written():
    value = {REFS[0]: {"amount": 3}, REFS[3]: {"amount": 5}, "items": [1, 2]}
    sig = captures.signature(value, None, returned=False, applies=None, holds=None, digests={})
    tree = json.dumps(sig["input"])
    assert REFS[0] in tree and REFS[3] in tree and "<id>" not in tree


def test_the_goal_and_what_a_capture_was_saved_against_keep_their_references():
    asked = f"Refund {REFS[0]} dated {REFS[1]} for {REFS[2]} (ref {REFS[3]}, digest {REFS[4]})."
    summary = f"## Asked\n{asked}\n## Approach\n- looked up {REFS[0]}\n"
    source = f'def run(state, env):\n    """Refunds {REFS[0]}."""\n    return None\n'
    assert captures.asked(summary) == asked
    saved = captures.saved_against(source, summary)
    assert saved == {"docstring": f"Refunds {REFS[0]}.", "asked": asked, "approach": f"- looked up {REFS[0]}"}
    sig = captures.signature(None, None, returned=False, applies=None, holds=None, digests={}, goal=asked)
    assert sig["goal"] == asked


def test_the_librarian_reads_the_question_and_the_situation_as_written():
    question = f"Which item handles {REFS[0]} from {REFS[1]}?"
    for evidence in (None, {}):
        sent = lib_mod.sent_to_librarian(
            question, {"commit": "c", "dispatch": "d"}, evidence=evidence,
            last_message=f"The customer wrote about {REFS[0]}.", rows=3,
        )  # fmt: skip
        assert question in sent["message"] and "<id>" not in sent["message"]
        # the same text, the same key: nothing is folded before the digest
        again = lib_mod.sent_to_librarian(
            question.replace(REFS[0], "TKT-004212"), {"commit": "c", "dispatch": "d"}, evidence=evidence,
            last_message=f"The customer wrote about {REFS[0]}.", rows=3,
        )  # fmt: skip
        assert again["message"] != sent["message"]
