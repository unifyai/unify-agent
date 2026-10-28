"""Trust, serving and ranking take no input from a test result or a judgement of an item's code.

Whether code is pure, acts, reads its input or gives a fixed answer cannot be told reliably, so nothing classifies it;
and trust comes only from real uses whose outcome confirmed an item, plus its source episodes. So what the actor and
the librarian are served (the package's files, the librarian's catalogue, its structural rows and its search rows) is
byte for byte the same whatever an item's tests gave, whatever the gate found, whatever effect or evidence level a
test run once recorded; and the functions that decide trust and status read none of those fields. The writer's own
tests stay its authoring aid: they may run in its box, and their results reach only the writer.
"""

from __future__ import annotations

import inspect

from tests.memory.test_layout import _tree
from unify.memory import item_records as ir
from unify.memory import library_export as lx
from unify.memory import library_search, lifecycle, serving, trust
from unify.memory.layout import discover

#: Everything a test run, the gate or a judgement of the code could have put on an item's record.
FROM_TESTS = {
    "label": {
        "tests": {"as_stated": 0, "not_as_stated": 5, "error": 2, "timeout": 0, "not_run": 0},
        "findings": ["G3: its tests hold no exact assertion", "not run: the gate runs no code"],
    },
    "level": "reproduces-its-case",
    "level_reasons": ["G3: its tests hold no exact assertion", "examples: special-cased"],
    "verification": {
        "tests": ["memory/text/tests/test_dates.py"],
        "exact_assertions": 0,
        "examples": {"calls": 3, "grounded_calls": 3, "episodes": ["e1"]},
        "mutation": {"killed": 0, "total": 4},
        "level": "reproduces-its-case",
    },
    "effect": "write",
}


def _records(tree, tested: bool) -> dict:
    lib = discover(tree)
    out = {}
    for f in lib.functions:
        rec = {**ir.empty_record(f.item_id, "function"), "provenance": {"episodes": ["e1"]}, "level": "unchecked",
               "label": {"uses": {"clean": 1, "failed": 0, "neutral": 0}}}  # fmt: skip
        if tested:
            rec.update({k: v for k, v in FROM_TESTS.items() if k != "label"})
            rec["label"] = {**rec["label"], **FROM_TESTS["label"]}
        out[f.item_id] = rec
    for n in lib.notes:
        out[n.item_id] = ir.empty_record(n.item_id, "note")
    return out


def test_what_is_served_is_the_same_whatever_the_tests_said(tmp_path):
    tree = _tree(tmp_path / "lib")
    plain, tested = _records(tree, False), _records(tree, True)
    assert plain != tested
    lib = discover(tree)
    for build in (
        lambda recs: lx.generated_library_mode(tree, records=recs, librarian=True),
        lambda recs: lx.catalogue_text(tree, lib, records=recs),
        lambda recs: lx.dispatch_data(lib, records=recs),
        lambda recs: library_search.rows(tree, lib, records=recs),
    ):
        assert build(plain) == build(tested)


def test_trust_and_status_read_no_test_result_and_no_code_judgement():
    """The functions that judge uses, decide status and build the served evidence take no test result, level reason,
    effect, purity or fixed-answer input, by their parameters or in their code."""
    banned = ("not_as_stated", "level_reasons", "label_findings", "exact_assertions", "fixed_answer", "reads_input",
              "purity", "worked_example", "\"effect\"", "grounded_calls")  # fmt: skip
    for fn in (trust.judge, trust.decide, trust.apply_outcome, trust.status_basis, lifecycle.decide_status,
               serving.item_evidence, serving.evidence_doc, serving.withdrawn_items):  # fmt: skip
        params = set(inspect.signature(fn).parameters)
        assert not params & {"tests", "labels", "label", "level_reasons", "effect", "pure", "constant"}, (fn, params)
        source = inspect.getsource(fn)
        assert not [b for b in banned if b in source], (fn.__qualname__, [b for b in banned if b in source])
