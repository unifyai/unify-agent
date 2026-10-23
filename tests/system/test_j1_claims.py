"""J1: the reviewed claims registry. Every in-scope sentence an agent is shown (the gate, library and input paths,
cards and recorded procedures, the librarian and its answer, and memory words in a memory-off text:
:func:`tests.system.render.scoped`) is in ``claims.tsv``, reviewed: a claim with the test that exercises it on the real
harness, a known-false claim with its audit id and the test that fails on it, or an instruction, example or format. A
new or changed sentence fails until it is reviewed; a factual sentence with no checker fails until one is written
(``untestable``: a claim about a configuration no run of this harness can have, with the reason in its note).
"""

from __future__ import annotations

import ast
import csv
from pathlib import Path

from tests.system.render import scoped

HERE = Path(__file__).parent
STATUSES = {"checked", "false", "nonfactual", "unchecked", "untestable"}


def _registry() -> dict[str, dict]:
    lines = [ln for ln in (HERE / "claims.tsv").read_text().splitlines() if ln and not ln.startswith("#")]
    rows = list(csv.DictReader(lines, delimiter="\t", quoting=csv.QUOTE_NONE))
    return {r["sentence"]: r for r in rows}


def _test_names() -> dict[str, set[str]]:
    out = {}
    for p in HERE.glob("test_*.py"):
        tree = ast.parse(p.read_text())
        out[f"tests/system/{p.name}"] = {n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    return out


def test_every_in_scope_sentence_is_reviewed(rendered):
    reg = _registry()
    now = scoped(rendered)
    new = sorted(s for s in now if s not in reg)
    gone = sorted(s for s in reg if s not in now)
    assert not new and not gone, {"unreviewed (new or changed)": new[:30], "stale rows (no longer shown)": gone[:30]}


def test_every_row_has_a_known_status_and_an_existing_checker():
    names = _test_names()
    bad = []
    for s, r in _registry().items():
        if r["status"] not in STATUSES:
            bad.append((r["status"], s[:80]))
        for chk in filter(None, r["checker"].split()):
            path, _, fn = chk.partition("::")
            if path not in names or (fn and fn not in names[path]):
                bad.append(("no such checker", chk))
        if r["status"] == "checked" and not r["checker"]:
            bad.append(("checked with no checker", s[:80]))
    assert not bad, bad[:20]


def test_no_factual_claim_is_unchecked():
    unchecked = [f"{r['roles']}: {s[:140]}" for s, r in _registry().items() if r["status"] == "unchecked"]
    assert not unchecked, f"{len(unchecked)} factual sentences have no checker:\n" + "\n".join(unchecked)
