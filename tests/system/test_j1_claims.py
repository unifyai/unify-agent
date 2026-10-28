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
import json
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


def _recorded() -> dict[str, str]:
    texts: dict[str, str] = {}
    for p in sorted((HERE / "goldens").glob("first_requests-*.recorded.json")):
        texts.update(json.loads(p.read_text()))
    return texts


def test_every_in_scope_sentence_is_reviewed(rendered):
    reg = _registry()
    now = scoped(rendered)
    shown = set(now) | set(scoped(_recorded()))  # a row may come from a first request only
    new = sorted(s for s in now if s not in reg)
    gone = sorted(s for s in reg if s not in shown)
    assert not new and not gone, {"unreviewed (new or changed)": new[:30], "stale rows (no longer shown)": gone[:30]}


def test_every_row_has_a_known_status_and_an_existing_checker():
    names = _test_names()
    bad = []
    for s, r in _registry().items():
        if r["status"] not in STATUSES:
            bad.append((r["status"], s[:80]))
        for chk in filter(None, r["checker"].split()):
            path, _, fn = chk.partition("::")
            if path not in names and (HERE.parents[1] / path).is_file():  # a checker outside the system suite
                tree = ast.parse((HERE.parents[1] / path).read_text())
                names[path] = {n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
            if path not in names or (fn and fn not in names[path]):
                bad.append(("no such checker", chk))
        if r["status"] == "checked" and not r["checker"]:
            bad.append(("checked with no checker", s[:80]))
    assert not bad, bad[:20]


def test_no_factual_claim_is_unchecked():
    unchecked = [f"{r['roles']}: {s[:140]}" for s, r in _registry().items() if r["status"] == "unchecked"]
    assert not unchecked, f"{len(unchecked)} factual sentences have no checker:\n" + "\n".join(unchecked)


def test_every_in_scope_sentence_of_a_first_request_is_reviewed():
    """The same for each role's first request in a real run (the recorded first-request goldens, normalised): every
    in-scope sentence of them is in the registry."""
    texts = _recorded()
    assert texts, "no recorded first requests"
    reg = _registry()
    new = sorted(s for s in scoped(texts) if s not in reg)
    assert not new, {"unreviewed (new or changed)": new[:30]}
