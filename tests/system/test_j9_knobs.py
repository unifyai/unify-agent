"""J9: the knob census. Every environment variable or setting ``unify/`` names is in the reviewed table
(``knobs.tsv``), every row is still named, and only ``UNIFY_MEMORY`` and ``UNIFY_MEMORY_CAP_USD`` are memory
switches (the lead's hard rule H1); everything else is deployment configuration, harness-internal, test-only or not a
knob. A new knob fails until it is reviewed.
"""

from __future__ import annotations

import csv
from pathlib import Path

from tests.system.knob_scan import scan

HERE = Path(__file__).parent
ROOT = HERE.parents[1]
CLASSES = {"memory-switch", "deployment", "internal", "test-only", "not-a-knob"}


def _table() -> dict[str, dict]:
    lines = [ln for ln in (HERE / "knobs.tsv").read_text().splitlines() if ln and not ln.startswith("#")]
    return {r["name"]: r for r in csv.DictReader(lines, delimiter="\t", quoting=csv.QUOTE_NONE)}


def test_every_knob_is_reviewed_and_every_row_is_named():
    found, table = scan(ROOT), _table()
    assert not sorted(set(found) - set(table)), {k: found[k] for k in sorted(set(found) - set(table))}
    assert not sorted(set(table) - set(found)), sorted(set(table) - set(found))
    assert all(r["class"] in CLASSES for r in table.values()), [r for r in table.values() if r["class"] not in CLASSES]


def test_only_the_two_allowed_knobs_switch_memory():
    switches = {k for k, r in _table().items() if r["class"] == "memory-switch"}
    assert switches == {"UNIFY_MEMORY", "UNIFY_MEMORY_CAP_USD"}, switches
