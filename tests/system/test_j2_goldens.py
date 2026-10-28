"""J2: the rendered goldens. Every static text each role is shown, per mode (tool lists, tool help, refusal templates,
the common section, the memory guide and block head, the briefs, ``help(memory)``, card templates), equals its
golden; the golden is the design-true text (:mod:`.goldens`), so a text the audit found false fails here until it is
fixed, and any other change fails until it is reviewed and re-recorded (``python -m tests.system.record_goldens``).
"""

from __future__ import annotations

import difflib
import json
import re
from pathlib import Path

import pytest

from tests.system.goldens import CORRECTIONS, GOLDENS

GOLDEN = json.loads((Path(__file__).parent / GOLDENS).read_text())
GROUPS = sorted({"/".join(k.split("/")[:2]) for k in GOLDEN})


def test_the_rendered_texts_are_the_recorded_ones(rendered):
    assert sorted(rendered) == sorted(GOLDEN), {
        "new": sorted(set(rendered) - set(GOLDEN))[:20], "gone": sorted(set(GOLDEN) - set(rendered))[:20]}


@pytest.mark.parametrize("group", GROUPS)
def test_each_text_reads_as_its_golden(rendered, group):
    keys = [k for k in GOLDEN if k.startswith(group + "/") and rendered.get(k) != GOLDEN[k]]
    diffs = []
    for k in keys:
        d = difflib.unified_diff(GOLDEN[k].splitlines(), str(rendered.get(k, "<absent>")).splitlines(),
                                 "golden", "rendered", n=0, lineterm="")
        diffs.append(f"== {k}\n" + "\n".join(list(d)[2:12]))
    assert not keys, "\n".join(diffs)


@pytest.mark.parametrize("c", CORRECTIONS, ids=[f"{c.id}:{c.span[:40]}" for c in CORRECTIONS])
def test_no_text_holds_a_span_the_audit_found_false(rendered, c):
    rx = re.compile(c.keys)
    hits = [k for k, v in rendered.items() if rx.search(k) and c.span in v]
    assert not hits, (c.id, c.span, hits[:8])
