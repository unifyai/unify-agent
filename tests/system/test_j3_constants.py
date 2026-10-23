"""J3: the numbers and rules an agent is told come from the constants that enforce them, and no text says the gate
runs anything (the gate runs no code: ``gate_policy``, the lead's ruling of 23 Oct).
"""

from __future__ import annotations

import re

import pytest

WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "eight": 8}


def _num(s: str) -> int:
    return WORDS[s] if s in WORDS else int(s.replace(",", ""))


#: "the gate runs/re-runs/replays …" or "… lands … only if …": a description of a gate that runs code.
GATE_RUNS = re.compile(r"\b[Tt]he gate (?:then )?(?:re-?runs|runs|replays|executes)\b(?! no\b| neither\b)"
                       r"|\blands?\b[^.]{0,60}\bonly if\b")


def test_no_text_says_the_gate_runs_code(rendered):
    hits = sorted({(k.split("/")[0], m.group(0), v[max(0, m.start() - 60):m.end() + 60].replace("\n", " "))
                   for k, v in rendered.items() for m in GATE_RUNS.finditer(v)})
    assert not hits, hits


def _stated(rendered: dict, pattern: str, keys: str = ".") -> list[tuple[str, int]]:
    rx, kx = re.compile(pattern), re.compile(keys)
    return [(k, _num(m.group(1))) for k, v in rendered.items() if kx.search(k) for m in rx.finditer(v)]


def _cases():
    from unify.common import tool_registry as R
    from unify.memory import librarian as LB
    from unify.memory import library_helper as LH
    from unify.memory import messaging as MS

    return [
        ("reader calls per model turn", r"reader calls per model turn: (\d+)", R.READERS_PER_TURN),
        ("read_episode parts per call", r"read_episode parts per call: (\d+)", R.READ_PARTS_PER_CALL),
        ("cells per model turn", r"cells per model turn: (\d+)", R.CELLS_PER_TURN),
        ("fits at most", r"`fits` holds at most (\w+)", LB.MAX_FITS),
        ("helpers at once", r"at most (\d+) at once and \d+", MS.RUN_PASS_OPTIONS.max_live),
        ("helpers in all", r"at most \d+ at once and (\d+)", MS.RUN_PASS_OPTIONS.max_total),
        ("a note's Rule as served", r"## Rule` section, at most ([\d,]+) characters", LH.NOTE_RULE_CHARS),
    ]


@pytest.mark.parametrize("i", range(7))
def test_each_stated_number_is_its_constant(rendered, i):
    name, pattern, value = _cases()[i]
    stated = _stated(rendered, pattern)
    assert stated, f"{name}: no text states it (the pattern {pattern!r} matched nothing)"
    wrong = [(k, n) for k, n in stated if n != value]
    assert not wrong, (name, value, wrong[:6])


def test_the_pass_limits_table_is_the_enforced_constants():
    from unify.common import tool_registry as R
    from unify.memory import agent_profiles as AP

    table = {r["limit"]: r["value"] for r in AP._PASS_LIMITS}
    assert table["reader calls per model turn"] == R.READERS_PER_TURN, table
    assert table["read_episode parts per call"] == R.READ_PARTS_PER_CALL, table
    assert table["cells per model turn"] == R.CELLS_PER_TURN, table
