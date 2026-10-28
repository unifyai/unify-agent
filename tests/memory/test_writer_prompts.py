"""The writer's brief (``library_prompts.write_brief_now``) and an attention pass's first message state what the
harness does: the relations a capture group is made of, as the grouping names them; the decisions the ledger takes;
the gate as its policy decides it (it runs no code, and only the listed findings refuse); the pass's limits read from
the constants that enforce them; and the same bytes for a whole run.
"""

from __future__ import annotations

import re
from types import SimpleNamespace

from unify.common import tool_registry as R
from unify.memory import agent_profiles as AP
from unify.memory import attention, capture_groups, writer_pass
from unify.memory import attention_tools as T
from unify.memory import library_prompts as LP
from unify.memory.gate_policy import GATE

BRIEF = LP.write_brief_now(helper=True, similar=True)


def test_the_brief_names_every_relation_a_group_is_made_of_with_its_meaning():
    for kind in capture_groups.RELATION_KINDS:
        assert f"`{kind}` ({capture_groups.RELATION_TEXT[kind]})" in BRIEF, kind
    assert set(capture_groups.RELATION_TEXT) == set(capture_groups.RELATION_KINDS)
    assert capture_groups.HISTORY_TEXT in BRIEF


def test_the_decisions_the_brief_lists_are_the_ones_the_ledger_takes():
    sentence = re.search(r"A decision is one of (.*?)\. An entry you do not decide", BRIEF, re.S)
    assert sentence, "the brief lists no decisions"
    named = re.findall(r"\b([a-z]+(?:_[a-z]+)*)\b(?= \(|,| and|$)", sentence.group(1))
    assert [n for n in named if n in attention.DECISIONS] == list(attention.DECISIONS), named


def test_the_brief_states_the_gate_as_its_policy_decides():
    assert LP.gate_text() in BRIEF
    assert GATE.runs == ()
    assert "The gate runs no code" in BRIEF
    for words in GATE.refusal_words():
        assert words in BRIEF, words
    # no sentence says the gate runs, re-runs or replays anything
    gate_runs = re.compile(r"\b[Tt]he gate (?:then )?(?:re-?runs|runs|replays|executes)\b(?! no\b| none\b| neither\b)")
    runs = gate_runs.findall(BRIEF)
    assert runs == [], runs


def test_the_limits_the_writer_is_told_are_the_constants_that_enforce_them():
    rows = {r["limit"]: r for r in AP.profile("writer")["limits"]}
    assert rows["reader calls per model turn"]["value"] == R.READERS_PER_TURN
    assert rows["read_episode parts per call"]["value"] == R.READ_PARTS_PER_CALL
    assert rows["cells per model turn"]["value"] == R.CELLS_PER_TURN == writer_pass._MAX_CELLS_PER_TURN
    assert rows["step guard"]["value"] == writer_pass.STEP_GUARD
    assert rows["file under /memory"]["value"] == writer_pass.QUOTA_FILE_BYTES
    for row in rows.values():
        assert AP._limit_text(row, None) in BRIEF, row
    limits = LP.limits_now()
    assert (limits.step_guard, limits.entries, limits.check_floor_s) == (
        writer_pass.STEP_GUARD, writer_pass.QUOTA_ENTRIES, writer_pass.CHECK_FLOOR_S)


def test_the_brief_is_the_same_bytes_for_a_run_and_names_only_the_tools_it_has():
    assert LP.write_brief_now(helper=True, similar=True) == BRIEF
    plain = LP.write_brief_now()
    # without a helper lane nothing in the brief points at what helpers stage
    assert "/inputs/helpers/" in BRIEF and "/inputs/helpers/" not in plain
    assert "<<" not in BRIEF and "<<" not in plain


def _entry(eid: str, kind: str = "new_experience") -> SimpleNamespace:
    return SimpleNamespace(entry=eid, kind=kind)


def test_a_pass_first_message_names_its_entries_where_their_material_is_and_its_wall_clock():
    att = SimpleNamespace(pass_id="p1", group=None)
    text = T.first_message(att, SimpleNamespace(attention=[_entry("a" * 16)]), "900")
    assert text.startswith(f"Pass p1: 1 entry needs your attention: {'a' * 16} (new_experience).\n")
    assert "/inputs/attention/index.json" in text and "tools.decide" in text and "900 s" in text
    group = {"related_closed": [{}], "library_items": ["notes/x/y.md"]}
    text = T.first_message(SimpleNamespace(pass_id="p2", group=group),
                           SimpleNamespace(attention=[_entry("a" * 16), _entry("b" * 16)]), "600")
    assert text.startswith("Pass p2: a group of 2 related captures needs your attention")
    assert "/inputs/attention/group.json" in text and "the 1 library item the group touches" in text
