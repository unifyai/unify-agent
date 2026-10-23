"""No text an agent reads claims the gate or the harness runs code it does not run.

:data:`unify.memory.gate_policy.GATE` says what the gate runs (``runs``: nothing), what refuses and what only labels.
Every agent-visible text (:mod:`tests.memory.agent_texts`: briefs, openings, first messages, tool texts per role,
refusals, the land_capture reply and the gate's own lines as real merges write them) is read for claims of running:
the gate (re)running, replaying or executing something, landing something only if a run passes, or the harness having
already run a capture. A claim passes only when it is negated in its own sentence ("the gate runs no code", "nothing
... is replayed"), or when :data:`GATE` says the gate runs that thing, which it never does. The tool texts that state
what the gate does are the same strings :mod:`unify.memory.gate_policy` holds, and the numbers the briefs state are
read from the constants that enforce them.
"""

from __future__ import annotations

import re

import pytest

from unify.memory import gate_policy as gp

#: Claims of running code, as regular expressions over flattened text.
CLAIMS = (
    re.compile(r"\bgate\b[^.;:]{0,30}?\b(?:re-?runs?|replays?|executes?|runs)\b", re.I),
    re.compile(r"\b(?:re-?runs?|re-?running|replay(?:s|ed|ing)?)\b", re.I),
    re.compile(r"\blands?\b[^.;]{0,80}\bonly if\b", re.I),
    re.compile(r"\bhas already run\b", re.I),
    re.compile(r"\boutput of any failing test\b", re.I),
)
#: A claim in a sentence that negates it is no claim: "the gate runs no code", "nothing ... is replayed".
NEGATION = re.compile(r"\b(?:no|not|never|nothing|none|neither|without)\b", re.I)


def _sentence(text: str, start: int, end: int) -> str:
    lo = max(text.rfind(". ", 0, start), text.rfind("; ", 0, start), text.rfind("\n", 0, start)) + 1
    hi = min([i for i in (text.find(". ", end), text.find("; ", end), text.find("\n", end)) if i != -1] or [len(text)])
    return text[lo:hi]


def claims(text: str) -> list[str]:
    """The claims of running in *text* that are not negated in their sentence and that :data:`GATE` does not make."""
    flat = " ".join(text.split())
    out = []
    for rx in CLAIMS:
        for m in rx.finditer(flat):
            sentence = _sentence(flat, m.start(), m.end())
            if NEGATION.search(sentence):
                continue
            if any(thing in sentence for thing in gp.GATE.runs):
                continue
            out.append(sentence.strip()[:300])
    return out


def test_the_lint_finds_the_claims_it_is_for():
    for false in (
        "The harness has already run each procedure on its recorded request, confined.",
        "The gate runs the recipe on every case and lands the procedure only if all pass.",
        "The gate runs the function on the recorded bytes again and refuses a fixture that differs.",
        "The gate replaces the fixture values the tests load with unseen values and reruns the tests.",
        "its tests replay the recorded calls",
        "'ok', or the gate's reasons with the output of any failing test",
    ):
        assert claims(false), false
    for true in (
        "The gate runs no code: no item, test or recipe is run to check or rank it.",
        "The test runs confined, and nothing the episode called is replayed.",
        gp.FIXTURE_TEXT,
        gp.CASES_TEXT,
        gp.LAND_GATE_TEXT,
        gp.CHECK_RETURNS,
        gp.NOT_RUN_GATE,
    ):
        assert not claims(true), true


def test_no_agent_visible_text_claims_a_run_the_gate_does_not_make(tmp_path):
    from tests.memory.agent_texts import rendered

    found = {name: claims(text) for name, text in rendered(tmp_path).items() if claims(text)}
    assert not found, found


def test_the_gate_runs_nothing_and_its_texts_are_the_tools_texts():
    from unify.common import tool_registry as R

    assert gp.GATE.runs == ()
    reg = R.registry()
    fixture = R.help_text("fixture", "writer")
    assert gp.FIXTURE_TEXT in " ".join(reg["fixture"].description.split()) or gp.FIXTURE_TEXT in fixture
    assert gp.CASES_TEXT in reg["fixture"].role_args["writer"]["properties"]["expected"]["description"]
    assert gp.LAND_GATE_TEXT in reg["land_capture"].description
    assert reg["check"].returns == gp.CHECK_RETURNS


@pytest.mark.parametrize("helper", (False, True))
def test_the_numbers_a_brief_states_are_the_enforcing_constants(helper):
    from unify.common import tool_registry as R
    from unify.memory import agent_profiles, helpers, messaging, views, writer_pass
    from unify.memory import library_prompts as lp
    from unify.memory.library_helper import NOTE_RULE_CHARS

    brief = " ".join(lp.write_brief_now(helper=helper).split())
    assert f"at most {NOTE_RULE_CHARS} characters" in brief
    assert f"at most {views.VIEW_BYTES} bytes" in brief
    assert f"after {writer_pass.STEP_GUARD} model and check calls" in brief
    delegate = " ".join(R.registry()["delegate"].description.split())
    opts = messaging.RUN_PASS_OPTIONS
    assert f"at most {opts.max_live} at once and {opts.max_total} in a pass" in delegate
    for agent, cell in (("writer", writer_pass.PassConfig.cell_timeout_s), ("helper", helpers.CELL_TIMEOUT_S)):
        limits = {row["limit"]: row["value"] for row in agent_profiles.profile(agent)["limits"]}
        assert limits["cell time"] == cell and "pass wall clock" not in limits
    writer = {row["limit"]: row["value"] for row in agent_profiles.profile("writer")["limits"]}
    assert writer["step guard"] == writer_pass.STEP_GUARD and writer["reader page"] == views.VIEW_BYTES
    assert writer["file under /memory"] == writer_pass.QUOTA_FILE_BYTES
    assert writer["time left for a check"] == writer_pass.CHECK_FLOOR_S
