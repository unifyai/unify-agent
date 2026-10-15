"""``UNIFY_ADAPTIVE_POLICY``: the harness chooses each session's drafts, hand-back decision and mirror itself
(unify/actor/adaptive_policy.py), from the deployment's own records.

The learner (discounted Gaussian Thompson sampling, a flat prior) finds the best of the five fixed arms and follows a
change. A session's own reward is its posted verdict (typed fields only) minus its model calls over the running median;
with no verdict posted it is the cost alone, and the delayed signals still flow: a later import of a draft credits the
arm of the session that wrote it, and a later rebuild of a definition a session did not keep debits that session's arm.
Every choice (with its propensity), credit and outcome is an event row. Off, nothing of it exists.

The model is scripted (no key, no network); the end-to-end test runs the minimal arm's setting, cells in the real
sandboxed worker.
"""

from __future__ import annotations

import json
import random
from collections import Counter
from types import SimpleNamespace

import pytest

from tests.actor.code_act.core_world import core_world, world  # noqa: F401 (fixtures)
from tests.actor.code_act.sandbox_world import needs_bwrap
from tests.actor.code_act.test_in_code_tools_wiring import _act, _Collect, _system_texts, _why, code
from tests.scripted_model import ScriptedModel, reply
from unify import cli
from unify.actor import adaptive_policy as P
from unify.actor import in_code_tools
from unify.settings import SETTINGS


class _Rows(list):
    def append(self, row):  # noqa: D401 - the EventFile's one method
        super().append(dict(row))


def _session(tmp_path, monkeypatch, sid, arm):
    monkeypatch.setattr(P.Learner, "choose", lambda self, rng: (arm, 1.0))
    return P.SessionPolicy(tmp_path / "home", sid, _Rows(), rng=random.Random(0))


def _stats(tmp_path):
    return json.loads((tmp_path / "home" / P.STATE).read_text())


# --- the settings ----------------------------------------------------------------------------------------------------


def test_the_policy_is_off_by_default_and_refuses_explicit_switches_memory_and_no_tools_in_code():
    on = {"UNIFY_ADAPTIVE_POLICY": "on", "UNIFY_TOOLS_IN_CODE": "on"}
    assert not P.enabled(SimpleNamespace(UNIFY_ADAPTIVE_POLICY=""))
    assert P.refusal(SimpleNamespace(**on)) is None
    assert "UNIFY_DRAFTS" in P.refusal(SimpleNamespace(**on, UNIFY_DRAFTS="on"))
    assert "UNIFY_CONSEQUENCES" in P.refusal(SimpleNamespace(**on, UNIFY_CONSEQUENCES="on"))
    assert "UNIFY_MEMORY" in P.refusal(SimpleNamespace(**on, UNIFY_MEMORY="on"))
    assert "UNIFY_TOOLS_IN_CODE" in P.refusal(SimpleNamespace(**{**on, "UNIFY_TOOLS_IN_CODE": "off"}))


def test_progress_comes_from_the_typed_fields_only():
    assert P.progress_of({"solved": True, "score": 0.1}) == 1.0
    assert P.progress_of({"solved": False}) == 0.0
    assert P.progress_of({"solved": None, "score": 0.25}) == 0.25
    assert P.progress_of({"solved": None, "score": 7, "checks_total": 4, "checks_passed": 3}) == 0.75
    assert P.progress_of({"checks": [{"passed": True}, {"passed": False}, {"passed": None}]}) == 0.5
    assert P.progress_of({"summary": "solved it"}) is None and P.progress_of(None) is None


# --- the learner -----------------------------------------------------------------------------------------------------


def test_the_learner_finds_the_best_arm_and_follows_a_change():
    rng, learner = random.Random(7), P.Learner()

    def run(best, n):
        picks = []
        for _ in range(n):
            arm, propensity = learner.choose(rng)
            assert 0.0 <= propensity <= 1.0
            learner.update(arm, (1.0 if arm == best else 0.0) + rng.gauss(0, 0.1))
            picks.append(arm)
        return Counter(picks[-60:]).most_common(1)[0][0]

    assert run(2, 240) == 2
    assert run(4, 300) == 4  # the discount forgets the old best
    flat = P.Learner()
    assert {flat.posterior(i) for i in range(len(P.ARMS))} == {flat.posterior(0)}  # a flat prior


# --- the rewards -----------------------------------------------------------------------------------------------------


def test_a_choice_is_recorded_with_its_propensity_and_the_state(tmp_path, monkeypatch):
    s = _session(tmp_path, monkeypatch, "A", 3)
    (choice,) = s.events
    assert choice["type"] == "policy_choice" and choice["arm"] == 3 and choice["propensity"] == 1.0
    assert choice["knobs"] == {"drafts": True, "decision": "built", "mirror": True}
    assert _stats(tmp_path)["sessions"]["A"] == {"arm": 3}


def test_without_a_verdict_the_own_reward_is_the_cost_alone(tmp_path, monkeypatch):
    s = _session(tmp_path, monkeypatch, "A", 1)
    s.close(4, set())
    (row,) = [r for r in s.events if r["type"] == "policy_outcome"]
    assert row["progress"] is None and row["progress_from"] == "none" and row["reward"] == -1.0
    s2 = _session(tmp_path, monkeypatch, "B", 1)
    s2.take_outcome({"solved": True})
    s2.close(2, set())
    (row2,) = [r for r in s2.events if r["type"] == "policy_outcome"]
    assert row2["progress"] == 1.0 and row2["reward"] == 1.0 - 2 / 4  # against the median of the earlier costs
    assert _stats(tmp_path)["costs"] == [4, 2]


def test_a_later_import_credits_the_arm_that_wrote_the_draft_not_the_importer(tmp_path, monkeypatch):
    writer = _session(tmp_path, monkeypatch, "A", 2)
    writer.close(3, set())
    importer = _session(tmp_path, monkeypatch, "B", 0)
    before = _stats(tmp_path)["stats"]
    importer.imported("dA", {"session": "A", "arm": 2})
    importer.imported("dA", {"session": "A", "arm": 2})  # once per draft in a session
    importer.imported("dB", {"session": "B", "arm": 0})  # never its own
    credits = [r for r in importer.events if r["type"] == "policy_credit"]
    assert [(r["arm"], r["reward"], r["why"]) for r in credits] == [(2, P.REUSE, "reuse")]
    after = _stats(tmp_path)["stats"]
    assert after[2]["s"] == pytest.approx(before[2]["s"] * P.GAMMA + P.REUSE)
    assert after[0]["s"] == pytest.approx(before[0]["s"] * P.GAMMA)


def test_rebuilding_what_a_session_did_not_keep_debits_that_sessions_arm(tmp_path, monkeypatch):
    src = "def area(w, h):\n    return w * h"
    first = _session(tmp_path, monkeypatch, "A", 0)
    first.defined({"area": src})
    first.close(3, set())  # not kept: it vanished with the interpreter
    kept = _session(tmp_path, monkeypatch, "K", 1)
    kept.defined({"volume": "def volume(a):\n    return a ** 3"})
    kept.close(3, {"volume"})
    later = _session(tmp_path, monkeypatch, "B", 3)
    later.defined({"area": "def area(w, h):\n    # same code, new layout\n    return w*h"})
    later.defined({"area": src})  # once per definition in a session
    later.defined({"volume": "def volume(a):\n    return a ** 3"})  # kept by K: no debit
    debits = [r for r in later.events if r["type"] == "policy_credit"]
    assert [(r["arm"], r["reward"], r["why"]) for r in debits] == [(0, -P.REWORK, "rework")]


# --- the outcome through the CLI ---------------------------------------------------------------------------------------


def test_the_cli_hands_a_posted_outcome_to_the_policy_in_a_session_that_does_not_store(tmp_path, monkeypatch):
    s = _session(tmp_path, monkeypatch, "A", 1)
    P.register(s)
    monkeypatch.setattr(SETTINGS, "UNIFY_ADAPTIVE_POLICY", "on")
    emitted = []
    fake = SimpleNamespace(_mv2=None, _handle=SimpleNamespace(), _progress=lambda text: None,
                           _emit=lambda **row: emitted.append(row))  # fmt: skip
    try:
        cli.Act._post_outcome(fake, {"solved": True, "checks": [{"name": "c", "passed": True}]})
    finally:
        P.unregister(s)
    assert emitted == [{"type": "outcome", "accepted": True, "solved": True, "checks": 1}]
    assert s.outcome["solved"] is True
    monkeypatch.setattr(SETTINGS, "UNIFY_ADAPTIVE_POLICY", "")
    emitted.clear()
    cli.Act._post_outcome(fake, {"solved": True})
    assert emitted[0]["accepted"] is False  # off: as before


# --- end to end ------------------------------------------------------------------------------------------------------


@pytest.fixture
def minimal(core_world, monkeypatch, tmp_path):  # noqa: F811
    monkeypatch.setattr(SETTINGS, "UNIFY_MEMORY", "")
    monkeypatch.setattr(SETTINGS, "UNIFY_LIBRARY", "off")
    monkeypatch.setattr(SETTINGS, "UNIFY_TOOLS_IN_CODE", "on")
    monkeypatch.setattr(SETTINGS, "UNIFY_CELL_STATE", "persistent")
    collect = _Collect()
    monkeypatch.setattr(in_code_tools, "SESSIONS", collect)
    yield {"workspace": core_world["workspace"], "opened": collect.opened}


def _rows(session) -> list[dict]:
    path = in_code_tools.request_folder(session.request_id) / "tool-events.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()]


@needs_bwrap
@pytest.mark.asyncio
@pytest.mark.timeout(500)
async def test_two_sessions_choose_arms_record_them_and_credit_the_writer(minimal, monkeypatch):
    from unify.db import store_home

    monkeypatch.setattr(SETTINGS, "UNIFY_ADAPTIVE_POLICY", "on")
    arms = iter([4, 0])
    monkeypatch.setattr(P.Learner, "choose", lambda self, rng: (next(arms), 0.5))
    keep = 'tools.decide("keep", names=["double"], why="doubling recurs")\n'
    build = ScriptedModel(actor=[reply(calls=[code("def double(v):\n    return v * 2\ntools.answer('ok')\n")]),
                                 reply(calls=[code(keep)])])  # fmt: skip
    assert await _act(build, "Build.") == "ok", _why(build)
    (draft,) = sorted((minimal["workspace"] / "drafts").glob("d*.py"))
    first = _rows(minimal["opened"][0])
    assert [r["arm"] for r in first if r.get("type") == "policy_choice"] == [4]
    assert [r["arm"] for r in first if r.get("type") == "policy_outcome"] == [4]
    use = ScriptedModel(actor=[reply(calls=[code(f"from drafts.{draft.stem} import double\nprint(double(2))\n"
                                                  "def triple(v):\n    return v * 3\n")]), reply("done")])  # fmt: skip
    assert await _act(use, "Use.") == "done", _why(use)
    second = _rows(minimal["opened"][1])
    credits = [r for r in second if r.get("type") == "policy_credit"]
    assert [(r["arm"], r["why"]) for r in credits] == [(4, "reuse")]
    assert len(list((minimal["workspace"] / "drafts").glob("d*.py"))) == 1  # arm 0 writes no draft of its own
    state = json.loads((store_home() / P.STATE).read_text())
    assert [v["arm"] for v in state["sessions"].values()] == [4, 0] and len(state["shadow"]) == 2


@needs_bwrap
@pytest.mark.asyncio
@pytest.mark.timeout(400)
async def test_off_nothing_changes_and_no_state_is_written(minimal, monkeypatch):
    from unify.db import store_home

    model = ScriptedModel(actor=[reply(calls=[code("def f():\n    return 1\ntools.answer('ok')\n")])])
    assert await _act(model, "Do it.") == "ok", _why(model)
    assert model.kinds() == ["actor"]
    assert not (store_home() / P.STATE).exists()
    assert not (minimal["workspace"] / "drafts").exists()
    assert "policy_choice" not in json.dumps(_rows(minimal["opened"][0]))
    assert "decide" not in json.dumps(model.calls[0].request.get("tools"))
    assert "\n".join(_system_texts(model.calls[0]))
