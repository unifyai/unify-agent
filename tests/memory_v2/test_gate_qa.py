"""Stage-5 QA-level test checks in the gate (memory v2.1): drawn inputs, mutants, determinism, replay, size.

The library items are abridged from items the offline sweep merged (docs/design/memory-v2-end-to-end-example.md
in continual-harness-research): ARC's ``feedback_state`` (a dialogue observation), AppWorld's
``current_datetime`` (the environment) and a Crafter screen reader (text). Their recordings have the recorded
shapes: ARC feedback and demonstration messages on one dialogue channel, the phone app's clock call, Crafter
screens.

Three kinds of test:

* decisions with injected runners (no sandbox): the switch-off identity, the budget, flakiness, the probe's
  verdicts, the seeded draw;
* the static checks through :meth:`Gate.preview` (no sandbox);
* end to end in bubblewrap (skipped without it): a planted weak suite the mutation check refuses and a strong
  one it accepts, drawn inputs that crash or are falsely refused, strict sample reads, the pins, replay.
"""

import json
import random
import shutil
import subprocess
import sys
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import pytest

from unify.memory_v2 import qa as qa_mod
from unify.memory_v2 import testkit
from unify.memory_v2.blobs import BlobStore
from unify.memory_v2.episodes import Action
from unify.memory_v2.evidence import EvidenceStore
from unify.memory_v2.gate import Gate
from unify.memory_v2.gitio import Repo
from unify.memory_v2.integration.adapters.dialogue import cap_text
from unify.memory_v2.qa import (
    QAConfig,
    builtin_error,
    draw,
    pytest_env,
    same_outputs,
    seed_of,
    stage_inputs,
)
from unify.memory_v2.sandbox_run import PytestOutcome
from tests.memory_v2.test_episodes import _ep

needs_bwrap = pytest.mark.skipif(
    shutil.which("bwrap") is None,
    reason="bubblewrap required",
)

# --- recordings ------------------------------------------------------------------------------------------


def F(n, correct=False, failed=False):
    """An ARC submission-feedback message."""
    return {
        "type": "SubmitFeedback",
        "valid": True,
        "correct": correct,
        "failed": failed,
        "attempts_used": n,
    }


def D(k):
    """An ARC demonstration reply: same dialogue channel and method, other fields."""
    return {
        "type": "DemoReply",
        "demo_requests_used": k,
        "pairs": [{"input": [[1, 7], [4, 4]], "output": [[7, 1], [4, 4]]}],
        "refused": False,
    }


def _dl(obs, channel="dialogue:user"):
    return Action(-1, channel, "reply", ["submit"], {}, obs, "ok", kind="dialogue")


ARC = {
    "e1": [_dl(F(1)), _dl(F(2, correct=True))],
    "e2": [_dl(F(3, failed=True)), _dl(D(1))],
    "e3": [_dl(F(0)), _dl(D(2)), _dl(F(4, correct=True))],
    "e4": [_dl(F(5, failed=True)), _dl(D(3))],
}
RECORDED_VALUES = (
    "SubmitFeedback",
    "DemoReply",
    "09:15 AM",
    "Wednesday",
    "health: 9",
    "You see",
)

SCREENS = {
    "c1": [
        "You see: tree, grass\n\nYour status:\nhealth: 9",
        "You see: water\n\nYour status:\nhealth: 8",
    ],
    "c2": [
        "You see: stone\n\nYour status:\nhealth: 9",
        "You see: tree\n\nYour status:\nhealth: 7",
    ],
    "c3": ["You see: sand\n\nYour status:\nhealth: 6"],
}
CRAFTER = {
    eid: [_dl(s, "dialogue:crafter") for s in screens]
    for eid, screens in SCREENS.items()
}

CLOCK = {"date": "Wednesday, October 07, 2026", "time": "09:15 AM"}
PHONE = {
    "p1": [
        Action(0, "phone", "get_current_date_and_time", [], {}, CLOCK, "ok", "read"),
    ],
}

# --- library items ---------------------------------------------------------------------------------------

FEEDBACK = '''"""ARC submission feedback (abridged from the offline sweep's merged feedback_state)."""

__all__ = ["feedback_state"]


class MemoryInputError(ValueError):
    pass


def feedback_state(observation):
    """Read a submission feedback message: (attempts used, correct, failed).

    Effect: read
    Input: observation
    """
    if not isinstance(observation, dict) or observation.get("type") != "SubmitFeedback":
        raise MemoryInputError("expected a SubmitFeedback object")
    attempts = observation.get("attempts_used")
    if not isinstance(attempts, int) or attempts < 0:
        raise MemoryInputError("attempts_used must be a non-negative integer")
    return attempts, observation.get("correct") is True, observation.get("failed") is True
'''
# a guard narrower than the recordings, on a boolean (G2's held-out check never perturbs booleans)
NARROW = FEEDBACK.replace(
    "    attempts = observation.get(",
    '    if observation.get("failed") is True:\n'
    '        raise MemoryInputError("a failed submission is not read here")\n'
    "    attempts = observation.get(",
)
# no shape check: a demonstration message makes it raise KeyError
CRASHING = '''"""ARC submission feedback."""

__all__ = ["attempts_used"]


class MemoryInputError(ValueError):
    pass


def attempts_used(observation):
    """The attempts a submission feedback message reports.

    Effect: read
    Input: observation
    """
    if not isinstance(observation, dict):
        raise MemoryInputError("expected an object")
    return observation["attempts_used"]
'''
OBS = 'OBS = {"type": "SubmitFeedback", "valid": True, "correct": False, "failed": False, "attempts_used": 1}\n'
WEAK_TEST = (
    "from env.dialogue_user import feedback_state\n\n" + OBS + "\n\n"
    "def test_feedback_state_is_stable():\n"
    "    assert feedback_state(OBS) == feedback_state(OBS)\n"
)
STRONG_TEST = """import pytest

from env.dialogue_user import MemoryInputError, feedback_state


def _obs(n, correct=False, failed=False):
    return {"type": "SubmitFeedback", "valid": True, "correct": correct, "failed": failed, "attempts_used": n}


def test_reads_attempts_and_flags():
    assert feedback_state(_obs(1)) == (1, False, False)
    assert feedback_state(_obs(2, correct=True)) == (2, True, False)
    assert feedback_state(_obs(3, failed=True)) == (3, False, True)


def test_zero_attempts_is_a_count():
    assert feedback_state(_obs(0)) == (0, False, False)


@pytest.mark.parametrize(
    "bad",
    [
        {"type": "DemoReply", "demo_requests_used": 1},
        {"type": "DemoReply", "attempts_used": 2},
        "SubmitFeedback",
        {**_obs(1), "attempts_used": "1"},
        {**_obs(1), "attempts_used": -1},
    ],
)
def test_refuses_other_shapes(bad):
    with pytest.raises(MemoryInputError):
        feedback_state(bad)
"""
CRASHING_TEST = (
    "from env.dialogue_user import attempts_used\n\n" + OBS + "\n\n"
    "def test_reads_attempts():\n"
    "    assert attempts_used(OBS) == 1\n"
)
PROPERTY_TEST = """import pytest
from memlab.inputs import from_action, inputs

from env.dialogue_user import MemoryInputError, feedback_state

ROW = {"kind": "dialogue", "channel": "dialogue:user", "method": "reply", "args": ["submit"], "kwargs": {},
       "response": {"type": "SubmitFeedback", "valid": True, "correct": False, "failed": False,
                    "attempts_used": 1},
       "status": "ok"}
CASES = inputs("env/dialogue_user:feedback_state", [from_action(ROW, form="observation", label="cover")])


@pytest.mark.parametrize("x", CASES, ids=repr)
def test_agrees_with_every_recorded_message(x):
    obs = x.value()
    try:
        got = feedback_state(obs)
    except MemoryInputError:
        assert not (isinstance(obs, dict) and obs.get("type") == "SubmitFeedback")
        return
    assert got == (obs["attempts_used"], obs["correct"] is True, obs["failed"] is True)
"""

HEALTH = '''"""Crafter screens."""

__all__ = ["health"]


class MemoryInputError(ValueError):
    pass


def health(text):
    """Read the health line of a Crafter screen.

    Effect: read
    Input: text
    """
    if not isinstance(text, str) or "Your status:" not in text:
        raise MemoryInputError("expected a Crafter screen")
    return int(text.rsplit("health: ", 1)[1].split()[0])
'''
HEALTH_TEST = (
    "from env.dialogue_crafter import health\n\n\n"
    "def test_health_reads_the_status_line():\n"
    '    assert health("You see: tree\\n\\nYour status:\\nhealth: 9") == 9\n'
)
PINNED_TEST = HEALTH_TEST + f"""

def test_the_gate_pins_clock_and_randomness():
    import datetime, os, random, time

    assert os.environ.get("PYTHONHASHSEED") == "0" and os.environ.get("TZ") == "UTC"
    a, b = time.time_ns(), time.time_ns()
    assert b - a == 1_000_000 and 1_000_000_000_000_000_000 < a < 1_000_000_001_000_000_000
    assert datetime.datetime.now().year == 2001
    assert random.random() == {random.Random(0).random()!r}
"""

CLOCK_MOD = '''"""The phone app's clock (abridged from the offline sweep's AppWorld current_datetime)."""

from datetime import datetime

__all__ = ["current_datetime"]


class MemoryInputError(ValueError):
    pass


def current_datetime(apis):
    """Read and check the phone app's clock.

    Effect: read
    Input: env
    """
    value = apis.phone.get_current_date_and_time()
    if not isinstance(value, dict) or not isinstance(value.get("date"), str) or not isinstance(value.get("time"), str):
        raise MemoryInputError("expected date and time strings")
    datetime.strptime(value["date"] + " " + value["time"], "%A, %B %d, %Y %I:%M %p")
    return value
'''
CLOCK_REPLAY_TEST = f"""from memlab.episodes import Action
from memlab.replay import RecordedEnv

from env.phone import current_datetime

CLOCK = {CLOCK!r}


def test_reads_the_recorded_clock():
    env = RecordedEnv([Action(0, "phone", "get_current_date_and_time", [], {{}}, CLOCK, "ok")])
    assert current_datetime(env) == CLOCK
"""
CLOCK_FAKE_TEST = f"""from types import SimpleNamespace

from env.phone import current_datetime

CLOCK = {CLOCK!r}


def test_reads_the_clock():
    apis = SimpleNamespace(phone=SimpleNamespace(get_current_date_and_time=lambda **kw: CLOCK))
    assert current_datetime(apis) == CLOCK
"""
# library code that imports the test kit (the working model imports the library without it)
CLOCK_MOD_KIT = CLOCK_MOD.replace(
    "from datetime import datetime\n",
    "from datetime import datetime\n\nimport memlab.replay\n",
)
# a Crafter reader with a second parameter no recording names (the probe must not invent its value)
SCALED = HEALTH.replace("def health(text):", "def health(text, scale):").replace(
    'return int(text.rsplit("health: ", 1)[1].split()[0])',
    'return int(text.rsplit("health: ", 1)[1].split()[0]) * int(scale)',
)
SCALED_TEST = (
    "from env.dialogue_crafter import health\n\n\n"
    "def test_health_reads_the_status_line():\n"
    '    assert health("You see: tree\\n\\nYour status:\\nhealth: 9", 2) == 18\n'
)


def _item(item, tests, covers, form, eid):
    return {
        "item": item,
        "kind": "env_function",
        "input": form,
        "source_episodes": [eid],
        "tests": tests,
        "covers": covers,
    }


def _manifest(item, test, covers, form, eid, **extra):
    channel = item.split(":")[0]
    return {
        "items": [_item(item, [test], covers, form, eid)],
        "skeleton": [channel],
        "support": [],
        "unlisted": [],
        "deleted": [],
        **extra,
    }


FB_ITEM, FB_TEST = (
    "env/dialogue_user:feedback_state",
    "env/dialogue_user/tests/test_feedback_state.py",
)
AU_ITEM, AU_TEST = (
    "env/dialogue_user:attempts_used",
    "env/dialogue_user/tests/test_attempts_used.py",
)
HP_ITEM, HP_TEST = (
    "env/dialogue_crafter:health",
    "env/dialogue_crafter/tests/test_health.py",
)
CL_ITEM, CL_TEST = (
    "env/phone:current_datetime",
    "env/phone/tests/test_current_datetime.py",
)
ARC_COVERS = [["e1", 0], ["e1", 1]]


def _fb(test=WEAK_TEST, module=FEEDBACK, **extra):
    files = {"env/dialogue_user/__init__.py": module, FB_TEST: test}
    return files, _manifest(FB_ITEM, FB_TEST, ARC_COVERS, "observation", "e1", **extra)


def _hp(test=HEALTH_TEST, module=HEALTH):
    files = {"env/dialogue_crafter/__init__.py": module, HP_TEST: test}
    return files, _manifest(HP_ITEM, HP_TEST, [["c1", 0], ["c1", 1]], "text", "c1")


def _cl(test=CLOCK_REPLAY_TEST, module=CLOCK_MOD):
    files = {"env/phone/__init__.py": module, CL_TEST: test}
    return files, _manifest(CL_ITEM, CL_TEST, [["p1", 0]], "env", "p1")


# --- worlds ----------------------------------------------------------------------------------------------


def _world(tmp_path, episodes, **gate_kw):
    tmp_path.mkdir(parents=True, exist_ok=True)
    mem = Repo.init_bare(tmp_path / "mem.git")
    ev = EvidenceStore(tmp_path / "e.sqlite")
    for n, (eid, acts) in enumerate(episodes.items()):
        ev.index_episode(_ep(episode_id=eid, actions=acts), f"{n + 1:040d}")

    def lookup(eid, i):
        acts = episodes.get(eid, [])
        return acts[i] if 0 <= i < len(acts) else None

    gate = Gate(mem, ev, BlobStore(tmp_path / "b"), action_lookup=lookup, **gate_kw)
    return mem, gate


def _commit(mem, files):
    with mem.temp_checkout("main") as wt:
        for rel, text in files.items():
            (wt / rel).parent.mkdir(parents=True, exist_ok=True)
            if isinstance(text, bytes):
                (wt / rel).write_bytes(text)
            else:
                (wt / rel).write_text(text)
        return mem.commit_all(wt, "pass", {"Pass": "p1"})


def _check(tmp_path, episodes, files, manifest, **gate_kw):
    mem, gate = _world(tmp_path, episodes, **gate_kw)
    parent = mem.head()
    return gate.check(parent, _commit(mem, files), manifest)


def _qa_lines(res):
    return [r for r in res.reasons if "[qa:" in r]


def _no_values(res):
    for line in _qa_lines(res):
        assert not any(v in line for v in RECORDED_VALUES), line


GREEN = PytestOutcome(passed={"t::a"}, returncode=0)
RED = PytestOutcome(failed={"t::a"}, returncode=1)


class FakePytest:
    """Answers like pytest: green where the module under test exists, red where it does not.

    *second*: the outcome of every later run of the same test on the candidate (a flaky test).
    """

    def __init__(self, module="env/dialogue_crafter/__init__.py", second=None):
        self.module, self.second, self.calls, self.seen = module, second, [], set()

    def __call__(self, target, *, python, ro, rw, cwd, timeout_s=300.0, env=None):
        self.calls.append(
            (
                target,
                sorted(ro.values()),
                sorted(rw.values()),
                cwd,
                timeout_s,
                dict(env or {}),
            ),
        )
        tree = next(p for p, d in ro.items() if d == "/memory")
        if not (tree / self.module).exists():
            return RED
        key = (str(tree), target)
        if self.second is not None and key in self.seen:
            return self.second
        self.seen.add(key)
        return GREEN


class Ticking:
    """A clock that jumps *step* seconds on every read after the first *free* reads."""

    def __init__(self, step, free=0):
        self.t, self.step, self.free = 0.0, step, free

    def __call__(self):
        if self.free > 0:
            self.free -= 1
        else:
            self.t += self.step
        return self.t


def _fake_probe(outcomes):
    """A probe runner writing *outcomes* (row role -> result row fields; a structural negative answers as
    a cover unless named) for every drawn input."""
    calls = []

    def run(argv, *, ro, rw, cwd, timeout_s, env=None):
        calls.append(argv)
        qa_dir = next(p for p, d in ro.items() if d == "/qa")
        out = next(p for p, d in rw.items() if d == "/out")
        rows = json.loads((qa_dir / "samples.json").read_text())["items"][argv[3]][
            "rows"
        ]
        with open(out / "results.jsonl", "w") as fh:
            for r in rows:
                got = outcomes.get(r["role"], outcomes["cover"])
                fh.write(json.dumps({"id": r["id"], **got}) + "\n")
        return SimpleNamespace(returncode=0, stdout="", stderr="", timed_out=False)

    run.calls = calls
    return run


# --- every switch off: the gate as before ------------------------------------------------------------------


def test_default_config_is_off_and_settings_parse():
    assert not QAConfig().on and not QAConfig().draws
    assert QAConfig.from_settings(SimpleNamespace()) == QAConfig()
    cfg = QAConfig.from_settings(
        SimpleNamespace(
            UNIFY_MEMORY_V2_QA_FIXTURES=" Strict ",
            UNIFY_MEMORY_V2_QA_MUTATION="on",
            UNIFY_MEMORY_V2_QA_MUTATION_MIN_KILL="0.75",
            UNIFY_MEMORY_V2_QA_DETERMINISM="off",
            UNIFY_MEMORY_V2_QA_REPLAY="on",
            UNIFY_MEMORY_V2_QA_FIXTURE_SIZE="",
        ),
    )
    assert (
        cfg.fixtures,
        cfg.mutation,
        cfg.min_kill,
        cfg.determinism,
        cfg.replay,
        cfg.fixture_size,
    ) == (
        "strict",
        True,
        Decimal("0.75"),
        False,
        True,
        False,
    )
    for bad in (
        {"UNIFY_MEMORY_V2_QA_FIXTURES": "yes"},
        {"UNIFY_MEMORY_V2_QA_MUTATION": "1"},
        {"UNIFY_MEMORY_V2_QA_MUTATION_MIN_KILL": "1.5"},
        {"UNIFY_MEMORY_V2_QA_MUTATION_MIN_KILL": "5e-1"},
    ):
        with pytest.raises(ValueError):
            QAConfig.from_settings(SimpleNamespace(**bad))


@pytest.mark.parametrize("variant", ["pass", "touched", "undeclared"])
def test_every_switch_off_is_the_gate_as_before(tmp_path, variant):
    """Same reasons, checks and test runs (arguments and environment) with no config and the default one."""
    files, man = _hp()
    if variant == "touched":
        files[HP_TEST] = files[HP_TEST] + "\n# touched\n"
    if variant == "undeclared":
        files["env/dialogue_crafter/tests/data.json"] = "{}"
    results = []
    for n, kw in enumerate(({}, {"qa": QAConfig()})):
        runner = FakePytest()
        res = _check(tmp_path / str(n), CRAFTER, files, man, pytest_runner=runner, **kw)
        results.append(
            (
                res.passed,
                res.checks,
                res.reasons,
                res.refused,
                [c[0] for c in runner.calls],
                [c[1:] for c in runner.calls],
            ),
        )
    assert results[0] == results[1]
    assert all(c[0] == ["/memory"] for c in results[0][5])  # nothing else mounted
    assert all(
        c[4]
        == {
            "PYTHONPATH": "/memory",
            "PYTEST_ADDOPTS": "-c /dev/null --import-mode=importlib",
        }
        for c in results[0][5]
    )
    assert not any("[qa:" in r for r in results[0][2])


# --- the seeded draw -------------------------------------------------------------------------------------


def _screens(n):
    return [
        _dl(
            f"You see: tile{i:03d}\n\nYour status:\nhealth: {i % 9 + 1}",
            "dialogue:crafter",
        )
        for i in range(n)
    ]


def test_the_draw_is_seeded_by_the_candidate_bounded_and_of_the_family():
    acts = _screens(30)
    covers = [("c0", 0, acts[0]), ("c0", 1, acts[1])]
    pool = [(f"c{i // 10}", a) for i, a in enumerate(acts)]
    pool += [
        (
            "c9",
            Action(
                -1,
                "dialogue:crafter",
                "reply",
                ["x"],
                {},
                acts[5].response,
                "error",
                error="busy",
                kind="dialogue",
            ),
        ),  # a rejection
        (
            "c9",
            Action(
                -1,
                "dialogue:crafter",
                "reset",
                ["x"],
                {},
                "You see: new world",
                "ok",
                kind="dialogue",
            ),
        ),
        (
            "c9",
            _dl(
                cap_text("You see: " + "grass " * 500 + "\nhealth: 9", 300),
                "dialogue:crafter",
            ),
        ),
    ]

    def run(candidate, k=4):
        return draw(
            HP_ITEM,
            covers,
            pool,
            form="text",
            blob=lambda s: b"",
            seed=seed_of(candidate),
            k=k,
        )

    rows, notes = run("a" * 40)
    assert [r.role for r in rows if r.role != "negative"] == [
        "cover",
        "cover",
        "sample",
        "sample",
        "sample",
        "sample",
    ]
    assert {r.role for r in rows[6:]} == {
        "negative",
    }  # the covers' structural negatives
    texts = [r.action.response for r in rows[2:6]]
    assert len(set(texts)) == 4 and not set(texts) & {
        acts[0].response,
        acts[1].response,
    }
    assert all(
        r.action.status == "ok" and r.action.method == "reply" for r in rows[2:6]
    )
    assert all("middle elided" not in t for t in texts)
    assert any("1 truncated recording" in n for n in notes)
    again, _ = run("a" * 40)
    assert [r.action.response for r in again] == [
        r.action.response for r in rows
    ]  # reproducible
    others = {
        tuple(r.action.response for r in run(f"{i:040x}")[0][2:6]) for i in range(12)
    }
    assert len(others) > 1  # another candidate commit draws another sample
    big, _ = run("a" * 40, k=100)
    assert (
        len([r for r in big if r.role == "sample"]) == 28
    )  # every distinct accepted input, no more


def test_an_env_input_replays_its_episodes_calls_with_the_drawn_call_first():
    me = Action(0, "venmo", "me", [], {}, {"user_id": "u-1"}, "ok")
    other_me = Action(0, "venmo", "me", [], {}, {"user_id": "u-2"}, "ok")
    pay = Action(1, "venmo", "pay", [], {"to": "u-3"}, {"ok": True}, "ok")
    rows, _ = draw(
        "env/venmo:me",
        [("v1", 0, me)],
        [("v1", me), ("v2", pay), ("v2", other_me)],
        form="env",
        blob=lambda s: b"",
        seed=seed_of("b" * 40),
    )
    sample = next(r for r in rows if r.role == "sample" and r.action is other_me)
    assert (
        sample.context[0] is other_me and pay in sample.context and sample.form == "env"
    )
    assert sample.covered_shape  # same keyword names as the cover


# --- budget, flakiness and the probe's verdicts (injected runners) --------------------------------------


def test_an_exhausted_budget_is_a_note_never_a_pass_or_a_crash(tmp_path):
    files, man = _hp()
    res = _check(
        tmp_path,
        CRAFTER,
        files,
        man,
        pytest_runner=FakePytest(),
        qa=QAConfig(determinism=True, clock=Ticking(1000.0)),
    )
    assert res.passed, res.reasons
    (note,) = [r for r in res.reasons if "[qa:budget]" in r]
    assert (
        note.startswith("note: ")
        and "determinism rerun" in note
        and "not judged" in note
    )


def test_the_budget_stops_mutants_midway_with_a_note(tmp_path):
    files, man = _hp()
    probe = _fake_probe(
        {
            "cover": {"outcome": "handled", "digest": "x"},
            "sample": {
                "outcome": "handled",
                "digest": "y",
            },
        },
    )
    res = _check(
        tmp_path,
        CRAFTER,
        files,
        man,
        pytest_runner=FakePytest(),
        qa=QAConfig(
            mutation=True,
            clock=Ticking(100.0, free=4),
            runner=probe,
            budget_s=300.0,
        ),
    )
    assert res.passed, res.reasons
    assert any(
        "[qa:budget]" in r and "the mutants of " + HP_ITEM in r for r in res.reasons
    )
    assert not any(r.startswith("G3: [qa:mutation]") for r in res.reasons)


def test_a_test_with_two_outcomes_under_the_pins_is_refused_as_flaky(tmp_path):
    files, man = _hp()
    flaky = PytestOutcome(passed={"t::a", "t::b"}, returncode=0)
    runner = FakePytest(second=flaky)
    res = _check(
        tmp_path,
        CRAFTER,
        files,
        man,
        pytest_runner=runner,
        qa=QAConfig(determinism=True),
    )
    assert not res.passed and res.refused == ["G3"]
    (reason,) = _qa_lines(res)
    assert (
        reason.startswith("G3: [qa:determinism] " + HP_TEST)
        and "1 test(s) differ" in reason
    )
    pinned = [c for c in runner.calls if c[0] == HP_TEST]
    assert all(
        c[5]["PYTHONHASHSEED"] == "0" and "-p _memv2_pin" in c[5]["PYTEST_ADDOPTS"]
        for c in pinned
    )
    assert all(c[1] == ["/inputs", "/memory"] for c in runner.calls)
    steady = _check(
        tmp_path / "steady",
        CRAFTER,
        files,
        man,
        pytest_runner=FakePytest(),
        qa=QAConfig(determinism=True),
    )
    assert steady.passed, steady.reasons


@pytest.mark.parametrize(
    "sample,refused,expect",
    [
        ({"outcome": "handled", "digest": "a"}, False, None),
        (
            {"outcome": "error", "raised": "KeyError"},
            True,
            "raises on 3 of 3 drawn recorded inputs of its family " "(KeyError (3))",
        ),
        (
            {"outcome": "error", "raised": "Leaked9ValueFromTheRecording"},
            True,
            "(another exception (3))",
        ),
        (
            {"outcome": "refused"},
            False,
            None,
        ),  # text inputs have no field names: a precondition, noted
        ({"outcome": "miss"}, False, None),
        ({"outcome": "timeout"}, False, None),
    ],
)
def test_the_probes_verdicts(tmp_path, sample, refused, expect):
    files, man = _hp()
    probe = _fake_probe(
        {"cover": {"outcome": "handled", "digest": "c"}, "sample": sample},
    )
    res = _check(
        tmp_path,
        CRAFTER,
        files,
        man,
        pytest_runner=FakePytest(),
        qa=QAConfig(fixtures="on", runner=probe),
    )
    assert res.passed is not refused, res.reasons
    if expect:
        assert any(
            r.startswith("G3: [qa:fixtures]") and expect in r for r in res.reasons
        )
    assert any(
        "own tests read 0 of 3 drawn inputs" in r for r in res.reasons
    )  # the plain test reads none
    argv = probe.calls[0]
    assert argv[1:] == ["-s", "/qa/probe.py", HP_ITEM, "2"]
    _no_values(res)
    assert not any("Leaked9" in r for r in res.reasons)


def test_strict_refuses_drawn_inputs_no_test_read(tmp_path):
    files, man = _hp()
    probe = _fake_probe(
        {
            "cover": {"outcome": "handled", "digest": "c"},
            "sample": {
                "outcome": "handled",
                "digest": "d",
            },
        },
    )
    res = _check(
        tmp_path,
        CRAFTER,
        files,
        man,
        pytest_runner=FakePytest(),
        qa=QAConfig(fixtures="strict", runner=probe),
    )
    assert not res.passed
    assert any(
        r.startswith("G3: [qa:fixtures]")
        and "read 0 of 3" in r
        and "must be exercised" in r
        for r in res.reasons
    )


def test_dynamic_checks_wait_for_a_candidate_that_passes_the_rest(tmp_path):
    files, man = _hp()
    files[HP_TEST] = files[HP_TEST] + "\n# touched\n"
    runner = FakePytest()
    mem, gate = _world(
        tmp_path,
        CRAFTER,
        pytest_runner=runner,
        qa=QAConfig(mutation=True, determinism=True),
    )
    parent = _commit(mem, files)
    mem.fast_forward("main", parent, expected_old=mem.head())
    files[HP_TEST] = files[HP_TEST] + "\n# again\n"
    res = gate.check(parent, _commit(mem, files), man)
    assert not res.passed and res.refused == ["G3"]  # not red on the parent
    assert any("[qa:skipped]" in r for r in res.reasons)


def test_same_outputs_and_builtin_errors():
    base = {
        0: {"outcome": "handled", "digest": "a"},
        1: {"outcome": "refused"},
        2: {"outcome": "timeout"},
    }
    assert same_outputs(
        base,
        {0: {"outcome": "handled", "digest": "a"}, 1: {"outcome": "refused"}},
    )
    assert not same_outputs(
        base,
        {0: {"outcome": "handled", "digest": "b"}, 1: {"outcome": "refused"}},
    )
    assert not same_outputs(base, {0: {"outcome": "handled", "digest": "a"}})
    assert (
        not same_outputs(base, None)
        and not same_outputs({}, {})
        and not same_outputs(
            {0: {"outcome": "timeout"}},
            {0: {"outcome": "timeout"}},
        )
    )
    # replay misses, recorded errors, unbuilt or unbound rows say nothing about the function
    missed = {
        i: {"outcome": o}
        for i, o in enumerate(["miss", "recorded_error", "unbuilt", "unbound"])
    }
    assert not same_outputs(missed, dict(missed))
    mixed = {0: {"outcome": "handled", "digest": "a"}, 1: {"outcome": "miss"}}
    assert same_outputs(
        mixed,
        {0: {"outcome": "handled", "digest": "a"}, 1: {"outcome": "handled"}},
    )
    assert (
        builtin_error("KeyError") == "KeyError"
        and builtin_error("JSONDecodeError") is None
    )
    assert (
        builtin_error("print") is None
        and builtin_error("A" * 100) is None
        and builtin_error(3) is None
    )


# --- the static checks, through the consolidator's check (preview) ---------------------------------------


def _preview(tmp_path, episodes, files, manifest, qa):
    mem, gate = _world(tmp_path, episodes, qa=qa)
    tree = tmp_path / "tree"
    for rel, text in files.items():
        (tree / rel).parent.mkdir(parents=True, exist_ok=True)
        if isinstance(text, bytes):
            (tree / rel).write_bytes(text)
        else:
            (tree / rel).write_text(text)
    return gate.preview(mem.head(), tree, manifest)


def test_preview_refuses_a_stand_in_environment_and_accepts_the_replay(tmp_path):
    files, man = _cl(CLOCK_FAKE_TEST)
    reasons = _preview(tmp_path / "a", PHONE, files, man, QAConfig(replay=True))
    (reason,) = [r for r in reasons if "[qa:replay]" in r]
    assert reason.startswith(f"G3: [qa:replay] {CL_TEST} line 10 passes {CL_ITEM}")
    assert not any(v in reason for v in RECORDED_VALUES)
    files, man = _cl(CLOCK_REPLAY_TEST)
    assert _preview(tmp_path / "b", PHONE, files, man, QAConfig(replay=True)) == []
    files, man = _cl(CLOCK_FAKE_TEST)
    assert (
        _preview(tmp_path / "c", PHONE, files, man, QAConfig()) == []
    )  # switch off: as before


def test_preview_refuses_an_oversized_fixture_and_accepts_a_blob_reference(tmp_path):
    files, man = _hp()
    man["support"] = ["env/dialogue_crafter/tests/screens.json"]
    big = json.dumps(SCREENS["c1"] * 4000).encode()
    assert len(big) > qa_mod.FIXTURE_MAX_BYTES
    files["env/dialogue_crafter/tests/screens.json"] = big
    reasons = _preview(tmp_path / "a", CRAFTER, files, man, QAConfig(fixture_size=True))
    (reason,) = [r for r in reasons if "[qa:fixture-size]" in r]
    assert (
        f"screens.json has {len(big)} bytes, over the {qa_mod.FIXTURE_MAX_BYTES}-byte bound"
        in reason
    )
    files["env/dialogue_crafter/tests/screens.json"] = json.dumps(
        {"blob": "ab" * 32},
    ).encode()
    assert (
        _preview(tmp_path / "b", CRAFTER, files, man, QAConfig(fixture_size=True)) == []
    )


def test_preview_refuses_an_assertion_on_the_cut_of_a_truncated_recording(tmp_path):
    full = (
        "You see:\n"
        + "".join(f"tile{i:03d} " for i in range(200))
        + "\n\nYour status:\nhealth: 9\n"
    )
    capped = cap_text(full, 400)
    marker_end = capped.index("elided ...]\n") + len("elided ...]\n")
    tail = capped[marker_end:]
    fragment = next(
        tail[:n] for n in range(7, 40) if capped.count(tail[:n]) == 1
    )  # only where the cut is
    episodes = {
        "c1": [
            _dl(capped, "dialogue:crafter"),
            _dl(SCREENS["c1"][1], "dialogue:crafter"),
        ],
    }
    on_cut = HEALTH_TEST + f"\n\ndef test_tail():\n    assert {fragment!r} in SCREEN\n"
    files, man = _hp(on_cut)
    reasons = _preview(
        tmp_path / "a",
        episodes,
        files,
        man,
        QAConfig(fixture_size=True),
    )
    (reason,) = [r for r in reasons if "[qa:truncation]" in r]
    assert reason.startswith(
        f"G3: [qa:truncation] {HP_TEST} line 9 asserts on text at the recorder's cut",
    )
    kept = HEALTH_TEST + "\n\ndef test_head():\n    assert 'You see:' in SCREEN\n"
    files, man = _hp(kept)
    assert (
        _preview(tmp_path / "b", episodes, files, man, QAConfig(fixture_size=True))
        == []
    )


# --- the /inputs mount and Sol's side --------------------------------------------------------------------


def test_stage_inputs_mounts_memlab_the_pin_and_only_referenced_recorded_blobs(
    tmp_path,
    monkeypatch,
):
    store = BlobStore(tmp_path / "b")
    a, b = store.put(b"recorded a"), store.put(b"recorded b")
    env, notes = stage_inputs(
        tmp_path / "in",
        QAConfig(fixture_size=True),
        store.has,
        store.get,
        store.size,
        [a, "0" * 64, b],
    )
    assert sorted(p.name for p in (tmp_path / "in" / "blobs").iterdir()) == sorted(
        [a, b],
    )
    assert (tmp_path / "in" / "memlab" / "inputs.py").is_file()
    assert (tmp_path / "in" / "memlab" / "replay.py").is_file()
    assert (
        "git is not available" in (tmp_path / "in" / "memlab" / "gitio.py").read_text()
    )
    assert (tmp_path / "in" / "_memv2_pin.py").read_text() == (
        Path(qa_mod.__file__).with_name("pin.py").read_text()
    )
    assert env.env == pytest_env(QAConfig(fixture_size=True)) and notes == []
    assert (
        "-p _memv2_pin" not in env.env["PYTEST_ADDOPTS"]
        and "PYTHONHASHSEED" not in env.env
    )
    assert (
        tmp_path / "in" / "memlab" / "analysis"
    ).is_dir()  # the same kit as Sol's box
    monkeypatch.setattr(testkit, "MAX_REF_BLOBS", 1)
    _, notes = stage_inputs(
        tmp_path / "in2",
        QAConfig(replay=True),
        store.has,
        store.get,
        store.size,
        [a, b],
    )
    assert len(notes) == 1 and notes[0].startswith(
        "[qa:blobs] 1 referenced blob(s) not mounted",
    )


def test_the_pin_installs_only_under_its_box_name(tmp_path):
    import time

    before = time.time
    import unify.memory_v2.pin  # noqa: F401 - imported on the host: changes nothing

    assert time.time is before
    box = tmp_path / "box"
    box.mkdir()
    shutil.copyfile(Path(qa_mod.__file__).with_name("pin.py"), box / "_memv2_pin.py")
    code = (
        "import sys; sys.path.insert(0, sys.argv[1]); import _memv2_pin, datetime, random, time; "
        "print(time.time_ns(), time.time_ns(), datetime.datetime.now().isoformat(), random.random())"
    )
    runs = [
        subprocess.run(
            [sys.executable, "-I", "-c", code, str(box)],
            capture_output=True,
            text=True,
            env={"TZ": "UTC", "PATH": "/usr/bin:/bin"},
            timeout=60,
            check=True,
        ).stdout
        for _ in range(2)
    ]
    assert runs[0] == runs[1]
    a, b, now, rnd = runs[0].split()
    assert int(b) - int(a) == 1_000_000 and now.startswith("2001-09-09T01:46:40")
    assert float(rnd) == random.Random(0).random()


# --- end to end in the sandbox ---------------------------------------------------------------------------


@needs_bwrap
def test_mutation_refuses_a_weak_suite_naming_kinds_and_lines(tmp_path):
    files, man = _fb(WEAK_TEST)
    res = _check(tmp_path, ARC, files, man, qa=QAConfig(mutation=True, max_mutants=64))
    assert not res.passed and res.refused == ["G3"], res.reasons
    (reason,) = [r for r in res.reasons if r.startswith("G3: [qa:mutation]")]
    # 12 sites. Killed (4): both negations, != to ==, < to >= (each makes OBS refused). Surviving (8): 0 to 1,
    # return None, both "is True" flips, and the two and/or swaps and two dropped raises, which equal the
    # original on every accepted recording (the other guard masks them) but not on the covers' structural
    # negatives (14 here: each field dropped or retyped, "type" emptied, an extra field, the object retyped
    # or emptied), so none is left out as equivalent. Re-derived from mutation.py and the probe's signature.
    assert (
        "kill 4 of 12 mutants that change its outputs" in reason
        and "0 likely equivalent" in reason
    )
    assert "surviving: " in reason and " at line " in reason
    _no_values(res)


@needs_bwrap
def test_mutation_accepts_a_strong_suite(tmp_path):
    files, man = _fb(STRONG_TEST)
    res = _check(tmp_path, ARC, files, man, qa=QAConfig(mutation=True, max_mutants=64))
    assert res.passed, res.reasons
    assert any(
        "[qa:mutation]" in r and "kill 12 of 12 mutants" in r for r in res.reasons
    )


@needs_bwrap
def test_mutation_with_the_default_bound_is_seeded_by_the_candidate(tmp_path):
    files, man = _fb(STRONG_TEST)
    res = _check(tmp_path, ARC, files, man, qa=QAConfig(mutation=True))
    assert res.passed, res.reasons
    assert any("[qa:mutation]" in r and "kill 8 of 8 mutants" in r for r in res.reasons)


@needs_bwrap
def test_drawn_inputs_that_crash_the_function_refuse_it(tmp_path):
    files = {"env/dialogue_user/__init__.py": CRASHING, AU_TEST: CRASHING_TEST}
    man = _manifest(AU_ITEM, AU_TEST, ARC_COVERS, "observation", "e1")
    res = _check(tmp_path, ARC, files, man, qa=QAConfig(fixtures="on"))
    assert not res.passed and res.refused == ["G3"], res.reasons
    assert any(
        r
        == f"G3: [qa:fixtures] {AU_ITEM} raises on 3 of 7 drawn recorded inputs of its family (KeyError (3)); "
        "it must return or raise MemoryInputError"
        for r in res.reasons
    ), res.reasons
    _no_values(res)


@needs_bwrap
def test_drawn_inputs_shaped_like_the_covers_must_not_be_refused(tmp_path):
    files, man = _fb(WEAK_TEST, module=NARROW)
    res = _check(tmp_path, ARC, files, man, qa=QAConfig(fixtures="on"))
    assert not res.passed and res.refused == ["G3"], res.reasons
    assert any(
        "refuses 2 of 7 drawn recorded inputs shaped like its covers" in r
        for r in res.reasons
    )
    assert any(
        "refuses 3 of 7 drawn inputs whose fields differ" in r for r in res.reasons
    )  # a precondition
    _no_values(res)


@needs_bwrap
def test_unread_drawn_inputs_are_a_note_and_under_strict_a_refusal(tmp_path):
    files, man = _fb(STRONG_TEST)
    on = _check(tmp_path / "on", ARC, files, man, qa=QAConfig(fixtures="on"))
    assert on.passed, on.reasons
    assert any("own tests read 0 of 7 drawn inputs" in r for r in on.reasons)
    strict = _check(
        tmp_path / "strict",
        ARC,
        files,
        man,
        qa=QAConfig(fixtures="strict"),
    )
    assert not strict.passed and strict.refused == ["G3"]


@needs_bwrap
def test_strict_accepts_tests_parametrised_over_the_drawn_inputs(tmp_path):
    files, man = _fb(PROPERTY_TEST)
    res = _check(tmp_path, ARC, files, man, qa=QAConfig(fixtures="strict"))
    assert res.passed, res.reasons
    assert not any("own tests read" in r for r in res.reasons)


@needs_bwrap
def test_the_pins_hold_in_the_gates_runs(tmp_path):
    files, man = _hp(PINNED_TEST)
    res = _check(
        tmp_path / "pinned",
        CRAFTER,
        files,
        man,
        qa=QAConfig(determinism=True),
    )
    assert res.passed, res.reasons
    unpinned = _check(
        tmp_path / "unpinned",
        CRAFTER,
        files,
        man,
        qa=QAConfig(replay=True),
    )
    assert not unpinned.passed and unpinned.refused == [
        "G3",
    ]  # the same test without the pins


@needs_bwrap
def test_the_recorded_replay_runs_in_the_gate_and_a_stand_in_is_refused(tmp_path):
    files, man = _cl(CLOCK_REPLAY_TEST)
    res = _check(tmp_path / "replay", PHONE, files, man, qa=QAConfig(replay=True))
    assert res.passed, res.reasons  # memlab imports in the gate's runs
    # every switch off: a library whose tests import memlab is checked with the kit all the same (B2)
    replay_off = _check(tmp_path / "replay-off", PHONE, files, man)
    assert replay_off.passed, replay_off.reasons
    files, man = _cl(CLOCK_FAKE_TEST)
    fake = _check(tmp_path / "fake", PHONE, files, man, qa=QAConfig(replay=True))
    assert not fake.passed and any("[qa:replay]" in r for r in fake.reasons)
    off = _check(tmp_path / "off", PHONE, files, man)
    assert off.passed, off.reasons  # switch off: the stand-in passes as before


@needs_bwrap
def test_the_probe_never_invents_an_argument_no_recording_names(tmp_path):
    """A required parameter beyond the input with no recorded value leaves the probe unbound: no
    "raises on" refusal caused by the probe's own argument (it used to pass "")."""
    files, man = _hp(SCALED_TEST, module=SCALED)
    res = _check(tmp_path, CRAFTER, files, man, qa=QAConfig(fixtures="on"))
    assert res.passed, res.reasons
    assert not any("raises on" in r for r in res.reasons)
    assert any(
        "[qa:fixtures]" in r
        and "returns on none of its covers under the probe's call" in r
        for r in res.reasons
    )


# --- the test kit: a property of the library, not of the switches (B2) ---------------------------------------


def test_a_library_whose_tests_use_the_kit_is_checked_the_same_under_every_switch_setting(
    tmp_path,
):
    """Every switch off, a library whose tests import memlab gets the kit mounted exactly as with a switch
    on: same runs, mounts and environment, so turning the switches off later never freezes it.
    """
    files, man = _cl(CLOCK_REPLAY_TEST)
    runs = []
    for n, kw in enumerate(({}, {"qa": QAConfig(replay=True)})):
        runner = FakePytest(module="env/phone/__init__.py")
        res = _check(tmp_path / str(n), PHONE, files, man, pytest_runner=runner, **kw)
        assert res.passed, res.reasons
        runs.append(list(runner.calls))
    assert runs[0] == runs[1]
    assert all(c[1] == ["/inputs", "/memory"] for c in runs[0])
    assert all(
        c[5]
        == {
            "PYTHONPATH": "/memory:/inputs",
            "PYTEST_ADDOPTS": "-c /dev/null --import-mode=importlib",
        }
        for c in runs[0]
    )


def _kit_preview(tmp_path, test, module=CLOCK_MOD, qa=None):
    files, man = _cl(test, module=module)
    reasons = _preview(
        tmp_path,
        PHONE,
        files,
        man,
        qa if qa is not None else QAConfig(),
    )
    return [r for r in reasons if "[qa:kit]" in r]


def test_library_code_must_not_use_the_kit(tmp_path):
    (reason,) = _kit_preview(tmp_path / "off", CLOCK_REPLAY_TEST, module=CLOCK_MOD_KIT)
    assert reason.startswith(
        "G3: [qa:kit] env/phone/__init__.py line 5 uses the test kit",
    )  # every switch off, but the library's tests use the kit
    assert _kit_preview(
        tmp_path / "on",
        CLOCK_FAKE_TEST,
        module=CLOCK_MOD_KIT,
        qa=QAConfig(replay=True),
    )
    # every switch off and no test uses the kit: nothing is checked, as before
    assert _kit_preview(tmp_path / "plain", CLOCK_FAKE_TEST, module=CLOCK_MOD_KIT) == []
    assert _kit_preview(tmp_path / "clean", CLOCK_REPLAY_TEST) == []


def test_tests_import_only_what_the_kit_provides_and_never_name_inputs(tmp_path):
    missing = CLOCK_REPLAY_TEST.replace(
        "from memlab.replay import RecordedEnv\n",
        "from memlab.replay import RecordedEnv, Replayer\n",
    )
    (reason,) = _kit_preview(tmp_path / "a", missing)
    assert reason == (
        f"G3: [qa:kit] {CL_TEST} line 2 imports memlab.replay.Replayer, which the test kit "
        f"(version {testkit.KIT_VERSION}) does not provide"
    )
    by_path = CLOCK_REPLAY_TEST + '\nSCREENS = "/inputs/blobs"\n'
    line = by_path.count("\n")
    (reason,) = _kit_preview(tmp_path / "b", by_path)
    assert reason.startswith(
        f"G3: [qa:kit] {CL_TEST} line {line} names a path under /inputs",
    )


# --- equivalence that cannot be judged, and the probe's arguments (I1, I2) -----------------------------------


def test_mutation_is_not_judged_when_the_probe_gets_no_output(tmp_path):
    """Every probe row a replay miss: no survivor can be judged equivalent, so the kill share is not
    judged either; a note under "on", a refusal under strict, never a silent pass."""
    files, man = _hp()
    missing = _fake_probe({"cover": {"outcome": "miss"}, "sample": {"outcome": "miss"}})
    on = _check(
        tmp_path / "on",
        CRAFTER,
        files,
        man,
        pytest_runner=FakePytest(),
        qa=QAConfig(mutation=True, runner=missing),
    )
    assert on.passed, on.reasons
    (note,) = [r for r in on.reasons if "[qa:mutation]" in r]
    assert note.startswith("note: [qa:mutation] ") and "mutation not judged" in note
    strict = _check(
        tmp_path / "strict",
        CRAFTER,
        files,
        man,
        pytest_runner=FakePytest(),
        qa=QAConfig(mutation=True, fixtures="strict", runner=missing),
    )
    assert not strict.passed and strict.refused == ["G3"]
    assert any(
        r.startswith("G3: [qa:mutation] ") and "mutation not judged" in r
        for r in strict.reasons
    )
    _no_values(strict)


def test_drawn_inputs_are_not_judged_when_the_covers_do_not_return_under_the_probe(
    tmp_path,
):
    files, man = _hp()
    probe = _fake_probe(
        {
            "cover": {"outcome": "unbound"},
            "sample": {"outcome": "error", "raised": "TypeError"},
        },
    )
    res = _check(
        tmp_path,
        CRAFTER,
        files,
        man,
        pytest_runner=FakePytest(),
        qa=QAConfig(fixtures="on", runner=probe),
    )
    assert res.passed, res.reasons
    assert not any("raises on" in r for r in res.reasons)
    assert any("returns on none of its covers" in r for r in res.reasons)


def test_the_probe_rows_hold_structural_negatives_of_the_covers(tmp_path):
    """The samples file gives the probe the covers, the draws and the covers' negatives (role
    "negative", the broken value itself); only the draws are appended to the tests' inputs.
    """
    acts = _screens(6)
    rows, _ = draw(
        HP_ITEM,
        [("c0", 0, acts[0])],
        [("c0", a) for a in acts],
        form="text",
        blob=lambda s: b"",
        seed=seed_of("c" * 40),
    )
    negatives = [r for r in rows if r.role == "negative"]
    assert [r.role for r in rows[:6]] == ["cover"] + ["sample"] * 5
    assert sorted(r.value for r in negatives if isinstance(r.value, str)) == [""]
    assert [r.value for r in negatives if isinstance(r.value, int)] == [
        len(acts[0].response),
    ]
    assert all(r.has_value and not r.covered_shape for r in negatives)
    me = Action(0, "venmo", "me", [], {}, {"user_id": "u-1", "name": "A"}, "ok")
    env_rows, _ = draw(
        "env/venmo:me",
        [("v1", 0, me)],
        [("v1", me)],
        form="env",
        blob=lambda s: b"",
        seed=seed_of("d" * 40),
    )
    broken = [r for r in env_rows if r.role == "negative"]
    assert broken and all(
        r.form == "env"
        and r.context[0] is r.action
        and r.action.response != me.response
        for r in broken
    )
    assert me.response == {"user_id": "u-1", "name": "A"}  # the recording is untouched
