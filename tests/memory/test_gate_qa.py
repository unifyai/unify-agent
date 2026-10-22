"""Stage-5 QA-level test checks in the gate (library mode): drawn inputs, mutants, determinism, replay, size.

The library items are abridged from items the offline sweep merged (the research repository's end-to-end example design note
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

from unify.memory.library_checks import LibraryConfig
from unify.memory import qa as qa_mod
from unify.memory import testkit
from unify.memory.blobs import BlobStore
from unify.memory.episodes import Action
from unify.memory.evidence import EvidenceStore
from unify.memory.gate import Gate
from unify.memory.gitio import Repo
from unify.memory.integration.adapters.dialogue import cap_text
from unify.memory.qa import QAConfig, pytest_env, stage_inputs
from unify.memory.sandbox_run import PytestOutcome
from tests.memory.test_episodes import _ep

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
# passes only on variant 0's pinned values: the determinism rerun (variant 1) turns it red
DEPENDS_ON_PINS_TEST = HEALTH_TEST + f"""

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
# the harness's replay helper (memlab.replay.env_from over exported rows), asserting the calls it served
CLOCK_ENV_FROM_TEST = f"""from memlab.replay import calls, env_from

from env.phone import current_datetime

CLOCK = {CLOCK!r}
ROWS = [
    {{"channel": "phone", "method": "get_current_date_and_time", "args": [], "kwargs": {{}},
      "response": CLOCK, "status": "ok", "effect": "read"}},
]


def test_reads_the_recorded_clock():
    env = env_from(ROWS)
    assert current_datetime(env) == CLOCK
    assert env.issued() == calls(ROWS) and not env.misses
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

    gate = Gate(mem, ev, BlobStore(tmp_path / "b"), action_lookup=lookup, **{"library_mode": LibraryConfig(), **gate_kw})
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
        # per call: whether import skips were asked to fail
        self.import_skips_fail: list[bool] = []

    def __call__(
        self,
        target,
        *,
        python,
        ro,
        rw,
        cwd,
        timeout_s=300.0,
        env=None,
        import_skips_fail=False,
    ):
        self.import_skips_fail.append(import_skips_fail)
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


# --- the seeded draw -------------------------------------------------------------------------------------


def _screens(n):
    return [
        _dl(
            f"You see: tile{i:03d}\n\nYour status:\nhealth: {i % 9 + 1}",
            "dialogue:crafter",
        )
        for i in range(n)
    ]


# --- budget, flakiness and the probe's verdicts (injected runners) --------------------------------------


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


# --- the /inputs mount and the writer's side --------------------------------------------------------------------


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
    assert (tmp_path / "in" / "_memory_pin.py").read_text() == (
        Path(qa_mod.__file__).with_name("pin.py").read_text()
    )
    assert env.env == pytest_env(QAConfig(fixture_size=True)) and notes == []
    assert (
        "-p _memory_pin" not in env.env["PYTEST_ADDOPTS"]
        and "PYTHONHASHSEED" not in env.env
    )
    assert (
        tmp_path / "in" / "memlab" / "analysis"
    ).is_dir()  # the same kit as the writer's box
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
    import unify.memory.pin  # noqa: F401 - imported on the host: changes nothing

    assert time.time is before
    box = tmp_path / "box"
    box.mkdir()
    shutil.copyfile(Path(qa_mod.__file__).with_name("pin.py"), box / "_memory_pin.py")
    code = (
        "import sys; sys.path.insert(0, sys.argv[1]); import _memory_pin, datetime, random, time; "
        "print(time.time_ns(), time.time_ns(), datetime.datetime.now().isoformat(), random.random())"
    )
    runs = [
        subprocess.run(
            [sys.executable, "-I", "-c", code, str(box)],
            capture_output=True,
            text=True,
            env={"TZ": "UTC", "PATH": "/usr/bin:/bin", "MEMORY_PIN": variant},
            timeout=60,
            check=True,
        ).stdout
        for variant in ("0", "0", "1")
    ]
    assert (
        runs[0] == runs[1] != runs[2]
    )  # reproducible per variant, different across them
    a, b, now, rnd = runs[0].split()
    assert int(b) - int(a) == 1_000_000 and now.startswith("2001-09-09T01:46:40")
    assert float(rnd) == random.Random(0).random()
    a, b, now, rnd = runs[2].split()
    assert int(b) - int(a) == 7_000_000 and now.startswith("2033-05-18T03:33:20")
    assert float(rnd) == random.Random(1).random()


# --- end to end in the sandbox ---------------------------------------------------------------------------


# --- the test kit: a property of the library, not of the switches ---------------------------------------


# the same test reaching the kit through pytest.importorskip, as a defensive test might
CLOCK_SKIP_TEST = CLOCK_REPLAY_TEST.replace(
    "from memlab.episodes import Action\nfrom memlab.replay import RecordedEnv\n",
    'import pytest\n\nAction = pytest.importorskip("memlab.episodes").Action\n'
    'RecordedEnv = pytest.importorskip("memlab.replay").RecordedEnv\n',
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


# --- equivalence that cannot be judged, and the probe's arguments -----------------------------------
