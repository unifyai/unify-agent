"""Rev 2's gate (``UNIFY_MEMORY_GATE_LABELS_ONLY``, :mod:`unify.memory.gate_policy`), and a landing that runs nothing.

* Purity is read from the syntax: a value or plain function that reads its input and calls no ``env``/``tools``, no
  I/O and no code from text is pure; a state form, an ``env`` call, a constant or I/O is not.
* The gate runs no code for an impure item: its tests are not run, its label says ``not run: <why>``, and nothing
  is refused for it.
* A finding about an item is a label, never a refusal; the hard checks (safety, the declared change set, recorded
  provenance, what breaks the library) still refuse.
* ``land_capture`` runs nothing: no replay of the capture, no reader on empty text, no parsed fixture.
* The rev 2 profile can never enable the v1 capture path (the own check, recurrence) or the recorded replays.

No model, network or key.
"""

from __future__ import annotations

import json
import os
import subprocess
from types import SimpleNamespace

import pytest

from tests.memory.gate_records import build, needs_box
from tests.memory.test_capture_end_to_end import recorded_episode
from tests.memory.test_land_capture import ROWS, _no_blob, _store
from unify.memory import gate_policy as gp
from unify.memory import land_capture as lc
from unify.memory import sandbox_run
from unify.memory.integration import switch


def _refuse(*a, **k):
    raise AssertionError("code was run")


# --- purity, from the syntax -----------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "src, pure",
    [
        ("def double(value):\n    return value * 2\n", True),
        ("def total(rows):\n    from memory.t.m import add\n    return add(rows)\n", True),
        ("def send(state, env):\n    env.mail.send(state.last)\n", False),
        ("def last(state):\n    return state.last\n", False),
        ("def answer(value):\n    return [6, 9]\n", False),  # a constant
        ("def read(path):\n    return open(path).read()\n", False),
        ("def run(text):\n    import subprocess\n    return subprocess.run(text)\n", False),
        ("def run(code):\n    return eval(code)\n", False),
        ("async def run(value):\n    return value\n", False),
        ("def call(value):\n    return tools.thing(value)\n", False),
    ],
)
def test_purity_is_read_from_the_syntax(src, pure):
    ok, why = gp.purity(src)
    assert ok is pure, why
    assert (why == "") is pure


def test_a_test_module_is_runnable_only_when_everything_it_uses_is_pure(tmp_path):
    bodies = {"memory.t.m:double": "def double(value):\n    return value * 2\n",
              "memory.t.m:send": "def send(state, env):\n    env.mail.send(state.last)\n"}  # fmt: skip
    policy = gp.Policy(bodies=bodies)
    (tmp_path / "memory" / "t" / "tests").mkdir(parents=True)
    (tmp_path / "memory/t/tests/test_a.py").write_text("from memory.t.m import double\n\ndef test_a():\n    assert 1\n")
    (tmp_path / "memory/t/tests/test_b.py").write_text("import memory.t.m as m\n\ndef test_b():\n    m.send\n")
    assert policy.test(tmp_path, "memory/t/tests/test_a.py") == (True, "")
    ok, why = policy.test(tmp_path, "memory/t/tests/test_b.py")
    assert not ok and "memory.t.m:send" in why and "state form" in why
    assert policy.suite(tmp_path, "memory/t/tests")[0] is False
    (tmp_path / "memory/t/tests/test_c.py").write_text("from memory.t.m import *\n")
    assert policy.test(tmp_path, "memory/t/tests/test_c.py")[0] is False


def test_the_hard_checks_stay_refusals():
    assert gp.hard("G6", "key-shaped string in memory/x.py", ["i"])
    assert gp.hard("G1", "undeclared change memory/x.py", [])
    assert gp.hard("G1", "fixture provenance: it does not re-derive", ["i"])
    assert gp.hard("G2", "memory/x.py does not parse", ["i"])
    assert not gp.hard("G3", "memory/x/tests/test_y.py is not green on the candidate", ["i"])
    assert not gp.hard("G2", "i: its recipe does not give the expected output", ["i"])
    assert not gp.hard("G1", "i docstring has no When to use section", ["i"])
    assert not gp.hard("G5", "no fewer items", [])


# --- the gate runs nothing for an impure item, and labels instead of refusing ----------------------------------------------

IMPURE = '''"""Mail the result."""


def send(state, env):
    """Mail the latest reply."""
    env.mail.send(state.last)
    return None
'''
IMPURE_TEST = '''from memory.ops.mail import send


def test_send_is_there():
    assert send is not None
'''


def _items(item: str, test: str) -> list[dict]:
    return [{"item": item, "kind": "function", "source_episodes": ["e7"], "tests": [test], "covers": []}]


def test_the_gate_runs_nothing_for_an_impure_item_and_labels_it(tmp_path, monkeypatch):
    # every way the gate runs code is refused: its pytest runner, its library checks' runner, the sandbox's own
    monkeypatch.setattr(sandbox_run, "run_confined", _refuse)
    built = build(
        tmp_path,
        episodes={"e7": recorded_episode("e7", rows=ROWS)},
        base={},
        files={"memory/ops/mail.py": IMPURE, "memory/ops/tests/test_mail.py": IMPURE_TEST},
        items=_items("memory.ops.mail:send", "memory/ops/tests/test_mail.py"),
        gate_kwargs={"labels_only": True, "pytest_runner": _refuse},
        config_kwargs={"runner": _refuse},
    )
    assert built.result.passed, built.result.reasons
    labels = built.result.verification["memory.ops.mail:send"]["label_findings"]
    assert any(line.startswith("not run:") and "state form" in line for line in labels), labels


PURE = '''"""Double a number."""


def double(value):
    """Twice the value."""
    return value * 2
'''
FAILING_TEST = '''from memory.calc.ops import double


def test_double_is_wrong_on_purpose():
    assert double(2) == 5
'''


@needs_box
@pytest.mark.timeout(600)
def test_a_failing_test_of_a_pure_item_is_a_label_not_a_refusal(tmp_path):
    world = dict(
        episodes={"e7": recorded_episode("e7", rows=ROWS)},
        base={},
        files={"memory/calc/ops.py": PURE, "memory/calc/tests/test_ops.py": FAILING_TEST},
        items=_items("memory.calc.ops:double", "memory/calc/tests/test_ops.py"),
    )
    as_built = build(tmp_path / "built", require_pass=False, **world)
    assert not as_built.result.passed  # before rev 2: refused
    rev2 = build(tmp_path / "rev2", gate_kwargs={"labels_only": True}, **world)
    assert rev2.result.passed, rev2.result.reasons
    labels = rev2.result.verification["memory.calc.ops:double"]["label_findings"]
    assert any("not green on the candidate" in line for line in labels), labels


# --- land_capture runs nothing ---------------------------------------------------------------------------------------------


def test_a_landing_runs_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(subprocess, "Popen", _refuse)
    monkeypatch.setattr(os, "fork", _refuse)
    ep = recorded_episode("e7", rows=ROWS)
    cid = _store(tmp_path / "store")
    box = tmp_path / "box"
    box.mkdir()
    reply = lc.land({"package": "readings", "name": "row_sums", "capture_id": cid}, store=tmp_path / "store", box=box,
                    episode={"e7": ep}.get, blob=_no_blob, batch=["e7"], runner=_refuse)  # fmt: skip
    assert reply.status == "ok", reply.text
    from unify.memory import captures

    landing = captures.landings(tmp_path / "store")[cid]
    assert landing["verified"]["ran"] is False and landing["recognition"]["ran"] is False
    assert not (box / "memory" / "readings" / "tests" / "data" / "row_sums.case1.json").exists()
    test = (box / "memory" / "readings" / "tests" / "test_row_sums.py").read_text()
    # no refusal test: whether the reader refuses text it was not built for is known only by running it
    assert "refuses_text_it_was_not_built_for" not in test


# --- the profile ------------------------------------------------------------------------------------------------------------


def test_the_rev2_profile_never_enables_the_v1_capture_path_or_the_recorded_replays():
    kit = {"UNIFY_MEMORY": "on", "UNIFY_MEMORY_LIBRARY": "on", "UNIFY_MEMORY_LIBRARY_FORK": "on"}
    for arm, values in switch.REDESIGN_ARMS.items():
        settings = SimpleNamespace(**{**kit, **values}) if arm != "no_memory" else SimpleNamespace(**values)
        if not switch.memory_on(settings):
            continue
        # the trace contract (no own check, no recurrence) and the labels-only gate, never the recorded replays
        assert switch.capture_trace(settings), arm
        assert switch.gate_labels_only(settings) and not switch.gate_recorded_replays(settings), arm
    # set otherwise under the profile, each is refused before anything runs
    for name, value in (("UNIFY_MEMORY_CAPTURE_TRACE", "off"), ("UNIFY_MEMORY_GATE_LABELS_ONLY", "off"),
                        ("UNIFY_MEMORY_GATE_RECORDED_REPLAYS", "on")):  # fmt: skip
        refused = switch.redesign_refusal(SimpleNamespace(**kit, UNIFY_MEMORY_PROFILE="redesign", **{name: value}))
        assert refused is not None and name in refused, (name, refused)
    assert switch.parse_gate_recorded_replays("") == "" and switch.parse_gate_labels_only("ON") == "on"
    assert json.dumps(switch.REDESIGN_SWITCHES)  # plain values a kit pins


# --- the writer's brief says labels, not refusals -------------------------------------------------------------------------


def test_the_brief_under_the_rev2_gate_describes_labels_and_stays_general(monkeypatch):
    from unify.common import in_code_switch
    from unify.memory import library_prompts as lp

    monkeypatch.setattr(in_code_switch, "tools_in_code", lambda: True)
    for impure in (False, True):
        text = lp.write_brief_now(staging=True, pool=True, switches=lp.WriterSwitches(impure=impure, labels_only=True))
        assert lp.LABELS_ONLY_GATE in text and lp.LABELS_ONLY_REPAIR in text
        for gone in (
            "The gate refuses a new or changed function or note",
            "The gate refuses the items of a folder",
            "the gate refuses a candidate with fewer items",
            "unseen values of the same type and reruns the tests",
            "more recorded inputs of the same kind to a copy of that file",
            "lands the procedure only if it gives every expected output",
            "A procedure is verified by replaying its episode",
            "has passed the gate or is kept as a draft with its failure",
        ):
            assert gone not in text, gone
        assert "is labelled so" in text and "has landed with its label" in text
        assert lp.benchmark_words(text) == [] and lp.example_checks(text) == []
    # off: the brief as built
    assert lp.write_brief_now(staging=True, pool=True, switches=lp.WriterSwitches()) == lp.write_brief_now(
        staging=True, pool=True,
    )
