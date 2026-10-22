"""Rev 2's gate (``UNIFY_MEMORY_GATE_LABELS_ONLY``, :mod:`unify.memory.gate_policy`), and a landing that runs nothing.

* Purity is read from the syntax: a value or plain function that reads its input and calls no ``env``/``tools``, no
  I/O and no code from text is pure; a state form, an ``env`` call, a constant or I/O is not.
* The gate runs no code at all (rev 2.1), a pure item included: its tests are not run, its label says
  ``not run: <why>`` (with whether it is pure), and nothing is refused for it.
* A finding about an item is a label, never a refusal; the hard checks (safety, the declared change set, recorded
  provenance, what breaks the library) still refuse.
* ``land_capture`` runs nothing: no replay of the capture, no reader on empty text, no parsed fixture.
* The rev 2 profile can never enable the v1 capture path (the own check, recurrence) or the recorded replays.

No model, network or key.
"""

from __future__ import annotations

import os
import subprocess

import pytest

from tests.memory.gate_records import build
from tests.memory.test_capture_end_to_end import recorded_episode
from tests.memory.test_land_capture import ROWS, _no_blob, _store
from unify.memory import gate_policy as gp
from unify.memory import land_capture as lc
from unify.memory import sandbox_run


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


def test_the_not_run_reason_keeps_purity_as_a_fact_and_runs_nothing():
    """Rev 2.1, fix 9: no capture or function is executed to classify, rank or test it, a statically pure one
    included; whether an item is pure stays a fact on its label (the gate's not-run labels: test_gate_runs_nothing)."""
    assert gp.not_run_reason("def double(value):\n    return value * 2\n") == gp.NOT_RUN_GATE
    why = gp.not_run_reason("def send(state, env):\n    env.mail.send(state.last)\n")
    assert why.startswith(gp.NOT_RUN_GATE + "; it is not pure: ") and "state form" in why
    assert not hasattr(gp, "Policy")


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
    # every way code could be run is refused: the confined runner and every box it would start
    monkeypatch.setattr(sandbox_run, "run_confined", _refuse)
    monkeypatch.setattr(sandbox_run, "_spawn_box", _refuse)
    built = build(
        tmp_path,
        episodes={"e7": recorded_episode("e7", rows=ROWS)},
        base={},
        files={"memory/ops/mail.py": IMPURE, "memory/ops/tests/test_mail.py": IMPURE_TEST},
        items=_items("memory.ops.mail:send", "memory/ops/tests/test_mail.py"),
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


def test_a_pure_items_test_is_not_run_by_the_gate_and_its_label_says_so(tmp_path, monkeypatch):
    """Rev 2.1, fix 9: rev 2 ran a pure item's own tests (and labelled this one's failure); the gate now runs none of
    them, so a test the item lists is a claim, and the writer checks it with tools.test."""
    monkeypatch.setattr(sandbox_run, "run_confined", _refuse)
    monkeypatch.setattr(sandbox_run, "_spawn_box", _refuse)
    built = build(
        tmp_path,
        episodes={"e7": recorded_episode("e7", rows=ROWS)},
        base={},
        files={"memory/calc/ops.py": PURE, "memory/calc/tests/test_ops.py": FAILING_TEST},
        items=_items("memory.calc.ops:double", "memory/calc/tests/test_ops.py"),
    )
    assert built.result.passed, built.result.reasons
    labels = built.result.verification["memory.calc.ops:double"]["label_findings"]
    assert any(line.startswith("not run:") and gp.NOT_RUN_GATE in line for line in labels), labels
    assert not any("not green on the candidate" in line for line in labels), labels


# --- land_capture runs nothing ---------------------------------------------------------------------------------------------


def test_a_landing_runs_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(subprocess, "Popen", _refuse)
    monkeypatch.setattr(os, "fork", _refuse)
    ep = recorded_episode("e7", rows=ROWS)
    cid = _store(tmp_path / "store")
    box = tmp_path / "box"
    box.mkdir()
    reply = lc.land({"package": "readings", "name": "row_sums", "capture_id": cid}, store=tmp_path / "store", box=box,
                    episode={"e7": ep}.get, blob=_no_blob, batch=["e7"])  # fmt: skip
    assert reply.status == "ok", reply.text
    from unify.memory import captures

    landing = captures.landings(tmp_path / "store")[cid]
    assert landing["verified"]["ran"] is False and landing["recognition"]["ran"] is False
    assert not (box / "memory" / "readings" / "tests" / "data" / "row_sums.case1.json").exists()
    test = (box / "memory" / "readings" / "tests" / "test_row_sums.py").read_text()
    # no refusal test: whether the reader refuses text it was not built for is known only by running it
    assert "refuses_text_it_was_not_built_for" not in test


# --- the profile ------------------------------------------------------------------------------------------------------------


# --- the writer's brief says labels, not refusals -------------------------------------------------------------------------


