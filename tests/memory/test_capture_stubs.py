"""Rev 2.1, fix 6: the fork's procedure is optional, and an empty stub is never a capture.

Rev 2's fork had to write a procedure "whether or not the work was accepted", so a conversation or a manual edit with
no code behind it left an empty ``run(...): return None``; those stubs folded into one shape and reopened its entry
again and again. Now the fork writes a procedure only when code that ran did the work, and a stub that reads nothing,
calls nothing and hands back nothing is not captured: its episode's summary is read as a fork with no procedure.
"""

from __future__ import annotations

import os

import pytest

from tests.memory.test_captures import _ep, _staging
from unify.memory import capture_trace as ct
from unify.memory import captures as cp
from unify.memory.attention import Ledger, SettledFork
from unify.memory.integration import fork

from tests.memory.test_attention_ledger import Fake


def test_the_fork_writes_a_procedure_only_when_code_did_the_work():
    text = fork.INSTRUCTION
    assert "`procedure.py`, when code that ran here did the work" in text
    assert "write no `procedure.py`" in text
    assert "Write it whether or not the work was accepted" not in text
    assert fork.CAPTURE_BULLETS in fork.SEGMENT_INSTRUCTION


@pytest.mark.parametrize(
    "source",
    [
        "def run(state, env):\n    return None\n",
        'def run(state):\n    """Nothing ran: the reply was the work. not shown"""\n',
        "def run(value):\n    pass\n",
        "def run(state, env):\n    result = 1\n    return\n",
    ],
)
def test_an_empty_run_is_a_stub(source):
    assert ct.empty_procedure(source) is True


@pytest.mark.parametrize(
    "source",
    [
        "def run(state, env):\n    env.mail.send(state.last, 'ok')\n    return None\n",
        "def run(state):\n    return state.last\n",
        "def run(value):\n    return 3\n",
        "def run(state, env):\n    print('x')\n",
        "def read_input(r):\n    return r\n",  # no run: not judged here (safety refuses it)
        "def run(v:\n",
    ],
)
def test_a_run_that_reads_calls_or_returns_is_no_stub(source):
    assert ct.empty_procedure(source) is False


def test_an_empty_stub_is_not_captured_and_its_episode_is_read_as_a_summary(tmp_path):
    stub = 'def run(state, env):\n    """Not shown."""\n    return None\n'
    store = cp.store_dir(tmp_path)
    rows = {}
    for eid in ("e1", "e2"):
        staging = _staging(tmp_path, eid, {"procedure.py": stub, "summary.md": "## Asked\nTalk it through.\n"})
        rows[eid] = cp.capture_trace_step(staging, _ep(eid), store)
    assert all(r["safety"] == "empty procedure" and r["stub"] is True and r["capture_id"] is None for r in rows.values())
    assert not [n for n in (os.listdir(store) if store.is_dir() else []) if len(n) == 16]  # nothing stored
    # the ledger reads each episode as a fork with its summary, never as the stub's capture (no fold, no reopen)
    led = Ledger(tmp_path / "home")
    new = led.sync(Fake(forks=[SettledFork(e, e, None, True, True, None) for e in ("e1", "e2")], captures=rows))
    assert sorted(e.refs.episodes[0] for e in new) == ["e1", "e2"]
    assert all(e.refs.captures == () and e.kind == "new_experience" for e in new)
