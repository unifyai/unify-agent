"""With v21 off (the default), the gate is v2's: no v2.1 check is built, the same pytest calls, no new record."""

from unify.memory_v2 import gate_v21
from unify.memory_v2.gate import Gate, GateResult
from unify.memory_v2.qa import QAConfig
from unify.memory_v2.sandbox_run import run_pytest
from tests.memory_v2.test_gate import (
    FILES,
    MAN,
    _candidate,
    world,
)  # noqa: F401 (fixture)

ENV = {
    "PYTHONPATH": "/memory",
    "PYTEST_ADDOPTS": "-c /dev/null --import-mode=importlib",
}


def test_defaults_are_v2(world):
    mem, ev, gate = world
    assert gate.v21 is None and gate.qa == QAConfig()
    r = GateResult(True)
    assert r.verification == {} and r.curate_due is False


def test_off_gate_never_builds_v21_checks_and_makes_v2_calls(world, monkeypatch):
    mem, ev, gate = world

    def boom(*a, **k):
        raise AssertionError("V21Checks built with v21 off")

    monkeypatch.setattr(gate_v21.V21Checks, "__init__", boom)
    calls = []

    def recording(target, **kw):
        calls.append((target, sorted(kw["ro"].values()), kw["env"]))
        return run_pytest(target, **kw)

    gate.pytest = recording
    res = gate.merge(
        mem.head(),
        _candidate(mem, FILES),
        MAN,
        "p1",
        "incremental",
        "venmo",
        "0.01",
    )
    assert res.passed, res.reasons
    assert res.verification == {} and res.curate_due is False
    # 4675a3c45's calls for this pass: the red run on the parent, the candidate run, the candidate's suite
    assert calls == [
        ("env/venmo/tests/test_me.py", ["/memory"], ENV),
        ("env/venmo/tests/test_me.py", ["/memory"], ENV),
        ("env/venmo/tests", ["/memory"], ENV),
    ]
    reasons = ev.db.execute("SELECT reasons FROM passes WHERE pass_id='p1'").fetchone()[
        0
    ]
    assert "v21-verification" not in reasons


def test_v21_checks_need_the_v21_layout(world):
    mem, ev, gate = world
    assert (
        Gate(mem, ev, gate.blobs, v21=gate_v21.V21Config()).qa == QAConfig()
    )  # P3's layout alone: no checks
    try:
        Gate(mem, ev, gate.blobs, v21=gate_v21.V21Config(layout=False, checks=True))
    except ValueError as exc:
        assert "V21Config.layout" in str(exc)
    else:
        raise AssertionError("v2.1 checks on the v2 layout were built")
