"""With library_mode off (the default), the gate is v2's: no library check, the same pytest calls, no new record."""

from unify.memory import library_checks
from unify.memory.gate import Gate, GateResult
from unify.memory.qa import QAConfig
from unify.memory.sandbox_run import run_pytest
from tests.memory.test_gate import (
    FILES,
    MAN,
    _candidate,
)  # noqa: F401 (fixture)

ENV = {
    "PYTHONPATH": "/memory",
    "PYTEST_ADDOPTS": "-c /dev/null --import-mode=importlib",
}


def test_defaults_are_v2(world):
    mem, ev, gate = world
    assert gate.library_mode is None and gate.qa == QAConfig()
    r = GateResult(True)
    assert r.verification == {} and r.curate_due is False


def test_off_gate_never_builds_library_checks_and_makes_v2_calls(world, monkeypatch):
    mem, ev, gate = world

    def boom(*a, **k):
        raise AssertionError("LibraryChecks built with library_mode off")

    monkeypatch.setattr(library_checks.LibraryChecks, "__init__", boom)
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
    assert "library_mode-verification" not in reasons


def test_library_checks_need_the_library_layout(world):
    mem, ev, gate = world
    assert (
        Gate(mem, ev, gate.blobs, library_mode=library_checks.LibraryConfig()).qa
        == QAConfig()
    )  # the library layout alone: no checks
    try:
        Gate(
            mem,
            ev,
            gate.blobs,
            library_mode=library_checks.LibraryConfig(layout=False, checks=True),
        )
    except ValueError as exc:
        assert "LibraryConfig.layout" in str(exc)
    else:
        raise AssertionError("library checks on the v2 layout were built")


def test_library_merge_checks_the_reduced_candidate_with_the_candidates_seed(
    world,
    monkeypatch,
):
    """Without a check() round the merge seeds stage 5 by the candidate, and the reduced re-check uses
    the same seed, so both draw the same recorded inputs and mutants."""
    from types import SimpleNamespace

    from unify.memory.qa import seed_of

    mem, ev, gate = world
    gate.library_mode = library_checks.LibraryConfig(checks=True)
    c = _candidate(mem, FILES)
    seeds = []

    def fake_check(parent, candidate, manifest, *, reduce_as=None, seed=None):
        seeds.append(seed)
        reduced = (
            SimpleNamespace(sha=candidate, manifest=manifest, refused={}, reasons=[])
            if reduce_as
            else None
        )
        return GateResult(False, reasons=["G3: x"]), set(), [], {}, reduced

    monkeypatch.setattr(gate, "_check", fake_check)
    gate.merge(mem.head(), c, MAN, "p9", "incremental", "venmo", "0")
    assert seeds == [seed_of(c), seed_of(c)]


def test_a_memory_test_id_is_root_relative_only_under_the_library_layout():
    """v2's test ids keep v2's mapping; a memory/ path is from the root only under the library layout."""
    from unify.memory.gate import _test_file

    tid = "memory/a/tests/test_b.py::test_c"
    assert _test_file(tid, "env/x/tests") == "env/x/tests/memory/a/tests/test_b.py"
    assert (
        _test_file(tid, "memory/a/tests", library_layout=True)
        == "memory/a/tests/test_b.py"
    )
    assert (
        _test_file("env/x/tests/test_a.py::t", "env/x/tests") == "env/x/tests/test_a.py"
    )
