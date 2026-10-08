"""``UNIFY_STORE_VERIFY`` never executes a model-written ``def`` in the harness's process.

A verifier (the factory ``UNIFY_STORE_VERIFY`` names) runs a candidate by
loading it (``Candidate.load``: its stored callees injected, its ``def``
executed) and calling it, in the process that asks: the harness, for the
review's ``FunctionManager_check_function``. With Python in the sandboxed worker that would
run model-written code outside the sandbox, beside the credentials, so the
verifier is refused there. These tests watch the two places a stored ``def``
is executed in this process (``_create_in_process_callable`` and
``_inject_dependencies``) and the verifier's own calls. No model is called.
"""

from __future__ import annotations

import sys
import types

import pytest

from tests.helpers import _handle_project
from unify.function_manager import store_verify
from unify.settings import SETTINGS

DOUBLE = (
    "def double(x: int) -> int:\n"
    '    """Return twice its argument."""\n'
    "    return 2 * x\n"
)


@pytest.fixture
def executed(monkeypatch):
    """The names of the functions this process executed from source."""
    from unify.function_manager.function_manager import FunctionManager

    seen: list[str] = []
    create = FunctionManager._create_in_process_callable
    inject = FunctionManager._inject_dependencies

    def watched_create(self, func_data, *args, **kwargs):
        seen.append(func_data.get("name"))
        return create(self, func_data, *args, **kwargs)

    def watched_inject(self, func_data, *args, **kwargs):
        seen.append(f"callees of {func_data.get('name')}")
        return inject(self, func_data, *args, **kwargs)

    monkeypatch.setattr(FunctionManager, "_create_in_process_callable", watched_create)
    monkeypatch.setattr(FunctionManager, "_inject_dependencies", watched_inject)
    return seen


class TinyVerifier:
    """Loads and calls the candidate in the asking process, as a verifier does."""

    def __init__(self):
        self.asked: list[str] = []

    def held_out(self, name):
        self.asked.append(f"held_out {name}")
        return {"available": True, "held_out_text": "Double 21."}

    def run(self, candidate, call_kwargs):
        self.asked.append(f"run {candidate.name}")
        value = candidate.load(types.SimpleNamespace(), None)(**call_kwargs)
        return {"ok": value == 42, "reason": f"returned {value}"}

    def recheck(self, candidate, call_kwargs):
        self.asked.append(f"recheck {candidate.name}")
        return self.run(candidate, call_kwargs)


@pytest.fixture
def tiny(monkeypatch, tmp_path):
    made: list[TinyVerifier] = []

    def factory():
        made.append(TinyVerifier())
        return made[-1]

    module = types.ModuleType("tiny_store_verifier")
    module.make = factory
    monkeypatch.setitem(sys.modules, "tiny_store_verifier", module)
    monkeypatch.setattr(SETTINGS, "UNIFY_STORE_ADMISSION", str(tmp_path / "v.json"))
    monkeypatch.setattr(SETTINGS, "UNIFY_STORE_VERIFY", "tiny_store_verifier:make")
    store_verify.reset()
    yield made
    store_verify.reset()


def test_the_verifier_is_refused_with_python_in_the_worker(tiny):
    with pytest.raises(store_verify.StoreVerifyError, match="sandboxed worker"):
        store_verify.verifier()
    assert tiny == [], "the factory was called"


@_handle_project
def test_check_function_executes_nothing_with_python_in_the_worker(
    tiny,
    executed,
):
    from unify.function_manager.function_manager import FunctionManager

    fm = FunctionManager(include_primitives=False)
    out = fm.check_function(implementation=DOUBLE, call_kwargs={"x": 21})
    assert executed == [], f"the harness executed {executed} while checking"
    assert all(not v.asked for v in tiny), [v.asked for v in tiny]
    assert "passed" not in out and "sandboxed worker" in out["error"]
    assert store_verify.passed(store_verify.source_sha256(DOUBLE)) is None


@_handle_project
def test_with_python_in_process_the_verifier_still_runs_the_candidate(
    tiny,
    executed,
    python_in_process,
):
    """With Python in this process (the function manager's non-actor mode),
    the check runs the candidate as before."""
    from unify.function_manager.function_manager import FunctionManager

    fm = FunctionManager(include_primitives=False)
    out = fm.check_function(implementation=DOUBLE, call_kwargs={"x": 21})
    assert out["passed"] is True, out
    assert "double" in executed
