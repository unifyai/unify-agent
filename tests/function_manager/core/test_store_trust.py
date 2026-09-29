"""Symbolic: ``UNIFY_STORE_TRUST=ramp`` keeps a trust record per stored function and updates it on reuse.

As shipped, a stored function is never looked at again once it is stored: in
past AppWorld runs 14 of 36 calls to a stored function raised, and each one
stayed in the library looking like a function that works. With the switch on,
every reuse is evidence. A call through the sandbox boundary, a proxy or
``execute_function`` that returns is a pass, counted with the hash of its
arguments; one that raises quarantines the function. A function that only
reads is trusted after 3 passes over 2 distinct inputs, one that can change
anything (directly or through a stored function it calls) after 5 over 3. An
overwrite, a patch or a change to a function it calls puts it back on
probation. With the switch off nothing is recorded and every call behaves as
shipped. Functions run in-process against fake ``primitives``; no model is
called.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from tests.helpers import _handle_project
from unify import db
from unify.function_manager import store_trust
from unify.function_manager.function_manager import FunctionManager
from unify.function_manager.primitives import (
    EnvironmentMethod,
    EnvironmentNamespace,
    EnvironmentSurface,
    register_environment,
)
from unify.function_manager.primitives.environment import clear_environment_namespaces
from unify.function_manager.steering import ExecutionStopped
from unify.settings import ProductionSettings, SETTINGS

DOUBLE = "def double(x: int) -> int:\n    return x * 2\n"
DIVIDE = "def divide(a: int, b: int) -> float:\n    return a / b\n"
ASYNC_HALVE = "async def halve(x: int) -> float:\n    return x / 2\n"
PURGE = (
    "def purge(message_id: int) -> str:\n"
    "    primitives.phone.delete_text_message(text_message_id=message_id)\n"
    "    return 'deleted'\n"
)
LOOKUP = (
    "def lookup(number: str) -> list:\n"
    "    return primitives.phone.search_text_messages(phone_number=number)\n"
)
PURGE_ALL = (
    "def purge_all(ids: list) -> int:\n"
    "    for message_id in ids:\n"
    "        purge(message_id)\n"
    "    return len(ids)\n"
)
LOUD = "def loud(n: int) -> None:\n" "    raise RuntimeError('x' * n)\n"
HALT = "def halt() -> None:\n    raise Stopped('the user stopped it')\n"


class FakePhone:
    def __init__(self):
        self.calls: list[tuple] = []

    def delete_text_message(self, **kwargs):
        self.calls.append(("delete_text_message", kwargs))
        return {"ok": True}

    def search_text_messages(self, **kwargs):
        self.calls.append(("search_text_messages", kwargs))
        return [{"id": 1}]


@pytest.fixture
def phone_env():
    clear_environment_namespaces()
    methods = tuple(
        EnvironmentMethod(name=n, call=lambda **kw: None, effect=e)
        for n, e in (
            ("search_text_messages", "read"),
            ("delete_text_message", "destructive"),
        )
    )
    register_environment(
        EnvironmentSurface(
            namespaces=(EnvironmentNamespace(name="phone", methods=methods),),
        ),
        source="tests:phone",
    )
    from unify.function_manager import function_manager as fm_module

    fm_module._PRIMITIVES_SEEDED_FOR.clear()
    yield
    clear_environment_namespaces()
    fm_module._PRIMITIVES_SEEDED_FOR.clear()


@pytest.fixture
def trust_on(monkeypatch):
    monkeypatch.setattr(SETTINGS, "UNIFY_STORE_TRUST", "ramp")


@pytest.fixture
def trust_off(monkeypatch):
    monkeypatch.setattr(SETTINGS, "UNIFY_STORE_TRUST", "")


def _FM() -> FunctionManager:
    return FunctionManager(include_primitives=False)


def _load(fm: FunctionManager, **extra) -> dict:
    """Load the library the way the actor's list tool does: into a sandbox namespace."""
    namespace = {"primitives": SimpleNamespace(phone=FakePhone()), **extra}
    fm.list_functions(_return_callable=True, _namespace=namespace)
    return namespace


def _id(fm: FunctionManager, name: str) -> int:
    return int(fm.list_function_name_to_ids()[name])


def _trust(fm: FunctionManager, name: str) -> store_trust.Trust:
    return store_trust.trust(_id(fm, name))


def _rows() -> list[dict]:
    return db.query("SELECT * FROM function_trust ORDER BY function_id")


def test_the_switch_is_off_by_default_and_takes_only_ramp():
    assert ProductionSettings.model_fields["UNIFY_STORE_TRUST"].default == ""
    assert ProductionSettings(UNIFY_STORE_TRUST=" Ramp ").UNIFY_STORE_TRUST == "ramp"
    with pytest.raises(ValueError, match="must be empty or 'ramp'"):
        ProductionSettings(UNIFY_STORE_TRUST="on")


# --------------------------------------------------------------------------- #
#  Promotion                                                                   #
# --------------------------------------------------------------------------- #


@_handle_project
def test_a_read_only_function_is_trusted_after_3_passes_over_2_inputs(trust_on):
    fm = _FM()
    fm.add_functions(implementations=[DOUBLE])
    double = _load(fm)["double"]
    assert _trust(fm, "double").state == "probation"  # no record yet
    assert _rows() == []
    assert double(1) == 2 and double(1) == 2 and double(x=1) == 2
    t = _trust(fm, "double")
    # three passes, but all on one input: f(1) and f(x=1) hash alike
    assert (t.state, t.passes, t.distinct_inputs, t.clean_uses) == (
        "probation",
        3,
        1,
        3,
    )
    assert t.effect_class == "read_only"
    assert double(2) == 4
    t = _trust(fm, "double")
    assert (t.state, t.passes, t.distinct_inputs) == ("trusted", 4, 2)


@_handle_project
def test_two_passes_over_two_inputs_are_not_enough(trust_on):
    fm = _FM()
    fm.add_functions(implementations=[DOUBLE])
    double = _load(fm)["double"]
    double(1), double(2)
    assert _trust(fm, "double").state == "probation"
    double(2)
    assert _trust(fm, "double").state == "trusted"


@_handle_project
def test_a_function_that_changes_things_needs_5_passes_over_3_inputs(
    trust_on,
    phone_env,
):
    fm = _FM()
    fm.add_functions(implementations=[PURGE, LOOKUP])
    ns = _load(fm)
    for message_id in (1, 1, 1, 2, 2):  # five passes, two inputs
        assert ns["purge"](message_id) == "deleted"
    t = _trust(fm, "purge")
    assert (t.effect_class, t.state, t.passes, t.distinct_inputs) == (
        "changes",
        "probation",
        5,
        2,
    )
    ns["purge"](3)
    assert _trust(fm, "purge").state == "trusted"
    fm.add_functions(implementations=[PURGE], overwrite=True)
    ns = _load(fm)
    for message_id in (1, 2, 3, 3):  # three inputs, four passes
        ns["purge"](message_id)
    t = _trust(fm, "purge")
    assert (t.state, t.passes, t.distinct_inputs) == ("probation", 4, 3)
    ns["purge"](3)
    assert _trust(fm, "purge").state == "trusted"
    # the same number of calls over the same inputs trusts a reader sooner
    for number in ("1", "2", "3"):
        ns["lookup"](number)
    t = _trust(fm, "lookup")
    assert (t.effect_class, t.state) == ("read_only", "trusted")


@_handle_project
def test_the_effect_class_follows_the_stored_functions_a_function_calls(
    trust_on,
    phone_env,
):
    fm = _FM()
    fm.add_functions(implementations=[PURGE])
    fm.add_functions(implementations=[PURGE_ALL])
    ns = _load(fm)
    for ids in ([1], [2], [3]):
        ns["purge_all"](ids)
    # purge_all names no environment method itself, but purge deletes
    t = _trust(fm, "purge_all")
    assert (t.effect_class, t.state, t.passes, t.distinct_inputs) == (
        "changes",
        "probation",
        3,
        3,
    )
    # each nested call of purge was a reuse of purge too
    assert _trust(fm, "purge").passes == 3


@_handle_project
def test_an_unknown_primitive_counts_as_a_change(trust_on):
    fm = _FM()
    fm.add_functions(implementations=[PURGE])  # no environment registered
    assert _trust(fm, "purge").effect_class == "changes"


# --------------------------------------------------------------------------- #
#  Demotion                                                                    #
# --------------------------------------------------------------------------- #


@_handle_project
def test_any_failure_quarantines_even_a_trusted_function(trust_on):
    fm = _FM()
    fm.add_functions(implementations=[DIVIDE])
    divide = _load(fm)["divide"]
    for a in (1, 2, 3):
        divide(a, 1)
    assert _trust(fm, "divide").state == "trusted"
    with pytest.raises(ZeroDivisionError):  # the caller still sees the error
        divide(1, 0)
    t = _trust(fm, "divide")
    assert (t.state, t.passes, t.failures, t.clean_uses) == ("quarantined", 3, 1, 0)
    assert t.last_failure == "ZeroDivisionError: division by zero"
    # later passes are counted but do not lift the quarantine
    divide(4, 1)
    t = _trust(fm, "divide")
    assert (t.state, t.passes) == ("quarantined", 4)


@_handle_project
def test_a_failure_reason_is_truncated(trust_on):
    fm = _FM()
    fm.add_functions(implementations=[LOUD])
    with pytest.raises(RuntimeError):
        _load(fm)["loud"](1000)
    reason = _trust(fm, "loud").last_failure
    assert len(reason) == store_trust.REASON_LIMIT
    assert reason.startswith("RuntimeError: xxx") and reason.endswith("...")


@_handle_project
def test_an_async_function_is_recorded_when_awaited(trust_on):
    import asyncio

    fm = _FM()
    fm.add_functions(implementations=[ASYNC_HALVE])
    halve = _load(fm)["halve"]
    pending = halve(4)
    assert _rows() == []  # not run yet
    assert asyncio.run(pending) == 2
    assert _trust(fm, "halve").passes == 1

    async def fail():
        return await halve("x")

    with pytest.raises(TypeError):
        asyncio.run(fail())
    t = _trust(fm, "halve")
    assert t.state == "quarantined" and t.last_failure.startswith("TypeError: ")


@_handle_project
def test_a_callee_that_raises_quarantines_its_caller_too(trust_on, phone_env):
    fm = _FM()
    fm.add_functions(implementations=[PURGE])
    fm.add_functions(implementations=[PURGE_ALL])
    ns = _load(fm)

    def broken(**kwargs):
        raise PermissionError("401 Unauthorized")

    ns["primitives"].phone.delete_text_message = broken
    with pytest.raises(PermissionError):
        ns["purge_all"]([1])
    for name in ("purge", "purge_all"):
        t = _trust(fm, name)
        assert (t.state, t.last_failure) == (
            "quarantined",
            "PermissionError: 401 Unauthorized",
        )


@_handle_project
def test_a_steering_stop_is_not_evidence(trust_on):
    fm = _FM()
    fm.add_functions(implementations=[HALT])
    with pytest.raises(ExecutionStopped):
        _load(fm, Stopped=ExecutionStopped)["halt"]()
    assert _rows() == []


# --------------------------------------------------------------------------- #
#  Restarting probation                                                        #
# --------------------------------------------------------------------------- #


def _quarantine_divide(fm: FunctionManager) -> None:
    fm.add_functions(implementations=[DIVIDE])
    divide = _load(fm)["divide"]
    divide(1, 1)
    with pytest.raises(ZeroDivisionError):
        divide(1, 0)
    assert _trust(fm, "divide").state == "quarantined"


@_handle_project
def test_an_overwrite_puts_the_function_back_on_probation(trust_on):
    fm = _FM()
    _quarantine_divide(fm)
    fixed = DIVIDE.replace("return a / b", "return a / b if b else 0.0")
    assert fm.add_functions(implementations=[fixed], overwrite=True) == {
        "divide": "updated",
    }
    t = _trust(fm, "divide")
    assert (t.state, t.passes, t.failures, t.last_failure) == (
        "probation",
        0,
        0,
        None,
    )
    assert _load(fm)["divide"](1, 0) == 0.0
    assert _trust(fm, "divide").passes == 1


@_handle_project
def test_an_identical_overwrite_restarts_probation_too(trust_on):
    fm = _FM()
    _quarantine_divide(fm)
    fm.add_functions(implementations=[DIVIDE], overwrite=True)
    assert _trust(fm, "divide").state == "probation"


@_handle_project
def test_a_patch_puts_the_function_back_on_probation(trust_on, monkeypatch):
    monkeypatch.setattr(SETTINGS, "UNIFY_FUNCTION_PATCH", True)
    fm = _FM()
    _quarantine_divide(fm)
    result = fm.patch_function(
        name="divide",
        old="return a / b",
        new="return a / b if b else 0.0",
        why="dividing by zero raised",
    )
    assert result["status"] == "patched"
    assert _trust(fm, "divide").state == "probation"


@_handle_project
def test_a_changed_callee_puts_its_callers_back_on_probation(trust_on, phone_env):
    fm = _FM()
    fm.add_functions(implementations=[PURGE])
    fm.add_functions(implementations=[PURGE_ALL])
    ns = _load(fm)
    with pytest.raises(TypeError):
        ns["purge_all"](None)  # purge_all's own failure: len(None)
    assert _trust(fm, "purge_all").state == "quarantined"
    assert _trust(fm, "purge").state == "probation"
    changed = PURGE.replace("return 'deleted'", "return 'gone'")
    fm.add_functions(implementations=[changed], overwrite=True)
    t = _trust(fm, "purge_all")
    assert (t.state, t.failures) == ("probation", 0)
    # the restart is written, not only reported
    row = db.query_one(
        "SELECT state FROM function_trust WHERE function_id = ?",
        (_id(fm, "purge_all"),),
    )
    assert row["state"] == "probation"


@_handle_project
def test_code_loaded_before_a_change_is_not_evidence_for_the_new_version(trust_on):
    fm = _FM()
    fm.add_functions(implementations=[DIVIDE])
    old = _load(fm)["divide"]
    fixed = DIVIDE.replace("return a / b", "return a / b if b else 0.0")
    fm.add_functions(implementations=[fixed], overwrite=True)
    with pytest.raises(ZeroDivisionError):
        old(1, 0)
    assert _trust(fm, "divide").state == "probation"
    assert _rows() == []


@_handle_project
def test_deleting_a_function_deletes_its_record(trust_on):
    fm = _FM()
    fm.add_functions(implementations=[DOUBLE, DIVIDE])
    ns = _load(fm)
    ns["double"](1), ns["divide"](1, 1)
    assert len(_rows()) == 2
    fm.delete_function(function_id=_id(fm, "double"))
    assert [r["function_id"] for r in _rows()] == [_id(fm, "divide")]
    db.clear()
    assert _rows() == []


# --------------------------------------------------------------------------- #
#  The other call paths                                                        #
# --------------------------------------------------------------------------- #


@_handle_project
@pytest.mark.asyncio
async def test_execute_function_records_passes_and_failures(trust_on):
    fm = _FM()
    fm.add_functions(implementations=[DIVIDE])
    out = await fm.execute_function(
        function_name="divide",
        call_kwargs={"a": 4, "b": 2},
    )
    assert out["error"] is None and out["result"] == 2.0
    assert _trust(fm, "divide").passes == 1
    out = await fm.execute_function(
        function_name="divide",
        call_kwargs={"a": 4, "b": 0},
    )
    assert "ZeroDivisionError" in out["error"]  # reported, not raised, as shipped
    t = _trust(fm, "divide")
    assert (t.state, t.last_failure) == (
        "quarantined",
        "ZeroDivisionError: division by zero",
    )


@_handle_project
@pytest.mark.asyncio
async def test_a_failed_install_is_a_failed_reuse(trust_on, monkeypatch):
    from unify.function_manager import function_manager as fm_module

    fm = _FM()
    fm.add_functions(implementations=[DOUBLE])

    def no_install(requirements):
        raise RuntimeError("uv pip install failed: no matching distribution")

    monkeypatch.setattr(fm_module.environment, "ensure", no_install)
    with pytest.raises(RuntimeError):
        await fm.execute_function(function_name="double", call_kwargs={"x": 1})
    assert _trust(fm, "double").last_failure.startswith("RuntimeError: uv pip")


@_handle_project
@pytest.mark.asyncio
async def test_a_proxy_call_is_recorded(trust_on):
    fm = _FM()
    fm.add_functions(implementations=[DIVIDE])
    namespace: dict = {}
    proxies = fm.list_functions(_return_callable=True, _namespace=namespace)
    assert await proxies["divide"](6, 3) == 2.0
    with pytest.raises(ZeroDivisionError):
        await proxies["divide"](a=1, b=0)
    t = _trust(fm, "divide")
    assert (t.passes, t.failures, t.state) == (1, 1, "quarantined")


# --------------------------------------------------------------------------- #
#  Switch off                                                                  #
# --------------------------------------------------------------------------- #


async def _scenario(fm: FunctionManager) -> list:
    """Every call path, a failure on each, and an overwrite; what the caller sees.

    Everything is loaded before the first failure: a later loaded read leaves
    a quarantined function out, by design (tested below).
    """
    seen: list = []
    fm.add_functions(implementations=[DOUBLE, DIVIDE, ASYNC_HALVE])
    ns = _load(fm)
    proxies = fm.list_functions(_return_callable=True, _namespace={})
    seen.append(type(ns["double"]).__name__)
    seen += [ns["double"](1), ns["double"](x=2), await ns["halve"](3)]
    try:
        ns["divide"](1, 0)
    except ZeroDivisionError as exc:
        seen.append(repr(exc))
    out = await fm.execute_function(
        function_name="divide",
        call_kwargs={"a": 1, "b": 0},
    )
    seen.append((out["result"], out["error"].strip().splitlines()[-1]))
    seen.append(await proxies["divide"](4, 2))
    seen.append(fm.add_functions(implementations=[DIVIDE], overwrite=True))
    seen.append(sorted(fm.list_functions()))
    return seen


@_handle_project
@pytest.mark.asyncio
async def test_switch_off_records_nothing_and_calls_behave_the_same(monkeypatch):
    monkeypatch.setattr(SETTINGS, "UNIFY_STORE_TRUST", "")
    fm = _FM()
    off = await _scenario(fm)
    assert _rows() == []
    ns = _load(fm)
    assert ns["double"]._observer is None
    proxies = fm.list_functions(_return_callable=True, _namespace={})
    assert proxies["double"]._observer is None

    db.clear()
    monkeypatch.setattr(SETTINGS, "UNIFY_STORE_TRUST", "ramp")
    fm = _FM()
    on = await _scenario(fm)
    assert on == off
    # divide's record was cleared by the overwrite that ends the scenario
    assert [r["function_id"] for r in _rows()] == [
        _id(fm, "double"),
        _id(fm, "halve"),
    ]


# --------------------------------------------------------------------------- #
#  Quarantine hides a function from the reads that load it                     #
# --------------------------------------------------------------------------- #


def _bag_of_words(texts: list[str]):
    import hashlib
    import re

    import numpy as np

    vectors = np.zeros((len(texts), 256), dtype=np.float32)
    for row, text in enumerate(texts):
        for word in re.findall(r"[a-z0-9]+", text.lower()):
            vectors[row, int(hashlib.sha256(word.encode()).hexdigest(), 16) % 256] += 1
        vectors[row, 0] += 1e-3  # never a zero vector
    return vectors


@pytest.fixture
def local_vectors(monkeypatch):
    from unify.common import embeddings
    from unify.common.embeddings import Embedder

    monkeypatch.setattr(
        embeddings,
        "embedder",
        lambda: Embedder("tests-bag-of-words/256", _bag_of_words),
    )


def _loaded_reads(fm: FunctionManager) -> dict:
    """What the actor's list, filter and search tools return, and what they load."""
    out = {}
    for read, kwargs in (
        ("list", {}),
        ("filter", {}),
        ("search", {"query": "divide or double numbers", "n": 5}),
    ):
        namespace: dict = {}
        method = getattr(fm, f"{read}_functions")
        result = method(
            _return_callable=True,
            _namespace=namespace,
            _also_return_metadata=True,
            **kwargs,
        )
        callables = result["callables"]
        names = sorted(
            (
                callables
                if isinstance(callables, dict)
                else [c.__name__ for c in callables]
            ),
        )
        metadata = result["metadata"]
        if isinstance(metadata, dict):
            warnings = [v for k, v in metadata.items() if k.startswith("(")]
            rows = sorted(k for k in metadata if not k.startswith("("))
        else:
            warnings = [r["warning"] for r in metadata if "warning" in r]
            rows = sorted(r["name"] for r in metadata if "name" in r)
        out[read] = {
            "callables": names,
            "rows": rows,
            "warnings": warnings,
            "loaded": sorted(k for k in ("double", "divide") if k in namespace),
        }
    return out


@_handle_project
def test_a_quarantined_function_is_left_out_of_loaded_reads_and_named(
    trust_on,
    local_vectors,
):
    fm = _FM()
    fm.add_functions(implementations=[DOUBLE])
    _quarantine_divide(fm)
    for read, seen in _loaded_reads(fm).items():
        assert seen["callables"] == seen["rows"] == seen["loaded"] == ["double"], read
        assert seen["warnings"] == [
            "Left out 1 stored function(s) that raised when last reused and wait "
            "for repair, so they are not callable here: divide (last failure: "
            "ZeroDivisionError: division by zero)",
        ], read
    # it stays in the store, and the reads that return rows (the review's) show it
    assert "divide" in fm.list_functions()
    assert "divide" in [r["name"] for r in fm.filter_functions()]
    assert "divide" in [
        r["name"] for r in fm.search_functions(query="divide numbers", n=5)
    ]


@_handle_project
def test_a_repaired_function_is_loaded_again(trust_on, local_vectors):
    fm = _FM()
    fm.add_functions(implementations=[DOUBLE])
    _quarantine_divide(fm)
    fixed = DIVIDE.replace("return a / b", "return a / b if b else 0.0")
    fm.add_functions(implementations=[fixed], overwrite=True)
    for read, seen in _loaded_reads(fm).items():
        assert seen["callables"] == ["divide", "double"], read
        assert seen["warnings"] == [], read


@_handle_project
def test_switch_off_loads_a_quarantined_function_as_shipped(
    monkeypatch,
    local_vectors,
):
    monkeypatch.setattr(SETTINGS, "UNIFY_STORE_TRUST", "ramp")
    fm = _FM()
    fm.add_functions(implementations=[DOUBLE])
    _quarantine_divide(fm)
    monkeypatch.setattr(SETTINGS, "UNIFY_STORE_TRUST", "")
    with_record = _loaded_reads(fm)
    db.execute("DELETE FROM function_trust")
    assert _loaded_reads(fm) == with_record
    for read, seen in with_record.items():
        assert seen["callables"] == ["divide", "double"], read
        assert seen["warnings"] == [], read
