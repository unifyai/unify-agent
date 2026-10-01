"""Symbolic: ``UNIFY_STORE_TRUST=ramp`` keeps a trust record per stored function and updates it on reuse.

As shipped, a stored function is never looked at again once it is stored: in
past AppWorld runs 14 of 36 calls to a stored function raised, and each one
stayed in the library looking like a function that works. With the switch on,
every reuse is evidence. A call through the sandbox boundary, a proxy or
``execute_function`` that returns is a pass, counted with the hash of its
arguments; failures on 2 distinct inputs quarantine the function (one, if the
call had no arguments or the fresh-world verifier found it), and a trusted
function's first failure only demotes it. A failure the caller caused (the
arguments do not fit the signature, or a credential parameter holds an
unfilled placeholder) is not recorded. A function that only reads is trusted
after 3 passes over 2 distinct inputs, one that can change anything (directly
or through a stored function it calls) after 5 over 3. An
overwrite, a patch or a change to a function it calls puts it back on
probation with its passes cleared and its failure history kept. With the
switch off nothing is recorded and every call behaves as shipped. Functions
run in-process against fake ``primitives``; no model is called.
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
def test_one_failure_leaves_a_function_on_probation_and_two_inputs_quarantine(
    trust_on,
):
    fm = _FM()
    fm.add_functions(implementations=[DIVIDE])
    divide = _load(fm)["divide"]
    with pytest.raises(ZeroDivisionError):  # the caller still sees the error
        divide(1, 0)
    t = _trust(fm, "divide")
    assert (t.state, t.passes, t.failures, t.clean_uses) == ("probation", 0, 1, 0)
    assert t.last_failure == "ZeroDivisionError: division by zero"
    # the same input again is not a second input
    with pytest.raises(ZeroDivisionError):
        divide(a=1, b=0)
    assert (_trust(fm, "divide").state, _trust(fm, "divide").failures) == (
        "probation",
        2,
    )
    # a function that failed is not promoted, however often it then passes
    for a in (1, 2, 3, 4):
        divide(a, 1)
    assert _trust(fm, "divide").state == "probation"
    with pytest.raises(ZeroDivisionError):
        divide(2, 0)
    t = _trust(fm, "divide")
    assert (t.state, t.passes, t.failures) == ("quarantined", 4, 3)
    assert store_trust.QUARANTINE_FAILING_INPUTS == 2
    # later passes are counted but do not lift the quarantine
    divide(5, 1)
    t = _trust(fm, "divide")
    assert (t.state, t.passes) == ("quarantined", 5)


@_handle_project
def test_a_trusted_function_is_demoted_first_and_quarantined_on_another_input(
    trust_on,
):
    fm = _FM()
    fm.add_functions(implementations=[DIVIDE])
    divide = _load(fm)["divide"]
    for a in (1, 2, 3):
        divide(a, 1)
    assert _trust(fm, "divide").state == "trusted"
    with pytest.raises(ZeroDivisionError):
        divide(1, 0)
    t = _trust(fm, "divide")
    assert (t.state, t.passes, t.failures, t.clean_uses) == ("probation", 3, 1, 0)
    # passes do not restore trust: it has failed in this version
    divide(4, 1), divide(5, 1)
    assert _trust(fm, "divide").state == "probation"
    with pytest.raises(ZeroDivisionError):
        divide(1, 0)  # the same input: still on probation
    assert _trust(fm, "divide").state == "probation"
    with pytest.raises(ZeroDivisionError):
        divide(9, 0)
    t = _trust(fm, "divide")
    assert (t.state, t.failures) == ("quarantined", 3)


@_handle_project
def test_a_call_with_no_arguments_quarantines_on_its_first_failure(trust_on):
    fm = _FM()
    fm.add_functions(implementations=[HALT.replace("Stopped", "RuntimeError")])
    with pytest.raises(RuntimeError):
        _load(fm)["halt"]()
    t = _trust(fm, "halt")
    assert (t.state, t.failures) == ("quarantined", 1)


# --------------------------------------------------------------------------- #
#  Caller faults                                                               #
# --------------------------------------------------------------------------- #

LOGIN_AS = (
    "def login_as(user: str, access_token: str) -> str:\n"
    "    if not access_token.startswith('tok-'):\n"
    "        raise PermissionError('401 Unauthorized')\n"
    "    return user\n"
)


@pytest.mark.parametrize(
    "token",
    [
        "{{access_token}}",
        "${ACCESS_TOKEN}",
        "$access_token",
        "<token>",
        "<your access token>",
        "YOUR_ACCESS_TOKEN",
        "%TOKEN%",
        "access_token",
        "placeholder",
        "xxx",
    ],
)
@_handle_project
def test_a_placeholder_credential_is_the_callers_fault(trust_on, token):
    fm = _FM()
    fm.add_functions(implementations=[LOGIN_AS])
    login_as = _load(fm)["login_as"]
    with pytest.raises(PermissionError):  # the caller still sees the error
        login_as("ada", access_token=token)
    with pytest.raises(PermissionError):
        login_as("bob", token)
    assert _rows() == []


@_handle_project
def test_a_real_looking_credential_failure_is_the_functions(trust_on):
    fm = _FM()
    fm.add_functions(implementations=[LOGIN_AS])
    login_as = _load(fm)["login_as"]
    for token in ("$uperman1", "expired-9"):  # "$..." names no credential
        with pytest.raises(PermissionError):
            login_as("ada", token)
    t = _trust(fm, "login_as")
    assert (t.state, t.failures) == ("quarantined", 2)


def test_placeholders_are_judged_only_for_credential_parameters():
    judge = store_trust.looks_like_placeholder
    assert judge("api_key", "${API_KEY}") and judge("password", "<password>")
    assert judge("client_secret", "{{secret}}") and judge("authorization", "***")
    assert not judge("user", "{{name}}")  # not a credential parameter
    assert not judge("access_token", "eyJhbGciOi.payload.sig")
    assert not judge("password", "<abc>")  # names no credential
    assert not judge("access_token", "") and not judge("access_token", 42)


@_handle_project
@pytest.mark.asyncio
async def test_arguments_that_do_not_fit_the_signature_are_the_callers_fault(
    trust_on,
):
    fm = _FM()
    fm.add_functions(implementations=[DIVIDE])
    divide = _load(fm)["divide"]
    with pytest.raises(TypeError):
        divide(1)
    with pytest.raises(TypeError):
        divide(1, 2, c=3)
    out = await fm.execute_function(
        function_name="divide",
        call_kwargs={"a": 1, "denominator": 2},
    )
    assert "TypeError" in out["error"]
    assert _rows() == []
    # a TypeError inside the body is the function's own
    with pytest.raises(TypeError):
        divide("x", 2)
    assert _trust(fm, "divide").failures == 1


def test_the_signature_is_read_from_the_source_without_running_it():
    source = "def f(a, /, b, c=print('never'), *args, d, e=2, **kwargs):\n    pass\n"
    signature = store_trust.source_signature(source, "f")
    assert str(signature) == "(a, /, b, c=..., *args, d, e=..., **kwargs)"
    assert store_trust.source_signature(source, "g") is None
    assert store_trust.source_signature("def f(:", "f") is None


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
    assert (t.state, t.failures) == ("probation", 1)
    assert t.last_failure.startswith("TypeError: ")


@_handle_project
def test_a_callee_that_raises_quarantines_its_caller_too(trust_on, phone_env):
    fm = _FM()
    fm.add_functions(implementations=[PURGE])
    fm.add_functions(implementations=[PURGE_ALL])
    ns = _load(fm)

    def broken(**kwargs):
        raise PermissionError("401 Unauthorized")

    ns["primitives"].phone.delete_text_message = broken
    for ids in ([1], [2]):
        with pytest.raises(PermissionError):
            ns["purge_all"](ids)
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
    for a in (1, 2):
        with pytest.raises(ZeroDivisionError):
            divide(a, 0)
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
    # passes belong to the old code and are cleared; its failure is kept
    assert (t.state, t.passes, t.distinct_inputs, t.clean_uses) == (
        "probation",
        0,
        0,
        0,
    )
    assert (t.failures, t.last_failure) == (2, "ZeroDivisionError: division by zero")
    assert t.input_hashes == () and t.failure_hashes == ()
    assert t.source_hash == store_trust.sha256(fixed)
    assert _load(fm)["divide"](1, 0) == 0.0
    t = _trust(fm, "divide")
    assert (t.state, t.passes, t.failures) == ("probation", 1, 2)


@_handle_project
def test_every_failure_before_an_overwrite_stays_on_record(trust_on):
    fm = _FM()
    fm.add_functions(implementations=[LOUD])
    loud = _load(fm)["loud"]
    for n in (1, 2, 3):
        with pytest.raises(RuntimeError):
            loud(n)
    quiet = LOUD.replace("raise RuntimeError('x' * n)", "return None")
    fm.add_functions(implementations=[quiet], overwrite=True)
    t = _trust(fm, "loud")
    assert (t.state, t.passes, t.failures, t.last_failure) == (
        "probation",
        0,
        3,
        "RuntimeError: xxx",
    )
    # a later overwrite keeps it again; the history is only ever added to
    fm.add_functions(implementations=[LOUD], overwrite=True)
    assert (_trust(fm, "loud").state, _trust(fm, "loud").failures) == ("probation", 3)
    # the new version starts counting its failing inputs afresh
    loud = _load(fm)["loud"]
    with pytest.raises(RuntimeError):
        loud(4)
    assert (_trust(fm, "loud").state, _trust(fm, "loud").failures) == ("probation", 4)
    with pytest.raises(RuntimeError):
        loud(5)
    t = _trust(fm, "loud")
    assert (t.state, t.failures, t.last_failure) == (
        "quarantined",
        5,
        "RuntimeError: xxxxx",
    )


@_handle_project
def test_an_overwrite_of_a_function_that_never_failed_leaves_no_record(trust_on):
    fm = _FM()
    fm.add_functions(implementations=[DOUBLE])
    for x in (1, 2, 3):
        _load(fm)["double"](x)
    assert _trust(fm, "double").state == "trusted"
    fm.add_functions(implementations=[DOUBLE], overwrite=True)
    assert _rows() == []  # nothing to keep: the same as a fresh record
    t = _trust(fm, "double")
    assert (t.state, t.passes, t.failures) == ("probation", 0, 0)


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
    for ids in (None, 5):
        with pytest.raises(TypeError):
            ns["purge_all"](ids)  # purge_all's own failure: len(None)
    assert _trust(fm, "purge_all").state == "quarantined"
    assert _trust(fm, "purge").state == "probation"
    changed = PURGE.replace("return 'deleted'", "return 'gone'")
    fm.add_functions(implementations=[changed], overwrite=True)
    t = _trust(fm, "purge_all")
    # back on probation, its own failures still on record
    assert (t.state, t.passes, t.failures) == ("probation", 0, 2)
    assert t.last_failure.startswith("TypeError: ")
    # the restart is written, not only reported
    row = db.query_one(
        "SELECT state, failures, last_failure FROM function_trust"
        " WHERE function_id = ?",
        (_id(fm, "purge_all"),),
    )
    assert (row["state"], row["failures"]) == ("probation", 2)
    assert row["last_failure"] == t.last_failure


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
    assert (_trust(fm, "divide").state, _trust(fm, "divide").failures) == (
        "probation",
        1,
    )
    await fm.execute_function(function_name="divide", call_kwargs={"a": 5, "b": 0})
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
    assert (t.passes, t.failures, t.state) == (1, 1, "probation")


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
    assert [r["function_id"] for r in _rows()] == [
        _id(fm, "double"),
        _id(fm, "divide"),
        _id(fm, "halve"),
    ]
    # the overwrite that ends the scenario restarted divide, keeping its two
    # failures (the raise and execute_function's error) but not its pass
    t = _trust(fm, "divide")
    assert (t.state, t.passes, t.failures) == ("probation", 0, 2)


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


# --------------------------------------------------------------------------- #
#  Re-checks in a fresh world, on a backoff                                    #
# --------------------------------------------------------------------------- #


class RecordingRng:
    """A seeded ``random.Random`` that remembers every draw."""

    def __init__(self, seed=None, values=None):
        import random

        self._random = random.Random(seed)
        self._values = list(values or [])
        self.draws: list[float] = []

    def random(self) -> float:
        value = self._values.pop(0) if self._values else self._random.random()
        self.draws.append(value)
        return value


class FreshWorldVerifier:
    """Runs a candidate against its own fake phone, as an environment's fresh copy would."""

    def __init__(self):
        self.available = True
        self.outcome = True
        self.raises = False
        self.runs: list[dict] = []

    def held_out(self, name):
        return {
            "available": self.available,
            "reason": "" if self.available else "no sibling task",
            "credential_params": ["access_token"],
        }

    def run(self, candidate, call_kwargs):
        if self.raises:
            raise RuntimeError("the world could not be copied")
        fn = candidate.load(SimpleNamespace(phone=FakePhone()), None)
        value = fn(**call_kwargs)
        self.runs.append(
            {"name": candidate.name, "kwargs": dict(call_kwargs), "value": value},
        )
        return {"ok": self.outcome, "reason": "" if self.outcome else "wrong outcome"}


@pytest.fixture
def rng():
    recording = RecordingRng(seed=7)
    previous = store_trust.set_rng(recording)
    yield recording
    store_trust.set_rng(previous)


@pytest.fixture
def enable_verify(monkeypatch, tmp_path):
    """Set UNIFY_STORE_VERIFY (after the test stored its functions: the gate would refuse them)."""
    import sys
    import types

    from unify.function_manager import store_verify

    verifier = FreshWorldVerifier()

    def enable() -> FreshWorldVerifier:
        module = types.ModuleType("fresh_world_verifier")
        module.make = lambda: verifier
        monkeypatch.setitem(sys.modules, "fresh_world_verifier", module)
        monkeypatch.setattr(SETTINGS, "UNIFY_STORE_ADMISSION", str(tmp_path / "v.json"))
        monkeypatch.setattr(SETTINGS, "UNIFY_STORE_VERIFY", "fresh_world_verifier:make")
        store_verify.reset()
        return verifier

    yield enable
    store_verify.reset()


def test_the_recheck_probability_halves_per_clean_use_and_stops_at_1_in_64():
    assert [store_trust.recheck_probability(k) for k in range(9)] == [
        1.0,
        0.5,
        0.25,
        0.125,
        0.0625,
        0.03125,
        0.015625,
        0.015625,
        0.015625,
    ]


@_handle_project
def test_the_first_reuse_is_always_rechecked_and_a_pass_counts(
    trust_on,
    rng,
    enable_verify,
):
    fm = _FM()
    fm.add_functions(implementations=[DOUBLE])
    verifier = enable_verify()
    rng._values = [0.999]  # k = 0: due whatever the draw
    assert _load(fm)["double"](5) == 10
    assert verifier.runs == [{"name": "double", "kwargs": {"x": 5}, "value": 10}]
    t = _trust(fm, "double")
    # the check and the call itself: two passes over one input
    assert (t.passes, t.distinct_inputs, t.clean_uses) == (2, 1, 2)


@_handle_project
def test_a_recheck_is_due_exactly_when_the_draw_is_under_1_over_2_to_the_k(
    trust_on,
    rng,
    enable_verify,
    monkeypatch,
):
    from unify.function_manager import store_verify

    monkeypatch.setattr(store_verify, "MAX_RUN_CHECKS", 1000)
    fm = _FM()
    fm.add_functions(implementations=[DOUBLE])
    verifier = enable_verify()
    double = _load(fm)["double"]
    expected_checks = 0
    for x in range(40):
        k = _trust(fm, "double").clean_uses
        before = len(rng.draws)
        double(x)
        assert len(rng.draws) == before + 1  # one draw per reuse
        due = rng.draws[-1] < 0.5 ** min(k, 6)
        expected_checks += due
        assert len(verifier.runs) == expected_checks, (x, k, rng.draws[-1])
    # the seed gives both outcomes, and the backoff makes checks rare
    assert 2 <= expected_checks < 10


@_handle_project
def test_a_failed_recheck_quarantines_but_the_call_still_runs(
    trust_on,
    rng,
    enable_verify,
):
    fm = _FM()
    fm.add_functions(implementations=[DOUBLE])
    verifier = enable_verify()
    verifier.outcome = False
    assert _load(fm)["double"](2) == 4
    t = _trust(fm, "double")
    # one failure, but the verifier's: quarantined at once
    assert (t.state, t.failures, t.passes) == ("quarantined", 1, 1)
    assert t.last_failure == "fresh-world check failed: wrong outcome"


@_handle_project
def test_a_placeholder_call_is_not_rechecked(trust_on, rng, enable_verify):
    source = LOGIN_AS.replace("login_as", "lookup_as")
    fm = _FM()
    fm.add_functions(implementations=[source])
    verifier = enable_verify()
    with pytest.raises(PermissionError):
        _load(fm)["lookup_as"]("ada", "{{access_token}}")
    assert rng.draws == [] and verifier.runs == [] and _rows() == []


@_handle_project
def test_a_quarantined_function_is_not_rechecked(trust_on, rng, enable_verify):
    fm = _FM()
    fm.add_functions(implementations=[DIVIDE])
    divide = _load(fm)["divide"]  # loaded before the quarantine, so still callable
    for a in (1, 2):
        with pytest.raises(ZeroDivisionError):
            divide(a, 0)
    assert _trust(fm, "divide").state == "quarantined"
    verifier = enable_verify()
    assert divide(4, 2) == 2.0
    assert rng.draws == [] and verifier.runs == []


@_handle_project
def test_rechecks_never_take_the_reviews_last_two_run_checks(
    trust_on,
    rng,
    enable_verify,
):
    from unify.function_manager import store_verify

    fm = _FM()
    fm.add_functions(implementations=[DOUBLE])
    verifier = enable_verify()
    assert store_verify.run_checks_left() == 4
    store_verify.take_run_check()  # the review used one: 3 left
    rng._values = [0.0] * 5  # every draw says due
    double = _load(fm)["double"]
    for x in range(5):
        double(x)
    assert len(verifier.runs) == 1
    assert store_verify.run_checks_left() == store_trust.RECHECK_RESERVE == 2


@_handle_project
def test_no_verifier_means_in_task_evidence_only(trust_on, rng, monkeypatch):
    monkeypatch.setattr(SETTINGS, "UNIFY_STORE_VERIFY", "")
    fm = _FM()
    fm.add_functions(implementations=[DOUBLE])
    double = _load(fm)["double"]
    for x in range(3):
        double(x)
    assert rng.draws == []
    assert _trust(fm, "double").passes == 3


@_handle_project
def test_no_held_out_task_means_no_check_and_no_budget_spent(
    trust_on,
    rng,
    enable_verify,
):
    from unify.function_manager import store_verify

    fm = _FM()
    fm.add_functions(implementations=[DOUBLE])
    verifier = enable_verify()
    verifier.available = False
    _load(fm)["double"](1)
    assert verifier.runs == [] and store_verify.run_checks_left() == 4
    assert _trust(fm, "double").passes == 1


@_handle_project
def test_credentials_are_not_passed_to_the_fresh_world(
    trust_on,
    rng,
    enable_verify,
    phone_env,
):
    source = (
        "def lookup_as(number: str, access_token: str = None) -> list:\n"
        "    return primitives.phone.search_text_messages(phone_number=number)\n"
    )
    fm = _FM()
    fm.add_functions(implementations=[source])
    verifier = enable_verify()
    _load(fm)["lookup_as"]("555", access_token="session-token")
    assert verifier.runs[0]["kwargs"] == {"number": "555"}


@_handle_project
def test_a_verifier_that_raises_records_nothing(trust_on, rng, enable_verify):
    fm = _FM()
    fm.add_functions(implementations=[DOUBLE])
    verifier = enable_verify()
    verifier.raises = True
    assert _load(fm)["double"](1) == 2
    t = _trust(fm, "double")
    assert (t.state, t.passes, t.failures) == ("probation", 1, 0)


@_handle_project
def test_a_verifier_recheck_method_is_preferred_to_run(trust_on, rng, enable_verify):
    fm = _FM()
    fm.add_functions(implementations=[DOUBLE])
    verifier = enable_verify()
    seen = []
    verifier.recheck = lambda candidate, call_kwargs: (
        seen.append((candidate.name, call_kwargs)) or {"ok": True}
    )
    _load(fm)["double"](3)
    assert seen == [("double", {"x": 3})] and verifier.runs == []


def test_a_store_created_before_failing_inputs_were_kept_gains_the_column(
    monkeypatch,
    tmp_path,
):
    import sqlite3

    path = tmp_path / "old.sqlite"
    old = sqlite3.connect(path)
    old.execute(
        "CREATE TABLE function_trust (function_id INTEGER PRIMARY KEY,"
        " state TEXT NOT NULL, source_hash TEXT NOT NULL,"
        " dependency_hash TEXT NOT NULL, effect_class TEXT NOT NULL,"
        " passes INTEGER NOT NULL DEFAULT 0, failures INTEGER NOT NULL DEFAULT 0,"
        " input_hashes TEXT NOT NULL DEFAULT '[]',"
        " distinct_inputs INTEGER NOT NULL DEFAULT 0,"
        " clean_uses INTEGER NOT NULL DEFAULT 0, last_failure TEXT,"
        " updated_at TEXT NOT NULL)",
    )
    old.execute(
        "INSERT INTO function_trust (function_id, state, source_hash,"
        " dependency_hash, effect_class, failures, updated_at)"
        " VALUES (7, 'quarantined', 's', 'd', 'read_only', 1, 'then')",
    )
    old.commit()
    old.close()
    monkeypatch.setenv("UNIFY_STORE_PATH", str(path))
    db.reset_store()
    try:
        row = db.query_one("SELECT * FROM function_trust WHERE function_id = 7")
        assert (row["state"], row["failures"], row["failure_hashes"]) == (
            "quarantined",
            1,
            "[]",
        )
    finally:
        db.reset_store()
