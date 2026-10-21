"""Symbolic: one observer seam for environment calls (``primitives.observers``).

Two features need to see every call an actor makes into its environment: an
evidence ledger that records requests and responses, and speculation that
answers a write without running it. Both subscribe to one hook in the wrapper
every ``primitives.<ns>.<method>`` call passes through, instead of each
editing it. With no observer pushed the wrapper takes the shipped path,
including the ``UNIFY_FUNCTION_CASES`` recording branch, and an environment's
raw globals (``apis``) are injected as the very objects it registered. With
the ledger or speculation switched on, those globals are wrapped in a proxy
that routes ``apis.app.api(...)`` calls through the same hook. Functions run
in process against a fake registered environment; no model is called.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from types import SimpleNamespace
from typing import Any, Optional

import pytest

from tests.helpers import _handle_project
from unify.common.tool_errors import ToolInputError
from unify.function_manager import store_cases
from unify.function_manager.execution_env import create_execution_globals
from unify.function_manager.function_manager import FunctionManager
from unify.function_manager.primitives import (
    EnvironmentMethod,
    EnvironmentNamespace,
    EnvironmentSurface,
    observers,
    register_environment,
)
from unify.function_manager.primitives.environment import (
    clear_environment_namespaces,
    namespace_object,
)
from unify.function_manager.primitives.observers import (
    EnvCall,
    Intercepted,
    observing,
)
from unify.settings import SETTINGS

CALLS: list[tuple[str, Any]] = []


class _App:
    """One app of a dynamic client, as AppWorld's ``apis.<app>`` is."""

    def __init__(self, name: str) -> None:
        self.name = name

    def login(self, *, username: str) -> dict:
        CALLS.append((f"{self.name}.login", {"username": username}))
        return {"token": f"t-{username}"}  # pragma: allowlist secret

    async def like(self, song_id: int) -> dict:
        CALLS.append((f"{self.name}.like", song_id))
        return {"liked": song_id}

    def _private(self) -> str:
        return "hidden"


class Apis:
    """A raw environment global: the same surface reached by attribute access."""

    def __init__(self) -> None:
        self.spotify = _App("spotify")
        self.version = "1.0"

    def __iter__(self):
        return iter([self.spotify])

    def __repr__(self) -> str:
        return "<apis>"


APIS = Apis()


def _forecast(city: str) -> dict:
    CALLS.append(("forecast", city))
    return {"city": city, "temp_c": 21}


def _set_alert(city: str) -> dict:
    CALLS.append(("set_alert", city))
    return {"alert": city}


def _delete_alerts() -> dict:
    CALLS.append(("delete_alerts", None))
    return {"deleted": 2}


def _fails(city: str) -> dict:
    CALLS.append(("fails", city))
    raise LookupError(f"no station in {city}")


async def _alerts(city: str) -> list:
    CALLS.append(("alerts", city))
    await asyncio.sleep(0)
    return [f"storm over {city}"]


async def _clear_alerts(city: str) -> dict:
    CALLS.append(("clear_alerts", city))
    return {"cleared": city}


def weather_surface() -> EnvironmentSurface:
    return EnvironmentSurface(
        namespaces=(
            EnvironmentNamespace(
                name="weather",
                methods=(
                    EnvironmentMethod("forecast", _forecast, "read", "(city: str)"),
                    EnvironmentMethod("set_alert", _set_alert, "write", "(city: str)"),
                    EnvironmentMethod("delete_alerts", _delete_alerts, "destructive"),
                    EnvironmentMethod("fails", _fails, "read", "(city: str)"),
                    EnvironmentMethod("alerts", _alerts, "read", "(city: str)"),
                    EnvironmentMethod(
                        "clear_alerts",
                        _clear_alerts,
                        "destructive",
                        "(city: str)",
                    ),
                ),
            ),
            EnvironmentNamespace(
                name="spotify",
                methods=(
                    EnvironmentMethod(
                        "login",
                        lambda **kw: APIS.spotify.login(**kw),
                        "read",
                        "(*, username: str)",
                    ),
                ),
            ),
        ),
        globals={"apis": APIS},
    )


@pytest.fixture
def weather():
    from unify.function_manager import function_manager as fm_module

    clear_environment_namespaces()
    register_environment(weather_surface(), source="tests:observers")
    fm_module._PRIMITIVES_SEEDED_FOR.clear()
    CALLS.clear()
    yield namespace_object("weather")
    clear_environment_namespaces()
    fm_module._PRIMITIVES_SEEDED_FOR.clear()


@pytest.fixture
def proxy_on(monkeypatch):
    """A feature that uses the seam is on. Neither switch exists yet, so the
    value is placed where ``getattr(SETTINGS, ...)`` finds it."""
    monkeypatch.setitem(vars(SETTINGS), "UNIFY_SPECULATE", "writes")


class Recorder:
    """Records every hook call; optionally intercepts or fails."""

    def __init__(
        self,
        name: str = "rec",
        *,
        intercept: Any = None,
        complete: bool = False,
        before_raises: Optional[BaseException] = None,
        after_raises: Optional[BaseException] = None,
        log: Optional[list] = None,
    ) -> None:
        self.name = name
        self.intercept = intercept
        self.complete = complete
        self.before_raises = before_raises
        self.after_raises = after_raises
        self.log = log if log is not None else []

    def before(self, call: EnvCall) -> Optional[Intercepted]:
        self.log.append((self.name, "before", call))
        if self.before_raises is not None:
            raise self.before_raises
        return self.intercept

    def after(self, call: EnvCall, **outcome: Any) -> None:
        self.log.append((self.name, "after", call, outcome))
        if self.after_raises is not None:
            raise self.after_raises


# --------------------------------------------------------------------------- #
#  No observer: as shipped                                                    #
# --------------------------------------------------------------------------- #


def test_no_observer_takes_the_shipped_path(weather, monkeypatch):
    def refuse(*_a, **_k):
        raise AssertionError("the observer path ran with no observer")

    monkeypatch.setattr(observers, "dispatch", refuse)
    monkeypatch.setattr(observers, "dispatch_async", refuse)
    assert observers.current() == () and not observers.complete_required()
    assert weather.forecast("Oslo") == {"city": "Oslo", "temp_c": 21}
    assert asyncio.run(weather.alerts("Oslo")) == ["storm over Oslo"]
    assert CALLS == [("forecast", "Oslo"), ("alerts", "Oslo")]


def test_switches_off_inject_the_registered_globals_themselves(weather):
    assert not observers.proxy_enabled()
    assert create_execution_globals()["apis"] is APIS
    values = {"apis": APIS}
    assert observers.observed_globals(values, enabled=False)["apis"] is APIS
    # A setting that is present but off is off.
    assert not observers.proxy_enabled(SimpleNamespace(UNIFY_SPECULATE="off"))
    assert observers.proxy_enabled(SimpleNamespace(UNIFY_SPECULATE="writes"))


@pytest.fixture
def python_in_process(monkeypatch):
    """Python in process (the function manager's in-process mode, for non-actor callers): these tests run stored functions in this process. With the
    sandboxed worker that is refused, and the worker's observer tests cover it (tests/actor/code_act/test_bind_load_confinement.py).
    """
    monkeypatch.setattr("unify.actor.execution.worker.enabled", lambda: False)


@pytest.mark.asyncio
async def test_switches_off_a_stored_function_gets_the_global_itself(
    weather,
    python_in_process,
):
    fm = FunctionManager(include_primitives=False)
    out = await fm._execute_python_function(
        implementation="def which():\n    return type(apis).__name__\n",
        call_kwargs={},
    )
    assert out["error"] is None and out["result"] == "Apis"


# --------------------------------------------------------------------------- #
#  One before/after pair per call                                             #
# --------------------------------------------------------------------------- #


def test_an_observer_sees_one_pair_per_sync_call_with_its_effect(weather):
    rec = Recorder()
    with observing(rec):
        assert observers.current() == (rec,)
        assert weather.forecast("Oslo") == {"city": "Oslo", "temp_c": 21}
        assert weather.set_alert(city="Bergen") == {"alert": "Bergen"}
        assert weather.delete_alerts() == {"deleted": 2}
    assert observers.current() == ()
    weather.forecast("unobserved")

    assert [(e[1], e[2].method) for e in rec.log] == [
        ("before", "forecast"),
        ("after", "forecast"),
        ("before", "set_alert"),
        ("after", "set_alert"),
        ("before", "delete_alerts"),
        ("after", "delete_alerts"),
    ]
    calls = [e[2] for e in rec.log if e[1] == "before"]
    assert [c.effect for c in calls] == ["read", "write", "destructive"]
    assert calls[0] == EnvCall(
        namespace="weather",
        method="forecast",
        effect="read",
        args=("Oslo",),
        kwargs={},
        via="primitives",
    )
    assert calls[1].args == () and calls[1].kwargs == {"city": "Bergen"}
    _, _, call, outcome = rec.log[1]
    assert call is calls[0]
    assert outcome["result"] == {"city": "Oslo", "temp_c": 21}
    assert outcome["error"] is None and outcome["intercepted"] is False
    assert isinstance(outcome["started"], float)
    assert isinstance(outcome["elapsed_s"], float) and outcome["elapsed_s"] >= 0
    assert set(outcome) == {"result", "error", "intercepted", "started", "elapsed_s"}
    assert len(CALLS) == 4


@pytest.mark.asyncio
async def test_an_observer_sees_one_pair_per_async_call(weather):
    rec = Recorder()
    with observing(rec):
        assert await weather.alerts("Oslo") == ["storm over Oslo"]
    b, a = rec.log
    assert b[1] == "before" and b[2].method == "alerts" and b[2].effect == "read"
    assert a[1] == "after" and a[3]["result"] == ["storm over Oslo"]
    assert a[3]["intercepted"] is False and a[3]["elapsed_s"] >= 0
    assert CALLS == [("alerts", "Oslo")]


def test_the_calls_error_reaches_after_and_is_reraised(weather):
    rec = Recorder()
    with observing(rec), pytest.raises(LookupError, match="no station in Oslo"):
        weather.fails("Oslo")
    _, _, _, outcome = rec.log[1]
    assert isinstance(outcome["error"], LookupError)
    assert outcome["result"] is None and outcome["intercepted"] is False


def test_before_runs_in_push_order_and_the_first_interception_wins(weather):
    log: list = []
    first = Recorder("first", log=log)
    second = Recorder("second", intercept=Intercepted({"spec": 2}), log=log)
    third = Recorder("third", intercept=Intercepted({"spec": 3}), log=log)
    with observing(first), observing(second), observing(third):
        assert weather.set_alert("Oslo") == {"spec": 2}
    assert [(e[0], e[1]) for e in log] == [
        ("first", "before"),
        ("second", "before"),
        ("third", "before"),
        ("first", "after"),
        ("second", "after"),
        ("third", "after"),
    ]
    assert all(e[3]["intercepted"] is True for e in log if e[1] == "after")
    assert CALLS == []


# --------------------------------------------------------------------------- #
#  Interception                                                               #
# --------------------------------------------------------------------------- #


def test_an_interception_replaces_a_sync_call(weather):
    rec = Recorder(intercept=Intercepted({"would_set": "Oslo"}))
    with observing(rec):
        assert weather.set_alert("Oslo") == {"would_set": "Oslo"}
    assert CALLS == []
    _, _, _, outcome = rec.log[1]
    assert outcome["intercepted"] is True
    assert outcome["result"] == {"would_set": "Oslo"} and outcome["error"] is None
    assert isinstance(outcome["started"], float) and outcome["elapsed_s"] >= 0


@pytest.mark.asyncio
async def test_an_interception_replaces_an_async_call(weather, recwarn):
    rec = Recorder(intercept=Intercepted(None))
    with observing(rec):
        assert await weather.clear_alerts("Oslo") is None
    assert CALLS == []
    _, _, _, outcome = rec.log[1]
    assert outcome["intercepted"] is True and outcome["result"] is None
    # The environment's coroutine was never even created.
    assert not [w for w in recwarn if "was never awaited" in str(w.message)]


def test_before_must_return_none_or_intercepted(weather):
    with observing(Recorder(intercept={"not": "wrapped"})):
        with pytest.raises(TypeError, match="Intercepted"):
            weather.set_alert("Oslo")
    assert CALLS == []


# --------------------------------------------------------------------------- #
#  Observer failures                                                          #
# --------------------------------------------------------------------------- #


@contextlib.contextmanager
def _logged():
    """The warnings the seam logs (the ``unify`` logger does not propagate to caplog)."""
    records: list = []
    handler = logging.Handler(logging.WARNING)
    handler.emit = records.append  # type: ignore[method-assign]
    log = logging.getLogger(observers.__name__)
    log.addHandler(handler)
    try:
        yield records
    finally:
        log.removeHandler(handler)


def test_an_exception_in_before_propagates_and_nothing_runs(weather):
    rec = Recorder(before_raises=RuntimeError("speculation broke"))
    with observing(rec), pytest.raises(RuntimeError, match="speculation broke"):
        weather.set_alert("Oslo")
    assert CALLS == []
    assert [e[1] for e in rec.log] == ["before"]


@pytest.mark.asyncio
async def test_an_exception_in_before_propagates_on_the_async_path(weather):
    with observing(Recorder(before_raises=RuntimeError("closed"))):
        with pytest.raises(RuntimeError, match="closed"):
            await weather.clear_alerts("Oslo")
    assert CALLS == []


def test_an_exception_in_after_is_logged_and_the_outcome_stands(weather):
    log: list = []
    broken = Recorder("broken", after_raises=ValueError("ledger full"), log=log)
    later = Recorder("later", log=log)
    with _logged() as records:
        with observing(broken), observing(later):
            assert weather.forecast("Oslo") == {"city": "Oslo", "temp_c": 21}
            with pytest.raises(LookupError):
                weather.fails("Oslo")
    messages = [r.getMessage() for r in records]
    assert len(messages) == 2
    assert "weather.forecast" in messages[0] and "ledger full" in messages[0]
    assert "weather.fails" in messages[1]
    # Every observer's after() ran even though an earlier one failed.
    assert [(e[0], e[1]) for e in log if e[1] == "after"] == [
        ("broken", "after"),
        ("later", "after"),
    ] * 2


@pytest.mark.asyncio
async def test_an_exception_in_after_is_swallowed_on_the_async_path(weather):
    with _logged() as records:
        with observing(Recorder(after_raises=ValueError("ledger full"))):
            assert await weather.alerts("Oslo") == ["storm over Oslo"]
    assert ["ledger full" in r.getMessage() for r in records] == [True]


# --------------------------------------------------------------------------- #
#  Scoping                                                                     #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_an_observer_pushed_in_one_task_is_not_seen_by_a_concurrent_one(
    weather,
):
    rec = Recorder()
    pushed = asyncio.Event()
    other_done = asyncio.Event()

    async def what_if_cell():
        with observing(rec):
            pushed.set()
            await other_done.wait()
            await weather.alerts("inside")

    async def concurrent_tool_call():
        await pushed.wait()
        assert observers.current() == ()
        weather.forecast("outside")
        await weather.alerts("outside")
        other_done.set()

    await asyncio.wait_for(
        asyncio.gather(what_if_cell(), concurrent_tool_call()),
        timeout=10,
    )
    seen = [(e[2].method, e[2].args) for e in rec.log if e[1] == "before"]
    assert seen == [("alerts", ("inside",))]
    assert observers.current() == ()


# --------------------------------------------------------------------------- #
#  Recorded cases are unchanged with an observer active                       #
# --------------------------------------------------------------------------- #

TOUCH = (
    "def touch_{n}(city: str) -> int:\n"
    "    primitives.weather.forecast(city)\n"
    "    primitives.weather.set_alert(city=city)\n"
    "    return len(city)\n"
)


def _case_view(case: store_cases.Case) -> tuple:
    return (
        case.kind,
        case.status,
        case.args_shown,
        case.call,
        case.result,
        case.error,
        case.trace,
        case.trace_complete,
    )


@_handle_project
def test_recorded_cases_are_the_same_with_an_observer_active(
    weather,
    monkeypatch,
    python_in_process,
):
    fm = FunctionManager(include_primitives=False)
    fm.add_functions(implementations=[TOUCH.format(n="a"), TOUCH.format(n="b")])
    namespace = {"primitives": SimpleNamespace(weather=weather)}
    fm.list_functions(_return_callable=True, _namespace=namespace)

    assert namespace["touch_a"]("Oslo") == 4
    rec = Recorder()
    with observing(rec):
        assert namespace["touch_b"]("Oslo") == 4

    ids = fm.list_function_name_to_ids()
    (plain,) = store_cases.cases(int(ids["touch_a"]))
    (observed,) = store_cases.cases(int(ids["touch_b"]))
    assert _case_view(observed) == _case_view(plain)
    assert [c["call"] for c in observed.trace] == [
        "weather.forecast",
        "weather.set_alert",
    ]
    assert [e[2].method for e in rec.log if e[1] == "before"] == [
        "forecast",
        "set_alert",
    ]


# --------------------------------------------------------------------------- #
#  Raw globals                                                                #
# --------------------------------------------------------------------------- #


def test_switch_on_a_raw_global_call_is_observed_with_no_effect(weather, proxy_on):
    assert observers.proxy_enabled()
    apis = create_execution_globals()["apis"]
    assert apis is not APIS and repr(apis) == "<apis>"
    assert apis.version == "1.0"  # plain data is returned as it is
    rec = Recorder()
    with observing(rec):
        assert apis.spotify.login(username="ada") == {
            "token": "t-ada",  # pragma: allowlist secret
        }
    b, a = rec.log
    assert b[2] == EnvCall(
        namespace="apis",
        method="spotify.login",
        effect="",
        args=(),
        kwargs={"username": "ada"},
        via="global",
    )
    assert a[3]["result"] == {"token": "t-ada"}  # pragma: allowlist secret
    assert a[3]["intercepted"] is False
    # With no observer the proxy just calls through.
    assert apis.spotify.login(username="bo")["token"] == "t-bo"
    assert len(rec.log) == 2
    assert [c[0] for c in CALLS] == ["spotify.login", "spotify.login"]


@pytest.mark.asyncio
async def test_switch_on_an_async_raw_global_call_is_observed_and_interceptable(
    weather,
    proxy_on,
):
    import inspect

    apis = observers.observed_globals({"apis": APIS})["apis"]
    assert inspect.iscoroutinefunction(apis.spotify.like)
    assert not inspect.iscoroutinefunction(apis.spotify.login)
    assert callable(apis.spotify.login) and not callable(apis.spotify)
    rec = Recorder()
    with observing(rec):
        assert await apis.spotify.like(7) == {"liked": 7}
    assert rec.log[0][2].method == "spotify.like" and rec.log[0][2].args == (7,)
    spec = Recorder(intercept=Intercepted({"liked": "speculative"}))
    with observing(spec):
        assert await apis.spotify.like(8) == {"liked": "speculative"}
        assert apis.spotify.login(username="x") == {"liked": "speculative"}
    assert [e[3]["intercepted"] for e in spec.log if e[1] == "after"] == [True, True]
    assert CALLS == [("spotify.like", 7)]


@pytest.mark.asyncio
async def test_switch_on_a_stored_function_reaches_the_global_through_the_proxy(
    weather,
    proxy_on,
    python_in_process,
):
    fm = FunctionManager(include_primitives=False)
    rec = Recorder()
    with observing(rec):
        out = await fm._execute_python_function(
            implementation=(
                "def sign_in(name: str) -> str:\n"
                "    return apis.spotify.login(username=name)['token']\n"
            ),
            call_kwargs={"name": "ada"},
        )
    assert out["error"] is None and out["result"] == "t-ada"
    assert [(e[1], e[2].method, e[2].via) for e in rec.log] == [
        ("before", "spotify.login", "global"),
        ("after", "spotify.login", "global"),
    ]


def test_a_complete_observer_refuses_what_it_cannot_dispatch(weather, proxy_on):
    apis = create_execution_globals()["apis"]
    # Not complete: the proxy is transparent, private attributes included.
    with observing(Recorder()):
        assert apis.spotify._private() == "hidden"
        assert [type(app).__name__ for app in apis] == ["_App"]
    with observing(Recorder(complete=True)):
        assert observers.complete_required()
        with pytest.raises(ToolInputError) as refused:
            apis.spotify._private()
        assert "primitives.spotify" in str(refused.value)
        with pytest.raises(ToolInputError, match="primitives.spotify.login"):
            apis.spotify.login._unwrapped  # noqa: B018
        with pytest.raises(ToolInputError):
            list(apis)
        # Probes with a default still see a missing attribute.
        assert getattr(apis, "__wrapped__", None) is None
        # What it can dispatch it still sees.
        assert apis.spotify.login(username="ada")["token"] == "t-ada"
    assert not observers.complete_required()
