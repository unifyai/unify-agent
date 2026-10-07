"""Symbolic: ``UNIFY_ENV_CARDS``, what earlier sessions verified about the environment, listed when a session starts.

Calls go through the environment seam of a registered test environment
(``weather``, plus a raw ``apis`` global as AppWorld binds one). Each test
records "sessions" with the recorder and reads the section a later session
would get. Nothing leaves the process.
"""

from __future__ import annotations

import sqlite3
from contextlib import closing

import pytest

from tests.function_manager.primitives.test_env_observers import (  # noqa: F401 (fixture)
    weather,
)
from unify.actor import env_cards as ec
from unify.function_manager.primitives import observers
from unify.function_manager.primitives.observers import EnvCall
from unify.settings import ProductionSettings, SETTINGS


@pytest.fixture
def on(monkeypatch):
    monkeypatch.setattr(SETTINGS, "UNIFY_ENV_CARDS", True)


def _session(run):
    """One session: its calls go through the seam under its own recorder."""
    scope, _ = ec.enter("")
    try:
        run()
    finally:
        ec.leave(scope)


def _rows():
    with closing(ec._connect()) as conn:
        return conn.execute(
            f"SELECT grp, method, args, ok, error, shape FROM {ec.TABLE} ORDER BY seq",
        ).fetchall()


def test_the_switch_is_off_by_default():
    assert ProductionSettings.model_fields["UNIFY_ENV_CARDS"].default is False


def test_off_no_recorder_no_section_and_globals_unproxied(monkeypatch):
    monkeypatch.setattr(SETTINGS, "UNIFY_ENV_CARDS", False)
    assert ec.enter("Check the weather in Oslo.") == (None, "")
    assert observers.current() == ()
    assert not observers.proxy_enabled()


def test_on_the_raw_globals_are_proxied(on):
    assert observers.proxy_enabled()


def test_a_call_keeps_argument_names_and_response_shape_never_values(on, weather):
    _session(lambda: weather.forecast(city="Oslo"))
    [(grp, method, args, ok, error, shape)] = _rows()
    assert (grp, method, args, ok, error) == ("weather", "forecast", "city", 1, None)
    assert shape == "{city: str, temp_c: int}"
    assert "Oslo" not in str(_rows())


def test_a_failed_call_keeps_its_error_without_the_calls_values(on, weather):
    def run():
        with pytest.raises(LookupError):
            weather.fails(city="Oslo")

    _session(run)
    [(_, method, _, ok, error, shape)] = _rows()
    assert (method, ok, shape) == ("fails", 0, None)
    assert error == "LookupError: no station in …"


def test_an_authentication_call_keeps_no_response_shape(on):
    rec = ec.Recorder("s1")
    call = EnvCall(
        namespace="apis",
        method="spotify.login",
        effect="",
        args=(),
        kwargs={"username": "kim", "password": "pw"},  # pragma: allowlist secret
        via="global",
    )
    rec.after(
        call,
        result={"access_token": "t"},  # pragma: allowlist secret
        error=None,
        intercepted=False,
        started=0.0,
        elapsed_s=0.0,
    )
    assert _rows() == [("apis.spotify", "login", "username, password", 1, None, None)]
    assert "kim" not in str(_rows())


def test_a_raw_global_call_is_grouped_by_its_app(on):
    rec = ec.Recorder("s1")
    call = EnvCall(
        namespace="apis",
        method="spotify.show_song",
        effect="",
        args=(),
        kwargs={"song_id": 7},
        via="global",
    )
    rec.after(
        call,
        result={"id": 7, "title": "x"},
        error=None,
        intercepted=False,
        started=0.0,
        elapsed_s=0.0,
    )
    assert _rows() == [
        ("apis.spotify", "show_song", "song_id", 1, None, "{id: int, title: str}"),
    ]


def test_a_value_that_names_the_surface_is_kept_others_are_not():
    names = {"spotify", "show_song"}
    assert (
        ec.describe_args((), {"app_name": "spotify", "api_name": "show_song"}, names)
        == "app_name='spotify', api_name='show_song'"
    )
    assert ec.describe_args(("Oslo",), {"city": "Oslo"}, names) == "…, city"


def test_a_later_session_sees_usage_and_the_call_that_worked_after_a_failure(
    on,
    weather,
):
    def first():
        with pytest.raises(LookupError):
            weather.fails(city="Oslo")
        weather.forecast(city="Oslo")

    _session(first)
    text = ec.section("What is the weather like in Bergen?")
    assert text.startswith(ec.HEADER)
    assert "### weather" in text
    assert "`weather.forecast(city)` → {city: str, temp_c: int}" in text
    assert (
        "`weather.fails(city)` failed (LookupError: no station in …; 1 session)" in text
    )
    assert "then `weather.forecast(city)` worked" in text


def test_a_group_is_shown_when_named_or_used_by_most_sessions(on, weather):
    _session(lambda: weather.forecast(city="Oslo"))
    # One earlier session: the share rule needs two, so only a naming request lists it.
    assert ec.section("Plan my day.") == ""
    assert "### weather" in ec.section("Will the weather hold?")
    _session(lambda: weather.forecast(city="Rome"))
    assert "### weather" in ec.section("Plan my day.")


def test_at_most_k_facts_per_group_heaviest_first(on, weather, monkeypatch):
    monkeypatch.setattr(ec, "K", 2)
    _session(lambda: (weather.forecast(city="A"), weather.set_alert(city="A")))
    _session(lambda: weather.forecast(city="B"))
    _session(lambda: weather.delete_alerts())
    lines = ec.facts("weather")["weather"]
    assert len(lines) == 2
    assert lines[0].startswith("`weather.forecast(city)`") and "2 sessions" in lines[0]


def test_the_current_session_is_never_its_own_evidence(on, weather):
    scope, text = ec.enter("weather please")
    try:
        assert text == ""
        weather.forecast(city="Oslo")
        session = observers.current()[-1].session
        assert ec.facts("weather", exclude_session=session) == {}
    finally:
        ec.leave(scope)


def test_a_sub_agent_keeps_its_callers_recorder(on):
    scope, _ = ec.enter("first")
    try:
        assert ec.enter("nested") == (None, "")
        assert sum(isinstance(o, ec.Recorder) for o in observers.current()) == 1
    finally:
        ec.leave(scope)
    assert observers.current() == ()


def test_a_store_that_cannot_be_written_never_breaks_the_call(on, weather, monkeypatch):
    def broken():
        raise sqlite3.OperationalError("disk full")

    monkeypatch.setattr(ec, "_connect", broken)
    scope, _ = ec.enter("")
    try:
        assert weather.forecast(city="Oslo") == {"city": "Oslo", "temp_c": 21}
    finally:
        ec.leave(scope)
