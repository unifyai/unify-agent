"""Symbolic: ``UNIFY_ENV_CARDS``, what earlier sessions verified about the environment, listed when a session starts.

Calls go through the environment seam of a registered test environment
(``weather``, plus a raw ``apis`` global as AppWorld binds one). Each test
records "sessions" with the recorder and reads the section a later session
would get. Nothing leaves the process.
"""

from __future__ import annotations

import sqlite3
from contextlib import closing

import numpy as np
import pytest

from tests.function_manager.primitives.test_env_observers import (  # noqa: F401 (fixture)
    weather,
)
from unify.actor import env_cards as ec
from unify.function_manager.primitives import observers
from unify.function_manager.primitives.observers import EnvCall
from unify.settings import ProductionSettings, SETTINGS

# The stand-in embedder: a fixed vector per text, so closeness is set by the test, never by shared words.
WEATHERY = [1.0, 0.0, 0.1]
MUSICAL = [0.0, 1.0, 0.1]
VECTORS: dict = {}


def _stand_in_embed(texts):
    return np.array([VECTORS.get(t, [0.0, 0.0, 1.0]) for t in texts], dtype=np.float32)


@pytest.fixture
def on(monkeypatch):
    monkeypatch.setattr(SETTINGS, "UNIFY_ENV_CARDS", True)
    monkeypatch.setattr(ec, "_embed", _stand_in_embed)
    VECTORS.clear()


def _session(run, request=""):
    """One session: its calls go through the seam under its own recorder."""
    scope, _ = ec.enter(request)
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
    VECTORS.update(
        {"Is it raining in Oslo?": WEATHERY, "Forecast for Bergen?": WEATHERY},
    )

    def first():
        with pytest.raises(LookupError):
            weather.fails(city="Oslo")
        weather.forecast(city="Oslo")

    _session(first, "Is it raining in Oslo?")
    request = "Forecast for Bergen?"
    text = ec.section(request, vector=ec._vector(request))
    assert text.startswith(ec.HEADER)
    assert "### weather" in text
    assert "`weather.forecast(city)` → {city: str, temp_c: int}" in text
    assert (
        "`weather.fails(city)` failed (LookupError: no station in …; 1 session)" in text
    )
    assert "then `weather.forecast(city)` worked" in text


def _fake_session(session, vector, group, method):
    ec._keep_session(session, ec._vector(vector) if isinstance(vector, str) else vector)
    rec = ec.Recorder(session)
    namespace, _, rest = group.partition(".")
    call = EnvCall(
        namespace=namespace,
        method=f"{rest}.{method}" if rest else method,
        effect="",
        args=(),
        kwargs={"q": 1},
        via="global",
    )
    rec.after(
        call,
        result={"ok": True},
        error=None,
        intercepted=False,
        started=0.0,
        elapsed_s=0.0,
    )


def test_a_group_is_shown_when_a_close_earlier_session_used_it_or_most_did(on):
    VECTORS.update(
        {
            "rain": WEATHERY,
            "song": MUSICAL,
            "Will it rain tomorrow?": WEATHERY,
            "Play something calm.": MUSICAL,
        },
    )
    _fake_session("s1", "rain", "apis.weather", "forecast")
    for name in ("s2", "s3", "s4"):
        _fake_session(name, "song", "apis.spotify", "play")
    near = ec.section(
        "Will it rain tomorrow?",
        vector=ec._vector("Will it rain tomorrow?"),
    )
    far = ec.section("Play something calm.", vector=ec._vector("Play something calm."))
    assert "### apis.weather" in near  # its closest earlier session used it
    assert "### apis.weather" not in far  # 1 of 4 sessions, none of them close
    assert "### apis.spotify" in near and "### apis.spotify" in far  # 3 of 4


def test_a_shared_word_never_selects_a_group(on):
    """The request names "weather" but is close, by embedding, only to the music sessions."""
    request = "Add the song Stormy Weather to my weather-free playlist."
    VECTORS.update({"rain": WEATHERY, "song": MUSICAL, request: MUSICAL})
    _fake_session("s1", "rain", "apis.weather", "forecast")
    for name in ("s2", "s3", "s4"):
        _fake_session(name, "song", "apis.spotify", "play")
    assert "### apis.weather" not in ec.section(request, vector=ec._vector(request))


def test_an_embedding_failure_leaves_only_the_share_rule(on, monkeypatch):
    def broken(texts):
        raise RuntimeError("provider down")

    _fake_session("s1", WEATHERY_BYTES(), "apis.weather", "forecast")
    for name in ("s2", "s3", "s4"):
        _fake_session(name, MUSICAL_BYTES(), "apis.spotify", "play")
    monkeypatch.setattr(ec, "_embed", broken)
    assert ec._vector("Will it rain?") is None
    text = ec.section("Will it rain?", vector=None)
    assert "### apis.spotify" in text and "### apis.weather" not in text


def _unit(v):
    a = np.array(v, dtype=np.float32)
    return (a / np.linalg.norm(a)).astype(np.float32).tobytes()


def WEATHERY_BYTES():
    return _unit(WEATHERY)


def MUSICAL_BYTES():
    return _unit(MUSICAL)


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
