"""Symbolic: ``UNIFY_STORE_CHECK=resolve`` refuses to store what would not run.

Each refusal names what failed, in the error ``add_functions`` returns (the
storage review reads it as the tool's result), and nothing is stored for the
refused function. The cases are the ways stored functions were broken in a
real run: a namespace that does not exist (``primitives.spotify``), a method
that does not exist (``primitives.actor.apis``), a name only the trajectory
had, ``import primitives``, and a declared dependency that cannot be
installed. With the check off, the same functions are stored as shipped. No
model is called.
"""

from __future__ import annotations

import pytest

from tests.helpers import _handle_project
from unify import environment
from unify.function_manager.function_manager import FunctionManager
from unify.function_manager.primitives import (
    EnvironmentMethod,
    EnvironmentNamespace,
    EnvironmentSurface,
    register_environment,
)
from unify.function_manager.primitives.environment import clear_environment_namespaces
from unify.settings import SETTINGS


def _reseed() -> None:
    from unify.function_manager import function_manager as fm_module

    fm_module._PRIMITIVES_SEEDED_FOR.clear()
    FunctionManager()


@pytest.fixture
def check_on(monkeypatch):
    monkeypatch.setattr(SETTINGS, "UNIFY_STORE_CHECK", "resolve")


@pytest.fixture
def weather():
    clear_environment_namespaces()
    register_environment(
        EnvironmentSurface(
            namespaces=(
                EnvironmentNamespace(
                    name="weather",
                    methods=(
                        EnvironmentMethod(
                            name="forecast",
                            call=lambda **kw: {"temp_c": 21, **kw},
                            effect="read",
                            signature="(city: str)",
                        ),
                    ),
                ),
            ),
            globals={"weather_client": object()},
        ),
        source="tests:weather",
    )
    _reseed()
    yield
    clear_environment_namespaces()
    _reseed()


def _refusal(fm: FunctionManager, source: str, **kwargs) -> str:
    name = source.split("def ", 1)[1].split("(", 1)[0]
    result = fm.add_functions(implementations=[source], raise_on_error=False, **kwargs)
    assert result[name].startswith("error: "), result
    assert name not in fm.list_functions()
    return result[name]


@_handle_project
def test_an_invented_namespace_is_refused_with_the_namespaces_that_exist(
    check_on,
    weather,
):
    fm = FunctionManager()
    message = _refusal(
        fm,
        "def rewind(username: str) -> dict:\n"
        "    return primitives.spotify.login(username=username)\n",
    )
    assert "'rewind' was not stored" in message
    assert "`primitives.spotify` does not exist" in message
    assert "`primitives.actor`, `primitives.weather`" in message


@_handle_project
def test_a_method_that_does_not_exist_is_refused(check_on, weather):
    fm = FunctionManager()
    message = _refusal(
        fm,
        "def call_phone() -> dict:\n    return primitives.actor.apis.phone.show()\n",
    )
    assert "`primitives.actor` has no method `apis`" in message
    assert "its methods are `act`" in message
    message = _refusal(
        fm,
        "def typo() -> dict:\n    return primitives.weather.forcast(city='Oslo')\n",
    )
    assert "did you mean `forecast`?" in message


@_handle_project
def test_a_name_only_the_trajectory_had_is_refused(check_on, weather):
    fm = FunctionManager()
    message = _refusal(
        fm,
        "def uses_apis() -> dict:\n    return apis.spotify.login()\n",
    )
    assert "`apis` is not defined where the function runs" in message
    message = _refusal(
        fm,
        "def bare() -> dict:\n    return weather.forecast(city='x')\n",
    )
    assert "did you mean `primitives.weather`?" in message


@_handle_project
def test_importing_a_sandbox_global_is_refused(check_on):
    fm = FunctionManager()
    message = _refusal(
        fm,
        "def imports() -> int:\n    import primitives\n    return 1\n",
    )
    assert "`primitives` is a sandbox global, not a module" in message


@_handle_project
def test_a_dependency_that_cannot_be_installed_is_refused(check_on, monkeypatch):
    def no_uv(requirements):
        if requirements:
            raise FileNotFoundError(2, "No such file or directory", "uv")

    monkeypatch.setattr(environment, "ensure", no_uv)
    fm = FunctionManager()
    message = _refusal(
        fm,
        "def rewind() -> int:\n    from appworld_client import apis\n    return 1\n",
        dependencies=["appworld-client"],
    )
    assert "does not load the way a search loads it" in message
    assert "FileNotFoundError" in message and "'appworld-client'" in message


@_handle_project
def test_code_that_resolves_is_stored(check_on, weather):
    fm = FunctionManager()
    helper = "def city_label(city: str) -> str:\n    return city.title()\n"
    main = (
        "async def report(cities: list, n: int = 1) -> list:\n"
        '    """Forecasts, labelled."""\n'
        "    import json\n"
        "    rows = [primitives.weather.forecast(city=c) for c in cities]\n"
        "    def label(row):\n"
        "        return city_label(str(row.get('city')))\n"
        "    display(len(rows))\n"
        "    handle = primitives.actor if n < 0 else None\n"
        "    _ = (weather_client, query_llm, handle)\n"
        "    return sorted((label(r) for r in rows), key=lambda s: json.dumps(s))\n"
    )
    assert fm.add_functions(implementations=[main, helper]) == {
        "report": "added",
        "city_label": "added",
    }
    assert fm.list_functions()["report"]["depends_on"] == [
        "city_label",
        "primitives.weather.forecast",
    ]


@_handle_project
def test_the_refusal_reaches_a_caller_that_raises(check_on):
    fm = FunctionManager()
    with pytest.raises(ValueError, match="`primitives.spotify` does not exist"):
        fm.add_functions(
            implementations=["def f() -> None:\n    primitives.spotify.play()\n"],
        )


@_handle_project
def test_without_the_check_the_same_code_is_stored_as_shipped():
    assert SETTINGS.UNIFY_STORE_CHECK == ""
    fm = FunctionManager()
    source = (
        "def rewind(u: str) -> dict:\n    return primitives.spotify.login(username=u)\n"
    )
    assert fm.add_functions(implementations=[source]) == {"rewind": "added"}
    assert fm.list_functions()["rewind"]["depends_on"] == ["primitives.spotify.login"]


def test_the_switch_accepts_only_resolve():
    from unify.settings import ProductionSettings

    assert (
        ProductionSettings(UNIFY_STORE_CHECK="Resolve").UNIFY_STORE_CHECK == "resolve"
    )
    assert ProductionSettings().UNIFY_STORE_CHECK == ""
    with pytest.raises(ValueError, match="UNIFY_STORE_CHECK"):
        ProductionSettings(UNIFY_STORE_CHECK="strict")
