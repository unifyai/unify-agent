"""Symbolic: an environment registers its own ``primitives.<name>`` namespaces.

A registered namespace must be treated like ``primitives.actor`` everywhere:
the scope, the sandbox's ``primitives`` object, the seeded primitive rows,
``depends_on`` and injection of a stored function that calls it. With nothing
registered (``UNIFY_ENV_NAMESPACES`` unset) every one of those is as shipped.
No model is called.
"""

from __future__ import annotations

import pytest

from tests.helpers import _handle_project
from unify.function_manager.execution_env import (
    ENVIRONMENT_MODULES,
    create_execution_globals,
    environment_modules,
)
from unify.function_manager.function_manager import FunctionManager
from unify.function_manager.primitives import (
    EnvironmentMethod,
    EnvironmentNamespace,
    EnvironmentSurface,
    Primitives,
    PrimitiveScope,
    VALID_MANAGER_ALIASES,
    default_runtime_scope,
    get_primitive_callable,
    get_registry,
    register_environment,
    valid_manager_aliases,
)
from unify.function_manager.primitives import environment as env_module
from unify.function_manager.primitives.environment import (
    EnvironmentNamespaceError,
    clear_environment_namespaces,
    load_from_spec,
)

CALLS: list[tuple[str, dict]] = []


class _WeatherService:
    """A stand-in environment: one read, one write and one destructive method."""

    def forecast(self, **kwargs):
        CALLS.append(("forecast", kwargs))
        return {"city": kwargs.get("city"), "temp_c": 21}

    def set_alert(self, **kwargs):
        CALLS.append(("set_alert", kwargs))
        return {"alert": "set"}

    def delete_alerts(self, **kwargs):
        CALLS.append(("delete_alerts", kwargs))
        return {"deleted": 2}


SERVICE = _WeatherService()


def weather_surface() -> EnvironmentSurface:
    """A factory, as ``UNIFY_ENV_NAMESPACES`` names one."""
    return EnvironmentSurface(
        namespaces=(
            EnvironmentNamespace(
                name="weather",
                description="A weather service",
                methods=(
                    EnvironmentMethod(
                        name="forecast",
                        call=SERVICE.forecast,
                        effect="read",
                        signature="(city: str)",
                        docstring=(
                            "The forecast for a city.\n\nParameters\n----------\n"
                            "city : str\n    The city's name."
                        ),
                    ),
                    EnvironmentMethod(
                        name="set_alert",
                        call=SERVICE.set_alert,
                        effect="write",
                        signature="(city: str)",
                    ),
                    EnvironmentMethod(
                        name="delete_alerts",
                        call=SERVICE.delete_alerts,
                        effect="destructive",
                    ),
                ),
            ),
        ),
        modules=frozenset({"json"}),
        globals={"weather_client": SERVICE},
    )


def _reseed() -> None:
    """Rows are seeded once per store; make the table match the registry again."""
    from unify.function_manager import function_manager as fm_module

    fm_module._PRIMITIVES_SEEDED_FOR.clear()
    FunctionManager()


@pytest.fixture
def weather():
    clear_environment_namespaces()
    register_environment(weather_surface(), source="tests:weather_surface")
    _reseed()
    CALLS.clear()
    yield
    clear_environment_namespaces()
    _reseed()


# ────────────────────────────────────────────────────────────────────────────
# Nothing registered: as shipped
# ────────────────────────────────────────────────────────────────────────────


def test_nothing_registered_changes_nothing():
    clear_environment_namespaces()
    assert valid_manager_aliases() == VALID_MANAGER_ALIASES == {"actor"}
    assert default_runtime_scope().scoped_managers == {"actor"}
    assert set(get_registry().collect_primitives()) == {"primitives.actor.act"}
    assert environment_modules() is ENVIRONMENT_MODULES
    assert "weather_client" not in create_execution_globals()
    with pytest.raises(AttributeError):
        Primitives().weather
    with pytest.raises(ValueError, match="Invalid manager aliases"):
        PrimitiveScope(scoped_managers=frozenset({"weather"}))


# ────────────────────────────────────────────────────────────────────────────
# Registered: the registry, the scope, the sandbox object
# ────────────────────────────────────────────────────────────────────────────


def test_registered_namespace_is_in_the_default_scope(weather):
    assert valid_manager_aliases() == {"actor", "weather"}
    assert default_runtime_scope().scoped_managers == {"actor", "weather"}
    assert PrimitiveScope.all_managers().scoped_managers == {"actor", "weather"}
    registry = get_registry()
    assert registry.primitive_methods(manager_alias="weather") == [
        "delete_alerts",
        "forecast",
        "set_alert",
    ]
    assert "primitives.weather.forecast" in registry.tool_names(default_runtime_scope())
    assert (
        registry.get_manager_spec("weather").primitive_class_path
        == "environment:weather"
    )
    assert "environment:weather" in registry.primitive_row_filter(
        default_runtime_scope(),
    )


def test_primitives_resolves_the_namespace_with_its_docs(weather):
    primitives = Primitives()
    assert primitives.weather.forecast(city="Oslo") == {"city": "Oslo", "temp_c": 21}
    assert CALLS == [("forecast", {"city": "Oslo"})]
    doc = primitives.weather.forecast.__doc__
    assert doc.startswith("The forecast for a city.")
    assert "`primitives.weather.forecast(city: str)` (effect: read)" in doc
    assert "city : str" in doc
    assert dir(primitives.weather) == ["delete_alerts", "forecast", "set_alert"]
    with pytest.raises(AttributeError, match="has no method 'forcast'"):
        primitives.weather.forcast
    scoped = Primitives(
        primitive_scope=PrimitiveScope(scoped_managers=frozenset({"actor"})),
    )
    with pytest.raises(AttributeError, match="not available to this actor"):
        scoped.weather


def test_rows_carry_signature_doc_and_effect(weather):
    rows = get_registry().collect_primitives()
    row = rows["primitives.weather.set_alert"]
    assert row["primitive_class"] == "environment:weather"
    assert row["argspec"] == "(city: str)"
    assert row["metadata"] == {"effect": "write", "environment": "weather"}
    assert "(effect: write)" in row["docstring"]
    assert (
        rows["primitives.weather.delete_alerts"]["metadata"]["effect"] == "destructive"
    )
    assert get_primitive_callable(rows["primitives.weather.forecast"])(
        city="Bergen",
    ) == {
        "city": "Bergen",
        "temp_c": 21,
    }


@_handle_project
def test_seeded_rows_are_searchable_primitives(weather):
    fm = FunctionManager()
    names = {row["name"] for row in fm._primitive_logs()}
    assert {"primitives.actor.act", "primitives.weather.forecast"} <= names
    listed = fm.list_functions()
    assert listed["primitives.weather.forecast"]["is_primitive"] is True
    assert listed["primitives.weather.forecast"]["metadata"]["effect"] == "read"


def test_environment_modules_and_globals(weather):
    assert environment_modules() == ENVIRONMENT_MODULES | {"json"}
    globals_ = create_execution_globals()
    assert globals_["weather_client"] is SERVICE
    # The sandbox's default ``primitives`` exposes nothing: an actor's
    # environments inject the namespaces it is granted.
    assert globals_["primitives"].primitive_scope.scoped_managers == frozenset()


# ────────────────────────────────────────────────────────────────────────────
# Stored functions: depends_on and injection, as for primitives.actor
# ────────────────────────────────────────────────────────────────────────────


@_handle_project
def test_stored_function_records_and_gets_the_namespace(weather):
    fm = FunctionManager()
    source = (
        "def two_forecasts(a: str, b: str) -> list:\n"
        '    """Forecasts for two cities."""\n'
        "    return [primitives.weather.forecast(city=a), primitives.weather.forecast(city=b)]\n"
    )
    assert fm.add_functions(implementations=[source]) == {"two_forecasts": "added"}
    assert fm.list_functions()["two_forecasts"]["depends_on"] == [
        "primitives.weather.forecast",
    ]

    namespace: dict = {}  # no ``primitives`` in it: injection must supply one
    fm.filter_functions(
        filter="name = 'two_forecasts'",
        _return_callable=True,
        _namespace=namespace,
    )
    assert namespace["two_forecasts"]("Oslo", "Rome") == [
        {"city": "Oslo", "temp_c": 21},
        {"city": "Rome", "temp_c": 21},
    ]


# ────────────────────────────────────────────────────────────────────────────
# Loading factories, and what a surface may not do
# ────────────────────────────────────────────────────────────────────────────


def test_factory_spec_loads_and_validates():
    surfaces = load_from_spec(f"{__name__}:weather_surface")
    assert [n.name for n in surfaces[0].namespaces] == ["weather"]
    assert surfaces[0].source == f"{__name__}:weather_surface"

    with pytest.raises(EnvironmentNamespaceError, match="not 'package.module:factory'"):
        load_from_spec("no_colon_here")
    with pytest.raises(EnvironmentNamespaceError, match="cannot be imported"):
        load_from_spec("no_such_module_xyz:factory")


def _raising_factory():
    raise RuntimeError("the environment is down")


def test_failing_factory_or_bad_surface_raises():
    with pytest.raises(EnvironmentNamespaceError, match="the environment is down"):
        load_from_spec(f"{__name__}:_raising_factory")

    def surface(**method):
        return {
            "namespaces": [
                {
                    "name": method.pop("namespace", "svc"),
                    "methods": [
                        {"name": "m", "call": print, "effect": "read", **method},
                    ],
                },
            ],
        }

    with pytest.raises(EnvironmentNamespaceError, match="effect 'mutate'"):
        env_module.coerce_surface(surface(effect="mutate"))
    with pytest.raises(EnvironmentNamespaceError, match="belongs to Unify"):
        env_module.coerce_surface(surface(namespace="actor"))
    with pytest.raises(
        EnvironmentNamespaceError,
        match="not a public Python identifier",
    ):
        env_module.coerce_surface(surface(name="not-a-name"))
    with pytest.raises(EnvironmentNamespaceError, match="global 'primitives'"):
        env_module.coerce_surface({"globals": {"primitives": object()}})
    # A mapping with the same fields is as good as the dataclasses.
    assert (
        env_module.coerce_surface(surface()).namespaces[0].methods[0].effect == "read"
    )


def test_the_switch_names_the_factory(monkeypatch):
    from unify.settings import SETTINGS

    clear_environment_namespaces()
    monkeypatch.setattr(SETTINGS, "UNIFY_ENV_NAMESPACES", f"{__name__}:weather_surface")
    try:
        assert valid_manager_aliases() == {"actor", "weather"}
    finally:
        monkeypatch.setattr(SETTINGS, "UNIFY_ENV_NAMESPACES", "")
        clear_environment_namespaces()
        _reseed()
    assert valid_manager_aliases() == {"actor"}
