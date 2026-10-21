"""Symbolic: an environment registers its own ``primitives.<name>`` namespaces.

A registered namespace is bound in the actor's cells as ``primitives.<name>``
and listed in the actor's prompt; its globals join the cells' namespace. With
nothing registered (``UNIFY_ENV_NAMESPACES`` unset) the actor has no
``primitives`` environment. No model is called.
"""

from __future__ import annotations

import pytest

from unify.function_manager.execution_env import create_execution_globals
from unify.function_manager.primitives import (
    EnvironmentMethod,
    EnvironmentNamespace,
    EnvironmentSurface,
    Primitives,
    register_environment,
)
from unify.function_manager.primitives import environment as env_module
from unify.function_manager.primitives.environment import (
    EnvironmentNamespaceError,
    clear_environment_namespaces,
    environment_aliases,
    environment_modules,
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


@pytest.fixture
def weather():
    clear_environment_namespaces()
    register_environment(weather_surface(), source="tests:weather_surface")
    CALLS.clear()
    yield
    clear_environment_namespaces()


def test_nothing_registered_binds_no_primitives():
    from unify.actor.environments import registered_environments

    clear_environment_namespaces()
    assert environment_aliases() == frozenset()
    assert registered_environments() == []
    assert dir(Primitives()) == []
    with pytest.raises(AttributeError, match="the environment registered: none"):
        Primitives().weather


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


def test_environment_modules_and_globals(weather):
    assert environment_modules() == frozenset({"json"})
    assert create_execution_globals()["weather_client"] is SERVICE


def test_the_actor_environment_lists_the_namespaces(weather):
    from unify.actor.environments import (
        EnvironmentNamespacesEnvironment,
        registered_environments,
    )
    from unify.actor.prompt_builders import build_code_act_prompt

    envs = registered_environments()
    assert len(envs) == 1 and isinstance(envs[0], EnvironmentNamespacesEnvironment)
    assert envs[0].namespace == "primitives"
    context = envs[0].get_prompt_context()
    assert "| `primitives.weather` | 3 (1 / 1 / 1) | A weather service |" in context
    assert "`weather_client`" in context and "`json`" in context
    prompt = build_code_act_prompt(environments={"primitives": envs[0]})
    assert context in prompt


def test_the_actor_prompt_names_no_namespace_when_nothing_is_registered():
    from unify.actor.prompt_builders import build_code_act_prompt

    clear_environment_namespaces()
    assert "primitives." not in build_code_act_prompt(environments={})


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


def test_the_setting_names_the_factory(monkeypatch):
    from unify.settings import SETTINGS

    clear_environment_namespaces()
    monkeypatch.setattr(SETTINGS, "UNIFY_ENV_NAMESPACES", f"{__name__}:weather_surface")
    try:
        assert environment_aliases() == {"weather"}
    finally:
        monkeypatch.setattr(SETTINGS, "UNIFY_ENV_NAMESPACES", "")
        clear_environment_namespaces()
    assert environment_aliases() == frozenset()
