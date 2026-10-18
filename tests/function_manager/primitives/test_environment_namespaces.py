"""Symbolic: an environment registers its own ``primitives.<name>`` namespaces.

A registered namespace must be treated like ``primitives.actor`` everywhere:
the scope, the sandbox's ``primitives`` object, the seeded primitive rows,
``depends_on`` and injection of a stored function that calls it, the actor's
environments and prompt, and the storage review's doctrine. With nothing
registered (``UNIFY_ENV_NAMESPACES`` unset) every one of those is as shipped.
No model is called.
"""

from __future__ import annotations

import re

from unify.actor.core_surface import PromptSurface
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
@pytest.mark.parametrize("python", ["worker", ""])
def test_stored_function_records_and_gets_the_namespace(weather, python, monkeypatch):
    if not python:
        # The function manager's in-process mode (non-actor callers).
        monkeypatch.setattr("unify.actor.execution.worker.enabled", lambda: False)
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
    if python:
        # Bound by its source for the sandboxed worker, with the namespace
        # root it reaches the environment through.
        assert namespace["two_forecasts"].source == source
        assert namespace["primitives"].weather.forecast(city="Oslo") == {
            "city": "Oslo",
            "temp_c": 21,
        }
        return
    assert namespace["two_forecasts"]("Oslo", "Rome") == [
        {"city": "Oslo", "temp_c": 21},
        {"city": "Rome", "temp_c": 21},
    ]


# ────────────────────────────────────────────────────────────────────────────
# The actor: its environments, prompt and the storage review's doctrine
# ────────────────────────────────────────────────────────────────────────────


def test_actor_environment_lists_the_namespaces(weather):
    from unify.actor.environments import (
        ActorEnvironment,
        EnvironmentNamespacesEnvironment,
        registered_environments,
    )
    from unify.actor.environments.base import _CompositeEnvironment

    envs = registered_environments()
    assert len(envs) == 1 and isinstance(envs[0], EnvironmentNamespacesEnvironment)
    tools = envs[0].get_tools()
    assert tools["primitives.weather.forecast"].is_impure is False
    assert tools["primitives.weather.set_alert"].is_impure is True
    assert envs[0].prompt_documented_names == frozenset()
    context = envs[0].get_prompt_context()
    assert "| `primitives.weather` | 3 (1 / 1 / 1) | A weather service |" in context
    assert "`weather_client`" in context and "`json`" in context

    merged = _CompositeEnvironment([ActorEnvironment(), *envs])
    assert merged.get_instance().primitive_scope.scoped_managers == {"actor", "weather"}
    assert "primitives.actor.act" in merged.get_tools()
    # The actor's inline docs are excluded from search as before; the
    # environment's methods stay searchable.
    assert "primitives.weather.forecast" not in merged.prompt_documented_names
    assert (
        "The one `primitives.*` surface." not in ActorEnvironment().get_prompt_context()
    )


def test_namespaces_context_names_the_library_only_when_it_is_on(weather, monkeypatch):
    from unify.actor.environments import registered_environments
    from unify.settings import SETTINGS

    on_text = (
        "`primitives.<namespace>.<method>(...)`. A stored function that calls them needs "
        "no import: they are recorded and injected when it runs later. Each method's "
        "signature and documentation come back from `await functions.search(...)` "
        "(as primitive rows) and from `help(primitives.<namespace>.<method>)`. "
    )
    monkeypatch.setattr(SETTINGS, "UNIFY_LIBRARY", "")
    on = registered_environments()[0].get_prompt_context()
    assert on_text in on

    monkeypatch.setattr(SETTINGS, "UNIFY_LIBRARY", "off")
    off = registered_environments()[0].get_prompt_context()
    assert "functions" not in off and "stored function" not in off
    assert (
        "Each method's signature and documentation come back from "
        "`help(primitives.<namespace>.<method>)`. "
    ) in off
    assert "| `primitives.weather` | 3 (1 / 1 / 1) | A weather service |" in off


#: What the namespaces section says only where the sandbox binds ``functions``.
LIBRARY_SENTENCES = (
    "A stored function that calls them needs no import: they are recorded and "
    "injected when it runs later. ",
    "`await functions.search(...)` (as primitive rows) and from ",
)


def _without_library(text: str) -> str:
    for sentence in LIBRARY_SENTENCES:
        text = text.replace(sentence, "")
    return text


def test_namespaces_context_names_functions_only_where_the_sandbox_binds_it(
    weather,
    monkeypatch,
):
    """A memory run's sandbox has no ``functions``, yet the section told the
    actor each method's docs come back from ``await functions.search(...)``:
    on tau2 airline that call raised NameError 96 times in 99. The section now
    follows the library objects the sandbox binds; the rest is unchanged."""
    from unify.actor.core_surface import bound_libraries
    from unify.actor.environments import registered_environments
    from unify.settings import SETTINGS

    env = registered_environments()[0]
    library = env.get_prompt_context_for(frozenset({"functions", "guidance"}))
    assert all(sentence in library for sentence in LIBRARY_SENTENCES)
    for libraries in (frozenset(), frozenset({"guidance"})):
        without = env.get_prompt_context_for(libraries)
        assert "functions" not in without and "stored function" not in without
        assert (
            "Each method's signature and documentation come back from "
            "`help(primitives.<namespace>.<method>)`. "
        ) in without
        assert without == _without_library(library)

    # Outside a session: what a session under the run's switches binds.
    monkeypatch.setattr(SETTINGS, "UNIFY_MEMORY", "")
    assert bound_libraries() == {"functions", "guidance"}
    assert env.get_prompt_context() == library
    monkeypatch.setattr(SETTINGS, "UNIFY_MEMORY", "on")
    assert bound_libraries() == frozenset()
    assert env.get_prompt_context() == _without_library(library)
    monkeypatch.setattr(SETTINGS, "UNIFY_MEMORY", "")
    monkeypatch.setattr(SETTINGS, "UNIFY_LIBRARY", "off")
    assert bound_libraries() == frozenset()
    assert env.get_prompt_context() == _without_library(library)


@pytest.mark.parametrize("in_code", ["on", ""], ids=["tools-in-code", "tools-off"])
def test_the_actor_prompt_names_functions_only_when_its_sandbox_binds_it(
    weather,
    monkeypatch,
    in_code,
):
    """The whole system prompt, with the namespaces registered: a session
    whose sandbox binds no library names no ``functions`` anywhere."""
    from unify.actor.environments import registered_environments
    from unify.actor.prompt_builders import build_code_act_prompt
    from unify.settings import SETTINGS

    monkeypatch.setattr(SETTINGS, "UNIFY_TOOLS_IN_CODE", in_code)
    environments = {"primitives": registered_environments()[0]}

    def prompt(core: PromptSurface) -> str:
        return build_code_act_prompt(
            environments=environments,
            core=core,
            session_sections=False,
        )

    bound = prompt(PromptSurface())
    assert "`await functions.search(...)`" in bound
    unbound = prompt(PromptSurface(functions=False, guidance=False))
    assert not re.search(r"\bfunctions\.\w", unbound), unbound
    assert "`functions`" not in unbound, unbound
    assert "`help(primitives.<namespace>.<method>)`" in unbound
    assert "| `primitives.weather` | 3 (1 / 1 / 1) | A weather service |" in unbound


@pytest.mark.asyncio
@pytest.mark.timeout(60)
@_handle_project
async def test_child_without_sub_agents_keeps_the_namespaces(weather):
    """A child that may not spawn still works in its parent's environment: it
    holds and calls the namespaces, and is offered no sub-actor."""
    from unify.actor.environments.actor import _build_inner_actor
    from unify.actor.execution import PythonExecutionSession, _CURRENT_SANDBOX
    from unify.actor.prompt_builders import build_code_act_prompt

    child, _guidelines = _build_inner_actor(
        guidelines=None,
        prompt_guidance=None,
        guidance_scope=None,
        prompt_functions=["primitives.weather.forecast"],
        discovery_scope=None,
        timeout=30,
        can_compose=True,
        can_store=False,
        can_spawn_sub_agents=False,
    )
    assert child.function_manager._primitive_scope.scoped_managers == {"weather"}
    tools = child.get_tools("act")
    prompt = build_code_act_prompt(
        environments=child.environments,
        core=PromptSurface(),
    )
    assert "`primitives.weather`" in prompt
    assert "primitives.actor" not in prompt

    sandbox = PythonExecutionSession(environments=child.environments)
    token = _CURRENT_SANDBOX.set(sandbox)
    try:
        await tools["execute_code"](
            thought="Checking the forecast.",
            code="primitives.weather.forecast(city='Oslo')",
        )
        refused = await tools["execute_code"](
            thought="Delegating to a sub-actor.",
            code="await primitives.actor.act(request='What is 2 + 2?')",
        )
    finally:
        _CURRENT_SANDBOX.reset(token)
    assert CALLS == [("forecast", {"city": "Oslo"})]
    error = refused.get("error") if isinstance(refused, dict) else refused.error
    assert "primitives.actor is not available to this actor" in (error or "")


def test_actor_prompt_and_review_doctrine_unchanged_when_nothing_registered():
    from unify.actor.code_act_actor import _storage_environment_note
    from unify.actor.environments import ActorEnvironment, registered_environments

    clear_environment_namespaces()
    assert registered_environments() == []
    # The note always states the storage check (baked in); it names no
    # namespace when nothing is registered.
    note = _storage_environment_note()
    assert "checks each function before storing it" in note
    assert "registered its own namespaces" not in note
    assert "The one `primitives.*` surface." in ActorEnvironment().get_prompt_context()


def test_review_doctrine_names_the_namespaces(weather):
    from unify.actor.code_act_actor import _storage_environment_note

    note = _storage_environment_note()
    assert "`primitives.weather`" in note
    assert "`weather_client`" in note
    assert "never declare them as `dependencies`" in note


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
