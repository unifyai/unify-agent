"""Symbolic: serving an environment global to the worker never probes it as a request.

AppWorld's ``apis`` answers any attribute name: ``apis.<name>`` looks up an app
and raises ``HTTPException(422, "No app named ...")`` for an unknown one, and
each app does the same for its APIs. Under worker Python
(``UNIFY_TOOL_SURFACE=core``) the harness described ``apis`` for the worker
with ``asyncio.iscoroutinefunction``, whose ``_is_coroutine_marker`` probe
reached that lookup: every cell failed before it ran with "No app named
'_is_coroutine_marker' found", and the zero-cost core rehearsal on 858817f98
solved 0/12 AppWorld tasks against 6/12 without core.

The harness now reads such an object's attributes only statically when it
describes it, so a fake environment whose ``__getattr__`` treats every name
as a remote call sees only the calls the cell makes. Underscored attributes
stay refused to cell code. Cells run in the real sandboxed worker; no model
is called.
"""

from __future__ import annotations

import pytest

from tests.actor.code_act.sandbox_world import needs_bwrap, world  # noqa: F401
from unify.actor.execution.session import SessionExecutor
from unify.settings import SETTINGS


class NoSuchApp(Exception):
    """Like AppWorld's HTTPException: not an AttributeError."""


class _App:
    def __init__(self, env: "Apis", name: str) -> None:
        object.__setattr__(self, "_env", env)
        object.__setattr__(self, "_name", name)

    def __getattr__(self, api: str):
        env, app = self._env, self._name
        env.seen.append(f"{app}.{api}")
        if api.startswith("_"):
            raise NoSuchApp(f"No API named '{api}' found in {app}.")

        def call(**kwargs):
            env.calls.append((app, api, kwargs))
            return {"app": app, "api": api, **kwargs}

        return call


class Apis:
    """Any attribute name is an app, as AppWorld's client does it."""

    def __init__(self) -> None:
        object.__setattr__(self, "seen", [])
        object.__setattr__(self, "calls", [])

    def __getattr__(self, name: str):
        self.seen.append(name)
        if name.startswith("_"):
            raise NoSuchApp(f"No app named '{name}' found.")
        return _App(self, name)

    def __repr__(self) -> str:
        return "<apis>"


@pytest.fixture
def apis_env():
    from unify.function_manager import function_manager as fm_module
    from unify.function_manager.primitives import (
        EnvironmentMethod,
        EnvironmentNamespace,
        EnvironmentSurface,
        register_environment,
    )
    from unify.function_manager.primitives.environment import (
        clear_environment_namespaces,
    )

    apis = Apis()
    clear_environment_namespaces()
    register_environment(
        EnvironmentSurface(
            namespaces=(
                EnvironmentNamespace(
                    name="spotify",
                    methods=(
                        EnvironmentMethod(
                            name="login",
                            call=lambda **kw: {"via": "primitives", **kw},
                            effect="read",
                            signature="(*, username: str)",
                        ),
                    ),
                ),
            ),
            globals={"apis": apis},
        ),
        source="tests:apis",
    )
    fm_module._PRIMITIVES_SEEDED_FOR.clear()
    yield apis
    clear_environment_namespaces()
    fm_module._PRIMITIVES_SEEDED_FOR.clear()


def test_describing_a_dynamic_object_reads_nothing_from_it(apis_env):
    from unify.actor.execution.worker import PythonWorker, _async_callable

    worker = PythonWorker()
    desc = worker._describe_global("apis", apis_env)
    assert desc["kind"] == "remote" and desc["async"] is False
    assert worker._describe(apis_env.spotify, "spotify")["kind"] == "namespace"
    apis_env.seen.clear()
    login = apis_env.spotify.login
    apis_env.seen.clear()
    assert worker._describe(login, "login") == {"kind": "callable", "async": False}
    assert worker._ref(apis_env)["async"] is False
    assert _async_callable(apis_env) is False
    assert apis_env.seen == []

    # Ordinary callables are described as before.
    async def coro():
        return None

    assert _async_callable(coro) is True and _async_callable(print) is False


@needs_bwrap
@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_no_probe_reaches_the_environment_and_real_calls_work(
    world,  # noqa: F811
    apis_env,
    monkeypatch,
):
    monkeypatch.setattr(SETTINGS, "UNIFY_WORKSPACE_PYTHON", "worker")
    ex = SessionExecutor(environments={})
    try:
        res = await ex.execute(
            code=(
                "import asyncio, inspect\n"
                "token = apis.spotify.login(username='ada')\n"
                "probe = (inspect.iscoroutinefunction(apis.spotify.login), "
                "asyncio.iscoroutinefunction(apis), callable(apis.spotify))\n"
                "(token, probe)"
            ),
            state_mode="stateful",
            session_id=0,
        )
        assert res["error"] is None, res["error"]
        token, probe = res["result"]
        assert token == {"app": "spotify", "api": "login", "username": "ada"}
        assert probe == (False, False, False)  # an app is a namespace, not a callable
        # Underscored attributes stay refused to cell code, in the worker.
        res = await ex.execute(
            code="apis._secret",
            state_mode="stateful",
            session_id=0,
        )
        assert "do not cross the worker boundary" in str(res["error"])
    finally:
        await ex.close()
    # The only names the environment saw are the ones the cell used.
    assert apis_env.calls == [("spotify", "login", {"username": "ada"})]
    assert [n for n in apis_env.seen if n.split(".")[-1].startswith("_")] == []
    assert set(apis_env.seen) <= {"spotify", "spotify.login"}
