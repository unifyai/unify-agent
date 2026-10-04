"""Symbolic: under worker Python an environment's globals are served from the harness, not re-imported.

A benchmark adapter registers, beside its namespaces, globals bound to the
harness's connection to the environment: AppWorld's ``apis`` is a client of a
relay socket under the harness's ``/tmp``. Under ``UNIFY_WORKSPACE_PYTHON=worker``
(which ``UNIFY_TOOL_SURFACE=core`` requires) such a global was installed in the
worker by importing it by name -- ``appworld_client.apis`` -- so cell code got
a fresh client inside the sandbox, whose private ``/tmp`` has no relay socket,
and every ``apis.*`` call raised ``RelayError``; only ``primitives.<app>.<api>``
worked, though the task prompt and the namespaces section point at ``apis``.

An environment global is now a remote object, as ``primitives`` is: attribute
access and calls go through the worker's channel and run on the harness's
object. Nothing new is mounted into the sandbox: the worker still cannot read
the relay's file itself, and the sandbox command line is the same with or
without the environment registered. Without worker Python nothing changes.
Cells run in the real sandboxed worker; no model is called.
"""

from __future__ import annotations

import os
import shutil
import tempfile

import pytest

from tests.actor.code_act import fake_relay_env
from tests.actor.code_act.sandbox_world import needs_bwrap, world  # noqa: F401
from unify import sandbox
from unify.actor.execution.session import SessionExecutor
from unify.actor.execution.types import parts_to_text
from unify.settings import SETTINGS

TOKEN = "relay-token-5d1f"  # pragma: allowlist secret


@pytest.fixture
def relay_env(monkeypatch):
    """A registered environment with one namespace and the global ``relay``,
    whose resource lives under the harness's /tmp."""
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

    root = tempfile.mkdtemp(prefix="unify-relay-", dir="/tmp")
    path = os.path.join(root, "relay.sock")
    with open(path, "w") as fh:
        fh.write(TOKEN + "\n")
    client = fake_relay_env.RelayClient(path)
    monkeypatch.setattr(fake_relay_env, "relay", client)
    clear_environment_namespaces()
    register_environment(
        EnvironmentSurface(
            namespaces=(
                EnvironmentNamespace(
                    name="music",
                    methods=(
                        EnvironmentMethod(
                            name="show",
                            call=lambda item: client.music.show(item),
                            effect="read",
                            signature="(item: str)",
                        ),
                    ),
                ),
            ),
            globals={"relay": client},
        ),
        source="tests:relay",
    )
    fm_module._PRIMITIVES_SEEDED_FOR.clear()
    yield client
    clear_environment_namespaces()
    fm_module._PRIMITIVES_SEEDED_FOR.clear()
    shutil.rmtree(root, ignore_errors=True)


async def _run(ex, code: str):
    res = await ex.execute(code=code, state_mode="stateful", session_id=0)
    return parts_to_text(res["stdout"]), res


@needs_bwrap
@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_an_environment_global_is_served_from_the_harness(
    world,  # noqa: F811
    relay_env,
    monkeypatch,
):
    monkeypatch.setattr(SETTINGS, "UNIFY_WORKSPACE_PYTHON", "worker")
    harness = fake_relay_env.process_id()
    ex = SessionExecutor(environments={})
    try:
        _, res = await _run(
            ex,
            "import os\n"
            "worker = f\"{os.getpid()}@{os.readlink('/proc/self/ns/pid')}\"\n"
            "(relay.ping('hi'), relay.music.show('song'), worker, repr(relay))",
        )
        assert res["error"] is None, res["error"]
        pong, shown, worker, shown_repr = res["result"]
        # The call ran on the harness's client, which reaches the relay.
        assert pong == {"word": "hi", "token": TOKEN, "ran_in": harness}
        assert shown["app"] == "music" and shown["token"] == TOKEN
        assert worker != harness
        assert shown_repr == "<relay client>"
        # A stored function defined in the worker reaches it the same way.
        _, res = await _run(
            ex,
            "def use_relay():\n    return relay.ping('fn')['token']\nuse_relay()",
        )
        assert res["error"] is None and res["result"] == TOKEN, res["error"]
        # Confinement is unchanged: the worker cannot read the relay itself.
        _, res = await _run(ex, f"open({relay_env.path!r}).read()")
        assert "FileNotFoundError" in str(res["error"])
        # Private attributes stay in the harness.
        _, res = await _run(ex, "relay._App")
        assert "do not cross the worker boundary" in str(res["error"])
    finally:
        await ex.close()


def test_registering_an_environment_mounts_nothing_new(
    world,
    monkeypatch,
):  # noqa: F811
    """The sandbox command line is the same with and without the environment."""
    from unify.function_manager.primitives import (
        EnvironmentMethod,
        EnvironmentNamespace,
        EnvironmentSurface,
        register_environment,
    )
    from unify.function_manager.primitives.environment import (
        clear_environment_namespaces,
    )

    def argv() -> list:
        monkeypatch.setattr(sandbox, "_POLICY_CACHE", None)
        policy = sandbox.build_policy()
        return sandbox.wrap_argv(["python3"], policy, cwd=str(policy.workspace))

    clear_environment_namespaces()
    try:
        before = argv()
        client = fake_relay_env.RelayClient("/tmp/unify-relay-x/relay.sock")
        register_environment(
            EnvironmentSurface(
                namespaces=(
                    EnvironmentNamespace(
                        name="music",
                        methods=(
                            EnvironmentMethod(
                                name="show",
                                call=lambda item: item,
                                effect="read",
                                signature="(item: str)",
                            ),
                        ),
                    ),
                ),
                globals={"relay": client},
            ),
            source="tests:relay",
        )
        assert argv() == before
        assert not any("unify-relay" in str(a) for a in before)
    finally:
        clear_environment_namespaces()


def test_the_worker_describes_an_environment_global_as_remote(relay_env):
    """The manifest entry: a remote object, never an import by name, whose
    module attribute the old path imported."""
    from unify.actor.execution.worker import PythonWorker, _importable

    assert _importable("relay", relay_env) == [
        "attr",
        "tests.actor.code_act.fake_relay_env",
        "relay",
    ]
    desc = PythonWorker()._describe_global("relay", relay_env)
    assert desc["kind"] == "remote" and desc["repr"] == "<relay client>"
