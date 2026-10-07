"""Symbolic: an environment observer sees a cell's calls in process and from the sandboxed worker.

Under ``UNIFY_WORKSPACE_PYTHON=worker`` a cell runs in a child process, and
its ``primitives.<ns>.<method>`` call (or a raw global's ``apis.app.api``)
comes back to the harness as an ``{"op": "call"}`` request, which the harness
serves by calling the very wrapper an in-process cell calls. An observer
pushed with ``observing()`` where the harness runs the cell must therefore be
in force when that request is served: each request is served in a task
created from the running cell's context, so ``ContextVar`` scoping is the same
as in process -- a cell outside the scope, or a cell of another session
running at the same time, is not observed. Cells run in process and in the
real sandboxed worker; no model is called.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any, Optional

import pytest

from tests.actor.code_act.sandbox_world import needs_bwrap, world  # noqa: F401
from unify.actor.execution.session import SessionExecutor
from unify.function_manager.primitives.observers import (
    EnvCall,
    Intercepted,
    observing,
)
from unify.settings import SETTINGS

CALLS: list[tuple] = []


class _App:
    def __init__(self, name: str) -> None:
        self.name = name

    def login(self, *, username: str) -> dict:
        CALLS.append((f"{self.name}.login", username))
        return {"user": username, "ran_in": os.getpid()}

    async def like(self, *, song_id: int) -> dict:
        CALLS.append((f"{self.name}.like", song_id))
        return {"liked": song_id}


class Apis:
    def __init__(self) -> None:
        self.spotify = _App("spotify")

    def __repr__(self) -> str:
        return "<apis>"


APIS = Apis()


class Recorder:
    def __init__(self, intercept: Any = None) -> None:
        self.intercept = intercept
        self.complete = False
        self.log: list = []

    def before(self, call: EnvCall) -> Optional[Intercepted]:
        self.log.append(("before", call))
        return self.intercept

    def after(self, call: EnvCall, **outcome: Any) -> None:
        self.log.append(("after", call, outcome))

    def calls(self) -> list[tuple]:
        return [
            (c.via, c.namespace, c.method, c.effect, c.args, c.kwargs)
            for kind, c, *_ in self.log
            if kind == "before"
        ]


@pytest.fixture
def spotify_env():
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

    async def like(*, song_id: int) -> dict:
        return await APIS.spotify.like(song_id=song_id)

    clear_environment_namespaces()
    register_environment(
        EnvironmentSurface(
            namespaces=(
                EnvironmentNamespace(
                    name="spotify",
                    methods=(
                        EnvironmentMethod(
                            name="login",
                            call=lambda **kw: APIS.spotify.login(**kw),
                            effect="read",
                            signature="(*, username: str)",
                        ),
                        EnvironmentMethod(
                            name="like",
                            call=like,
                            effect="write",
                            signature="(*, song_id: int)",
                        ),
                    ),
                ),
            ),
            globals={"apis": APIS},
        ),
        source="tests:observers",
    )
    fm_module._PRIMITIVES_SEEDED_FOR.clear()
    CALLS.clear()
    yield
    clear_environment_namespaces()
    fm_module._PRIMITIVES_SEEDED_FOR.clear()


def _executor() -> SessionExecutor:
    from unify.actor.environments import EnvironmentNamespacesEnvironment

    return SessionExecutor(
        environments={"primitives": EnvironmentNamespacesEnvironment()},
    )


async def _cell(ex: SessionExecutor, code: str, session_id: int = 0) -> Any:
    res = await asyncio.wait_for(
        ex.execute(code=code, state_mode="stateful", session_id=session_id),
        timeout=60,
    )
    assert res["error"] is None, res["error"]
    return res["result"]


MODES = [
    pytest.param("in_process", id="in_process"),
    pytest.param("worker", id="worker", marks=needs_bwrap),
]


def _mode(monkeypatch, mode: str) -> None:
    if mode == "worker":
        monkeypatch.setattr(SETTINGS, "UNIFY_WORKSPACE_PYTHON", "worker")


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.asyncio
@pytest.mark.timeout(180)
async def test_a_cells_environment_calls_reach_the_observer(
    world,  # noqa: F811
    spotify_env,
    monkeypatch,
    mode,
):
    _mode(monkeypatch, mode)
    ex = _executor()
    try:
        # The session (and its worker) starts before any observer exists: the
        # observer is in force per cell, not fixed when the worker started.
        where = await _cell(ex, "import os\nos.readlink('/proc/self/ns/pid')")
        assert (where != os.readlink("/proc/self/ns/pid")) == (mode == "worker")

        rec = Recorder()
        with observing(rec):
            out = await _cell(
                ex,
                "def sign_in(name):\n"
                "    return primitives.spotify.login(username=name)\n"
                "(sign_in('ada'), await primitives.spotify.like(song_id=3))",
            )
        assert out[0]["user"] == "ada" and out[1] == {"liked": 3}
        # The environment ran in the harness either way.
        assert out[0]["ran_in"] == os.getpid()
        assert rec.calls() == [
            ("primitives", "spotify", "login", "read", (), {"username": "ada"}),
            ("primitives", "spotify", "like", "write", (), {"song_id": 3}),
        ]
        after = [entry for entry in rec.log if entry[0] == "after"]
        assert [a[2]["intercepted"] for a in after] == [False, False]
        assert after[1][2]["result"] == {"liked": 3}

        # A speculative answer replaces the call, sync and async alike.
        spec = Recorder(intercept=Intercepted({"speculative": True}))
        with observing(spec):
            out = await _cell(
                ex,
                "(primitives.spotify.login(username='bo'), "
                "await primitives.spotify.like(song_id=4))",
            )
        assert list(out) == [{"speculative": True}] * 2
        assert [e[2]["intercepted"] for e in spec.log if e[0] == "after"] == [
            True,
            True,
        ]

        # Outside the scope nothing is observed.
        await _cell(ex, "primitives.spotify.login(username='cy')")
        assert len(rec.log) == 4 and len(spec.log) == 4
        assert CALLS == [
            ("spotify.login", "ada"),
            ("spotify.like", 3),
            ("spotify.login", "cy"),
        ]
    finally:
        await ex.close()


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.asyncio
@pytest.mark.timeout(180)
async def test_a_raw_global_call_reaches_the_observer_when_a_feature_is_on(
    world,  # noqa: F811
    spotify_env,
    monkeypatch,
    mode,
):
    _mode(monkeypatch, mode)
    # The switch does not exist yet: placed where ``getattr(SETTINGS, ...)`` finds it.
    monkeypatch.setitem(vars(SETTINGS), "UNIFY_SPECULATE", "writes")
    ex = _executor()
    try:
        rec = Recorder()
        with observing(rec):
            out = await _cell(
                ex,
                "(repr(apis), apis.spotify.login(username='ada'), "
                "await apis.spotify.like(song_id=5))",
            )
        assert out[0] == "<apis>"
        assert out[1]["user"] == "ada" and out[2] == {"liked": 5}
        assert rec.calls() == [
            ("global", "apis", "spotify.login", "", (), {"username": "ada"}),
            ("global", "apis", "spotify.like", "", (), {"song_id": 5}),
        ]
    finally:
        await ex.close()


@needs_bwrap
@pytest.mark.asyncio
@pytest.mark.timeout(180)
async def test_concurrent_worker_cells_keep_their_own_observer_scope(
    world,  # noqa: F811
    spotify_env,
    monkeypatch,
):
    """Two sessions' workers serve calls at the same time; only the cell run
    inside ``observing()`` is observed, as two in-process tasks would be."""
    monkeypatch.setattr(SETTINGS, "UNIFY_WORKSPACE_PYTHON", "worker")
    ex = _executor()
    rec = Recorder()
    code = (
        "import asyncio\n"
        "for i in range(3):\n"
        "    primitives.spotify.login(username='{who}' + str(i))\n"
        "    await asyncio.sleep(0.05)\n"
    )
    try:
        # Start both workers first, so the cells overlap.
        await _cell(ex, "1", session_id=1)
        await _cell(ex, "1", session_id=2)

        async def observed() -> None:
            with observing(rec):
                await _cell(ex, code.format(who="seen"), session_id=1)

        await asyncio.gather(
            observed(),
            _cell(ex, code.format(who="unseen"), session_id=2),
        )
    finally:
        await ex.close()
    assert [c[5]["username"] for c in rec.calls()] == ["seen0", "seen1", "seen2"]
    assert sorted(c[1] for c in CALLS if c[1].startswith("unseen")) == [
        "unseen0",
        "unseen1",
        "unseen2",
    ]
