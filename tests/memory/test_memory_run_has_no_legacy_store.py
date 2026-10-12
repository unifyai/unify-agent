"""Symbolic: a memory run (``UNIFY_MEMORY=on``) builds no function or guidance store, and no text the
actor is sent names the old store or the run's own directory.

The model is the scripted transport (``tests/cache_discipline_helpers.py``): no key, no network. The cell
runs in the real sandboxed worker. ``UNIFY_HOME`` is put under a directory whose name is a distinctive
run label, as a benchmark launcher's attempt directory is; that label must reach no request.
"""

from __future__ import annotations

import asyncio
import json
import os

import pytest

from tests import cache_discipline_helpers as h
from tests.actor.code_act.sandbox_world import needs_bwrap, world  # noqa: F401 (fixture)
from unify import db, sandbox
from unify.settings import SETTINGS

pytestmark = [pytest.mark.asyncio, pytest.mark.timeout(240)]

LABEL = "runlabel-qz7-arm-k4"

_PROBE = (
    "import os\n"
    "print('NS', sorted(n for n in ('primitives', 'functions', 'guidance') if n in globals()))\n"
    "print('CWD', os.getcwd())\n"
)


@pytest.fixture
def labelled_home(world, monkeypatch):  # noqa: F811
    """``UNIFY_HOME`` (and so the workspace under it) inside a directory named by :data:`LABEL`."""
    home = world["state"].parent / "runs" / LABEL / "attempt-1" / "state"
    (home / "workspace").mkdir(parents=True)
    monkeypatch.setenv("UNIFY_HOME", str(home))
    monkeypatch.setattr(SETTINGS, "UNIFY_LOCAL_ROOT", "")
    monkeypatch.setattr(sandbox, "_POLICY_CACHE", None)
    db.reset_store()
    yield home
    db.reset_store()


@pytest.fixture
def constructed(monkeypatch):
    """Every function or guidance manager built: each constructor and registry getter counts."""
    from unify.function_manager.function_manager import FunctionManager
    from unify.function_manager.simulated import SimulatedFunctionManager
    from unify.guidance_manager.guidance_manager import GuidanceManager
    from unify.guidance_manager.simulated import SimulatedGuidanceManager
    from unify.manager_registry import ManagerRegistry

    calls: list[str] = []
    for cls in (
        FunctionManager,
        SimulatedFunctionManager,
        GuidanceManager,
        SimulatedGuidanceManager,
    ):
        original = cls.__init__

        def counted(self, *a, _original=original, _name=cls.__name__, **k):
            calls.append(_name)
            return _original(self, *a, **k)

        monkeypatch.setattr(cls, "__init__", counted)
    for name in ("get_function_manager", "get_guidance_manager"):
        original = getattr(ManagerRegistry, name)

        def getter(*a, _original=original, _name=name, **k):
            calls.append(_name)
            return _original(*a, **k)

        monkeypatch.setattr(ManagerRegistry, name, getter)
    return calls


def _texts(request: dict) -> dict[str, str]:
    """What one request sends, by part: the system prompt, the first user message, the tool list."""
    messages = request["messages"]
    first_user = next(m for m in messages if m["role"] == "user")
    return {
        "system": json.dumps(messages[0]["content"]),
        "first_user": json.dumps(first_user["content"]),
        "tools": json.dumps(request["tools"]),
    }


async def _act(replies) -> tuple[object, list[dict]]:
    from unify.actor.code_act_actor import CodeActActor

    actor = CodeActActor()
    try:
        with h.scripted(replies) as provider:
            handle = await actor.act("Count the files in the workspace.", persist=False)
            await asyncio.wait_for(handle.result(), 120)
    finally:
        await actor.close()
    return actor, provider.requests


@needs_bwrap
async def test_a_memory_run_builds_no_store_and_names_neither_it_nor_the_run(
    labelled_home,
    constructed,
    monkeypatch,
):
    monkeypatch.setattr(SETTINGS, "UNIFY_MEMORY", "on")
    replies = [
        lambda: h.completion(calls=[("execute_code", {"thought": "probe", "code": _PROBE})]),
        lambda: h.completion(content="done"),
    ]
    actor, requests = await _act(replies)
    assert constructed == []
    assert actor.function_manager is None and actor.guidance_manager is None
    # The cell's namespace: no store objects and no empty primitives object; it starts in /workspace.
    results = [
        json.dumps(m.get("content"))
        for m in requests[-1]["messages"]
        if m.get("role") == "tool"
    ]
    assert results, [m.get("role") for m in requests[-1]["messages"]]
    shown = results[0]
    assert "NS []" in shown and "CWD /workspace" in shown, shown[:1500]
    root = str(labelled_home)
    for request in requests:
        sent = json.dumps(request)
        for banned in ("FunctionManager", LABEL, root):
            assert banned not in sent, banned
        assert "stored function" not in sent.lower()
    system = requests[0]["messages"][0]["content"]
    assert "Your working directory is `/workspace`" in system
    assert "importable in your cells as `memory`" in system


@needs_bwrap
@pytest.mark.parametrize("memory", ["", "on"])
async def test_no_prompt_names_the_run_whatever_the_memory_switch(
    labelled_home,
    monkeypatch,
    memory,
):
    monkeypatch.setattr(SETTINGS, "UNIFY_MEMORY", memory)
    _, requests = await _act([lambda: h.completion(content="done")] * 8)
    first = h.session_requests(requests)[0]
    for part, text in _texts(first).items():
        assert LABEL not in text, part
        assert str(labelled_home) not in text, part
    assert os.path.realpath(labelled_home) not in json.dumps(first)
