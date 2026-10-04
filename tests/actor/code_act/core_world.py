"""The world the ``UNIFY_TOOL_SURFACE=core`` tests run in, and their helpers.

Cells run in the real sandboxed worker (``sandbox_world``); the model, where
there is one, is the scripted transport of ``tests/cache_discipline_helpers``.
"""

from __future__ import annotations

from typing import Any

import pytest

from tests.actor.code_act.sandbox_world import world  # noqa: F401 (fixture)
from unify import db
from unify.settings import SETTINGS


@pytest.fixture
def core_world(world, monkeypatch):  # noqa: F811
    """The sandbox world with worker Python, the gate off and the switch on.

    The world seeds a stand-in store to prove the sandbox hides it; these
    tests need a real one, so it is replaced by an empty store.
    """
    (world["state"] / "store.sqlite").unlink()
    db.reset_store()
    monkeypatch.setattr(SETTINGS, "UNIFY_WORKSPACE_PYTHON", "worker")
    monkeypatch.setattr(SETTINGS, "UNIFY_DISCOVERY_GATE", False)
    monkeypatch.setattr(SETTINGS, "UNIFY_TOOL_SURFACE", "core")
    yield world
    db.reset_store()


def new_actor(**kwargs: Any):
    """A CodeActActor over a fresh function and guidance library."""
    from unify.actor.code_act_actor import CodeActActor
    from unify.function_manager.function_manager import FunctionManager
    from unify.guidance_manager.guidance_manager import GuidanceManager

    kwargs.setdefault("environments", [])
    kwargs.setdefault("function_manager", FunctionManager(include_primitives=False))
    kwargs.setdefault("guidance_manager", GuidanceManager())
    return CodeActActor(**kwargs)


def tool_names(request: dict) -> list[str]:
    """The names of the tools a recorded request carried, in order."""
    return [t["function"]["name"] for t in request["tools"] or []]
