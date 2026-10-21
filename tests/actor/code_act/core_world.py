"""The world the actor tests run in, and their helpers.

Cells run in the real sandboxed worker (``sandbox_world``); the model, where
there is one, is the scripted transport of ``tests/cache_discipline_helpers``.
"""

from __future__ import annotations

from typing import Any

import pytest

from tests.actor.code_act.sandbox_world import world  # noqa: F401 (fixture)


@pytest.fixture
def core_world(world):  # noqa: F811
    """The sandbox world the actor's cells run in."""
    yield world


def new_actor(**kwargs: Any):
    """A CodeActActor with no environment unless *kwargs* names some."""
    from unify.actor.code_act_actor import CodeActActor

    kwargs.setdefault("environments", [])
    return CodeActActor(**kwargs)


def tool_names(request: dict) -> list[str]:
    """The names of the tools a recorded request carried, in order."""
    return [t["function"]["name"] for t in request["tools"] or []]
