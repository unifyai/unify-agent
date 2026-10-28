"""Symbolic: the actor refuses to start where it cannot confine its cells.

Every cell runs in the sandboxed worker (bubblewrap); without bubblewrap the
actor refuses before anything starts, and holds no slot. The model is a
scripted transport (tests/cache_discipline_helpers.py); nothing leaves the
process.
"""

from __future__ import annotations

import pytest

from tests.actor.code_act.core_world import new_actor, world  # noqa: F401 (fixture)
from unify import sandbox
from unify.actor import code_act_actor, core_surface


@pytest.mark.asyncio
async def test_the_actor_refuses_to_start_without_bubblewrap(world, monkeypatch):  # noqa: F811
    actor = new_actor()

    def missing() -> None:
        raise sandbox.SandboxRefusal("sandbox-required", "bwrap is not installed")

    monkeypatch.setattr(sandbox, "require_bwrap", missing)
    with pytest.raises(core_surface.ToolSurfaceError, match="sandboxed worker"):
        await actor.act("Do the task.")
    # Nothing was started: no slot is held.
    assert actor._slots._value == code_act_actor.MAX_CONCURRENT_REQUESTS
