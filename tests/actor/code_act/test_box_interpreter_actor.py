"""The persistent interpreter of the actor and its teammates (unify/common/box_interpreter.py): their sandboxed
Python worker with the tool channel's two pipe ends passed in, ``tools`` built in the box, state across cells, the
verified kill on a timeout and exactly the expected descriptors. Keyless; the cells run in the real sandbox."""

from __future__ import annotations

import json
import time
from collections import Counter

import pytest

from tests.actor.code_act.sandbox_world import needs_bwrap, world  # noqa: F401
from tests.memory.box_channel_standin import StandInChannel, fd_listing_code
from tests.memory.test_box_interpreter import EXPECTED_FDS, _kind
from unify.common.box_interpreter import ActorProfile, BoxInterpreter

LOST = "the interpreter was restarted: variables are gone, files remain"


async def _start(channel=None, limit: float = 60.0) -> BoxInterpreter:
    return await BoxInterpreter.start(
        agent="actor",
        profile=ActorProfile(idle_restart_s=None),
        binds=None,
        env={},
        channel=channel,
        stub_source="VALUE = 7\ndef twice(v):\n    return 2 * v\n",
        cell_limit_s=limit,
    )


@needs_bwrap
@pytest.mark.asyncio
@pytest.mark.timeout(180)
async def test_actor_state_tools_and_descriptors(world):  # noqa: F811
    ch = StandInChannel()
    interp = await _start(ch)
    try:
        r1 = await interp.run_cell("x = tools.twice(tools.VALUE)\nx")
        assert r1.error is None and r1.result == 14, r1.text()
        r2 = await interp.run_cell("print(x + 1, __import__('os').environ.get('TOOLS_CHANNEL'))")
        assert r2.output.strip() == f"15 {ch.env_value()}" and not r2.restarted
        r3 = await interp.run_cell(fd_listing_code())
        got = json.loads(r3.output.strip().splitlines()[-1])
        fds = {int(k): v for k, v in got["fds"].items()}
        req, rep = ch.box_fds()
        assert fds[req].startswith("pipe:") and fds[rep].startswith("pipe:")
        assert fds[0] == "/dev/null"
        # the worker child no longer opens the random-number device
        assert Counter(_kind(t) for t in fds.values()) == EXPECTED_FDS - Counter({"urandom": 1}), fds
    finally:
        await interp.stop()
    assert not interp.box_alive()
    assert ("after_spawn",) in ch.events and ch.closed


@needs_bwrap
@pytest.mark.asyncio
@pytest.mark.timeout(180)
async def test_actor_timeout_kills_the_box_and_restarts(world):  # noqa: F811
    opened: list[StandInChannel] = []

    def open_channel() -> StandInChannel:
        opened.append(StandInChannel())
        return opened[-1]

    interp = await _start(open_channel, limit=2.0)
    try:
        await interp.run_cell("y = 1")
        pid = interp._backend.host_pid
        started = time.monotonic()
        r = await interp.run_cell("import time\ntime.sleep(60)")
        assert time.monotonic() - started < 40
        assert r.timed_out and r.notes_after == [f"(the cell timed out after 2 s; {LOST})"]
        assert not interp.box_alive()
        from unify.memory import sandbox_run as sr

        assert not sr._descendants(pid)
        r2 = await interp.run_cell("print('y' in dir(), tools.VALUE)")
        assert r2.output.strip() == "False 7" and len(opened) == 2
    finally:
        await interp.stop()
