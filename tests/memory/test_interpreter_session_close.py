"""A memory agent's interpreter session still reports a box that survived its stop.

The pass's end closes the session and then asks whether its box is alive, so that the gate never judges /memory while
a box could still write it. Closing clears the session's running interpreter; the liveness it reports afterwards is
the stopped interpreter's own. Keyless and boxless: the interpreter is a stand-in.
"""

from __future__ import annotations

import pytest

from unify.memory.tool_handlers import InterpreterSession


class StandIn:
    """An interpreter whose stop leaves its box alive (or not, or raises)."""

    def __init__(self, survives: bool, raises: bool = False) -> None:
        self.survives = survives
        self.raises = raises
        self.stopped = False

    async def stop(self) -> None:
        self.stopped = True
        if self.raises:
            raise RuntimeError("the box did not stop")

    def box_alive(self) -> bool:
        return self.survives or not self.stopped


def _session(interp) -> InterpreterSession:
    session = object.__new__(InterpreterSession)
    session.interp = interp
    session.channel = None
    session._root_fds = {}
    return session


@pytest.mark.asyncio
async def test_a_box_that_survives_its_stop_is_still_seen_alive_after_close():
    interp = StandIn(survives=True)
    session = _session(interp)
    await session.close()
    assert interp.stopped and session.interp is None
    assert session.box_alive() is True


@pytest.mark.asyncio
async def test_a_box_that_stopped_is_seen_stopped_after_close():
    session = _session(StandIn(survives=False))
    await session.close()
    assert session.box_alive() is False


@pytest.mark.asyncio
async def test_a_session_with_no_interpreter_closes_with_no_box():
    session = _session(None)
    await session.close()
    assert session.box_alive() is False


@pytest.mark.asyncio
async def test_a_stop_that_raises_propagates_so_the_pass_fails_closed():
    interp = StandIn(survives=True, raises=True)
    session = _session(interp)
    with pytest.raises(RuntimeError):
        await session.close()
    assert session.box_alive() is True
