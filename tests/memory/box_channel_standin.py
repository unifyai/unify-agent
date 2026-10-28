"""A stand-in for the tool channel (unify/common/tool_channel.py ``Channel``) with the interface the interpreter uses:
two real pipes whose box ends go into the box, ``TOOLS_CHANNEL`` naming them, and no call dispatch."""

from __future__ import annotations

import asyncio
import os


class StandInChannel:
    def __init__(self) -> None:
        self.req_r, self.req_w = os.pipe()  # the box writes requests on req_w
        self.rep_r, self.rep_w = os.pipe()  # the box reads replies on rep_r
        self.closed = False
        self.close_reason: str | None = None
        self.in_flight = False
        self.events: list[tuple] = []
        self._done = asyncio.Event()

    def box_fds(self) -> tuple[int, int]:
        return (self.req_w, self.rep_r)

    def env_value(self) -> str:
        return f"fd:{self.req_w},{self.rep_r}"

    def after_spawn(self) -> None:
        self.events.append(("after_spawn",))
        for fd in (self.req_w, self.rep_r):
            try:
                os.close(fd)
            except OSError:
                pass

    async def serve(self) -> None:
        await self._done.wait()

    def cell_started(self, cell: int) -> None:
        self.events.append(("cell_started", cell))

    def cell_ended(self, cell: int) -> None:
        self.events.append(("cell_ended", cell))

    async def close(self, reason: str) -> None:
        self.events.append(("close", reason))
        if not self.closed:
            self.closed = True
            self.close_reason = self.close_reason or reason
        for fd in (self.req_r, self.rep_w):
            try:
                os.close(fd)
            except OSError:
                pass
        self._done.set()


def fd_listing_code() -> str:
    """A cell printing, as JSON, every descriptor the interpreter's process holds (number -> link target) and the
    value of ``TOOLS_CHANNEL``. The directory descriptor ``os.listdir`` opens is closed again before any link is
    read, so it is not listed."""
    return (
        "import json as _j, os as _o\n"
        "_fds = {}\n"
        "for _n in _o.listdir('/proc/self/fd'):\n"
        "    try:\n"
        "        _fds[int(_n)] = _o.readlink('/proc/self/fd/' + _n)\n"
        "    except OSError:\n"
        "        pass\n"
        "print(_j.dumps({'fds': _fds, 'channel': _o.environ.get('TOOLS_CHANNEL')}))\n"
    )
