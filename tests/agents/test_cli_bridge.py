"""Symbolic: the user is a participant. Messages become posts, a parked main agent
is woken with them once, cancels are recorded, and replies too."""

import asyncio
import json
import os
import sys
from types import SimpleNamespace

import pytest

from unify.agents.cli_bridge import CliBridge, attach_bridge
from unify.agents.options import Options
from unify.agents.pool import Pool
from unify.agents.record import Record

pytestmark = pytest.mark.no_unify_context


class _Handle:
    def __init__(self, pool=None):
        self.interjected = []
        self.cancelled = 0
        self.stopped = None
        if pool is not None:
            self.agents_pool = pool

    async def interject(self, text, **_):
        self.interjected.append(text)

    async def cancel_request(self):
        self.cancelled += 1
        return True

    async def stop(self, reason=None):
        self.stopped = reason


def _pool():
    async def start(*a):
        return "ok"

    rec = Record(None, options=Options(min_post_interval_s=0), user_reads=True)
    return Pool(rec, start_helper=start, spawn_allowed=True)


def _bridge():
    pool = _pool()
    out = []
    handle = _Handle(pool)
    return CliBridge(pool, handle, out.append), pool.record, handle, out


@pytest.mark.asyncio
async def test_a_message_while_the_root_works_is_only_posted():
    bridge, rec, handle, _ = _bridge()
    await bridge.user_message("use 2024")
    assert rec.entries[-1].author == "user" and handle.interjected == []


@pytest.mark.asyncio
async def test_a_message_while_the_root_is_parked_wakes_it_with_the_block_once():
    bridge, rec, handle, _ = _bridge()
    await bridge.root_replied("first answer")
    await bridge.user_message("next question")
    assert len(handle.interjected) == 1
    assert "user: next question" in handle.interjected[0]
    assert rec.take_block("root") is None
    await bridge.user_message("and one more while it works")
    assert len(handle.interjected) == 1


@pytest.mark.asyncio
async def test_cancel_and_replies_are_recorded_and_entries_for_the_user_are_emitted():
    bridge, rec, _, out = _bridge()
    bridge.cancel_posted()
    assert rec.entries[-1].kind == "cancel" and rec.entries[-1].mentions == ("root",)
    rec.append("root", "@user progress: 3 of 4 files")
    await bridge.root_replied("the answer")
    assert [o["text"] for o in out if o["type"] == "record"] == [
        "@user progress: 3 of 4 files",
    ]
    assert rec.entries[-1].kind == "reply" and rec.entries[-1].mentions == ("user",)


def test_no_pool_no_bridge():
    assert attach_bridge(_Handle(), lambda **_: None) is None


@pytest.mark.asyncio
async def test_the_act_driver_posts_messages_records_cancels_and_never_interjects(
    monkeypatch,
):
    from unify.cli import Act

    read_fd, write_fd = os.pipe()
    monkeypatch.setattr(sys, "stdin", os.fdopen(read_fd, "r"))
    session = Act(SimpleNamespace(persist=True, quiet=True, jsonl=True))
    pool = _pool()
    handle = _Handle(pool)
    session._handle = handle
    session._bridge = attach_bridge(handle, session._emit)
    reader = asyncio.create_task(session._read_lines())
    for item in ({"message": "use the 2024 file"}, {"cancel": True}, {"quit": True}):
        os.write(write_fd, (json.dumps(item) + "\n").encode())
    os.close(write_fd)
    await asyncio.wait_for(reader, timeout=5)
    kinds = [(e.author, e.kind, e.text) for e in pool.record.entries]
    assert kinds == [
        ("user", "post", "use the 2024 file"),
        ("user", "cancel", "@root cancel"),
    ]
    assert handle.interjected == [] and handle.cancelled == 1
    assert handle.stopped is not None


@pytest.mark.asyncio
async def test_a_long_line_is_posted_and_mentions_in_user_lines_resolve():
    bridge, rec, _, _ = _bridge()
    rec.add_agent("h1", spawner="root")
    await bridge.user_message("x" * 40_000)
    assert len(rec.entries[-1].text.encode()) <= 16 * 1024
    await bridge.user_message("@h1 stop that")
    assert rec.entries[-1].mentions == ("h1",)


@pytest.mark.asyncio
async def test_a_message_during_the_last_model_call_wakes_the_parked_session():
    bridge, rec, handle, _ = _bridge()
    await bridge.user_message("posted while the main agent finished")
    await bridge.root_replied("the answer")
    assert len(handle.interjected) == 1
    assert "posted while the main agent finished" in handle.interjected[0]
